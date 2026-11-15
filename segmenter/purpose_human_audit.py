"""Prepare blinded purpose review; score only completed independent human labels."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from .ledger import atomic, encoded, sha
from .purpose_probe import LABELS, trajectories

DEFAULT_PROBE = Path('eval/purpose_probe_v1')


def select_cases(ts, probe, per_source=30, per_trajectory=4):
    candidates = {'SWE':[], 'Terminal-Bench':[]}
    for t in ts:
        source = 'SWE' if int(t['trajectory_id'][1:]) <= 10 else 'Terminal-Bench'
        for position, step in enumerate(t['steps']):
            rid = f"{t['trajectory_id']}_s{step['idx']:05d}"
            response = json.loads((probe/'responses'/f'{rid}.json').read_text())
            if response['status'] != 'scored':
                continue
            packet = dict(trajectory_id=t['trajectory_id'],task=t['task'],
                          current_step_id=step['idx'],steps=t['steps'][:position+1])
            if response['input_sha256'] != sha(encoded(packet)):
                raise ValueError('Probe input changed: '+rid)
            candidates[source].append(dict(request_id=rid,source=source,
                                          predicted=response['purpose'],packet=packet))
    selected=[]
    counts=Counter()
    def choose(row):
        trace=row['packet']['trajectory_id']
        if counts[trace] >= per_trajectory:
            return False
        selected.append(row);counts[trace]+=1
        return True
    for source, rows in candidates.items():
        rows.sort(key=lambda row:hashlib.sha256(('purpose-audit-v1:'+row['request_id']).encode()).hexdigest())
        for label in LABELS:
            picked=0
            for row in rows:
                if row['predicted']==label and row not in selected and choose(row):
                    picked+=1
                    if picked==2:break
        for row in rows:
            if sum(x['source']==source for x in selected)>=per_source:
                break
            if row not in selected:
                choose(row)
        if sum(x['source']==source for x in selected)!=per_source:
            raise ValueError('Insufficient eligible independent examples: '+source)
    return sorted(selected,key=lambda row:row['request_id'])


def agreement(a,b):
    n=len(a)
    observed=sum(x==y for x,y in zip(a,b))/n
    ca,cb=Counter(a),Counter(b)
    expected=sum(ca[k]*cb[k] for k in LABELS)/(n*n)
    return dict(exact=observed,cohen_kappa=(observed-expected)/(1-expected) if expected<1 else None)


def wilson(k,n):
    if not n:return None
    z=1.96;p=k/n;denom=1+z*z/n
    midpoint=(p+z*z/(2*n))/denom
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [midpoint-radius,midpoint+radius]


def load_labels(path,expected,annotator):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('annotator')!=annotator or set(data.get('labels',{}))!=set(expected):
        raise ValueError('Wrong annotator or incomplete request set: '+annotator)
    for rid, label in data['labels'].items():
        if label not in LABELS:
            raise ValueError('Missing/invalid human label: '+rid)
    return data['labels']


def make(probe,out):
    if out.exists():raise FileExistsError('Use a fresh review directory')
    out.mkdir(parents=True)
    selected=select_cases(trajectories(),probe)
    packets=[dict(request_id=x['request_id'],source=x['source'],packet=x['packet']) for x in selected]
    atomic(out/'blind_packets.json',packets)
    for name in ('A','B'):
        atomic(out/f'annotator_{name}.json',dict(annotator=name,labels={x['request_id']:None for x in selected}))
    atomic(out/'adjudication.json',dict(annotator='adjudicator',labels={}))
    atomic(out/'selection.json',dict(protocol='purpose-two-human-v1',seed='purpose-audit-v1',
        n=len(selected),by_source=dict(Counter(x['source'] for x in selected)),
        by_model_label=dict(Counter(x['predicted'] for x in selected)),
        blind_packets_sha256=sha((out/'blind_packets.json').read_bytes()),
        probe_identity=json.loads((probe/'manifest.json').read_text())['identity'],
        note='Model predictions never appear in annotator packets; UNKNOWN is a valid human label.'))
    print(json.dumps(dict(n=len(selected),out=str(out)),ensure_ascii=False))


def score(probe,out):
    packets=json.loads((out/'blind_packets.json').read_text(encoding='utf-8'))
    meta=json.loads((out/'selection.json').read_text(encoding='utf-8'))
    if sha((out/'blind_packets.json').read_bytes())!=meta['blind_packets_sha256']:
        raise ValueError('Blind packet changed')
    if json.loads((probe/'manifest.json').read_text())['identity']!=meta['probe_identity']:
        raise ValueError('Probe identity changed')
    ids=[x['request_id'] for x in packets]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate request ID')
    a=load_labels(out/'annotator_A.json',ids,'A')
    b=load_labels(out/'annotator_B.json',ids,'B')
    ad=json.loads((out/'adjudication.json').read_text(encoding='utf-8'))
    disputed={rid for rid in ids if a[rid]!=b[rid]}
    if ad.get('annotator')!='adjudicator' or set(ad.get('labels',{}))!=disputed:
        raise ValueError('Adjudicate exactly the human disagreements before scoring')
    if any(v not in LABELS for v in ad['labels'].values()):
        raise ValueError('Incomplete adjudication')
    gold={rid:(a[rid] if rid not in disputed else ad['labels'][rid]) for rid in ids}
    results=[]
    for p in packets:
        rid=p['request_id']; raw=json.loads((probe/'responses'/f'{rid}.json').read_text())
        if raw['status']!='scored' or raw['input_sha256']!=sha(encoded(p['packet'])):
            raise ValueError('Probe response changed: '+rid)
        results.append(dict(request_id=rid,source=p['source'],prediction=raw['purpose'],
                            gold=gold[rid],correct=raw['purpose']==gold[rid]))
    def stats(rows):
        n=len(rows);k=sum(x['correct'] for x in rows)
        return dict(n=n,correct=k,accuracy=k/n if n else None,wilson95=wilson(k,n),
                    gold_counts=dict(Counter(x['gold'] for x in rows)))
    report=dict(protocol='purpose-two-human-v1',agreement=agreement([a[i] for i in ids],[b[i] for i in ids]),
        disagreements=len(disputed),overall=stats(results),
        by_source={source:stats([x for x in results if x['source']==source])
                   for source in ('SWE','Terminal-Bench')},
        known_gold=stats([x for x in results if x['gold']!='UNKNOWN']),
        rows=results,selection_sha256=sha((out/'selection.json').read_bytes()),
        human_file_hashes={name:sha((out/f'{name}.json').read_bytes())
                           for name in ('annotator_A','annotator_B','adjudication')},
        limitations=['Purpose accuracy on stratified 60-case sample only, not full 735-step prevalence',
                     'Two independent human labels plus adjudication required; model never creates gold'])
    atomic(out/'human_accuracy_report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False))


def score_assisted(probe,out):
    """Score a fully human-reviewed AI draft, never call it independent gold."""
    draft_path=out/'ai_draft.json'
    draft=json.loads(draft_path.read_text(encoding='utf-8'))
    review=json.loads((out/'ai_review.json').read_text(encoding='utf-8'))
    packets=json.loads((out/'blind_packets.json').read_text(encoding='utf-8'))
    meta=json.loads((out/'selection.json').read_text(encoding='utf-8'))
    if sha((out/'blind_packets.json').read_bytes())!=meta['blind_packets_sha256']:
        raise ValueError('Blind packet changed')
    if review.get('draft_sha256')!=sha(draft_path.read_bytes()):
        raise ValueError('AI draft changed after review template creation')
    if not isinstance(review.get('reviewer_id'),str) or not review['reviewer_id'].strip():
        raise ValueError('Set a nonempty reviewer_id before scoring')
    ids=[x['request_id'] for x in packets]
    if len(ids)!=len(set(ids)) or set(ids)!=set(draft['suggestions']) or set(ids)!=set(review.get('labels',{})):
        raise ValueError('Draft/review IDs differ from blind packet')
    if json.loads((probe/'manifest.json').read_text())['identity']!=meta['probe_identity']:
        raise ValueError('Probe identity changed')
    results=[]
    for packet in packets:
        rid=packet['request_id']; item=review['labels'][rid]
        suggested=draft['suggestions'][rid]['label']
        if item.get('suggested_label')!=suggested or item.get('reviewed') is not True:
            raise ValueError('Unreviewed or altered suggestion: '+rid)
        label=item.get('final_label')
        if label not in LABELS:
            raise ValueError('Missing/invalid reviewed label: '+rid)
        raw=json.loads((probe/'responses'/f'{rid}.json').read_text())
        if raw['status']!='scored' or raw['input_sha256']!=sha(encoded(packet['packet'])):
            raise ValueError('Probe response changed: '+rid)
        results.append(dict(request_id=rid,source=packet['source'],qwen=raw['purpose'],
                            ai_draft=suggested,reviewed_reference=label,
                            qwen_agrees=raw['purpose']==label,review_changed_draft=suggested!=label))
    def stats(rows):
        n=len(rows);k=sum(x['qwen_agrees'] for x in rows)
        return dict(n=n,qwen_agrees=k,agreement=k/n if n else None,wilson95=wilson(k,n),
                    ai_draft_changed=sum(x['review_changed_draft'] for x in rows),
                    reviewed_label_counts=dict(Counter(x['reviewed_reference'] for x in rows)))
    report=dict(protocol='purpose-ai-assisted-single-review-v1',reviewer_id=review['reviewer_id'],
                reference_provenance='AI suggestions exposed before one human review; not independent human gold',
                overall=stats(results),
                by_source={source:stats([x for x in results if x['source']==source])
                           for source in ('SWE','Terminal-Bench')},
                known_reference=stats([x for x in results if x['reviewed_reference']!='UNKNOWN']),
                rows=results,draft_sha256=sha(draft_path.read_bytes()),
                review_sha256=sha((out/'ai_review.json').read_bytes()),
                limitations=['AI-assisted single-reviewer agreement on stratified sample; not blind human accuracy',
                             'Not representative of all 735 steps without sample weighting'])
    atomic(out/'assisted_agreement_report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['make','score','score-assisted'])
    ap.add_argument('--probe',type=Path,default=DEFAULT_PROBE)
    ap.add_argument('--out',type=Path,default=Path('eval/purpose_human_audit_v1'))
    args=ap.parse_args()
    {'make':make,'score':score,'score-assisted':score_assisted}[args.command](args.probe,args.out)


if __name__=='__main__':main()
