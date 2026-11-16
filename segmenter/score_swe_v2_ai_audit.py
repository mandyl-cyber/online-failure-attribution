"""Score a blind Codex AI sample against first-pass weak labels; no human accuracy claim."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .purpose_pilot_remote_v2 import read, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    selection=read(a.sample/'selection.json')
    blind=(a.sample/'blind_packets.jsonl').read_bytes()
    ai=read(a.sample/'ai_labels.json')
    first_bytes=(a.review/'review_pending.json').read_bytes()
    if ai['blind_packets_sha256']!=hashlib.sha256(blind).hexdigest(): raise ValueError('Blind packets changed')
    if 'candidate_sha256' not in selection and selection['review_sha256']!=hashlib.sha256(first_bytes).hexdigest():
        raise ValueError('First review table changed')
    first=read(a.review/'review_pending.json')['labels']
    packets={x['request_id']:x for x in map(json.loads,blind.decode().splitlines())}
    ids=[x['request_id'] for x in selection['items']]
    if len(ids)!=len(set(ids)) or set(ids)!=set(packets) or set(ids)!=set(ai['labels']):
        raise ValueError('Selection, blind packets, and AI labels differ')
    for rid,digest in selection.get('candidate_sha256',{}).items():
        actual=hashlib.sha256(json.dumps(dict(input_sha256=first[rid]['input_sha256'],
            candidate=first[rid]['candidate']),ensure_ascii=False,sort_keys=True,
            separators=(',',':')).encode()).hexdigest()
        if actual!=digest: raise ValueError('First candidate changed: '+rid)
    pairs=Counter(); strata={}; targeted=[]; disagreements=[]
    for item in selection['items']:
        rid=item['request_id']; label=ai['labels'][rid]
        if label['reviewed'] is not True or not label['note'] or label['label'] not in selection['stratum_sizes']:
            raise ValueError('Unreviewed or invalid AI label: '+rid)
        current={e['id'] for e in packets[rid]['packet']['evidence'] if e['id'].startswith('C_')}
        if label['evidence_id'] not in current: raise ValueError('Invalid current evidence: '+rid)
        if packets[rid]['input_sha256']!=first[rid]['input_sha256']:
            raise ValueError('Packet identity mismatch: '+rid)
        predicted=first[rid]['candidate']['purpose']; agreed=predicted==label['label']
        pairs[(predicted,label['label'])]+=1
        if item['sampling']=='probability':
            if item['stratum']!=predicted: raise ValueError('Stratum mismatch')
            strata.setdefault(predicted,[]).append(agreed)
        elif item['sampling']=='targeted': targeted.append(dict(request_id=rid,first=predicted,ai=label['label'],agreed=agreed))
        else: raise ValueError('Unknown sampling mode')
        if not agreed: disagreements.append(dict(request_id=rid,first=predicted,ai=label['label'],sampling=item['sampling'],note=label['note']))
    per_stratum={k:dict(population=n,sample=len(strata[k]),agree=sum(strata[k]),rate=sum(strata[k])/len(strata[k]))
                 for k,n in selection['stratum_sizes'].items()}
    weighted=sum(v['population']*v['rate'] for v in per_stratum.values())/sum(selection['stratum_sizes'].values())
    report=dict(protocol='candidate-stratified-codex-blind-ai-audit-v1',reviewer_kind='ai',
                sample_size=len(ids),probability_sample=sum(len(v) for v in strata.values()),targeted_sample=len(targeted),
                raw_agreement=sum(k==v and n or 0 for (k,v),n in pairs.items())/len(ids),
                design_weighted_agreement=weighted,per_stratum=per_stratum,
                targeted_agreement=sum(x['agreed'] for x in targeted),targeted=targeted,
                confusion=[dict(first=k,ai=v,n=n) for (k,v),n in sorted(pairs.items())],
                disagreements=disagreements,
                blind_packets_sha256=hashlib.sha256(blind).hexdigest(),
                ai_labels_sha256=hashlib.sha256((a.sample/'ai_labels.json').read_bytes()).hexdigest(),
                first_review_sha256=selection['review_sha256'],
                current_review_sha256=hashlib.sha256(first_bytes).hexdigest(),
                limitations=['Codex AI reference is not human gold.',
                             '70 items are drawn from 10 previously exposed development trajectories.',
                             '64 are candidate-stratified probability samples; 6 are targeted CHECK cases.',
                             'Weighted agreement is descriptive; no valid independence-based confidence interval is claimed.',
                             'Same trajectory steps are correlated and prior work may have exposed the corpus.'])
    save(a.out,report)
    print(json.dumps(dict(sample_size=len(ids),raw_agreement=report['raw_agreement'],
                          weighted_agreement=weighted,targeted_agreement=report['targeted_agreement'],
                          disagreements=len(disagreements)),ensure_ascii=False))


if __name__=='__main__': main()
