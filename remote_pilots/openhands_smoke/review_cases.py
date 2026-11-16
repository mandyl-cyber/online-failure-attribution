"""Deterministic diagnostic sample; preserves prefix/future separation."""
import hashlib
import json
from pathlib import Path

from segmenter.adapters.openhands import load_trajectory
from segmenter.eval_online import matched_pairs


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    out = Path('pilot/swe_case_review_v1')
    out.mkdir(exist_ok=False)
    base = Path('eval/boundary_supervised/data_fixed')
    records = {r['traj_id']: r for r in read(base/'records.json')}
    labels = read(base/'labels.json')
    preds = read('pilot/swe_only_boundary_v1/predictions.json')['swe_train_swe_dev']
    features = {r['traj_id']: r['rows'] for r in map(json.loads,(base/'features.jsonl').read_text(encoding='utf-8').splitlines())}
    buckets = {'plan_read_fp': [], 'read_read_fn': []}
    for tid, events in preds.items():
        gold = labels[tid]['boundaries']
        pairs = matched_pairs(events,gold,1,0)
        pi, gi = {i for i,j in pairs}, {j for i,j in pairs}
        lookup = {r['idx']:r['features'] for r in features[tid]}
        for i,e in enumerate(events):
            idx=e['boundary_step_id']; f=lookup[idx]
            if i not in pi and f.get('lag1:tool')=='PLAN' and f['lag0:tool']=='READ':
                buckets['plan_read_fp'].append((tid,idx))
        for j,idx in enumerate(gold):
            f=lookup[idx]
            if j not in gi and f.get('lag1:tool')=='READ' and f['lag0:tool']=='READ':
                buckets['read_read_fn'].append((tid,idx))
    cases=[]
    for bucket,candidates in buckets.items():
        used=set()
        ordered=sorted(candidates,key=lambda x:hashlib.sha256(json.dumps(x).encode()).hexdigest())
        for tid,idx in ordered:
            r=records[tid]
            if r['group'] in used:continue
            used.add(r['group'])
            path=Path(r['raw_source'])
            steps,meta=load_trajectory(path.parent if r['format']=='completions' else path,
                                       base=r['ids'][0],expect_steps=r['n_steps'])
            steps=steps[:r['n_steps']]
            assert [s.idx for s in steps]==r['ids']
            digest=hashlib.sha256()
            for s in steps:digest.update(json.dumps([s.action,s.observation],ensure_ascii=False).encode())
            assert digest.hexdigest()==r['content_sha256'], 'Source mismatch'
            def export(s):
                return dict(idx=s.idx,tool=s.tool,tool_class=s.tool_class,action=s.action,
                            observation=s.observation,paths=s.touched_paths,raw_ref=s.raw_ref)
            cases.append(dict(case_id=f'{bucket}_{len(used)}',bucket=bucket,traj_id=tid,idx=idx,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source=str(path),
                alignment_verified=True, gold_boundaries=labels[tid]['boundaries'],
                prefix_window=[export(s) for s in steps if idx-3<=s.idx<=idx],
                future_audit_only=[export(s) for s in steps if idx<s.idx<=idx+2]))
            if len(used)==6:break
    (out/'packets.json').write_text(json.dumps(dict(policy='6 per bucket, SHA256 ordering, distinct task/content group within bucket; prefix last four steps and future two separated; no thought; diagnostic AI review only',pool_sizes={k:len(v) for k,v in buckets.items()},cases=cases),ensure_ascii=False,indent=2),encoding='utf-8')
    print('PACKETS='+str(out/'packets.json'))


if __name__=='__main__':main()
