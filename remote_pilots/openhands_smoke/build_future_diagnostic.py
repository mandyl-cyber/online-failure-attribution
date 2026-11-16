"""Offline-only suffix packets; never feed this directory to a predictor."""
import hashlib
import json
from pathlib import Path

from segmenter.adapters.openhands import load_trajectory
from segmenter.audit_plan_payloads import find_calls
from segmenter.relation_evidence import clipped


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    source=Path('pilot/swe_relation_balanced_v2')
    out=Path('pilot/swe_future_diagnostic_v1');out.mkdir(exist_ok=False)
    protocol=dict(purpose='Offline diagnostic only; no training, no performance or causal impossibility claim.',
        future_actions=3,selection='Same frozen 16 cases, including uncertain and near-end cases; no replacements.',
        review='Prior AI judgments frozen before suffix generation. Reviewer already knows gold buckets; not blinded.',
        categories=['prefix_supported','future_adds_support','granularity_ambiguous','insufficient_suffix'],
        interpretation='Future support does not prove the boundary could not be inferred from a better prefix representation.',
        frozen_hashes={str(source/name):sha(source/name) for name in ['packets.json','answer_key.json','ai_review.json']})
    save(out/'protocol.json',protocol)
    save(out/'prefix_review_frozen.json',read(source/'ai_review.json'))
    records={r['traj_id']:r for r in read('eval/boundary_supervised/data_fixed/records.json')}
    packets=[]
    for k in read(source/'answer_key.json'):
        r=records[k['traj_id']];path=Path(r['raw_source']);target=k['target_step']
        assert sha(path)==k['source_sha256']
        steps,_=load_trajectory(path.parent if r['format']=='completions' else path,
                               base=r['ids'][0],expect_steps=r['n_steps'])
        steps=steps[:r['n_steps']];h=hashlib.sha256()
        for s in steps:h.update(json.dumps([s.action,s.observation],ensure_ascii=False).encode())
        assert h.hexdigest()==r['content_sha256'] and [s.idx for s in steps]==r['ids']
        suffix=[s for s in steps if s.idx>target][:3]
        plans=find_calls(read(path),{s.raw_ref.get('action_id') for s in suffix if s.tool in ('task_tracking','task_tracker')})
        packets.append(dict(case_id=k['case_id'],target_step=target,offline_only=True,
            available_future_steps=len(suffix),trajectory_ends_within_window=len(suffix)<3,
            future=[dict(idx=s.idx,tool_class=s.tool_class,action=clipped(s.action,2400),
                         observation=clipped(s.observation,3200),public_plan=plans.get(s.raw_ref.get('action_id')))
                    for s in suffix]))
    save(out/'future_packets.json',packets)
    print(json.dumps([dict(case_id=p['case_id'],target=p['target_step'],n=p['available_future_steps']) for p in packets]))


if __name__=='__main__':main()
