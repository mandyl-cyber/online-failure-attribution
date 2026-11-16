"""Build task + complete observed-prefix evidence for spatially unmatched cuts."""
import hashlib
import json
from pathlib import Path

from segmenter.adapters.openhands import load_trajectory
from segmenter.audit_plan_payloads import find_calls


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    out=Path('pilot/swe_full_prefix_review_v1')
    out.mkdir(exist_ok=False)
    records={r['traj_id']:r for r in read('eval/boundary_supervised/data_fixed/records.json')}
    labels=read('eval/boundary_supervised/data_fixed/labels.json')
    errors=read('pilot/swe_timing_audit_v1/swe_error_details.json')
    selected=[]
    for kind in ('FP_outside_tolerance','FN_no_nearby_prediction'):
        candidates=[e for e in errors if e['delay']==0 and e['kind']==kind]
        candidates.sort(key=lambda e:hashlib.sha256(json.dumps(e,sort_keys=True).encode()).hexdigest())
        used=set()
        for e in candidates:
            r=records[e['traj_id']]
            if r['group'] in used:continue
            used.add(r['group']); selected.append(e)
            if len(used)==3:break
    packets=[]; key=[]
    for i,e in enumerate(selected,1):
        tid=e['traj_id'];r=records[tid];idx=e.get('boundary_step_id',e.get('gold_boundary'))
        path=Path(r['raw_source']);raw=read(path)
        steps,meta=load_trajectory(path.parent if r['format']=='completions' else path,
                                   base=r['ids'][0],expect_steps=r['n_steps'])
        steps=steps[:r['n_steps']]
        digest=hashlib.sha256()
        for s in steps:digest.update(json.dumps([s.action,s.observation],ensure_ascii=False).encode())
        assert digest.hexdigest()==r['content_sha256'] and [s.idx for s in steps]==r['ids']
        prefix=[s for s in steps if s.idx<=idx]
        wanted={s.raw_ref.get('action_id') for s in prefix if s.tool in ('task_tracking','task_tracker')}
        plans=find_calls(raw,wanted) if r['format']=='completions' else {}
        users=[m.get('content') for m in raw.get('messages',[]) if m.get('role')=='user'] if isinstance(raw,dict) else []
        task=users[0] if users else meta.get('task','')
        packets.append(dict(case_id=f'case_{i:02d}',target_step=idx,initial_user_message=task,
            prefix=[dict(idx=s.idx,tool_class=s.tool_class,tool=s.tool,action=s.action,
                         observation=s.observation,paths=s.touched_paths,
                         public_plan=plans.get(s.raw_ref.get('action_id'))) for s in prefix]))
        key.append(dict(case_id=f'case_{i:02d}',traj_id=tid,kind=e['kind'],target_step=idx,
                        gold_boundaries=labels[tid]['boundaries'],source=str(path),
                        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),alignment_verified=True))
    (out/'prefix_packets.json').write_text(json.dumps(packets,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'answer_key.json').write_text(json.dumps(key,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Created',len(packets),'full-prefix packets; no future steps, thought or gold in packet file.')


if __name__=='__main__':main()
