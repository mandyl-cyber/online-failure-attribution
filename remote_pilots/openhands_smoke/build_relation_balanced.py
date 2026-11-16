"""Frozen deterministic diagnostic sampling; never trains on the sampled cases."""
import hashlib
import json
from pathlib import Path

from segmenter.adapters.openhands import load_trajectory
from segmenter.audit_plan_payloads import find_calls
from segmenter.eval_online import matched_pairs
from segmenter.relation_evidence import RelationEvidence


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def main():
    out=Path('pilot/swe_relation_balanced_v2')
    out.mkdir(exist_ok=True)
    protocol=dict(version=2,per_bucket=4,seed='relation-balanced-v2',
        buckets=['TP_matched','TN_far','FP_far','FN_far'],
        matching='Existing task-group OOF SWE predictions; tolerance 1, delay 0.',
        sampling='SHA256 order; distinct groups globally; exclude six v1 case groups. No manual substitutions.',
        definitions={'TP_matched':'matched prediction target; gold can be one step later',
                     'TN_far':'noninitial step; no prediction here and no gold within +/-1',
                     'FP_far':'unmatched prediction with no gold within +/-1',
                     'FN_far':'unmatched gold with no prediction boundary within +/-1'},
        purpose='Diagnostic enriched sample, not representative accuracy or held-out test.',
        review='AI review of bounded evidence; not human labels. No training authorization from labels.')
    protocol_path=out/'protocol.json'
    if protocol_path.exists() and read(protocol_path)!=protocol:
        raise ValueError('Frozen protocol differs')
    protocol_path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    base=Path('eval/boundary_supervised/data_fixed')
    records={r['traj_id']:r for r in read(base/'records.json')}
    labels=read(base/'labels.json')
    preds=read('pilot/swe_only_boundary_v1/predictions.json')['swe_train_swe_dev']
    used={records[k['traj_id']]['group'] for k in read('pilot/swe_full_prefix_review_v1/answer_key.json')}
    pools={k:[] for k in protocol['buckets']}
    for tid,events in preds.items():
        gold=labels[tid]['boundaries']; pairs=matched_pairs(events,gold,1,0)
        pi={i for i,j in pairs};gi={j for i,j in pairs}
        predicted={e['boundary_step_id'] for e in events}
        for i,e in enumerate(events):
            idx=e['boundary_step_id']
            if i in pi:pools['TP_matched'].append((tid,idx))
            elif all(abs(idx-g)>1 for g in gold):pools['FP_far'].append((tid,idx))
        for j,idx in enumerate(gold):
            if j not in gi and all(abs(idx-p)>1 for p in predicted):pools['FN_far'].append((tid,idx))
        for idx in records[tid]['ids'][1:]:
            if idx not in predicted and all(abs(idx-g)>1 for g in gold):pools['TN_far'].append((tid,idx))
    selection=[]
    for bucket in protocol['buckets']:
        count=0
        for tid,idx in sorted(pools[bucket],key=lambda x:digest([protocol['seed'],bucket,x])):
            if records[tid]['group'] in used:continue
            used.add(records[tid]['group'])
            selection.append(dict(bucket=bucket,traj_id=tid,target_step=idx))
            count+=1
            if count==protocol['per_bucket']:break
        if count!=protocol['per_bucket']:raise ValueError('Insufficient distinct groups')
    selection.sort(key=lambda x:digest([protocol['seed'],x]))
    packets=[];key=[]
    for number,s in enumerate(selection,1):
        r=records[s['traj_id']];path=Path(r['raw_source'])
        if r['format']!='completions':raise ValueError('Unexpected SWE source format')
        raw=read(path)
        steps,meta=load_trajectory(path.parent,base=r['ids'][0],expect_steps=r['n_steps'])
        steps=steps[:r['n_steps']]
        h=hashlib.sha256()
        for step in steps:h.update(json.dumps([step.action,step.observation],ensure_ascii=False).encode())
        if h.hexdigest()!=r['content_sha256'] or [x.idx for x in steps]!=r['ids']:
            raise ValueError('Source alignment mismatch')
        prefix=[x for x in steps if x.idx<=s['target_step']]
        plans=find_calls(raw,{x.raw_ref.get('action_id') for x in prefix if x.tool in ('task_tracking','task_tracker')})
        users=[m.get('content') for m in raw.get('messages',[]) if m.get('role')=='user']
        builder=RelationEvidence(users[0] if users else meta.get('task',''))
        for x in prefix:
            evidence=builder.feed(dict(idx=x.idx,tool=x.tool,tool_class=x.tool_class,
                action=x.action,observation=x.observation,public_plan=plans.get(x.raw_ref.get('action_id'))))
        cid=f'relation_{number:02d}'
        packets.append(dict(case_id=cid,evidence=evidence))
        key.append(dict(case_id=cid,**s,group=r['group'],gold_boundaries=labels[s['traj_id']]['boundaries'],
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),alignment_verified=True))
    for name,value in [('packets.json',packets),('answer_key.json',key),
                       ('selection_audit.json',dict(pool_sizes={k:len(v) for k,v in pools.items()},
                                                   packet_sha256=digest(packets),selected=len(packets)))]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(selected=len(packets),pool_sizes={k:len(v) for k,v in pools.items()})))


if __name__=='__main__':main()
