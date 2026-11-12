"""Recover public task-tracker arguments by exact native call ID, as a sidecar.

Does not mutate the legacy adapter or any frozen experiment inputs. Task IDs are
retained for audit/reconciliation, never interpreted as gold stage annotations.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def public_payload(args):
    tasks=args.get('task_list')
    if not isinstance(tasks,list):return None
    return dict(command=str(args.get('command','')),task_list=[
        {k:v[k] for k in ('id','title','status') if k in v and isinstance(v[k],str)}
        for v in tasks if isinstance(v,dict)])


def find_calls(obj,wanted):
    found={};stack=[obj]
    while stack:
        value=stack.pop()
        if isinstance(value,dict):
            call_id=value.get('id')
            if isinstance(call_id,str) and call_id in wanted and isinstance(value.get('function'),dict):
                fn=value['function']
                if fn.get('name') in ('task_tracker','task_tracking'):
                    args=fn.get('arguments') or {}
                    if isinstance(args,str):args=json.loads(args)
                    payload=public_payload(args)
                    if payload is not None:
                        if call_id in found and found[call_id]!=payload:
                            raise ValueError('Conflicting contents for native call '+call_id)
                        found[call_id]=payload
            stack.extend(value.values())
        elif isinstance(value,list):stack.extend(value)
    return found


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='eval/plan_payload_audit')
    a=ap.parse_args();out=Path(a.out)
    if out.exists():raise SystemExit('Output exists')
    out.mkdir(parents=True)
    data=Path('eval/boundary_supervised/data_fixed')
    records=json.loads((data/'records.json').read_text(encoding='utf-8'))
    texts={t['traj_id']:t['rows'] for t in map(json.loads,Path('eval/boundary_encoder/text_v1/steps.jsonl').read_text(encoding='utf-8').splitlines())}
    counts=Counter();affected=set();recovered=[];missing=[];sources=[]
    for r in records:
        rows=[x for x in texts[r['traj_id']] if x['tool']=='task_tracking']
        counts['task_tracking_steps']+=len(rows)
        empty=[x for x in rows if not x['action'].strip()]
        counts['empty_action_steps']+=len(empty)
        if not empty:continue
        affected.add(r['traj_id'])
        mapping={m['idx']:m['raw_ref']['action_id'] for m in r['mapping']}
        wanted={mapping[x['idx']] for x in rows}
        if r['format']!='completions':
            missing.extend(dict(traj_id=r['traj_id'],idx=x['idx'],reason='unsupported raw format') for x in empty)
            continue
        path=Path(r['raw_source']);blob=path.read_bytes()
        calls=find_calls(json.loads(blob.decode('utf-8')),wanted)
        sources.append(dict(traj_id=r['traj_id'],path=str(path),sha256=hashlib.sha256(blob).hexdigest()))
        for x in rows:
            payload=calls.get(mapping[x['idx']])
            if payload is not None:
                recovered.append(dict(traj_id=r['traj_id'],idx=x['idx'],native_call_id=mapping[x['idx']],
                    was_empty=not x['action'].strip(),payload=payload))
            elif not x['action'].strip():
                missing.append(dict(traj_id=r['traj_id'],idx=x['idx'],reason='native call/task_list not found'))
        if len(affected)%50==0:print(f'{len(affected)} affected trajectories inspected',flush=True)
    counts['affected_trajectories']=len(affected)
    counts['recovered_empty_steps']=sum(x['was_empty'] for x in recovered)
    counts['unrecovered_empty_steps']=len(missing)
    report=dict(counts=dict(counts),missing=missing,sources=sources,
        cause='adapters.openhands._s treats every dictionary in a list as a text content part and reads only its text key; task dictionaries use id/title/status',
        interpretation='Public tool input was dropped, not absent from the source. This limits conclusions from old text representations. No evidence yet that restoring it improves boundary quality.',
        next_input_version='Use this exact-call-ID sidecar for a separate controlled input repair; keep original cached features, splits and experiments immutable',
        thought_included=False,labels_used=False,legacy_files_modified=False)
    (out/'payloads.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in recovered)+'\n',encoding='utf-8')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['counts']),flush=True)


if __name__=='__main__':main()
