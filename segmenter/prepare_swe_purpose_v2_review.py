"""Export blind packets and a separate pending AI review table for v2 candidates."""
import argparse
import json
from pathlib import Path

from .prepare_purpose_round2 import write
from .purpose_pilot_remote_v2 import jobs, read, validate, verify_package


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--first',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists(): raise SystemExit('Refusing overwrite: '+str(args.out))
    verify_package(args.package)
    run=read(args.first/'run_manifest.json')
    args.out.mkdir(parents=True)
    entries={}
    seen_construct={}
    final_step={t['trajectory_id']:t['steps'][-1]['idx'] for t in
                (json.loads(line) for line in (args.package/'trajectories.jsonl').read_text(encoding='utf-8').splitlines())}
    with (args.out/'blind_packets.jsonl').open('w',encoding='utf-8') as f:
        for job in jobs(args.package):
            f.write(json.dumps(job,ensure_ascii=False)+'\n')
            entry=dict(input_sha256=job['input_sha256'],candidate=None,label_source=None,
                       priority_reason=None,reviewed=False,final_label=None,review_note='')
            try:
                r=read(args.first/'responses'/(job['request_id']+'.json'))
                if r['identity']!=run['identity'] or r['input_sha256']!=job['input_sha256'] or r['request_id']!=job['request_id']:
                    raise ValueError('Run/input identity mismatch')
                entry['candidate']=validate(r['answer'],job['packet'])
                entry['label_source']=r.get('label_source','qwen')
                if r.get('label_source')=='codex_ai_adjudication': entry['priority_reason']='codex_ai_adjudication'
                elif r.get('review_priority'): entry['priority_reason']='normalized_model_alias'
                elif r.get('label_source')=='deterministic_control_rule': entry['priority_reason']='control_rule'
                elif entry['candidate']['purpose']=='CHECK' and not seen_construct.get(job['packet']['trajectory_id']):
                    entry['priority_reason']='check_before_candidate_construction'
                elif entry['candidate']['purpose']=='DELIVER' and job['packet']['current_step_id']<final_step[job['packet']['trajectory_id']]:
                    entry['priority_reason']='delivery_before_trajectory_end'
                if entry['candidate']['purpose']=='CONSTRUCT': seen_construct[job['packet']['trajectory_id']]=True
            except (OSError,ValueError,KeyError,TypeError) as e:
                entry['priority_reason']='missing_or_invalid: '+str(e)
            entries[job['request_id']]=entry
    write(args.out/'review_pending.json',dict(reviewer_kind='ai',reviewer_id=None,
          note='Blind packets omit the first candidate. Review independently before opening review_pending.json. '
               'No entry is approved; agreement and structural validity are not semantic accuracy.',
          first_run_identity=run['identity'],labels=entries))
    print(json.dumps(dict(total=len(entries),candidates=sum(x['candidate'] is not None for x in entries.values()),
                          pending=sum(not x['reviewed'] for x in entries.values())),ensure_ascii=False))


if __name__=='__main__': main()
