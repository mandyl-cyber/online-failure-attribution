"""Collect frozen new synthetic tasks; failures remain in the acceptance pool."""
import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from scripts.run_stage_claim_pipeline import read,save


def checked_tasks(inputs):
    protocol=read(inputs/'protocol.json')
    result=[]
    for row in protocol['tasks']:
        path=inputs/row['path']
        if not path.resolve().is_relative_to(inputs.resolve()):
            raise ValueError('Task bundle outside inputs')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Frozen task changed: '+row['task_id'])
        result.append((row,path))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','run'])
    parser.add_argument('--inputs',type=Path,default=Path('pilot/claim_prospective_tasks_v1'))
    parser.add_argument('--out',type=Path,default=Path('out/claim_prospective_native_v1'))
    parser.add_argument('--limit',type=int,default=4)
    args=parser.parse_args()
    tasks=checked_tasks(args.inputs)
    if not 1<=args.limit<=len(tasks):
        parser.error('--limit must be between 1 and task count')
    if args.command=='check':
        print(dict(frozen_tasks=len(tasks),claim_model_calls=0,task_hashes='passed'))
        return
    if sys.platform!='linux':
        raise SystemExit('Run collection on the Linux Docker host; local fixture checking needs no Docker.')
    from .run import APP,RUNTIME
    subprocess.run(['docker','ps'],check=True,stdout=subprocess.DEVNULL)
    for image in (APP,RUNTIME):
        subprocess.run(['docker','image','inspect',image],check=True,stdout=subprocess.DEVNULL)
    with urllib.request.urlopen('http://127.0.0.1:30001/v1/models',timeout=10) as response:
        if not any(m['id']=='qwen-local' for m in json.load(response)['data']):
            raise ValueError('qwen-local unavailable; start the tool-compatible server before collection')
    config=dict(protocol_sha256=hashlib.sha256((args.inputs/'protocol.json').read_bytes()).hexdigest(),
                mode='native_agent_collection_only',claim_outputs_generated=False)
    if (args.out/'config.json').exists() and read(args.out/'config.json')!=config:
        raise ValueError('Collection identity changed; use a new output directory')
    save(args.out/'config.json',config)
    reports=[]
    for selected,path in tasks[:args.limit]:
        task_id=selected['task_id']
        directory=args.out/path.stem
        receipt=directory/'receipt.json'
        if receipt.exists():
            result=read(receipt)
            if result['task_id']!=task_id or result['bundle_sha256']!=selected['sha256']:
                raise ValueError('Stored task identity changed')
            if result['status']=='collected':
                saved=list(directory.glob('runs/run-*/state/store/sessions/*/events/*.json'))
                if len(saved)!=result['native_event_files']:
                    raise ValueError('Stored native events missing or changed in count')
            # Never overwrite an interrupted or failed first attempt with a
            # successful retry and silently improve the acceptance set.
            reports.append(result)
            continue
        result=dict(task_id=task_id,bundle_sha256=selected['sha256'],status='started',native_event_files=0)
        save(receipt,result)
        command=[sys.executable,'-m','remote_pilots.openhands_smoke.run','--task-bundle',str(path.resolve()),
                 '--out-root',str((directory/'runs').resolve()),'--prompt-mode','action-guided','--quiet','--collect']
        with (directory/'driver.log').open('w',encoding='utf-8') as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        run_dirs=sorted((directory/'runs').glob('run-*'))
        run_dirs=[p for p in run_dirs if p.is_dir()]
        if len(run_dirs)>1:
            raise ValueError('More than one native attempt for frozen task')
        native=list(directory.glob('runs/run-*/state/store/sessions/*/events/*.json'))
        result.update(status='collected' if native else 'collection_failed',
                      driver_exit_code=completed.returncode,native_event_files=len(native))
        if run_dirs:
            result['run_directory']=run_dirs[0].relative_to(args.out).as_posix()
            if (run_dirs[0]/'result.json').exists():
                detail=read(run_dirs[0]/'result.json')
                result.update(controller_exit_code=detail['controller_exit_code'],verification_exit_code=detail['verification_exit_code'],
                              visible_tests_unchanged=detail['visible_tests_unchanged'])
        save(receipt,result)
        reports.append(result)
        print(dict(task=task_id,status=result['status'],native_events=len(native)),flush=True)
    summary=dict(selected_total=len(tasks),attempts_in_requested_prefix=len(reports),requested_prefix=args.limit,
        all_selected_attempted=len(reports)==len(tasks),
        native_sessions_available=sum(r['status']=='collected' for r in reports),
        interrupted_attempts=sum(r['status']=='started' for r in reports),
        collection_failed=sum(r['status']=='collection_failed' for r in reports),
        agent_tasks_passed=sum(r.get('controller_exit_code')==0 and r.get('verification_exit_code')==0
                               and r.get('visible_tests_unchanged') is True for r in reports),
        claim_outputs_generated=False,semantic_references_frozen=False,tasks=reports,
        scope='New synthetic engineering tasks. No semantic claim accuracy. Preserve failed and interrupted attempts; do not select successful runs only.')
    save(args.out/'summary.json',summary)
    print(dict(summary=str(args.out/'summary.json'),native_sessions_available=summary['native_sessions_available']))
    return int(summary['collection_failed']>0 or summary['interrupted_attempts']>0)


if __name__=='__main__':
    sys.exit(main())
