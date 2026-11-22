"""Same event streams and claim extractor, rules versus fixed CPU GRU candidate."""
import argparse
from collections import Counter
from pathlib import Path
import json
import subprocess
import sys

from scripts.run_stage_claim_pipeline import read,save
from segmenter.ledger import sha,encoded
from .neural_ledger_acceptance import distribution


def timing(directory):
    rows=[json.loads(x) for x in (directory/'timing/timeline.jsonl').read_text(encoding='utf-8').splitlines()]
    ready={}
    inference=[];published=[]
    for row in rows:
        if row.get('recovery'):continue
        key=(row['process_id'],row.get('step_id'))
        if row['kind']=='boundary_inference':inference.append(row['inference_ms'])
        if row['kind']=='step_ready':ready[key]=row
        if row['kind']=='boundary_published' and key in ready:
            published.append((row['monotonic_ns']-ready[key]['monotonic_ns'])/1e6)
    return dict(model_feature_and_inference_ms=distribution(inference),
        ready_to_boundary_published_ms=distribution(published),
        scope='Matching observer process clocks, excluding recovery. Replay, not live latency SLO.')


def summarize(out,inputs,model):
    facts=read(inputs/'references.json')['facts'] if (inputs/'references.json').exists() else []
    reports={}
    for arm in ('rules','neural'):
        summary=read(out/arm/'summary.json')
        config=read(out/arm/'config.json')
        expected='existing_phase_rules' if arm=='rules' else 'supervised_candidate'
        if config['boundary']!=expected:raise ValueError('Wrong boundary arm')
        sessions=[]
        for report in summary['sessions']:
            directory=out/arm/report['session']
            snapshot=read(directory/'stages.json')
            stages=snapshot['stages']
            covered=[i for s in stages for i in range(s['start_step_id'],s['end_step_id']+1)]
            if covered!=list(range(1,snapshot['observed_steps']+1)):
                raise ValueError('Stage gap/overlap')
            jobs=[read(p) for p in sorted((directory/'stage_jobs').glob('*.json'))]
            for job in jobs:
                if job['stage']['end_step_id']>job['detected_at_seq']:raise ValueError('Seal precedes evidence')
            projection=[]
            for fact in facts:
                if fact['session']!=report['session']:continue
                matches=[j for j in jobs if j['stage']['start_step_id']<=fact['step_id']<=j['stage']['end_step_id']]
                if len(matches)>1:raise ValueError('Overlapping fact assignments')
                if matches:
                    refs=[e for e in matches[0]['evidence'] if e['source_step_id']==fact['step_id'] and e['field']==fact['source_field']]
                    if len(refs)!=1 or refs[0]['text']!=fact['record_text']:raise ValueError('Fact evidence changed')
                projection.append(dict(fact_id=fact['fact_id'],importance=fact['importance'],
                    job_id=matches[0]['job_id'] if matches else None,status='sealed' if matches else 'open'))
            sessions.append(dict(session=report['session'],steps=snapshot['observed_steps'],
                sealed_jobs=len(jobs),open_stages=sum(not s['sealed'] for s in stages),
                stage_lengths=[s['end_step_id']-s['start_step_id']+1 for s in stages],
                statuses=dict(Counter(read(p)['status'] for p in (directory/'claim_batches').glob('*.json'))),
                recovery_passed=report['recovered_stages_and_jobs_identical'] and report['recovered_batches_identical'],
                timings=timing(directory),fact_projection=projection))
        reports[arm]=dict(processor=config['processor'],sessions=sessions,
            sealed_jobs=sum(s['sealed_jobs'] for s in sessions),open_stages=sum(s['open_stages'] for s in sessions))
    if reports['rules']['processor']!=reports['neural']['processor']:raise ValueError('Claim extractor differs between arms')
    result=dict(arms=reports,candidate_json_sha256=sha(model.read_bytes()),
        semantic_quality='pending_AI_comparison' if summary['mode']=='run' else 'not_measured_rule_claim_dry_run',
        default_replacement='pending_GPU_claim_quality_and_cost_acceptance',
        scope='Known synthetic development traces. Same claim extractor, frozen policy. Fact projection is input availability, not semantic coverage. No gold stage F1.')
    save(out/'comparison.json',result)
    print(dict(comparison=str(out/'comparison.json'),rules_jobs=reports['rules']['sealed_jobs'],
               neural_jobs=reports['neural']['sealed_jobs'],default_replacement=result['default_replacement']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','dry-run','run'])
    parser.add_argument('--inputs',type=Path,default=Path('pilot/claim_prospective_eval_v1'))
    parser.add_argument('--boundary-model',type=Path,default=Path('artifacts/boundary_gru_adjacent_v1/model.json'))
    parser.add_argument('--out',type=Path,default=Path('out/boundary_ledger_rollout_v1'))
    parser.add_argument('--model-path',type=Path)
    parser.add_argument('--training-run',type=Path,default=Path('out/claim_sft_7b_reviewed_v2'))
    parser.add_argument('--interval',type=float,default=.05)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(1)
    from segmenter.temporal_boundary import TemporalPredictor
    candidate=TemporalPredictor.load(args.boundary_model)
    if candidate.policy_config!=dict(threshold=.55,margin=0.,max_delay=0,adjacent_threshold=.65):
        raise ValueError('Candidate policy differs from frozen selection')
    protocol=read(args.inputs/'protocol.json')
    for session in protocol['sessions']:
        if sha((args.inputs/session['path']).read_bytes())!=session['sha256']:raise ValueError('Event stream changed')
    if args.command=='check':
        print(dict(sessions=len(protocol['sessions']),candidate_parameters=sum(p.numel() for p in candidate.net.parameters()),
                   candidate_weights='verified',policy=candidate.policy_config,claim_model_calls=0))
        return
    if args.command=='run' and args.model_path is None:parser.error('--model-path required')
    for arm in ('rules','neural'):
        command=[sys.executable,'-m','remote_pilots.openhands_smoke.neural_ledger_acceptance',
            args.command,'--inputs',str(args.inputs),'--out',str(args.out/arm),'--interval',str(args.interval),
            '--cpu-threads','1','--training-run',str(args.training_run)]
        if arm=='neural':command+=['--boundary-model',str(args.boundary_model)]
        if args.model_path is not None:command+=['--model-path',str(args.model_path)]
        args.out.mkdir(parents=True,exist_ok=True)
        with (args.out/(arm+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(dict(arm=arm,status='completed',log=str(args.out/(arm+'.log'))),flush=True)
    summarize(args.out,args.inputs,args.boundary_model)


if __name__=='__main__':main()
