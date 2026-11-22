"""Run the fixed 7B SFT extractor on frozen prospective sealed jobs; no training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from scripts.run_stage_claim_pipeline import read,save
from segmenter.ledger import sha,encoded
from segmenter.claim_extraction import StageProcessor,build_request
from segmenter.reviewed_claim_processor import ReviewedClaimProcessor
from segmenter.claim_sft_backend import ClaimSFTBackend
from segmenter import stage_claim_summary as core
from segmenter.adapters.openhands import OpenHandsStreamAdapter
from segmenter.input_timeline import validate_event


def verify_inputs(inputs):
    protocol=read(inputs/'protocol.json')
    for filename,key in [('references.json','reference_sha256'),('stage_jobs.jsonl','jobs_sha256')]:
        if sha((inputs/filename).read_bytes())!=protocol[key]:
            raise ValueError('Frozen file changed: '+filename)
    if not protocol['frozen_before_model_calls'] or protocol['posthoc_verification_in_input']:
        raise ValueError('Invalid reference freeze protocol')
    steps={}
    for session in protocol['sessions']:
        path=inputs/session['path']
        if not path.resolve().is_relative_to(inputs.resolve()) or sha(path.read_bytes())!=session['sha256']:
            raise ValueError('Native input changed')
        adapter=OpenHandsStreamAdapter(fold_agent_messages=False)
        current={}
        for line in path.read_text(encoding='utf-8').splitlines():
            event=json.loads(line)
            validate_event(adapter,event)
            current.update({s.idx:s for s in adapter.feed(event)})
        if adapter._pending or adapter.diagnostics:
            raise ValueError('Incomplete native input')
        steps[session['session_id']]=current
    jobs=[json.loads(line) for line in (inputs/'stage_jobs.jsonl').read_text(encoding='utf-8').splitlines()]
    known={}
    for row in jobs:
        job=row['job']
        if sha(encoded(job))!=row['input_sha256']:
            raise ValueError('Sealed job changed')
        build_request(job)
        for evidence in job['evidence']:
            step=steps[row['session']][evidence['source_step_id']]
            if getattr(step,evidence['field'])!=evidence['text']:
                raise ValueError('Non-native evidence in frozen job')
        known[job['job_id']]=job
    facts=read(inputs/'references.json')['facts']
    for fact in facts:
        step=steps[fact['session']][fact['step_id']]
        text=getattr(step,fact['source_field'])
        anchor=fact['anchor']
        if (text!=fact['record_text'] or sha(text.encode())!=fact['record_sha256']
                or text[anchor['start']:anchor['end']]!=anchor['quote']):
            raise ValueError('Reference evidence changed')
        if fact['sealed_job_id'] is not None:
            if fact['sealed_job_id'] not in known:
                raise ValueError('Reference names an unknown job')
            job=known[fact['sealed_job_id']]
            if not job['stage']['start_step_id']<=fact['step_id']<=job['stage']['end_step_id']:
                raise ValueError('Fact outside sealed stage')
    return jobs,protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','run'])
    parser.add_argument('--inputs',type=Path,default=Path('pilot/claim_prospective_eval_v1'))
    parser.add_argument('--out',type=Path,default=Path('out/claim_prospective_eval_v1'))
    parser.add_argument('--model-path',type=Path)
    parser.add_argument('--training-run',type=Path,default=Path('out/claim_sft_7b_reviewed_v2'))
    args=parser.parse_args()
    jobs,protocol=verify_inputs(args.inputs)
    if args.command=='check':
        print(dict(sealed_jobs=len(jobs),native_sessions=len(protocol['sessions']),input_hashes='passed',model_calls=0))
        return
    if args.model_path is None:
        parser.error('--model-path required')
    if Path(read(args.training_run/'config.json')['model_path']).resolve()!=args.model_path.resolve():
        raise ValueError('Use the original training base model')
    backend=ClaimSFTBackend(args.model_path,args.training_run/'adapter')
    processor=ReviewedClaimProcessor(StageProcessor(backend))
    config=dict(protocol_sha256=sha((args.inputs/'protocol.json').read_bytes()),processor=processor.identity,
        model_path=str(args.model_path.resolve()),mode='fixed_sft_frozen_sealed_jobs',training=False,
        reference_used_as_model_input=False,extractor_version='v4',boundary='frozen_phase_rules')
    if (args.out/'config.json').exists() and read(args.out/'config.json')!=config:
        raise ValueError('Run identity changed; use a new output directory')
    save(args.out/'config.json',config)
    records=[]
    for index,row in enumerate(jobs,1):
        job_path=args.out/'stage_jobs'/f'{index:04d}.json'
        if job_path.exists() and read(job_path)!=row['job']:
            raise ValueError('Saved frozen job changed')
        save(job_path,row['job'])
        result_dir=args.out/'claim_batches'
        cached=(result_dir/job_path.name).exists()
        started=time.perf_counter()
        batch=processor(job_path,result_dir) # Only job evidence; references never passed.
        records.append(dict(session=row['session'],job_id=row['job']['job_id'],file=job_path.name,
            status=batch['status'],claims=len(batch.get('claims',[])),cached=cached,
            current_call_seconds=time.perf_counter()-started,metrics=batch.get('metrics',{})))
        summary=dict(jobs=len(jobs),processed=len(records),statuses=dict(Counter(r['status'] for r in records)),
            claims=sum(r['claims'] for r in records),newly_processed=sum(not r['cached'] for r in records),
            semantic_quality='pending_AI_review_against_frozen_references',records=records,
            reference_sha256=protocol['reference_sha256'],
            scope='Extractor-only frozen sealed jobs. No streaming queue latency, boundary F1, training or automatic entailment score.')
        save(args.out/'summary.json',summary)
        print(dict(done=index,total=len(jobs),session=row['session'],status=batch['status'],cached=cached),flush=True)


if __name__=='__main__':
    main()
