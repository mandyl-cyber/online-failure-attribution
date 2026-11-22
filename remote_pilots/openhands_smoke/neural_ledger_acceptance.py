"""Controlled native-event replay through stage sealing and the trained 7B."""
import argparse
import json
from pathlib import Path
import time
import math
import statistics
from collections import Counter

from scripts.run_stage_claim_pipeline import read,save
from segmenter.ledger import sha
from segmenter.online_ledger import OnlineLedger
from segmenter.claim_extraction import StageProcessor
from segmenter.reviewed_claim_processor import ReviewedClaimProcessor


def distribution(values):
    values=sorted(values)
    return dict(n=len(values),p50=statistics.median(values) if values else None,
                p95=values[math.ceil(len(values)*.95)-1] if values else None,
                p99=values[math.ceil(len(values)*.99)-1] if values else None,
                maximum=max(values) if values else None)


def successful_timings(directory, published_names):
    timeline=[json.loads(line) for line in (directory/'timing/timeline.jsonl').read_text(encoding='utf-8').splitlines()]
    starts={}
    measured=set()
    queue_ms,processing_ms,publication_ms=[],[],[]
    for event in timeline:
        key=(event.get('job'),event['process_id'])
        if event['kind']=='claim_started':
            starts[key]=event
        if event['kind']!='claim_completed' or key not in starts or event['job'] not in published_names or event['job'] in measured:
            continue
        measured.add(event['job'])
        start=starts[key]
        processing_ms.append((event['monotonic_ns']-start['monotonic_ns'])/1e6)
        queued_path=directory/'timing'/('queue-'+event['job'])
        if queued_path.exists():
            queued=read(queued_path)
            if queued['process_id']==start['process_id']:
                queue_ms.append((start['monotonic_ns']-queued['monotonic_ns'])/1e6)
                publication_ms.append((event['monotonic_ns']-queued['monotonic_ns'])/1e6)
    return dict(queue_ms=distribution(queue_ms),processing_ms=distribution(processing_ms),
                queued_to_publication_ms=distribution(publication_ms),
                timing_scope='Successful persisted batches only, deduplicated by job. Matching clock domains. Median and nearest-rank p95/p99.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check-inputs','dry-run','run'])
    parser.add_argument('--inputs',type=Path,default=Path('pilot/neural_ledger_acceptance_v1'))
    parser.add_argument('--out',type=Path,default=Path('out/neural_ledger_acceptance_v1'))
    parser.add_argument('--model-path',type=Path)
    parser.add_argument('--training-run',type=Path,default=Path('out/claim_sft_7b_reviewed_v2'))
    parser.add_argument('--interval',type=float,default=.05)
    parser.add_argument('--boundary-model',type=Path,help='Optional fixed JSON GRU or trusted local LR checkpoint')
    parser.add_argument('--cpu-threads',type=int,help='Set matching PyTorch CPU thread count for controlled arms')
    args=parser.parse_args()
    if args.interval<0:
        parser.error('--interval must be nonnegative')
    if args.cpu_threads is not None:
        if args.cpu_threads<1:parser.error('--cpu-threads must be positive')
        import torch
        torch.set_num_threads(args.cpu_threads)
    protocol=read(args.inputs/'protocol.json')
    for session in protocol['sessions']:
        if sha((args.inputs/session['path']).read_bytes())!=session['sha256']:
            raise ValueError('Session input changed')
    if args.command=='check-inputs':
        print(dict(sessions=len(protocol['sessions']),events=sum(s['events'] for s in protocol['sessions'])))
        return
    backend=None
    if args.command=='run':
        if args.model_path is None:
            parser.error('--model-path required')
        training=read(args.training_run/'config.json')
        if Path(training['model_path']).resolve()!=args.model_path.resolve():
            raise ValueError('Use the training base model')
        from segmenter.claim_sft_backend import ClaimSFTBackend
        backend=ClaimSFTBackend(args.model_path,args.training_run/'adapter')
    processor=ReviewedClaimProcessor(StageProcessor(backend))
    config=dict(mode=args.command,protocol_sha256=sha((args.inputs/'protocol.json').read_bytes()),
                processor=processor.identity,interval=args.interval,boundary='existing_phase_rules')
    if args.boundary_model is not None:
        config.update(boundary='supervised_candidate',boundary_model_sha256=sha(args.boundary_model.read_bytes()))
    if args.cpu_threads is not None:config['cpu_threads']=args.cpu_threads
    if (args.out/'config.json').exists() and read(args.out/'config.json')!=config:
        raise ValueError('Acceptance run configuration changed')
    save(args.out/'config.json',config)
    reports=[]
    for session in protocol['sessions']:
        directory=args.out/session['session_id']
        events=[json.loads(line) for line in (args.inputs/session['path']).read_text(encoding='utf-8').splitlines()]
        def open_runtime():
            return OnlineLedger(directory,session['session_id'],claim_processor=processor,input_mode='replay',
                                boundary_model=args.boundary_model)
        runtime=open_runtime()
        for index,event in enumerate(events):
            runtime.ingest(event)
            if index==len(events)//2:
                runtime.close()
                runtime=open_runtime()
            if args.interval:
                time.sleep(args.interval)
        runtime.drain()
        runtime.close()
        before={p.name:p.read_bytes() for p in runtime.results.glob('*.json')}
        before_jobs={p.name:p.read_bytes() for p in runtime.jobs.glob('*.json')}
        before_stages=(directory/'stages.json').read_bytes()
        restored=open_runtime()
        restored.drain()
        restored.close()
        after={p.name:p.read_bytes() for p in restored.results.glob('*.json')}
        if before!=after:
            raise ValueError('Published batches changed after recovery')
        if before_jobs!={p.name:p.read_bytes() for p in restored.jobs.glob('*.json')} or before_stages!=(directory/'stages.json').read_bytes():
            raise ValueError('Boundary stages or sealed jobs changed after recovery')
        batches=[read(p) for p in restored.results.glob('*.json')]
        success_names={name for name,payload in after.items() if json.loads(payload)['status']=='ready'}
        partial_names={name for name,payload in after.items() if json.loads(payload)['status']=='partial'}
        timing=successful_timings(directory,success_names)
        jobs=len(list(restored.jobs.glob('*.json')))
        attempts=[read(p) for p in (directory/'review_backend/extraction').glob('*/status.json')]
        report=dict(session=session['session_id'],events=len(events),steps=len(restored.segmenter.steps),
            boundary=config['boundary'],
            sealed_jobs=jobs,published_receipts=len(batches),published_batches=len(success_names),
            evidence_only_batches=sum(b['status']=='evidence_only' for b in batches),unpublished_jobs=jobs-len(batches),
            partial_batches=sum(b['status']=='partial' for b in batches),
            usable_claim_batches=len(success_names|partial_names),
            partial_batch_timings=successful_timings(directory,partial_names),
            generation_parts_attempted=sum(len(b.get('extraction_details',{}).get('parts',[])) for b in batches),
            partition_diagnostics=[dict(job_id=b['job_id'],status=b['status'],
                input_complete=b['extraction_details']['input_complete'],
                parts=b['extraction_details']['parts'],oversized_steps=b['extraction_details']['oversized_steps'],
                unprocessed_pages=b['extraction_details'].get('unprocessed_pages',[]),
                rejected_claim_count=len(b['extraction_details']['rejected_claims']),
                duplicate_claim_count=len(b['extraction_details']['duplicate_claims']))
                for b in batches if 'extraction_details' in b],
            failed_extractions=sum(a['status']=='failed' for a in attempts),attempts=sum(a['attempts'] for a in attempts),
            recovered_batches_identical=True,terminal_seen=restored.finished,
            recovered_stages_and_jobs_identical=True,
            **timing,
            signals=sum(len(b.get('observation_signals',[])) for b in batches),
            signal_counts=dict(Counter(s['kind'] for b in batches for s in b.get('observation_signals',[]))),
            uncited_signals=sum(len(b.get('uncited_signal_ids',[])) for b in batches),
            claims_needing_review=sum(c.get('semantic_status')=='needs_review' for b in batches for c in b.get('claims',[])),
            generation_metrics=[b.get('metrics',{}) for b in batches],semantic_quality='pending_AI_review')
        reports.append(report)
        save(args.out/'summary.json',dict(mode=args.command,sessions=reports,
            timing_scope='Controlled replay timings within matching observer clock domains. Replay speed controls queue load. Midpoint restart gracefully drains workers.'))
        print(report,flush=True)


if __name__=='__main__':
    main()
