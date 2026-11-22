"""Controlled native-event replay through stage sealing and the trained 7B."""
import argparse
import json
from pathlib import Path
import time

from scripts.run_stage_claim_pipeline import read,save
from segmenter.ledger import sha
from segmenter.online_ledger import OnlineLedger
from segmenter.claim_extraction import StageProcessor
from segmenter.reviewed_claim_processor import ReviewedClaimProcessor


def distribution(values):
    values=sorted(values)
    return dict(n=len(values),**{name:values[min(len(values)-1,int((len(values)-1)*p))] if values else None
                for name,p in [('p50',.5),('p95',.95),('p99',.99)]})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check-inputs','dry-run','run'])
    parser.add_argument('--inputs',type=Path,default=Path('pilot/neural_ledger_acceptance_v1'))
    parser.add_argument('--out',type=Path,default=Path('out/neural_ledger_acceptance_v1'))
    parser.add_argument('--model-path',type=Path)
    parser.add_argument('--training-run',type=Path,default=Path('out/claim_sft_7b_reviewed_v2'))
    parser.add_argument('--interval',type=float,default=.05)
    args=parser.parse_args()
    if args.interval<0:
        parser.error('--interval must be nonnegative')
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
    if (args.out/'config.json').exists() and read(args.out/'config.json')!=config:
        raise ValueError('Acceptance run configuration changed')
    save(args.out/'config.json',config)
    reports=[]
    for session in protocol['sessions']:
        directory=args.out/session['session_id']
        events=[json.loads(line) for line in (args.inputs/session['path']).read_text(encoding='utf-8').splitlines()]
        def open_runtime():
            return OnlineLedger(directory,session['session_id'],claim_processor=processor,input_mode='replay')
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
        restored=open_runtime()
        restored.drain()
        restored.close()
        after={p.name:p.read_bytes() for p in restored.results.glob('*.json')}
        if before!=after:
            raise ValueError('Published batches changed after recovery')
        batches=[read(p) for p in restored.results.glob('*.json')]
        timeline=[json.loads(line) for line in (directory/'timing/timeline.jsonl').read_text(encoding='utf-8').splitlines()]
        starts={}
        queue_ms,processing_ms,publication_ms=[],[],[]
        for event in timeline:
            key=(event.get('job'),event['process_id'])
            if event['kind']=='claim_started':
                starts[key]=event
            if event['kind']=='claim_completed' and key in starts:
                start=starts[key]
                processing_ms.append((event['monotonic_ns']-start['monotonic_ns'])/1e6)
                queued_path=directory/'timing'/('queue-'+event['job'])
                if queued_path.exists():
                    queued=read(queued_path)
                    if queued['process_id']==start['process_id']:
                        queue_ms.append((start['monotonic_ns']-queued['monotonic_ns'])/1e6)
                        publication_ms.append((event['monotonic_ns']-queued['monotonic_ns'])/1e6)
        jobs=len(list(restored.jobs.glob('*.json')))
        attempts=[read(p) for p in (directory/'review_backend/extraction').glob('*/status.json')]
        report=dict(session=session['session_id'],events=len(events),steps=len(restored.segmenter.steps),
            sealed_jobs=jobs,published_batches=len(batches),unpublished_jobs=jobs-len(batches),
            failed_extractions=sum(a['status']=='failed' for a in attempts),attempts=sum(a['attempts'] for a in attempts),
            recovered_batches_identical=True,terminal_seen=restored.finished,
            queue_ms=distribution(queue_ms),processing_ms=distribution(processing_ms),
            queued_to_publication_ms=distribution(publication_ms),
            signals=sum(len(b.get('observation_signals',[])) for b in batches),
            claims_needing_review=sum(c.get('semantic_status')=='needs_review' for b in batches for c in b.get('claims',[])),
            generation_metrics=[b.get('metrics',{}) for b in batches],semantic_quality='pending_AI_review')
        reports.append(report)
        save(args.out/'summary.json',dict(mode=args.command,sessions=reports,
            timing_scope='Controlled replay timings within matching observer clock domains. Replay speed controls queue load. Midpoint restart gracefully drains workers.'))
        print(report,flush=True)


if __name__=='__main__':
    main()
