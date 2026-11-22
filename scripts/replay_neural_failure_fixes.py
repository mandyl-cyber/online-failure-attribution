"""Replay the 19 frozen extraction outcomes through the corrected state machine."""
from pathlib import Path
import argparse

from scripts.run_stage_claim_pipeline import read,save
from segmenter.claim_extraction import StageProcessor
from segmenter.claim_sft_backend import ClaimSFTBackend
from segmenter.reviewed_claim_processor import ReviewedClaimProcessor


class FrozenBackend:
    identity='frozen-neural-outcomes-regression-v1'
    mode='saved_outcome_regression'
    retryable_error=staticmethod(ClaimSFTBackend.retryable_error)

    def __init__(self,source):
        self.source=source
        self.calls=0
        self.filenames={read(p)['job_id']:p.name for p in (source/'stage_jobs').glob('*.json')}

    def extract(self,request):
        self.calls+=1
        filename=self.filenames[request['job_id']]
        target=self.source/'review_backend/claim_batches'/filename
        if not target.exists():
            status=read(self.source/'review_backend/extraction'/Path(filename).stem/'status.json')
            raise ValueError(status['error'])
        batch=read(target)
        return dict(claims=[{k:c[k] for k in ('local_id','kind','text','basis','objects','conditions','unknowns','evidence_refs')}
                            for c in batch['claims']])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('out/neural_failure_fixes_v1'))
    args=parser.parse_args()
    root=Path('neural_ledger_acceptance_v1')
    out=args.out
    if out.exists():
        raise FileExistsError('Use a fresh output location for the frozen-outcome regression')
    reports=[]
    for session in ('swe','terminal','live_smoke'):
        source=root/session
        backend=FrozenBackend(source)
        processor=ReviewedClaimProcessor(StageProcessor(backend))
        results=out/session/'claim_batches'
        results.mkdir(parents=True)
        jobs=sorted((source/'stage_jobs').glob('*.json'))
        for path in jobs:
            processor(path,results)
        before={p.name:p.read_bytes() for p in results.glob('*.json')}
        first_calls=backend.calls
        for path in jobs:
            processor(path,results)
        if backend.calls!=first_calls or before!={p.name:p.read_bytes() for p in results.glob('*.json')}:
            raise ValueError('Cached publication changed or backend was called again')
        batches=[read(p) for p in results.glob('*.json')]
        reports.append(dict(session=session,jobs=len(jobs),receipts=len(batches),
            semantic_batches=sum(b['status']=='ready' for b in batches),
            evidence_only=sum(b['status']=='evidence_only' for b in batches),
            retained_failed_observations=sum(len(b.get('retained_observations',[])) for b in batches),
            backend_calls=first_calls,repeat_calls=backend.calls-first_calls,recovery_bytes_identical=True))
    save(out/'summary.json',dict(mode='saved_outcome_regression_no_model_calls',sessions=reports,
        note='Successful claims and failures replay original outcomes. No new semantic accuracy or GPU latency measurement.'))
    print(reports)


if __name__=='__main__':
    main()
