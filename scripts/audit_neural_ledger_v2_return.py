"""Validate terminal-failure evidence receipts and independently replay recovery."""
from pathlib import Path
from collections import Counter
import json
import shutil
import tempfile

from scripts.run_stage_claim_pipeline import read,save
from segmenter.ledger import sha,encoded
from segmenter.claim_extraction import validate_response
from segmenter.online_ledger import OnlineLedger
from remote_pilots.openhands_smoke.neural_ledger_acceptance import successful_timings,distribution


class CachedProcessor:
    def __init__(self,identity):
        self.identity=identity

    def __call__(self,job_path,result_dir):
        result=read(Path(result_dir)/Path(job_path).name)
        if result['input_sha256']!=sha(encoded(read(Path(job_path)))):
            raise ValueError('Cached input mismatch')
        return result


def main():
    root=Path('neural_ledger_acceptance_v2')
    pack=Path('pilot/neural_ledger_acceptance_v1')
    out=Path('out/neural_ledger_v2_return_audit')
    out.mkdir(parents=True,exist_ok=True)
    protocol=read(pack/'protocol.json')
    config=read(root/'config.json')
    if config['protocol_sha256']!=sha((pack/'protocol.json').read_bytes()):
        raise ValueError('Input protocol changed')
    reports,failures,metrics=[],[],[]
    claims_unchanged=0
    restore_root=Path(tempfile.mkdtemp(prefix='restore-',dir=out))
    for session in protocol['sessions']:
        name=session['session_id']
        directory=root/name
        original=[json.loads(line) for line in (pack/session['path']).read_text(encoding='utf-8').splitlines()]
        native=[r['event'] for r in (read(p) for p in sorted((directory/'native_events').glob('*.json'))) if r.get('kind')=='event']
        if native!=original:
            raise ValueError('Native event mismatch')
        jobs={p.name:read(p) for p in (directory/'stage_jobs').glob('*.json')}
        batches={p.name:read(p) for p in (directory/'claim_batches').glob('*.json')}
        if set(jobs)!=set(batches):
            raise ValueError('Missing or extra stage receipt')
        retained=0
        attempts=0
        for filename,batch in batches.items():
            job=jobs[filename]
            if batch['input_sha256']!=sha(encoded(job)) or batch['evidence_review_identity']!=config['processor']:
                raise ValueError('Receipt input/processor mismatch')
            audit=directory/'review_backend/extraction'/Path(filename).stem
            state=read(audit/'status.json')
            attempts+=state['attempts']
            if state['attempts']!=1 or len(list(audit.glob('attempt_*.json')))!=1:
                raise ValueError('Unexpected repeated extraction attempt')
            if batch['status']=='ready':
                request=read(audit/'request.json')
                response=dict(claims=[{k:c[k] for k in ('local_id','kind','text','basis','objects','conditions','unknowns','evidence_refs')} for c in batch['claims']])
                validate_response(response,request)
                old=read(Path('neural_ledger_acceptance_v1')/name/'claim_batches'/filename)
                def content(claim):
                    return {k:claim[k] for k in ('text','basis','evidence_refs')}
                if [content(c) for c in batch['claims']]!=[content(c) for c in old['claims']]:
                    raise ValueError('Claim changed. Perform new semantic review.')
                claims_unchanged+=len(batch['claims'])
                metrics.append(batch['metrics'])
            elif batch['status']=='evidence_only':
                if batch['claims'] or batch['review_status']!='needs_review' or state['retryable'] is not False:
                    raise ValueError('Invalid failure receipt')
                expected=[e for e in job['evidence'] if e['field']=='observation']
                if batch['retained_observations']!=expected:
                    raise ValueError('Raw failure observation changed or omitted')
                for observation in expected:
                    if sha(observation['text'].encode())!=observation['content_sha256']:
                        raise ValueError('Retained observation checksum mismatch')
                retained+=len(expected)
                failures.append(dict(session=name,job=filename,error=state['error'],attempts=state['attempts'],
                    retained_observations=len(expected),category='budget' if 'token_budget_exceeded' in state['error'] else 'source_citation'))
            else:
                raise ValueError('Unexpected receipt status')
            for signal in batch['observation_signals']:
                ev=next(e for e in job['evidence'] if e['record_id']==signal['record_id'])
                if ev['text'][signal['start']:signal['end']]!=signal['quote'] or signal['available_after_seq']!=job['detected_at_seq']:
                    raise ValueError('Signal provenance mismatch')
        copy=restore_root/name
        shutil.copytree(directory,copy)
        before={p.name:p.read_bytes() for p in (copy/'claim_batches').glob('*.json')}
        online=read(copy/'online.json')
        runtime=OnlineLedger(copy,name,background=False,claim_processor=CachedProcessor(config['processor']),input_mode=online['input_mode'])
        runtime.drain()
        runtime.close()
        if before!={p.name:p.read_bytes() for p in (copy/'claim_batches').glob('*.json')}:
            raise ValueError('Receipts changed on local recovery')
        success={filename for filename,b in batches.items() if b['status']=='ready'}
        reports.append(dict(session=name,jobs=len(jobs),receipts=len(batches),semantic_batches=len(success),
            evidence_only=len(batches)-len(success),attempts=attempts,retained_failed_observations=retained,
            independent_cached_recovery_passed=True,timings=successful_timings(directory,success)))
    result=dict(sessions=reports,total_jobs=sum(r['jobs'] for r in reports),total_receipts=sum(r['receipts'] for r in reports),
        semantic_batches=sum(r['semantic_batches'] for r in reports),evidence_only=len(failures),
        total_attempts=sum(r['attempts'] for r in reports),retained_failed_observations=sum(r['retained_failed_observations'] for r in reports),
        claims_matching_v1=claims_unchanged,failure_categories=dict(Counter(f['category'] for f in failures)),
        generation_seconds=distribution([m['generation_seconds'] for m in metrics]),peak_gpu_GiB=max(m['peak_gpu_bytes'] for m in metrics)/1024**3,
        scope='Same three controlled-replay sessions. Evidence receipts and retries fixed. Semantic extraction remains 14/19. Cached recovery uses no GPU.')
    save(out/'summary.json',result)
    save(out/'failures.json',failures)
    print(result)


if __name__=='__main__':
    main()
