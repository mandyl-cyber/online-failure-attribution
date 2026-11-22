"""Audit successful neural ledger publication and recompute success-only timing."""
from pathlib import Path
from collections import Counter
import hashlib
import json

from scripts.run_stage_claim_pipeline import read,save
from segmenter.ledger import sha,encoded
from segmenter.claim_extraction import validate_response
from remote_pilots.openhands_smoke.neural_ledger_acceptance import successful_timings,distribution


def main():
    root=Path('neural_ledger_acceptance_v1')
    pack=Path('pilot/neural_ledger_acceptance_v1')
    out=Path('out/neural_ledger_acceptance_return_v1')
    config=read(root/'config.json')
    snapshot=hashlib.sha256()
    for path in sorted(root.glob('*/claim_batches/*.json')):
        snapshot.update(path.relative_to(root).as_posix().encode())
        snapshot.update(path.read_bytes())
    if snapshot.hexdigest()!='bd01b442de2d2e3d647761cae480ae4c229c51b8c44198bfb0ca6143d4374dec':
        raise ValueError('Reviewed claim snapshot changed. Re-review before applying AI judgments.')
    if config['protocol_sha256']!=sha((pack/'protocol.json').read_bytes()):
        raise ValueError('Frozen protocol mismatch')
    reports,failures,reviews=[],[],[]
    generations=[]
    for session in read(pack/'protocol.json')['sessions']:
        name=session['session_id']
        directory=root/name
        original=[json.loads(line) for line in (pack/session['path']).read_text(encoding='utf-8').splitlines()]
        native=[read(p)['event'] for p in sorted((directory/'native_events').glob('*.json')) if read(p).get('kind')=='event']
        if original!=native:
            raise ValueError('Native event replay mismatch')
        jobs={p.name:read(p) for p in (directory/'stage_jobs').glob('*.json')}
        published={p.name:read(p) for p in (directory/'claim_batches').glob('*.json')}
        for filename,batch in published.items():
            job=jobs[filename]
            if batch['input_sha256']!=sha(encoded(job)) or batch['evidence_review_identity']!=config['processor']:
                raise ValueError('Published input or processor mismatch')
            request=read(directory/'review_backend/extraction'/Path(filename).stem/'request.json')
            response=dict(claims=[{k:c[k] for k in ['local_id','kind','text','basis','objects','conditions','unknowns','evidence_refs']} for c in batch['claims']])
            validate_response(response,request)
            for signal in batch['observation_signals']:
                ev=next(e for e in job['evidence'] if e['record_id']==signal['record_id'])
                if ev['text'][signal['start']:signal['end']]!=signal['quote']:
                    raise ValueError('Observation signal quote mismatch')
            generations.append(batch['metrics'])
            for index,claim in enumerate(batch['claims']):
                special=(name=='swe' and filename=='00000006.json' and index==1)
                reviews.append(dict(session=name,job=filename,claim_index=index,text=claim['text'],reviewer_type='AI',reviewed=True,
                    support='partial' if special else 'supported',
                    note='Printed expressions and expected ValueError are observed. The qualifier correct exceeds the cited output alone.' if special else 'Claim checked against its cited source excerpt.',
                    source_label_note='Commit-message purpose should retain agent-assertion attribution.' if name=='swe' and filename=='00000005.json' else None))
        for path in (directory/'review_backend/extraction').glob('*/status.json'):
            state=read(path)
            if state['status']=='ready':
                continue
            attempts=[read(p) for p in sorted(path.parent.glob('attempt_*.json'))]
            failures.append(dict(session=name,job=path.parent.name,error=state.get('error'),attempts=state['attempts'],
                category='input_budget' if 'token_budget_exceeded' in state.get('error','') else 'source_type_citation_mismatch',
                deterministic_repeat=len(attempts)==2 and attempts[0].get('raw_output')==attempts[1].get('raw_output')))
        reports.append(dict(session=name,jobs=len(jobs),published=len(published),unpublished=len(jobs)-len(published),
                            corrected_timings=successful_timings(directory,set(published))))
    summary=dict(events=sum(s['events'] for s in read(pack/'protocol.json')['sessions']),jobs=sum(r['jobs'] for r in reports),
        published=sum(r['published'] for r in reports),failed=len(failures),failure_categories=dict(Counter(f['category'] for f in failures)),
        sessions=reports,published_claims=len(reviews),AI_support_counts=dict(Counter(r['support'] for r in reviews)),
        generation_seconds=distribution([m['generation_seconds'] for m in generations]),
        peak_gpu_GiB=max(m['peak_gpu_bytes'] for m in generations)/1024**3,
        caveat='AI review only on published claims. Failed stages excluded from support counts. Controlled replay queue timings, graceful restart only. Peak GPU counter is cumulative.')
    save(out/'summary.json',summary)
    save(out/'failures.json',failures)
    save(out/'AI_claim_review.json',reviews)
    print(summary)


if __name__=='__main__':
    main()
