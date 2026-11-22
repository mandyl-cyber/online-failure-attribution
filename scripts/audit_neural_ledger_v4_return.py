"""Verify returned v4 page coverage, citations, signals and cached recovery."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import shutil
import tempfile

from scripts.run_stage_claim_pipeline import read, save
from scripts.audit_neural_ledger_v2_return import CachedProcessor
from scripts.audit_neural_ledger_v3_return import content, require
from segmenter.claim_extraction import validate_response, build_request
from segmenter.claim_sft_backend import messages_for, convert_partial
from segmenter.claim_evidence_review import review
from segmenter import stage_claim_summary as core
from segmenter.ledger import sha, encoded
from segmenter.online_ledger import OnlineLedger
from remote_pilots.openhands_smoke.neural_ledger_acceptance import successful_timings, distribution


def main():
    root, pack = Path('neural_ledger_acceptance_v4'), Path('pilot/neural_ledger_acceptance_v1')
    out = Path('out/neural_ledger_v4_return_audit')
    out.mkdir(parents=True, exist_ok=True)
    config, protocol = read(root/'config.json'), read(pack/'protocol.json')
    require(config['mode']=='run' and config['protocol_sha256']==sha((pack/'protocol.json').read_bytes()), 'Wrong mode/protocol')
    restored=Path(tempfile.mkdtemp(prefix='restore-',dir=out))
    sessions,records,claims,part_metrics,rejections,signals=[],[],[],[],[],[]
    for session in protocol['sessions']:
        name=session['session_id']
        directory=root/name
        original=[json.loads(line) for line in (pack/session['path']).read_text(encoding='utf-8').splitlines()]
        native=[r['event'] for r in (read(p) for p in sorted((directory/'native_events').glob('*.json'))) if r.get('kind')=='event']
        require(original==native, 'Native stream changed')
        jobs={p.name:read(p) for p in (directory/'stage_jobs').glob('*.json')}
        batches={p.name:read(p) for p in (directory/'claim_batches').glob('*.json')}
        require(set(jobs)==set(batches), 'Missing/extra receipts')
        for filename,batch in sorted(batches.items()):
            job=jobs[filename]
            require(job==read(Path('neural_ledger_acceptance_v3')/name/'stage_jobs'/filename), 'Sealed stage changed')
            require(batch['input_sha256']==sha(encoded(job)) and batch['evidence_review_identity']==config['processor'], 'Receipt fingerprint mismatch')
            audit=directory/'review_backend/extraction'/Path(filename).stem
            request,state,attempt=read(audit/'request.json'),read(audit/'status.json'),read(audit/'attempt_1.json')
            require(request==build_request(job), 'Persisted request differs from sealed input')
            require(state['attempts']==1 and len(list(audit.glob('attempt_*.json')))==1, 'Unexpected retry')
            details=batch['extraction_details']
            require(len(attempt['raw_output'])==len(details['parts']), 'Part count mismatch')
            visible,rebuilt,rejected=[],[],[]
            for raw,part in zip(attempt['raw_output'],details['parts']):
                require({k:v for k,v in raw.items() if k not in ('raw_output','model_request')}==part, 'Part metadata mismatch')
                child=deepcopy(request)
                child['_claim_policy']='v4'
                child['stage']=part['stage']
                start,end=part['stage']['start_step_id'],part['stage']['end_step_id']
                child['evidence']=[e for e in request['evidence'] if start<=e['source_step_id']<=end]
                ids=part['page_evidence_ids']
                if ids is not None:
                    child['_claim_page_ids']=ids
                row,messages=messages_for(child)
                require(messages==raw['model_request'], 'Prompt/evidence/seal changed')
                visible.extend(ids if ids is not None else row['evidence_map'])
                require(part['status']=='generated', 'Generation failure requires separate review')
                metrics=part['metrics']
                require(metrics['input_tokens']+512<=8192 and metrics['output_tokens']<512, 'Budget violation')
                converted,invalid=convert_partial(core.strict_json(raw['raw_output']),row,allowed_ids=ids,enforce_result_scope=True)
                rebuilt.extend(converted['claims'])
                rejected.extend(dict(r,part_index=part['part_index']) for r in invalid)
                part_metrics.append(metrics)
            gaps=details['unprocessed_pages']
            require(Counter(visible+[i for g in gaps for i in g['evidence_ids']])==Counter(messages_for(request)[0]['evidence_map'].keys()), 'Missing/repeated chunks')
            require(details['input_complete']==(not gaps), 'Input completeness mismatch')
            require(details['oversized_steps']==sorted({g['step_id'] for g in gaps}), 'Gap summary mismatch')
            unique,duplicates,seen=[],[],set()
            for claim in rebuilt:
                fingerprint=sha(encoded({k:v for k,v in claim.items() if k!='local_id'}))
                if fingerprint in seen:
                    duplicates.append(claim)
                else:
                    seen.add(fingerprint)
                    unique.append(claim)
            require(details['duplicate_claims']==duplicates, 'Deduplication changed')
            require(len(unique)<=64, 'New claim-cap case requires review')
            require(rejected==details['rejected_claims'], 'Rejected claims changed')
            require([content(c) for c in unique]==[content(c) for c in batch['claims']], 'Published claim changed')
            response=dict(claims=[{k:c[k] for k in ('local_id','kind','text','basis','objects','conditions','unknowns','evidence_refs')} for c in batch['claims']])
            validate_response(response,request)
            complete=not gaps and not rejected
            expected_status='ready' if complete else 'partial' if batch['claims'] else 'evidence_only'
            require(details['complete']==complete and batch['status']==expected_status, 'Batch status mismatch')
            retained=[]
            if not complete:
                retained=[e for e in job['evidence'] if e['field']=='observation']
                require(batch['retained_observations']==retained and batch['review_status']=='needs_review', 'Raw observation lost')
            expected_signals=review(dict(claims=[]),core.build(job))['observation_signals']
            actual_signals=batch['observation_signals']
            require(len(expected_signals)==len(actual_signals), 'Observation signal missing')
            for expected,actual in zip(expected_signals,actual_signals):
                require(all(actual[k]==v for k,v in expected.items() if k!='citing_claim_ids'), 'Signal span/source changed')
                require(actual['available_after_seq']==job['detected_at_seq'], 'Signal timing changed')
            old=read(Path('neural_ledger_acceptance_v3')/name/'claim_batches'/filename)
            for claim in batch['claims']:
                claims.append(dict(session=name,job=filename,claim=claim,
                    identical_to_v3=any(content(claim)==content(c) for c in old['claims'])))
            rejections.extend(dict(session=name,job=filename,**r) for r in rejected)
            signals.extend(dict(session=name,job=filename,**s) for s in actual_signals)
            records.append(dict(session=name,job=filename,status=batch['status'],claims=len(batch['claims']),
                parts=len(details['parts']),paged_parts=sum(p['page_evidence_ids'] is not None for p in details['parts']),
                rejected=len(rejected),gaps=len(gaps),retained_observations=len(retained)))
        copy=restored/name
        shutil.copytree(directory,copy)
        before={p.name:p.read_bytes() for p in (copy/'claim_batches').glob('*.json')}
        online=read(copy/'online.json')
        runtime=OnlineLedger(copy,name,background=False,claim_processor=CachedProcessor(config['processor']),input_mode=online['input_mode'])
        runtime.drain()
        runtime.close()
        require(before=={p.name:p.read_bytes() for p in (copy/'claim_batches').glob('*.json')}, 'Recovery changed receipts')
        sessions.append(dict(session=name,statuses=dict(Counter(b['status'] for b in batches.values())),cached_recovery_passed=True,
            full_timings=successful_timings(directory,{n for n,b in batches.items() if b['status']=='ready'}),
            partial_timings=successful_timings(directory,{n for n,b in batches.items() if b['status']=='partial'})))
    require(len(records)==19,'Missing stage')
    result=dict(jobs=len(records),receipts=len(records),statuses=dict(Counter(r['status'] for r in records)),
        claims=len(claims),identical_claims_to_v3=sum(c['identical_to_v3'] for c in claims),
        generation_parts=len(part_metrics),paged_parts=sum(r['paged_parts'] for r in records),
        input_gaps=sum(r['gaps'] for r in records),rejected_claims=len(rejections),
        retained_observations=sum(r['retained_observations'] for r in records),
        signal_counts=dict(Counter(s['kind'] for s in signals)),uncited_signals=sum(not s['citing_claim_ids'] for s in signals),
        generation_seconds_per_part=distribution([m['generation_seconds'] for m in part_metrics]),
        total_generation_seconds=sum(m['generation_seconds'] for m in part_metrics),
        peak_gpu_GiB=max(m['peak_gpu_bytes'] for m in part_metrics)/1024**3,sessions=sessions,
        scope='Three known sessions at 0.05s/event, default phase rules, shared serial 7B. Structural validation is not semantic accuracy. Cached recovery did not force a GPU crash.')
    for filename,value in [('summary.json',result),('batches.json',records),('claims.json',claims),('rejected_claims.json',rejections),('observation_signals.json',signals)]:
        save(out/filename,value)
    print(json.dumps({k:v for k,v in result.items() if k!='sessions'},ensure_ascii=False))


if __name__=='__main__':
    main()
