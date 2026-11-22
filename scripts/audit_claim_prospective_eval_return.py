"""Audit returned prospective extractor outputs against the frozen inputs."""
from collections import Counter
from copy import deepcopy
from pathlib import Path

from scripts.run_stage_claim_pipeline import read,save
from scripts.audit_neural_ledger_v3_return import require,content
from remote_pilots.openhands_smoke.claim_prospective_eval import verify_inputs
from remote_pilots.openhands_smoke.neural_ledger_acceptance import distribution
from segmenter.claim_extraction import build_request,validate_response
from segmenter.claim_sft_backend import messages_for,convert_partial
from segmenter import stage_claim_summary as core
from segmenter.ledger import sha,encoded
from segmenter.reviewed_claim_processor import ReviewedClaimProcessor


def main():
    root=Path('claim_prospective_eval_v1')
    pack=Path('pilot/claim_prospective_eval_v1')
    out=Path('out/claim_prospective_eval_return_audit_v1')
    out.mkdir(parents=True,exist_ok=True)
    jobs,protocol=verify_inputs(pack)
    config,summary=read(root/'config.json'),read(root/'summary.json')
    require(config['protocol_sha256']==sha((pack/'protocol.json').read_bytes()),'Frozen protocol changed')
    require(config['reference_used_as_model_input'] is False and config['training'] is False,'Wrong experiment')
    require(config['processor']==read(Path('neural_ledger_acceptance_v4/config.json'))['processor'],'Fixed extractor identity changed')
    require(summary['reference_sha256']==protocol['reference_sha256'],'Reference mismatch')
    require(len(summary['records'])==len(jobs),'Missing run records')
    records,claims,metrics=[],[],[]
    for index,(item,record) in enumerate(zip(jobs,summary['records']),1):
        filename=f'{index:04d}.json'
        job=read(root/'stage_jobs'/filename)
        batch=read(root/'claim_batches'/filename)
        require(job==item['job'],'Sealed job changed')
        require(record['file']==filename and record['job_id']==job['job_id'] and record['session']==item['session'],'Run mapping changed')
        require(batch['input_sha256']==item['input_sha256'] and batch['evidence_review_identity']==config['processor'],'Receipt identity changed')
        directory=root/'review_backend/extraction'/Path(filename).stem
        request,attempt,state=read(directory/'request.json'),read(directory/'attempt_1.json'),read(directory/'status.json')
        require(request==build_request(job),'Backend request changed')
        require(state['attempts']==1 and len(list(directory.glob('attempt_*.json')))==1,'Unexpected retry')
        details=batch['extraction_details']
        rebuilt,rejected=[],[]
        for part in attempt['raw_output']:
            child=deepcopy(request)
            child['_claim_policy']='v4'
            child['stage']=part['stage']
            start,end=part['stage']['start_step_id'],part['stage']['end_step_id']
            child['evidence']=[e for e in request['evidence'] if start<=e['source_step_id']<=end]
            ids=part['page_evidence_ids']
            if ids is not None:child['_claim_page_ids']=ids
            row,messages=messages_for(child)
            require(part['model_request']==messages,'Model prompt changed or reference leakage')
            require(part['status']=='generated','New failed generation case')
            require(part['metrics']['input_tokens']+512<=8192 and part['metrics']['output_tokens']<512,'Budget violation')
            converted,invalid=convert_partial(core.strict_json(part['raw_output']),row,allowed_ids=ids,enforce_result_scope=True)
            rebuilt.extend(converted['claims'])
            rejected.extend(dict(r,part_index=part['part_index']) for r in invalid)
            metrics.append(part['metrics'])
        require(not details['unprocessed_pages'] and not details['duplicate_claims'],'Input gap or new dedup case')
        require(details['rejected_claims']==rejected,'Rejected outputs changed')
        require([content(c) for c in rebuilt]==[content(c) for c in batch['claims']],'Accepted claims changed')
        validate_response(dict(claims=[{k:c[k] for k in ('local_id','kind','text','basis','objects','conditions','unknowns','evidence_refs')} for c in batch['claims']]),request)
        require(batch['status']=='ready' and details['complete'],'Incomplete batch')
        # Exercise the real cache-hit path; a miss must not invoke a backend.
        processor=ReviewedClaimProcessor.__new__(ReviewedClaimProcessor)
        processor.identity=config['processor']
        def forbidden(*args):raise AssertionError('Unexpected backend call during cached recovery')
        processor.processor=forbidden
        path=root/'claim_batches'/filename
        before=path.read_bytes()
        require(processor(root/'stage_jobs'/filename,root/'claim_batches')==batch,'Cache result changed')
        require(path.read_bytes()==before,'Published bytes changed')
        for claim in batch['claims']:
            claims.append(dict(file=filename,session=item['session'],job_id=job['job_id'],claim=claim))
        records.append(dict(file=filename,session=item['session'],claims=len(batch['claims']),status=batch['status'],
            cached=record['cached'],call_seconds=record['current_call_seconds'],cache_reuse_passed=True))
    result=dict(jobs=len(jobs),receipts=len(records),statuses=dict(Counter(r['status'] for r in records)),claims=len(claims),
        generation_parts=len(metrics),rejected_claims=0,attempts=len(records),
        newly_processed=sum(not r['cached'] for r in records),cache_recovery_checked=len(records),
        generation_seconds_per_part=distribution([m['generation_seconds'] for m in metrics]),
        total_generation_seconds=sum(m['generation_seconds'] for m in metrics),
        fresh_call_seconds=distribution([r['call_seconds'] for r in records if not r['cached']]),
        peak_gpu_GiB=max(m['peak_gpu_bytes'] for m in metrics)/1024**3,
        reference_sha256=protocol['reference_sha256'],claims_sha256=sha(encoded(claims)),
        scope='Frozen sealed-job extractor evaluation, not streaming latency, stage F1 or human semantic accuracy. No reference edits or model calls in this audit.')
    save(out/'summary.json',result)
    save(out/'claims.json',claims)
    save(out/'batches.json',records)
    print(result)


if __name__=='__main__':
    main()
