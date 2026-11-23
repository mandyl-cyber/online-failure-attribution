"""Verify both real boundary rollout arms without changing their artifacts."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json

from scripts.run_stage_claim_pipeline import read,save
from scripts.audit_neural_ledger_v3_return import require,content
from remote_pilots.openhands_smoke.neural_ledger_acceptance import distribution
from remote_pilots.openhands_smoke.boundary_ledger_rollout import timing
from segmenter.claim_extraction import build_request,validate_response
from segmenter.claim_sft_backend import messages_for,convert_partial
from segmenter import stage_claim_summary as core
from segmenter.ledger import sha,encoded


def main():
    root=Path('boundary_ledger_rollout_v1')
    inputs=Path('pilot/claim_prospective_eval_v1')
    out=Path('out/boundary_ledger_return_audit_v1')
    protocol=read(inputs/'protocol.json')
    model=Path('artifacts/boundary_gru_adjacent_v1/model.json')
    require(read(root/'comparison.json')['candidate_json_sha256']==sha(model.read_bytes()),'Candidate changed')
    old_claims=read(Path('out/claim_prospective_eval_return_audit_v1/claims.json'))
    old={(r['session'],r['job_id'],r['claim']['local_id']):r['claim'] for r in old_claims}
    reports,claims={},[]
    for arm in ('rules','neural'):
        config=read(root/arm/'config.json')
        require(config['protocol_sha256']==sha((inputs/'protocol.json').read_bytes()),'Input protocol changed')
        require(config['cpu_threads']==1,'CPU thread mismatch')
        require(config['boundary']==('existing_phase_rules' if arm=='rules' else 'supervised_candidate'),'Wrong boundary mode')
        if arm=='neural':require(config['boundary_model_sha256']==sha(model.read_bytes()),'Candidate identity mismatch')
        metrics,sessions=[],[]
        for session in protocol['sessions']:
            name=session['session_id'];directory=root/arm/name
            original=[json.loads(x) for x in (inputs/session['path']).read_text(encoding='utf-8').splitlines()]
            native=[r['event'] for r in (read(p) for p in sorted((directory/'native_events').glob('*.json'))) if r['kind']=='event']
            require(original==native,'Native input changed')
            snapshot=read(directory/'stages.json')
            spans=snapshot['stages']
            require([i for s in spans for i in range(s['start_step_id'],s['end_step_id']+1)]==list(range(1,snapshot['observed_steps']+1)),'Stage gap/overlap')
            jobs={p.name:read(p) for p in (directory/'stage_jobs').glob('*.json')}
            batches={p.name:read(p) for p in (directory/'claim_batches').glob('*.json')}
            require(set(jobs)==set(batches),'Missing receipt')
            for filename,batch in sorted(batches.items()):
                job=jobs[filename]
                audit=directory/'review_backend/extraction'/Path(filename).stem
                request,attempt,state=read(audit/'request.json'),read(audit/'attempt_1.json'),read(audit/'status.json')
                require(request==build_request(job),'Non-native model request')
                require(batch['input_sha256']==sha(encoded(job)) and batch['evidence_review_identity']==config['processor'],'Receipt identity mismatch')
                require(state['attempts']==1 and len(list(audit.glob('attempt_*.json')))==1,'Repeated extraction')
                rebuilt,rejected=[],[]
                for part in attempt['raw_output']:
                    child=deepcopy(request);child['_claim_policy']='v4';child['stage']=part['stage']
                    start,end=part['stage']['start_step_id'],part['stage']['end_step_id']
                    child['evidence']=[e for e in request['evidence'] if start<=e['source_step_id']<=end]
                    ids=part['page_evidence_ids']
                    if ids is not None:child['_claim_page_ids']=ids
                    row,prompt=messages_for(child)
                    require(prompt==part['model_request'],'Model prompt changed')
                    require(part['status']=='generated' and part['metrics']['input_tokens']+512<=8192 and part['metrics']['output_tokens']<512,'Budget/generation failure')
                    converted,invalid=convert_partial(core.strict_json(part['raw_output']),row,allowed_ids=ids,enforce_result_scope=True)
                    rebuilt.extend(converted['claims']);rejected.extend(invalid);metrics.append(part['metrics'])
                require(not rejected and not batch['extraction_details']['rejected_claims'],'New rejection case')
                require(not batch['extraction_details']['unprocessed_pages'] and not batch['extraction_details']['duplicate_claims'],'Gap/dedup requires review')
                require([content(c) for c in rebuilt]==[content(c) for c in batch['claims']],'Published claims changed')
                validate_response(dict(claims=[{k:c[k] for k in ('local_id','kind','text','basis','objects','conditions','unknowns','evidence_refs')} for c in batch['claims']]),request)
                require(batch['status']=='ready','Non-ready batch')
                for claim in batch['claims']:
                    if arm=='rules':require(content(claim)==content(old[(name,job['job_id'],claim['local_id'])]),'Rule control changed; cannot transfer review')
                    claims.append(dict(arm=arm,session=name,file=filename,job_id=job['job_id'],stage=job['stage'],claim=claim))
            reported=next(r for r in read(root/arm/'summary.json')['sessions'] if r['session']==name)
            require(reported['recovered_batches_identical'] and reported['recovered_stages_and_jobs_identical'],'Recovery failed')
            sessions.append(dict(session=name,jobs=len(jobs),claims=sum(len(b['claims']) for b in batches.values()),
                steps=snapshot['observed_steps'],open_stages=sum(not s['sealed'] for s in spans),
                recovery_passed=True,timing=timing(directory),queued_to_publication_ms=reported['queued_to_publication_ms']))
        reports[arm]=dict(processor=config['processor'],interval=config['interval'],sessions=sessions,
            jobs=sum(s['jobs'] for s in sessions),claims=sum(s['claims'] for s in sessions),
            generation_parts=len(metrics),total_generation_seconds=sum(m['generation_seconds'] for m in metrics),
            generation_seconds=distribution([m['generation_seconds'] for m in metrics]),
            peak_gpu_GiB=max(m['peak_gpu_bytes'] for m in metrics)/1024**3)
    require(reports['rules']['processor']==reports['neural']['processor'] and reports['rules']['interval']==reports['neural']['interval'],'Unmatched control')
    require(sum(c['arm']=='rules' for c in claims)==39,'Rule claim count changed')
    result=dict(arms=reports,claims_sha256=sha(encoded(claims)),reference_sha256=protocol['reference_sha256'],
        scope='Same known synthetic traces, fixed old 7B SFT, same replay rate and CPU threads. Fewer stages do not establish boundary accuracy. Rule claims exactly match prior review; neural claims require new review.')
    save(out/'summary.json',result);save(out/'claims.json',claims)
    print({a:{k:v for k,v in r.items() if k not in ('sessions','processor')} for a,r in reports.items()})


if __name__=='__main__':main()
