"""Freeze AI evidence facts and causal sealed jobs before prospective claim generation."""
from pathlib import Path
from collections import Counter
import json

from scripts.run_stage_claim_pipeline import read,save
from scripts.audit_neural_ledger_v3_return import require
from segmenter.ledger import encoded,sha
from segmenter.online_ledger import OnlineLedger
from segmenter import stage_claim_summary as core
from segmenter.claim_extraction import build_request
from segmenter.claim_sft_backend import messages_for,partition_paged_requests


class NoClaims:
    identity='reference-freeze-no-claim-generation-v1'

    def __call__(self,*args):
        raise AssertionError('Claim generation forbidden before reference freeze')


def freeze_write(path,payload):
    if path.exists():
        require(path.read_bytes()==payload,'Frozen input/reference changed; create a new version: '+str(path))
    else:
        path.write_bytes(payload)


def specifications():
    # Manually reviewed against native actions/results, not posthoc verification.
    items=[]
    def add(task,step,category,text,anchor,field='observation',key=True):
        items.append((task,step,category,text,anchor,field,key))
    for task,steps in [('stable_unique',[1,5]),('bounded_chunks',[1,5])]:
        for sid in steps:
            add(task,sid,'zero_tests','The command reports running zero tests; it does not verify the implementation.','Ran 0 tests')
    add('stable_unique',2,'source_observation','The source listing returns sorted(set(values)).','return sorted(set(values))',key=False)
    add('stable_unique',3,'edit_failed','The replacement was not performed because old_str did not appear verbatim.','did not appear verbatim')
    add('stable_unique',4,'edit_acknowledged','The edit receipt displays a seen dictionary and a result list populated in input order; runtime correctness is not established by the receipt.','result.append(value)')
    add('stable_unique',6,'source_observation','The test source defines TaskTests and three test methods.','class TaskTests',key=False)
    add('stable_unique',7,'verification_pass','The native test run reports 3 tests and OK within the shown test scope.','Ran 3 tests')
    add('bounded_chunks',2,'source_observation','The source listing uses len(values)-size+1 as the range stop.','len(values)-size+1',key=False)
    add('bounded_chunks',3,'probe_result','The negative-size probe prints an empty list.','[]')
    add('bounded_chunks',4,'edit_acknowledged','The edit receipt adds a size <= 0 ValueError guard while still displaying len(values) - size + 1 in the loop bound.','len(values) - size + 1')
    add('bounded_chunks',6,'agent_assertion','The agent claims the implementation is fixed, but also says automated verification is unavailable because no tests were found.','we can conclude that the implementation is fixed',field='action')
    for sid,n in [(1,3),(6,2),(12,1)]:
        add('config_overrides',sid,'verification_failed',f'The native test run reports 4 tests with {n} failure(s).',f'FAILED (failures={n})')
    add('config_overrides',3,'source_observation','The source updates base in place and filters overrides by truthiness.','base.update({k:v for k,v in overrides.items() if v})',key=False)
    for sid in [5,7,9,11]:
        add('config_overrides',sid,'edit_acknowledged','The tool acknowledges an edit of task.py; this alone does not establish a working fix.','has been edited',key=False)
    for sid in [8,10]:
        add('config_overrides',sid,'runtime_exception','The test command raises IndentationError in task.py.','IndentationError:')
    add('config_overrides',13,'edit_acknowledged','The edit receipt displays result = base.copy(), conditional assignment into result, and return result.','result = base.copy()')
    add('config_overrides',14,'verification_pass','The native test run reports 4 tests and OK within the shown test scope.','Ran 4 tests')
    add('config_overrides',15,'agent_assertion','The agent says the changes have been committed; this is an agent assertion, not a commit receipt.','changes have been committed',field='action')
    add('prefix_totals',1,'verification_failed','The initial native test run reports 3 tests with 3 failures.','FAILED (failures=3)')
    for sid in [2,4]:
        add('prefix_totals',sid,'edit_failed','The replacement was not performed because the supplied old_str did not match the file.','did not appear verbatim')
    add('prefix_totals',3,'source_observation','The grep result shows def solve(values).','def solve(values)',key=False)
    add('prefix_totals',5,'source_observation','The source listing modifies values in place while accumulating.','values[i] += values[i-1]',key=False)
    add('prefix_totals',6,'edit_acknowledged','The edit receipt displays an initial [0] and a loop over range(1, len(values)); it does not establish full behavior.','range(1, len(values))')
    for sid in [7,10,13,18,19,20]:
        add('prefix_totals',sid,'verification_failed','The shown run reports 3 tests with 1 failure and displays [0, 2, 1] instead of [0, 2, 1, 5].','FAILED (failures=1)')
    for sid in [8,11,14,15,16]:
        add('prefix_totals',sid,'edit_failed','The tool performs no replacement because new_str and old_str must be different.','must be different')
    for sid in [9,12,17]:
        add('prefix_totals',sid,'edit_acknowledged','The edit receipt displays an added comment for clarity; it does not establish a behavioral fix.','# Added comment for clarity',key=False)
    return items


def main():
    root=Path('out/claim_prospective_native_v1')
    pack=Path('pilot/claim_prospective_eval_v1')
    pack.mkdir(parents=True,exist_ok=True)
    sessions,facts,all_jobs,coverage=[],[],[],[]
    specs=specifications()
    for selected in read(Path('pilot/claim_prospective_tasks_v1/protocol.json'))['tasks']:
        task=Path(selected['path']).stem
        receipt=read(root/task/'receipt.json')
        run=root/receipt['run_directory']
        dirs=list((run/'state/store').glob('sessions/*/events'))
        require(len(dirs)==1,'Expected one session')
        events=[read(p) for p in sorted(dirs[0].glob('*.json'),key=lambda p:int(p.stem))]
        payload=b''.join(encoded(e)+b'\n' for e in events)
        freeze_write(pack/f'{task}.jsonl',payload)
        runtime=OnlineLedger(Path('out/claim_prospective_reference_replay_v1')/task,task,
                             background=False,claim_processor=NoClaims(),input_mode='replay')
        for event in events:
            runtime.ingest(event)
        runtime.close() # No drain, no forced end-of-stream, no claims.
        require(not list(runtime.results.glob('*.json')),'Unexpected claim output')
        jobs=[read(p) for p in sorted(runtime.jobs.glob('*.json'))]
        steps={s.idx:s for s in runtime.segmenter.steps}
        for job in jobs:
            all_jobs.append(dict(session=task,job=job,input_sha256=sha(encoded(job))))
        for session,sid,category,text,anchor,field,key in specs:
            if session!=task:
                continue
            step=steps[sid]
            source_text=getattr(step,field)
            require(anchor in source_text,'Reviewed anchor changed: '+str((task,sid,anchor)))
            matching=[job for job in jobs if job['stage']['start_step_id']<=sid<=job['stage']['end_step_id']]
            require(len(matching)<=1,'Overlapping sealed stages')
            action_id=step.raw_ref['action_id']
            observation_id=step.raw_ref.get('observation_id')
            source_id=action_id if field=='action' else observation_id
            require(source_id is not None,'No native evidence for fact')
            start=source_text.index(anchor)
            # Preserve the exact full normalized record plus a narrow literal
            # anchor; no posthoc files enter the source snapshot.
            fact=dict(fact_id=f'{task}:step{sid}:{category}',session=task,step_id=sid,
                category=category,importance='key_outcome' if key else 'context',text=text,
                basis='agent_assertion' if field=='action' else 'tool_observation',reviewer_kind='AI',reviewed=True,
                source_field=field,native_event_id=source_id,record_text=source_text,record_sha256=sha(source_text.encode()),
                anchor=dict(start=start,end=start+len(anchor),quote=anchor),
                available_at_event_seq=step.attrs['input_integrity']['available_at_seq'],
                sealed_job_id=matching[0]['job_id'] if matching else None,
                eligibility='sealed_stage' if matching else 'open_stage_not_sealed')
            if matching:
                evidence=[e for e in matching[0]['evidence'] if e['source_step_id']==sid and e['field']==field]
                require(len(evidence)==1 and evidence[0]['text']==source_text,'Fact/source mismatch')
                fact['record_id']=evidence[0]['record_id']
                fact['seal_cutoff_step']=matching[0]['detected_at_seq']
            facts.append(fact)
        sessions.append(dict(session_id=task,path=f'{task}.jsonl',sha256=sha(payload),events=len(events),
                             source='fresh synthetic native collection',terminal_finish_present=runtime.finished))
        coverage.append(dict(session=task,normalized_steps=len(steps),sealed_jobs=len(jobs),
            sealed_step_ids=sorted({i for j in jobs for i in range(j['stage']['start_step_id'],j['stage']['end_step_id']+1)}),
            native_finish_present=runtime.finished,claim_outputs_generated=False))
    # Tokenizer only; all annotation precedes any model generation.
    from transformers import AutoTokenizer
    from remote_pilots.openhands_smoke.claim_sft_train import chat_ids
    tokenizer=AutoTokenizer.from_pretrained('claim_sft_7b_reviewed_v2/adapter',local_files_only=True)
    preflight=[]
    for item in all_jobs:
        request=build_request(item['job'])
        request['_claim_policy']='v4'
        def count(child):
            return len(chat_ids(tokenizer,messages_for(child)[1],True))
        parts,gaps=partition_paged_requests(request,lambda child:count(child)+512<=8192)
        require(not gaps,'Budget gap in prospective inputs')
        preflight.append(dict(job_id=item['job']['job_id'],session=item['session'],parts=len(parts),input_tokens=[count(p) for p in parts]))
    references=dict(reviewer_kind='AI',blinded_to_claim_outputs=True,
        scope='Focused native evidence inventory, not human gold or exhaustive stage semantics; context facts optional. Posthoc tests excluded.',facts=facts)
    jobs_payload=b''.join(encoded(j)+b'\n' for j in all_jobs)
    references_payload=core.encoded(references)+b'\n'
    freeze_write(pack/'stage_jobs.jsonl',jobs_payload)
    freeze_write(pack/'references.json',references_payload)
    save(pack/'token_preflight.json',preflight)
    protocol=dict(protocol='claim-prospective-eval-v1',sessions=sessions,boundary='existing_phase_rules',
        reference_sha256=sha(references_payload),jobs_sha256=sha(jobs_payload),
        frozen_before_model_calls=True,claim_model_calls=0,posthoc_verification_in_input=False,
        evaluate_only_sealed_stages=True,scope='Four new synthetic tasks; small engineering acceptance, not CodeTraceBench generalization.')
    freeze_write(pack/'protocol.json',core.encoded(protocol)+b'\n')
    summary=dict(sessions=coverage,facts=len(facts),categories=dict(Counter(f['category'] for f in facts)),
        key_outcomes=sum(f['importance']=='key_outcome' for f in facts),
        sealed_facts=sum(f['eligibility']=='sealed_stage' for f in facts),
        open_stage_facts=sum(f['eligibility']!='sealed_stage' for f in facts),
        sealed_jobs=len(all_jobs),generation_parts=sum(p['parts'] for p in preflight),input_gaps=0,
        reviewer_kind='AI',claim_model_calls=0,semantic_references_frozen=True,
        limits=['References are focused AI annotations','Open-stage facts stay in the denominator report but receive no artificial summary job',
                'No stage boundary gold or failure alarm labels','No posthoc verification facts used'])
    save(pack/'summary.json',summary)
    print({k:v for k,v in summary.items() if k!='sessions'})


if __name__=='__main__':
    main()
