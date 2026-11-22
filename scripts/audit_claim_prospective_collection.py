"""Validate prospective native inputs without generating claims or using posthoc tests as evidence."""
from collections import Counter
from pathlib import Path

from scripts.run_stage_claim_pipeline import read,save
from scripts.audit_neural_ledger_v3_return import require
from segmenter.ledger import sha,encoded
from segmenter.adapters.openhands import OpenHandsStreamAdapter,NO_OBS_ACTIONS
from segmenter.input_timeline import validate_event
from remote_pilots.openhands_smoke.run import ACTION_GUIDANCE


def main():
    root=Path('out/claim_prospective_native_v1')
    inputs=Path('pilot/claim_prospective_tasks_v1')
    out=Path('out/claim_prospective_return_audit_v1')
    protocol,config=read(inputs/'protocol.json'),read(root/'config.json')
    require(config['protocol_sha256']==sha((inputs/'protocol.json').read_bytes()),'Collection protocol changed')
    require(config['claim_outputs_generated'] is False,'Claim outputs already generated')
    rows=[]
    for selected in protocol['tasks']:
        bundle=read(inputs/selected['path'])
        require(sha((inputs/selected['path']).read_bytes())==selected['sha256'],'Task definition changed')
        receipt=read(root/Path(selected['path']).stem/'receipt.json')
        require(receipt['task_id']==selected['task_id'] and receipt['bundle_sha256']==selected['sha256'],'Wrong task receipt')
        directory=root/receipt['run_directory']
        require(directory.resolve().is_relative_to(root.resolve()),'Run outside collection')
        manifest,result=read(directory/'manifest.json'),read(directory/'result.json')
        require(manifest['task_bundle_sha256']==selected['sha256'] and manifest['reference_solution_mounted'] is False,'Run identity/reference leakage')
        require(manifest['initial_code_sha256']==sha(bundle['code'].encode()),'Initial code changed')
        require(manifest['initial_test_sha256']==sha(bundle['tests'].encode()),'Initial tests changed')
        expected_tests=bundle['tests'].encode()
        require((directory/'workspace'/bundle['test_name']).read_bytes()==expected_tests,'Visible tests modified')
        require((directory/'checks'/bundle['test_name']).read_bytes()==expected_tests,'Immutable checks changed')
        require((directory/'state/task.txt').read_text(encoding='utf-8')==bundle['task']+ACTION_GUIDANCE,'Task prompt changed')
        sessions=list((directory/'state/store').glob('sessions/*/events'))
        require(len(sessions)==1,'Expected one native session')
        paths=sorted(sessions[0].glob('*.json'),key=lambda p:int(p.stem))
        require(len(paths)==receipt['native_event_files'],'Native event count mismatch')
        require([int(p.stem) for p in paths]==list(range(len(paths))),'Event file sequence gap')
        adapter=OpenHandsStreamAdapter(fold_agent_messages=False)
        steps=[]
        event_manifest=[]
        tool_observations=0
        for path in paths:
            event=read(path)
            require(event['id']==int(path.stem),'Native event ID mismatch')
            validate_event(adapter,event)
            steps.extend(adapter.feed(event))
            if event.get('observation') not in (None,'agent_state_changed','null','recall'):
                tool_observations+=1
            event_manifest.append(dict(path=path.relative_to(root).as_posix(),sha256=sha(path.read_bytes()),event_id=event['id']))
        require(not adapter._pending,'Uncompleted native tool action')
        require(not adapter.diagnostics,'Pairing diagnostics require review')
        tools=[s for s in steps if s.tool not in NO_OBS_ACTIONS]
        require(len(tools)==tool_observations,'Observation/step count mismatch')
        for step in tools:
            integrity=step.attrs['input_integrity']
            require(integrity['pairing']=='exact_cause' and not integrity['issues'],'Incomplete/ambiguous input')
        state=read(directory/'state/controller_result.json')
        for key in ('controller_exit_code','verification_exit_code','visible_tests_unchanged'):
            require(result[key]==receipt[key],'Receipt/result mismatch')
        row=dict(task_id=selected['task_id'],events=len(paths),complete_tool_steps=len(tools),
            normalized_steps=len(steps),pending=0,exact_cause_pairing=True,
            native_finish_present=any(s.tool=='finish' for s in steps),
            controller_state=state['agent_state'],controller_error=state['last_error'],
            controller_exit_code=result['controller_exit_code'],posthoc_verification_exit_code=result['verification_exit_code'],
            tests_unchanged=True,posthoc_verification_sha256=sha((directory/'verification.log').read_bytes()),
            native_event_manifest_sha256=sha(encoded(event_manifest)),
            posthoc_verification_usage='Task outcome audit only; not visible to the online claim model',
            model_claim_outputs_generated=False)
        rows.append(row)
        save(out/Path(selected['path']).stem/'native_event_manifest.json',event_manifest)
        save(out/Path(selected['path']).stem/'normalized_steps.json',[s.to_dict() for s in steps])
    summary=dict(tasks=len(rows),events=sum(r['events'] for r in rows),
        complete_tool_steps=sum(r['complete_tool_steps'] for r in rows),pending=sum(r['pending'] for r in rows),
        tests_unchanged=all(r['tests_unchanged'] for r in rows),
        posthoc_checks_passed=sum(r['posthoc_verification_exit_code']==0 for r in rows),
        controller_and_posthoc_passed=sum(r['controller_exit_code']==0 and r['posthoc_verification_exit_code']==0 for r in rows),
        controller_states=dict(Counter(r['controller_state'] for r in rows)),rows=rows,
        semantic_references_frozen=False,claim_model_calls=0,
        scope='New synthetic engineering traces, not CodeTraceBench. No stage F1, semantic claim accuracy or alarm metric. No forced EOF flush; posthoc tests excluded from online evidence.')
    save(out/'summary.json',summary)
    print({k:v for k,v in summary.items() if k!='rows'})


if __name__=='__main__':
    main()
