"""Engineering acceptance only: controlled contrasts plus existing native replay."""
from pathlib import Path
import json
from segmenter.evidence_state import EvidenceState,native_fact
from segmenter.stage_claim_ablation import read,save,require,digest


def fact(step,kind='test_run',status='passed',obj='module-a',version='v1',source='tool_observation'):
    return dict(fact_id=f'{step}:{kind}:{obj}:{status}:{version}:{source}',step=step,available_step=step,
        kind=kind,status=status,object_id=obj,version=version,source=source,evidence_ids=[f'R{step}'],
        text='Controlled fixture, not a real execution',scope='fixture')


def controls():
    claim=dict(kind='current_test_pass',object_id='module-a',version='v1',prefix_step=10)
    scenarios=[
        ('matching_pass',[fact(1)],claim,'supported'),
        ('other_object_pass',[fact(1,obj='module-b')],claim,'not_established'),
        ('failed_then_recovered',[fact(1,status='failed'),fact(2)],claim,'supported'),
        ('pass_then_failed',[fact(1),fact(2,status='failed')],claim,'not_established'),
        ('zero_tests',[fact(1,status='zero_tests')],claim,'not_established'),
        ('blocked',[fact(1,status='blocked')],claim,'not_established'),
        ('unrelated_recovery',[fact(1,status='failed'),fact(2,obj='module-b')],claim,'not_established'),
        ('edit_after_pass',[fact(1),fact(2,kind='edit',status='acknowledged',version='v2')],dict(claim,version='v2'),'not_established'),
        ('reverify_after_edit',[fact(1),fact(2,kind='edit',status='acknowledged',version='v2'),fact(3,version='v2')],dict(claim,version='v2'),'supported'),
        ('unknown_version',[fact(1,version=None)],dict(claim,version=None),'not_established'),
        ('agent_commit',[fact(1,kind='commit',status='confirmed',source='agent_assertion')],dict(claim,kind='commit_confirmed'),'not_established'),
        ('tool_commit',[fact(1,kind='commit',status='confirmed')],dict(claim,kind='commit_confirmed'),'supported'),
        ('other_object_commit',[fact(1,kind='commit',status='confirmed',obj='module-b')],dict(claim,kind='commit_confirmed'),'not_established'),
        ('historical_scope',[fact(1),fact(2,status='failed')],dict(claim,prefix_step=1),'supported'),
    ]
    rows=[]
    for name,facts,c,expected in scenarios:
        state=EvidenceState()
        for f in facts: state.append(f)
        result=state.assess(c)
        rows.append(dict(case=name,facts=facts,claim=c,expected=expected,decision=result,
                         correct=result['assertion_support']==expected))
    return rows


def native_replay():
    root=Path('pilot/claim_prospective_eval_v1')
    protocol=read(root/'protocol.json')
    paths={s['session_id']:s for s in protocol['sessions']}
    data=read(Path('pilot/stage_claim_ablation_v1/inputs.json'))
    gold=read(Path('pilot/memory_mechanism_probe_v1/labels.ai.json'))
    rows=[]
    for session in data['sessions']:
        item=paths[session['session']]; raw=(root/item['path']).read_bytes()
        require(digest(raw)==item['sha256'],'Native source drift')
        events={x['id']:x for line in raw.decode().splitlines() for x in [json.loads(line)] if 'id' in x}
        facts=[]
        for step in sorted({x['source_step_id'] for x in session['records']}):
            records=[x for x in session['records'] if x['source_step_id']==step]
            ref=records[0]['raw_ref']; action=events[ref['action_id']]
            observation=events.get(ref.get('observation_id'))
            f=native_fact(action,observation,step,[x['record_id'] for x in records])
            # Unknown object/version is intentional; this replay scores observed
            # run status only, never current-version correctness.
            facts.append(f)
        state=EvidenceState()
        for f in facts: state.append(f)
        for g in gold:
            if not g['case_id'].startswith(session['session']+':'): continue
            prefix=int(g['case_id'].split(':')[1][6:])
            runs=state.query(prefix,kind='test_run',limit=1)
            predicted=runs[0]['status'] if runs else 'unknown'
            rows.append(dict(case_id=g['case_id'],expected_status=g['latest_run_status'],
                predicted_status=predicted,correct=predicted==g['latest_run_status'],facts=facts,
                scope='Existing seen native traces; narrow unittest parser acceptance, not failure detector accuracy'))
    return rows


def main():
    contrasts=controls(); native=native_replay()
    result=dict(controlled_cases=contrasts,native_cases=native,
        controlled_correct=sum(x['correct'] for x in contrasts),controlled_total=len(contrasts),
        native_status_correct=sum(x['correct'] for x in native),native_total=len(native),
        scope='Engineering acceptance, NOT held-out benchmark, learned comparison or online alarm efficacy')
    save(Path('out/evidence_state_acceptance_v1/results.json'),result)
    print({k:v for k,v in result.items() if k not in ('controlled_cases','native_cases')})


if __name__=='__main__': main()
