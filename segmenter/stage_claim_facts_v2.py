"""Opt-in sidecar experiment separating outer status from reported inner results.

The production v1 extractor and v4/v5 job hashes remain unchanged. These signals
describe recorded output, not proof of patch correctness or subprocess identity.
"""
import re
from . import stage_claim_facts as legacy

VERSION='stage-observed-facts-v2-experimental'
INNER=re.compile(r'(?m)^\s*EXIT:\s*(-?\d+)\s*$')
ERROR=re.compile(r'(?m)^\s*(?:E\s+)?((?:[A-Za-z_]\w*\.)*(?:[A-Za-z_]\w*(?:Error|Exception)|Error|Exception)):\s*([^\n]*)')
WARNING=re.compile(r'(?im)^\s*bash:.*warning: here-document[^\n]*')
PATH=re.compile(r"(?m)^Testing against Django installed in '([^']+)'")


def extract(source,evidence_map=None):
    facts=legacy.extract(source,evidence_map)
    grouped=legacy._records(source)
    by_step={f['step_id']:f for f in facts}
    for step,unit in sorted(grouped.items()):
        action=unit.get('action',{}).get('text','')
        obs=unit.get('observation',{}).get('text','')
        outer=legacy._last_int(legacy.EXIT,obs)
        # A source listing can contain exception-like text; do not treat it as an
        # executed exception. Full shell/source-echo parsing is not claimed here.
        read_only=bool(legacy.READ.search(action))
        inner=[] if read_only else [int(m.group(1)) for m in INNER.finditer(obs)]
        errors=[] if read_only else [dict(name=m.group(1),message=m.group(2)) for m in ERROR.finditer(obs)]
        warning=[] if read_only else [m.group(0).strip() for m in WARNING.finditer(obs)]
        paths=[] if read_only else PATH.findall(obs)
        fact=by_step.get(step)
        if fact is None and not (inner or errors or warning or paths):continue
        if fact is None:
            records=[unit[k] for k in ('action','observation') if k in unit]
            record_ids=[r['record_id'] for r in records]
            fact=dict(step_id=step,type='command_observation',summary='Recorded command output; outcome not established.',
                exit_code=outer,test_count=None,skipped=None,evidence_record_ids=record_ids,
                evidence_ids=sorted(e for e,r in (evidence_map or {}).items() if r['record_id'] in record_ids))
            facts.append(fact)
        original_type=fact['type']
        if original_type in ('read_only','plan_update','edit_acknowledged') and not (inner or errors or warning or paths):
            fact['fact_id']=f'{VERSION}:{step}:{original_type}'
            continue
        positive=original_type=='verification_pass'
        nonzero=any(x!=0 for x in inner)
        verify=original_type.startswith('verification_') or original_type=='zero_tests'
        fact.update(outer_exit_code=outer,reported_inner_exit_codes=inner,
            reported_inner_exit_code=inner[-1] if inner else None,
            reported_errors=errors,warnings=warning,reported_execution_paths=paths,
            positive_test_result_observed=positive,
            verification_result='pass_in_shown_scope' if positive else 'not_established',
            status_scope='Outer status and standalone EXIT markers are separate observations; marker-to-subprocess identity is not independently verified.')
        if original_type=='interrupted':
            fact['verification_result']='interrupted'
        elif original_type in ('verification_unavailable','zero_tests'):
            fact['verification_result']='unavailable' if original_type=='verification_unavailable' else 'zero_tests'
        elif positive and (nonzero or errors):
            fact.update(type='mixed_outcomes',verification_result='mixed',summary='Positive test text coexists with reported errors/nonzero inner status; do not summarize the entire command as passing.')
        elif nonzero:
            fact.update(type='verification_failed' if verify else 'reported_inner_failure',verification_result='reported_failure',
                summary='Output reports a nonzero inner EXIT marker independently of the outer shell status; a pass is not established.')
        elif errors:
            fact.update(type='verification_failed' if verify else 'reported_error',verification_result='reported_error',
                summary='Output reports error text independently of outer shell status. Caught/uncaught process failure is not inferred from error text alone.')
        elif original_type=='verification_failed':
            fact['verification_result']='failed'
        elif warning and not positive:
            fact.update(type='command_warning',verification_result='not_established',summary='Command output reports a shell heredoc warning; exit zero does not establish that the intended verification ran.')
        fact['fact_id']=f'{VERSION}:{step}:{fact["type"]}'
    return sorted(facts,key=lambda f:f['step_id'])
