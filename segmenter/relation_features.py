"""Fixed numeric ablation, restricted to current step and eight prior actions."""
from copy import deepcopy


def extend_history(rows, position):
    result=deepcopy(rows[position])
    for lag in range(4,9):
        if position<lag:break
        for key,value in rows[position-lag]['features'].items():
            if key.startswith('lag0:'):
                result['features'][f'lag{lag}:'+key[5:]]=value
    return result


def relation_features(evidence):
    history=evidence['recent_action_history']
    ids={h['step_id'] for h in history}
    current=evidence['observed_through_step']
    failures=[e for e in evidence['recent_failure_candidates'] if e['step_id'] in ids or e['step_id']==current]
    links=[e for e in evidence['relation_candidates'] if e['from_step'] in ids]
    edit=evidence['recent_edit']
    return {'lag0:relation:'+k:v for k,v in dict(
        prior_exit_zero=sum(h['exit_code']==0 for h in history),
        prior_exit_nonzero=sum(h['exit_code'] is not None and h['exit_code']!=0 for h in history),
        prior_unknown_exit=sum(h['exit_code'] is None for h in history),
        prior_failure_count=sum(e['step_id']!=current for e in failures),
        prior_zero_tests=sum(e['step_id']!=current and e['kind']=='zero_tests' for e in failures),
        current_failure=int(any(e['step_id']==current for e in failures)),
        current_zero_tests=int(any(e['step_id']==current and e['kind']=='zero_tests' for e in failures)),
        shared_object_count=sum(e['kind']=='shared_lexical_object' for e in links),
        reads_failed_object=sum(e['kind']=='reads_object_named_in_prior_failed_action' for e in links),
        recent_reported_edit=int(bool(edit and edit['step_id'] in ids)),
    ).items()}
