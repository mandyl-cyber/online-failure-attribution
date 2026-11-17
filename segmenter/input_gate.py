"""Deterministic stage input gate; unknown completeness is not proof of completeness."""
from .ledger import atomic, encoded, sha
from pathlib import Path
import json


def stage_check(stage, checks, incomplete=False):
    reasons=[]
    if incomplete: reasons.append(dict(reason='stage_interrupted'))
    for idx in range(stage['start_step_id'],stage['end_step_id']+1):
        check=checks.get(idx)
        if check is None:
            reasons.append(dict(step_id=idx,reason='missing_input_integrity'))
            continue
        reasons.extend(dict(step_id=idx,reason=issue) for issue in check.get('issues',[]))
    return dict(version=1,status='blocked' if reasons else 'eligible',reasons=reasons,
                completeness='unknown_unless_explicitly_incomplete',
                scope='known input defects only; not semantic validation')


def blocked(job):
    return bool(job.get('incomplete') or job.get('input_integrity',{}).get('status')=='blocked')


def dispatch(processor, job_path, result_dir):
    job=json.loads(Path(job_path).read_text(encoding='utf-8'))
    if not blocked(job): return processor(job_path,result_dir)
    result=dict(job_id=job['job_id'],input_sha256=sha(encoded(job)),status='blocked_input',
                input_integrity=job.get('input_integrity'),incomplete=job.get('incomplete',False),
                claims=[],edges=[],semantic_correctness='not_evaluated')
    target=Path(result_dir)/Path(job_path).name
    if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=result:
        raise ValueError('Existing batch conflicts with blocked input')
    atomic(target,result)
    return result
