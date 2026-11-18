"""Stage claim v4: v3 evidence plus conservative observed facts and a narrow guard."""
from copy import deepcopy

from . import stage_claim_summary as v3
from .stage_claim_facts import VERSION as FACT_VERSION, extract, guard_claim, missing_critical_facts


PROTOCOL = 'stage-claim-summary-v4'
SCHEMA = v3.SCHEMA
INSTRUCTION = v3.INSTRUCTION + '''
The observed_facts field is a deterministic sidecar derived from paired actions
and observations. Use it to preserve interrupted commands, unavailable test
runners, zero-test runs, failed verification and runtime exceptions. It is not a
substitute for citations: every output claim must cite raw evidence IDs from units.
A zero exit code without an observed positive test result is not a passing test.
Reading test source is not running tests. If a task list says a test passed, report
that only as an attributed agent assertion; never convert it to a tool outcome.
Do not claim that a patch is verified merely because one scoped test passed.'''


def build(source, chunk_chars=1200):
    base = v3.build(source, chunk_chars)
    facts = extract(base['source_snapshot'], base['evidence_map'])
    request = deepcopy(base['request'])
    request['protocol'] = PROTOCOL
    request['observed_facts_version'] = FACT_VERSION
    request['observed_facts'] = deepcopy(facts)
    key = v3.digest(v3.encoded(dict(protocol=PROTOCOL, base_job_id=base['job_id'], facts=facts)))
    return dict(job_id=key, source_job_id=base['source_job_id'], run_id=base['run_id'],
                source_sha256=base['source_sha256'], input_sha256=v3.digest(v3.encoded(request)),
                chunk_chars=chunk_chars, request=request, evidence_map=deepcopy(base['evidence_map']),
                source_snapshot=deepcopy(base['source_snapshot']), base_row=base,
                observed_facts=facts)


def verify_row(row):
    if build(row['source_snapshot'], row['chunk_chars']) != row:
        raise ValueError('v4 input or fact sidecar drift')


def validate(response, row):
    """Validate a whole batch; rejected raw responses remain with the caller."""
    verify_row(row)
    claims = v3.validate(response, row['base_row'])
    for i, raw in enumerate(response['claims'], 1):
        reason = guard_claim(raw, row['evidence_map'], row['observed_facts'])
        if reason:
            raise ValueError(f'claim {i}: {reason}')
    for i, claim in enumerate(claims, 1):
        claim['claim_id'] = f'{row["job_id"]}:c{i}'
    return dict(claims=claims,
                missing_critical_fact_ids=missing_critical_facts(response['claims'], row['observed_facts']),
                facts_version=FACT_VERSION,
                semantic_quality='not_fully_evaluated')
