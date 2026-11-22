"""Narrow claim checks and verbatim observations for downstream ledger review."""
import re

from . import stage_claim_summary as core
from . import stage_claim_facts as facts

VERSION = 'claim-evidence-review-v2-result-scope'
CLAIM_EXIT = re.compile(r'(?i)\bexit(?:ed|s)?\s+(?:with\s+)?(?:code\s+)?(-?\d+)\b')
CLAIM_COUNT = re.compile(r'(?i)\b(\d+)\s+tests?\b')
OBSERVED_COUNT = re.compile(r'(?im)^\s*Ran\s+(\d+)\s+tests?\b|\b(\d+)\s+passed\b')
SIGNALS = {
    'replacement_not_performed': re.compile(r'(?m)^No replacement was performed[^\n]*'),
    'missing_tool_or_module': re.compile(r'(?m)^bash: [^\n]*command not found[^\n]*|^[^\n]*: No module named [^\n]+'),
    'reported_traceback': re.compile(r'(?m)^Traceback \(most recent call last\):'),
    'zero_tests_reported': re.compile(r'(?m)^\s*Ran 0 tests?[^\n]*'),
    'externally_managed_environment': re.compile(r'(?m)^error: externally-managed-environment[^\n]*'),
    'venv_creation_failed': re.compile(r'(?m)^The virtual environment was not created successfully because ensurepip is not\s*\navailable\.[^\n]*'),
}


def result_scope_issues(text, cited_observations, *, has_observation=True):
    """Narrow completion checks, not general entailment or a correctness score."""
    issues=[]
    qualified=bool(facts.QUALIFIED_ASSERTION.search(text))
    if not has_observation and not qualified and not re.search(r'(?i)\b(?:requested|plans?|intends?|attempts?)\b',text):
        if re.search(r'(?i)\b(?:is|was|were|has been)\s+(?:viewed|read|created|edited|modified|installed)\b',text):
            issues.append(dict(code='completed_action_without_observed_result'))
    installation=re.search(r'(?i)\b(?:installs|installed)\b|\binstallation of (?:the )?packages\b',text)
    negative=re.search(r'(?i)\b(?:not|never|cannot|could not|failed|fails|failure|unavailable)\b',text)
    completion=re.search(r'(?im)^\s*(?:Successfully installed\b|Setting up\s+\S+)',cited_observations)
    if installation and not negative and not qualified and not completion:
        issues.append(dict(code='installation_completion_absent_from_cited_result'))
    return issues


def citation_issues(text, cited_observations):
    exits = {int(m.group(1)) for m in facts.EXIT.finditer(cited_observations)}
    counts = {int(m.group(1) or m.group(2)) for m in OBSERVED_COUNT.finditer(cited_observations)}
    return ([dict(code='test_count_absent_from_cited_result',value=int(m.group(1)))
             for m in CLAIM_COUNT.finditer(text) if int(m.group(1)) not in counts]
            + [dict(code='exit_code_absent_from_cited_result',value=int(m.group(1)))
               for m in CLAIM_EXIT.finditer(text) if int(m.group(1)) not in exits])


def review(response, row):
    """Preserve raw claims, flag narrow citation conflicts, retain source signals.

    Passing these checks does not certify general semantic support. No evidence
    IDs or claim text are repaired. Observation signals remain separate records.
    """
    claims = core.validate(response, row)
    records = facts._records(row['source_snapshot'])
    mapping = row['evidence_map']
    signals = []
    for step, unit in sorted(records.items()):
        action = unit.get('action', {}).get('text', '')
        observation = unit.get('observation')
        if not observation or facts.READ.search(action):
            continue
        text = observation['text']
        for kind, pattern in SIGNALS.items():
            for match in pattern.finditer(text):
                refs = {eid:ref for eid,ref in mapping.items()
                        if ref['record_id']==observation['record_id']
                        and ref['start']<match.end() and ref['end']>match.start()}
                if not refs:
                    raise ValueError('Observation signal has no supporting chunk')
                signals.append(dict(signal_id=f'{row["job_id"]}:{step}:{kind}:{match.start()}',
                    kind=kind,source_step_id=step,record_id=observation['record_id'],
                    quote=match.group(0),start=match.start(),end=match.end(),
                    evidence_ids=sorted(refs),content_sha256=observation['content_sha256'],
                    availability_seq=observation['availability_seq']))
    reviewed = []
    for claim in claims:
        cited_observations = '\n'.join(mapping[e]['quote'] for e in claim['evidence_ids'] if mapping[e]['field']=='observation')
        issues = []
        # These checks apply to observation assertions, avoiding plan-text numbers.
        if claim['source']=='tool_observation':
            issues = citation_issues(claim['text'], cited_observations)
        reviewed.append(dict(claim=claim,issues=issues,status='needs_review' if issues else 'narrow_checks_passed'))
    for signal in signals:
        # Citation presence is reported as such. It does not imply text coverage.
        signal['citing_claim_ids'] = [c['claim_id'] for c in claims
                                     if set(c['evidence_ids']) & set(signal['evidence_ids'])]
    return dict(version=VERSION,job_id=row['job_id'],input_sha256=row['input_sha256'],
                reviewed_claims=reviewed,observation_signals=signals,
                uncited_signal_ids=[s['signal_id'] for s in signals if not s['citing_claim_ids']],
                semantic_review='pending_except_separately_reviewed_cases')
