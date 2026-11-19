"""Conservative, causal facts from sealed action/observation evidence.

Facts are observations about the recorded interaction, not claims that a patch
works. They are sidecar evidence for stage summaries and later alarms.
"""
import re


VERSION = 'stage-observed-facts-v1'

VERIFY = re.compile(
    r'(?i)(?:^|[;&|]\s*)(?:pytest\b|python(?:\d+(?:\.\d+)?)?\s+(?:-m\s+pytest\b|'
    r'-m\s+unittest\b|tests/runtests\.py\b)|npm\s+(?:(?:-s|--silent)\s+)?'
    r'(?:run\s+)?test[\w:-]*\b|cargo\s+test\b|go\s+test\b|'
    r'bash\s+[^\n]*test[^\n]*\.sh\b)')
READ = re.compile(r'(?i)^\s*(?:sed\b|grep\b|rg\b|cat\b|head\b|tail\b|view\b|read\b|ls\b)')
EXIT = re.compile(r'(?i)(?:command finished with exit code|the command completed with exit code|exit[_ ]code)\s*[=:]?\s*(-?\d+)')
RAN = re.compile(r'(?i)\bRan\s+(\d+)\s+tests?\b')
FOUND = re.compile(r'(?i)\bFound\s+(\d+)\s+test\(s\)')
SKIPPED = re.compile(r'(?i)\bskipped\s*[=:]\s*(\d+)')
EXCEPTION = re.compile(r'(?m)^\s*([A-Za-z_]\w*(?:Error|Exception))(?::|\s*$)')
NO_RUNNER = re.compile(r'(?i)(?:pytest:\s*command not found|no module named pytest|'
                       r'(?:npm|cargo|go):\s*command not found)')
PYTEST_PASS = re.compile(r'(?m)^\s*\.+\s*\[100%\]')
SUCCESS_TEXT = re.compile(r'(?i)\b(?:tests?|suite)\s+(?:\w+\s+){0,8}(?:pass(?:ed)?|succeed(?:ed)?|successful(?:ly)?)\b|'
                          r'\ball\s+tests?\s+pass(?:ed)?\b|\b(?:fix|patch|change)\s+(?:(?:was|is|has been)\s+)?verified\b')
TEST_ACTIVITY = re.compile(r'(?i)\b(?:tested|ran\s+tests?|executed\s+tests?)\b')
QUALIFIED_ASSERTION = re.compile(r'(?i)\b(?:plan|agent|task list|notes|commit message)\b.{0,70}'
                                 r'\b(?:says?|states?|asserts?|claims?|reports?|describes?|notes?|indicates?)\b|'
                                 r'\b(?:according to|per)\s+(?:the\s+)?(?:plan|agent|task list|notes|commit message)\b')


def _last_int(pattern, text):
    matches = list(pattern.finditer(text))
    return int(matches[-1].group(1)) if matches else None


def _records(source):
    grouped = {}
    for record in source['evidence']:
        unit = grouped.setdefault(record['source_step_id'], {})
        if record['field'] in unit:
            raise ValueError('Ambiguous duplicate evidence field')
        unit[record['field']] = record
    return grouped


def extract(source, evidence_map=None):
    """Return stable, supported facts for the sealed stage only.

    The caller should first validate source integrity and temporal membership,
    e.g. with stage_claim_summary.build(source). `evidence_map` maps v3 chunk
    IDs back to source records and is optional for ledger-sidecar use.
    """
    facts = []
    for sid, unit in sorted(_records(source).items()):
        action_record = unit.get('action')
        observation_record = unit.get('observation')
        action = action_record['text'] if action_record else ''
        obs = observation_record['text'] if observation_record else ''
        exit_code = _last_int(EXIT, obs)
        test_count = _last_int(RAN, obs)
        if test_count is None:
            test_count = _last_int(re.compile(r'(?i)\b(\d+)\s+passed\b'),obs)
        if test_count is None:
            test_count = _last_int(FOUND, obs)
        skipped = _last_int(SKIPPED, obs)
        is_verify = bool(VERIFY.search(action))
        is_verify = is_verify or bool(re.search(
            r'(?:^|[\s;&])(?:/[^\s]+/)?python[\d.]*\s+(?:-[\w]+\s+)*-m\s+pytest\b',action))
        kind = summary = None
        if obs and (re.search(r'(?i)\bCTRL\+C\b|\^C\b', obs) or action.strip() == 'C-c' or exit_code == 130):
            kind, summary = 'interrupted', 'The recorded command was interrupted; no successful verification is established.'
        elif is_verify and obs and NO_RUNNER.search(obs):
            kind, summary = 'verification_unavailable', 'The verification command could not run because its test runner is unavailable.'
        elif is_verify and obs and test_count == 0:
            kind, summary = 'zero_tests', 'The verification command reported running zero tests; this does not verify the target behavior.'
        elif is_verify and obs and exit_code is not None and exit_code != 0:
            kind, summary = 'verification_failed', f'The verification command exited with code {exit_code}; success is not established.'
        elif is_verify and obs and ((test_count is not None and test_count > 0 and re.search(r'(?im)^\s*OK\b|\b\d+\s+passed\b', obs))
                                or (PYTEST_PASS.search(obs) and exit_code == 0)):
            kind, summary = 'verification_pass', 'The recorded verification reports success within the shown test scope; patch correctness beyond that scope is unknown.'
        elif is_verify and obs and exit_code == 0:
            kind, summary = 'verification_exit_zero', 'The verification command exited with code 0; a positive test count or full pass is not established.'
        elif obs and 'Traceback' in obs and EXCEPTION.search(obs):
            name = EXCEPTION.search(obs).group(1)
            kind, summary = 'runtime_exception', f'The recorded command raised {name}; success is not established.'
        elif obs and exit_code == 0 and re.search(r'(?i)^\s*(?:/\S+/)?python[\d.]*\s+(?:-\s*<<|[^\s;]+\.py\b)',action):
            kind, summary = 'script_exit_zero', 'The inline script exited with code zero within its recorded scope.'
        elif obs and re.search(r'(?i)\bfile\s+[^\n]{1,250}\s+has been edited\b', obs):
            kind, summary = 'edit_acknowledged', 'The tool acknowledged a file edit; this does not establish a working fix.'
        elif obs and READ.search(action):
            kind, summary = 'read_only', 'The step read or searched existing material; it did not itself run tests or modify code.'
        elif obs and ('task list has been updated' in obs.lower() or '"command":"plan"' in action.replace(' ','')):
            kind, summary = 'plan_update', 'The agent task list was updated; listed work is an assertion, not proof of execution.'
        if kind is None:
            continue
        record_ids = [r['record_id'] for r in (action_record, observation_record) if r]
        ids = sorted(eid for eid, ref in (evidence_map or {}).items() if ref['record_id'] in record_ids)
        facts.append(dict(fact_id=f'{VERSION}:{sid}:{kind}', type=kind, step_id=sid,
                          summary=summary, exit_code=exit_code, test_count=test_count,
                          skipped=skipped, evidence_record_ids=record_ids,
                          evidence_ids=ids))
    return facts


def guard_claim(claim, evidence_map, facts):
    """Reject only high-risk claims whose cited step lacks a supporting result.

    This is a narrow safety check, never a general semantic entailment score.
    """
    text = claim['text']
    cited = set(claim['evidence_ids'])
    cited_steps = {evidence_map[eid]['source_step_id'] for eid in cited}
    if TEST_ACTIVITY.search(text) and claim['source'] == 'tool_observation' and not any(
            fact['step_id'] in cited_steps and (fact['type'].startswith('verification_') or fact['type']=='script_exit_zero')
            for fact in facts):
        return 'test_activity_without_verification_action'
    if not SUCCESS_TEXT.search(text):
        return None
    if claim['source'] == 'agent_assertion' and QUALIFIED_ASSERTION.search(text):
        return None
    if claim['source'] != 'tool_observation':
        return 'unqualified_success_assertion'
    if (re.search(r'(?i)\b(?:executed|ran) successfully\b',text)
            and not re.search(r'(?i)\b(?:suite|all tests|patch|fix|passed)\b',text)
            and any(f['type']=='script_exit_zero' and any(
                eid in cited and evidence_map[eid]['field']=='observation' for eid in f['evidence_ids'])
                for f in facts)):
        return None
    supporting = [fact for fact in facts if fact['type'] == 'verification_pass'
                  and any(eid in cited and evidence_map[eid]['field'] == 'observation'
                          for eid in fact['evidence_ids'])]
    if not supporting:
        return 'verification_success_without_observed_pass'
    if re.search(r'(?i)\b(?:fix|patch|change)\s+(?:(?:was|is|has been)\s+)?verified\b', text):
        return 'patch_correctness_not_established'
    return None


def missing_critical_facts(claims, facts):
    """Return unmentioned high-risk observations; never mutate claims to hide them."""
    patterns = {
        'interrupted': re.compile(r'(?i)interrupt|ctrl\+c|exit\s*(?:code\s*)?130|\^C'),
        'verification_unavailable': re.compile(r'(?i)command.{0,25}not (?:be )?found|no module named|unavailable|missing pytest|pytest.{0,50}not (?:be )?found|absence of.{0,30}pytest.{0,15}module'),
        'zero_tests': re.compile(r'(?i)\b(?:0|zero|no)\s+tests?\b'),
        'verification_failed': re.compile(r'(?i)fail|error|exit\s*(?:code\s*)?[1-9]\d*'),
        'runtime_exception': re.compile(r'(?i)exception|error|traceback'),
    }
    missing = []
    for fact in facts:
        pattern = patterns.get(fact['type'])
        if pattern is None:
            continue
        if not any(pattern.search(claim['text'])
                   and any(eid in claim['evidence_ids'] for eid in fact['evidence_ids'])
                   for claim in claims):
            missing.append(fact['fact_id'])
    return missing
