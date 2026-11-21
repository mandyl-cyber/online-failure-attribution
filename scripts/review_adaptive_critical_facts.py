"""Materialize a source-first AI review of selected critical observed results."""
import hashlib
from pathlib import Path
from scripts.run_stage_claim_pipeline import read, save


# Group, source step, reviewed fact. The list is selective and predates generation.
FACTS = [
    ('x005', 43, 'The reproduction aborts with RuntimeError because model __main__.A lacks an explicit app_label and is outside INSTALLED_APPS. Exit 1.'),
    ('x005', 44, 'A later reproduction aborts with OperationalError: no such table: auth_user. Exit 1.'),
    ('x008', 34, 'The Django test runner aborts importing test_sqlite. Exit 1. This invocation does not establish passing tests.'),
    ('x009', 37, 'The runner reports 20 tests, OK, exit 0, using Django installed in /testbed/django.'),
    ('x009', 39, 'The runner reports 7 tests, OK, exit 0, using Django installed in /testbed/django.'),
    ('x012', 45, 'pytest is not found. Exit 127. No tests execute in this invocation.'),
    ('x012', 46, 'python -m pytest fails because the pytest module is absent. Exit 1.'),
    ('x015', 36, 'The probe aborts with ImproperlyConfigured because Django settings are not configured. Exit 1.'),
    ('x015', 43, 'The runner reports 461 tests, OK, 11 skipped and 2 expected failures, exit 0, against /workspace/django__django__4.2/django.'),
    ('x015', 45, 'The runner reports 109 tests, OK, exit 0, against /workspace/django__django__4.2/django.'),
    ('x015', 48, 'The runner reports 461 tests, OK, 11 skipped and 2 expected failures, exit 0, against /workspace/django__django__4.2/django.'),
    ('x016', 23, 'pytest is not found. Exit 127.'),
    ('x016', 26, 'The runner reports Ran 0 tests and OK, exit 0, against /testbed/django. This provides no executed-test evidence.'),
    ('x018', 32, 'pytest is not found. Exit 127.'),
    ('x018', 33, 'The Python interpreter reports No module named pytest. Exit 1.'),
    ('x021', 26, 'The probe shows the figure registry becoming empty after get_backend, then raises AssertionError and exits 1. The Assertion passed text is in the echoed script, not its executed output.'),
    ('x025', 22, 'The test run reports 1 failed, 1384 passed, 59 skipped and 206 deselected, exit 1. recursion_error_3152 is the failed functional test.'),
    ('x027', 25, 'The replacement fails because old_str does not occur verbatim. No replacement was performed.'),
    ('x027', 27, 'Another replacement fails because old_str does not occur verbatim. No replacement was performed.'),
    ('x027', 29, 'The replacement again reports No replacement was performed.'),
    ('x027', 30, 'The replacement again reports No replacement was performed.'),
    ('x028', 16, 'The test run reports 126 failed, 1485 passed and 21 skipped, exit 1. The invocation is not a passing suite.'),
    ('x031', 19, 'The replacement in monospaceLineBreaksComputer.ts fails because old_str does not occur verbatim. No replacement was performed.'),
]


def main():
    root = Path('pilot/adaptive_seal_claim_v1')
    packet = read(root/'fact_review.json')
    groups = {g['group_id']: g for g in packet['groups']}
    reviewed = []
    for group_id, step, text in FACTS:
        group = groups[group_id]
        point = next(p for p in group['candidate_points'] if p['source_step_id'] == step and p['field'] == 'observation')
        reviewed.append(dict(point_id=point['point_id'], group_id=group_id, task=group['task'],
            source_step_id=step, source_sha256=point['source_sha256'], fact=text,
            membership=point['membership'], reviewed=True, reviewer_type='AI',
            coverage=dict(immediate=None, adaptive=None),
            prohibited_inference='Do not extrapolate command success, edit receipt or agent status into verified correctness.'))
    save(root/'critical_facts.json', dict(scope='Selected critical execution outcomes reviewed before new 7B generation',
        exhaustive=False, human_reviewed=False, facts=reviewed,
        group_coverage={g:sum(p['group_id']==g for p in reviewed) for g in groups}))
    protocol = read(root/'protocol.json')
    protocol['critical_facts_sha256'] = hashlib.sha256((root/'critical_facts.json').read_bytes()).hexdigest()
    save(root/'protocol.json', protocol)
    print(dict(reviewed_facts=len(reviewed), groups_with_selected_facts=len({p['group_id'] for p in reviewed}), total_groups=len(groups)))


if __name__ == '__main__':
    main()
