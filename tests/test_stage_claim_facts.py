import unittest

from segmenter import stage_claim_summary as v3
from segmenter import stage_claim_summary_v4 as v4


def source(action, observation, step=1):
    evidence = []
    for field, text in [('action',action),('observation',observation)]:
        evidence.append(dict(record_id=f'run:{step}:{field}', source_step_id=step,
                             field=field, text=text,
                             content_sha256=v3.digest(text.encode('utf-8')),
                             availability_seq=step))
    return dict(job_id='job',run_id='run',stage=dict(stage_id=1,start_step_id=step,end_step_id=step),
                detected_at_seq=step,evidence=evidence,task_context=None,incomplete=False)


def claim(text, source_type='tool_observation', action=False):
    ids = ['S1.observation.1.1']
    if action:
        ids.insert(0,'S1.action.1.1')
    return dict(text=text,source=source_type,evidence_ids=ids)


class FactTests(unittest.TestCase):
    def test_partial_write_and_absence_expression(self):
        from segmenter import stage_claim_summary_v5 as v5
        row = v5.build(source("python -m pytest -q",
            "No module named pytest\n[Command finished with exit code 1]"))
        response = {"claims":[
            claim("Executing python -m pytest failed due to the absence of the pytest module."),
            claim("The test suite ran successfully.")]}
        checked = v5.validate_partial(response,row)
        self.assertEqual(len(checked["claims"]),1)
        self.assertEqual(len(checked["rejected_claims"]),1)
        self.assertTrue(checked["incomplete"])
        self.assertEqual(checked["missing_critical_fact_ids"],[])

    def test_zero_tests_is_not_pass(self):
        row = v4.build(source('python tests/runtests.py -k target',
                              'Found 0 test(s).\nRan 0 tests in 0.000s\nOK\n[Command finished with exit code 0]'))
        self.assertEqual(row['observed_facts'][0]['type'],'zero_tests')
        with self.assertRaisesRegex(ValueError,'verification_success_without_observed_pass'):
            v4.validate({'claims':[claim('All tests passed.')]},row)
        self.assertEqual(v4.validate({'claims':[]},row)['missing_critical_fact_ids'],
                         [row['observed_facts'][0]['fact_id']])

    def test_positive_scoped_pass_and_no_patch_verification(self):
        row = v4.build(source('python tests/runtests.py model_forms',
                              'Ran 406 tests in 0.659s\nOK (skipped=14)\n[Command finished with exit code 0]'))
        fact = row['observed_facts'][0]
        self.assertEqual((fact['type'],fact['test_count'],fact['skipped']),('verification_pass',406,14))
        self.assertEqual(len(v4.validate({'claims':[claim('406 tests passed with 14 skipped.')]},row)['claims']),1)
        with self.assertRaisesRegex(ValueError,'patch_correctness_not_established'):
            v4.validate({'claims':[claim('The patch was verified by these tests.')]},row)

    def test_unavailable_runner_and_reading_are_distinct(self):
        row = v4.build(source('pytest -q','bash: pytest: command not found\n[Command finished with exit code 127]'))
        self.assertEqual(row['observed_facts'][0]['type'],'verification_unavailable')
        with self.assertRaisesRegex(ValueError,'verification_success_without_observed_pass'):
            v4.validate({'claims':[claim('The test suite ran successfully.')]},row)
        read = v4.build(source('sed -n 1,20p tests/test_a.py','def test_a(): pass\n[Command finished with exit code 0]'))
        self.assertEqual(read['observed_facts'][0]['type'],'read_only')
        with self.assertRaisesRegex(ValueError,'test_activity_without_verification_action'):
            v4.validate({'claims':[claim('Tested specific sections of test_a.py.')]},read)

    def test_interruption_and_runtime_exception(self):
        interrupted = v4.build(source('C-c','^C\nCTRL+C was sent.\n[Command finished with exit code 130]'))
        self.assertEqual(interrupted['observed_facts'][0]['type'],'interrupted')
        self.assertEqual(v4.validate({'claims':[claim('The command was interrupted with Ctrl+C.')]},interrupted)
                         ['missing_critical_fact_ids'],[])
        error = v4.build(source('python - <<PY\nraise NotImplementedError\nPY',
                                'Traceback (most recent call last):\nNotImplementedError\n[Command finished with exit code 1]'))
        self.assertEqual(error['observed_facts'][0]['type'],'runtime_exception')

    def test_plan_success_must_be_attributed(self):
        row = v4.build(source('{"command":"plan","task_list":[{"notes":"auth_tests passed"}]}',
                              'Task list has been updated with 1 items.'))
        self.assertEqual(row['observed_facts'][0]['type'],'plan_update')
        bad = claim('The auth_tests suite ran successfully.','tool_observation',True)
        with self.assertRaisesRegex(ValueError,'verification_success_without_observed_pass'):
            v4.validate({'claims':[bad]},row)
        attributed = claim('The task list says auth_tests passed.','agent_assertion',True)
        self.assertEqual(len(v4.validate({'claims':[attributed]},row)['claims']),1)

    def test_test_path_is_not_test_success_and_input_is_immutable(self):
        row = v4.build(source('sed -n 1,20p tests/test_a.py',
                              'Command executed successfully.\ndef test_a(): pass'))
        text = 'The tests/test_a.py source was read; the command executed successfully.'
        self.assertEqual(len(v4.validate({'claims':[claim(text)]},row)['claims']),1)
        row['observed_facts'][0]['type'] = 'verification_pass'
        with self.assertRaisesRegex(ValueError,'input or fact sidecar drift'):
            v4.validate({'claims':[claim(text)]},row)


if __name__=='__main__':
    unittest.main()
