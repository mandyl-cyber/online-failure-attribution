import unittest

from segmenter import stage_claim_summary as core
from segmenter.claim_evidence_review import review


def row_for(action, observation):
    records = [dict(record_id='r:'+field, source_step_id=1, field=field,
                    text=text, content_sha256=core.digest(text.encode()), availability_seq=1)
               for field,text in [('action',action),('observation',observation)]]
    return core.build(dict(job_id='job',run_id='run',stage=dict(stage_id=1,start_step_id=1,end_step_id=1),
                           detected_at_seq=1,evidence=records,incomplete=False,task_context=None))


class EvidenceReviewTests(unittest.TestCase):
    def test_wrong_test_count_is_flagged_without_rewriting(self):
        row=row_for('python tests/runtests.py','Ran 7 tests in 0.1s\nOK\n[Command finished with exit code 0]')
        response=dict(claims=[dict(text='The runner reports 109 tests.',source='tool_observation',evidence_ids=['S1.observation.1.1'])])
        result=review(response,row)
        self.assertEqual(result['reviewed_claims'][0]['issues'][0]['code'],'test_count_absent_from_cited_result')
        self.assertEqual(response['claims'][0]['text'],'The runner reports 109 tests.')

    def test_cited_test_count_and_exit_are_allowed(self):
        row=row_for('python tests/runtests.py','Ran 7 tests in 0.1s\nOK\n[Command finished with exit code 0]')
        response=dict(claims=[dict(text='The runner reports 7 tests and exits with code 0.',source='tool_observation',evidence_ids=['S1.observation.1.1'])])
        self.assertEqual(review(response,row)['reviewed_claims'][0]['issues'],[])

    def test_uncited_failed_edit_is_retained_with_exact_offsets(self):
        text='ERROR:\nNo replacement was performed, old_str did not appear verbatim.'
        row=row_for('str_replace file.py',text)
        result=review(dict(claims=[]),row)
        signal=result['observation_signals'][0]
        self.assertEqual(text[signal['start']:signal['end']],signal['quote'])
        self.assertEqual(result['uncited_signal_ids'],[signal['signal_id']])

    def test_source_listing_does_not_create_traceback_event(self):
        row=row_for('cat log_fixture.txt','Traceback (most recent call last):\nRuntimeError: expected fixture')
        self.assertEqual(review(dict(claims=[]),row)['observation_signals'],[])

    def test_zero_tests_preserved_and_exit_mismatch_flagged(self):
        row=row_for('python tests/runtests.py','Ran 0 tests in 0.0s\nOK\n[Command finished with exit code 0]')
        response=dict(claims=[dict(text='The command exits with code 2.',source='tool_observation',evidence_ids=['S1.observation.1.1'])])
        result=review(response,row)
        self.assertEqual(result['observation_signals'][0]['kind'],'zero_tests_reported')
        self.assertEqual(result['reviewed_claims'][0]['issues'][0]['code'],'exit_code_absent_from_cited_result')


if __name__=='__main__':
    unittest.main()
