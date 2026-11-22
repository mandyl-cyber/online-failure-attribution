import unittest

from segmenter.claim_extraction import build_request,validate_response
from segmenter.claim_sft_backend import messages_for,convert_response
from segmenter import stage_claim_summary as core


class BridgeTests(unittest.TestCase):
    def request(self):
        records=[dict(record_id=field,run_id='r',source_step_id=1,field=field,text=text,
                      content_sha256=core.digest(text.encode()),availability_seq=1)
                 for field,text in [('action','python -m pytest'),('observation','2 passed\n[Command finished with exit code 0]')]]
        job=dict(job_id='j',run_id='r',stage=dict(stage_id=1,start_step_id=1,end_step_id=1),
                 detected_at_seq=1,incomplete=False,evidence=records,candidates=[],task_context=None)
        return build_request(job)

    def test_exact_chunk_quotes_convert_to_native_record_spans(self):
        request=self.request()
        row,messages=messages_for(request)
        result=convert_response(dict(claims=[dict(text='2 tests passed.',source='tool_observation',
                                  evidence_ids=['S1.observation.1.1'])]),row)
        self.assertEqual(validate_response(result,request)['status'],'passed')
        ref=result['claims'][0]['evidence_refs'][0]
        self.assertEqual(ref['evidence_id'],'observation')
        self.assertEqual(ref['quote'],request['evidence'][1]['text'][ref['start']:ref['end']])
        self.assertEqual(messages[0]['content'],core.INSTRUCTION)

    def test_unknown_evidence_is_rejected(self):
        row,_=messages_for(self.request())
        with self.assertRaisesRegex(ValueError,'unknown evidence'):
            convert_response(dict(claims=[dict(text='Passed',source='tool_observation',evidence_ids=['foreign'])]),row)

    def test_plan_stays_agent_assertion(self):
        request=self.request()
        row,_=messages_for(request)
        result=convert_response(dict(claims=[dict(text='The agent intends to run pytest.',source='agent_assertion',
                                                evidence_ids=['S1.action.1.1'])]),row)
        self.assertEqual(result['claims'][0]['basis'],'agent_assertion')
        self.assertEqual(validate_response(result,request)['status'],'passed')


if __name__=='__main__':
    unittest.main()
