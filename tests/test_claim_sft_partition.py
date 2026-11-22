from copy import deepcopy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from segmenter.claim_sft_backend import partition_requests,convert_partial,messages_for,ClaimSFTBackend,convert_response
import test_claim_sft_backend as fixtures


class PartitionTests(unittest.TestCase):
    def request(self):
        request=fixtures.BridgeTests().request()
        request['stage']['end_step_id']=3
        request['detected_at_seq']=3
        records=[]
        for step in (1,2,3):
            for original in request['evidence']:
                record=deepcopy(original)
                record.update(record_id=f'{step}:{original["field"]}',source_step_id=step,availability_seq=step)
                records.append(record)
        request['evidence']=records
        return request

    def test_whole_step_split_preserves_every_record_and_cutoff(self):
        request=self.request()
        parts,oversized=partition_requests(request,lambda child:len(child['evidence'])<=2)
        self.assertEqual(oversized,[])
        self.assertEqual([e for p in parts for e in p['evidence']],request['evidence'])
        self.assertTrue(all(p['detected_at_seq']==3 for p in parts))
        self.assertEqual([p['stage']['start_step_id'] for p in parts],[1,2,3])

    def test_single_step_oversize_is_explicit(self):
        request=self.request()
        parts,oversized=partition_requests(request,lambda child:child['stage']['start_step_id']==child['stage']['end_step_id']!=2)
        self.assertEqual(oversized,[2])
        self.assertEqual([p['stage']['start_step_id'] for p in parts],[1,3])

    def test_bad_claim_does_not_discard_good_neighbor_or_change_source(self):
        row,_=messages_for(fixtures.BridgeTests().request())
        response=dict(claims=[dict(text='Two tests passed.',source='tool_observation',evidence_ids=['S1.observation.1.1']),
                              dict(text='A result',source='tool_observation',evidence_ids=['S1.action.1.1'])])
        converted,rejected=convert_partial(response,row)
        self.assertEqual(len(converted['claims']),1)
        self.assertEqual(len(rejected),1)
        self.assertEqual(rejected[0]['raw_claim']['source'],'tool_observation')
        self.assertEqual(rejected[0]['error'],'tool observation needs observation evidence')

    def test_empty_claims_is_valid_without_invented_summary(self):
        row,_=messages_for(fixtures.BridgeTests().request())
        self.assertEqual(convert_partial(dict(claims=[]),row),(dict(claims=[]),[]))

    def test_later_generation_failure_keeps_earlier_part_claims(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                import json
                stage=json.loads(messages[1]['content'])['stage']
                return [1]*((stage['end_step_id']-stage['start_step_id']+1)*10+1)
        backend=ClaimSFTBackend.__new__(ClaimSFTBackend)
        backend.tokenizer=Tokenizer()
        backend.max_length=22
        backend.max_new_tokens=1
        def generate(child):
            if child['stage']['start_step_id']==3:
                raise RuntimeError('simulated generator failure')
            backend.part_rejections=[]
            row,_=messages_for(child)
            return convert_response(dict(claims=[dict(text='Observed two passing tests.',source='tool_observation',
                evidence_ids=['S1.observation.1.1'])]),row)
        backend._generate=generate
        with patch.dict('sys.modules',{'transformers':SimpleNamespace(AutoTokenizer=None)}):
            result=backend.extract(self.request())
        self.assertEqual(len(result['claims']),1)
        self.assertFalse(backend.extraction_details['complete'])
        self.assertEqual([p['status'] for p in backend.extraction_details['parts']],['generated','failed'])


if __name__=='__main__':
    unittest.main()
