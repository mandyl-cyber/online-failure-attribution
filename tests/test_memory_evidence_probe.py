import unittest
from copy import deepcopy
from segmenter import memory_evidence_probe as e


class EvidenceTests(unittest.TestCase):
    def test_pool_is_whole_prefix_and_no_future(self):
        jobs,_,_=e.load()
        pools=[j for j in jobs if j['arm']=='model_lookup']
        self.assertEqual(len(pools),32)
        for j in pools:
            self.assertTrue(all(x['source_step_id']<=j['query']['prefix_step'] and
                                x['availability_seq']<=j['query']['prefix_step'] for x in j['pool']))
            self.assertTrue(any(x['source_step_id']!=j['query']['target_step'] for x in j['pool']))
            _,mapping=e.lookup_prompt(j)
            with self.assertRaises(ValueError): e.parse_lookup('{"lookup_ids":["E99999"]}',mapping,j['pool'])

    def test_lookup_retrieves_selected_source_only(self):
        jobs,_,_=e.load(); j=next(x for x in jobs if x['arm']=='model_lookup')
        _,mapping=e.lookup_prompt(j)
        record=j['pool'][0]
        alias=next(k for k,v in mapping.items() if v==record['record_id'])
        messages=[]
        def generate(prompt,remaining):
            messages.append(prompt)
            if len(messages)==1: return '{"lookup_ids":["'+alias+'"]}',10,100,.1
            return '{"category":"unknown","evidence_ids":[]}',10,100,.1
        result=e.evaluate(j,generate)
        self.assertEqual(result['selected_original_ids'],[record['record_id']])
        self.assertEqual(result['input_tokens'],200)
        self.assertEqual(len(result['rounds']),2)
        self.assertTrue(result['answer']['valid'])
        self.assertEqual(result['rounds'][0]['messages'][0]['content'],messages[0][0]['content'])

    def test_self_review_does_not_see_labels_and_soft_limits_do_not_invalidate(self):
        jobs,_,_=e.load(); j=next(x for x in jobs if x['arm']=='self_review')
        calls=[]
        def generate(messages,remaining):
            calls.append(deepcopy(messages))
            return '{"latest_run_status":"unknown","assertion_support":"not_established","evidence_ids":[],"limitations":"'+('uncertain '*25)+'"}',60,100,.1
        result=e.evaluate(j,generate)
        self.assertEqual(len(calls),2)
        self.assertTrue(result['answer']['valid'])
        self.assertGreater(result['protocol']['explanation_words'],20)
        self.assertNotIn('latest_run_step',str(calls))
        self.assertNotIn('human_reviewed',str(calls))


if __name__=='__main__': unittest.main()
