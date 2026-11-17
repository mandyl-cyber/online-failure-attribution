from pathlib import Path
from collections import UserDict
import tempfile
import unittest
from unittest.mock import patch

from remote_pilots.openhands_smoke import claims_qwen32 as wrapper
from remote_pilots.openhands_smoke import claims_v3 as pilot


class Tokenizer:
    def __init__(self, length): self.length=length; self.messages=[]
    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        return [1]*self.length


class ComparisonTests(unittest.TestCase):
    def test_preflight_includes_output_budget(self):
        tok=Tokenizer(6144)
        checks=wrapper.token_preflight(tok)
        self.assertEqual(len(checks),13)
        self.assertTrue(all(c['fits'] for c in checks))
        self.assertTrue(all(c['output_budget']==2048 for c in checks))
        self.assertTrue(all(m[0]['content']==wrapper.core.INSTRUCTION for m in tok.messages))
        self.assertFalse(any(c['fits'] for c in wrapper.token_preflight(Tokenizer(6145))))

    def test_mapping_counts_tokens_not_mapping_keys(self):
        tok=Tokenizer(1)
        tok.apply_chat_template=lambda *args,**kwargs: UserDict(input_ids=[[1]*6145], attention_mask=[[1]*6145])
        checks=wrapper.token_preflight(tok)
        self.assertTrue(all(c['input_tokens']==6145 and not c['fits'] for c in checks))

    def test_tensor_and_single_batch(self):
        class Tensor:
            def tolist(self): return [[4,5,6]]
        self.assertEqual(wrapper.token_ids({'input_ids':Tensor()}),[4,5,6])
        self.assertEqual(wrapper.token_ids([4,5,6]),[4,5,6])

    def test_refuse_malformed_or_multiple_sequences(self):
        for value in ({'attention_mask':[1]}, [[1],[2]], [], [True], ['1'], [-1], 'text'):
            with self.subTest(value=value), self.assertRaises(ValueError): wrapper.token_ids(value)

    def test_weight_mismatch_not_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'fake').write_bytes(b'bad')
            with patch.object(wrapper,'WEIGHT_SHA256',{'fake':'wrong'}):
                with self.assertRaises(ValueError): wrapper.verify_weights(root)

    def test_comparison_preserves_failure_and_reviews(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); (out/'stages').mkdir()
            pilot.save(out/'stages'/'a.json',dict(status='failed',seconds=2,error='bad JSON'))
            baseline=[dict(arm='stages',job_id='a',result=dict(status='ready',seconds=1,claims=[{}])),
                      dict(arm='short',job_id='b',result=dict(status='ready',seconds=1,claims=[{}]))]
            pairs=wrapper.compare(out,baseline)
            self.assertEqual(pairs[0]['candidate_status'],'failed')
            self.assertEqual(pairs[1]['candidate_status'],'missing')
            self.assertIsNone(pairs[0]['citation_support'])
            (out/'paired_review.json').write_text('human work',encoding='utf-8')
            wrapper.compare(out,baseline)
            self.assertEqual((out/'paired_review.json').read_text(),'human work')


if __name__=='__main__': unittest.main()
