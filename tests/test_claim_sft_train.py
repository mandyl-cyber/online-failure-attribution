import unittest
from remote_pilots.openhands_smoke.claim_sft_train import encode_reference


class Tokenizer:
    def apply_chat_template(self,messages,tokenize,add_generation_prompt,return_dict=False):
        return [10,20] if add_generation_prompt else [10,20,30,40]


class CompletionMaskTests(unittest.TestCase):
    def test_only_answer_tokens_supervised(self):
        encoded=encode_reference(Tokenizer(),dict(prompt=[{}],completion=[{}]))
        self.assertEqual(encoded["labels"],[-100,-100,30,40])
        self.assertEqual(encoded["input_ids"],[10,20,30,40])

    def test_incompatible_template_rejected(self):
        class Incompatible(Tokenizer):
            def apply_chat_template(self,messages,tokenize,add_generation_prompt,return_dict=False):
                return [1,2] if add_generation_prompt else [1,3,4]
        with self.assertRaises(ValueError):
            encode_reference(Incompatible(),dict(prompt=[{}],completion=[{}]))

    def test_mapping_return_normalized_before_masking(self):
        from collections import UserDict
        class MappingTokenizer(Tokenizer):
            def apply_chat_template(self,messages,tokenize,add_generation_prompt,return_dict=False):
                ids=super().apply_chat_template(messages,tokenize,add_generation_prompt,return_dict)
                return UserDict(input_ids=[ids],attention_mask=[[1]*len(ids)])
        encoded=encode_reference(MappingTokenizer(),dict(prompt=[{}],completion=[{}]))
        self.assertEqual(encoded["labels"],[-100,-100,30,40])

    def test_generation_ids_are_flat(self):
        from remote_pilots.openhands_smoke.claim_sft_train import chat_ids
        class NestedTokenizer(Tokenizer):
            def apply_chat_template(self,*args,**kwargs):
                return [[10,20]]
        self.assertEqual(chat_ids(NestedTokenizer(),[{}],True),[10,20])


if __name__=="__main__":
    unittest.main()
