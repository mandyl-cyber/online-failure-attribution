import unittest
from remote_pilots.openhands_smoke.claim_sft_train import encode_reference


class Tokenizer:
    def apply_chat_template(self,messages,tokenize,add_generation_prompt):
        return [10,20] if add_generation_prompt else [10,20,30,40]


class CompletionMaskTests(unittest.TestCase):
    def test_only_answer_tokens_supervised(self):
        encoded=encode_reference(Tokenizer(),dict(prompt=[{}],completion=[{}]))
        self.assertEqual(encoded["labels"],[-100,-100,30,40])
        self.assertEqual(encoded["input_ids"],[10,20,30,40])

    def test_incompatible_template_rejected(self):
        class Incompatible(Tokenizer):
            def apply_chat_template(self,messages,tokenize,add_generation_prompt):
                return [1,2] if add_generation_prompt else [1,3,4]
        with self.assertRaises(ValueError):
            encode_reference(Incompatible(),dict(prompt=[{}],completion=[{}]))


if __name__=="__main__":
    unittest.main()
