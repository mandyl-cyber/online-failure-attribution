import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_claim_binding_sft import verify_package
from remote_pilots.openhands_smoke.claim_binding_compare import structural_check
from remote_pilots.openhands_smoke.claim_sft_train import require_full_budget,main as train_main


class BindingSFTTests(unittest.TestCase):
    def test_data_roles_and_task_groups_verified(self):
        rows,protocol=verify_package(Path('pilot/claim_binding_sft_v1'))
        self.assertEqual((len(rows['train']),len(rows['dev'])),(126,38))
        self.assertFalse(protocol['independent_acceptance'])
        self.assertEqual(sum(r.get('origin')=='retired_prospective_diagnostic' for r in rows['train']),26)

    def test_canonical_alias_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for filename in ['train.jsonl','dev.jsonl','regression.jsonl','protocol.json']:
                shutil.copyfile(Path('pilot/claim_binding_sft_v1')/filename,root/filename)
            rows,protocol=verify_package(root)
            protocol['canonical_group_aliases'][rows['train'][0]['group']]=protocol['canonical_group_aliases'][rows['dev'][0]['group']]
            (root/'protocol.json').write_text(json.dumps(protocol),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Canonical evaluation task overlaps'):
                verify_package(root)

    def test_oversize_is_not_silently_filtered(self):
        with self.assertRaisesRegex(ValueError,'no silent filtering'):
            require_full_budget([dict(job_id='too-long',eligible=False)])
        require_full_budget([dict(job_id='fits',eligible=True)])

    def test_development_only_pack_blocked_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'protocol.json').write_text(json.dumps(dict(training_eligible=False)),encoding='utf-8')
            with patch('sys.argv',['train','train','--data',str(root),'--model-path','unused']):
                with self.assertRaisesRegex(ValueError,'development-only'):
                    train_main()

    def test_structure_check_never_certifies_semantics(self):
        row=json.loads(Path('pilot/claim_binding_sft_v1/dev.jsonl').read_text(encoding='utf-8').splitlines()[0])
        result=structural_check(row['completion'][0]['content'],row['prompt'])
        self.assertTrue(result['valid'])
        self.assertEqual(result['semantic_correctness'],'not_scored')
        self.assertFalse(structural_check('{"claims":[{"text":"x","source":"tool_observation","evidence_ids":["unknown"]}]}',row['prompt'])['valid'])


if __name__=='__main__':unittest.main()
