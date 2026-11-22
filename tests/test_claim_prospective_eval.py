from pathlib import Path
import tempfile
import unittest

from scripts.prepare_claim_prospective_eval import freeze_write
from remote_pilots.openhands_smoke.claim_prospective_eval import verify_inputs


class ProspectiveEvalTests(unittest.TestCase):
    def test_reference_cannot_silently_change_after_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'frozen.json'
            freeze_write(path,b'original')
            freeze_write(path,b'original')
            with self.assertRaisesRegex(ValueError,'Frozen input/reference changed'):
                freeze_write(path,b'revised')
            self.assertEqual(path.read_bytes(),b'original')

    def test_real_inputs_validate_without_model_calls(self):
        jobs,protocol=verify_inputs(Path('pilot/claim_prospective_eval_v1'))
        self.assertEqual(len(jobs),26)
        self.assertTrue(protocol['frozen_before_model_calls'])
        self.assertFalse(protocol['posthoc_verification_in_input'])
        self.assertTrue(all(row['job']['run_id']==row['session'] for row in jobs))


if __name__=='__main__':
    unittest.main()
