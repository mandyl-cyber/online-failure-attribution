import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_claim_next_stage import authored_tasks,preflight_tasks
from remote_pilots.openhands_smoke.run import load_task_bundle,CODE,TESTS
from remote_pilots.openhands_smoke.claim_prospective import checked_tasks
from remote_pilots.openhands_smoke.claim_prospective import main as collect_main


class NextStageTests(unittest.TestCase):
    def test_legacy_smoke_keeps_original_task_files(self):
        bundle=load_task_bundle()
        self.assertEqual(bundle['code'],CODE)
        self.assertEqual(bundle['tests'],TESTS)
        self.assertEqual((bundle['code_name'],bundle['test_name']),('ranges.py','test_ranges.py'))

    def test_new_tasks_distinguish_real_bug_and_reference(self):
        reports=preflight_tasks(authored_tasks())
        self.assertEqual(len(reports),4)
        self.assertTrue(all(r['buggy']!=0 and r['reference']==0 for r in reports))

    def test_task_path_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'task.json'
            bundle=authored_tasks()[0]
            bundle['code_name']='../answer.py'
            path.write_text(json.dumps(bundle),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'filenames'):
                load_task_bundle(path)

    def test_frozen_task_hashes_pass(self):
        tasks=checked_tasks(Path('pilot/claim_prospective_tasks_v1'))
        self.assertEqual(len(tasks),4)
        for _,path in tasks:
            bundle=load_task_bundle(path)
            self.assertNotIn(bundle['reference_code'],bundle['task'])

    def test_unavailable_model_does_not_consume_a_frozen_task_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'not-started'
            with patch('sys.argv',['claim_prospective','run','--out',str(output)]), \
                 patch('sys.platform','linux'), \
                 patch('remote_pilots.openhands_smoke.claim_prospective.subprocess.run'), \
                 patch('remote_pilots.openhands_smoke.claim_prospective.urllib.request.urlopen',side_effect=OSError('server offline')):
                with self.assertRaisesRegex(OSError,'server offline'):
                    collect_main()
            self.assertFalse(output.exists())


if __name__=='__main__':
    unittest.main()
