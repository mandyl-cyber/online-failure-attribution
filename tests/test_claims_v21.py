import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from remote_pilots.openhands_smoke import claims_v21 as v


class RevisionTests(unittest.TestCase):
    def test_selection_task_disjoint_and_deterministic(self):
        rows, jobs = [], {}
        for task in ('a', 'b'):
            for arm in ('gold', 'structure', 'rule'):
                for step in (1, 2, 3):
                    row = dict(job_id=f'{task}-{arm}-{step}', run_id=task,
                               request={'stage': {'start_step_id': step}})
                    rows.append(row); jobs[row['job_id']] = {'arm': arm}
        selected = v.select_holdout(rows, [rows[0]], jobs)
        self.assertEqual(len(selected), 6)
        self.assertEqual({r['run_id'] for r in selected}, {'b'})
        self.assertEqual({r['request']['stage']['start_step_id'] for r in selected}, {1, 3})
        self.assertEqual(selected, v.select_holdout(list(reversed(rows)), [rows[0]], jobs))

    def test_instruction_restored_after_failure(self):
        old = v.v2.INSTRUCTION
        with patch.object(v.v2, 'run', side_effect=RuntimeError('offline')):
            with self.assertRaises(RuntimeError):
                v.run_one(Path('x'), Path('y'), 'http://example', 'qwen-local', v.INSTRUCTION)
        self.assertEqual(v.v2.INSTRUCTION, old)

    def test_paired_inputs_and_prompt_drift_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'inputs'; out = Path(tmp) / 'results'
            v.prepare(root)
            dev = v.read_rows(root / 'development.jsonl')
            held = v.read_rows(root / 'holdout.jsonl')
            self.assertFalse({r['run_id'] for r in dev} & {r['run_id'] for r in held})
            seen = []
            def fake_run(inputs, dest, base_url, model, instruction):
                seen.append((v.read_rows(inputs), instruction))
                dest.mkdir(parents=True)
                v.v2.save(dest / 'summary.json', {'mock': True})
            with patch.object(v, 'run_one', side_effect=fake_run):
                v.run(root, out, 'http://example', 'qwen-local')
            self.assertEqual(len(seen), 3)
            self.assertEqual(seen[1][0], seen[2][0])
            self.assertEqual(seen[1][1], v.v2.INSTRUCTION)
            self.assertEqual(seen[2][1], v.INSTRUCTION)
            protocol = json.loads((root / 'protocol.json').read_text())
            protocol['revised_instruction_sha256'] = 'changed'
            v.v2.save(root / 'protocol.json', protocol)
            with patch.object(v, 'run_one') as call:
                with self.assertRaises(ValueError): v.run(root, out, 'http://example', 'qwen-local')
                call.assert_not_called()


if __name__ == '__main__': unittest.main()
