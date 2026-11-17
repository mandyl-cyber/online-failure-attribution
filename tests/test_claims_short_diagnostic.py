import json
from pathlib import Path
import tempfile
import unittest

from remote_pilots.openhands_smoke import claims_short_diagnostic as diagnostic
from remote_pilots.openhands_smoke import claims_v3 as pilot
from segmenter import stage_claim_summary as core


class DiagnosticTests(unittest.TestCase):
    def test_prepare_exact_complete_steps_and_unchanged_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'pack'; diagnostic.prepare(root)
            rows = pilot.rows_at(root / 'development.jsonl')
            parents = {r['job_id']: r for r in pilot.rows_at(pilot.ROOT / 'development.jsonl')}
            selection = pilot.read(root / 'selection.json')
            self.assertEqual(len(rows), 7)
            self.assertEqual(len({r['job_id'] for r in rows}), 7)
            for row, selected in zip(rows, selection):
                core.verify_row(row)
                sid = selected['source_step_id']
                expected = [e for e in parents[selected['parent_job_id']]['source_snapshot']['evidence']
                            if e['source_step_id'] == sid]
                self.assertEqual(row['source_snapshot']['evidence'], expected)
                self.assertEqual(row['request']['units'][0]['missing_fields'], [])
                self.assertEqual(len(row['request']['units']), 1)
                self.assertNotIn('expected', row['request'])
            self.assertEqual(pilot.read(root / 'protocol.json')['instruction_sha256'],
                             core.digest(core.INSTRUCTION.encode()))

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError): diagnostic.prepare(Path(tmp))

    def test_args_do_not_record_secrets(self):
        args = diagnostic.allowed_args(['python', 'server.py', '--model-path', '/models/qwen',
                                        '--api-key', 'SECRET', '--port=30001'])
        self.assertEqual(args, {'model-path': '/models/qwen', 'port': '30001'})
        self.assertNotIn('SECRET', json.dumps(args))

    def test_model_files_selected_config_and_no_weight_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config.json').write_text(json.dumps(dict(model_type='qwen2',
                hidden_size=3584, private_token='SECRET')))
            (root / 'model.safetensors').write_bytes(b'not-a-real-weight')
            report = diagnostic.model_files(root)
            fields = report['config_files']['config.json']['selected_fields']
            self.assertEqual(fields['hidden_size'], 3584)
            self.assertNotIn('SECRET', json.dumps(report))
            self.assertFalse(report['weights_hashed'])
            self.assertEqual(report['weight_files'][0]['bytes'], 17)


if __name__ == '__main__': unittest.main()
