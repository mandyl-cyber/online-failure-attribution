import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from remote_pilots.openhands_smoke.shadow import follow


class ShadowTests(unittest.TestCase):
    def test_snapshot_retries_temporary_reader_lock(self):
        from remote_pilots.openhands_smoke.shadow import atomic
        with patch('remote_pilots.openhands_smoke.shadow._atomic',
                   side_effect=[PermissionError('reader lock'), None]) as write:
            atomic(Path('unused'), {'complete': True})
            self.assertEqual(write.call_count, 2)

    def test_pending_restart_duplicate_and_conflict(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            events = root / 'state/store/sessions/test/events'
            events.mkdir(parents=True)
            action = dict(id=0, source='agent', action='run', args={'command': 'cat file.py'})
            (events / '0.json').write_text(json.dumps(action))
            follow(root, once=True)
            first = json.loads((root / 'shadow/stages.json').read_text())
            self.assertEqual(first['observed_steps'], 0)
            self.assertEqual(first['pending_action_ids'], [0])
            (events / '1.json').write_text(json.dumps(dict(id=1, source='agent',
                observation='run', cause=0, content='hello', extras={'exit_code': 0})))
            follow(root, once=True)
            completed = json.loads((root / 'shadow/stages.json').read_text())
            self.assertEqual(completed['observed_steps'], 1)
            follow(root, once=True)
            self.assertEqual(completed, json.loads((root / 'shadow/stages.json').read_text()))
            action['args']['command'] = 'different'
            (events / '0.json').write_text(json.dumps(action))
            with self.assertRaises(ValueError):
                follow(root, once=True)


if __name__ == '__main__':
    unittest.main()
