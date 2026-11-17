import json
import tempfile
import unittest
from pathlib import Path
from segmenter.input_timeline import InputTimeline
from segmenter.online_ledger import OnlineLedger
from segmenter.timeline_report import report


class TimelineTests(unittest.TestCase):
    def test_report_excludes_replay_and_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            live = InputTimeline(Path(d)/'live')
            live.emit('boundary_published', ready_to_publish_ms=2, recovery=False)
            live.emit('boundary_published', ready_to_publish_ms=999, recovery=True)
            InputTimeline(Path(d)/'replay', 'replay').emit('boundary_published', ready_to_publish_ms=999)
            summary=report(d)
            self.assertEqual(summary['metrics']['ready_to_publish_ms'], dict(n=1,p50=2,p95=2,p99=2))

    def test_first_receipt_survives_restart_no_cross_process_duration(self):
        with tempfile.TemporaryDirectory() as d:
            a = InputTimeline(d)
            ev = dict(id=1, timestamp='upstream-unsynchronised')
            receipt = a.receive(ev)
            self.assertGreaterEqual(a.elapsed(receipt, a.stamp()), 0)
            b = InputTimeline(d)
            self.assertEqual(b.receive(ev, recovery=True), receipt)
            self.assertIsNone(b.elapsed(receipt, b.stamp()))
            replay = InputTimeline(Path(d)/'replay', 'replay')
            self.assertIsNone(replay.elapsed(replay.receive(ev), replay.stamp()))

    def test_ledger_restore_and_reused_claims_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            a = OnlineLedger(d, 'test', background=False)
            a.ingest(dict(id=1, action='run', source='agent', args={'command':'pytest'}))
            with self.assertRaises(ValueError):
                a.ingest(dict(id=2, observation='run', content='orphan'))
            a.close()
            b = OnlineLedger(d, 'test', background=False)
            b.ingest(dict(id=2, observation='run', cause=1, content='1 passed', extras={'exit_code':0}))
            b.ingest(dict(id=3, action='finish', source='agent', args={}))
            b.drain(); b.drain(); b.close()
            rows = [json.loads(s) for s in (Path(d)/'timing/timeline.jsonl').read_text().splitlines()]
            ready = next(r for r in rows if r['kind']=='step_ready' and r['step_id']==1)
            self.assertIsNone(ready['action_to_ready_ms'])
            self.assertGreaterEqual(ready['observation_to_ready_ms'], 0)
            self.assertTrue(any(r['kind']=='claim_reused' for r in rows))
            self.assertTrue(any(r['kind']=='boundary_published' for r in rows))


if __name__ == '__main__': unittest.main()
