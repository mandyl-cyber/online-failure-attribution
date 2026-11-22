from pathlib import Path
import tempfile
import unittest

from segmenter.online_ledger import OnlineLedger


class TerminalLifecycleTests(unittest.TestCase):
    def test_trailing_lifecycle_is_durable_without_reopening_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime=OnlineLedger(directory,'terminal-test',background=False,input_mode='replay')
            runtime.ingest(dict(id=0,source='user',action='message',args=dict(content='Check code.')))
            runtime.ingest(dict(id=1,source='agent',action='finish',args=dict(final_thought='Done')))
            jobs={p.name:p.read_bytes() for p in runtime.jobs.glob('*.json')}
            step_count=len(runtime.segmenter.steps)
            trailing=dict(id=2,source='environment',observation='agent_state_changed',cause=999,extras=dict(agent_state='finished'))
            self.assertTrue(runtime.ingest(trailing))
            self.assertFalse(runtime.ingest(trailing))
            self.assertTrue(runtime.finished)
            self.assertEqual(step_count,len(runtime.segmenter.steps))
            self.assertEqual(jobs,{p.name:p.read_bytes() for p in runtime.jobs.glob('*.json')})
            runtime.close()
            restored=OnlineLedger(directory,'terminal-test',background=False,input_mode='replay')
            self.assertEqual(len(list(restored.raw.glob('*.json'))),3)
            self.assertEqual(jobs,{p.name:p.read_bytes() for p in restored.jobs.glob('*.json')})
            restored.close()

    def test_new_work_and_tool_results_still_rejected_after_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime=OnlineLedger(directory,'terminal-test',background=False)
            runtime.ingest(dict(id=1,source='agent',action='finish',args={}))
            for event in [dict(id=2,source='agent',action='run',args=dict(command='echo work')),
                          dict(id=3,source='environment',observation='run',cause=1,content='late result')]:
                with self.assertRaisesRegex(ValueError,'already closed'):
                    runtime.ingest(event)
            runtime.close()


if __name__=='__main__':
    unittest.main()
