import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from segmenter.input_gate import stage_check,dispatch
from segmenter.online_ledger import OnlineLedger


class GateTests(unittest.TestCase):
    def test_plan_notes_reach_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            runtime=OnlineLedger(d,'plan',background=False)
            runtime.ingest(dict(id=1,action='task_tracking',source='agent',args={'command':'plan','task_list':[{'title':'test','notes':'test blocked','status':'todo'}]}))
            runtime.ingest(dict(id=2,cause=1,observation='task_tracking',content='saved'))
            self.assertIn('test blocked',runtime.ledger.steps[1]['step']['action'])
            runtime.close()
    def test_missing_and_truncated(self):
        stage=dict(start_step_id=1,end_step_id=2)
        check=stage_check(stage,{1:dict(issues=['upstream_truncation_declared'])})
        self.assertEqual(check['status'],'blocked')
        self.assertEqual(len(check['reasons']),2)

    def test_block_before_backend_and_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);results=root/'results';results.mkdir();job=root/'job.json'
            job.write_text(json.dumps(dict(job_id='j',incomplete=True)),encoding='utf-8')
            processor=Mock(side_effect=AssertionError('must not call'))
            a=dispatch(processor,job,results);b=dispatch(processor,job,results)
            self.assertEqual(a,b);self.assertEqual(a['status'],'blocked_input');processor.assert_not_called()

    def test_live_stage_gate_and_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            runtime=OnlineLedger(d,'run',background=False)
            runtime.ingest(dict(id=1,source='agent',action='run',args={'command':'pytest'}))
            runtime.ingest(dict(id=2,cause=1,observation='run',content='Output truncated',extras={'truncated':True}))
            runtime.ingest(dict(id=3,source='agent',action='finish',args={'final_thought':'done'}))
            runtime.drain();jobs=[json.loads(p.read_text()) for p in runtime.jobs.glob('*.json')]
            self.assertTrue(any(j['input_integrity']['status']=='blocked' for j in jobs))
            batches=[json.loads(p.read_text()) for p in runtime.results.glob('*.json')]
            self.assertTrue(any(b['status']=='blocked_input' for b in batches))
            before={p.name:p.read_bytes() for p in runtime.results.glob('*.json')};runtime.close()
            recovered=OnlineLedger(d,'run',background=False);recovered.drain()
            self.assertEqual(before,{p.name:p.read_bytes() for p in recovered.results.glob('*.json')})
            recovered.close()


if __name__=='__main__':unittest.main()
