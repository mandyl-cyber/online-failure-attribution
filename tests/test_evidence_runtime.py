from pathlib import Path
import tempfile
import unittest
from segmenter.evidence_runtime import RuntimeEvidence


class RuntimeTests(unittest.TestCase):
    def setup_runtime(self,base):
        root=Path(base)/'work';root.mkdir();path=root/'task.py';path.write_text('x=1')
        runtime=RuntimeEvidence('test',root,Path(base)/'out')
        return runtime,path

    def emit_test(self,runtime,path):
        runtime.bind(1,path)
        runtime.feed(dict(id=1,source='agent',action='run',args=dict(command='python -m unittest test_task')))
        return runtime.feed(dict(id=2,source='environment',observation='run',cause=1,
            content='Ran 1 test in 0.1s\n\nOK',extras=dict(exit_code=0)))[0]['fact']

    def emit_statement(self,runtime,path,text='All tests passed.'):
        runtime.bind(3,path)
        runtime.feed(dict(id=3,source='agent',action='message',args=dict(content=text)))
        return runtime.statement(3,text,path)

    def test_publication_gate_and_actual_file(self):
        with tempfile.TemporaryDirectory() as base:
            runtime,path=self.setup_runtime(base);f=self.emit_test(runtime,path)
            runtime.seal('stage',1);item=self.emit_statement(runtime,path)
            self.assertEqual(runtime.decide(item)['decision']['state'],'pending_publication')
            runtime.publish('stage',[f['fact_id']])
            result=runtime.decide(item)
            self.assertEqual(result['decision']['assertion_support'],'supported')
            self.assertEqual(len(list((runtime.out/'batches').glob('*.json'))),1)
            publication=runtime.publications['stage']
            self.assertGreaterEqual(publication['monotonic_ns'],publication['sealed_at_monotonic_ns'])

    def test_unknown_prose_and_source_mismatch(self):
        with tempfile.TemporaryDirectory() as base:
            runtime,path=self.setup_runtime(base);self.emit_test(runtime,path)
            item=self.emit_statement(runtime,path,'Possibly fixed.')
            self.assertEqual(runtime.decide(item)['decision']['state'],'pending_decomposition')
            with self.assertRaises(ValueError):runtime.statement(3,'All tests passed.',path)
            with self.assertRaises(ValueError):runtime.bind(1,path)

    def test_changed_file_during_command_loses_version(self):
        with tempfile.TemporaryDirectory() as base:
            runtime,path=self.setup_runtime(base);runtime.bind(1,path)
            runtime.feed(dict(id=1,source='agent',action='run',args=dict(command='python -m unittest test_task')))
            path.write_text('x=2')
            f=runtime.feed(dict(id=2,source='environment',observation='run',cause=1,
                content='Ran 1 test in 0.1s\n\nOK',extras=dict(exit_code=0)))[0]['fact']
            self.assertIsNone(f['version']);self.assertIsNone(runtime.shadow.state.facts[0]['version'])


if __name__=='__main__':unittest.main()
