import json
from pathlib import Path
import tempfile
import unittest

from segmenter.claim_extraction import StageProcessor
from segmenter.online_ledger import OnlineLedger
from segmenter.reviewed_claim_processor import ReviewedClaimProcessor


class WrongCountBackend:
    identity='wrong-count-regression-fixture'

    def extract(self,request):
        observation=next(e for e in request['evidence'] if e['field']=='observation')
        return dict(claims=[dict(local_id='c1',kind='verification',text='The runner reports 109 tests.',
            basis='tool_observation',objects=[],conditions=[],unknowns=[],
            evidence_refs=[dict(evidence_id=observation['record_id'],start=0,end=len(observation['text']),quote=observation['text'])])])


def events(text):
    return [dict(id=1,source='agent',action='run',args=dict(command='python tests/runtests.py')),
            dict(id=2,source='environment',observation='run',cause=1,content=text,extras=dict(exit_code=0)),
            dict(id=3,source='agent',action='finish',args=dict(final_thought='Done'))]


class ReviewedProcessorTests(unittest.TestCase):
    def test_online_seal_marks_claim_and_resume_preserves_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            processor=ReviewedClaimProcessor(StageProcessor(WrongCountBackend()))
            runtime=OnlineLedger(directory,'r',background=False,claim_processor=processor)
            stream=events('Ran 7 tests in 0.1s\nOK\n[Command finished with exit code 0]')
            for event in stream[:2]:
                runtime.ingest(event)
            self.assertFalse(list(runtime.results.glob('*.json')))
            runtime.ingest(stream[2])
            runtime.drain()
            path=next(runtime.results.glob('*.json'))
            result=json.loads(path.read_text(encoding='utf8'))
            self.assertEqual(result['claims'][0]['semantic_status'],'needs_review')
            self.assertEqual(result['claims'][0]['text'],'The runner reports 109 tests.')
            before=path.read_bytes()
            runtime.close()
            recovered=OnlineLedger(directory,'r',background=False,claim_processor=processor)
            recovered.drain()
            recovered.close()
            self.assertEqual(path.read_bytes(),before)

    def test_rule_batch_publishes_exact_failed_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime=OnlineLedger(directory,'r',background=False,claim_processor=ReviewedClaimProcessor())
            for event in events('ERROR:\nNo replacement was performed, old_str did not appear verbatim.'):
                runtime.ingest(event)
            runtime.drain()
            result=json.loads(next(runtime.results.glob('*.json')).read_text(encoding='utf8'))
            signal=next(s for s in result['observation_signals'] if s['kind']=='replacement_not_performed')
            self.assertIn('No replacement',signal['quote'])
            self.assertEqual(signal['available_after_seq'],result['detected_at_seq'])
            self.assertEqual(signal['stage_id'],result['stage']['stage_id'])
            runtime.close()

    def test_restore_rejects_processor_configuration_change(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime=OnlineLedger(directory,'r',background=False,claim_processor=ReviewedClaimProcessor())
            runtime.close()
            with self.assertRaisesRegex(ValueError,'configuration|mismatch'):
                OnlineLedger(directory,'r',background=False)


if __name__=='__main__':
    unittest.main()
