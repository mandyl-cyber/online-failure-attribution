import tempfile
from pathlib import Path
import unittest

from segmenter.claim_extraction import StageProcessor
from segmenter.claim_sft_backend import ClaimSFTBackend
from segmenter.reviewed_claim_processor import ReviewedClaimProcessor
from segmenter.ledger import atomic,sha


class FailingBackend:
    identity='deterministic-failure-test'
    retryable_error=staticmethod(ClaimSFTBackend.retryable_error)

    def __init__(self):
        self.calls=0

    def extract(self,request):
        self.calls+=1
        raise ValueError('token_budget_exceeded')


def job():
    text='ERROR:\nNo replacement was performed, old_str was not found.'
    records=[dict(record_id=field,run_id='r',source_step_id=1,field=field,text=content,
                  content_sha256=sha(content.encode()),availability_seq=1)
             for field,content in [('action','str_replace a.py'),('observation',text)]]
    return dict(job_id='j',run_id='r',stage=dict(stage_id=1,start_step_id=1,end_step_id=1),
                detected_at_seq=1,incomplete=False,candidates=[],task_context=None,evidence=records)


class FailureTests(unittest.TestCase):
    def test_partial_claims_and_rejections_survive_parent_publication(self):
        class PartialBackend:
            identity='partial-validation-test'
            extraction_details=dict(complete=False,rejected_claims=[dict(error='bad source')],oversized_steps=[])

            def extract(self,request):
                observation=next(e for e in request['evidence'] if e['field']=='observation')
                return dict(claims=[dict(local_id='c1',kind='open_issue',text='No replacement was performed.',
                    basis='tool_observation',objects=[],conditions=[],unknowns=[],
                    evidence_refs=[dict(evidence_id=observation['record_id'],start=0,end=len(observation['text']),quote=observation['text'])])])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            atomic(root/'job.json',job())
            result=ReviewedClaimProcessor(StageProcessor(PartialBackend()))(root/'job.json',root/'claims')
            self.assertEqual(result['status'],'partial')
            self.assertEqual(len(result['claims']),1)
            self.assertEqual(len(result['retained_observations']),1)
            self.assertEqual(result['extraction_details']['rejected_claims'][0]['error'],'bad source')

    def test_terminal_failure_does_not_retry_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            atomic(root/'job.json',job())
            backend=FailingBackend()
            processor=StageProcessor(backend)
            first=processor(root/'job.json',root/'claims')
            self.assertEqual(first['attempts'],1)
            self.assertFalse(first['retryable'])
            StageProcessor(backend)(root/'job.json',root/'claims')
            self.assertEqual(backend.calls,1)

    def test_failed_summary_publishes_raw_observation_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=job()
            atomic(root/'job.json',source)
            backend=FailingBackend()
            processor=ReviewedClaimProcessor(StageProcessor(backend))
            batch=processor(root/'job.json',root/'claims')
            self.assertEqual(batch['status'],'evidence_only')
            self.assertEqual(batch['claims'],[])
            self.assertEqual(batch['retained_observations'],[source['evidence'][1]])
            self.assertEqual(batch['observation_signals'][0]['kind'],'replacement_not_performed')
            before=(root/'claims/job.json').read_bytes()
            processor(root/'job.json',root/'claims')
            self.assertEqual((root/'claims/job.json').read_bytes(),before)
            self.assertEqual(backend.calls,1)


if __name__=='__main__':
    unittest.main()
