import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import tempfile
import unittest
from copy import deepcopy
from segmenter.claim_extraction import build_request, validate_response, LocalRuleBackend, StageProcessor, ExtractionError
from segmenter.ledger import atomic

JOB = Path('eval/online_ledger_smoke_v1/SWE/stage_jobs/00000001.json')

class Tests(unittest.TestCase):
    def test_online_processor(self):
        from segmenter.online_ledger import OnlineLedger
        with tempfile.TemporaryDirectory() as d:
            runtime = OnlineLedger(d, 'test', claim_processor=StageProcessor())
            try:
                runtime.ingest(dict(id=0, source='user', action='message', args={'content': 'Check tests'}))
                runtime.ingest(dict(id=1, source='agent', action='run', args={'command': 'pytest'}))
                runtime.ingest(dict(id=2, observation='run', cause=1, content='1 passed', extras={'exit_code': 0}))
                runtime.ingest(dict(id=3, source='agent', action='finish', args={}))
                runtime.drain()
                result = json.loads(next(runtime.results.glob('*.json')).read_text(encoding='utf-8'))
                self.assertEqual(result['validation']['status'], 'passed')
                request = json.loads(next((Path(d)/'extraction').glob('*/request.json')).read_text(encoding='utf-8'))
                self.assertEqual(request['task_context']['text'], 'Check tests')
            finally: runtime.close()

    def test_validation(self):
        job = json.loads(JOB.read_text(encoding='utf-8'))
        request = build_request(job)
        response = LocalRuleBackend().extract(request)
        validate_response(response, request)
        validate_response({'claims': []}, request)
        with self.assertRaises(ExtractionError): build_request(job, 1)
        bad = deepcopy(job); bad['evidence'][0]['text'] += 'tampered'
        with self.assertRaises(ExtractionError): build_request(bad)
        if response['claims']:
            for change in ({'evidence_id': 'foreign'}, {'quote': 'fabricated'}, {'start': True}, {'end': 999999999}):
                bad = deepcopy(response); bad['claims'][0]['evidence_refs'][0].update(change)
                with self.assertRaises(ExtractionError): validate_response(bad, request)

    def test_retry_and_cache(self):
        class Backend(LocalRuleBackend):
            identity = 'test-retry'
            calls = 0
            def extract(self, request):
                self.calls += 1
                if self.calls == 1: raise RuntimeError('temporary')
                return super().extract(request)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'claim_batches'; out.mkdir()
            b = Backend(); p = StageProcessor(b)
            first = p(JOB, out)
            self.assertEqual(first['status'], 'ready')
            self.assertEqual(p(JOB, out), first)
            self.assertEqual(b.calls, 2)
            with self.assertRaises(ExtractionError): StageProcessor(b, max_attempts=3)(JOB, out)

    def test_failure_budget(self):
        class Backend(LocalRuleBackend):
            identity = 'test-failure'
            calls = 0
            def extract(self, request):
                self.calls += 1
                return {'invalid': 'output'}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'claim_batches'; out.mkdir()
            b = Backend()
            self.assertEqual(StageProcessor(b)(JOB, out)['status'], 'failed')
            self.assertEqual(StageProcessor(b)(JOB, out)['attempts'], 2)
            self.assertEqual(b.calls, 2)
            self.assertEqual(list(out.glob('*.json')), [])

if __name__ == '__main__': unittest.main()
