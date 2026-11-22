import json
from pathlib import Path
import tempfile
import unittest

from remote_pilots.openhands_smoke.neural_ledger_acceptance import successful_timings,distribution


class TimingTests(unittest.TestCase):
    def test_failed_completions_and_duplicate_jobs_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'timing').mkdir()
            events=[]
            for job,start,end in [('ready.json',100,400),('failed.json',450,460),('ready.json',500,501)]:
                events.extend([dict(kind='claim_started',job=job,process_id='p',monotonic_ns=start*1000000),
                               dict(kind='claim_completed',job=job,process_id='p',monotonic_ns=end*1000000)])
            (path/'timing/timeline.jsonl').write_text('\n'.join(json.dumps(e) for e in events),encoding='utf8')
            (path/'timing/queue-ready.json').write_text(json.dumps(dict(process_id='p',monotonic_ns=50000000)),encoding='utf8')
            result=successful_timings(path,{'ready.json'})
            self.assertEqual(result['processing_ms']['n'],1)
            self.assertEqual(result['processing_ms']['p50'],300)
            self.assertEqual(result['queued_to_publication_ms']['p50'],350)

    def test_quantiles_include_small_sample_tail(self):
        result=distribution([1,2,20])
        self.assertEqual(result['p50'],2)
        self.assertEqual(result['p95'],20)
        self.assertEqual(result['p99'],20)

    def test_empty_distribution_is_explicit(self):
        result=distribution([])
        self.assertEqual(result['n'],0)
        self.assertIsNone(result['p95'])


if __name__=='__main__':
    unittest.main()
