import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from remote_pilots.openhands_smoke import claims_v2 as v2


def source(text='test failed\n'):
    return dict(job_id='abc', run_id='task', stage={'start_step_id': 1, 'end_step_id': 1},
                detected_at_seq=1, evidence=[dict(record_id='r', source_step_id=1,
                availability_seq=1, field='observation', text=text,
                content_sha256=v2.digest(text.encode()))])


def response(eid='e0000'):
    return {'claims': [dict(text='The test failed.', source='tool_observation',
                           objects=[], evidence_ids=[eid])]}


class ContractTests(unittest.TestCase):
    def test_unicode_chunk_coverage_and_exact_spans(self):
        text = ('中文🙂\r\n' * 700) + 'last'
        row = v2.build(source(text))
        refs = list(row['evidence_map'].values())
        self.assertEqual(''.join(r['quote'] for r in refs), text)
        for a, b in zip(refs, refs[1:]):
            self.assertEqual(a['end'], b['start'])
        for ref in refs:
            self.assertEqual(text[ref['start']:ref['end']], ref['quote'])
            self.assertLessEqual(len(ref['quote']), 1200)

    def test_reject_future_and_corruption(self):
        for field, value in [('availability_seq', 2), ('content_sha256', 'bad')]:
            old = source(); old['evidence'][0][field] = value
            with self.assertRaises(ValueError): v2.build(old)

    def test_validate_and_reject_foreign_duplicate_wrong_basis(self):
        row = v2.build(source())
        claims = v2.validate(response(), row)
        self.assertEqual(claims[0]['source_steps'], [1])
        self.assertEqual(claims[0]['evidence_refs'][0]['quote'], 'test failed\n')
        with self.assertRaises(ValueError): v2.validate(response('foreign'), row)
        bad = response(); bad['claims'][0]['evidence_ids'] *= 2
        with self.assertRaises(ValueError): v2.validate(bad, row)
        row['evidence_map']['e0000']['field'] = 'action'
        with self.assertRaises(ValueError): v2.validate(response(), row)
        self.assertEqual(v2.validate({'claims': []}, row), [])

    def test_schema_rejects_invented_fields_and_excess_claims(self):
        row = v2.build(source())
        bad = response(); bad['claims'][0]['cause'] = 'invented'
        with self.assertRaises(ValueError): v2.validate(bad, row)
        bad = response(); bad['claims'] *= 7
        with self.assertRaises(ValueError): v2.validate(bad, row)

    def test_runner_resume_preserves_raw_and_review(self):
        from io import BytesIO
        reply = {'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps(response())}}], 'usage': {'total_tokens': 10}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); inputs = root / 'in.jsonl'; out = root / 'out'
            inputs.write_bytes(v2.encoded(v2.build(source())) + b'\n')
            with patch.object(v2.urllib.request, 'urlopen', side_effect=[
                    BytesIO(b'{"data":[{"id":"qwen-local"}]}'), BytesIO(json.dumps(reply).encode())]) as mock:
                v2.run(inputs, out, 'http://example/v1', 'qwen-local')
                self.assertEqual(mock.call_count, 2)
            saved = json.loads((out / 'abc.json').read_text())
            self.assertEqual(saved['raw_response'], reply)
            self.assertEqual(saved['status'], 'valid')
            (out / 'semantic_review.json').write_text('user annotation', encoding='utf-8')
            with patch.object(v2.urllib.request, 'urlopen', return_value=BytesIO(
                    b'{"data":[{"id":"qwen-local"}]}')) as mock:
                v2.run(inputs, out, 'http://example/v1', 'qwen-local')
                self.assertEqual(mock.call_count, 1)
            self.assertEqual((out / 'semantic_review.json').read_text(), 'user annotation')

    def test_truncated_reply_is_saved_but_not_accepted(self):
        from io import BytesIO
        reply = {'choices': [{'finish_reason': 'length', 'message': {'content': '{'}}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); inputs = root / 'in.jsonl'; out = root / 'out'
            inputs.write_bytes(v2.encoded(v2.build(source())) + b'\n')
            with patch.object(v2.urllib.request, 'urlopen', side_effect=[
                    BytesIO(b'{"data":[{"id":"qwen-local"}]}'), BytesIO(json.dumps(reply).encode())]):
                v2.run(inputs, out, 'http://example/v1', 'qwen-local')
            saved = json.loads((out / 'abc.json').read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual(saved['raw_response'], reply)


if __name__ == '__main__': unittest.main()
