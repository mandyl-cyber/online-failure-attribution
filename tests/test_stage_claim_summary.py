from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from segmenter import stage_claim_summary as core
from remote_pilots.openhands_smoke import claims_v3 as pilot


def source():
    records = []
    for field, text in [('action', 'pytest test_x.py'), ('observation', '1 failed\n')]:
        records.append(dict(record_id='run:1:' + field, source_step_id=1, field=field,
                            text=text, content_sha256=core.digest(text.encode()), availability_seq=1))
    return dict(job_id='old', run_id='run', stage=dict(stage_id=1, start_step_id=1, end_step_id=1),
                detected_at_seq=1, task_context=None, evidence=records)


def response():
    return dict(claims=[dict(text='The test run reports one failure.', source='tool_observation',
                            evidence_ids=['S1.action.1.1', 'S1.observation.1.1'])])


class SummaryTests(unittest.TestCase):
    def test_unicode_exact_coverage_and_parent_pairing(self):
        old = source(); text = '中文🙂\r\n' * 600
        old['evidence'][1].update(text=text, content_sha256=core.digest(text.encode()))
        row = core.build(old)
        refs = [v for v in row['evidence_map'].values() if v['field'] == 'observation']
        self.assertEqual(''.join(r['quote'] for r in refs), text)
        for ref in refs:
            self.assertEqual(ref['quote'], text[ref['start']:ref['end']])
            self.assertLessEqual(len(ref['quote']), 1200)
        self.assertEqual(len(row['request']['units']), 1)
        core.verify_row(row)

    def test_no_future_outside_or_corrupt_records(self):
        for key, value in [('availability_seq', 2), ('source_step_id', 2), ('content_sha256', 'bad')]:
            old = source(); old['evidence'][0][key] = value
            with self.assertRaises(ValueError): core.build(old)
        old = source(); old['detected_at_seq'] = 4
        old['evidence'][0]['source_step_id'] = 2
        with self.assertRaises(ValueError): core.build(old)  # delayed seal is not stage membership

    def test_missing_and_ambiguous_pairing(self):
        old = source(); old['evidence'] = old['evidence'][:1]
        old['stage']['end_step_id'] = old['detected_at_seq'] = 2
        row = core.build(old)
        self.assertEqual(row['request']['units'][0]['missing_fields'], ['observation'])
        self.assertEqual(row['request']['units'][1]['missing_fields'], ['action', 'observation'])
        old = source(); extra = deepcopy(old['evidence'][0]); extra['record_id'] = 'other'
        old['evidence'].append(extra)
        with self.assertRaises(ValueError): core.build(old)

    def test_mapping_tampering_rejected(self):
        row = core.build(source()); row['evidence_map']['S1.action.1.1']['quote'] = 'invented'
        with self.assertRaises(ValueError): core.validate(response(), row)

    def test_contract_and_no_six_claim_cap(self):
        row = core.build(source()); good = response(); good['claims'] *= 7
        self.assertEqual(len(core.validate(good, row)), 7)
        self.assertEqual(core.validate({'claims': []}, row), [])
        for mutate in [lambda c: c.update(objects=[]),
                       lambda c: c.update(evidence_ids=['foreign']),
                       lambda c: c.update(evidence_ids=['S1.action.1.1']),
                       lambda c: c.update(text='   '),
                       lambda c: c.update(evidence_ids=['S1.observation.1.1'] * 2)]:
            bad = response(); mutate(bad['claims'][0])
            with self.assertRaises(ValueError): core.validate(bad, row)
        # A tool may display an agent plan: field is not epistemic source.
        good = response(); good['claims'][0]['source'] = 'agent_assertion'
        self.assertEqual(len(core.validate(good, row)), 1)

    def test_duplicate_json_keys_and_nonfinite_rejected(self):
        for text in ['{"claims":[],"claims":[]}', '{"claims":NaN}']:
            with self.assertRaises(ValueError): core.strict_json(text)

    def prepared(self, root):
        rows = [core.build(source())]
        root.mkdir()
        (root / 'development.jsonl').write_bytes(core.encoded(rows[0]) + b'\n')
        pilot.save(root / 'reference_points.json', [])
        pilot.save(root / 'protocol.json', dict(input_sha256=core.digest(core.encoded(rows)),
            instruction_sha256=core.digest(core.INSTRUCTION.encode()),
            schema_sha256=core.digest(core.encoded(core.SCHEMA)),
            references_sha256=core.digest(core.encoded([])),
            limits=dict(max_tokens=2048, max_request_bytes=100000, max_response_bytes=1000000)))
        return rows[0]

    def test_resume_no_duplicate_calls_and_preserve_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'inputs'; out = Path(tmp) / 'out'; row = self.prepared(root)
            reply = dict(choices=[dict(finish_reason='stop', message=dict(content=json.dumps(response())))])
            with patch.object(pilot.urllib.request, 'urlopen', return_value=BytesIO(core.encoded(reply))) as call:
                self.assertEqual(pilot.run(root, out, 'http://local/v1', 'qwen-local'), 0)
                self.assertEqual(call.call_count, 1)
            result = pilot.read(out / (row['job_id'] + '.json'))
            self.assertEqual(result['status'], 'ready')
            self.assertIsNone(result['available_at_seq'])
            (out / 'semantic_review.json').write_text('review in progress')
            with patch.object(pilot.urllib.request, 'urlopen') as call:
                pilot.run(root, out, 'http://local/v1', 'qwen-local')
                call.assert_not_called()
            self.assertEqual((out / 'semantic_review.json').read_text(), 'review in progress')

    def test_length_and_invalid_json_saved_not_repaired(self):
        for reason, content in [('length', json.dumps(response())), ('stop', '{bad')]:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'inputs'; out = Path(tmp) / 'out'; row = self.prepared(root)
                reply = dict(choices=[dict(finish_reason=reason, message=dict(content=content))])
                with patch.object(pilot.urllib.request, 'urlopen', return_value=BytesIO(core.encoded(reply))):
                    self.assertEqual(pilot.run(root, out, 'http://local/v1', 'qwen-local'), 1)
                result = pilot.read(out / (row['job_id'] + '.json'))
                self.assertEqual(result['raw_response'], reply)
                self.assertNotIn('claims', result)

    def test_budget_blocks_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'inputs'; out = Path(tmp) / 'out'; row = self.prepared(root)
            protocol = pilot.read(root / 'protocol.json'); protocol['limits']['max_request_bytes'] = 1
            pilot.save(root / 'protocol.json', protocol)
            with patch.object(pilot.urllib.request, 'urlopen') as call:
                self.assertEqual(pilot.run(root, out, 'http://local/v1', 'qwen-local'), 1)
                call.assert_not_called()
            self.assertEqual(pilot.read(out / (row['job_id'] + '.json'))['status'], 'blocked_input_budget')

    def test_reserved_is_not_resent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'inputs'; out = Path(tmp) / 'out'; row = self.prepared(root)
            out.mkdir(); pilot.save(out / (row['job_id'] + '.json'), dict(status='reserved'))
            with patch.object(pilot.urllib.request, 'urlopen') as call:
                self.assertEqual(pilot.run(root, out, 'http://local/v1', 'qwen-local'), 1)
                call.assert_not_called()
            self.assertEqual(pilot.read(out / 'summary.json')['counts']['reserved'], 1)

    def test_empty_success_is_distinct_from_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'inputs'; out = Path(tmp) / 'out'; row = self.prepared(root)
            reply = dict(choices=[dict(finish_reason='stop', message=dict(content='{"claims":[]}'))])
            with patch.object(pilot.urllib.request, 'urlopen', return_value=BytesIO(core.encoded(reply))):
                self.assertEqual(pilot.run(root, out, 'http://local/v1', 'qwen-local'), 0)
            self.assertEqual(pilot.read(out / (row['job_id'] + '.json'))['status'], 'ready_empty')

    def test_http_error_body_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'inputs'; out = Path(tmp) / 'out'; row = self.prepared(root)
            error = pilot.urllib.error.HTTPError('http://local', 400, 'Bad Request', {},
                                                BytesIO(b'{"error":"context overflow"}'))
            with patch.object(pilot.urllib.request, 'urlopen', side_effect=error):
                self.assertEqual(pilot.run(root, out, 'http://local/v1', 'qwen-local'), 1)
            result = pilot.read(out / (row['job_id'] + '.json'))
            self.assertIn('context overflow', result['http_error_body'])


if __name__ == '__main__':
    unittest.main()
