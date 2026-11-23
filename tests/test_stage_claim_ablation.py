"""Scientific controls: causality, equal raw evidence, budgets and denominators."""
import unittest

from segmenter.stage_claim_ablation import (fixed_jobs, representation, queue_schedule,
    messages_for_query, parse_answer, score, pack)


def evidence(step, field='observation'):
    return dict(record_id=f'r{step}-{field}', source_step_id=step, field=field,
                text=f'record at step {step}', availability_seq=step,
                run_id='s', content_sha256='fixture')


def fixture():
    records = [evidence(i) for i in range(1, 9)]
    job = fixed_jobs('s', records, 4, False, None)[0]
    stage = dict(session='s', job=job, seal_event_seq=10)
    return dict(sessions=[dict(session='s', records=records)], units=dict(stage=[stage], fixed=[stage]))


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(len(str(messages))))


class AblationTests(unittest.TestCase):
    def test_open_tail_is_not_artificially_sealed(self):
        records = [evidence(i) for i in range(1, 7)]
        self.assertEqual(len(fixed_jobs('s', records, 4, False, None)), 1)
        jobs = fixed_jobs('s', records, 4, True, None)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[-1]['detected_at_seq'], 6)

    def test_raw_stage_grouping_does_not_duplicate_or_drop_records(self):
        data = fixture()
        q = dict(session='s', prefix_step=5, prefix_event_seq=12)
        for arm in ('raw', 'stage_raw'):
            units = representation(data, q, arm, {}, 'representation', .05)
            ids = [r['evidence_id'] for u in units for r in u.get('records', [])]
            self.assertEqual(sorted(ids), [f'r{i}-observation' for i in range(1, 6)])
            self.assertEqual(len(ids), len(set(ids)))

    def test_future_seal_and_future_records_are_invisible(self):
        data = fixture()
        data['units']['stage'][0]['job']['detected_at_seq'] = 6
        q = dict(session='s', prefix_step=5, prefix_event_seq=12)
        units = representation(data, q, 'stage_raw', {}, 'representation', .05)
        self.assertFalse(any(u['type'] == 'stage_raw' for u in units))
        self.assertTrue(all(r['step'] <= 5 for u in units for r in u['records']))

    def test_queue_does_not_publish_claim_before_completion(self):
        data = fixture()
        item = data['units']['stage'][0]
        key = item['job']['job_id']
        outputs = {key: dict(service_seconds=10., status='ready', claims=[])}
        q = dict(session='s', prefix_step=5, prefix_event_seq=12)
        offline = representation(data, q, 'stage_claim', outputs, 'representation', .05)
        queued = representation(data, q, 'stage_claim', outputs, 'queued', .05)
        self.assertTrue(any(u['type'] == 'claims' for u in offline))
        self.assertFalse(any(u['type'] == 'claims' for u in queued))
        schedule = queue_schedule([item], outputs, .05)
        self.assertAlmostEqual(schedule[key]['ready_seconds'], 10.5)

    def test_no_reference_answer_enters_question(self):
        q = dict(target_step=3, target_field='observation', prefix_step=7,
                 category='SECRET_LABEL', text='SECRET_REFERENCE')
        messages = str(messages_for_query(q, []))
        self.assertNotIn('SECRET', messages)

    def test_invalid_and_invisible_citations_are_rejected(self):
        self.assertFalse(parse_answer('{"category":"edit_failed","evidence_ids":["future"]}', ['r1'])['valid'])
        self.assertFalse(parse_answer('{"category":"edit_failed","evidence_ids":[]}', ['r1'])['valid'])
        self.assertTrue(parse_answer('{"category":"unknown","evidence_ids":[]}', [])['valid'])

    def test_truncation_is_by_whole_unit_and_reported(self):
        q = dict(target_step=1, target_field='observation', prefix_step=2)
        units = [dict(type='raw', records=[dict(evidence_id='old', text='x' * 2000)]),
                 dict(type='raw', records=[dict(evidence_id='new', text='small')])]
        _, ids, details = pack(Tokenizer(), q, units, 1600, 100)
        self.assertLessEqual(len(ids) + 100, 1600)
        self.assertEqual(details['omitted_record_ids'], ['old'])
        self.assertEqual(details['visible_ids'], ['new'])

    def test_invalid_answers_remain_in_denominator(self):
        labels = [dict(query_id='q1', session='s', category='edit_failed', record_id='r1'),
                  dict(query_id='q2', session='s', category='edit_failed', record_id='r2')]
        rows = [dict(query_id=q, arm='raw', answer=dict(valid=q == 'q1', category='edit_failed',
                    evidence_ids=['r1']), packing=dict(input_tokens=2, omitted_units=0), output_tokens=2, seconds=1.)
                for q in ('q1', 'q2')]
        result = score(rows, labels)['arms']['raw']
        self.assertEqual(result['category_accuracy'], .5)
        self.assertEqual(result['joint_rate'], .5)


if __name__ == '__main__':
    unittest.main()
