import unittest
from copy import deepcopy

from scripts.prepare_memory_mechanism_probe import prefix_records, reference_atoms
from segmenter.memory_mechanism_probe import evaluate, parse, score


class MechanismTests(unittest.TestCase):
    def test_lookup_fetches_only_selected_visible_source(self):
        import json
        case = dict(case_id='case', session='task', prefix_step=1,
                    assertion='Tests passed', questions=['Supported?'])
        packet = dict(raw=[dict(record_id='r', source_step_id=1, field='observation', text='FAILED')],
                      learned_stage_claims=[dict(type='claims', claims=[dict(text='Tests failed',
                          basis='tool_observation', source_steps=[1], evidence_ids=['r'])])])
        outputs = iter([json.dumps(dict(lookup_ids=['r'])), json.dumps(dict(latest_run_status='failed',
            assertion_support='not_established', evidence_ids=['r'], limitations='Shown suite failed'))])
        messages = []
        def tokenize(value):
            messages.append(deepcopy(value))
            return [0]*10
        row = evaluate(case, packet, 'learned_claim_with_lookup', tokenize,
                       lambda ids: (next(outputs), 20, .1))
        self.assertEqual(row['lookup_calls'], 1)
        self.assertEqual(row['input_tokens'], 20)
        self.assertNotIn('FAILED', str(messages[0]))
        self.assertIn('FAILED', str(messages[1]))
        self.assertTrue(row['answer']['valid'])
        self.assertFalse(parse('{"lookup_ids":["future"]}', ['r'], True)['valid'])
        self.assertFalse(parse('{"lookup_ids":["r"]}', ['r'], False)['valid'])

    def test_lookup_budget_failure_is_counted_not_dropped(self):
        case = dict(case_id='case', session='task', prefix_step=1, assertion='Passed', questions=[])
        packet = dict(raw=[dict(record_id='r', source_step_id=1, field='observation', text='FAILED')],
            learned_stage_claims=[dict(type='claims', claims=[dict(text='failed', evidence_ids=['r'])])])
        calls=[]
        def generate(ids):
            calls.append(1)
            return '{"lookup_ids":["r"]}', 2, .1
        row=evaluate(case, packet, 'learned_claim_with_lookup', lambda m:[0]*10,
                     generate, max_length=25, new_tokens=10)
        self.assertEqual(len(calls), 1)
        self.assertEqual(row['answer']['error'], 'cumulative_input_budget_exceeded')
        result=score([row],[dict(case_id='case',latest_run_status='failed',assertion_support='not_established')])
        self.assertEqual(result['arms']['learned_claim_with_lookup']['evaluated'], 1)
        self.assertEqual(result['arms']['learned_claim_with_lookup']['invalid'], 1)

    def test_future_record_and_future_fact_are_excluded(self):
        records = [dict(record_id='old', source_step_id=1, field='observation', text='FAILED', availability_seq=1),
                   dict(record_id='future', source_step_id=2, field='observation', text='OK', availability_seq=2)]
        facts = [dict(fact_id='f1', step_id=1, source_field='observation', record_text='FAILED',
                      basis='tool_observation', text='Shown tests failed.', category='SECRET_LABEL',
                      anchor=dict(start=0,end=6,quote='FAILED')),
                 dict(fact_id='f2', step_id=2, source_field='observation', record_text='OK',
                      basis='tool_observation', text='Shown tests passed.', anchor=dict(start=0,end=2,quote='OK'))]
        atoms = reference_atoms(facts, records, 1)
        self.assertEqual(len(atoms), 1)
        self.assertEqual(atoms[0]['evidence_ids'], ['old'])
        self.assertNotIn('SECRET_LABEL', str(atoms))
        before = deepcopy(atoms)
        records[1]['text'] = 'DIFFERENT FUTURE'
        self.assertEqual(reference_atoms(facts, records, 1), before)

    def test_reference_does_not_drift_from_native_evidence(self):
        records = [dict(record_id='x', source_step_id=1, field='observation', text='FAILED', availability_seq=1)]
        facts = [dict(fact_id='f1', step_id=1, source_field='observation', record_text='OK',
                      basis='tool_observation', text='passed', anchor=dict(start=0,end=2,quote='OK'))]
        with self.assertRaises(ValueError):
            reference_atoms(facts, records, 1)

    def test_prefix_is_a_copy_and_rejects_late_evidence(self):
        records = [dict(source_step_id=1, availability_seq=2)]
        with self.assertRaises(ValueError):
            prefix_records(records, 1)
        records[0]['availability_seq'] = 1
        selected = prefix_records(records, 1)
        selected[0]['availability_seq'] = 99
        self.assertEqual(records[0]['availability_seq'], 1)


if __name__ == '__main__':
    unittest.main()
