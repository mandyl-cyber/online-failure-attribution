import unittest
from scripts.check_evidence_state import controls,fact
from segmenter.evidence_state import EvidenceState,native_fact,observed_test_status,short_view


class EvidenceTests(unittest.TestCase):
    def test_bounded_query_and_reversible_short_view(self):
        s=EvidenceState();s.append(fact(1));s.append(fact(2));s.append(fact(3,obj='other'))
        selected=s.query(3,object_id='module-a',kind='test_run',limit=1)
        self.assertEqual(len(selected),1);self.assertEqual(selected[0]['step'],2)
        view=short_view(selected)
        self.assertEqual(view['id_mapping'][view['facts'][0]['evidence_ids'][0]],'R2')
        self.assertEqual(selected[0]['evidence_ids'],['R2'])
        self.assertEqual(view['facts'][0]['text'],selected[0]['text'])

    def test_controlled_positive_and_negative_pairs(self):
        rows=controls()
        self.assertTrue(all(x['correct'] for x in rows),[(x['case'],x['decision']) for x in rows if not x['correct']])
        self.assertGreater(sum(x['expected']=='supported' for x in rows),1)

    def test_future_and_late_arrival_do_not_change_prefix(self):
        s=EvidenceState(); s.append(fact(1))
        c=dict(kind='current_test_pass',prefix_step=1,object_id='module-a',version='v1')
        before=s.assess(c)
        f=fact(2,status='failed');s.append(f)
        late=fact(1,status='failed');late['available_step']=3;s.append(late)
        self.assertEqual(s.assess(c),before)

    def test_native_agent_text_is_never_tool_receipt(self):
        f=native_fact(dict(action='finish',id=1,args=dict(final_thought='All tests passed. Changes committed.')),None,1,['A'])
        self.assertEqual(f['source'],'agent_assertion');self.assertEqual(f['kind'],'agent_statement')
        command=dict(action='run',id=1,args=dict(command='git commit -m fix'))
        receipt=dict(observation='run',id=2,content='[main abc1234] fix',extras=dict(exit_code=0))
        self.assertEqual(native_fact(command,receipt,1,['A','B'])['status'],'confirmed')
        receipt['extras']['exit_code']=1
        self.assertEqual(native_fact(command,receipt,1,['A','B'])['status'],'unknown')

    def test_scope_and_combined_commands_remain_unknown(self):
        a=dict(action='run',args=dict(command='python -m unittest; echo OK'))
        o=dict(observation='run',content='Ran 3 tests in 0.1s\n\nOK')
        self.assertEqual(native_fact(a,o,1,['A'])['kind'],'unclassified')
        self.assertEqual(observed_test_status('Ran 0 tests in 0.1s\nOK'),'zero_tests')
        self.assertEqual(observed_test_status('Ran 2 tests in 0.1s\nOK\nRan 1 test in 0.1s\nFAILED'),'unknown')


if __name__=='__main__':unittest.main()
