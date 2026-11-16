from copy import deepcopy
import unittest
from segmenter.relation_evidence import RelationEvidence


def step(idx,action,obs='',**kwargs):
    return dict(idx=idx,action=action,observation=obs,tool='run',tool_class='EXEC',**kwargs)


class RelationTests(unittest.TestCase):
    def test_zero_tests_then_related_read(self):
        b=RelationEvidence('fix issue')
        b.feed(step(1,'python tests/runtests.py','Ran 0 tests\nOK',exit_code=0))
        s=step(2,'view /repo/tests/runtests.py');s['tool_class']='READ'
        out=b.feed(s)
        self.assertEqual(out['recent_failure_candidates'][0]['kind'],'zero_tests')
        self.assertEqual(out['relation_candidates'][0]['from_step'],1)
        self.assertIsNone(out['boundary_decision'])

    def test_success_elsewhere_does_not_clear(self):
        b=RelationEvidence('x')
        b.feed(step(1,'pytest test_x.py','error',exit_code=1))
        self.assertTrue(b.feed(step(2,'git status','ok',exit_code=0))['recent_failure_candidates'])
        self.assertFalse(b.feed(step(3,'pytest test_x.py','1 passed',exit_code=0))['recent_failure_candidates'])

    def test_plan_is_declaration_and_failed_edit_not_write(self):
        b=RelationEvidence('x')
        b.feed(step(1,'python check.py','ModuleNotFoundError: missing',exit_code=1))
        s=step(2,'');s.update(tool_class='PLAN',public_plan={'task_list':[dict(id='v',title='Verify fix',status='done')]})
        self.assertTrue(b.feed(s)['declaration_evidence_tension_candidate']['present'])
        s=step(3,'str_replace x.py','ERROR: no replacement');s['tool_class']='EDIT'
        self.assertIsNone(b.feed(s)['recent_edit'])

    def test_ignores_hidden_fields_and_freezes_past(self):
        plain=step(1,'cat x.py','hello')
        altered=deepcopy(plain);altered.update(thought='hidden',gold=1,future='answer')
        a=RelationEvidence('x').feed(plain)
        b=RelationEvidence('x');saved=b.feed(altered)
        self.assertEqual(a,saved)
        b.feed(step(2,'new','new'))
        self.assertEqual(a,saved)

    def test_memory_bounded(self):
        b=RelationEvidence('x')
        for i in range(1,50):
            out=b.feed(step(i,'python '+str(i)+'.py','error',exit_code=1))
        self.assertEqual(len(out['recent_failure_candidates']),8)

    def test_reading_traceback_is_not_execution_failure(self):
        b=RelationEvidence('x')
        s=step(1,'cat failure.log','Traceback (most recent call last)\nModuleNotFoundError: x',exit_code=0)
        s['tool_class']='READ'
        self.assertFalse(b.feed(s)['recent_failure_candidates'])

    def test_successful_history_retained_without_goal_claim(self):
        b=RelationEvidence('x')
        b.feed(step(1,'pytest test_x.py','2 passed',exit_code=0))
        out=b.feed(step(2,'cat test_x.py','source'))
        self.assertEqual(out['recent_action_history'][0]['outcome'],'exit_zero_not_goal_success')
        self.assertEqual(out['relation_candidates'][0]['from_step'],1)
        self.assertEqual(out['purpose_relation'],'undetermined')

    def test_history_is_bounded_and_never_contains_current_or_future(self):
        b=RelationEvidence('x')
        for i in range(1,20):
            out=b.feed(step(i,'echo ok','ok',exit_code=0))
        self.assertEqual([h['step_id'] for h in out['recent_action_history']],list(range(11,19)))
        self.assertEqual(b.feed(step(100,'echo ok'))['recent_action_history'],[])


if __name__=='__main__':unittest.main()
