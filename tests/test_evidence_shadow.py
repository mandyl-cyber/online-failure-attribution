import unittest
from segmenter.evidence_shadow import Shadow


def action(i):
    return dict(id=i,source='agent',action='run',args=dict(command='python -m unittest -v test_task'))


def observation(i,cause,text):
    return dict(id=i,source='environment',observation='run',cause=cause,content=text,extras=dict(exit_code=0))


class ShadowTests(unittest.TestCase):
    def test_pending_duplicate_and_no_future_backfill(self):
        s=Shadow('case',bindings=[dict(action_id=1,object_id='task',version='hash',
            available_event_seq=3,provenance='trusted_runtime')])
        self.assertEqual(s.feed(action(1)),[])
        self.assertEqual(s.summary()['pending_actions'],1)
        rows=s.feed(observation(2,1,'Ran 1 test in 0.1s\n\nOK'))
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['latest_observed_test_status'],'passed')
        self.assertIsNone(rows[0]['fact']['object_id'])
        s.feed(action(1))
        self.assertEqual(s.summary()['steps'],1)
        self.assertIsNone(s.snapshots[0]['fact']['object_id'])
        bad=action(1);bad['args']['command']='different'
        with self.assertRaises(ValueError):s.feed(bad)

    def test_available_binding_and_explicit_claim(self):
        s=Shadow('case',bindings=[dict(action_id=1,object_id='task',version='h',
            available_event_seq=1,provenance='trusted_runtime')],claims=[dict(kind='current_test_pass',
            object_id='task',version='h',prefix_step=1)])
        s.feed(action(1)); row=s.feed(observation(2,1,'Ran 1 test in 0.1s\n\nOK'))[0]
        self.assertEqual(row['claim_decisions'][0]['decision']['assertion_support'],'supported')
        self.assertIsNone(row['timing']['alarm_latency'])

    def test_truncated_observation_cannot_establish_pass(self):
        s=Shadow('case');s.feed(action(1))
        o=observation(2,1,'Ran 1 test in 0.1s\n\nOK');o['extras']['truncated']=True
        row=s.feed(o)[0]
        self.assertEqual(row['fact']['status'],'unknown')


if __name__=='__main__':unittest.main()
