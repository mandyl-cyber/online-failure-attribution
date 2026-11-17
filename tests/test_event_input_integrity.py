import json
import unittest
from copy import deepcopy
from segmenter.adapters.openhands import OpenHandsStreamAdapter, _s, messages_to_events


def action(i,kind='run',args=None):
    return dict(id=i,source='agent',action=kind,args=args or {'command':'echo ok'})


def observation(i,cause=None,content='ok',**extra):
    return dict(id=i,cause=cause,observation='run',content=content,**extra)


class InputTests(unittest.TestCase):
    def test_structured_plan_and_nontext_parts_preserved(self):
        payload={'command':'plan','task_list':[{'id':'p','title':'verify','status':'done','notes':'reported failure'}]}
        adapter=OpenHandsStreamAdapter();adapter.feed(action(1,'task_tracking',payload))
        step=adapter.feed(observation(2,1))[0]
        self.assertEqual(json.loads(step.action),payload)
        self.assertEqual(json.loads(_s(payload['task_list'])),payload['task_list'])
        self.assertEqual(_s([{'type':'text','text':'a'},{'type':'text','text':'b'}]),'a\nb')
        image=[{'type':'image_url','image_url':{'url':'local-ref'}}]
        self.assertEqual(json.loads(_s(image)),image)

    def test_out_of_order_results_keep_actions_and_availability(self):
        a=OpenHandsStreamAdapter();a.feed(action(1));a.feed(action(2))
        self.assertEqual(a.feed(observation(3,2,'second')),[])
        steps=a.feed(observation(4,1,'first'))
        self.assertEqual([s.observation for s in steps],['first','second'])
        self.assertEqual([s.attrs['input_integrity']['observation_received_seq'] for s in steps],[4,3])
        self.assertEqual([s.attrs['input_integrity']['available_at_seq'] for s in steps],[4,4])

    def test_duplicates_and_conflicts(self):
        a=OpenHandsStreamAdapter();ev=action(1);before=deepcopy(ev)
        a.feed(ev);self.assertEqual(a.feed(ev),[])
        self.assertEqual(ev,before)
        self.assertEqual(len(a.feed(observation(2,1))),1)
        self.assertEqual(a.feed(observation(2,1)),[])
        with self.assertRaises(ValueError):a.feed(action(1,args={'command':'changed'}))

    def test_ambiguous_fallback_rejected(self):
        a=OpenHandsStreamAdapter();a.feed(action(1));a.feed(action(2))
        with self.assertRaisesRegex(ValueError,'Ambiguous'):a.feed(observation(3))
        b=OpenHandsStreamAdapter();b.feed(action(1))
        self.assertEqual(b.feed(observation(2))[0].attrs['input_integrity']['pairing'],'single_pending_fallback')

    def test_missing_orphan_and_truncation(self):
        a=OpenHandsStreamAdapter();a.feed(observation(1,99))
        self.assertEqual(a.diagnostics[0]['kind'],'unmatched_observation')
        a.feed(action(2));step=a.flush()[0]
        self.assertIn('missing_observation_at_flush',step.attrs['input_integrity']['issues'])
        self.assertFalse(step.attrs['input_integrity']['observation_complete'])
        b=OpenHandsStreamAdapter();b.feed(action(1))
        step=b.feed(observation(2,1,'Output truncated',extras={'truncated':True}))[0]
        self.assertIn('upstream_truncation_declared',step.attrs['input_integrity']['issues'])
        self.assertIn('possible_truncation_marker',step.attrs['input_integrity']['issues'])

    def test_chat_messages_preserve_plan(self):
        payload={'command':'plan','task_list':[{'title':'test','notes':'not yet run'}]}
        messages=[{'role':'assistant','tool_calls':[{'id':'c','function':{'name':'task_tracker','arguments':json.dumps(payload)}}]},
                  {'role':'tool','tool_call_id':'c','content':'updated'}]
        a=OpenHandsStreamAdapter();steps=[]
        for event in messages_to_events(messages):steps+=a.feed(event)
        self.assertEqual(json.loads(steps[0].action),payload)
        self.assertEqual(steps[0].raw_ref,{'action_id':'c','observation_id':'m1'})


if __name__=='__main__':unittest.main()
