import json
import tempfile
import unittest
from pathlib import Path
from segmenter.durable_input import DurableInput
from segmenter.run_online import main


def action(i):
    return dict(id=i, source='agent', action='run', args={'command':'echo ok'})


def obs(i, cause):
    return dict(id=i, observation='run', cause=cause, content='ok', extras={'exit_code':0})


class DurableTests(unittest.TestCase):
    def test_notifications_and_partial_results_do_not_complete_action(self):
        with tempfile.TemporaryDirectory() as d:
            a = DurableInput(d, {})
            a.feed(action(1))
            self.assertEqual(a.feed(dict(id=2, observation='agent_state_changed', extras={'agent_state':'running'})), [])
            self.assertEqual(a.feed(dict(id=6, observation='recall', cause=99, content='workspace context')), [])
            with self.assertRaises(ValueError): a.feed(dict(obs(3,1), is_final=False))
            self.assertEqual(len(a.feed(obs(4,1))), 1)
            self.assertTrue((Path(d)/'rejected.json').exists())

    def test_pending_restore_duplicates_and_order(self):
        with tempfile.TemporaryDirectory() as d:
            a = DurableInput(d, {})
            a.feed(action(1)); a.feed(action(2)); a.feed(obs(3, 2))
            b = DurableInput(d, {})
            self.assertEqual(b.recovered, [])
            self.assertEqual(b.feed(action(1)), [])
            steps = b.feed(obs(4, 1))
            self.assertEqual([s.raw_ref['action_id'] for s in steps], [1, 2])
            c = DurableInput(d, {})
            self.assertEqual([s.to_dict() for s in c.recovered], [s.to_dict() for s in steps])

    def test_contract_and_conflict(self):
        with tempfile.TemporaryDirectory() as d:
            a = DurableInput(d, {})
            for ev in [obs(1, 99), obs(2, None), {'action':'run'}]:
                with self.assertRaises(ValueError): a.feed(ev)
            a.feed(action(3)); a.feed(obs(4, 3))
            with self.assertRaises(ValueError): a.feed(obs(5, 3))
            with self.assertRaises(ValueError): a.feed(dict(action(3), args={'command':'bad'}))
            with self.assertRaises(ValueError): DurableInput(d, {'changed':True})

    def test_follow_pause_restart_finalize_and_partial_line(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d)/'source.jsonl'; out = Path(d)/'out'
            src.write_text(json.dumps(action(1))+'\n'+json.dumps(obs(2,1))[:15], encoding='utf-8')
            args = ['follow',str(src),'--out',str(out),'--poll','0.001','--idle','0.003']
            main(args)
            status = lambda: json.loads((out/'follow_status.json').read_text())
            self.assertEqual((status()['status'],status()['pending'],status()['steps']), ('paused',1,0))
            src.write_text(json.dumps(action(1))+'\n'+json.dumps(obs(2,1))+'\n', encoding='utf-8')
            main(args)
            self.assertEqual((status()['status'],status()['steps']), ('paused',1))
            before = (out/'steps.jsonl').read_text()
            main(args+['--finalize'])
            self.assertEqual(status()['status'],'finished')
            self.assertEqual((out/'steps.jsonl').read_text(), before)
            main(args)
            self.assertEqual((out/'steps.jsonl').read_text(), before)

    def test_finalize_missing_observation_marks_issue(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d)/'s.jsonl'; out = Path(d)/'out'
            src.write_text(json.dumps(action(1))+'\n')
            main(['follow',str(src),'--out',str(out),'--idle','0.001','--poll','0.001','--finalize'])
            step = json.loads((out/'steps.jsonl').read_text())
            self.assertIn('missing_observation_at_flush',step['attrs']['input_integrity']['issues'])


if __name__ == '__main__': unittest.main()
