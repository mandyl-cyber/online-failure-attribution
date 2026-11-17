from collections import UserDict
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from remote_pilots.openhands_smoke import seal_claim_diagnostic as pilot
from remote_pilots.openhands_smoke.repair_seal_claim_inputs import full_calls


class PilotTests(unittest.TestCase):
    def test_pack_pairing_and_recovery(self):
        _,rows,requests=pilot.load(pilot.ROOT)
        self.assertEqual((len(rows),len(requests)),(24,48))
        self.assertFalse(any(pilot.input_issues(row) for row in rows.values()))
        repairs=pilot.io.read(pilot.ROOT/'repairs.json')
        self.assertEqual(len(repairs),16)
        self.assertTrue(any('notes' in task for c in repairs for task in c['payload']['task_list']))

    def test_call_identity_and_conflicts(self):
        call=dict(id='a',function=dict(name='task_tracker',arguments='{"task_list":[{"notes":"keep"}]}'))
        self.assertIn('notes',full_calls([dict(id={}),call],{'a'})['a']['task_list'][0])
        other=deepcopy(call);other['function']['arguments']='{}'
        with self.assertRaises(ValueError): full_calls([call,other],{'a'})
        self.assertEqual(full_calls([call],{'b'}),{})

    def test_tokenizer_mapping_and_blank_detection(self):
        class Tokenizer:
            def apply_chat_template(self,*a,**kw): return UserDict(input_ids=[[1,2,3]])
        self.assertEqual(pilot.count_tokens(Tokenizer(),[]),3)
        _,rows,_=pilot.load(pilot.ROOT);row=deepcopy(next(iter(rows.values())))
        row['request']['units'][0]['action'][0]['chunks'][0]['text']=' \n'
        self.assertTrue(pilot.input_issues(row))

    def test_pending_requests_never_call_api(self):
        protocol,rows,_=pilot.load(pilot.ROOT)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'preflight.json'
            pilot.io.save(path,dict(request_sha256=protocol['requests_sha256'],input_sha256=protocol['input_sha256'],
                                   pair_eligible={j:False for j in rows}))
            with patch.object(pilot.urllib.request,'urlopen',side_effect=AssertionError('must not call')):
                pilot.run(pilot.ROOT,Path(directory)/'out',path,'http://localhost:1/v1','test')
            summary=pilot.io.read(Path(directory)/'out'/'summary.json')
            self.assertEqual(summary['counts'],{'blocked_input':48})


if __name__=='__main__': unittest.main()
