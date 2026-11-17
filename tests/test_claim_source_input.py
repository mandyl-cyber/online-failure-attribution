from copy import deepcopy
import unittest
from segmenter.claim_source_input import render
from remote_pilots.openhands_smoke import seal_claim_diagnostic as pilot


class SourceTests(unittest.TestCase):
    def test_lossless_and_no_mutation(self):
        _,rows,_=pilot.load(pilot.ROOT)
        for row in rows.values():
            before=deepcopy(row);typed=render(row)
            typed.pop('source_policy')
            for unit in typed['units']: unit.pop('source_roles')
            self.assertEqual(typed,row['request'])
            self.assertEqual(row,before)

    def test_plan_receipt_and_edit_distinguished(self):
        _,rows,_=pilot.load(pilot.ROOT);values=list(rows.values())
        roles=render(values[0])['units'][-1]['source_roles']
        self.assertEqual(roles['action'],'agent_plan')
        self.assertEqual(roles['observation'],'plan_update_receipt')
        self.assertEqual(render(values[6])['units'][0]['source_roles']['observation'],'observation')
        self.assertEqual(render(values[15])['units'][0]['source_roles']['embedded_authored_text'],'commit_message')

    def test_new_pack(self):
        from pathlib import Path
        protocol,rows,requests=pilot.load(Path('pilot/claim_source_input_v1'))
        self.assertEqual((len(rows),len(requests)),(8,16))
        self.assertEqual(len(set(protocol['instruction_sha256'].values())),1)


if __name__=='__main__': unittest.main()
