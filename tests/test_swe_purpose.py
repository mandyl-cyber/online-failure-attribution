import unittest
from segmenter.prepare_swe_purpose import select
from segmenter.purpose_pilot_remote import jobs
import json
import tempfile
from pathlib import Path


class SwePurposeTests(unittest.TestCase):
    def test_selection_is_deterministic_group_isolated(self):
        records=[dict(traj_id=str(i),n_steps=i+1) for i in range(90)]
        assignments={str(i):dict(split='train' if i<60 else 'dev',group=str(i//2)) for i in range(90)}
        a=select(records,assignments,'train',12,350)
        self.assertEqual(a,select(list(reversed(records)),assignments,'train',12,350))
        self.assertEqual(len(a),12)
        self.assertEqual(len({assignments[r['traj_id']]['group'] for r in a}),12)
        self.assertTrue(all(assignments[r['traj_id']]['split']=='train' for r in a))

    def test_remote_prefix_does_not_include_future(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            row=dict(trajectory_id='T',task='task',steps=[dict(idx=i,action=f'secret-{i}') for i in (1,2,3)])
            (root/'trajectories.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
            requests=list(jobs(root))
            self.assertEqual([len(r['packet']['steps']) for r in requests],[1,2,3])
            self.assertNotIn('secret-3',json.dumps(requests[1]))


if __name__=='__main__':
    unittest.main()
