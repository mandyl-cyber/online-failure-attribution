"""Exercise real child training processes on tiny grouped fixtures."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from remote_pilots.temporal_boundary_v1.launch import verify_data, REPO
from segmenter.temporal_boundary import VERSION, INPUT_DIM


class LauncherTests(unittest.TestCase):
    def test_parallel_launch_and_completed_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';data.mkdir();records=[];vectors={}
            for i in range(5):
                key=str(i);vectors[key]=np.zeros((4,INPUT_DIM),dtype=np.float32)
                records.append(dict(key=key,traj_id=key,group=key,fold=i,ids=[1,2,3,4],boundaries=[3],
                    baseline_rows=[dict(idx=j,features={'lag0:x':int(j==3)}) for j in range(1,5)]))
            (data/'records.json.gz').write_bytes(gzip.compress(json.dumps(records).encode()))
            (data/'audit.json').write_text(json.dumps(dict(failures=[],version=VERSION)))
            np.savez_compressed(data/'vectors.npz',**vectors)
            manifest=dict(trajectories=5,steps=20,files={n:dict(sha256=hashlib.sha256((data/n).read_bytes()).hexdigest())
                for n in ('records.json.gz','vectors.npz','audit.json')})
            (data/'manifest.json').write_text(json.dumps(manifest))
            command=[sys.executable,str(REPO/'remote_pilots/temporal_boundary_v1/launch.py'),
                     '--data',str(data),'--out',str(root/'result'),'--epochs','1','--workers','2','--threads','1']
            first=subprocess.run(command,cwd=REPO,capture_output=True,text=True,timeout=240)
            self.assertEqual(first.returncode,0,first.stdout+first.stderr)
            summary=json.loads((root/'result/summary.json').read_text())
            self.assertEqual(summary['status'],'complete');self.assertEqual(len(summary['folds']),5)
            model=root/'result/run_fold0/fold0/confidence/weights.pt';before=model.read_bytes()
            second=subprocess.run(command,cwd=REPO,capture_output=True,text=True,timeout=240)
            self.assertEqual(second.returncode,0,second.stdout+second.stderr)
            self.assertEqual(model.read_bytes(),before)
            self.assertIn('already complete',(root/'result/fold0.log').read_text())
            (data/'vectors.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):verify_data(data)


if __name__=='__main__':unittest.main()
