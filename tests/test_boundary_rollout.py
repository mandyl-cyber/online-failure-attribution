import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from segmenter.online_ledger import OnlineLedger
from segmenter.temporal_boundary import TemporalPredictor

MODEL=Path('artifacts/boundary_gru_adjacent_v1/model.json')


class BoundaryRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_packaged_real_model_policy_and_cpu(self):
        predictor=TemporalPredictor.load(MODEL)
        self.assertEqual(sum(p.numel() for p in predictor.net.parameters()),190081)
        self.assertEqual(predictor.device.type,'cpu')
        self.assertEqual(predictor.policy_config,dict(threshold=.55,margin=0.,max_delay=0,adjacent_threshold=.65))

    def test_native_recovery_works_without_lr_dependency(self):
        events=[json.loads(line) for line in Path('pilot/claim_prospective_eval_v1/stable_unique.jsonl').read_text(encoding='utf-8').splitlines()]
        with tempfile.TemporaryDirectory() as directory,patch.dict('sys.modules',{'segmenter.boundary_model':None}):
            def open_runtime():return OnlineLedger(directory,'real-gru',background=False,boundary_model=MODEL,input_mode='replay')
            runtime=open_runtime()
            for event in events[:10]:runtime.ingest(event)
            before=(Path(directory)/'stages.json').read_bytes()
            jobs={p.name:p.read_bytes() for p in runtime.jobs.glob('*.json')}
            runtime.close();runtime=open_runtime()
            self.assertEqual((Path(directory)/'stages.json').read_bytes(),before)
            self.assertEqual(jobs,{p.name:p.read_bytes() for p in runtime.jobs.glob('*.json')})
            for event in events[10:]:runtime.ingest(event)
            self.assertFalse(runtime.finished)
            self.assertFalse(runtime.adapter._pending)
            self.assertEqual(len(runtime.segmenter.steps),7)
            self.assertTrue(all('ms' not in e for e in runtime.boundary_events))
            runtime.close()

    def test_policy_change_requires_new_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            model=root/'model.json'
            model.write_bytes(MODEL.read_bytes())
            shutil.copyfile(MODEL.parent/'weights.pt',root/'weights.pt')
            runtime=OnlineLedger(root/'run','frozen-policy',background=False,boundary_model=model)
            runtime.close()
            metadata=json.loads(model.read_text(encoding='utf-8'))
            metadata['policy']['adjacent_threshold']=.7
            model.write_text(json.dumps(metadata),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'configuration mismatch'):
                OnlineLedger(root/'run','frozen-policy',background=False,boundary_model=model)


if __name__=='__main__':unittest.main()
