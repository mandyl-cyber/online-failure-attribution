import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from copy import deepcopy
import numpy as np
import torch
from segmenter.temporal_boundary import TemporalNet,TemporalPredictor,Confirmation,StepEncoder,save_model,INPUT_DIM
from segmenter.schema import Step
from segmenter.online_ledger import OnlineLedger


def constant_net(logit=0.):
    net=TemporalNet(16)
    for p in net.parameters():p.data.zero_()
    net.head[-1].bias.data.fill_(logit)
    return net


class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_epoch_resume_matches_uninterrupted_training(self):
        import segmenter.train_temporal as training
        records=[];vectors={}
        for i in range(5):
            k=str(i);vectors[k]=np.random.default_rng(i).normal(size=(4,INPUT_DIM)).astype('float32')
            records.append(dict(traj_id=k,key=k,group=k,fold=i,ids=[1,2,3,4],boundaries=[3]))
        args=SimpleNamespace(seed=41,hidden=8,device='cpu',epochs=2,benchmark=True,input_sha256={},resume=False)
        with tempfile.TemporaryDirectory() as d:
            full=Path(d)/'full';part=Path(d)/'part';full.mkdir();part.mkdir()
            training.train_fold(args,records,vectors,0,full)
            save=training.atomic_torch_save
            def interrupt(value,path):
                save(value,path)
                raise RuntimeError('simulated process stop after epoch commit')
            with patch.object(training,'atomic_torch_save',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'simulated'):training.train_fold(args,records,vectors,0,part)
            args.resume=True
            training.train_fold(args,records,vectors,0,part)
            left=torch.load(full/'benchmark_only/weights.pt',weights_only=True)
            right=torch.load(part/'benchmark_only/weights.pt',weights_only=True)
            for key in left:torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)

    def test_single_encode_loss_matches_stepwise_rollout(self):
        from segmenter.train_temporal import trajectory_loss,policy
        from torch.nn import functional as F
        torch.manual_seed(9);net=TemporalNet(8);reference=deepcopy(net)
        x=np.random.default_rng(3).normal(size=(7,INPUT_DIM)).astype('float32')
        r=dict(ids=list(range(1,8)),boundaries=[3,5])
        actual=trajectory_loss(net,r,x,torch.tensor(2.))
        roll=TemporalPredictor(reference,policy());starts=[]
        for idx,vector in zip(r['ids'],x):starts.append(roll.controller.start);roll.feed_vector(vector,idx)
        z,h,_=reference.encode(torch.from_numpy(x).unsqueeze(0));z,h=z[0],h[0]
        prefix=torch.cat((z.new_zeros(1,8),z.cumsum(0)));inputs=[];ys=[];ws=[]
        for t in range(1,len(x)):
            for b in range(max(1,t-2),t+1):
                start=min(starts[t],b-1)
                inputs.append(torch.cat((h[b-1],h[b],h[t],(prefix[b]-prefix[start])/(b-start),h.new_tensor([np.log1p(b-start),(t-b)/2.]))))
                ys.append(float(r['ids'][b] in r['boundaries']));ws.append(1/min(3,len(x)-b))
        logits=reference.head(torch.stack(inputs)).squeeze(-1);weights=torch.tensor(ws)
        expected=(F.binary_cross_entropy_with_logits(logits,torch.tensor(ys),pos_weight=torch.tensor(2.),reduction='none')*weights).sum()/weights.sum()
        torch.testing.assert_close(actual,expected)
        actual.backward();expected.backward()
        for p,q in zip(net.parameters(),reference.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-6)

    def test_training_selection_and_checkpoint_roundtrip(self):
        from segmenter.train_temporal import train_fold
        records=[];vectors={}
        for i in range(5):
            key=str(i);vectors[key]=np.random.default_rng(i).normal(size=(5,INPUT_DIM)).astype('float32')
            records.append(dict(traj_id=key,key=key,group=key,fold=i,ids=[1,2,3,4,5],boundaries=[3],
                baseline_rows=[dict(idx=j,features={'lag0:value':float(j==3)},text='') for j in range(1,6)]))
        args=SimpleNamespace(seed=41,hidden=8,device='cpu',epochs=1,benchmark=False,input_sha256={})
        with tempfile.TemporaryDirectory() as d:
            result=train_fold(args,records,vectors,0,Path(d))
            self.assertEqual(set(result['comparisons']),{'lr','immediate','confidence'})
            predictor=TemporalPredictor.load(Path(d)/'confidence/model.json')
            for idx,x in enumerate(vectors['0'],1): predictor.feed_vector(x,idx)
            predictor.finish()
            predicted=json.loads((Path(d)/'predictions.json').read_text())
            self.assertEqual(predictor.events,predicted['confidence']['0'])

    def test_confirmation_same_candidate_deadline_and_order(self):
        c=Confirmation(.4,.2,2)
        self.assertEqual(c.advance(1,lambda b,t,s:.5),[])
        self.assertEqual(c.advance(2,lambda b,t,s:.5 if b==1 else .99),[])
        out=c.advance(3,lambda b,t,s:.5 if b==1 else .99)
        self.assertEqual([x['position'] for x in out],[1,2,3])
        self.assertEqual(out[0]['reason'],'deadline')

    def test_prefix_invariance_and_bounded_cache(self):
        torch.manual_seed(3);net=TemporalNet(16)
        xs=np.random.default_rng(4).normal(size=(30,INPUT_DIM)).astype('float32')
        a=TemporalPredictor(net);b=TemporalPredictor(net)
        for i,x in enumerate(xs[:12],1):a.feed_vector(x,i);b.feed_vector(x,i)
        prefix=json.dumps(a.stages)
        for i,x in enumerate(xs[12:],13):
            a.feed_vector(x,i)
            self.assertLessEqual(len(a.cache),3)
        self.assertEqual(json.dumps(a.stages[:len(b.stages)]),prefix)
        for i,x in enumerate(xs[12:],13):b.feed_vector(x,i)
        a.finish();b.finish()
        self.assertEqual(a.snapshot(),b.snapshot())

    def test_finish_flushes_candidates_without_backdating(self):
        p=TemporalPredictor(constant_net(),dict(threshold=.4,margin=.2,max_delay=2))
        for i in range(1,4):p.feed_vector(np.zeros(INPUT_DIM),i)
        self.assertEqual(p.stages,[])
        p.finish()
        self.assertEqual([(e['boundary_step_id'],e['detected_at_step_id']) for e in p.events],[(2,3),(3,3)])
        self.assertEqual([(s['start_step_id'],s['end_step_id']) for s in p.snapshot()],[(1,1),(2,2),(3,3)])

    def test_encoder_no_thought_and_incremental_equivalence(self):
        a=Step(idx=1,tool='task_tracking',tool_class='PLAN',action='plan',attrs={'public_action_args':{'task_list':[{'id':'x','status':'done'}]}})
        b=Step.from_dict(a.to_dict());b.thought='future secret'
        np.testing.assert_array_equal(StepEncoder().feed(a),StepEncoder().feed(b))
        net=TemporalNet(16).eval();x=torch.randn(1,8,INPUT_DIM)
        with torch.no_grad():
            _,full,_=net.encode(x);state=None;parts=[]
            for i in range(8):
                _,h,state=net.encode(x[:,i:i+1],state);parts.append(h)
        torch.testing.assert_close(full,torch.cat(parts,dim=1),rtol=1e-5,atol=1e-6)

    def test_ledger_delayed_seal_recovery_exactly_once(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);save_model(constant_net(),root/'model',dict(threshold=.4,margin=.2,max_delay=2),{})
            model=root/'model/model.json'
            def open_run():return OnlineLedger(root/'run','temporal-test',background=False,boundary_model=model)
            r=open_run()
            for i in range(2):
                r.ingest(dict(id=2*i,source='agent',action='run',args={'command':'pytest'}))
                r.ingest(dict(id=2*i+1,observation='run',cause=2*i,content='1 passed'))
            self.assertEqual(len(list(r.jobs.glob('*.json'))),0)
            r.close();r=open_run()
            r.ingest(dict(id=4,source='agent',action='finish',args={}))
            r.drain();before={p.name:p.read_bytes() for p in r.results.glob('*.json')};r.close()
            r=open_run();r.drain();r.close()
            self.assertEqual(before,{p.name:p.read_bytes() for p in r.results.glob('*.json')})
            self.assertTrue(all(s['sealed'] for s in r._selected_stages()))


if __name__=='__main__':torch.set_num_threads(1);unittest.main()
