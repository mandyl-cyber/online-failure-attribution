"""One fixed GRU architecture, grouped train/dev/test, three deployment comparisons."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time
import os
import gzip
import numpy as np
import torch
from torch.nn import functional as F
from .temporal_boundary import TemporalNet, TemporalPredictor, Confirmation, VERSION, save_model
from .eval_online import aggregate
from .durable_input import atomic_json


def atomic_torch_save(value,path):
    temporary=path.with_suffix('.tmp')
    torch.save(value,temporary)
    os.replace(temporary,path)


def policy(threshold=.5,margin=.15,delay=2):
    return dict(threshold=threshold,margin=margin,max_delay=delay)


def predict(net, records, vectors, config):
    output={}
    for r in records:
        p=TemporalPredictor(net,config)
        for x,idx in zip(vectors[r['key']],r['ids']): p.feed_vector(x,idx)
        p.finish()
        output[r['traj_id']]=p.events
    return output


def trajectory_loss(net, r, x, positive_weight):
    # Encode the causal GRU once; use detached prefix states for the current-model
    # rollout. No gold reset or future hidden state enters any score.
    net.train()
    tensor=torch.as_tensor(x,dtype=torch.float32,device=next(net.parameters()).device).unsqueeze(0)
    z,h,_=net.encode(tensor); z,h=z[0],h[0]
    prefix=torch.cat((z.new_zeros(1,net.hidden),z.cumsum(0)),dim=0)
    controller=Confirmation(**policy())
    starts=[]
    with torch.no_grad():
        def score(b,t,start):
            return torch.sigmoid(net.score(h[b-1],h[b],h[t],
                (prefix[b]-prefix[start])/(b-start),b-start,t-b)).item()
        for t in range(len(x)):
            starts.append(controller.start)
            controller.advance(t,score)
    gold=set(r['boundaries']); inputs=[]; labels=[]; weights=[]
    for t in range(1,len(x)):
        for b in range(max(1,t-2),t+1):
            start=min(starts[t],b-1)
            inputs.append(torch.cat((h[b-1],h[b],h[t],(prefix[b]-prefix[start])/(b-start),
                                    h.new_tensor([np.log1p(b-start),(t-b)/2.]))))
            labels.append(float(r['ids'][b] in gold))
            weights.append(1/min(3,len(x)-b))
    if not inputs: return None
    logits=net.head(torch.stack(inputs)).squeeze(-1)
    target=logits.new_tensor(labels); weight=logits.new_tensor(weights)
    loss=F.binary_cross_entropy_with_logits(logits,target,reduction='none',pos_weight=positive_weight)
    return (loss*weight).sum()/weight.sum()


def train_fold(args, records, vectors, fold, out):
    dev_fold=(fold+1)%5
    train=[r for r in records if r['fold'] not in (fold,dev_fold)]
    dev=[r for r in records if r['fold']==dev_fold]
    test=[r for r in records if r['fold']==fold]
    if not train or not dev or not test: raise ValueError('Empty task split')
    splits=[{r['group'] for r in rs} for rs in (train,dev,test)]
    if any(splits[i]&splits[j] for i in range(3) for j in range(i)): raise ValueError('Task leakage')
    labels={r['traj_id']:dict(boundaries=r['boundaries']) for r in records}
    torch.manual_seed(args.seed); random.seed(args.seed)
    net=TemporalNet(args.hidden).to(args.device)
    optim=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
    positives=sum(len(r['boundaries']) for r in train)
    n=sum(len(r['ids'])-1 for r in train)
    pos_weight=torch.tensor(min(5.,max(1.,(n-positives)/max(1,positives))),device=args.device)
    best=None; best_loss=float('inf'); stale=0; logs=[]
    resume_path=out/'resume.pt'
    if getattr(args,'resume',False) and resume_path.exists():
        state=torch.load(resume_path,map_location=args.device,weights_only=True)
        net.load_state_dict(state['model']);optim.load_state_dict(state['optimizer'])
        best=state['best'];best_loss=state['best_loss'];stale=state['stale'];logs=state['logs']
        random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'].cpu())
        if args.device.startswith('cuda') and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        print(json.dumps(dict(fold=fold,resumed_after_epoch=len(logs))),flush=True)
    # Early stopping uses dev multi-prefix BCE, not test boundaries or repeated test F1.
    for epoch in range(len(logs),args.epochs):
        if stale>=3: break
        started=time.perf_counter(); total=0.; order=train.copy();random.shuffle(order)
        for r in order:
            optim.zero_grad(set_to_none=True)
            loss=trajectory_loss(net,r,vectors[r['key']],pos_weight)
            if loss is None: continue
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            optim.step();total+=float(loss.detach())
        net.eval()
        with torch.no_grad():
            losses=[trajectory_loss(net,r,vectors[r['key']],pos_weight) for r in dev]
            validation=float(torch.stack([l for l in losses if l is not None]).mean())
        elapsed=time.perf_counter()-started
        log=dict(epoch=epoch+1,train_loss=total/len(train),dev_loss=validation,seconds=elapsed)
        logs.append(log); print(json.dumps(dict(fold=fold,**log)),flush=True)
        if validation<best_loss-1e-4:
            best_loss=validation;best=deepcopy(net.state_dict());stale=0
        else: stale+=1
        atomic_torch_save(dict(model=net.state_dict(),optimizer=optim.state_dict(),best=best,
            best_loss=best_loss,stale=stale,logs=logs,python_rng=random.getstate(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else None),resume_path)
        if stale>=3: break
    net.load_state_dict(best);net.eval()
    provenance=dict(fold=fold,seed=args.seed,training_groups=sorted(splits[0]),development_groups=sorted(splits[1]),
                    test_groups=sorted(splits[2]),input_sha256=args.input_sha256,
                    stage_context='current-model rollout on training tasks; never gold resets',log=logs,
                    score_semantics='uncalibrated weighted-BCE score; thresholds selected on dev only')
    if args.benchmark:
        save_model(net,out/'benchmark_only',policy(),provenance)
        return dict(fold=fold,benchmark_only=True,parameters=sum(p.numel() for p in net.parameters()),epochs=logs,
                    train_steps=sum(len(r['ids']) for r in train),dev_steps=sum(len(r['ids']) for r in dev),
                    estimated_15_epoch_seconds=np.median([l['seconds'] for l in logs])*15,
                    cuda_peak_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None)
    trials=[]
    for threshold in (.25,.4,.55):
        for margin,delay in ((0.,0),(.15,2)):
            config=policy(threshold,margin,delay)
            metrics=aggregate(predict(net,dev,vectors,config),labels)
            trials.append(dict(policy=config,metrics=metrics))
    immediate=max((r for r in trials if r['policy']['max_delay']==0),key=lambda r:r['metrics']['tol1_D0']['f1'])
    # Delayed candidate must compete at a common D1 deadline, not only eventual F1.
    confirmed=max((r for r in trials if r['policy']['max_delay']==2),key=lambda r:r['metrics']['tol1_D1']['f1'])
    from .boundary_model import BoundaryModel
    def flat(rs):
        rows=[]; ys=[]
        for r in rs:
            gold=set(r['boundaries'])
            for row in r['baseline_rows'][1:]: rows.append(row);ys.append(int(row['idx'] in gold))
        return rows,ys
    x,y=flat(train); baseline=BoundaryModel(window=4,text=False,C=1.).fit(x,y)
    def base_events(rs,threshold):
        result={}
        for r in rs:
            prob=baseline.probabilities(r['baseline_rows'][1:]) if len(r['ids'])>1 else []
            result[r['traj_id']]=[dict(boundary_step_id=i,detected_at_step_id=i) for i,p in zip(r['ids'][1:],prob) if p>=threshold]
        return result
    baseline.threshold=max((.1,.15,.2,.225,.25,.3,.4,.5,.6),
        key=lambda t:aggregate(base_events(dev,t),labels)['tol1_D0']['f1'])
    import joblib
    joblib.dump(baseline,out/'baseline.joblib')
    result=dict(fold=fold,parameters=sum(p.numel() for p in net.parameters()),training=provenance,development_trials=trials,
                baseline_threshold=baseline.threshold,comparisons={})
    preds={'lr':base_events(test,baseline.threshold)}
    for name,chosen in (('immediate',immediate),('confidence',confirmed)):
        save_model(net,out/name,chosen['policy'],provenance)
        preds[name]=predict(net,test,vectors,chosen['policy'])
    for name,events in preds.items():
        result['comparisons'][name]=aggregate(events,labels)
        delays=[e['detected_at_step_id']-e['boundary_step_id'] for es in events.values() for e in es]
        result['comparisons'][name]['confirmation_delay_steps']=dict(n=len(delays),
            mean=float(np.mean(delays)) if delays else None,max=max(delays) if delays else None,
            histogram={str(d):delays.count(d) for d in (0,1,2)})
    (out/'predictions.json').write_text(json.dumps(preds),encoding='utf-8')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--fold',default='0',help='0..4 or all; default development pilot only')
    ap.add_argument('--epochs',type=int,default=15);ap.add_argument('--hidden',type=int,default=128)
    ap.add_argument('--device',default='cpu');ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--seed',type=int,default=41);ap.add_argument('--benchmark',action='store_true')
    ap.add_argument('--resume',action='store_true',help='Continue last completed epoch; configuration must match')
    a=ap.parse_args();torch.set_num_threads(a.threads)
    if a.epochs<1: raise ValueError('epochs must be positive')
    root=Path(a.data);out=Path(a.out)
    if out.exists() and not a.resume: raise ValueError('Output exists; use --resume or a new directory')
    out.mkdir(parents=True,exist_ok=True)
    audit=json.loads((root/'audit.json').read_text(encoding='utf-8'))
    if audit['failures'] or audit['version']!=VERSION: raise ValueError('Invalid prepared dataset')
    record_name='records.json' if (root/'records.json').exists() else 'records.json.gz'
    if record_name.endswith('.gz'):
        with gzip.open(root/record_name,'rt',encoding='utf-8') as fh: records=json.load(fh)
    else: records=json.loads((root/record_name).read_text(encoding='utf-8'))
    vectors=dict(np.load(root/'vectors.npz',allow_pickle=False))
    a.input_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('audit.json',record_name,'vectors.npz')}
    a.torch_version=torch.__version__
    a.code_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                   for name in ('temporal_boundary.py','train_temporal.py','boundary_features.py','eval_online.py')}
    protocol={k:v for k,v in vars(a).items() if k!='resume'}
    protocol_path=out/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding='utf-8'))!=protocol:
            raise ValueError('Resume configuration/data/code mismatch; use a new directory')
    elif a.resume and any(out.iterdir()):
        raise ValueError('Missing protocol in nonempty output')
    atomic_json(protocol_path,protocol)
    folds=range(5) if a.fold=='all' else [int(a.fold)]
    results=[]
    for fold in folds:
        if fold not in range(5): raise ValueError('Invalid fold')
        folder=out/f'fold{fold}';folder.mkdir(exist_ok=a.resume)
        if a.resume and (folder/'report.json').exists():
            results.append(json.loads((folder/'report.json').read_text(encoding='utf-8')))
            print(f'fold={fold} already complete',flush=True)
            continue
        result=train_fold(a,records,vectors,fold,folder);results.append(result)
        atomic_json(folder/'report.json',result)
    report=dict(scope='Previously inspected SWE task-group evaluation, not pristine test',folds=results,
                benchmark_only=a.benchmark)
    if not a.benchmark:
        labels={r['traj_id']:dict(boundaries=r['boundaries']) for r in records}
        combined={name:{} for name in ('lr','immediate','confidence')}
        for fold in folds:
            p=json.loads((out/f'fold{fold}'/'predictions.json').read_text())
            for name in combined:combined[name].update(p[name])
        report['aggregate']={name:aggregate(p,labels) for name,p in combined.items()}
    atomic_json(out/'report.json',report)
    print('REPORT='+str(out/'report.json'))


if __name__=='__main__':main()
