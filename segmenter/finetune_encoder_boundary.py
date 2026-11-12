"""One-layer ModernBERT adaptation with a frozen, label-free prefix cache.

Only the first 21 immutable pretrained layers are cached. The final layer and
pooling are recomputed after every update, including historical step vectors.
Training and checkpoint selection use the existing task-separated folds.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MaxAbsScaler

from .step_encoder import StepEncoder
from .train_encoder_boundary import Head, select_threshold, to_events
from .eval_online import aggregate
from .train_boundary import bootstrap_difference


def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temp.replace(path)


def prepare(a):
    cache,out=Path(a.cache),Path(a.prefix)
    audit=json.loads((cache/'audit.json').read_text())
    assert audit['model']=='modernbert' and audit['frozen_pretrained']
    tokens=json.loads((cache/'tokens.json').read_text())
    lengths=np.array(list(map(len,tokens)),dtype=np.int64)
    offsets=np.concatenate([[0],np.cumsum(lengths)])
    shape=(int(offsets[-1]),768)
    out.mkdir(parents=True,exist_ok=True)
    progress=out/'progress.json'
    signature=dict(revision=audit['revision'],text_sha256=audit['text_sha256'],
                   policy=audit['policy'],shape=list(shape),split_before_layer=21,
                   tokens_sha256=hashlib.sha256((cache/'tokens.json').read_bytes()).hexdigest())
    if progress.exists():
        state=json.loads(progress.read_text())
        assert state['signature']==signature
        start=state['next_row']
        if state.get('complete'):
            print('Frozen prefix already complete',flush=True); return
        arr=np.lib.format.open_memmap(out/'hidden.npy',mode='r+')
    else:
        start=0
        arr=np.lib.format.open_memmap(out/'hidden.npy',mode='w+',dtype=np.float16,shape=shape)
        np.save(out/'offsets.npy',offsets)
        state=dict(signature=signature,next_row=0,complete=False,min_cosine=1.)
        write_json(progress,state)
    enc=StepEncoder(audit['model_path'],a.device,'float16' if a.device=='cuda' else 'float32','sdpa')
    assert len(enc.model.layers)==22
    captured=[]
    def capture(module,args,kwargs):
        captured.append(args[0].detach().cpu().numpy().astype(np.float16))
    hook=enc.model.layers[-1].register_forward_pre_hook(capture,with_kwargs=True)
    expected=np.load(cache/'embeddings.npy',mmap_mode='r')
    began=time.perf_counter()
    for i in range(start,len(tokens),a.batch_size):
        stop=min(i+a.batch_size,len(tokens))
        actual=enc.embed(tokens[i:stop])
        hidden=captured.pop()
        cosine=float(np.min(np.sum(actual*expected[i:stop],axis=1)))
        assert cosine>.999, ('Frozen cache drift',cosine)
        for j in range(i,stop):
            arr[offsets[j]:offsets[j+1]]=hidden[j-i,:lengths[j]]
        arr.flush()
        state.update(next_row=stop,min_cosine=min(state['min_cosine'],cosine),complete=stop==len(tokens))
        write_json(progress,state)
        if (i-start)%(a.batch_size*20)==0:
            print(f'prefix {stop}/{len(tokens)}; elapsed={time.perf_counter()-began:.1f}s',flush=True)
    hook.remove()
    print(json.dumps(state),flush=True)


class Tail(nn.Module):
    def __init__(self,base,struct_dim):
        super().__init__()
        self.layer=copy.deepcopy(base.layers[-1]).float()
        self.norm=copy.deepcopy(base.final_norm).float()
        self.head=Head(struct_dim+5*768+4)
        self.mask_builder=base._update_attention_mask

    def encode(self,hidden,mask):
        global_mask,local_mask=self.mask_builder(mask,output_attentions=False)
        position=torch.arange(hidden.shape[1],device=hidden.device).unsqueeze(0)
        h=self.layer(hidden,attention_mask=global_mask,sliding_window_mask=local_mask,
                     position_ids=position,output_attentions=False)[0]
        h=self.norm(h).float()
        pooled=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
        return nn.functional.normalize(pooled,dim=1)

    def classify(self,history,mask,structure):
        values=history*mask.unsqueeze(-1)
        return self.head(torch.cat([structure,values.flatten(1),
                                   (values[:,0]-values[:,1]).abs(),mask],dim=1))


def train(a):
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    data,cache,prefix,out=map(Path,(a.data,a.cache,a.prefix,a.out))
    if out.exists(): raise SystemExit('Output exists; use a fresh output directory')
    state=json.loads((prefix/'progress.json').read_text())
    assert state['complete']
    audit=json.loads((cache/'audit.json').read_text())
    assert state['signature']['revision']==audit['revision']
    assert state['signature']['text_sha256']==audit['text_sha256']
    assert state['signature']['tokens_sha256']==hashlib.sha256((cache/'tokens.json').read_bytes()).hexdigest()
    records=json.loads((data/'records.json').read_text(encoding='utf-8'))
    labels=json.loads((data/'labels.json').read_text(encoding='utf-8'))
    features={t['traj_id']:t['rows'] for t in map(json.loads,(data/'features.jsonl').read_text(encoding='utf-8').splitlines())}
    index=json.loads((cache/'index.json').read_text())
    lookup={(r['traj_id'],r['idx']):i for i,r in enumerate(index)}
    offsets=np.load(prefix/'offsets.npy')
    hidden=np.load(prefix/'hidden.npy',mmap_mode='r')
    history,struct,y,meta=[],[],[],[]
    for r in records:
        assert [row['idx'] for row in features[r['traj_id']]]==r['ids']
        for j,row in enumerate(features[r['traj_id']]):
            if not j: continue
            history.append([lookup[(r['traj_id'],r['ids'][j-lag])] if j>=lag else -1 for lag in range(4)])
            struct.append(row['features']);y.append(int(row['idx'] in labels[r['traj_id']]['boundaries']))
            meta.append(dict(traj_id=r['traj_id'],idx=row['idx'],fold=r['fold']))
    history=np.asarray(history); y=np.asarray(y,dtype=np.float32)
    folds=np.asarray([r['fold'] for r in meta])
    out.mkdir(parents=True)
    base=StepEncoder(audit['model_path'],'cpu','float32','sdpa').model
    all_predictions={}; reports=[]
    def token_batch(ids):
        lengths=offsets[ids+1]-offsets[ids]
        h=np.zeros((len(ids),int(lengths.max()),768),dtype=np.float32)
        m=np.zeros(h.shape[:2],dtype=np.int64)
        for j,idx in enumerate(ids):
            h[j,:lengths[j]]=hidden[offsets[idx]:offsets[idx+1]];m[j,:lengths[j]]=1
        return torch.from_numpy(h).to(a.device),torch.from_numpy(m).to(a.device)
    for fold in ([a.fold] if a.fold is not None else range(5)):
        torch.manual_seed(a.seed)
        dev=(fold+1)%5
        train_ids=np.where((folds!=fold)&(folds!=dev))[0]
        dev_ids=np.where(folds==dev)[0];test_ids=np.where(folds==fold)[0]
        vectorizer=DictVectorizer(dtype=np.float32)
        scaler=MaxAbsScaler()
        scaler.fit(vectorizer.fit_transform([struct[i] for i in train_ids]))
        sx=scaler.transform(vectorizer.transform(struct)).toarray()
        model=Tail(base,sx.shape[1]).to(a.device)
        opt=torch.optim.AdamW([dict(params=list(model.layer.parameters())+list(model.norm.parameters()),lr=1e-5),
                              dict(params=model.head.parameters(),lr=1e-3)],weight_decay=.01)
        amp=a.device=='cuda'
        grad_scaler=torch.amp.GradScaler('cuda',enabled=amp)
        def predict(ids):
            model.eval()
            unique=np.unique(history[ids]);unique=unique[unique>=0]
            vectors=np.zeros((len(index),768),dtype=np.float32)
            with torch.inference_mode():
                for k in range(0,len(unique),64):
                    rows=unique[k:k+64];h,m=token_batch(rows)
                    with torch.autocast(a.device,dtype=torch.float16,enabled=amp):
                        vectors[rows]=model.encode(h,m).float().cpu().numpy()
                probabilities=[]
                for batch in np.array_split(ids,max(1,(len(ids)+255)//256)):
                    hm=history[batch]>=0
                    v=torch.from_numpy(vectors[np.maximum(history[batch],0)]).to(a.device)
                    logits=model.classify(v,torch.from_numpy(hm.astype(np.float32)).to(a.device),
                                          torch.from_numpy(sx[batch]).to(a.device))
                    probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            return np.asarray(probabilities)
        best=None;stale=0;curve=[]
        rng=np.random.default_rng(a.seed)
        for epoch in range(a.epochs):
            began=time.perf_counter();model.train()
            shuffled=rng.permutation(train_ids); losses=[]
            for k in range(0,len(shuffled),a.train_batch_size):
                batch=shuffled[k:k+a.train_batch_size]
                hm=history[batch]>=0
                unique,inverse=np.unique(np.maximum(history[batch],0),return_inverse=True)
                h,m=token_batch(unique)
                opt.zero_grad()
                with torch.autocast(a.device,dtype=torch.float16,enabled=amp):
                    v=model.encode(h,m)[torch.from_numpy(inverse.reshape(len(batch),4)).to(a.device)]
                    logits=model.classify(v,torch.from_numpy(hm.astype(np.float32)).to(a.device),
                                          torch.from_numpy(sx[batch]).to(a.device))
                    loss=nn.functional.binary_cross_entropy_with_logits(logits,torch.from_numpy(y[batch]).to(a.device))
                assert torch.isfinite(loss), 'Nonfinite training loss'
                grad_scaler.scale(loss).backward();grad_scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                grad_scaler.step(opt);grad_scaler.update();losses.append(float(loss.detach()))
                if k%(a.train_batch_size*100)==0:
                    print(f'fold {fold} epoch {epoch+1} samples {k}/{len(shuffled)} elapsed {time.perf_counter()-began:.1f}s',flush=True)
                if epoch==0 and len(losses)==(a.probe_batches or 8):
                    seconds=time.perf_counter()-began
                    estimate=seconds/(k+len(batch))*len(shuffled)
                    write_json(out/'resource_probe.json',dict(fold=fold,batches=len(losses),samples=k+len(batch),
                        seconds=seconds,estimated_training_seconds_per_epoch=estimate,
                        upper_schedule_train_only_hours=estimate*5*5/3600,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated() if amp else None,
                        finite_loss=bool(np.isfinite(losses).all()),
                        note='Small-batch measured estimate; excludes validation, prefix construction and checkpoint I/O; no quality conclusion'))
                    print('Resource probe complete: '+str(out/'resource_probe.json'),flush=True)
                    if a.probe_batches:
                        return
            dp=predict(dev_ids)
            dev_meta=[meta[i] for i in dev_ids];dtids=[r['traj_id'] for r in records if r['fold']==dev]
            metric,threshold=select_threshold(dev_meta,dp,labels,dtids)
            curve.append(dict(epoch=epoch+1,dev_D0=metric,threshold=threshold,loss=float(np.mean(losses)),seconds=time.perf_counter()-began))
            print(json.dumps(curve[-1]),flush=True)
            write_json(out/f'fold_{fold}_curve.json',curve)
            if best is None or metric>best['dev_D0']:
                best=dict(curve[-1]);best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
                save_file(best_weights,str(out/f'fold_{fold}.safetensors'))
            else: stale+=1
            if stale>=2: break
        model.load_state_dict(best_weights)
        tp=predict(test_ids);tids=[r['traj_id'] for r in records if r['fold']==fold]
        predictions=to_events([meta[i] for i in test_ids],tp,best['threshold'],tids)
        all_predictions.update(predictions)
        joblib.dump(dict(vectorizer=vectorizer,scaler=scaler),out/f'fold_{fold}_preprocess.joblib')
        reports.append(dict(fold=fold,selected=best,curve=curve,test=aggregate(predictions,labels)))
        sources={source:aggregate({r['traj_id']:all_predictions[r['traj_id']] for r in records
                 if r['source']==source and r['traj_id'] in all_predictions},labels)
                 for source in sorted({r['source'] for r in records})}
        write_json(out/'predictions.json',all_predictions)
        write_json(out/'summary.json',dict(folds=reports,metrics=aggregate(all_predictions,labels),seed=a.seed,
                   complete_five_fold=len(reports)==5,by_source=sources,encoder_layers_trained=[21],cache_signature=state['signature'],
                   scope='last layer and final norm tuned only on training tasks; causal four-step fusion head; development checkpoint and threshold selection; no external data'))
        del model,opt
        if amp: torch.cuda.empty_cache()
    if len(reports)==5:
        old=json.loads(Path('eval/boundary_supervised/experiment/predictions.json').read_text())['supervised']
        report=json.loads((out/'summary.json').read_text())
        report['D0_delta_CI_vs_previous']=bootstrap_difference(records,all_predictions,old,labels)
        report['eventual_delta_CI_vs_previous']=bootstrap_difference(records,all_predictions,old,labels,delay=None)
        structural=json.loads(Path('eval/boundary_encoder/minilm_cv/predictions.json').read_text())['struct_mlp']
        report['eventual_delta_CI_vs_struct_mlp']=bootstrap_difference(records,all_predictions,structural,labels,delay=None)
        report['D0_delta_CI_vs_struct_mlp']=bootstrap_difference(records,all_predictions,structural,labels)
        baseline=aggregate(old,labels)
        report['baseline']=baseline
        delta=report['metrics']['tol1_eventual']['f1']-baseline['tol1_eventual']['f1']
        report['quality_screen']=dict(eventual_gain_percentage_points=delta*100,
            at_least_three_points=delta>=.03,
            eventual_bootstrap_lower_positive=report['eventual_delta_CI_vs_previous']['ci95'][0]>0,
            semantic_gain_over_struct_mlp_supported=report['eventual_delta_CI_vs_struct_mlp']['ci95'][0]>0,
            D0_not_lower=report['metrics']['tol1_D0']['f1']>=baseline['tol1_D0']['f1'],
            deployment_approved=False,
            note='Quality screen only; source stability, repeated seeds and serving costs still require review')
        write_json(out/'summary.json',report)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['prepare','train'])
    ap.add_argument('--cache',default='eval/boundary_encoder/modernbert_headtail')
    ap.add_argument('--prefix',default='eval/boundary_encoder/modernbert_prefix')
    ap.add_argument('--data',default='eval/boundary_supervised/data_fixed')
    ap.add_argument('--out',default='eval/boundary_encoder/modernbert_finetune')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--train-batch-size',type=int,default=16)
    ap.add_argument('--epochs',type=int,default=5)
    ap.add_argument('--fold',type=int,choices=range(5))
    ap.add_argument('--seed',type=int,default=41)
    ap.add_argument('--probe-batches',type=int,default=0)
    a=ap.parse_args()
    torch.set_num_threads(4)
    if a.mode=='prepare': prepare(a)
    else: train(a)


if __name__=='__main__': main()
