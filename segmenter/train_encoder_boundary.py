"""Task-group nested CV for cached causal semantic representations and heads."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import joblib
from safetensors.torch import save_file
from torch import nn
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from .eval_online import aggregate
from .train_boundary import THRESHOLDS, bootstrap_difference, quick_f1


def load_data(data, cache):
    records = json.loads((data/'records.json').read_text(encoding='utf-8'))
    labels = json.loads((data/'labels.json').read_text(encoding='utf-8'))
    features = {r['traj_id']:r['rows'] for r in map(json.loads,(data/'features.jsonl').read_text(encoding='utf-8').splitlines())}
    index = json.loads((cache/'index.json').read_text())
    vectors = np.load(cache/'embeddings.npy')
    texts = json.loads((cache/'packed_texts.json').read_text(encoding='utf-8'))
    lookup = {(r['traj_id'],r['idx']):i for i,r in enumerate(index)}
    assert len(lookup)==len(index)
    struct, semantic, y, meta, packed = [],[],[],[],[]
    for r in records:
        gold = set(labels[r['traj_id']]['boundaries'])
        for j,row in enumerate(features[r['traj_id']]):
            if j==0:
                continue
            assert row['idx']==r['ids'][j]
            history, mask, words = [],[],[]
            for lag in range(4):
                if j-lag>=0:
                    position = lookup[(r['traj_id'],r['ids'][j-lag])]
                    history.append(vectors[position]); mask.append(1.); words.append(texts[position])
                else:
                    history.append(np.zeros(vectors.shape[1],dtype=np.float32)); mask.append(0.)
            semantic.append(np.concatenate(history+[np.abs(history[0]-history[1]),np.array(mask,dtype=np.float32)]))
            struct.append(row['features']); packed.append(' '.join(words))
            y.append(int(row['idx'] in gold))
            meta.append(dict(traj_id=r['traj_id'],idx=row['idx'],fold=r['fold']))
    return records,labels,struct,np.asarray(semantic,dtype=np.float32),np.array(y),meta,packed


def to_events(meta, probabilities, threshold, tids):
    result = {tid:[] for tid in tids}
    for r,p in zip(meta,probabilities):
        if p>=threshold:
            result[r['traj_id']].append(dict(boundary_step_id=r['idx'],detected_at_step_id=r['idx']))
    return result


def select_threshold(meta,p,labels,tids):
    values = [(quick_f1(to_events(meta,p,t, tids),labels),t) for t in THRESHOLDS]
    return max(values,key=lambda x:(x[0],x[1]))


class Head(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,1))
    def forward(self,x):
        return self.net(x).squeeze(-1)


def fit_mlp(x,y,xdev,dev_meta,labels,dev_tids,seed=41):
    torch.manual_seed(seed)
    model = Head(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    tx,ty,dx = torch.from_numpy(x),torch.from_numpy(y.astype(np.float32)),torch.from_numpy(xdev)
    generator = torch.Generator().manual_seed(seed)
    best, stale, curve = None,0,[]
    for epoch in range(20):
        model.train()
        order = torch.randperm(len(tx),generator=generator)
        for ids in order.split(256):
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(tx[ids]),ty[ids])
            loss.backward(); opt.step()
        model.eval()
        with torch.inference_mode():
            p = torch.sigmoid(model(dx)).numpy()
        metric,threshold = select_threshold(dev_meta,p,labels,dev_tids)
        curve.append(dict(epoch=epoch+1,dev_D0=metric,threshold=threshold))
        if best is None or metric>best[0]:
            best=(metric,threshold,copy.deepcopy(model.state_dict()),epoch+1); stale=0
        else:
            stale+=1
        if stale>=2:
            break
    model.load_state_dict(best[2]); model.eval()
    return model,best[0],best[1],curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',default='eval/boundary_supervised/data_fixed')
    ap.add_argument('--cache',default='eval/boundary_encoder/minilm_headtail')
    ap.add_argument('--out',default='eval/boundary_encoder/minilm_cv')
    ap.add_argument('--seed',type=int,default=41)
    a = ap.parse_args()
    data,cache,out = Path(a.data),Path(a.cache),Path(a.out)
    if out.exists():
        raise SystemExit('Output exists')
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    records,labels,struct,semantic,y,meta,packed = load_data(data,cache)
    names = ['struct_lr','struct_mlp','newtext_tfidf','semantic_lr','fusion_lr','fusion_mlp']
    predictions = {k:{} for k in names+['selected']}
    fold_ids = np.array([r['fold'] for r in meta])
    folds=[]
    for fold in range(5):
        start=time.perf_counter()
        devfold=(fold+1)%5
        tr=np.where((fold_ids!=fold)&(fold_ids!=devfold))[0]
        dv=np.where(fold_ids==devfold)[0]; te=np.where(fold_ids==fold)[0]
        dvmeta=[meta[i] for i in dv]; temeta=[meta[i] for i in te]
        devtids=[r['traj_id'] for r in records if r['fold']==devfold]
        testtids=[r['traj_id'] for r in records if r['fold']==fold]
        vectorizer=DictVectorizer(dtype=np.float32)
        scaler=MaxAbsScaler()
        train_s=scaler.fit_transform(vectorizer.fit_transform([struct[i] for i in tr])).toarray()
        dev_s=scaler.transform(vectorizer.transform([struct[i] for i in dv])).toarray()
        test_s=scaler.transform(vectorizer.transform([struct[i] for i in te])).toarray()
        trials=[]; best=None
        for name in names:
            tv=None
            if name.startswith('struct'):
                x,dx,ex=train_s,dev_s,test_s
            elif name=='semantic_lr':
                x,dx,ex=semantic[tr],semantic[dv],semantic[te]
            elif name=='newtext_tfidf':
                tv=TfidfVectorizer(max_features=16000,min_df=3,ngram_range=(1,2),sublinear_tf=True,dtype=np.float32)
                x=hstack([csr_matrix(train_s),tv.fit_transform([packed[i] for i in tr])]).tocsr()
                dx=hstack([csr_matrix(dev_s),tv.transform([packed[i] for i in dv])]).tocsr()
                ex=hstack([csr_matrix(test_s),tv.transform([packed[i] for i in te])]).tocsr()
            else:
                x=np.concatenate([train_s,semantic[tr]],axis=1)
                dx=np.concatenate([dev_s,semantic[dv]],axis=1)
                ex=np.concatenate([test_s,semantic[te]],axis=1)
            if name.endswith('mlp'):
                model,metric,threshold,curve=fit_mlp(x,y[tr],dx,dvmeta,labels,devtids,a.seed)
                with torch.inference_mode():
                    p=torch.sigmoid(model(torch.from_numpy(ex))).numpy()
            else:
                model=LogisticRegression(C=1.,solver='liblinear',max_iter=500,random_state=a.seed).fit(x,y[tr])
                metric,threshold=select_threshold(dvmeta,model.predict_proba(dx)[:,1],labels,devtids)
                p=model.predict_proba(ex)[:,1]; curve=[]
            pred=to_events(temeta,p,threshold,testtids)
            predictions[name].update(pred)
            trial=dict(name=name,dev_D0=metric,threshold=threshold,curve=curve)
            trials.append(trial)
            artifact=out/f'fold_{fold}_{name}'
            joblib.dump(dict(vectorizer=vectorizer,scaler=scaler,text_vectorizer=tv),str(artifact)+'.preprocess.joblib')
            if name.endswith('mlp'):
                save_file(model.state_dict(),str(artifact)+'.safetensors')
            else:
                joblib.dump(model,str(artifact)+'.joblib')
            Path(str(artifact)+'.json').write_text(json.dumps(dict(name=name,threshold=threshold,
                input_dimension=x.shape[1],cache=str(cache),seed=a.seed)),encoding='utf-8')
            # Selection never examines the outer-test result.
            if best is None or metric>best['dev_D0']:
                best=dict(trial); bestpred=pred
            print(f'fold {fold} {name} dev D0={metric:.4f}',flush=True)
        predictions['selected'].update(bestpred)
        folds.append(dict(fold=fold,selected=best,trials=trials,seconds=time.perf_counter()-start))
        (out/f'fold_{fold}.json').write_text(json.dumps(folds[-1],indent=2),encoding='utf-8')
    old=json.loads(Path('eval/boundary_supervised/experiment/predictions.json').read_text())['supervised']
    summaries={name:aggregate(p,labels) for name,p in predictions.items()}
    groups={}
    for source in sorted({r['source'] for r in records}):
        tids={r['traj_id'] for r in records if r['source']==source}
        groups[source]={name:aggregate({t:p[t] for t in tids},labels) for name,p in predictions.items()}
    report=dict(folds=folds,methods=summaries,by_source=groups,seed=a.seed,
                baseline=aggregate(old,labels),
                D0_delta_CI_vs_previous=bootstrap_difference(records,predictions['selected'],old,labels),
                eventual_delta_CI_vs_previous=bootstrap_difference(records,predictions['selected'],old,labels,delay=None),
                cache_audit=json.loads((cache/'audit.json').read_text()),
                raw_text_audit_sha256=hashlib.sha256((data/'records.json').read_bytes()).hexdigest(),
                scope='frozen pretrained encoder; same 5 task-group outer folds; model/threshold/epoch selected on inner development fold; external set untouched')
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'predictions.json').write_text(json.dumps(predictions),encoding='utf-8')
    print(json.dumps({k:{m:round(v[m]['f1'],4) for m in ('tol0_eventual','tol1_eventual','tol1_D0')} for k,v in summaries.items()},indent=2))


if __name__=='__main__':
    main()
