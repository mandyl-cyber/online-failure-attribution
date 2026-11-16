"""Three predeclared SWE arms, fixed task-group folds, no AI training labels."""
import hashlib
import json
import platform
import time
import warnings
from copy import deepcopy
from pathlib import Path

from segmenter.boundary_model import BoundaryModel
from segmenter.adapters.openhands import load_trajectory
from segmenter.relation_evidence import RelationEvidence
from segmenter.relation_features import extend_history, relation_features
from segmenter.train_boundary import THRESHOLDS, events_for, quick_f1, bootstrap_difference
from segmenter.eval_online import aggregate
import numpy as np
import sklearn


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    out=Path('pilot/swe_relation_ablation_v1');out.mkdir(exist_ok=False)
    base=Path('eval/boundary_supervised/data_fixed')
    records=[r for r in read(base/'records.json') if r['source']=='SWE']
    labels=read(base/'labels.json')
    cached={r['traj_id']:r['rows'] for r in map(json.loads,(base/'features.jsonl').read_text(encoding='utf-8').splitlines())}
    arms={'baseline4':4,'history9':9,'history9_relations':9}
    paths=[base/'records.json',base/'features.jsonl',base/'labels.json',Path(__file__),
           Path('segmenter/relation_features.py'),Path('segmenter/relation_evidence.py')]
    save(out/'protocol.json',dict(arms=arms,thresholds=THRESHOLDS,C=1,text=False,
        folds='Existing five task-group folds: test=f, dev=(f+1)%5, remaining train; SWE only.',
        features='history9 adds lag4..8 from prior cached lag0; relations add ten numeric features on same eight prior steps plus current. No plan/task text or AI labels.',
        matching='tol1 D0, after current observation; threshold maximizes dev F1, ties choose larger threshold.',
        gate='F1 and precision improve; task-group paired F1 CI lower >0. Historical development data only; no deployment.',
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    data={a:{} for a in arms};timings=[]
    for number,r in enumerate(records,1):
        tid=r['traj_id'];rows=cached[tid];path=Path(r['raw_source'])
        steps,_=load_trajectory(path.parent if r['format']=='completions' else path,
                                base=r['ids'][0],expect_steps=r['n_steps'])
        steps=steps[:r['n_steps']];h=hashlib.sha256()
        for s in steps:h.update(json.dumps([s.action,s.observation],ensure_ascii=False).encode())
        assert h.hexdigest()==r['content_sha256']
        assert [s.idx for s in steps]==r['ids']==[x['idx'] for x in rows]
        data['baseline4'][tid]=rows;long=[];relation=[];builder=RelationEvidence('')
        for i,s in enumerate(steps):
            started=time.perf_counter()
            e=builder.feed(dict(idx=s.idx,tool=s.tool,tool_class=s.tool_class,action=s.action,observation=s.observation))
            extra=relation_features(e)
            timings.append((time.perf_counter()-started)*1000)
            row=extend_history(rows,i);long.append(row)
            row=deepcopy(row);row['features'].update(extra);relation.append(row)
        data['history9'][tid]=long;data['history9_relations'][tid]=relation
        if number%50==0:print('Prepared',number,flush=True)
    predictions={a:{} for a in arms};folds=[]
    from sklearn.exceptions import ConvergenceWarning
    for fold in range(5):
        train=[r for r in records if r['fold'] not in (fold,(fold+1)%5)]
        dev=[r for r in records if r['fold']==(fold+1)%5]
        test=[r for r in records if r['fold']==fold]
        groups=[{r['group'] for r in rs} for rs in (train,dev,test)]
        assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        for arm,window in arms.items():
            def flatten(rs):
                x=[];y=[]
                for r in rs:
                    for row in data[arm][r['traj_id']][1:]:
                        x.append(row);y.append(int(row['idx'] in labels[r['traj_id']]['boundaries']))
                return x,y
            started=time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                model=BoundaryModel(window=window,text=False,C=1.).fit(*flatten(train))
            def probabilities(rs):
                return {r['traj_id']:model.probabilities(data[arm][r['traj_id']][1:]) for r in rs}
            dp=probabilities(dev)
            score,threshold=max((quick_f1(events_for(dev,dp,t),labels),t) for t in THRESHOLDS)
            pred=events_for(test,probabilities(test),threshold);predictions[arm].update(pred)
            folds.append(dict(fold=fold,arm=arm,threshold=threshold,dev_f1=score,
                              seconds=time.perf_counter()-started,test=aggregate(pred,labels)))
            print(fold,arm,threshold,flush=True)
    reference=read('pilot/swe_only_boundary_v1/predictions.json')['swe_train_swe_dev']
    assert predictions['baseline4']==reference,'Baseline must reproduce exactly'
    metrics={a:aggregate(p,labels) for a,p in predictions.items()}
    comparisons={}
    for a,b in [('history9','baseline4'),('history9_relations','history9'),('history9_relations','baseline4')]:
        comparisons[a+' minus '+b]=dict(delta_points=100*(metrics[a]['tol1_D0']['f1']-metrics[b]['tol1_D0']['f1']),
            bootstrap=bootstrap_difference(records,predictions[a],predictions[b],labels))
    report=dict(n_trajectories=len(records),n_steps=sum(r['n_steps'] for r in records),
        baseline_exactly_reproduced=True,metrics=metrics,comparisons=comparisons,folds=folds,
        evidence_preparation_ms=dict(n=len(timings),median=float(np.median(timings)),p99=float(np.quantile(timings,.99)),
            scope='Local prefix evidence+numeric extraction only; excludes IO, structure construction and model inference.'),
        versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__),
        limitations=['Repeatedly exposed development data; not independent test.',
                    'Bootstrap conditions on fitted models, not training uncertainty.',
                    'Failure memory exact-retry handling can reflect earlier context; counts restricted to recent eight actions.',
                    'No model deployment, no default change, no Terminal result.'])
    save(out/'predictions.json',predictions);save(out/'report.json',report)
    print(json.dumps(dict(metrics={a:m['tol1_D0'] for a,m in metrics.items()},comparisons=comparisons),indent=2))


if __name__=='__main__':main()
