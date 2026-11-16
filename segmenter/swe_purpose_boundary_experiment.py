"""Exploratory SWE purpose student and D0 boundary comparison using reviewed AI samples."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time

LABELS=['ACQUIRE','ENABLE','CONSTRUCT','OPERATE','CHECK','DELIVER','UNKNOWN']
THRESHOLDS=[x/100 for x in range(10,81,5)]


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_split(root,split,mapping,review,gold):
    from .purpose_student import feature_text
    result=[]
    with (root/split/'trajectories.jsonl').open(encoding='utf-8') as f:
        for line in f:
            traj=json.loads(line); alias=traj['trajectory_id']; info=mapping[alias]
            if info['split']!=split or len(traj['steps'])!=info['steps']:
                raise ValueError('Selection mismatch: '+alias)
            true_id=info['traj_id']; stages=gold[true_id]['stages']
            covered=[i for s in stages for i in range(s['start_step_id'],s['end_step_id']+1)]
            if covered!=[s['idx'] for s in traj['steps']]:
                raise ValueError('Boundary/step mismatch: '+alias)
            texts=[feature_text(traj['task'],traj['steps'][:j+1]) for j in range(len(traj['steps']))]
            result.append(dict(alias=alias,true_id=true_id,group=info['group'],task=traj['task'],
                               steps=traj['steps'],texts=texts,gold=gold[true_id]['boundaries'],
                               reviewed={int(k.rsplit('_s',1)[1]):v['final_label'] for k,v in review.items()
                                         if k.startswith(alias+'_s') and v['reviewed']}))
    return result


def fit_purpose(trajectories):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    texts=[]; labels=[]
    for t in trajectories:
        for step,text in zip(t['steps'],t['texts']):
            label=t['reviewed'].get(step['idx'])
            if label is not None: texts.append(text);labels.append(label)
    if len(set(labels))<2: raise ValueError('Too few reviewed purpose classes')
    vec=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=8000,sublinear_tf=True)
    model=LogisticRegression(C=1.,max_iter=500,class_weight='balanced')
    model.fit(vec.fit_transform(texts),labels)
    return vec,model


def predict_purpose(fitted,trajectories):
    vec,model=fitted; out={}
    for t in trajectories:
        probs=model.predict_proba(vec.transform(t['texts']))
        out[t['alias']]=[{label:float(row[list(model.classes_).index(label)]) if label in model.classes_ else 0.
                          for label in LABELS} for row in probs]
    return out


def crossfit_purpose(trajectories,n_splits=5):
    from sklearn.model_selection import GroupKFold
    groups=[t['group'] for t in trajectories]
    out={}; folds=[]
    for fold,(train,test) in enumerate(GroupKFold(n_splits=n_splits).split(trajectories,groups=groups)):
        trained=[trajectories[i] for i in train];held=[trajectories[i] for i in test]
        predicted=predict_purpose(fit_purpose(trained),held)
        out.update(predicted)
        folds.append(dict(fold=fold,train_groups=sorted({x['group'] for x in trained}),
                          test_groups=sorted({x['group'] for x in held}),
                          reviewed_train=sum(len(x['reviewed']) for x in trained)))
    if set(out)!={t['alias'] for t in trajectories}: raise AssertionError('Missing purpose OOF')
    return out,folds


def structural(t,j):
    row=t['steps'][j];prev=t['steps'][j-1] if j else {}
    action=str(row.get('action') or '').strip();obs=str(row.get('observation') or '')
    def verb(a):
        first=str(a or '').strip().split(maxsplit=1)
        return re.sub(r'[^a-z0-9_-]','',first[0].lower())[:28] if first else 'empty'
    since_edit=next((j-k for k in range(j,-1,-1) if t['steps'][k]['tool']=='edit'),j+1)
    return dict(tool=str(row['tool']),prev_tool=str(prev.get('tool') or 'START'),
                verb=verb(action),prev_verb=verb(prev.get('action')),
                tool_changed=int(row['tool']!=prev.get('tool')),
                action_len=math.log1p(len(action)),observation_len=math.log1p(len(obs)),
                prev_observation_len=math.log1p(len(str(prev.get('observation') or ''))),
                exit='none' if row['exit_code'] is None else ('ok' if row['exit_code']==0 else 'fail'),
                test_pass=int(bool(re.search(r'\b(passed|success|\bOK\b)\b',obs,re.I))),
                test_fail=int(bool(re.search(r'\b(failed|traceback|error)\b',obs,re.I))),
                since_edit=math.log1p(since_edit))


def rows_for(trajectories,purpose,augmented):
    x=[];y=[];meta=[]
    for t in trajectories:
        ps=purpose[t['alias']] if augmented else None
        for j in range(1,len(t['steps'])):
            row=structural(t,j)
            if augmented:
                now=ps[j];previous=ps[j-1]
                for label in LABELS:
                    row['purpose_'+label]=now[label]
                    row['purpose_delta_'+label]=now[label]-previous[label]
                row['purpose_same_top']=int(max(now,key=now.get)==max(previous,key=previous.get))
                row['purpose_max']=max(now.values())
            x.append(row);y.append(int(t['steps'][j]['idx'] in t['gold']))
            meta.append((t['alias'],t['steps'][j]['idx']))
    return x,y,meta


def fit_boundary(x,y):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    vec=DictVectorizer(sparse=True)
    clf=LogisticRegression(C=1.,max_iter=500,class_weight='balanced')
    clf.fit(vec.fit_transform(x),y)
    return vec,clf


def boundary_probs(model,x):
    vec,clf=model
    return clf.predict_proba(vec.transform(x))[:,list(clf.classes_).index(1)].tolist()


def nest_boundary_oof(train):
    from sklearn.model_selection import GroupKFold
    groups=[t['group'] for t in train]
    oof={'structure':{},'structure_purpose':{}}
    for fold,(tr,te) in enumerate(GroupKFold(n_splits=5).split(train,groups=groups)):
        fit=[train[i] for i in tr];test=[train[i] for i in te]
        inner,_=crossfit_purpose(fit,n_splits=4)
        held=predict_purpose(fit_purpose(fit),test)
        for mode in oof:
            augmented=mode=='structure_purpose'
            x,y,_=rows_for(fit,inner,augmented)
            model=fit_boundary(x,y)
            xx,_,meta=rows_for(test,held,augmented)
            for key,prob in zip(meta,boundary_probs(model,xx)):
                oof[mode][key]=prob
        print('nested fold',fold,'reviewed training',sum(len(x['reviewed']) for x in fit),flush=True)
    return oof


def events(trajectories,probabilities,threshold):
    return {t['alias']:[dict(boundary_step_id=s['idx'],detected_at_step_id=s['idx'])
                        for s in t['steps'][1:] if probabilities[(t['alias'],s['idx'])]>=threshold]
            for t in trajectories}


def metrics(trajectories,events_by_alias):
    from .eval_online import counts
    from .eval_segmenter import prf
    result={}
    for tol,delay,name in [(0,0,'exact_D0'),(1,0,'tol1_D0')]:
        c=[0,0,0]
        for t in trajectories:
            x=counts(events_by_alias[t['alias']],t['gold'],tol,delay)
            c=[a+b for a,b in zip(c,x)]
        result[name]=dict(counts=c,precision=prf(*c)[0],recall=prf(*c)[1],f1=prf(*c)[2])
    return result


def pick_threshold(train,probabilities):
    scored=[(metrics(train,events(train,probabilities,t))['tol1_D0']['f1'],t) for t in THRESHOLDS]
    return max(scored,key=lambda x:(x[0],x[1]))


def purpose_audit(student_pred,dev,audit_root):
    from sklearn.metrics import accuracy_score,f1_score
    ai=read(audit_root/'ai_labels.json')['labels']
    selection=read(audit_root/'selection.json')
    pairs=[]
    for item in selection['items']:
        alias,step=item['request_id'].rsplit('_s',1)
        probs=student_pred[alias][next(j for j,s in enumerate(next(t for t in dev if t['alias']==alias)['steps']) if s['idx']==int(step))]
        pairs.append(dict(request_id=item['request_id'],sampling=item['sampling'],stratum=item['stratum'],
                          reference=ai[item['request_id']]['label'],prediction=max(probs,key=probs.get)))
    probabilistic=[x for x in pairs if x['sampling']=='probability']
    strata=selection['stratum_sizes'];weighted=0.
    for label,n in strata.items():
        subset=[x for x in probabilistic if x['stratum']==label]
        weighted+=n*sum(x['prediction']==x['reference'] for x in subset)/len(subset)
    weighted/=sum(strata.values())
    return dict(n=len(pairs),unweighted_accuracy=float(accuracy_score([x['reference'] for x in pairs],[x['prediction'] for x in pairs])),
                probability_sample_weighted_accuracy=weighted,
                macro_f1=float(f1_score([x['reference'] for x in pairs],[x['prediction'] for x in pairs],
                                        labels=LABELS,average='macro',zero_division=0)),
                reference_counts=dict(Counter(x['reference'] for x in pairs)),
                prediction_counts=dict(Counter(x['prediction'] for x in pairs)),rows=pairs)


def main():
    import numpy as np
    import sklearn
    from .purpose_student import FEATURE_VERSION,PurposeStudent
    from .eval_online import counts
    from .eval_segmenter import prf
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,default=Path('remote_pilots/swe_purpose_v2'))
    p.add_argument('--selection',type=Path,default=Path('pilot/swe_purpose_continuous_v1/selection.json'))
    p.add_argument('--gold',type=Path,default=Path('eval/boundary_supervised/data_fixed/labels.json'))
    p.add_argument('--reviews',type=Path,default=Path('pilot/swe_purpose_qwen_v2'))
    p.add_argument('--out',type=Path,default=Path('pilot/swe_purpose_qwen_v2/experiment_ai200_v1'))
    a=p.parse_args()
    if a.out.exists(): raise SystemExit('Refusing overwrite: '+str(a.out))
    mapping={x['trajectory_id']:x for x in read(a.selection)};gold=read(a.gold)
    review={split:read(a.reviews/split/'review_pending'/'review_pending.json')['labels'] for split in ('train','dev')}
    train=load_split(a.package,'train',mapping,review['train'],gold)
    dev=load_split(a.package,'dev',mapping,review['dev'],gold)
    if {t['group'] for t in train}&{t['group'] for t in dev}: raise ValueError('Train/dev task group overlap')
    if sum(len(t['reviewed']) for t in train)!=130 or sum(len(t['reviewed']) for t in dev)!=70:
        raise ValueError('Expected frozen 130/70 reviewed AI sample')
    a.out.mkdir(parents=True)
    started=time.perf_counter()
    boundary_oof=nest_boundary_oof(train)
    thresholds={name:pick_threshold(train,values) for name,values in boundary_oof.items()}
    print('thresholds',thresholds,flush=True)
    train_purpose_oof,purpose_folds=crossfit_purpose(train)
    final_purpose=fit_purpose(train)
    dev_purpose=predict_purpose(final_purpose,dev)
    boundary_predictions={};boundary_metrics={};boundary_scores={}
    for name in ('structure','structure_purpose'):
        augmented=name=='structure_purpose'
        x,y,_=rows_for(train,train_purpose_oof,augmented)
        model=fit_boundary(x,y)
        xx,_,meta=rows_for(dev,dev_purpose,augmented)
        probs=dict(zip(meta,boundary_probs(model,xx)))
        boundary_scores[name]=[dict(trajectory_id=alias,step_id=step,probability=prob)
                               for (alias,step),prob in probs.items()]
        chosen=events(dev,probs,thresholds[name][1])
        boundary_predictions[name]=chosen
        boundary_metrics[name]=metrics(dev,chosen)
    # Existing online rule events are read only for a same-trajectory reference.
    records={r['traj_id']:r for r in read('eval/boundary_supervised/data_fixed/records.json')}
    rule={t['alias']:records[t['true_id']]['rules_online']['phase'] for t in dev}
    boundary_predictions['phase_rule']=rule
    boundary_metrics['phase_rule']=metrics(dev,rule)
    structure_sets={alias:{e['boundary_step_id'] for e in rows}
                    for alias,rows in boundary_predictions['structure'].items()}
    augmented_sets={alias:{e['boundary_step_id'] for e in rows}
                    for alias,rows in boundary_predictions['structure_purpose'].items()}
    added=sum(len(augmented_sets[a]-structure_sets[a]) for a in structure_sets)
    removed=sum(len(structure_sets[a]-augmented_sets[a]) for a in structure_sets)
    # Diagnostic only: selecting the dev prediction count is not a valid frozen threshold.
    equal_count=sum(map(len,structure_sets.values()))
    ranked=sorted(boundary_scores['structure_purpose'],
                  key=lambda x:(-x['probability'],x['trajectory_id'],x['step_id']))[:equal_count]
    equal_events={t['alias']:[] for t in dev}
    for x in ranked:
        equal_events[x['trajectory_id']].append(dict(boundary_step_id=x['step_id'],detected_at_step_id=x['step_id']))
    equal_count_metrics=metrics(dev,equal_events)
    reference=purpose_audit(dev_purpose,dev,a.reviews/'dev'/'codex_blind_audit_v1')
    vec,clf=final_purpose
    artifact=dict(feature_version=FEATURE_VERSION,classes=[str(x) for x in clf.classes_],
        vocabulary={k:int(v) for k,v in vec.vocabulary_.items()},idf=vec.idf_.tolist(),
        coef=clf.coef_.tolist(),intercept=clf.intercept_.tolist(),ngram_range=[3,5],sublinear_tf=True,
        label_source='130 Codex AI blind reviewed SWE train steps; not human gold',
        review_sha256=sha(a.reviews/'train'/'review_pending'/'review_pending.json'))
    model_path=a.out/'purpose_student.json.gz'
    with gzip.open(model_path,'wt',encoding='utf-8',compresslevel=9) as f:
        json.dump(artifact,f,ensure_ascii=False,separators=(',',':'))
    runtime=PurposeStudent.load(model_path)
    sample=dev[0];reference_probs=dev_purpose[sample['alias']][0]
    runtime_probs=runtime.probabilities_text(sample['texts'][0])
    error=max(abs(runtime_probs[k]-reference_probs[k]) for k in runtime_probs)
    if error>1e-9: raise AssertionError('Exported runtime differs: '+str(error))
    times=[]
    for t in dev:
        for text in t['texts']:
            begin=time.perf_counter();runtime.probabilities_text(text);times.append((time.perf_counter()-begin)*1000)
    # Paired task-group bootstrap on frozen dev predictions.
    rng=np.random.default_rng(41)
    groups=[]
    for t in dev:
        alias=t['alias']
        groups.append([counts(boundary_predictions[name][alias],t['gold'],1,0)
                       for name in ('structure_purpose','structure')])
    groups=np.asarray(groups,dtype=int);deltas=[]
    for _ in range(2000):
        sample_counts=groups[rng.integers(0,len(groups),len(groups))].sum(axis=0)
        deltas.append((prf(*sample_counts[0])[2]-prf(*sample_counts[1])[2])*100)
    delta=(boundary_metrics['structure_purpose']['tol1_D0']['f1']-
           boundary_metrics['structure']['tol1_D0']['f1'])*100
    report=dict(protocol='swe-purpose-reviewed-ai200-boundary-v1',
        scope='25 SWE train trajectories; 10 disjoint SWE dev trajectories; previously exposed corpus',
        reviewed_train=130,reviewed_dev=70,reviewer_kind='ai',human_accuracy_evaluated=False,
        purpose_train_classes=dict(Counter(label for t in train for label in t['reviewed'].values())),
        purpose_unlearned_classes=[x for x in LABELS if x not in clf.classes_],
        purpose_dev_audit={k:v for k,v in reference.items() if k!='rows'},
        threshold_selection={k:dict(train_nested_oof_f1=v[0],threshold=v[1]) for k,v in thresholds.items()},
        boundary_dev=boundary_metrics,boundary_delta_points=delta,
        added_cuts=added,removed_cuts=removed,
        posthoc_equal_count_diagnostic=dict(predictions=equal_count,metrics=equal_count_metrics,
            warning='Uses development-set prediction count; diagnostic only, not a deployable threshold.'),
        paired_group_bootstrap_ci95=np.quantile(deltas,[.025,.975]).tolist(),
        purpose_model_bytes=model_path.stat().st_size,
        inference_ms=dict(median=statistics.median(times),p95=float(np.quantile(times,.95)),n=len(times)),
        exported_probability_error=error,purpose_folds=purpose_folds,
        input_sha256=dict(selection=sha(a.selection),gold=sha(a.gold),
                          train_review=sha(a.reviews/'train'/'review_pending'/'review_pending.json'),
                          dev_review=sha(a.reviews/'dev'/'review_pending'/'review_pending.json')),
        seconds=round(time.perf_counter()-started,2),sklearn_version=sklearn.__version__,
        limitations=['AI-only sparse review; no human purpose accuracy.',
                     'Only four purpose classes were present in reviewed training sample.',
                     'Audit sample was stratified by first-pass Qwen labels and targeted CHECK; not random steps.',
                     'Train and dev trajectories were previously exposed in other experiments.',
                     'No reserved-set or genuinely new-task boundary test.',
                     'Student is exploratory and not enabled in the online default partitioner.'])
    save(a.out/'report.json',report)
    save(a.out/'purpose_dev_audit_rows.json',reference['rows'])
    save(a.out/'boundary_predictions.json',boundary_predictions)
    save(a.out/'boundary_scores.json',boundary_scores)
    save(a.out/'purpose_dev_predictions.json',dev_purpose)
    print(json.dumps(dict(purpose_audit=report['purpose_dev_audit'],boundary_dev=boundary_metrics,
                          delta_points=delta,ci95=report['paired_group_bootstrap_ci95'],
                          model_bytes=report['purpose_model_bytes'],seconds=report['seconds']),ensure_ascii=False))


if __name__=='__main__': main()
