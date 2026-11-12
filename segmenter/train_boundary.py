"""Nested task-grouped cross-validation; test folds never select thresholds."""
import argparse
import hashlib
import json
import platform
import statistics
import time
from collections import Counter
from pathlib import Path

from .boundary_model import BoundaryModel, joblib
import numpy as np
import sklearn
from .eval_online import aggregate, counts, window_diff
from .eval_segmenter import prf


CONFIGS = [dict(window=1,text=False,C=1.), dict(window=4,text=False,C=1.),
           dict(window=4,text=True,C=.1), dict(window=4,text=True,C=1.)]
THRESHOLDS = [.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.75,.80]


def events_for(records, probabilities, threshold):
    return {r['traj_id']: [dict(boundary_step_id=i, detected_at_step_id=i)
                          for i,p in zip(r['ids'][1:], probabilities[r['traj_id']]) if p >= threshold]
            for r in records}


def quick_f1(items, labels):
    c = [0,0,0]
    for t, events in items.items():
        x = counts(events,labels[t]['boundaries'],tol=1,delay=0)
        c = [a+b for a,b in zip(c,x)]
    return prf(*c)[2]


def bootstrap_difference(records, predictions, baseline, labels, seed=41, delay=0):
    groups = {}
    for r in records:
        t = r['traj_id']
        pair = np.array([counts(predictions[t],labels[t]['boundaries'],1,delay),
                         counts(baseline[t],labels[t]['boundaries'],1,delay)],dtype=int)
        groups[r['group']] = groups.get(r['group'],np.zeros((2,3),dtype=int)) + pair
    arrays = np.array(list(groups.values()))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(1000):
        c = arrays[rng.integers(0,len(arrays),len(arrays))].sum(axis=0)
        diffs.append((prf(*c[0])[2]-prf(*c[1])[2])*100)
    return dict(unit='percentage_points',samples=1000,group='task/content connected component',
                ci95=np.quantile(diffs,[.025,.975]).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',default='eval/boundary_supervised/data_fixed')
    ap.add_argument('--legacy-data',default='eval/boundary_supervised/data')
    ap.add_argument('--out',default='eval/boundary_supervised/experiment')
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    if out.exists():
        raise SystemExit('Output exists; use a new directory')
    out.mkdir(parents=True)
    records = json.loads((data/'records.json').read_text(encoding='utf-8'))
    labels = json.loads((data/'labels.json').read_text(encoding='utf-8'))
    features = {r['traj_id']:r['rows'] for r in map(json.loads,(data/'features.jsonl').read_text(encoding='utf-8').splitlines())}
    def flatten(rs):
        xs, ys = [], []
        for r in rs:
            gold = set(labels[r['traj_id']]['boundaries'])
            for row in features[r['traj_id']][1:]:
                xs.append(row); ys.append(int(row['idx'] in gold))
        return xs,ys
    def predict(model,rs):
        xs,_ = flatten(rs)
        ps = model.probabilities(xs)
        result, pos = {}, 0
        for r in rs:
            n = len(r['ids'])-1
            result[r['traj_id']] = ps[pos:pos+n]
            pos += n
        return result
    predictions, fixed, folds, windows = {}, {}, [], {}
    for fold in range(5):
        dev_fold = (fold+1)%5
        test = [r for r in records if r['fold']==fold]
        dev = [r for r in records if r['fold']==dev_fold]
        train = [r for r in records if r['fold'] not in (fold,dev_fold)]
        assert not ({r['group'] for r in train} & {r['group'] for r in dev+test})
        x,y = flatten(train)
        trials, best = [], None
        started = time.perf_counter()
        for config in CONFIGS:
            model = BoundaryModel(**config).fit(x,y)
            p = predict(model,dev)
            scored = [(quick_f1(events_for(dev,p,threshold),labels),threshold) for threshold in THRESHOLDS]
            metric, threshold = max(scored,key=lambda z:(z[0],z[1]))
            trials.append(dict(config=config,threshold=threshold,dev_timely_f1=metric))
            if best is None or metric > best[0]:
                model.threshold = threshold
                best = metric, model
        model = best[1]
        pred = events_for(test,predict(model,test),model.threshold)
        predictions.update(pred)
        joblib.dump(model,out/f'fold_{fold}.joblib')
        # Training-only duration baseline: median gold stage length, no test count/length oracle.
        lens = [s['end_step_id']-s['start_step_id']+1 for r in train for s in labels[r['traj_id']]['stages']]
        interval = max(1,round(statistics.median(lens)))
        window = max(1,round(statistics.fmean(lens)/2))
        for r in test:
            fixed[r['traj_id']] = [dict(boundary_step_id=i,detected_at_step_id=i) for i in r['ids'][interval::interval]]
            windows[r['traj_id']] = window
        entry = dict(fold=fold,dev_fold=dev_fold,train_n=len(train),dev_n=len(dev),test_n=len(test),
                     selected=model.config,threshold=model.threshold,trials=trials,
                     test=aggregate(pred,labels),train_interval=interval,seconds=time.perf_counter()-started)
        folds.append(entry)
        (out/f'fold_{fold}.json').write_text(json.dumps(entry,indent=2),encoding='utf-8')
        print(f"fold {fold}: selected={model.config}, threshold={model.threshold}, test timely F1={entry['test']['tol1_D0']['f1']:.4f}",flush=True)
    rule = {r['traj_id']:r['rules_online']['phase'] for r in records}
    node = {r['traj_id']:r['rules_online']['node'] for r in records}
    final_rule = {r['traj_id']:[dict(boundary_step_id=i,detected_at_step_id=i) for i in r['rules_final']['phase']] for r in records}
    methods = dict(supervised=predictions,phase_fixed_online=rule,node_online=node,fixed_interval=fixed)
    old_path = Path(a.legacy_data)/'records.json'
    if old_path.exists():
        old = {r['traj_id']:r for r in json.loads(old_path.read_text(encoding='utf-8'))}
        if set(old) != {r['traj_id'] for r in records}:
            raise ValueError('Legacy and corrected baseline samples differ')
        methods['phase_legacy_online'] = {t:r['rules_online']['phase'] for t,r in old.items()}
    summaries = {name:aggregate(p,labels) for name,p in methods.items()}
    groups = {}
    for key in ('source','has_thought'):
        for value in sorted({r[key] for r in records},key=str):
            tids = {r['traj_id'] for r in records if r[key]==value}
            groups[f'{key}={value}'] = dict(n=len(tids),methods={name:aggregate({t:e for t,e in p.items() if t in tids},labels) for name,p in methods.items()})
    for name,p in methods.items():
        ds = [window_diff([e['boundary_step_id'] for e in p[r['traj_id']]],labels[r['traj_id']]['boundaries'],r['ids'],windows[r['traj_id']]) for r in records]
        summaries[name]['window_diff'] = statistics.fmean(d for d in ds if d is not None)
        summaries[name]['stage_count_ratio'] = (sum(map(len,p.values()))+len(records))/(sum(len(labels[t]['boundaries']) for t in p)+len(records))
    ci = bootstrap_difference(records,predictions,rule,labels)
    delta = (summaries['supervised']['tol1_D0']['f1']-summaries['phase_fixed_online']['tol1_D0']['f1'])*100
    report = dict(n_traj=len(records),n_steps=sum(r['n_steps'] for r in records),folds=folds,methods=summaries,
                  by_group=groups,paired_delta_points=delta,paired_bootstrap=ci,
                  phase_fixed_final_offline=aggregate(final_rule,labels)['tol1_eventual'],
                  suggested_five_point_gate=delta>=5 and ci['ci95'][0]>0,
                  scope='5 outer task-group folds; inner held-out dev selects config and threshold; previously inspected corpus',
                  timing='D0 is after current action AND observation, not pre-action',
                  versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__),
                  dataset_audit_sha256=hashlib.sha256((data/'audit.json').read_bytes()).hexdigest())
    # Deployment artifact uses all development data; its in-sample score is NOT reported.
    choice = Counter(json.dumps(f['selected'],sort_keys=True) for f in folds).most_common(1)[0][0]
    config = json.loads(choice)
    deployment = BoundaryModel(**config).fit(*flatten(records))
    deployment.threshold = statistics.median(f['threshold'] for f in folds if f['selected']==config)
    joblib.dump(deployment,out/'candidate.joblib')
    report['candidate'] = dict(config=config,threshold=deployment.threshold,trained_on_all_development=True,
                               default_enabled=False,not_independently_evaluated=True)
    (out/'predictions.json').write_text(json.dumps(methods),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(n=len(records),delta_points=delta,ci=ci,gate=report['suggested_five_point_gate']),indent=2))


if __name__ == '__main__':
    main()
