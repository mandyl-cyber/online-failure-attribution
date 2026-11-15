"""Task-group held-out pilot comparison. Gold is read only here, never by Qwen."""
import argparse
import json
from pathlib import Path

from .boundary_features import BoundaryFeatures
from .eval_online import aggregate, counts
from .eval_segmenter import prf
from .ledger import atomic, encoded, sha
from .purpose_probe import LABELS, trajectories, audited_mapping
from .rules import Rules
from .schema import Step, ToolClass
from .shell import classify_command


def reconstructed_step(row, rules):
    """Rebuild only fields recoverable from the package, not missing native metadata."""
    s = Step.from_dict(row)
    if s.tool == 'run':
        info = classify_command(s.action, rules)
        s.tool_class, s.touched_paths, s.written_paths = info.tool_class, info.paths, info.written
        s.attrs['cmd_classes'] = sorted(info.classes)
    elif s.tool == 'run_ipython':
        write = bool(rules.sh_ipython_write.search(s.action))
        s.tool_class = ToolClass.EXEC_WRITE if write else ToolClass.EXEC
        s.attrs['cmd_classes'] = ['state' if write else 'probe']
    elif s.tool in ('edit', 'read', 'write'):
        head = s.action.split('\n',1)[0].split()
        sub = head[0] if head else ''
        path = head[1] if len(head)>1 else ''
        read = s.tool == 'read' or sub in ('view','read')
        s.tool_class = ToolClass.READ if read else ToolClass.EDIT
        s.touched_paths = [path] if path else []
        s.written_paths = [] if read else s.touched_paths[:]
    else:
        s.tool_class = {'think':ToolClass.THINK, 'finish':ToolClass.FINISH,
                        'task_tracking':ToolClass.PLAN, 'message':ToolClass.MESSAGE}.get(s.tool,ToolClass.OTHER)
    return s


def feature_rows(t, responses):
    bf, rules, history, rows = BoundaryFeatures(4), Rules(), [], []
    previous = None
    for step in t['steps']:
        r = responses[f"{t['trajectory_id']}_s{step['idx']:05d}"]
        available = r['status'] == 'scored'
        purpose = r['purpose'] if available else 'MISSING'
        entry = dict(available=int(available), purpose=purpose)
        entry.update({f'p_{label}':float(r['scores'][i]) if available else 0.
                      for i,label in enumerate(LABELS)})
        entry['changed'] = int(previous is not None and previous not in ('MISSING','UNKNOWN')
                               and purpose not in ('MISSING','UNKNOWN') and previous != purpose)
        previous = purpose
        history.append(entry)
        pf, af = {}, {}
        for lag, e in enumerate(reversed(history[-4:])):
            pf.update({f'purpose_lag{lag}:{k}':v for k,v in e.items()})
            af[f'purpose_lag{lag}:available'] = e['available']
        sf = bf.feed(reconstructed_step(step,rules))['features']
        rows.append(dict(idx=step['idx'], structure=sf, purpose=pf, fusion=dict(sf,**pf),
                         availability=dict(sf,**af), changed=entry['changed']))
    return rows


def main():
    import numpy as np
    import sklearn
    import platform
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MaxAbsScaler
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probe', type=Path, default=Path('eval/purpose_probe_v1'))
    ap.add_argument('--data', type=Path, default=Path('eval/boundary_supervised/data_fixed'))
    ap.add_argument('--packed', type=Path, default=Path('eval/boundary_encoder/minilm_headtail'))
    ap.add_argument('--out', type=Path, default=Path('eval/purpose_probe_evaluation_v1'))
    args = ap.parse_args()
    if args.out.exists(): raise SystemExit('Use a new evaluation directory')
    ts = trajectories()
    records = json.loads((args.data/'records.json').read_text())
    labels = json.loads((args.data/'labels.json').read_text())
    mapping = audited_mapping(ts, records, args.packed)
    manifest = json.loads((args.probe/'manifest.json').read_text())
    responses = {}
    for t in ts:
        for j,s in enumerate(t['steps']):
            rid = f"{t['trajectory_id']}_s{s['idx']:05d}"
            r = json.loads((args.probe/'responses'/f'{rid}.json').read_text())
            packet = dict(trajectory_id=t['trajectory_id'],task=t['task'],current_step_id=s['idx'],steps=t['steps'][:j+1])
            if r['identity'] != manifest['identity'] or r['input_sha256'] != sha(encoded(packet)):
                raise ValueError('Input/config mismatch')
            if r['status'] not in ('scored','context_overflow','failed'):
                raise ValueError('Incomplete probe; finish or explicitly resolve pending jobs')
            responses[rid]=r
    matched = {m['pilot_id']:m['match'] for m in mapping if m['match'] is not None}
    features = {t['trajectory_id']:feature_rows(t,responses) for t in ts if t['trajectory_id'] in matched}
    folds = sorted({r['fold'] for r in matched.values()})
    if len(folds)<3: raise ValueError('Need >=3 distinct original folds')
    methods = ('structure','purpose','fusion','availability')
    predictions = {m:{} for m in methods}
    predictions['purpose_change'] = {}
    details = []
    def flatten(ids,method):
        x,y = [],[]
        for pid in ids:
            gold = set(labels[matched[pid]['traj_id']]['boundaries'])
            for row in features[pid][1:]:
                x.append(row[method]); y.append(int(row['idx'] in gold))
        return x,y
    def predict(model,ids,method,threshold):
        result={}
        for pid in ids:
            rows = features[pid][1:]
            probabilities = model.predict_proba([r[method] for r in rows])[:,1]
            result[matched[pid]['traj_id']] = [dict(boundary_step_id=r['idx'],detected_at_step_id=r['idx'])
                                                for r,p in zip(rows,probabilities) if p>=threshold]
        return result
    for j,fold in enumerate(folds):
        dev_fold=folds[(j+1)%len(folds)]
        test=[p for p,r in matched.items() if r['fold']==fold]
        dev=[p for p,r in matched.items() if r['fold']==dev_fold]
        train=[p for p,r in matched.items() if r['fold'] not in (fold,dev_fold)]
        groups=[{matched[p]['group'] for p in part} for part in (train,dev,test)]
        assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        for method in methods:
            model=make_pipeline(DictVectorizer(),MaxAbsScaler(),
                LogisticRegression(C=1,solver='liblinear',max_iter=1000,random_state=41))
            x,y=flatten(train,method)
            if len(set(y))!=2: raise ValueError('Training fold has only one class')
            model.fit(x,y)
            trials=[]
            for threshold in np.arange(.10,.801,.05):
                score=aggregate(predict(model,dev,method,threshold),labels)['tol1_D0']['f1']
                trials.append((score,float(threshold)))
            best,threshold=max(trials,key=lambda x:(x[0],x[1]))
            predictions[method].update(predict(model,test,method,threshold))
            details.append(dict(fold=fold,dev_fold=dev_fold,method=method,train=train,dev=dev,test=test,
                                threshold=threshold,dev_f1=best,trials=trials))
    for pid,r in matched.items():
        predictions['purpose_change'][r['traj_id']] = [dict(boundary_step_id=x['idx'],detected_at_step_id=x['idx'])
                                                       for x in features[pid][1:] if x['changed']]
    metrics={m:aggregate(p,labels) for m,p in predictions.items()}
    for m in metrics:
        hit,pred,gold=metrics[m]['tol1_D0']['counts']
        metrics[m].update(false_positives=pred-hit,false_negatives=gold-hit)
    def bootstrap(a,b):
        group_counts={}
        for r in matched.values():
            pair=np.array([counts(predictions[m][r['traj_id']],labels[r['traj_id']]['boundaries'],1,0)
                           for m in (a,b)])
            group_counts[r['group']]=group_counts.get(r['group'],np.zeros((2,3),dtype=int))+pair
        arr=np.array(list(group_counts.values()));rng=np.random.default_rng(41);ds=[]
        for _ in range(2000):
            c=arr[rng.integers(0,len(arr),len(arr))].sum(axis=0)
            ds.append(100*(prf(*c[0])[2]-prf(*c[1])[2]))
        return dict(delta_points=100*(metrics[a]['tol1_D0']['f1']-metrics[b]['tol1_D0']['f1']),
                    ci95_points=np.quantile(ds,[.025,.975]).tolist(),bootstrap_groups=len(arr),resamples=2000)
    comparison=bootstrap('fusion','structure')
    coverage={}
    for pid in features:
        rr=[responses[f'{pid}_s{x["idx"]:05d}'] for x in features[pid]]
        coverage[pid]=dict(steps=len(rr),scored=sum(x['status']=='scored' for x in rr),
                           source=matched[pid]['source'])
    report=dict(matched_trajectories=len(matched),steps=sum(len(x) for x in features.values()),
        excluded=[m['pilot_id'] for m in mapping if m['match'] is None],coverage=coverage,methods=metrics,
        comparison=comparison,fusion_vs_availability=bootstrap('fusion','availability'),
        positive_pilot_evidence=comparison['delta_points']>=1 and comparison['ci95_points'][0]>0,
        by_source={source:{m:aggregate({t:e for t,e in p.items() if t in
                        {r['traj_id'] for r in matched.values() if r['source']==source}},labels)
                          for m,p in predictions.items()} for source in {r['source'] for r in matched.values()}},
        limitations=['Previously studied development data; not blind holdout',
                     'SWE mapping uses independent bounded original step text, not an exact full-content hash',
                     'Reconstructed structural features, not original 461-trajectory baseline',
                     'Small and imbalanced original folds; unstable threshold estimates',
                     'D0 is information timing, not zero wall-clock latency',
                     'Label-score probe has no validated semantic accuracy'],
        data_sha256=sha((args.data/'labels.json').read_bytes()),probe_identity=manifest['identity'],
        packed_index_sha256=sha((args.packed/'index.json').read_bytes()),
        packed_text_sha256=sha((args.packed/'packed_texts.json').read_bytes()),
        versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__),
        evaluator_sha256=sha(Path(__file__).read_bytes()),
        bootstrap_note='Resamples held-out prediction counts by task group; does not refit models.')
    args.out.mkdir(parents=True)
    atomic(args.out/'mapping.json',mapping)
    atomic(args.out/'splits.json',details)
    atomic(args.out/'predictions.json',predictions)
    atomic(args.out/'report.json',report)
    print(json.dumps(dict(matched=len(matched),comparison=comparison,positive=report['positive_pilot_evidence']),indent=2))


if __name__ == '__main__': main()
