"""Fixed SWE folds: error audit and one causal tool-transition interaction."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import warnings

from segmenter.boundary_model import BoundaryModel
from segmenter.eval_online import aggregate, matched_pairs
from segmenter.train_boundary import THRESHOLDS, quick_f1, events_for, bootstrap_difference


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def transition(row):
    f = row['features']
    return str(f.get('lag1:tool', 'START')) + ' -> ' + str(f['lag0:tool'])


def augment(row):
    result = deepcopy(row)
    result['features']['lag0:tool_transition'] = transition(row)
    return result


def audit(predictions, labels, features):
    groups = defaultdict(Counter)
    examples = defaultdict(list)
    for tid, events in predictions.items():
        gold = labels[tid]['boundaries']
        pairs = matched_pairs(events, gold, tol=1, delay=0)
        matched_pred, matched_gold = {i for i, j in pairs}, {j for i, j in pairs}
        lookup = {r['idx']: r for r in features[tid]}
        for i, event in enumerate(events):
            idx = event['boundary_step_id']
            kind = 'TP' if i in matched_pred else 'FP'
            key = transition(lookup[idx])
            groups[key][kind] += 1
            if kind == 'FP' and len(examples[key]) < 3:
                examples[key].append(dict(traj_id=tid, idx=idx, kind=kind,
                    nearest_gold=min((abs(idx-g) for g in gold), default=None)))
        for j, idx in enumerate(gold):
            if j not in matched_gold:
                key = transition(lookup[idx])
                groups[key]['FN'] += 1
                if len(examples[key]) < 6:
                    examples[key].append(dict(traj_id=tid, idx=idx, kind='FN'))
    return dict(groups=[dict(transition=k, **dict(v), examples=examples[k])
                        for k, v in sorted(groups.items(), key=lambda x: x[1]['FP']+x[1]['FN'], reverse=True)],
                note='Maximum one-to-one tol1 D0 assignment; FP/FN categories depend on matching. These are historical development errors, not blind evidence.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--out', default='pilot/swe_transition_v1')
    args = parser.parse_args()
    base = Path('eval/boundary_supervised/data_fixed')
    records = [r for r in read(base / 'records.json') if r['source'] == 'SWE']
    labels = read(base / 'labels.json')
    features = {r['traj_id']: r['rows'] for r in map(json.loads, (base / 'features.jsonl').read_text(encoding='utf-8').splitlines())}
    baseline = read('pilot/swe_only_boundary_v1/predictions.json')['swe_train_swe_dev']
    diagnostics = audit(baseline, labels, features)
    if args.audit_only:
        print(json.dumps(diagnostics['groups'][:12], indent=2))
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(change='One categorical feature: previous tool class -> current tool class',
        training='SWE only; original 5 task/content group folds; train 3/dev next/test heldout',
        model=dict(window=4,text=False,C=1.), thresholds=THRESHOLDS,
        gate='Both tol1 D0 F1 and precision improve; task-group paired CI lower >0; no default replacement without new-task validation',
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (base/'records.json',base/'labels.json',base/'features.jsonl')})
    (out/'protocol.json').write_text(json.dumps(plan,indent=2))
    (out/'errors_before.json').write_text(json.dumps(diagnostics,indent=2))
    changed = {r['traj_id']:[augment(x) for x in features[r['traj_id']]] for r in records}
    predictions, folds = {}, []
    from sklearn.exceptions import ConvergenceWarning
    for fold in range(5):
        train=[r for r in records if r['fold'] not in (fold,(fold+1)%5)]
        dev=[r for r in records if r['fold']==(fold+1)%5]
        test=[r for r in records if r['fold']==fold]
        groups=[{r['group'] for r in rs} for rs in (train,dev,test)]
        assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        def flatten(rs):
            rows, y = [], []
            for r in rs:
                assert [x['idx'] for x in changed[r['traj_id']]] == r['ids']
                for x in changed[r['traj_id']][1:]:
                    rows.append(x); y.append(int(x['idx'] in labels[r['traj_id']]['boundaries']))
            return rows, y
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model=BoundaryModel(window=4,text=False,C=1.).fit(*flatten(train))
        def probabilities(rs):
            return {r['traj_id']:model.probabilities(changed[r['traj_id']][1:]) for r in rs}
        dp=probabilities(dev)
        score, threshold=max((quick_f1(events_for(dev,dp,t),labels),t) for t in THRESHOLDS)
        model.threshold=threshold
        events=events_for(test, probabilities(test), threshold)
        predictions.update(events)
        folds.append(dict(fold=fold, threshold=threshold, dev_f1=score, test_metrics=aggregate(events,labels)))
        print('Completed fold',fold,flush=True)
    assert set(predictions)==set(baseline)
    metrics={k:aggregate(v,labels) for k,v in [('baseline',baseline),('transition',predictions)]}
    report=dict(metrics=metrics,folds=folds,paired_interval=bootstrap_difference(records,predictions,baseline,labels),
        delta_points=100*(metrics['transition']['tol1_D0']['f1']-metrics['baseline']['tol1_D0']['f1']),
        limitations=['Historical development corpus, not pristine testing.',
                    'One feature family; no claim that purpose or semantic stage is recovered.',
                    'No runtime/model replacement; transformed training input requires matching inference support.'])
    for name,value in [('report',report),('predictions',predictions),('errors_after',audit(predictions,labels,features))]:
        (out/(name+'.json')).write_text(json.dumps(value,indent=2))
    print(json.dumps(dict(delta_points=report['delta_points'],paired_interval=report['paired_interval'],
                         metrics={k:v['tol1_D0'] for k,v in metrics.items()}),indent=2))


if __name__ == '__main__':
    main()
