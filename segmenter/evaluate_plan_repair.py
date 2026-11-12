"""One fixed, zero-delay A/B probe of restored public plan-state information."""
import argparse
import json
from pathlib import Path
from .boundary_model import BoundaryModel,joblib
from .plan_features import enrich
from .train_boundary import THRESHOLDS,events_for,quick_f1,bootstrap_difference
from .eval_online import aggregate


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='eval/boundary_plan_repair')
    a=ap.parse_args();out=Path(a.out)
    if out.exists():raise SystemExit('Output exists')
    out.mkdir(parents=True)
    data=Path('eval/boundary_supervised/data_fixed')
    records=json.loads((data/'records.json').read_text(encoding='utf-8'))
    labels=json.loads((data/'labels.json').read_text(encoding='utf-8'))
    payloads={}
    for x in map(json.loads,Path('eval/plan_payload_audit/payloads.jsonl').read_text(encoding='utf-8').splitlines()):
        payloads.setdefault(x['traj_id'],{})[x['idx']]=x['payload']
    features={x['traj_id']:enrich(x['rows'],payloads.get(x['traj_id'],{}))
              for x in map(json.loads,(data/'features.jsonl').read_text(encoding='utf-8').splitlines())}
    protocol=dict(window=4,text=False,C=1.,seed=41,delay=0,
        input='legacy structure plus four-step public plan status/count deltas; no titles, item IDs or thought as features',
        selection='only development D0 threshold; original five task-group folds',
        scope='one exploratory input repair probe prompted by error audit; not independent confirmation or deployment; external tasks unused')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    predictions={};folds=[]
    for fold in range(5):
        tr=[r for r in records if r['fold'] not in (fold,(fold+1)%5)]
        dv=[r for r in records if r['fold']==(fold+1)%5];te=[r for r in records if r['fold']==fold]
        assert not ({r['group'] for r in tr}&{r['group'] for r in dv+te})
        x=[row for r in tr for row in features[r['traj_id']][1:]]
        y=[int(row['idx'] in labels[r['traj_id']]['boundaries']) for r in tr for row in features[r['traj_id']][1:]]
        model=BoundaryModel(window=4,text=False,C=1.).fit(x,y)
        def probabilities(rs):
            return {r['traj_id']:(model.probabilities(features[r['traj_id']][1:]) if len(features[r['traj_id']])>1 else []) for r in rs}
        dp=probabilities(dv)
        metric,threshold=max((quick_f1(events_for(dv,dp,t),labels),t) for t in THRESHOLDS)
        model.threshold=threshold
        # This artifact requires the opt-in plan feature builder, not the existing
        # online loader. Wrap it to prevent accidental use as a legacy model.
        joblib.dump(dict(model=model,feature_schema='public_plan_state_v1',requires_plan_payloads=True),out/f'fold_{fold}.joblib')
        predictions.update(events_for(te,probabilities(te),threshold))
        folds.append(dict(fold=fold,threshold=threshold,dev_D0=metric))
        print(f'fold {fold} dev D0={metric:.4f}',flush=True)
    baseline=json.loads(Path('eval/boundary_confirmation/predictions.json').read_text())['confirm0']
    report=dict(protocol=protocol,folds=folds,metrics=aggregate(predictions,labels),baseline=aggregate(baseline,labels),
        by_source={s:aggregate({r['traj_id']:predictions[r['traj_id']] for r in records if r['source']==s},labels) for s in sorted({r['source'] for r in records})},
        D0_delta_CI=bootstrap_difference(records,predictions,baseline,labels),
        eventual_delta_CI=bootstrap_difference(records,predictions,baseline,labels,delay=None))
    (out/'predictions.json').write_text(json.dumps(predictions),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report['metrics'][k]['f1'] for k in ('tol0_eventual','tol1_eventual','tol1_D0')}),flush=True)


if __name__=='__main__':main()
