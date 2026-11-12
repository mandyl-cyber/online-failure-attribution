"""Evaluate the frozen candidate once on task-disjoint external trajectories."""
import json
from pathlib import Path
from .boundary_model import joblib
from .train_boundary import events_for, bootstrap_difference
from .eval_online import aggregate


def main():
    data = Path('eval/boundary_supervised/external_data')
    exp = Path('eval/boundary_supervised/experiment')
    target = exp/'external.json'
    if target.exists():
        raise SystemExit('External result exists; do not overwrite or tune on it')
    records = json.loads((data/'records.json').read_text(encoding='utf-8'))
    labels = json.loads((data/'labels.json').read_text(encoding='utf-8'))
    features = {r['traj_id']:r['rows'] for r in map(json.loads,(data/'features.jsonl').read_text(encoding='utf-8').splitlines())}
    model = joblib.load(exp/'candidate.joblib')
    probs = {r['traj_id']:model.probabilities(features[r['traj_id']][1:]) if len(r['ids'])>1 else [] for r in records}
    pred = events_for(records,probs,model.threshold)
    rule = {r['traj_id']:r['rules_online']['phase'] for r in records}
    final_rule = {r['traj_id']:[dict(boundary_step_id=i,detected_at_step_id=i) for i in r['rules_final']['phase']] for r in records}
    report = dict(n=len(records),steps=sum(r['n_steps'] for r in records),config=model.config,threshold=model.threshold,
                  supervised=aggregate(pred,labels),phase_online=aggregate(rule,labels),
                  phase_final_offline=aggregate(final_rule,labels)['tol1_eventual'],
                  bootstrap=bootstrap_difference(records,pred,rule,labels),by_source={})
    for source in sorted({r['source'] for r in records}):
        tids = {r['traj_id'] for r in records if r['source']==source}
        report['by_source'][source] = dict(n=len(tids),supervised=aggregate({t:pred[t] for t in tids},labels),
                                          phase=aggregate({t:rule[t] for t in tids},labels))
    report['scope'] = 'full task names absent from entire verified manifest; exact development trace duplicates excluded; no hyperparameter selection on this data; all available sources reported'
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    (exp/'external_predictions.json').write_text(json.dumps(pred),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('n','steps','config','threshold','bootstrap')},indent=2))


if __name__ == '__main__':
    main()
