"""Fixed 0/1/2-step lookahead experiment and descriptive error audit.

The algorithm really observes subsequent steps and records delayed emission.
Simply relaxing a score's deadline is reported separately, never as a new model.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from .boundary_model import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from .confirmation import confirmation_rows
from .eval_online import aggregate,counts,matched_pairs
from .eval_segmenter import prf
from .train_boundary import THRESHOLDS,bootstrap_difference


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def event_map(records,rows,probabilities,threshold):
    result={r['traj_id']:[] for r in records};pos=0
    for r in records:
        for x in rows[r['traj_id']]:
            p=float(probabilities[pos]);pos+=1
            if p>=threshold:
                result[r['traj_id']].append({k:x[k] for k in ('boundary_step_id','detected_at_step_id')})
    assert pos==len(probabilities)
    return result


def score_at(events,labels,delay):
    total=np.zeros(3,dtype=np.int64)
    for tid,ev in events.items(): total+=counts(ev,labels[tid]['boundaries'],1,delay)
    return prf(*total)[2]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='eval/boundary_supervised/data_fixed')
    ap.add_argument('--out',default='eval/boundary_confirmation')
    ap.add_argument('--audit-only',action='store_true')
    a=ap.parse_args();data,out=Path(a.data),Path(a.out)
    if out.exists() and not a.audit_only: raise SystemExit('Output exists; use a fresh directory')
    out.mkdir(parents=True,exist_ok=a.audit_only)
    records=json.loads((data/'records.json').read_text(encoding='utf-8'))
    labels=json.loads((data/'labels.json').read_text(encoding='utf-8'))
    features={t['traj_id']:t['rows'] for t in map(json.loads,(data/'features.jsonl').read_text(encoding='utf-8').splitlines())}
    if a.audit_only:
        audit(records,labels,features,json.loads((out/'predictions.json').read_text()),out)
        return
    for r in records:
        assert [x['idx'] for x in features[r['traj_id']]]==r['ids']
        assert all(b-a==1 for a,b in zip(r['ids'],r['ids'][1:])), 'Native IDs must be contiguous for step-delay scoring'
    protocol=dict(delays=[0,1,2],C=1.,window=4,text=False,seed=41,
        folds='original task/content-group folds; dev=(test+1)%5',
        threshold_selection='development tol1 F1 at the configured delay deadline; no outer-test tuning',
        tail_policy='no EOS flush; unconfirmed final d candidates remain in gold denominator',
        scope='offline causal replay of bounded confirmation; no online backend promotion; no external tasks')
    dump(out/'protocol.json',protocol)
    predictions={};fold_reports=[];prob_rows=[]
    for delay in (0,1,2):
        rows={tid:confirmation_rows(rs,delay) for tid,rs in features.items()}
        result={}
        for fold in range(5):
            started=time.perf_counter();devfold=(fold+1)%5
            tr=[r for r in records if r['fold'] not in (fold,devfold)]
            dv=[r for r in records if r['fold']==devfold];te=[r for r in records if r['fold']==fold]
            assert not ({r['group'] for r in tr}&{r['group'] for r in dv+te})
            assert not ({r['group'] for r in dv}&{r['group'] for r in te})
            def flatten(rs):
                return [x for r in rs for x in rows[r['traj_id']]]
            train=flatten(tr)
            y=[int(x['boundary_step_id'] in labels[r['traj_id']]['boundaries']) for r in tr for x in rows[r['traj_id']]]
            model=make_pipeline(DictVectorizer(),MaxAbsScaler(),
                LogisticRegression(C=1.,solver='liblinear',max_iter=500,random_state=41))
            model.fit([x['features'] for x in train],y)
            dp=model.predict_proba([x['features'] for x in flatten(dv)])[:,1]
            metric,threshold=max([(score_at(event_map(dv,rows,dp,t),labels,delay),t) for t in THRESHOLDS])
            tp=model.predict_proba([x['features'] for x in flatten(te)])[:,1]
            result.update(event_map(te,rows,tp,threshold))
            joblib.dump(dict(model=model,delay=delay,threshold=threshold,feature_version=1),out/f'delay{delay}_fold{fold}.joblib')
            pos=0
            for r in te:
                for x in rows[r['traj_id']]:
                    prob_rows.append(dict(traj_id=r['traj_id'],fold=fold,delay=delay,
                        boundary_step_id=x['boundary_step_id'],detected_at_step_id=x['detected_at_step_id'],
                        probability=float(tp[pos]),threshold=threshold));pos+=1
            fold_reports.append(dict(delay=delay,fold=fold,threshold=threshold,dev_f1=metric,seconds=time.perf_counter()-started))
            print(f'delay={delay} fold={fold} dev F1={metric:.4f}',flush=True)
        predictions[f'confirm{delay}']=result
    old=json.loads(Path('eval/boundary_supervised/experiment/predictions.json').read_text())['supervised']
    predictions['previous_lr']=old
    # Wait-only control: same D0 predictions, no new evidence, true delayed time.
    for delay in (1,2):
        predictions[f'wait_only{delay}']={r['traj_id']:[dict(boundary_step_id=e['boundary_step_id'],
            detected_at_step_id=e['detected_at_step_id']+delay) for e in predictions['confirm0'][r['traj_id']]
            if e['detected_at_step_id']+delay<=r['ids'][-1]] for r in records}
    report=dict(protocol=protocol,folds=fold_reports,metrics={k:aggregate(v,labels) for k,v in predictions.items()},
        by_source={s:{k:aggregate({r['traj_id']:v[r['traj_id']] for r in records if r['source']==s},labels)
                   for k,v in predictions.items()} for s in sorted({r['source'] for r in records})},
        tail_gold_unconfirmable={str(d):sum(b>r['ids'][-1]-d for r in records for b in labels[r['traj_id']]['boundaries']) for d in (0,1,2)},
        comparisons={str(d):dict(
            eventual_CI_vs_confirm0=bootstrap_difference(records,predictions[f'confirm{d}'],predictions['confirm0'],labels,delay=None),
            same_deadline_CI_vs_wait_only=bootstrap_difference(records,predictions[f'confirm{d}'],predictions[f'wait_only{d}'],labels,delay=d),
            same_deadline_CI_vs_immediate=bootstrap_difference(records,predictions[f'confirm{d}'],predictions['confirm0'],labels,delay=d)) for d in (1,2)})
    dump(out/'predictions.json',predictions);dump(out/'summary.json',report)
    (out/'probabilities.jsonl').write_text('\n'.join(json.dumps(x) for x in prob_rows)+'\n',encoding='utf-8')
    audit(records,labels,features,predictions,out)
    print(json.dumps({k:{m:round(v[m]['f1'],4) for m in ('tol0_eventual','tol1_eventual','tol1_D0','tol1_D1','tol1_D2')} for k,v in report['metrics'].items()},indent=2))


def audit(records,labels,features,predictions,out):
    raw={t['traj_id']:{r['idx']:r for r in t['rows']} for t in map(json.loads,Path('eval/boundary_encoder/text_v1/steps.jsonl').read_text(encoding='utf-8').splitlines())}
    cases=[];totals=Counter();transitions=Counter();review=[]
    for r in records:
        tid=r['traj_id'];gold=labels[tid]['boundaries'];ev=predictions['previous_lr'][tid]
        pairs=matched_pairs(ev,gold,1,0)
        hit_gold={j for i,j in pairs};hit_pred={i for i,j in pairs}
        recovered={d:{j for i,j in matched_pairs(predictions[f'confirm{d}'][tid],gold,1,d)} for d in (0,1,2)}
        lookup={x['idx']:x['features'] for x in features[tid]}
        for kind,positions in [('FN',[(j,b) for j,b in enumerate(gold) if j not in hit_gold]),
                               ('FP',[(None,e['boundary_step_id']) for i,e in enumerate(ev) if i not in hit_pred])]:
            for j,b in positions:
                f=lookup[b];transition=str(f.get('lag1:tool'))+' -> '+str(f.get('lag0:tool'))
                flags=dict(tool_changed=bool(f.get('lag0:tool_changed')),exit_failed=f.get('lag0:exit')=='fail',
                           test_signal=bool(f.get('lag0:test_pass') or f.get('lag0:test_fail')),
                           same_tool_next_two=all(lookup.get(b+k,{}).get('lag0:tool')==f.get('lag0:tool') for k in (1,2)))
                late_nearby=any(abs(e['boundary_step_id']-b)<=1 and e['detected_at_step_id']>b for e in ev) if kind=='FN' else False
                category=('late_nearby_prediction' if late_nearby else 'visible_tool_transition' if flags['tool_changed'] else 'same_tool')
                recovered_by=[d for d in (0,1,2) if j in recovered[d]] if j is not None else []
                item=dict(traj_id=tid,source=r['source'],fold=r['fold'],kind=kind,step_id=b,
                          transition=transition,proxy_category=category,flags=flags,recovered_by=recovered_by)
                cases.append(item);totals[f'{kind}:{category}']+=1;transitions[f'{kind}:{transition}']+=1
                # Fixed deterministic excerpts; retrospective evidence only.
                if (sum(x['kind']==kind and x['proxy_category']==category and x['source']==r['source'] for x in review)<3
                    and not any(x['kind']==kind and x['proxy_category']==category and x['traj_id']==tid for x in review)):
                    item=dict(item,context=[dict(step_id=i,tool=raw[tid][i]['tool'],
                        action=raw[tid][i]['action'][:1200],observation=raw[tid][i]['observation'][:1000])
                        for i in range(b-1,b+3) if i in raw[tid]])
                    review.append(item)
    (out/'errors.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in cases)+'\n',encoding='utf-8')
    dump(out/'review_samples.json',review)
    dump(out/'error_audit.json',dict(baseline='previous_lr, tol1_D0 one-to-one matching',counts=dict(totals),
        top_transitions=transitions.most_common(20),
        FN_recovery={str(d):sum(x['kind']=='FN' and d in x['recovered_by'] for x in cases) for d in (0,1,2)},
        by_source={s:dict(Counter(f"{x['kind']}:{x['proxy_category']}" for x in cases if x['source']==s)) for s in sorted({r['source'] for r in records})},
        limitations='Observable proxies only. A missed boundary does not prove evidence was insufficient; recovery after lookahead does not prove future evidence was necessary. Label ambiguity cannot be inferred from a single annotation.'))


if __name__=='__main__':main()
