"""Partition unmatched boundary events without assuming semantic errors."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from segmenter.eval_online import aggregate, matched_pairs


def decompose(events, gold, delay):
    pairs = matched_pairs(events, gold, tol=1, delay=delay)
    pi, gi = {i for i, j in pairs}, {j for i, j in pairs}
    fp, fn, details = Counter(), Counter(), []
    for i, e in enumerate(events):
        if i in pi:
            continue
        nearby = [j for j, g in enumerate(gold) if abs(e['boundary_step_id']-g)<=1]
        eligible = [j for j in nearby if e['detected_at_step_id']<=gold[j]+delay]
        # A maximum matching leaves no edge between two unmatched vertices.
        assert all(j in gi for j in eligible)
        kind = ('no_gold' if not gold else 'outside_tolerance' if not nearby else
                'competition' if eligible else 'timing_only')
        fp[kind] += 1
        details.append(dict(kind='FP_'+kind, event_index=i, **e,
                            nearby_gold=[gold[j] for j in nearby]))
    for j, g in enumerate(gold):
        if j in gi:
            continue
        nearby = [i for i, e in enumerate(events) if abs(e['boundary_step_id']-g)<=1]
        eligible = [i for i in nearby if events[i]['detected_at_step_id']<=g+delay]
        assert all(i in pi for i in eligible)
        kind = 'no_nearby_prediction' if not nearby else 'competition' if eligible else 'timing_only'
        fn[kind] += 1
        details.append(dict(kind='FN_'+kind, gold_boundary=g,
                            nearby_predictions=[events[i] for i in nearby]))
    offsets = Counter(str(events[i]['boundary_step_id']-gold[j]) for i, j in pairs)
    assert sum(fp.values()) == len(events)-len(pairs)
    assert sum(fn.values()) == len(gold)-len(pairs)
    return dict(tp=len(pairs), fp=dict(fp), fn=dict(fn), matched_position_offsets=dict(offsets), details=details)


def main():
    out=Path('pilot/swe_timing_audit_v1')
    out.mkdir(exist_ok=False)
    source=Path('pilot/swe_only_boundary_v1/predictions.json')
    candidate=Path('pilot/swe_transition_v1/predictions.json')
    base=Path('eval/boundary_supervised/data_fixed')
    labels=json.loads((base/'labels.json').read_text())
    records=json.loads((base/'records.json').read_text())
    original=json.loads(source.read_text())
    arms={**original,'transition':json.loads(candidate.read_text()),
          'rules':{r['traj_id']:r['rules_online']['phase'] for r in records if r['source']=='SWE'}}
    ids=set(arms['swe_train_swe_dev'])
    assert all(set(p)==ids for p in arms.values())
    report=dict(trajectories=len(ids), gold_boundaries=sum(len(labels[t]['boundaries']) for t in ids),
        policy='Fixed predictions and thresholds; tol=1; D0/D1 scored separately. Matching may reassign pairs. Competition is structural, not proof of redundant semantic stages.',
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (source,candidate,base/'labels.json',base/'records.json')},arms={})
    details=[]
    for arm,pred in arms.items():
        scores=aggregate(pred,labels)
        totals={}
        for delay in (0,1):
            tp=0; fp=Counter(); fn=Counter(); offsets=Counter()
            for tid in sorted(ids):
                row=decompose(pred[tid],labels[tid]['boundaries'],delay)
                tp+=row['tp']; fp.update(row['fp']); fn.update(row['fn']); offsets.update(row['matched_position_offsets'])
                if arm=='swe_train_swe_dev':
                    details.extend(dict(traj_id=tid,delay=delay,**x) for x in row['details'])
            counts=scores[f'tol1_D{delay}']['counts']
            assert [tp,sum(fp.values()),sum(fn.values())]==[counts[0],counts[1]-counts[0],counts[2]-counts[0]]
            totals[f'D{delay}']=dict(tp=tp,fp=dict(fp),fn=dict(fn),matched_position_offsets=dict(offsets))
        report['arms'][arm]=dict(scores={k:scores[k] for k in ('tol0_D0','tol1_D0','tol1_D1','tol1_D2','tol1_eventual')},
            decomposition=totals, net_additional_matches_D1=totals['D1']['tp']-totals['D0']['tp'],
            all_emissions_at_boundary=all(e['boundary_step_id']==e['detected_at_step_id'] for es in pred.values() for e in es))
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'swe_error_details.json').write_text(json.dumps(details,indent=2))
    print(json.dumps({k:dict(D0=v['scores']['tol1_D0'],D1=v['scores']['tol1_D1'],
                    decomposition=v['decomposition']) for k,v in report['arms'].items()},indent=2))


if __name__=='__main__':main()
