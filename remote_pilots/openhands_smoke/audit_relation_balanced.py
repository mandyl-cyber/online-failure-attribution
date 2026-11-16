"""Validate packet provenance constraints and summarize diagnostic AI judgments."""
import json
from collections import Counter
from pathlib import Path
from segmenter.relation_evidence import RelationEvidence


def main():
    root=Path('pilot/swe_relation_balanced_v2')
    def read(name):return json.loads((root/name).read_text(encoding='utf-8'))
    packets=read('packets.json');key=read('answer_key.json');review=read('ai_review.json')
    assert len(packets)==len(key)==len(review['cases'])==16
    assert len({k['group'] for k in key})==16
    assert Counter(k['bucket'] for k in key)==dict(TP_matched=4,TN_far=4,FP_far=4,FN_far=4)
    lookup={r['case_id']:r for r in review['cases']}
    keys={r['case_id']:r for r in key}
    report=[]
    for r in packets:
        cid=r['case_id'];e=r['evidence'];idx=e['observed_through_step'];a=lookup[cid]
        assert a['reviewed'] and keys[cid]['alignment_verified']
        assert idx==keys[cid]['target_step']==e['current']['step_id']
        assert all(h['step_id']<idx for h in e['recent_action_history'])
        assert len(e['recent_action_history'])<=8
        assert all(x<=idx for x in a['evidence_steps'])
        assert all(link['from_step']<link['to_step']==idx for link in e['relation_candidates'])
        assert e['boundary_decision'] is None and e['purpose_relation']=='undetermined'
        report.append(dict(case_id=cid,bucket=keys[cid]['bucket'],suggestion=a['suggestion'],
                           history_count=len(e['recent_action_history']),
                           exit_zero_history=sum(h['exit_code']==0 for h in e['recent_action_history']),
                           relation_count=len(e['relation_candidates'])))
    old=json.loads(Path('pilot/swe_full_prefix_review_v1/prefix_packets.json').read_text(encoding='utf-8'))
    restored={}
    for case in old:
        if case['case_id'] not in ('case_02','case_06'):continue
        builder=RelationEvidence(case['initial_user_message'])
        for step in case['prefix']:evidence=builder.feed(step)
        wanted=15 if case['case_id']=='case_02' else 18
        assert wanted in [h['step_id'] for h in evidence['recent_action_history']]
        restored[case['case_id']]=wanted
    result=dict(cases=report,counts=dict(Counter(a['suggestion'] for a in review['cases'])),
                prior_pilot_missing_history_restored=restored,
                checks_passed=True,accuracy_estimate=None,
                caveat='Selected diagnostic sample; plausible AI judgments are not binary labels or performance measurements.')
    (root/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
