"""Paired cached baseline / prefix evidence on six previously audited cases."""
import json
from pathlib import Path
from segmenter.relation_evidence import RelationEvidence


def main():
    source=Path('pilot/swe_full_prefix_review_v1')
    packets=json.loads((source/'prefix_packets.json').read_text(encoding='utf-8'))
    keys={r['case_id']:r for r in json.loads((source/'answer_key.json').read_text(encoding='utf-8'))}
    cached={r['traj_id']:r['rows'] for r in map(json.loads,Path('eval/boundary_supervised/data_fixed/features.jsonl').read_text(encoding='utf-8').splitlines())}
    # Preserve the frozen v1 evidence artifact when the builder changes.
    out=Path('pilot/swe_relation_input_v2');out.mkdir(exist_ok=True)
    paired=[]
    for case in packets:
        builder=RelationEvidence(case['initial_user_message'])
        for step in case['prefix']:
            evidence=builder.feed(step)
        # Key is only used to retrieve identical cached baseline. No gold enters builder.
        rows=cached[keys[case['case_id']]['traj_id']]
        baseline=next(r for r in rows if r['idx']==case['target_step'])
        paired.append(dict(case_id=case['case_id'],target_step=case['target_step'],
                           baseline_structure=baseline['features'],relation_evidence=evidence))
    (out/'paired_inputs.json').write_text(json.dumps(paired,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([dict(case_id=r['case_id'],relations=r['relation_evidence']['relation_candidates'],
        failures=[(e['step_id'],e['kind']) for e in r['relation_evidence']['recent_failure_candidates']],
        tension=r['relation_evidence']['declaration_evidence_tension_candidate']['present']) for r in paired],indent=2))


if __name__=='__main__':main()
