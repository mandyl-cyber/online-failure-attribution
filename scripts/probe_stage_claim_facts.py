"""Retrospective v4 fact/guard audit on frozen v3 responses; no model calls."""
import json
from collections import Counter
from pathlib import Path

from segmenter import stage_claim_summary as v3
from segmenter import stage_claim_summary_v4 as v4


INPUT = Path('pilot/adjacent_claim_pair_v1/requests.jsonl')
RETURN = Path('adjacent_claim_pair_v1')
OUT = Path('pilot/stage_claim_facts_v1_probe')


def main():
    rows = [v3.strict_json(line) for line in INPUT.read_text(encoding='utf-8').splitlines() if line.strip()]
    groups = json.loads(Path('pilot/adjacent_claim_pair_v1/groups.json').read_text(encoding='utf-8'))
    metadata = {job:(group['task'],arm) for group in groups for arm in ('original','adjacent') for job in group['arms'][arm]}
    counts = Counter()
    details = []
    for old in rows:
        new = v4.build(old['source_snapshot'],old['chunk_chars'])
        v4.verify_row(new)
        facts = new['observed_facts']
        counts.update(f['type'] for f in facts)
        path = RETURN/(old['job_id']+'.json')
        result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
        item = dict(task=metadata[old['job_id']][0],arm=metadata[old['job_id']][1],
                    old_job_id=old['job_id'],new_job_id=new['job_id'],
                    facts=[dict(type=f['type'],step_id=f['step_id'],test_count=f['test_count'],
                                exit_code=f['exit_code']) for f in facts],
                    old_status=result['status'] if result else 'excluded',
                    retrospective_v4_guard='not_scored',critical_missing=[])
        if result and result.get('raw_response'):
            try:
                response = v3.strict_json(result['raw_response']['choices'][0]['message']['content'])
                checked = v4.validate(response,new)
                item['retrospective_v4_guard']='passed'
                item['critical_missing']=checked['missing_critical_fact_ids']
            except Exception as exc:
                item['retrospective_v4_guard']='rejected'
                item['guard_error']=f'{type(exc).__name__}: {exc}'
        details.append(item)
    summary = dict(scope='Retrospective v3 outputs under a new guard; not v4 model quality or general accuracy.',
                   jobs=len(details), fact_types=dict(counts),
                   old_ready=sum(x['old_status']=='ready' for x in details),
                   guard_rejected_old_ready=sum(x['old_status']=='ready' and x['retrospective_v4_guard']=='rejected' for x in details),
                   guard_passed_old_ready=sum(x['old_status']=='ready' and x['retrospective_v4_guard']=='passed' for x in details),
                   missing_critical_after_guard=sum(bool(x['critical_missing']) for x in details),
                   details=details)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'report.json').write_bytes(v3.encoded(summary)+b'\n')
    print(json.dumps({k:summary[k] for k in ('jobs','fact_types','old_ready',
                                           'guard_rejected_old_ready','guard_passed_old_ready',
                                           'missing_critical_after_guard')},ensure_ascii=False,indent=2))
    print('Rejected:',[(x['task'],x['arm'],x.get('guard_error')) for x in details if x['old_status']=='ready' and x['retrospective_v4_guard']=='rejected'])


if __name__=='__main__':
    main()
