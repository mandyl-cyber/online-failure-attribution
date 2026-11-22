"""Re-pair unchanged stage requests between common emitted boundaries."""
import argparse
import hashlib
from pathlib import Path

from scripts.run_stage_claim_pipeline import read, save
from segmenter import stage_claim_summary as core


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('pilot/adaptive_seal_local_v2'))
    args = parser.parse_args()
    source = Path('pilot/adaptive_seal_claim_v1')
    returned = Path('adaptive_seal_claim_v1')
    parent_protocol = read(source/'protocol.json')
    rows = {r['job_id']:r for r in map(core.strict_json,(source/'requests.jsonl').read_text(encoding='utf-8').splitlines())}
    predictions = read(source/'predictions.json')
    preflight = read(returned/'preflight.json')
    windows, selected = [], {}
    for parent in read(source/'groups.json'):
        cuts = {arm:{e['boundary_step_id'] for e in predictions[parent['traj_id']][arm]['events']}
                for arm in ('immediate','adaptive')}
        common = sorted({parent['envelope'][0],parent['envelope'][1]+1} |
                        {b for b in cuts['immediate'] & cuts['adaptive'] if parent['envelope'][0]<b<=parent['envelope'][1]})
        for left,right in zip(common,common[1:]):
            if not any(left<b<right for b in cuts['immediate'] ^ cuts['adaptive']):
                continue
            window = dict(group_id=f'{parent["group_id"]}_s{left}_{right-1}', parent_group_id=parent['group_id'],
                          traj_id=parent['traj_id'],task=parent['task'],envelope=[left,right-1],arms={})
            source_records = {}
            for arm, ids in parent['arms'].items():
                members = [j for j in ids if left<=rows[j]['request']['stage']['start_step_id']
                           and rows[j]['request']['stage']['end_step_id']<right]
                coverage, records = [], {}
                for job_id in members:
                    row = rows[job_id]
                    core.verify_row(row)
                    stage = row['request']['stage']
                    coverage.extend(range(stage['start_step_id'],stage['end_step_id']+1))
                    for record in row['source_snapshot']['evidence']:
                        key = (record['source_step_id'],record['field'])
                        if key in records:
                            raise ValueError('Duplicate source record')
                        records[key] = record
                    if job_id in selected:
                        raise ValueError('Request appears in multiple local windows')
                    selected[job_id] = row
                if coverage != list(range(left,right)):
                    raise ValueError('Local window coverage gap')
                source_records[arm] = records
                window['arms'][arm] = members
            if source_records['immediate'] != source_records['adaptive']:
                raise ValueError('Paired source differs')
            windows.append(window)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'requests.jsonl').write_bytes(b''.join(core.encoded(r)+b'\n' for r in selected.values()))
    save(args.out/'groups.json',windows)
    candidates = read(source/'fact_review.json')
    save(args.out/'fact_review.json',dict(source_sha256=sha(source/'fact_review.json'),
        selection='Same source candidates, filtered to local disagreement windows',
        groups=[dict(group_id=w['group_id'],parent_group_id=w['parent_group_id'],envelope=w['envelope'],
                     candidate_points=[p for g in candidates['groups'] if g['group_id']==w['parent_group_id']
                                       for p in g['candidate_points'] if w['envelope'][0]<=p['source_step_id']<=w['envelope'][1]])
                for w in windows]))
    critical = []
    for w in windows:
        for f in read(source/'critical_facts.json')['facts']:
            if f['group_id']==w['parent_group_id'] and w['envelope'][0]<=f['source_step_id']<=w['envelope'][1]:
                critical.append(dict(f,group_id=w['group_id'],parent_group_id=f['group_id']))
    save(args.out/'critical_facts.json',dict(exhaustive=False,reviewer_type='AI',facts=critical,
        source_sha256=sha(source/'critical_facts.json')))
    eligible = [w for w in windows if all(preflight['prompt_tokens'][j]+512<=8192 for ids in w['arms'].values() for j in ids)]
    eligible_ids = {j for w in eligible for ids in w['arms'].values() for j in ids}
    reused = [j for j in eligible_ids if (returned/(j+'.json')).exists()]
    manifest = dict(existing_results={j:sha(returned/(j+'.json')) for j in sorted(reused)},
                    first_generation_required=sorted(eligible_ids-set(reused)))
    save(args.out/'reuse_manifest.json',manifest)
    protocol = dict(parent_protocol,scope='All common-boundary disagreement windows from the same 18 previously evaluated tasks',
                    parent_protocol_sha256=sha(source/'protocol.json'),grouping='local_common_boundary_windows')
    for name,key in [('requests.jsonl','requests_sha256'),('groups.json','groups_sha256'),
                     ('fact_review.json','fact_review_sha256'),('critical_facts.json','critical_facts_sha256')]:
        protocol[key] = sha(args.out/name)
    save(args.out/'protocol.json',protocol)
    report = dict(windows=len(windows),tasks=len({w['parent_group_id'] for w in windows}),requests=len(selected),
        historically_token_eligible_windows=len(eligible),eligible_requests=len(eligible_ids),
        reusable_outputs=len(reused),new_requests=len(eligible_ids-set(reused)),selected_critical_facts=len(critical),
        note='Token counts reused from identical v1 stage inputs. Remote tokenizer preflight still required. Local windows from one task are correlated.')
    save(args.out/'report.json',report)
    print(report)


if __name__=='__main__':
    main()
