"""Audit released stage supervision without loading trajectory contents."""
import collections
import hashlib
import json
import statistics
from pathlib import Path


def main():
    root = Path('codetrace_bench')
    datasets = {}
    report = {}
    for split in ('verified', 'full'):
        path = root / f'bench_manifest.{split}.jsonl'
        rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
        datasets[split] = rows
        lengths, keys, invalid = [], collections.Counter(), []
        for r in rows:
            stages = r['stages']
            ids = []
            for s in stages:
                keys.update(s.keys())
                lengths.append(s['end_step_id'] - s['start_step_id'] + 1)
                ids.extend(range(s['start_step_id'], s['end_step_id'] + 1))
            if not ids or ids != list(range(ids[0], ids[0] + r['step_count'])):
                invalid.append(r['traj_id'])
        report[split] = dict(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            trajectories=len(rows), unique_ids=len({r['traj_id'] for r in rows}),
            tasks=len({r['task_name'] for r in rows}),
            agents=dict(collections.Counter(r['agent'] for r in rows)),
            stages=len(lengths), stage_keys=dict(keys),
            stage_length_median=statistics.median(lengths),
            singleton_stages=sum(n == 1 for n in lengths),
            invalid_coverage_ids=invalid,
            openhands=sum(r['agent'] == 'OpenHands' for r in rows))
    a, b = datasets['verified'], datasets['full']
    by_full = {r['traj_id']: r for r in b}
    report['overlap'] = dict(
        trajectory_ids=len({r['traj_id'] for r in a} & {r['traj_id'] for r in b}),
        task_names=len({r['task_name'] for r in a} & {r['task_name'] for r in b}),
        full_rows_on_verified_tasks=sum(r['task_name'] in {x['task_name'] for x in a} for r in b),
        differing_stage_annotations=sum(r['traj_id'] in by_full and r['stages'] != by_full[r['traj_id']]['stages'] for r in a))
    report['openhands_verified'] = dict(
        stages=sum(len(r['stages']) for r in a if r['agent'] == 'OpenHands'),
        stage_length_median=statistics.median(s['end_step_id']-s['start_step_id']+1 for r in a if r['agent']=='OpenHands' for s in r['stages']))
    out = Path('eval/stage_label_audit')
    out.mkdir(parents=True, exist_ok=True)
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
