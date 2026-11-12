"""Build blind, independent CodeTraceBench annotation packets (20 SWE + 20 terminal)."""
import argparse
import hashlib
import json
from pathlib import Path

from .bench import iter_bench, read_manifest
from .adapters.openhands import load_trajectory
from .eval_segmenter import source_family
from .rules import Rules
from .run_online import run_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='pilot/m2_codetrace')
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit('Output exists; choose a new directory to preserve annotations.')
    manifest = 'codetrace_bench/bench_manifest.verified.jsonl'
    raw = {r['traj_id']: r for r in read_manifest(manifest)}
    entries = sorted(iter_bench(manifest, 'codetrace_bench'),
                     key=lambda e: hashlib.sha256(('m2-v1:' + e['traj_id']).encode()).hexdigest())
    counts, tasks, selected = {}, set(), []
    rules = Rules()
    def save(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    for e in entries:
        source = source_family(e)
        key = (source, bool(e.get('solved')))
        task = raw[e['traj_id']].get('task_name') or e['traj_id']
        if source not in ('SWE', 'Terminal-Bench') or counts.get(key, 0) >= 10 or task in tasks or e.get('error'):
            continue
        steps, meta = load_trajectory(e['traj_path'], rules, base=e['base'], expect_steps=e['step_count'])
        if meta.get('n_core', len(steps)) != e['step_count']:
            continue
        counts[key] = counts.get(key, 0) + 1
        tasks.add(task)
        pid = f'p{len(selected)+1:03}'
        selected.append(dict(packet_id=pid, traj_id=e['traj_id'], task=task, source=source,
                             solved=e.get('solved'), n_steps=len(steps)))
        # Full text, no rule events, original gold or success labels in annotator packets.
        packet = dict(packet_id=pid, task=meta.get('task', ''),
                      steps=[{k: s.to_dict()[k] for k in ('idx', 'thought', 'action', 'observation', 'raw_ref')} for s in steps])
        save(out / 'trajectories' / f'{pid}.json', packet)
        for annotator in ('A', 'B'):
            save(out / annotator / f'{pid}.json', dict(packet_id=pid, annotator=annotator,
                 status='unreviewed', errors=[], notes='',
                 error_template=dict(error_id='', claim='', t_err=None, t_lock=None, t_obs=None,
                     lock_status=None, lock_type=None, outcome=None, evidence=[], confidence=None)))
        result = run_steps(steps, rules)
        save(out / 'reviewer_only' / f'{pid}.json', dict(gold=e['incorrect_steps'],
             candidates=[r for r in result['rows'] if set(r['events']) & {'REFERENCE','VERIFY_PASS','HANDOFF','IRREVERSIBLE'}]))
        if len(selected) == 40:
            break
    save(out / 'reviewer_only' / 'selection.json', selected)
    save(out / 'reviewer_only' / 'rules.json', rules.raw)
    save(out / 'status.json', dict(codetrace_count=len(selected), expected_codetrace=40,
         longrca_count=0, expected_longrca=40, annotation_status='not_started',
         missing='LongRCA local trajectories and two human annotators',
         strata={f'{k[0]} solved={k[1]}':v for k,v in counts.items()}))
    print(json.dumps({'selected': len(selected), 'out': str(out), 'longrca': 'pending'}, ensure_ascii=False))
    if len(selected) != 40:
        raise SystemExit('Insufficient aligned tasks; inspect status.json')


if __name__ == '__main__':
    main()
