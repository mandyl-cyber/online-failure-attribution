"""Reproducible M1 development-only phase search and EXEC audit.

Run: python -m segmenter.audit_m1 --out eval/m1_audit
Task-level hash split prevents runs of the same task crossing partitions.
Holdout is evaluated only for the baseline and development-selected setting.
"""
import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

from .adapters.openhands import load_trajectory
from .bench import iter_bench, read_manifest
from .eval_segmenter import match_boundaries, prf, source_family
from .events import Event
from .rules import Rules
from .run_online import run_steps
from .segment import Segmenter, cut_points


def partition(task):
    return 'holdout' if int(hashlib.sha256(('m1-v1:' + task).encode()).hexdigest(), 16) % 5 == 0 else 'dev'


def phase_cuts(steps, rows, rules):
    seg = Segmenter(rules, 'phase')
    for step, row in zip(steps, rows):
        seg.update(step, [Event(t, step.idx) for t in row['events']],
                   {'node': row['node'], 'closed': []})
    seg.finish()
    return cut_points(seg.snapshot())


def score(records, rules=None):
    counts = [0, 0, 0]
    for e, steps, res in records:
        pred = phase_cuts(steps, res['rows'], rules) if rules else cut_points(res['stages']['phase'])
        gold = cut_points(e['stages'])
        counts[0] += match_boundaries(pred, gold)
        counts[1] += len(pred)
        counts[2] += len(gold)
    return dict(zip(('precision', 'recall', 'f1'), prf(*counts)), tp=counts[0], pred=counts[1], gold=counts[2], n=len(records))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='eval/m1_audit')
    ap.add_argument('--rules', help='Rule snapshot (YAML or JSON) for reproducible reruns')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    def save(name, data):
        (out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = 'codetrace_bench/bench_manifest.verified.jsonl'
    raw = {r['traj_id']: r for r in read_manifest(manifest)}
    rules = Rules(a.rules)
    save('baseline_rules.json', rules.raw)
    splits, skipped, execs, boundaries = [], [], [], []
    records = {'dev': [], 'holdout': []}
    for e in iter_bench(manifest, 'codetrace_bench'):
        task = str(raw[e['traj_id']].get('task_name') or e['traj_id'])
        split = partition(task)
        source = source_family(e)
        splits.append(dict(traj_id=e['traj_id'], task=task, split=split, source=source))
        if e.get('error'):
            skipped.append(dict(traj_id=e['traj_id'], reason=e['error']))
            continue
        steps, meta = load_trajectory(e['traj_path'], rules, base=e['base'], expect_steps=e['step_count'])
        if meta.get('n_core', len(steps)) != e['step_count']:
            skipped.append(dict(traj_id=e['traj_id'], reason='misaligned'))
            continue
        res = run_steps(steps, rules, modes=['phase'])
        by_id = {s.idx: s for s in steps}
        rows = {r['idx']: r for r in res['rows']}
        for idx in e['incorrect_steps']:
            s = by_id[idx]
            if s.tool_class == 'EXEC':
                execs.append(dict(traj_id=e['traj_id'], source=source, split=split, step=idx,
                    action=s.action, observation=s.observation[-1500:], classes=s.attrs.get('cmd_classes'),
                    events=rows[idx]['events']))
        if source != 'Terminal-Bench':
            continue
        records[split].append((e, steps, res))
        if split == 'dev':
            gold = cut_points(e['stages'])
            for idx in cut_points(res['stages']['phase']):
                if not any(abs(idx-g) <= 1 for g in gold):
                    boundaries.append(dict(traj_id=e['traj_id'], step=idx, gold=gold,
                        context=[dict(step=s.idx, action=s.action[:800], events=rows[s.idx]['events'])
                                 for s in steps if idx-1 <= s.idx <= idx+1]))
    save('split.json', splits)
    save('skipped.json', skipped)
    save('exec_errors.json', execs)
    save('dev_extra_boundaries.json', boundaries)
    counts = Counter((r['action'].split() or ['<empty>'])[0] for r in execs)
    save('exec_categories.json', dict(Counter('recognized_probe' if 'probe' in (r['classes'] or [])
                                             else 'unknown' for r in execs)))
    save('exec_heads.json', counts.most_common())
    candidates = []
    for patch, verify, inspect in itertools.product((1, 2, 3), (1, 2, 3), (2, 3, 4)):
        overrides = {f'segment.phase_min_run.{k}': v for k, v in
                     [('PATCH', patch), ('VERIFY', verify), ('INSPECT', inspect)]}
        candidates.append(dict(overrides=overrides, dev=score(records['dev'], Rules(a.rules, overrides=overrides))))
    # Deterministic tie-break follows grid order, never holdout performance.
    best = max(candidates, key=lambda c: c['dev']['f1'])
    save('sweep.json', candidates)
    result = dict(selected=best['overrides'], baseline={k: score(v) for k,v in records.items()},
                  tuned={k: score(v, Rules(a.rules, overrides=best['overrides'])) for k,v in records.items()},
                  exec_count=len(execs), exec_heads=counts.most_common(), skipped=len(skipped))
    save('summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
