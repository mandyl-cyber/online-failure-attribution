"""Build audited boundary supervision and freeze causal rule baseline emissions."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .adapters.openhands import load_trajectory
from .bench import iter_bench, read_manifest
from .boundary_features import BoundaryFeatures, FEATURE_VERSION
from .eval_segmenter import source_family
from .prepare_downstream import validate_ranges
from .rules import Rules
from .run_online import OnlineSegmenter
from .segment import cut_points


def canonical_task(name):
    return re.sub(r'\s+', ' ', str(name).strip().lower())


def assign_folds(records, n_folds=5):
    # Union task identities AND exact normalized trace duplicates before assigning folds.
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, r in enumerate(records):
        for key in ('task:' + canonical_task(r['task']), 'trace:' + r['content_sha256']):
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = {}
    for i, r in enumerate(records):
        groups.setdefault(find(i), []).append(r)
    buckets = {}
    for members in groups.values():
        identity = '|'.join(sorted({canonical_task(r['task']) for r in members}))
        gid = hashlib.sha256(identity.encode()).hexdigest()
        source = '+'.join(sorted({r['source'] for r in members}))
        buckets.setdefault(source, []).append((gid, members))
    for group_list in buckets.values():
        for j, (gid, members) in enumerate(sorted(group_list, key=lambda x: x[0])):
            for r in members:
                r.update(group=gid, fold=j % n_folds)
    for field in ('task', 'content_sha256'):
        check = {}
        for r in records:
            key = canonical_task(r[field]) if field == 'task' else r[field]
            if key in check and check[key] != r['fold']:
                raise AssertionError('Cross-fold leakage')
            check[key] = r['fold']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='codetrace_bench/bench_manifest.verified.jsonl')
    ap.add_argument('--artifacts', default='codetrace_bench')
    ap.add_argument('--out', default='eval/boundary_supervised/data')
    ap.add_argument('--exclude-task-manifest', help='Exclude all tasks present in this development manifest')
    ap.add_argument('--exclude-data', help='Exclude exact trace fingerprints present in an earlier dataset')
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit('Output exists; use a new directory')
    out.mkdir(parents=True)
    rules = Rules()
    raw = {r['traj_id']: r for r in read_manifest(a.manifest)}
    selected = list(raw.values())
    if a.exclude_task_manifest:
        excluded = {canonical_task(r['task_name']) for r in read_manifest(a.exclude_task_manifest)}
        selected = [r for r in selected if canonical_task(r['task_name']) not in excluded]
    selected_path = out/'selected_manifest.jsonl'
    selected_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
    excluded_hashes = set()
    if a.exclude_data:
        excluded_hashes = {r['content_sha256'] for r in json.loads((Path(a.exclude_data)/'records.json').read_text(encoding='utf-8'))}
    labels, records, skipped = {}, [], []
    with (out/'features.jsonl').open('w', encoding='utf-8') as fh:
        for n, e in enumerate(iter_bench(selected_path, a.artifacts), 1):
            tid = e['traj_id']
            if e.get('error'):
                skipped.append(dict(traj_id=tid, reason=e['error']))
                continue
            try:
                steps, meta = load_trajectory(e['traj_path'], rules, base=e['base'], expect_steps=e['step_count'])
                if meta.get('n_core', len(steps)) != e['step_count']:
                    raise ValueError('Raw trajectory step count does not align')
                # Only drop explicitly unannotated completion-log FINISH suffix.
                if len(steps) > e['step_count']:
                    suffix = steps[e['step_count']:]
                    if meta['format'] != 'completions' or any(s.tool_class != 'FINISH' for s in suffix):
                        raise ValueError('Unexplained extra steps')
                    steps = steps[:e['step_count']]
                ids = [s.idx for s in steps]
                validate_ranges(e['stages'], ids)
                features = BoundaryFeatures(4)
                seg = OnlineSegmenter(rules, modes=['node', 'phase'])
                emissions = {'node': [], 'phase': []}
                known = {'node': set(), 'phase': set()}
                digest = hashlib.sha256()
                rows = []
                for s in steps:
                    digest.update(json.dumps([s.action, s.observation], ensure_ascii=False).encode())
                    rows.append(dict(idx=s.idx, **features.feed(s)))
                    seg.feed(s)
                    for mode, cutter in seg.segmenters.items():
                        # A sealed preceding span commits the following start.
                        for stage in cutter.stages:
                            boundary = stage['end_step_id'] + 1
                            if boundary <= s.idx and boundary not in known[mode]:
                                emissions[mode].append(dict(boundary_step_id=boundary, detected_at_step_id=s.idx))
                                known[mode].add(boundary)
                result = seg.finish()
                if digest.hexdigest() in excluded_hashes:
                    raise ValueError('Exact trace duplicate of development data')
                record = dict(traj_id=tid, task=raw[tid]['task_name'], source=source_family(e),
                              content_sha256=digest.hexdigest(), ids=ids, n_steps=len(ids),
                              has_thought=any(s.thought.strip() for s in steps),
                              format=meta['format'], raw_source=meta['source'],
                              mapping=[dict(idx=s.idx, raw_ref=s.raw_ref) for s in steps],
                              rules_final={m: cut_points(result['stages'][m]) for m in ('node','phase')},
                              rules_online=emissions)
                records.append(record)
                labels[tid] = dict(boundaries=cut_points(e['stages']), stages=e['stages'], solved=e['solved'])
                fh.write(json.dumps(dict(traj_id=tid, rows=rows), ensure_ascii=False)+'\n')
            except (ValueError, KeyError, OSError) as exc:
                skipped.append(dict(traj_id=tid, reason=str(exc)))
            if n % 25 == 0:
                print(f'{n} entries: {len(records)} aligned, {len(skipped)} skipped', flush=True)
    assign_folds(records)
    report = dict(n_traj=len(records), n_steps=sum(r['n_steps'] for r in records),
                  groups=len({r['group'] for r in records}), folds=dict(Counter(r['fold'] for r in records)),
                  sources=dict(Counter(r['source'] for r in records)), skipped=skipped,
                  feature_version=FEATURE_VERSION,
                  exclude_task_manifest=a.exclude_task_manifest,exclude_data=a.exclude_data,
                  selected_openhands=sum(r['agent']=='OpenHands' for r in selected),
                  manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
                  label_policy='start-of-stage binary target; first step excluded',
                  alignment_scope='offline benchmark normalized steps; not proof of native live-event alignment',
                  evaluation_scope=('external task-name-disjoint pool; no tuning' if a.exclude_task_manifest else
                                    'grouped cross-validation on previously inspected development corpus, not pristine test'))
    for name, obj in [('records.json', records), ('labels.json', labels), ('audit.json', report), ('rules.json', rules.raw)]:
        (out/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'skipped'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
