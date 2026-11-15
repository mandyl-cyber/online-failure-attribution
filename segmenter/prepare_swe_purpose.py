"""Freeze SWE continuous-prefix annotation packages and a local boundary baseline."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil

from .eval_online import aggregate
from .prepare_purpose_round2 import packet_for, read, write


def select(records, assignments, split, count, target):
    # One trajectory per connected task/content group; length strata, no gold access.
    pool = sorted((r for r in records if assignments[r['traj_id']]['split'] == split),
                  key=lambda r: (r['n_steps'], r['traj_id']))
    strata = [pool[i*len(pool)//3:(i+1)*len(pool)//3] for i in range(3)]
    best = None
    for seed in range(256):
        rng = random.Random(f'swe-purpose-v1:{split}:{seed}')
        chosen, groups = [], set()
        for i, items in enumerate(strata):
            items = list(items)
            rng.shuffle(items)
            quota = count//3 + (i < count % 3)
            for r in items:
                group = assignments[r['traj_id']]['group']
                if group in groups:
                    continue
                chosen.append(r)
                groups.add(group)
                quota -= 1
                if quota == 0:
                    break
            if quota:
                raise ValueError('Insufficient independent groups in length stratum')
        rank = abs(sum(r['n_steps'] for r in chosen)-target)
        if best is None or rank < best[0]:
            best = rank, chosen
    return sorted(best[1], key=lambda r: r['traj_id'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('pilot/swe_purpose_continuous_v1'))
    ap.add_argument('--package', type=Path, default=Path('remote_pilots/swe_purpose_v1'))
    ap.add_argument('--data-out', type=Path, default=Path('eval/swe_pipeline_v1/data'))
    a = ap.parse_args()
    for path in (a.out, a.package, a.data_out):
        if path.exists():
            raise SystemExit('Refusing overwrite: '+str(path))
    paths = dict(records=Path('eval/boundary_supervised/data_fixed/records.json'),
                 labels=Path('eval/boundary_supervised/data_fixed/labels.json'),
                 split=Path('pilot/purpose_round2_v1/split_manifest.json'),
                 rows=Path('eval/boundary_encoder/text_v1/steps.jsonl'),
                 contracts=Path('eval/purpose_context_v2b/contracts.json'),
                 intents=Path('eval/semantic_input_ablation_v1/inputs/intent_provenance.json'))
    records = [r for r in read(paths['records']) if r['source']=='SWE']
    tids = {r['traj_id'] for r in records}
    splits = [s for s in read(paths['split']) if s['traj_id'] in tids]
    assignments = {s['traj_id']:s for s in splits}
    if set(assignments) != tids:
        raise ValueError('Missing split assignment')
    group_splits = {}
    for s in splits:
        group_splits.setdefault(s['group'], set()).add(s['split'])
    if any(len(v)!=1 for v in group_splits.values()):
        raise ValueError('Group leakage')
    rows = {x['traj_id']:x['rows'] for x in map(json.loads, paths['rows'].read_text(encoding='utf-8').splitlines()) if x['traj_id'] in tids}
    contracts, intents, labels = read(paths['contracts']), read(paths['intents']), read(paths['labels'])
    for r in records:
        tid = r['traj_id']
        if r['ids'] != [x['idx'] for x in rows[tid]]:
            raise ValueError('Row alignment mismatch')
        covered = [i for s in labels[tid]['stages'] for i in range(s['start_step_id'],s['end_step_id']+1)]
        if covered != r['ids'] or labels[tid]['boundaries'] != [s['start_step_id'] for s in labels[tid]['stages'][1:]]:
            raise ValueError('Gold alignment mismatch')
    selected = {s: select(records, assignments, s, n, target)
                for s,n,target in [('train',25,1000),('dev',10,350)]}
    for path in (a.out, a.package, a.data_out):
        path.mkdir(parents=True)
    write(a.out/'split_manifest.json', splits)
    write(a.data_out/'records.json', records)
    write(a.data_out/'labels.json', {tid:labels[tid] for tid in sorted(tids)})
    with (a.data_out/'steps.jsonl').open('w',encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(dict(traj_id=r['traj_id'],rows=rows[r['traj_id']]),ensure_ascii=False)+'\n')
    mapping, summary = [], {}
    template = Path('remote_pilots/qwen_purpose_v1')
    for split, chosen in selected.items():
        root = a.package/split
        root.mkdir()
        trajectories = []
        for i,r in enumerate(chosen,1):
            tid = r['traj_id']
            alias = f'{split.upper()}_{i:03d}'
            packet = packet_for(r, rows[tid], contracts[tid]['text'], intents[tid], len(r['ids'])-1, alias)
            trajectories.append({k:packet[k] for k in ('trajectory_id','task','steps')})
            mapping.append(dict(trajectory_id=alias,traj_id=tid,split=split,group=assignments[tid]['group'],steps=len(r['ids'])))
        with (root/'trajectories.jsonl').open('w',encoding='utf-8') as f:
            for item in trajectories:
                f.write(json.dumps(item,ensure_ascii=False)+'\n')
        for name in ('remote.py','SYSTEM_PROMPT.txt','OUTPUT_SCHEMA.json','ENVIRONMENT_TEMPLATE.txt'):
            shutil.copyfile(template/name, root/name)
        # Hash portable LF bytes for Windows and Linux checkouts.
        for path in root.iterdir():
            path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
        summary[split] = dict(trajectories=len(chosen),steps=sum(r['n_steps'] for r in chosen),
                              truncated_steps=sum(s['input_truncated'] for t in trajectories for s in t['steps']))
        write(root/'package_manifest.json', dict(protocol='swe-continuous-purpose-v1',
              trajectories=len(chosen),requests=summary[split]['steps'],
              files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir())}))
    write(a.out/'selection.json',mapping)
    # Report only train/dev; do not inspect reserved model performance.
    baseline = {}
    for split in ('train','dev'):
        subset = [r for r in records if assignments[r['traj_id']]['split']==split]
        baseline[split] = {mode:aggregate({r['traj_id']:r['rules_online'][mode] for r in subset},labels)
                           for mode in ('phase','node')}
    write(a.out/'rule_baseline.json',baseline)
    write(a.out/'protocol.json',dict(selection=summary,corpus=dict(Counter(s['split'] for s in splits)),
          selection_policy='25 train / 10 dev; one per group; length thirds; deterministic 256 candidate draws closest to 1000/350 steps; no gold used',
          task_type_coverage='Repository repair tasks; semantic task-type diversity not manually verified',
          label_policy='AI weak references, not human gold; independent second pass before adjudication; same labels do not imply same stage',
          baseline_policy='Existing frozen rule events rescored; no new model training or live replay',
          historical_exposure='All data previously exposed; reserved is not pristine test',
          input_policy='Full completed prefix, public intent optional; no gold/failure fields; overflow remains missing',
          future_training='Group cross-fit purpose predictions for boundary training; dev only for selection; reserved untouched',
          inputs={k:dict(path=str(v),sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k,v in paths.items()}))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
