"""Freeze an exposed-corpus development split and prefix-only AI labeling pilot.

No model predictions, failure annotations or stage labels enter labeling packets.
This does not create a pristine external test set from a previously used corpus.
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


LABELS = ['ACQUIRE', 'ENABLE', 'CONSTRUCT', 'OPERATE', 'CHECK', 'DELIVER', 'UNKNOWN']
SEED = 'purpose-round2-v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def grouped_split(records, pilot_ids):
    """Union repeated tasks, existing groups and identical content before splitting."""
    parent = {r['traj_id']: r['traj_id'] for r in records}
    if len(parent) != len(records):
        raise ValueError('Duplicate trajectory ID')
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    seen = {}
    for r in records:
        for field in ('task', 'group', 'content_sha256'):
            if not r.get(field):
                raise ValueError('Missing grouping field: ' + field)
            key = field, r[field]
            if key in seen:
                a, b = find(r['traj_id']), find(seen[key])
                parent[max(a, b)] = min(a, b)
            seen[key] = r['traj_id']
    components = defaultdict(list)
    for r in records:
        components[find(r['traj_id'])].append(r['traj_id'])
    if not set(pilot_ids) <= set(parent):
        raise ValueError('Pilot contains unmapped trajectories')
    assignments = {}
    for ids in components.values():
        component = sha('\n'.join(sorted(ids)).encode())
        bucket = int(sha((SEED + component).encode())[:8], 16) % 100
        split = 'train' if bucket < 70 else 'dev' if bucket < 85 else 'reserved'
        if set(ids) & set(pilot_ids):
            split = 'dev'
        for tid in ids:
            assignments[tid] = (component, split)
    return [dict(traj_id=r['traj_id'], source=r['source'], task=r['task'],
                 group=assignments[r['traj_id']][0], split=assignments[r['traj_id']][1],
                 legacy_fold=r['fold'], legacy_content_sha256=r['content_sha256'],
                 historical_exposure='used_in_previous_experiments') for r in records]


def packet_for(record, rows, contract, intent, position, alias):
    if [s['idx'] for s in rows] != record['ids']:
        raise ValueError('Step alignment mismatch')
    if not contract.strip():
        raise ValueError('Missing initial task')
    if [s['idx'] for s in intent] != record['ids']:
        raise ValueError('Intent alignment mismatch')
    steps = []
    for row, public in zip(rows[:position+1], intent[:position+1]):
        steps.append(dict(idx=row['idx'], tool=row['tool'], action=row['action'],
                          observation=row['observation'], exit_code=row.get('exit_code'),
                          public_intent=public['text'],
                          input_truncated=bool(row.get('scan_truncated', False))))
    return dict(trajectory_id=alias, task=contract, current_step_id=steps[-1]['idx'], steps=steps)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('pilot/purpose_round2_v1'))
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit('Refusing to overwrite a frozen round')
    paths = dict(records=Path('eval/boundary_supervised/data_fixed/records.json'),
                 rows=Path('eval/boundary_encoder/text_v1/steps.jsonl'),
                 contracts=Path('eval/purpose_context_v2b/contracts.json'),
                 intents=Path('eval/semantic_input_ablation_v1/inputs/intent_provenance.json'),
                 mapping=Path('eval/qwen_purpose_pilot_v1/LOCAL_ONLY_mapping_gold.json'),
                 guide=Path('pilot/purpose_human_audit_v1/LABELING_GUIDE.md'))
    records = read(paths['records'])
    pilot = read(paths['mapping'])
    splits = grouped_split(records, [r['traj_id'] for r in pilot])
    assignments = {r['traj_id']: r for r in splits}
    with paths['rows'].open(encoding='utf-8') as f:
        rows = {r['traj_id']: r['rows'] for r in map(json.loads, f)}
    contracts, intents = read(paths['contracts']), read(paths['intents'])
    lookup = {r['traj_id']: r for r in records}
    for r in records:
        assert r['ids'] == [s['idx'] for s in rows[r['traj_id']]]
    # Candidate retrieval motifs are NOT labels. Random samples remain separate.
    motifs = dict(prerequisite=r'\b(pip\d* install|apt(?:-get)? install|npm install|conda install)\b',
                  execution=r'\b(python\d*|ffmpeg|sox|make|bash)\b',
                  verification=r'\b(pytest|unittest|test|diff|assert|sha256sum)\b',
                  delivery=r'\b(git push|submit|deliver)\b|^finish$')
    candidates = defaultdict(list)
    for r in records:
        if assignments[r['traj_id']]['split'] != 'train':
            continue
        for pos, step in enumerate(rows[r['traj_id']]):
            entry = (r['traj_id'], pos)
            candidates[(r['source'], 'random')].append(entry)
            for motif, pattern in motifs.items():
                if re.search(pattern, step['action'] + '\n' + step['tool'], re.I):
                    candidates[(r['source'], motif)].append(entry)
    rng = random.Random(SEED)
    chosen, selected, used = [], [], set()
    # 12 natural step samples; 12 targeted candidates. More Terminal coverage.
    quotas = [('SWE', 'random', 4), ('Terminal-Bench', 'random', 8)]
    quotas += [(source, motif, 1 if source == 'SWE' else 2)
               for source in ('SWE', 'Terminal-Bench') for motif in motifs]
    shortfalls = []
    for source, motif, count in quotas:
        pool = list(candidates[(source, motif)])
        rng.shuffle(pool)
        fresh = [x for x in pool if x not in used][:count]
        if len(fresh) != count:
            shortfalls.append(dict(source=source, motif=motif, wanted=count, found=len(fresh)))
        for tid, pos in fresh:
            used.add((tid, pos))
            alias = f'R2_{len(chosen)+1:03d}'
            packet = packet_for(lookup[tid], rows[tid], contracts[tid]['text'], intents[tid], pos, alias)
            chosen.append(dict(request_id=alias, source=source, packet=packet))
            selected.append(dict(request_id=alias, traj_id=tid, idx=packet['current_step_id'],
                                 group=assignments[tid]['group'], split='train', sampling=motif,
                                 packet_sha256=sha(json.dumps(packet, sort_keys=True, ensure_ascii=False).encode())))
    availability = {}
    for name in ('purpose_probe_v1', 'purpose_probe_evaluation_v2', 'purpose_student_v3',
                 'purpose_ai_boundary_diagnostic_v2'):
        availability['eval/' + name] = Path('eval', name).exists()
    args.out.mkdir(parents=True)
    write(args.out/'split_manifest.json', splits)
    write(args.out/'selection.json', selected)
    write(args.out/'blind_packets.json', chosen)
    write(args.out/'ai_labels.json', dict(protocol='purpose-round2-ai-reference-v1', reviewer_kind='ai',
          reviewer_id=None, labels={p['request_id']: dict(final_label=None, reviewed=False,
          evidence=[], review_note='') for p in chosen}))
    write(args.out/'protocol.json', dict(version=SEED, labels=LABELS,
          label_source='AI weak supervision; not human gold',
          test_status='reserved_after_historical_exposure; NOT pristine external holdout',
          split_policy='task/content/existing-group connected components; 70/15/15 hash; old pilot groups forced dev',
          input_policy='initial task and completed prefix only; public intent optional auxiliary evidence',
          main_metric='D0 boundary F1 with +/-1 step one-to-one matching',
          secondary_metrics=['exact boundary F1', 'per-source precision/recall/F1', 'extra/missing cuts', 'coverage'],
          selection_policy='Select models/thresholds only on train/dev; reserve consumed only after frozen model selection',
          boundary_training='Use task-group out-of-fold purpose predictions for boundary training',
          deployment_gate=dict(minimum_f1_gain=.01, paired_interval_lower_bound_gt=0,
                               requires_new_independent_tasks=True),
          current_round_scope='audit, split freeze, 24 training-prefix AI annotations; no default replacement'))
    report = dict(records=len(records), steps=sum(r['n_steps'] for r in records),
          groups=len({r['group'] for r in splits}),
          counts=dict(Counter(r['source'] + ':' + r['split'] for r in splits)),
          candidate_counts={source + ':' + motif:len(v) for (source,motif),v in candidates.items()},
          shortfalls=shortfalls, packets=len(chosen),
          raw_files_present=sum(Path(r['raw_source']).is_file() for r in records),
          missing_remote_experiments=availability,
          inputs={k:dict(path=str(p), sha256=sha(p.read_bytes())) for k,p in paths.items()},
          code_sha256=sha(Path(__file__).read_bytes()),
          pristine_test_established=False,
          limitations=['All 461 trajectories were previously included in feature/experiment work.',
                      'Input rows may be truncated; this run verifies row IDs, not new raw content hashes.',
                      'Targeted candidate motifs do not determine labels or population prevalence.',
                      'No purpose labels from the reserved split are generated in this round.'])
    write(args.out/'audit.json', report)
    print(json.dumps({k:report[k] for k in ('records','steps','groups','counts','packets','shortfalls')}, indent=2))


if __name__ == '__main__':
    main()
