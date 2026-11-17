"""Compare two frozen SWE purpose runs and their downstream D0 boundary value.

The AI audit is a reference, not human gold. The boundary development tasks were
previously inspected, so this is an exploratory comparison, not a blind test.
"""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from .swe_purpose_boundary_experiment import (
    LABELS, boundary_probs, crossfit_purpose, events, fit_boundary, fit_purpose,
    load_split, metrics, nest_boundary_oof, pick_threshold, predict_purpose,
    purpose_audit, read, rows_for, save,
)


def answers(root, split, aliases):
    run = read(root / split / 'run_manifest.json')
    found = {}
    for path in (root / split / 'responses').glob('*.json'):
        row = read(path)
        if row['request_id'] != path.stem or row['identity'] != run['identity']:
            raise ValueError(f'Invalid response identity: {path}')
        if row['request_id'].split('_s')[0] not in aliases:
            raise ValueError(f'Unexpected trajectory: {path}')
        if row['answer']['purpose'] not in LABELS:
            raise ValueError(f'Invalid purpose: {path}')
        found[row['request_id']] = row
    return found


def audit_reference(predictions, selection, reference):
    rows = []
    for item in selection['items']:
        rid = item['request_id']
        if rid not in predictions:
            continue
        rows.append(dict(request_id=rid, sampling=item['sampling'],
                         stratum=item['stratum'], reference=reference[rid]['label'],
                         prediction=predictions[rid]['answer']['purpose']))
    sampled = [r for r in rows if r['sampling'] == 'probability']
    weighted = 0.0
    weighted_complete = True
    for stratum, size in selection['stratum_sizes'].items():
        subset = [r for r in sampled if r['stratum'] == stratum]
        expected = sum(i['sampling'] == 'probability' and i['stratum'] == stratum
                       for i in selection['items'])
        if len(subset) != expected or not subset:
            weighted_complete = False
            break
        weighted += size * sum(r['prediction'] == r['reference'] for r in subset) / len(subset)
    return dict(n=len(rows), selected=len(selection['items']),
                agreement=sum(r['prediction'] == r['reference'] for r in rows),
                weighted_agreement=(weighted / sum(selection['stratum_sizes'].values())
                                    if weighted_complete else None),
                prediction_counts=dict(Counter(r['prediction'] for r in rows)), rows=rows)


def model_latency(rows):
    times = sorted(float(row['seconds']) for row in rows.values()
                   if row['label_source'] in ('qwen', 'qwen_search_alias'))
    if not times:
        return dict(n=0, median_seconds=None, p95_seconds=None)
    return dict(n=len(times), median_seconds=times[len(times) // 2],
                p95_seconds=times[int(0.95 * (len(times) - 1))])


def one_hot(trajectories, predictions):
    out = {}
    for t in trajectories:
        labels = []
        for step in t['steps']:
            rid = f"{t['alias']}_s{step['idx']:05d}"
            row = predictions.get(rid)
            if row is None:
                raise ValueError('Incomplete purpose run: ' + rid)
            label = row['answer']['purpose']
            labels.append({name: float(label == name) for name in LABELS})
        out[t['alias']] = labels
    return out


def paired_group_ci(dev, by_model, pairs, seed=41):
    import numpy as np

    groups = sorted({t['group'] for t in dev})
    by_group = {group: [t for t in dev if t['group'] == group] for group in groups}
    rng = np.random.default_rng(seed)
    differences = {name: [] for name in pairs}
    for _ in range(2000):
        sample = [t for index in rng.integers(len(groups), size=len(groups))
                  for t in by_group[groups[index]]]
        scores = {mode: metrics(sample, chosen)['tol1_D0']['f1']
                  for mode, chosen in by_model.items()}
        for name, (candidate, baseline) in pairs.items():
            differences[name].append(100 * (scores[candidate] - scores[baseline]))
    return {name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            for name, values in differences.items()}


def boundary_comparison(train, dev, purpose):
    from sklearn.model_selection import GroupKFold

    train_groups = [t['group'] for t in train]
    modes = ['structure', 'qwen7', 'qwen32']
    thresholds = {}
    dev_metrics = {}
    dev_events = {}
    for mode in modes:
        augmented = mode != 'structure'
        train_p = purpose[mode]['train'] if augmented else {}
        dev_p = purpose[mode]['dev'] if augmented else {}
        oof = {}
        for fit_idx, held_idx in GroupKFold(n_splits=5).split(train, groups=train_groups):
            fit = [train[i] for i in fit_idx]
            held = [train[i] for i in held_idx]
            x, y, _ = rows_for(fit, train_p, augmented)
            model = fit_boundary(x, y)
            xx, _, meta = rows_for(held, train_p, augmented)
            oof.update(zip(meta, boundary_probs(model, xx)))
        if len(oof) != sum(len(t['steps']) - 1 for t in train):
            raise AssertionError('Incomplete boundary out of fold predictions')
        thresholds[mode] = pick_threshold(train, oof)[1]
        x, y, _ = rows_for(train, train_p, augmented)
        model = fit_boundary(x, y)
        xx, _, meta = rows_for(dev, dev_p, augmented)
        probabilities = dict(zip(meta, boundary_probs(model, xx)))
        chosen = events(dev, probabilities, thresholds[mode])
        dev_events[mode] = chosen
        dev_metrics[mode] = metrics(dev, chosen)
    ci = paired_group_ci(dev, dev_events,
                         {'qwen7_vs_structure': ('qwen7', 'structure'),
                          'qwen32_vs_structure': ('qwen32', 'structure'),
                          'qwen32_vs_qwen7': ('qwen32', 'qwen7')})
    return dict(thresholds=thresholds, metrics=dev_metrics, paired_group_ci95_points=ci,
                events=dev_events)


def student_boundary_comparison(train, dev, runs, audit_root, out_dir):
    """Crossfit small purpose students; deploy-time features come from students only."""
    from .purpose_student import FEATURE_VERSION, PurposeStudent

    result = {}
    for mode in ('qwen7', 'qwen32'):
        labeled = []
        for t in train:
            copy = dict(t)
            copy['reviewed'] = {
                step['idx']: runs['train'][mode][f"{t['alias']}_s{step['idx']:05d}"]['answer']['purpose']
                for step in t['steps']
            }
            labeled.append(copy)
        oof = nest_boundary_oof(labeled)['structure_purpose']
        threshold = pick_threshold(labeled, oof)[1]
        train_purpose, folds = crossfit_purpose(labeled)
        fitted = fit_purpose(labeled)
        dev_purpose = predict_purpose(fitted, dev)
        x, y, _ = rows_for(labeled, train_purpose, True)
        model = fit_boundary(x, y)
        xx, _, meta = rows_for(dev, dev_purpose, True)
        probabilities = dict(zip(meta, boundary_probs(model, xx)))
        chosen = events(dev, probabilities, threshold)
        reference = purpose_audit(dev_purpose, dev, audit_root)
        vec, clf = fitted
        artifact = dict(feature_version=FEATURE_VERSION,
                        classes=[str(label) for label in clf.classes_],
                        vocabulary={key: int(value) for key, value in vec.vocabulary_.items()},
                        idf=vec.idf_.tolist(), coef=clf.coef_.tolist(),
                        intercept=clf.intercept_.tolist(), ngram_range=[3, 5],
                        sublinear_tf=True,
                        label_source=f'935 frozen {mode} SWE train candidate labels; not human gold')
        path = out_dir / f'purpose_student_{mode}.json.gz'
        with gzip.open(path, 'wt', encoding='utf-8', compresslevel=9) as file:
            json.dump(artifact, file, ensure_ascii=False, separators=(',', ':'))
        runtime = PurposeStudent.load(path)
        sample = dev[0]
        actual = runtime.probabilities_text(sample['texts'][0])
        expected = dev_purpose[sample['alias']][0]
        if max(abs(actual[label] - expected[label]) for label in LABELS) > 1e-9:
            raise AssertionError('Exported student differs from training inference')
        result[mode] = dict(threshold=threshold, boundary=metrics(dev, chosen),
                            purpose_reference={key: value for key, value in reference.items()
                                               if key != 'rows'},
                            student_bytes=path.stat().st_size,
                            student_classes=artifact['classes'],
                            student_artifact=str(path),
                            train_crossfit_folds=len(folds), events=chosen)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qwen7', type=Path, default=Path('pilot/swe_purpose_qwen_v2'))
    parser.add_argument('--qwen32', type=Path, default=Path('eval/swe_purpose_qwen32_awq_v1'))
    parser.add_argument('--package', type=Path, default=Path('remote_pilots/swe_purpose_v2'))
    parser.add_argument('--selection', type=Path, default=Path('pilot/swe_purpose_continuous_v1/selection.json'))
    parser.add_argument('--gold', type=Path, default=Path('eval/boundary_supervised/data_fixed/labels.json'))
    parser.add_argument('--out', type=Path, default=Path('eval/swe_purpose_qwen32_awq_v1/comparison.json'))
    args = parser.parse_args()
    mapping = {r['trajectory_id']: r for r in read(args.selection)}
    gold = read(args.gold)
    reviewed = {split: read(args.qwen7 / split / 'review_pending' / 'review_pending.json')['labels']
                for split in ('train', 'dev')}
    trajectories = {split: load_split(args.package, split, mapping, reviewed[split], gold)
                    for split in ('train', 'dev')}
    train_groups = {t['group'] for t in trajectories['train']}
    dev_groups = {t['group'] for t in trajectories['dev']}
    if train_groups & dev_groups:
        raise ValueError('Train/development task groups overlap')
    runs = {}
    audit = {}
    purpose = {mode: {} for mode in ('qwen7', 'qwen32')}
    for split in ('train', 'dev'):
        aliases = {t['alias'] for t in trajectories[split]}
        runs[split] = {
            'qwen7': answers(args.qwen7 / split / 'full_v2', '', aliases),
            'qwen32': answers(args.qwen32, split, aliases),
        }
        for rid in runs[split]['qwen32'].keys() & runs[split]['qwen7'].keys():
            if runs[split]['qwen32'][rid]['input_sha256'] != runs[split]['qwen7'][rid]['input_sha256']:
                raise ValueError('Input hash mismatch between models: ' + rid)
        selection = read(args.qwen7 / split / 'codex_blind_audit_v1' / 'selection.json')
        reference = read(args.qwen7 / split / 'codex_blind_audit_v1' / 'ai_labels.json')['labels']
        audit[split] = {mode: audit_reference(rows, selection, reference)
                        for mode, rows in runs[split].items()}
        for mode, rows in runs[split].items():
            expected = sum(len(t['steps']) for t in trajectories[split])
            if len(rows) == expected:
                purpose[mode][split] = one_hot(trajectories[split], rows)
    paired = {}
    paired_ai = {}
    for split, by_model in runs.items():
        both = by_model['qwen7'].keys() & by_model['qwen32'].keys()
        raw = {rid for rid in both if by_model['qwen7'][rid]['label_source'] in
               ('qwen', 'qwen_search_alias', 'qwen_field_alias') and
               by_model['qwen32'][rid]['label_source'] in
               ('qwen', 'qwen_search_alias', 'qwen_field_alias')}
        paired[split] = dict(n=len(both), raw_model_n=len(raw),
                             equal=sum(by_model['qwen7'][rid]['answer']['purpose'] ==
                                       by_model['qwen32'][rid]['answer']['purpose'] for rid in both),
                             raw_equal=sum(by_model['qwen7'][rid]['answer']['purpose'] ==
                                           by_model['qwen32'][rid]['answer']['purpose'] for rid in raw))
        old_ref = {row['request_id']: row for row in audit[split]['qwen7']['rows']}
        new_ref = {row['request_id']: row for row in audit[split]['qwen32']['rows']}
        states = Counter()
        for rid in old_ref.keys() & new_ref.keys():
            old_correct = old_ref[rid]['prediction'] == old_ref[rid]['reference']
            new_correct = new_ref[rid]['prediction'] == new_ref[rid]['reference']
            states[('both' if old_correct and new_correct else
                    'qwen7_only' if old_correct else
                    'qwen32_only' if new_correct else 'neither')] += 1
        paired_ai[split] = dict(states)
    report = dict(scope='Exploratory: previously inspected SWE task groups; AI reference is not human gold',
                  coverage={split: {mode: len(rows) for mode, rows in by_model.items()}
                            for split, by_model in runs.items()},
                  label_sources={split: {mode: dict(Counter(r['label_source'] for r in rows.values()))
                                         for mode, rows in by_model.items()}
                                 for split, by_model in runs.items()},
                  successful_model_call_latency={split: {mode: model_latency(rows)
                                                         for mode, rows in by_model.items()}
                                                 for split, by_model in runs.items()},
                  paired_predictions=paired,
                  paired_ai_reference=paired_ai,
                  ai_reference={split: {mode: {k: v for k, v in result.items() if k != 'rows'}
                                        for mode, result in by_model.items()}
                                for split, by_model in audit.items()})
    if all(split in purpose[mode] for mode in ('qwen7', 'qwen32') for split in ('train', 'dev')):
        report['boundary'] = boundary_comparison(trajectories['train'], trajectories['dev'], purpose)
        report['student_boundary'] = student_boundary_comparison(
            trajectories['train'], trajectories['dev'], runs,
            args.qwen7 / 'dev' / 'codex_blind_audit_v1', args.out.parent)
        student_events = {'structure': report['boundary']['events']['structure'],
                          **{mode: value['events'] for mode, value in
                             report['student_boundary'].items()}}
        report['student_boundary_ci95_points'] = paired_group_ci(
            trajectories['dev'], student_events,
            {'qwen7_vs_structure': ('qwen7', 'structure'),
             'qwen32_vs_structure': ('qwen32', 'structure'),
             'qwen32_vs_qwen7': ('qwen32', 'qwen7')}, seed=43)
    else:
        report['boundary'] = None
        report['student_boundary'] = None
        report['student_boundary_ci95_points'] = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(args.out, report)
    print(json.dumps({k: report[k] for k in
                      ('scope', 'coverage', 'paired_predictions',
                       'paired_ai_reference', 'ai_reference')}, ensure_ascii=False))
    if report['boundary']:
        print(json.dumps(report['boundary']['metrics'], ensure_ascii=False))
        print(json.dumps({mode: value['boundary'] for mode, value in
                          report['student_boundary'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
