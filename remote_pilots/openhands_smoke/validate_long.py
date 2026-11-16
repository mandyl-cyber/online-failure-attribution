"""Local fixed-fold re-evaluation and recorded long-source recovery checks."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from segmenter.boundary_model import joblib
from segmenter.eval_online import aggregate
from .shadow import follow


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def score_fixed():
    base = Path('eval/boundary_supervised/data_fixed')
    records = read(base / 'records.json')
    labels = read(base / 'labels.json')
    features = {x['traj_id']: x['rows'] for x in
                map(json.loads, (base / 'features.jsonl').read_text(encoding='utf-8').splitlines())}
    reference = read('pilot/swe_only_boundary_v1/predictions.json')
    arms = ('mixed_train_mixed_dev', 'mixed_train_swe_dev', 'swe_train_swe_dev')
    predictions = {a: {} for a in arms}
    for fold in range(5):
        train = {r['group'] for r in records if r['fold'] not in (fold, (fold+1)%5)}
        dev = {r['group'] for r in records if r['fold'] == (fold+1)%5}
        test = [r for r in records if r['source'] == 'SWE' and r['fold'] == fold]
        assert not (train & dev or train & {r['group'] for r in test} or dev & {r['group'] for r in test})
        for arm in arms:
            model = joblib.load(f'eval/swe_only_boundary_v1/models/{arm}_fold{fold}.joblib')
            for record in test:
                rows = features[record['traj_id']]
                assert [r['idx'] for r in rows] == record['ids']
                probabilities = model.probabilities(rows[1:])
                predictions[arm][record['traj_id']] = [dict(boundary_step_id=i, detected_at_step_id=i)
                    for i, p in zip(record['ids'][1:], probabilities) if p >= model.threshold]
    assert predictions == reference, 'Fixed-fold predictions changed'
    swe = [r for r in records if r['source'] == 'SWE']
    rules = {r['traj_id']: r['rules_online']['phase'] for r in swe}
    return dict(trajectories=len(swe), steps=sum(r['n_steps'] for r in swe),
                metrics={a: aggregate(p, labels) for a, p in dict(predictions, rules=rules).items()},
                predictions_reproduced=True, group_overlap=False,
                scope='Existing development folds, no retraining or tuning; not pristine test. Full-fit deployment model is not evaluated on its training tasks.')


def stable(snapshot):
    return {k: snapshot[k] for k in ('event_hashes', 'boundary_events', 'stages',
                                     'terminal', 'pending_action_ids', 'observed_steps')}


def long_check(record):
    source = Path(record['raw_source'])
    events = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len({e['id'] for e in events}) != len(events):
        raise ValueError('Duplicate source IDs')
    events.sort(key=lambda e: e['id'])
    # Pause after an action around halfway through, leaving its observation pending.
    split = next(i+1 for i, e in enumerate(events) if i >= len(events)//2 and e.get('action') == 'run')
    def write(run, subset):
        directory = run / 'state/store/sessions/check/events'
        directory.mkdir(parents=True, exist_ok=True)
        for e in subset:
            target = directory / f'{e["id"]}.json'
            temp = target.with_suffix('.tmp')
            temp.write_text(json.dumps(e), encoding='utf-8')
            temp.replace(target)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        full, resumed = root / 'full', root / 'resumed'
        write(full, events)
        start = time.perf_counter()
        follow(full, once=True)
        full_seconds = time.perf_counter() - start
        expected = stable(read(full / 'shadow/stages.json'))
        resumed.mkdir()
        with (root / 'watch.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-m', 'remote_pilots.openhands_smoke.shadow', str(resumed)],
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 120
                while not (resumed / 'shadow/ready.json').exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('Observer startup failed')
                    time.sleep(.1)
                # Recorded events injected into a growing directory; no agent is executing.
                write(resumed, events[:split])
                while True:
                    path = resumed / 'shadow/stages.json'
                    if path.exists() and len(read(path)['event_hashes']) == split:
                        break
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('Prefix processing failed: ' + (root / 'watch.log').read_text())
                    time.sleep(.1)
                prefix = read(path)
                assert prefix['pending_action_ids'], 'Expected pending action at interruption'
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=15)
        write(resumed, events[split:])
        start = time.perf_counter()
        follow(resumed, once=True)
        recovery_seconds = time.perf_counter() - start
        actual = stable(read(resumed / 'shadow/stages.json'))
        assert actual == expected, 'Recovery differs from uninterrupted replay'
        follow(resumed, once=True)
        assert stable(read(resumed / 'shadow/stages.json')) == expected
        return dict(traj_id=record['traj_id'], source=record['source'], native_events=len(events),
                    annotated_steps=record['n_steps'], online_steps=actual['observed_steps'],
                    stages=len(actual['stages']), full_replay_seconds=full_seconds,
                    recovery_and_remaining_seconds=recovery_seconds,
                    process_termination_recovered=True, duplicate_replay_identical=True,
                    prefix_pending=prefix['pending_action_ids'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='eval/live_shadow_validation_v1/report.json')
    args = parser.parse_args()
    target = Path(args.out)
    if target.exists():
        raise SystemExit('Use a new output path')
    report = dict(fixed_fold=score_fixed(), long_replay=[], limitations=[
        'Recorded native events, not new live agent tasks; no long-duration soak test.',
        'Online steps may differ from offline alignment; no gold scoring of native replay.',
        'Single observer, local filesystem; no power-loss or multi-session test.'])
    target.parent.mkdir(parents=True, exist_ok=True)
    records = read('eval/boundary_supervised/data_fixed/records.json')
    selected = sorted([r for r in records if r['format'] == 'events'], key=lambda r: r['n_steps'], reverse=True)[:3]
    for record in selected:
        print('Checking', record['traj_id'], flush=True)
        report['long_replay'].append(long_check(record))
        target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('REPORT=' + str(target), flush=True)


if __name__ == '__main__':
    main()
