"""Read-only native-event boundary observer. Run from repository root as a module."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from segmenter.adapters.openhands import OpenHandsStreamAdapter
from segmenter.boundary_model import OnlineBoundaryPredictor
from segmenter.ledger import atomic as _atomic


def atomic(path, value):
    # Windows readers can briefly deny replacement while opening a snapshot.
    # Keep the old complete snapshot and retry the same new value, never truncate.
    for attempt in range(10):
        try:
            return _atomic(path, value)
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.02)


def follow(run, once=False, poll=0.2):
    run = Path(run).resolve()
    model = Path(__file__).with_name('swe.joblib')
    model_meta = json.loads(model.with_suffix('.json').read_text())
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if digest != model_meta['sha256']:
        raise ValueError('Frozen model checksum mismatch')
    out = run / 'shadow'
    out.mkdir(exist_ok=True)
    snapshot = out / 'stages.json'
    previous = json.loads(snapshot.read_text()) if snapshot.exists() else None
    if previous and previous['model_sha256'] != digest:
        raise ValueError('Cannot resume with a different model')
    # Recover by replaying immutable source events, never flushing pending actions.
    predictor = OnlineBoundaryPredictor.load(model)
    adapter = OpenHandsStreamAdapter(fold_agent_messages=False)
    atomic(out / 'ready.json', dict(model_sha256=digest, ready_at=time.time()))
    seen, decisions, actions = {}, [], set()
    terminal = None
    session = None
    while True:
        dirs = list((run / 'state/store').glob('sessions/*/events'))
        if len(dirs) > 1:
            raise ValueError('Expected exactly one OpenHands session in this run')
        if dirs:
            session = str(dirs[0].relative_to(run))
            if previous and previous['session'] != session:
                raise ValueError('Session changed since restart')
            files = sorted(dirs[0].glob('*.json'), key=lambda p: int(p.stem))
            for path in files:
                try:
                    event = json.loads(path.read_text(encoding='utf-8'))
                except (json.JSONDecodeError, UnicodeError):
                    if once or (run / 'shadow.stop').exists():
                        raise ValueError('Incomplete event file: ' + str(path))
                    break  # producer may still be writing; retry, preserving order
                key = str(event['id'])
                fingerprint = hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()
                if key in seen:
                    if seen[key] != fingerprint:
                        raise ValueError('Native event changed: ' + key)
                    continue
                if previous and key in previous['event_hashes'] and previous['event_hashes'][key] != fingerprint:
                    raise ValueError('Native event changed since restart: ' + key)
                if 'observation' in event and event.get('cause') is not None and event['cause'] not in actions:
                    raise ValueError('Observation precedes its action')
                if 'action' in event:
                    actions.add(event['id'])
                for step in adapter.feed(event):
                    prediction = predictor.feed(step)
                    decisions.append({k: v for k, v in prediction.items() if k not in ('ms', 'emitted_at')})
                    with (out / 'timing.jsonl').open('a', encoding='utf-8') as timing:
                        timing.write(json.dumps(dict(event_id=event['id'], step_id=step.idx,
                            predictor_ms=prediction['ms'], observed_at=time.time(),
                            recovery=previous is not None and key in previous['event_hashes'])) + '\n')
                if event.get('action') == 'finish':
                    terminal = 'finished'
                if event.get('observation') == 'agent_state_changed':
                    value = event.get('extras', {}).get('agent_state')
                    if value in ('error', 'stopped', 'rejected'):
                        terminal = 'interrupted'
                seen[key] = fingerprint
                atomic(snapshot, dict(model_sha256=digest, session=session,
                    event_hashes=seen, boundary_events=decisions,
                    stages=predictor.snapshot(final=terminal is not None), terminal=terminal,
                    pending_action_ids=list(adapter._pending),
                    input_mode='recorded_replay' if once else 'live_directory_watch',
                    observed_steps=len(decisions)))
        if once or (run / 'shadow.stop').exists():
            if previous and not set(previous['event_hashes']).issubset(seen):
                raise ValueError('Source events missing on recovery')
            break
        time.sleep(poll)
    print(json.dumps(dict(observed_steps=len(decisions), stages=len(predictor.snapshot()),
                          terminal=terminal, output=str(out))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    parser.add_argument('--once', action='store_true', help='Recorded replay, not a live validation')
    args = parser.parse_args()
    follow(args.run_directory, args.once)
