"""Single-writer, replayable OpenHands input. No model or external side effects."""
import hashlib
import json
import os
from pathlib import Path

from .adapters.openhands import OpenHandsStreamAdapter


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as fh:
        json.dump(value, fh, ensure_ascii=False, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class DurableInput:
    """Replay accepted raw events to reconstruct pending calls and step numbers.

    Source must retain stable IDs. Only one writer per output directory is supported.
    A crash after journal commit and before consumer output is repaired by replay.
    """
    def __init__(self, directory, config, rules=None, base=1):
        from .input_timeline import InputTimeline
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.root / 'config.json'
        if manifest.exists():
            if json.loads(manifest.read_text(encoding='utf-8')) != config:
                raise ValueError('Input/config changed; use a new output directory')
        else:
            atomic_json(manifest, config)
        self.adapter = OpenHandsStreamAdapter(rules, base=base)
        self.timeline = InputTimeline(self.root/'timing', config.get('input_mode', 'live'))
        self.seen = {}
        self.count = 0
        self.recovered = []
        for p in sorted(self.root.glob('event-*.json')):
            event = json.loads(p.read_text(encoding='utf-8'))
            self.recovered.extend(self._feed(event, persist=False))

    def _feed(self, event, persist):
        if not isinstance(event, dict) or type(event.get('id')) not in (int, str):
            raise ValueError('Live contract requires a stable integer/string event id')
        key = (type(event['id']).__name__, str(event['id']))
        digest = hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()
        if key in self.seen:
            if self.seen[key] != digest:
                raise ValueError('Conflicting duplicate event ID')
            return []
        from .input_timeline import validate_event
        validate_event(self.adapter, event)
        self.timeline.receive(event, recovery=not persist)
        steps = self.adapter.feed(event)
        if persist:
            atomic_json(self.root / f'event-{self.count:09d}.json', event)
        self.count += 1
        self.seen[key] = digest
        for step in steps:
            self.timeline.ready(step, recovery=not persist)
        return steps

    def feed(self, event):
        try:
            return self._feed(event, persist=True)
        except ValueError as exc:
            atomic_json(self.root / 'rejected.json', dict(event=event, error=str(exc)))
            raise
