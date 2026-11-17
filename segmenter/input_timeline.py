"""Shared observer clock. Never infer latency across process/clock domains."""
import hashlib
import json
import threading
import time
import uuid
from pathlib import Path
from .durable_input import atomic_json


def validate_event(adapter, event):
    if not isinstance(event, dict) or type(event.get('id')) not in (int, str):
        raise ValueError('Stable integer/string event ID required')
    if event.get('observation') not in (None, 'agent_state_changed', 'null', 'recall'):
        for part in (event, event.get('extras') or {}):
            if part.get('is_final') is False or part.get('partial') is True or part.get('stream') is True:
                raise ValueError('Streaming observations not supported')
        cause = event.get('cause')
        slot = adapter._pending.get(cause) if type(cause) in (int, str) else None
        if slot is None or slot[2]:
            raise ValueError('Expected one final observation with an outstanding exact cause')


class InputTimeline:
    def __init__(self, root, mode='live'):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.receipts = self.root/'receipts'
        self.receipts.mkdir(exist_ok=True)
        self.process = uuid.uuid4().hex
        self.mode = mode
        self.lock = threading.Lock()
        self.ready_times = {}

    def stamp(self):
        return dict(utc_ns=time.time_ns(), monotonic_ns=time.perf_counter_ns(),
                    process_id=self.process, mode=self.mode)

    def path(self, event_id):
        key = json.dumps(event_id, ensure_ascii=False).encode()
        return self.receipts/(hashlib.sha256(key).hexdigest()+'.json')

    def receive(self, event, recovery=False):
        path = self.path(event['id'])
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
        record = dict(event_id=event['id'], source_timestamp=event.get('timestamp'),
                      recovery=recovery, **self.stamp())
        atomic_json(path, record)
        return record

    def elapsed(self, start, end):
        if not start or start.get('process_id') != end['process_id'] or start.get('recovery'):
            return None
        if start.get('mode') != 'live' or end['mode'] != 'live':
            return None
        return (end['monotonic_ns']-start['monotonic_ns'])/1e6

    def receipt(self, event_id):
        path = self.path(event_id)
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

    def emit(self, kind, **fields):
        row = dict(kind=kind, **self.stamp(), **fields)
        with self.lock, (self.root/'timeline.jsonl').open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(row, ensure_ascii=False)+'\n')
        return row

    def ready(self, step, recovery=False):
        now = self.stamp()
        action = self.receipt(step.raw_ref.get('action_id'))
        observation = self.receipt(step.raw_ref.get('observation_id'))
        self.ready_times[step.idx] = dict(**now, recovery=recovery)
        self.emit('step_ready', step_id=step.idx, recovery=recovery,
                  action_to_ready_ms=None if recovery else self.elapsed(action, now),
                  observation_to_ready_ms=None if recovery else self.elapsed(observation, now))

    def published(self, step_id, recovery=False):
        now = self.stamp()
        self.emit('boundary_published', step_id=step_id, recovery=recovery,
                  ready_to_publish_ms=None if recovery else self.elapsed(self.ready_times.get(step_id), now))
