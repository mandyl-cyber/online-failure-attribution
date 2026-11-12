"""Bounded, causal features; never consumes labels, metadata or hidden thought."""
import math
import re
from collections import deque

FEATURE_VERSION = 1


class BoundaryFeatures:
    def __init__(self, window=4):
        if not 1 <= window <= 16:
            raise ValueError('window must be 1..16')
        self.window = window
        self.history = deque(maxlen=window)
        self.previous_paths = set()
        self.previous_class = None
        self.class_run = 0
        self.since_write = 1000
        self.last_idx = None

    def feed(self, step):
        if self.last_idx is not None and step.idx <= self.last_idx:
            raise ValueError('Steps must have increasing native IDs')
        self.last_idx = step.idx
        tc = str(step.tool_class)
        paths = set(step.touched_paths)
        self.class_run = self.class_run + 1 if tc == self.previous_class else 1
        self.since_write = 0 if step.written_paths else min(1000, self.since_write + 1)
        obs = step.observation or ''
        action = step.action or ''
        d = {'tool': tc, 'tool_name': step.tool, 'tool_changed': int(tc != self.previous_class),
             'class_run': math.log1p(self.class_run), 'since_write': math.log1p(self.since_write),
             'exit': 'unknown' if step.exit_code is None else ('ok' if step.exit_code == 0 else 'fail'),
             'path_overlap': len(paths & self.previous_paths) / max(1, len(paths | self.previous_paths)),
             'writes': math.log1p(len(step.written_paths)), 'paths': math.log1p(len(paths)),
             'action_len': math.log1p(len(action)), 'observation_len': math.log1p(len(obs)),
             'test_pass': int(bool(re.search(r'\b(passed|success|test result: ok)\b', obs, re.I))),
             'test_fail': int(bool(re.search(r'\b(failed|traceback|error)\b', obs, re.I)))}
        for c in step.attrs.get('cmd_classes', []):
            d['command_' + str(c)] = 1
        # Do not encode absolute paths, task IDs, full-history summaries or stage age.
        text = 'ACTION ' + action[:1000] + ' OBS ' + obs[:200] + ' ' + obs[-400:]
        self.history.append((d, text))
        features, texts = {}, []
        for lag, (past, txt) in enumerate(reversed(self.history)):
            features.update({f'lag{lag}:{k}': v for k, v in past.items()})
            texts.append(f' LAG{lag} ' + txt)
        self.previous_paths, self.previous_class = paths, tc
        return {'features': features, 'text': ''.join(texts)}


def select_window(row, window):
    features = {k: v for k, v in row['features'].items() if int(k.split(':')[0][3:]) < window}
    # Cached text is only used for the four-step configuration.
    return features
