"""Small causal GRU with ordered, bounded confidence confirmation.

No generated text, future inputs, or gold-reset memory. Checkpoints are local JSON
plus PyTorch weights; the JSON file is the deployment entry point.
"""
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .boundary_features import BoundaryFeatures

VERSION = 'temporal-boundary-v1'
INPUT_DIM = 192


class StepEncoder:
    """Stable signed feature hashing. No task IDs, raw paths, thought or gold."""
    def __init__(self):
        self.features = BoundaryFeatures(1)
        self.plan = {}

    def feed(self, step):
        features = self.features.feed(step)['features']
        public = step.attrs.get('public_action_args', {})
        tasks = public.get('task_list', []) if isinstance(public, dict) else []
        if isinstance(tasks, list):
            states = {str(t.get('id', i)): str(t.get('status', 'unknown'))
                      for i, t in enumerate(tasks) if isinstance(t, dict)}
            if step.tool == 'task_tracking':
                features['plan:changed'] = sum(self.plan.get(k) != v for k,v in states.items())
                features['plan:removed'] = len(self.plan.keys()-states.keys())
                features['plan:count'] = len(states)
                self.plan = states
            for status in ('todo', 'in_progress', 'done', 'completed'):
                features['plan:'+status] = sum(v == status for v in self.plan.values())
        issues = step.attrs.get('input_integrity', {}).get('issues', [])
        features['input:known_issues'] = len(issues)
        out = np.zeros(INPUT_DIM, dtype=np.float32)
        for name, value in sorted(features.items()):
            if isinstance(value, str):
                name, value = name+'='+value, 1.
            value = float(value)
            # Bound counts while preserving their sign. Feature extraction is shared
            # between preparation and live inference, including repaired plan input.
            value = math.copysign(math.log1p(abs(value)), value)
            digest = hashlib.sha256(name.encode()).digest()
            out[int.from_bytes(digest[:4], 'little') % INPUT_DIM] += value * (1 if digest[4] & 1 else -1)
        return out


class TemporalNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.project = nn.Sequential(nn.Linear(INPUT_DIM, hidden), nn.LayerNorm(hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        # Continuous left, boundary and current context, stage mean, stage age,
        # and number of observed steps since the candidate boundary.
        self.head = nn.Sequential(nn.Linear(4*hidden+2, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def encode(self, x, state=None):
        z = self.project(x)
        h, state = self.gru(z, state)
        return z, h, state

    def score(self, left, boundary, current, stage_mean, age, lag):
        meta = current.new_tensor([math.log1p(age), lag/2.])
        return self.head(torch.cat((left, boundary, current, stage_mean, meta), dim=-1)).squeeze(-1)


class Confirmation:
    """Resolve oldest candidate first. Every position is considered exactly once."""
    def __init__(self, threshold=.5, margin=.15, max_delay=2):
        if max_delay not in (0,1,2) or not 0 < threshold < 1 or margin < 0:
            raise ValueError('Invalid confirmation configuration')
        self.threshold, self.margin, self.max_delay = threshold, margin, max_delay
        self.cursor = 1
        self.start = 0
        self.closed = False

    def advance(self, t, scorer, final=False):
        if self.closed:
            raise ValueError('Already finalized')
        resolved = []
        while self.cursor <= t:
            b = self.cursor
            p = float(scorer(b, t, self.start))
            certain = p <= max(0.,self.threshold-self.margin) or p >= min(1.,self.threshold+self.margin)
            expired = t-b >= self.max_delay
            if not (certain or expired or final):
                break
            cut = p >= self.threshold
            resolved.append(dict(position=b, observed_at=t, score=p, cut=cut,
                reason='stream_end' if final else ('confidence' if certain else 'deadline')))
            if cut:
                self.start = b
            self.cursor += 1
        if final:
            self.closed = True
        return resolved


class TemporalPredictor:
    """Bounded tensor cache; GRU is never reset at predicted boundaries."""
    def __init__(self, net, policy=None):
        self.net = net.eval()
        self.model = SimpleNamespace(feature_version=VERSION)
        self.device = next(net.parameters()).device
        self.policy_config = policy or dict(threshold=.5, margin=.15, max_delay=2)
        self.controller = Confirmation(**self.policy_config)
        self.encoder = StepEncoder()
        self.state = None
        self.prefix = torch.zeros(net.hidden, device=self.device)
        self.base_prefix = self.prefix.clone()
        self.cache = OrderedDict()
        self.stages = []
        self.position = -1
        self.previous_id = None
        self.start_id = None
        self.events = []
        self.last_emissions = []
        self.closed = False

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads(path.read_text(encoding='utf-8'))
        if meta['version'] != VERSION:
            raise ValueError('Temporal checkpoint version mismatch')
        weights = path.parent/meta['weights']
        if hashlib.sha256(weights.read_bytes()).hexdigest() != meta['weights_sha256']:
            raise ValueError('Temporal weight hash mismatch')
        net = TemporalNet(meta['hidden'])
        net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        return cls(net, meta['policy'])

    def _probability(self, b, t, start):
        left_h, b_h, before_b, _, _, _ = self.cache[b]
        h = self.cache[t][1]
        mean = (before_b-self.base_prefix)/max(1,b-start)
        logit = self.net.score(left_h, b_h, h, mean, b-start, t-b)
        return torch.sigmoid(logit).item()

    def _resolve(self, final=False):
        emitted = []
        initial_start = self.controller.start
        def score(b,t,start):
            # A cut committed earlier in this tick changes the next candidate's context.
            if start != initial_start:
                self.base_prefix = self.cache[start][2].clone()
            return self._probability(b,t,start)
        decisions = self.controller.advance(self.position,score,final=final)
        for decision in decisions:
            b,p,reason = decision['position'],decision['score'],decision['reason']
            if decision['cut']:
                boundary_id = self.cache[b][4]
                event = dict(boundary_step_id=boundary_id, detected_at_step_id=self.previous_id,
                             score=p, reason=reason)
                emitted.append(event)
                self.stages.append(dict(stage_id=len(self.stages)+1,start_step_id=self.start_id,
                    end_step_id=self.cache[b][5], sealed=True,
                    sealed_at_step_id=self.previous_id, boundary_step_id=boundary_id))
                self.start_id = boundary_id
                self.base_prefix = self.cache[b][2].clone()
        self.events.extend(emitted)
        self.last_emissions = emitted
        return emitted

    @torch.inference_mode()
    def feed_vector(self, vector, idx):
        if self.closed or (self.previous_id is not None and idx != self.previous_id+1):
            raise ValueError('Expected contiguous steps on an open stream')
        started = time.perf_counter()
        old_id = self.previous_id
        old_h = self.state[0,0].clone() if self.state is not None else torch.zeros(self.net.hidden,device=self.device)
        x = torch.as_tensor(vector, dtype=torch.float32, device=self.device).view(1,1,-1)
        z,h,self.state = self.net.encode(x, self.state)
        before = self.prefix.clone()
        self.prefix += z[0,0]
        self.position += 1
        self.previous_id = idx
        if self.start_id is None:
            self.start_id = idx
        self.cache[self.position] = (old_h,h[0,0],before,self.prefix.clone(),idx,old_id)
        self._resolve()
        while self.cache and next(iter(self.cache)) < self.controller.cursor:
            self.cache.popitem(last=False)
        return dict(status='committed' if self.last_emissions else 'continue',
            boundary_step_id=self.last_emissions[-1]['boundary_step_id'] if self.last_emissions else None,
            detected_at_step_id=idx, emissions=self.last_emissions,
            pending_candidates=max(0,self.position-self.controller.cursor+1),
            model_version=VERSION, ms=(time.perf_counter()-started)*1000)

    def feed(self, step):
        return self.feed_vector(self.encoder.feed(step), step.idx)

    @torch.inference_mode()
    def finish(self, incomplete=False):
        if self.closed:
            return []
        self.last_emissions = []
        if self.position >= 0:
            if not incomplete:
                self._resolve(final=True)
            self.stages.append(dict(stage_id=len(self.stages)+1,start_step_id=self.start_id,
                end_step_id=self.previous_id,sealed=True,sealed_at_step_id=self.previous_id,
                incomplete=incomplete,reason='interrupted' if incomplete else 'stream_end'))
        self.closed = True
        self.cache.clear()
        return self.last_emissions

    def snapshot(self, final=False):
        if final and not self.closed:
            raise ValueError('Call finish explicitly before final snapshot')
        return self.stages + ([dict(stage_id=len(self.stages)+1,start_step_id=self.start_id,
            end_step_id=self.previous_id,sealed=False)] if self.start_id is not None and not self.closed else [])


def save_model(net, directory, policy, training):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    weights = directory/'weights.pt'
    torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()}, weights)
    meta = dict(version=VERSION, hidden=net.hidden, weights=weights.name,
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(), policy=policy,
        parameter_count=sum(p.numel() for p in net.parameters()),training=training)
    (directory/'model.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
