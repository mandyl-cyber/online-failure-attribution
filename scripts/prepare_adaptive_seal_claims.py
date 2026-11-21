"""Causal D0/D1 replay using one frozen model, then prepare paired claims."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from scripts.run_stage_claim_pipeline import read, save
from scripts.prepare_adjacent_claim_pair import source_job
from segmenter.temporal_boundary import TemporalPredictor
from segmenter.eval_online import counts
from segmenter.adapters.openhands import load_trajectory
from segmenter.ledger import Ledger
from segmenter.rules import Rules
from segmenter import stage_claim_summary as core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('pilot/adaptive_seal_claim_v1'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base = Path('eval/temporal_boundary_v1/external_candidates')
    model_path = Path('temporal_boundary_train_v1/run_fold0/fold0/immediate/model.json')
    model = TemporalPredictor.load(model_path)
    vectors = np.load(base / 'vectors.npz', allow_pickle=False)
    records = read(base / 'records.json')
    frozen = read(base / 'predictions.json')['fold0']['original']
    # Exploratory settings fixed before this replay. Same weights in both arms.
    policies = dict(immediate=dict(threshold=.55, margin=0., max_delay=0),
                    adaptive=dict(threshold=.55, margin=.10, max_delay=1))
    torch.set_num_threads(2)
    predictions, groups, jobs = {}, [], []
    totals = {arm: {str(d): [0, 0, 0] for d in (0, 1)} for arm in policies}
    changes = dict(confirmed=0, revoked=0, promoted=0)
    for row in records:
        arms = {}
        for arm, policy in policies.items():
            predictor = TemporalPredictor(model.net, policy)
            for idx, vector in zip(row['ids'], vectors[row['key']], strict=True):
                predictor.feed_vector(vector, idx)
            predictor.finish()
            arms[arm] = dict(events=predictor.events, stages=predictor.stages)
            covered = [i for s in predictor.stages for i in range(s['start_step_id'], s['end_step_id']+1)]
            if covered != row['ids']:
                raise ValueError('Stage coverage mismatch')
            if any(e['detected_at_step_id']-e['boundary_step_id'] not in (0, 1) for e in predictor.events):
                raise ValueError('Unexpected publication lag')
            for delay in (0, 1):
                score = counts(predictor.events, row['boundaries'], 1, delay)
                totals[arm][str(delay)] = [a+b for a, b in zip(totals[arm][str(delay)], score)]
        old = {e['boundary_step_id'] for e in arms['immediate']['events']}
        if old != {e['boundary_step_id'] for e in frozen[row['traj_id']]}:
            raise ValueError('Immediate baseline reproduction failed')
        new = {e['boundary_step_id'] for e in arms['adaptive']['events']}
        changes['confirmed'] += len(old & new)
        changes['revoked'] += len(old-new)
        changes['promoted'] += len(new-old)
        predictions[row['traj_id']] = arms
        if old == new:
            continue
        common = sorted({1, row['ids'][-1]+1} | (old & new))
        changed = old ^ new
        left = max(b for b in common if b < min(changed))
        right = min(b for b in common if b > max(changed))
        steps, metadata = load_trajectory(row['source_path'], Rules(), expect_steps=len(row['ids']))
        if len(steps) != len(row['ids'])+1 or metadata.get('n_core') != len(row['ids']):
            raise ValueError('Source alignment changed')
        if steps[-1].raw_ref.get('observation_id') is not None or any(s.raw_ref.get('observation_id') is None for s in steps[:-1]):
            raise ValueError('Incomplete source alignment')
        ledger = Ledger(args.out / 'journals' / row['key'], 'adaptive-'+row['key'], defer_claims=True)
        for step in steps[:-1]:
            ledger.feed(step)
        group = dict(group_id=row['key'], traj_id=row['traj_id'], task=row['task'], envelope=[left, right-1], arms={})
        for arm, result in arms.items():
            ids = []
            for stage in result['stages']:
                start, end = stage['start_step_id'], stage['end_step_id']
                if not left <= start <= end < right:
                    continue
                job = source_job(ledger, row['task'], arm, stage['stage_id'], start, end, row['ids'][-1])
                job['detected_at_seq'] = stage['sealed_at_step_id']
                built = core.build(job)
                core.verify_row(built)
                jobs.append(built)
                ids.append(built['job_id'])
            group['arms'][arm] = ids
        groups.append(group)
    metrics = {}
    for arm, delays in totals.items():
        metrics[arm] = {}
        for delay, (tp, predicted, gold) in delays.items():
            metrics[arm][delay] = dict(tp=tp, predicted=predicted, gold=gold,
                precision=tp/predicted if predicted else 0., recall=tp/gold if gold else 0.,
                f1=2*tp/(predicted+gold) if predicted+gold else 0.)
    request_bytes = b''.join(core.encoded(j)+b'\n' for j in jobs)
    (args.out / 'requests.jsonl').write_bytes(request_bytes)
    save(args.out / 'groups.json', groups)
    save(args.out / 'predictions.json', predictions)
    report = dict(trajectories=len(records), metrics=metrics, changed_tasks=len(groups), requests=len(jobs),
                  cut_set_changes=changes, policies=policies,
                  status='causal_replay_complete_claim_generation_pending',
                  caveat='Exploratory policy on previously evaluated tasks. Lagged rescoring reuses frozen weights and requires validation. No claim quality inference yet.')
    save(args.out / 'report.json', report)
    save(args.out / 'protocol.json', dict(policies=policies, checkpoint_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        weights_sha256=read(model_path)['weights_sha256'], requests_sha256=hashlib.sha256(request_bytes).hexdigest(),
        model='same frozen 7B adapter for both arms', max_length=8192, max_new_tokens=512,
        scope='All fold0 policy-disagreement envelopes on the same 37 SWE tasks',
        instruction_sha256=hashlib.sha256(core.INSTRUCTION.encode()).hexdigest()))
    print(report)


if __name__ == '__main__':
    main()
