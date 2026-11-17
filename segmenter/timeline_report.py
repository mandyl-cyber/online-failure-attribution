"""Report local live samples only; source timestamps are never subtracted."""
import argparse
import json
import math
from pathlib import Path
from .durable_input import atomic_json


def report(root):
    rows = []
    for path in Path(root).rglob('timeline.jsonl'):
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                raise ValueError('Incomplete timeline; stop the writer before reporting: '+str(path))
    live = [r for r in rows if r.get('mode')=='live' and not r.get('recovery')]
    stats = {}
    for key in ('action_to_ready_ms', 'observation_to_ready_ms', 'ready_to_publish_ms', 'queue_ms', 'processing_ms'):
        values = sorted(r[key] for r in live if isinstance(r.get(key), (int,float)))
        stats[key] = dict(n=len(values), **{f'p{p}': values[max(0, math.ceil(len(values)*p/100)-1)] if values else None for p in (50,95,99)})
    return dict(scope='Observer-local durations only; no synchronized upstream latency; replay excluded',
                rows=len(rows), recovery_rows=sum(bool(r.get('recovery')) for r in rows),
                replay_rows=sum(r.get('mode')=='replay' for r in rows), metrics=stats,
                backlog_samples=[{k:r[k] for k in ('utc_ns','pending_actions','uncommitted_claim_jobs') if k in r}
                                 for r in live if r['kind']=='backlog'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root'); ap.add_argument('--out', required=True)
    a=ap.parse_args(); result=report(a.root)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    atomic_json(a.out, result); print(json.dumps(result, indent=2))


if __name__=='__main__': main()
