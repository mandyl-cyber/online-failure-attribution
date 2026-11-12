"""Rebuild immutable per-step text from the already audited supervised corpus."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from .adapters.openhands import load_trajectory
from .bench import iter_bench

TEXT_VERSION = 1
SCAN_CHARS = 65536


def bounded(text, limit=SCAN_CHARS):
    text = str(text or '')
    if len(text) <= limit:
        return text
    return text[:limit//2] + '\n[TRUNCATED]\n' + text[-limit//2:]


def clean(text):
    # Normalize ephemeral workspace roots; retain the relative path and code case.
    return re.sub(r'/workspace/[^/\s\"\x27]+/', '/workspace/', bounded(text))


def step_text(step):
    return dict(idx=step.idx, tool=str(step.tool), action=clean(step.action),
                observation=clean(step.observation), exit_code=step.exit_code,
                raw_action_chars=len(step.action or ''), raw_observation_chars=len(step.observation or ''),
                scan_truncated=len(step.action or '')>SCAN_CHARS or len(step.observation or '')>SCAN_CHARS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='eval/boundary_supervised/data_fixed')
    ap.add_argument('--out', default='eval/boundary_encoder/text')
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    manifest = data/'selected_manifest.jsonl'
    if not manifest.exists():
        manifest = Path('codetrace_bench/bench_manifest.verified.jsonl')
    if out.exists():
        raise SystemExit('Output exists; use a fresh path')
    out.mkdir(parents=True)
    records = json.loads((data/'records.json').read_text(encoding='utf-8'))
    wanted = {r['traj_id']:r for r in records}
    n = steps_n = truncated = 0
    with (out/'steps.jsonl').open('w',encoding='utf-8') as fh:
        for e in iter_bench(manifest,'codetrace_bench'):
            if e['traj_id'] not in wanted:
                continue
            r = wanted[e['traj_id']]
            steps, meta = load_trajectory(e['traj_path'],base=e['base'],expect_steps=e['step_count'])
            if len(steps)>r['n_steps']:
                assert meta['format']=='completions' and all(s.tool_class=='FINISH' for s in steps[r['n_steps']:])
                steps = steps[:r['n_steps']]
            assert [s.idx for s in steps]==r['ids']
            h = hashlib.sha256()
            for s in steps:
                h.update(json.dumps([s.action,s.observation],ensure_ascii=False).encode())
            if h.hexdigest()!=r['content_sha256']:
                raise ValueError('Source drift: '+r['traj_id'])
            rows = [step_text(s) for s in steps]
            fh.write(json.dumps(dict(traj_id=r['traj_id'],rows=rows),ensure_ascii=False)+'\n')
            steps_n += len(rows); n += 1
            truncated += sum(s['scan_truncated'] for s in rows)
            if n%50==0:
                print(f'{n}/{len(records)} trajectories rebuilt',flush=True)
    assert n==len(records)
    audit = dict(trajectories=n,steps=steps_n,text_version=TEXT_VERSION,scan_limit=SCAN_CHARS,
                 scan_truncated_steps=truncated,source=data.as_posix(),
                 source_records_sha256=hashlib.sha256((data/'records.json').read_bytes()).hexdigest(),
                 text_sha256=hashlib.sha256((out/'steps.jsonl').read_bytes()).hexdigest(),
                 labels_in_text=False,thought_in_text=False,all_raw_fingerprints_verified=True)
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit))


if __name__=='__main__':
    main()
