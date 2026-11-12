"""Lossless LongRCA coding-domain pilot packets, preserving native message IDs."""
import argparse
import hashlib
import json
from pathlib import Path


def validate_history(data):
    history = data['history']
    ids = [r['step'] for r in history]
    if not ids or any(type(i) is not int for i in ids) or ids != sorted(set(ids)):
        raise ValueError('History IDs must be unique, ordered integers')
    if data['mistake_step'] not in ids:
        raise ValueError('Gold root cause does not resolve to a native message')
    if any(not isinstance(r.get('content'), str) for r in history):
        raise ValueError('Expected textual history content')
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='longrca_bench')
    ap.add_argument('--out', default='pilot/m2_longrca')
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    if out.exists():
        raise SystemExit('Output exists; select a fresh directory to preserve human annotations.')
    selected, seen = [], set()
    for source in ('swe_bench_pro', 'terminal_bench_2'):
        files = sorted((root / 'data/full' / source).glob('*.json'),
                       key=lambda p: hashlib.sha256(('m2-longrca-v1:' + p.stem).encode()).hexdigest())
        count = 0
        for path in files:
            data = json.loads(path.read_text(encoding='utf-8'))
            history = validate_history(data)
            # Deduplicate exact task text across both sources before selecting.
            task = next((r['content'] for r in history if r.get('role') == 'user'), '')
            task_hash = hashlib.sha256(task.strip().encode()).hexdigest()
            if task_hash in seen:
                continue
            seen.add(task_hash)
            selected.append((source, path, data, history, task, task_hash))
            count += 1
            if count == 20:
                break
        if count != 20:
            raise ValueError(f'Insufficient unique tasks in {source}: {count}')

    def save(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

    index = []
    for i, (source, path, data, history, task, task_hash) in enumerate(selected, 1):
        pid = f'l{i:03}'
        steps = []
        for position, record in enumerate(history):
            # Native message is retained verbatim; do not invent thought/action/observation separation.
            steps.append(dict(idx=record['step'], role=record.get('role'), name=record.get('name'),
                content=record['content'], raw_message=record,
                raw_ref=dict(history_position=position, native_step=record['step'])))
        save(out / 'trajectories' / f'{pid}.json', dict(packet_id=pid, task=task,
             step_unit='native_message', steps=steps))
        for annotator in ('A', 'B'):
            save(out / annotator / f'{pid}.json', dict(packet_id=pid, annotator=annotator,
                 status='unreviewed', errors=[], notes='', error_template=dict(error_id='', claim='',
                 t_err=None, t_lock=None, t_obs=None, lock_status=None, lock_type=None,
                 outcome=None, evidence=[], confidence=None)))
        save(out / 'reviewer_only' / f'{pid}.json', dict(
             original_metadata={k:v for k,v in data.items() if k != 'history'},
             candidates=None, candidate_status='unsupported_native_multi_agent_format'))
        index.append(dict(packet_id=pid, question_ID=data['question_ID'], source=source,
             source_file=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             task_sha256=task_hash, n_steps=len(history), first_step=history[0]['step'],
             last_step=history[-1]['step']))
    save(out / 'reviewer_only/selection.json', index)
    save(out / 'status.json', dict(dataset='LongRCA', longrca_count=40, expected_longrca=40,
         annotation_status='not_started', strata={'swe_bench_pro':20, 'terminal_bench_2':20},
         all_failed=True, rule_candidates_available=False, step_unit='native_message'))
    # Verify round-trip fidelity, label separation and independent annotation templates.
    for item, original in zip(index, selected):
        pid = item['packet_id']
        packet = json.loads((out / 'trajectories' / f'{pid}.json').read_text(encoding='utf-8'))
        assert [s['raw_message'] for s in packet['steps']] == original[3]
        assert not set(packet) & {'mistake_agent', 'mistake_step', 'mistake_reason'}
        for who in ('A', 'B'):
            label = json.loads((out / who / f'{pid}.json').read_text(encoding='utf-8'))
            assert label['annotator'] == who and label['errors'] == []
    print(f'Validated {len(index)} lossless packets and 80 blank annotation files: {out}')


if __name__ == '__main__':
    main()
