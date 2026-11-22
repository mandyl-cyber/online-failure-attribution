"""Collect one existing smoke run for diagnosis; never rerun the agent."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import zipfile


def collect(run):
    run = Path(run).resolve(strict=True)
    if not run.is_dir() or not (run / 'controller.log').is_file():
        raise ValueError('Expected a smoke run directory containing controller.log')
    candidates = [run / name for name in (
        'controller.log', 'verification.log', 'manifest.json', 'result.json', 'shadow.log',
        'shadow/stages.json', 'shadow/timing.jsonl', 'shadow/ready.json',
        'state/controller_result.json', 'state/task.txt', 'state/trajectory.json',
        'workspace/ranges.py', 'workspace/test_ranges.py', 'checks/test_ranges.py',
    )]
    candidates += list((run / 'state/store').glob('sessions/*/events/*.json'))
    candidates += list((run / 'state/completions').rglob('*.json'))
    candidates += list((run / 'shadow/input_timing').rglob('*.json'))
    candidates += list((run / 'shadow/input_timing').rglob('*.jsonl'))
    candidates += list((run / 'workspace').glob('*.py'))
    candidates += list((run / 'checks').glob('*.py'))
    candidates += [run / 'shadow/latency_report.json']
    files, errors, events = [], [], []
    for path in sorted(set(candidates)):
        # Do not follow symlinks outside the selected run.
        if not path.resolve().is_relative_to(run):
            errors.append({'file': str(path), 'error': 'outside run directory'})
            continue
        if path.is_file():
            files.append(path)
    target = run.parent / (run.name + '-diagnostics-' + uuid.uuid4().hex[:8] + '.zip')
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = path.relative_to(run).as_posix()
            try:
                data = path.read_bytes()
            except OSError as exc:
                errors.append({'file': name, 'error': str(exc)})
                continue
            archive.writestr(name, data)
            if path.parent.name == 'events':
                try:
                    events.append({'file': name, 'event': json.loads(data)})
                except (ValueError, UnicodeError) as exc:
                    errors.append({'file': name, 'error': str(exc)})
        # Convenience copy; raw events above remain intact and untruncated.
        events.sort(key=lambda row: (str(Path(row['file']).parent),
                    int(Path(row['file']).stem) if Path(row['file']).stem.isdigit() else -1))
        archive.writestr('events_readable.json', json.dumps(events, ensure_ascii=False, indent=2))
        types = Counter()
        for row in events:
            event = row['event']
            if isinstance(event, dict):
                types[str(event.get('action') or event.get('observation') or 'other')] += 1
        archive.writestr('collection_summary.json', json.dumps({
            'run_directory': str(run), 'collected_at': datetime.now(timezone.utc).isoformat(),
            'event_count': len(events), 'event_types': dict(types),
            'collected_files': archive.namelist(), 'errors': errors,
            'note': 'Contains local task/model text. No Docker inspection or model weights collected.',
        }, ensure_ascii=False, indent=2))
    return target, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    args = parser.parse_args()
    target, errors = collect(args.run_directory)
    print(f'DIAGNOSTICS={target}')
    print(f'SIZE={target.stat().st_size / 1024:.1f} KiB; collection_errors={len(errors)}')


if __name__ == '__main__':
    main()
