"""Check strict contract and replay equivalence on returned smoke archives."""
import json
import tempfile
import zipfile
from pathlib import Path
from segmenter.durable_input import DurableInput, atomic_json


def main():
    reports = []
    for path in sorted(Path('.').glob('run-*-diagnostics-*.zip')):
        with tempfile.TemporaryDirectory() as d, zipfile.ZipFile(path) as archive:
            names = sorted((n for n in archive.namelist() if '/events/' in n and n.endswith('.json')),
                           key=lambda n: int(Path(n).stem))
            stream = DurableInput(d, {})
            steps = []
            error = None
            for name in names:
                try:
                    steps.extend(stream.feed(json.loads(archive.read(name))))
                except ValueError as exc:
                    error = dict(file=name, message=str(exc))
                    break
            recovered = DurableInput(d, {})
            reports.append(dict(archive=path.name, events=len(names), accepted=stream.count,
                steps=len(steps), error=error, pending=len(stream.adapter._pending),
                replay_equal=[s.to_dict() for s in steps] == [s.to_dict() for s in recovered.recovered]))
    if not reports:
        raise ValueError('No archives found')
    out = Path('eval/durable_input_v1')
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'report.json', dict(runs=reports, scope='Historical replay; not live latency or segmentation accuracy'))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__': main()
