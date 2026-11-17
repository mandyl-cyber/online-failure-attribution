"""Record a narrow, auditable JSON field-name repair for Qwen32 AWQ outputs.

Only the exact key typo ``rationationale`` is accepted. The original answers and
all failed attempts remain in ``errors/``; no purpose label is inferred here.
"""

import argparse
import importlib.util
import json
from pathlib import Path


def protocol(path):
    spec = importlib.util.spec_from_file_location('swe_purpose_v2_protocol', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    p = protocol(args.package / 'remote.py')
    p.verify_package(args.package)
    jobs = {item['request_id']: item for item in p.jobs(args.package)}
    run = p.read(args.run / 'run_manifest.json')
    previous = args.run / 'format_recovery.json'
    repaired = list(p.read(previous).get('repaired', [])) if previous.exists() else []
    unresolved = []
    for path in sorted((args.run / 'errors').glob('*.json')):
        rid = path.stem
        if (args.run / 'responses' / path.name).exists():
            continue
        error = p.read(path)
        failures = error.get('failures', [])
        if rid not in jobs or len(failures) != 3:
            unresolved.append(rid)
            continue
        if any(x.get('error') != 'Wrong output fields' for x in failures):
            unresolved.append(rid)
            continue
        raw = failures[-1].get('raw_content')
        try:
            answer = json.loads(raw)
        except (TypeError, ValueError):
            unresolved.append(rid)
            continue
        if not isinstance(answer, dict) or set(answer) != {'purpose', 'evidence_id', 'rationationale'}:
            unresolved.append(rid)
            continue
        answer['rationale'] = answer.pop('rationationale')
        try:
            p.validate(answer, jobs[rid]['packet'])
        except (KeyError, TypeError, ValueError):
            unresolved.append(rid)
            continue
        record = dict(request_id=rid, input_sha256=jobs[rid]['input_sha256'],
                      identity=run['identity'], answer=answer,
                      label_source='qwen_field_alias',
                      normalization='rationationale->rationale',
                      raw_answer=raw, review_priority=True, usage=None,
                      server_model=run['config']['model'], seconds=None,
                      attempt=3)
        p.save(args.run / 'responses' / path.name, record)
        if rid not in repaired:
            repaired.append(rid)
    summary = dict(repaired=repaired, unresolved=unresolved,
                   normalization='exact rationationale key renamed to rationale; purpose unchanged')
    p.save(args.run / 'format_recovery.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
