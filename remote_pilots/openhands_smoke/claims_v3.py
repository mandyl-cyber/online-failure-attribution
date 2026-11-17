"""Small development pilot for v3 summaries; stdlib only, one attempt per job."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from segmenter import stage_claim_summary as core

ROOT = Path('pilot/swe_claims_v3')
# Pre-output AI reference points. Not exhaustive human gold and not model input.
CASES = {
    'd5cd394a43': ('planning', [
        ('agent_assertion', 'The displayed plan proposes inspecting ops.linspace and TensorFlow code, reproducing the issue, applying a minimal fix and verifying it; these are planned actions.', [(2, 'observation')])],
        ['Do not report plan items as completed work.']),
    '104c381337': ('edit_and_source_conditions', [
        ('tool_observation', 'The edit initializes the NaN step with start.dtype and casts the divisors to start.dtype in the displayed TensorFlow linspace branches.', [(12, 'action'), (12, 'observation')]),
        ('tool_observation', 'The displayed ops implementation passes axis to the backend; its documentation declares axis=0 by default and separately notes a Torch limitation.', [(13, 'observation')])],
        ['Do not generalize a backend-specific documentation note to all backends.', 'Edit confirmation does not establish a verified fix.']),
    'd49ee42e57': ('verification_blocked', [
        ('tool_observation', 'The reproduction attempt stops during import with missing absl; the later environment check reports absl absent.', [(15, 'observation'), (17, 'observation')]),
        ('tool_observation', 'The subsequent TensorFlow reproduction stops with missing tensorflow, so this invocation does not verify linspace outputs.', [(18, 'action'), (18, 'observation')]),
        ('tool_observation', 'Git status reports numpy.py modified and Dockerfile untracked.', [(19, 'observation')]),
        ('tool_observation', 'Displayed requirement files list absl-py and tensorflow[and-cuda]~=2.16.1; this does not show installation.', [(16, 'observation'), (20, 'observation')])],
        ['Echoed script is not successful execution.', 'Untracked does not mean unmodified.']),
    'ae1e47e246': ('edit_failed_then_successful_example', [
        ('tool_observation', 'The MethodDefinition edit adds printing of nonempty decorators before the static modifier and method.', [(27, 'action'), (27, 'observation')]),
        ('tool_observation', 'The node -e attempt reports shell errors and a text.startsWith TypeError with exit 1.', [(28, 'action'), (28, 'observation')]),
        ('tool_observation', 'The subsequent heredoc example using babylon prints the shown OrderLine decorators inline and exits 0.', [(31, 'action'), (31, 'observation')])],
        ['The successful example is not evidence that the whole test suite passed.', 'Do not invent task-list contents from update counts.']),
    'b6c9a1a652': ('type_edit', [
        ('tool_observation', 'Tooltip.d.ts changes componentsProps.popper from PopperProps to Partial<PopperProps>, retaining TooltipComponentsPropsOverrides.', [(27, 'action'), (27, 'observation')]),
        ('tool_observation', 'The yarn version query prints 1.22.10.', [(28, 'action'), (28, 'observation')])],
        ['A version query is not a TypeScript check or a passing test.']),
    'c3c276d0ad': ('commit', [
        ('tool_observation', 'Git reports commit f2ded26 with one file changed, 20 insertions and one deletion.', [(35, 'observation')]),
        ('agent_assertion', 'The commit message describes inline class-member decorators; its description alone does not verify runtime correctness.', [(35, 'action')])],
        ['Do not upgrade the commit message to a verified implementation result.'])
}


def read(path):
    return core.strict_json(path.read_text(encoding='utf-8'))


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(core.encoded(value) + b'\n')
    temp.replace(path)


def rows_at(path):
    return [core.strict_json(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def prepare(out, source):
    old = [r['request'] for r in rows_at(source)]
    rows, references = [], []
    for prefix, (category, points, prohibitions) in CASES.items():
        found = [r for r in old if r['job_id'].startswith(prefix)]
        if len(found) != 1:
            raise ValueError('source selection missing/ambiguous: ' + prefix)
        row = core.build(found[0]); rows.append(row)
        refs = []
        for i, (basis, text, locations) in enumerate(points, 1):
            ids = [eid for eid, ref in row['evidence_map'].items()
                   if (ref['source_step_id'], ref['field']) in locations]
            if not ids:
                raise ValueError('reference lacks evidence')
            refs.append(dict(point_id=f'p{i}', text=text, source=basis, evidence_ids=ids,
                             priority='supporting' if prefix == 'b6c9a1a652' and i == 2 else 'key',
                             coverage=None, matched_claim_ids=[]))
        references.append(dict(job_id=row['job_id'], run_id=row['run_id'], category=category,
                               reviewer_type='AI', reviewed=True, reviewed_before_v3_output=True,
                               exhaustive=False, reference_points=refs, forbidden_inferences=prohibitions,
                               coverage_reviewed=False))
    out.mkdir(parents=True, exist_ok=False)
    (out / 'development.jsonl').write_bytes(b''.join(core.encoded(r) + b'\n' for r in rows))
    save(out / 'reference_points.json', references)
    save(out / 'schema.json', core.SCHEMA)
    save(out / 'protocol.json', dict(protocol=core.PROTOCOL, split='historical_development',
         input_sha256=core.digest(core.encoded(rows)), source_sha256=core.digest(source.read_bytes()),
         references_sha256=core.digest(core.encoded(references)),
         instruction_sha256=core.digest(core.INSTRUCTION.encode()),
         schema_sha256=core.digest(core.encoded(core.SCHEMA)), calls=len(rows),
         limits=dict(max_tokens=2048, max_request_bytes=100000, max_response_bytes=1000000),
         decoder='prompt_only; current tool server rejects response_format',
         caveats=['All tasks previously inspected: NOT held out; no population accuracy claim.',
                  'AI reference points, not human gold; categories not training labels.',
                  'Not exhaustive category coverage: interrupted stages need future examples.',
                  'Normalized source_step_id grouping, not raw concurrent event pairing.',
                  'Byte limit is not tokenizer context check; server may reject token overflow.',
                  'No live available_at_seq fabricated from retrospective requests.']))
    print(f'Prepared {len(rows)} development stages in {out}')


def now():
    return datetime.now(timezone.utc).isoformat()


def summarize(rows, out):
    counts = {s: 0 for s in ('ready', 'ready_empty', 'failed', 'reserved', 'blocked_input_budget', 'missing')}
    review = []
    for row in rows:
        path = out / (row['job_id'] + '.json')
        r = read(path) if path.exists() else dict(status='missing')
        counts[r['status']] += 1
        review.append(dict(job_id=row['job_id'], status=r['status'], claims=r.get('claims', []),
                           reviewed=False, reviewer_type=None, supported=None,
                           missing_key_points=[], severe_errors=[], note=''))
    save(out / 'summary.json', dict(counts=counts, semantic_quality='not_evaluated',
                                   scope='development only; readiness is schema/provenance, not truth'))
    if not (out / 'semantic_review.json').exists():
        save(out / 'semantic_review.json', review)
    print(json.dumps(counts))
    return counts


def run(root, out, base_url, model):
    rows = rows_at(root / 'development.jsonl'); protocol = read(root / 'protocol.json')
    if protocol['input_sha256'] != core.digest(core.encoded(rows)):
        raise ValueError('frozen input drift')
    if protocol['instruction_sha256'] != core.digest(core.INSTRUCTION.encode()) or protocol['schema_sha256'] != core.digest(core.encoded(core.SCHEMA)):
        raise ValueError('contract drift; prepare a new version')
    if protocol['references_sha256'] != core.digest(core.encoded(read(root / 'reference_points.json'))):
        raise ValueError('reference points changed after freezing')
    if len({r['job_id'] for r in rows}) != len(rows) or not rows:
        raise ValueError('duplicate/empty jobs')
    for row in rows:
        core.verify_row(row)
    config = dict(protocol_sha256=core.digest(core.encoded(protocol)), base_url=base_url, model=model,
                  runner_sha256=core.digest(Path(__file__).read_bytes().replace(b'\r\n', b'\n')),
                  core_sha256=core.digest(Path(core.__file__).read_bytes().replace(b'\r\n', b'\n')),
                  decoder='prompt_only', attempts=1, temperature=0)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists() and read(out / 'config.json') != config:
        raise ValueError('run configuration changed; choose a new output directory')
    save(out / 'config.json', config)
    # Preserve the exact evidence and reference pack alongside returned outputs.
    for name in ('development.jsonl', 'reference_points.json', 'protocol.json'):
        dest = out / name
        if dest.exists() and dest.read_bytes() != (root / name).read_bytes():
            raise ValueError('saved input pack changed: ' + name)
        if not dest.exists():
            dest.write_bytes((root / name).read_bytes())
    limits = protocol['limits']
    for i, row in enumerate(rows, 1):
        dest = out / (row['job_id'] + '.json')
        if dest.exists():
            continue
        body = dict(model=model, temperature=0, max_tokens=limits['max_tokens'], messages=[
            dict(role='system', content=core.INSTRUCTION),
            dict(role='user', content=core.encoded(row['request']).decode('utf-8'))])
        payload = core.encoded(body)
        result = dict(job_id=row['job_id'], status='reserved', started_at=now(),
                      input_sha256=row['input_sha256'], semantic_quality='not_evaluated',
                      request=body, request_bytes=len(payload), available_at_seq=None,
                      mode='retrospective_pilot')
        try:
            with dest.open('x', encoding='utf-8') as stream:
                json.dump(result, stream)
        except FileExistsError:
            continue
        started = time.perf_counter()
        try:
            if len(payload) > limits['max_request_bytes']:
                result['status'] = 'blocked_input_budget'
            else:
                request = urllib.request.Request(base_url.rstrip('/') + '/chat/completions',
                    data=payload, headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=240) as response:
                    raw = response.read(limits['max_response_bytes'] + 1)
                result['raw_response_text'] = raw.decode('utf-8', errors='replace')
                if len(raw) > limits['max_response_bytes']:
                    result['raw_response_truncated_by_client'] = True
                    raise ValueError('response byte budget exceeded')
                reply = core.strict_json(raw.decode('utf-8')); result['raw_response'] = reply
                choice = reply['choices'][0]
                if choice['finish_reason'] != 'stop':
                    raise ValueError('incomplete completion')
                claims = core.validate(core.strict_json(choice['message']['content']), row)
                result.update(status='ready' if claims else 'ready_empty', claims=claims)
        except urllib.error.HTTPError as exc:
            result.update(status='failed', error=f'HTTP {exc.code}',
                          http_error_body=exc.read(limits['max_response_bytes']).decode('utf-8', errors='replace'))
        except Exception as exc:
            result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        result.update(seconds=time.perf_counter() - started, completed_at=now())
        save(dest, result)
        print(f'{i}/{len(rows)} {result["status"]}', flush=True)
    counts = summarize(rows, out)
    return 0 if counts['ready'] + counts['ready_empty'] == len(rows) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--out', type=Path, default=ROOT)
    p.add_argument('--source', type=Path, default=Path('pilot/swe_ledger_sensitivity_v1/requests.jsonl'))
    r = sub.add_parser('run')
    r.add_argument('--inputs', type=Path, default=ROOT)
    r.add_argument('--out', type=Path, default=Path('out/swe_claims_v3'))
    r.add_argument('--base-url', default='http://127.0.0.1:30001/v1')
    r.add_argument('--model', default='qwen-local')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out, args.source)
        return 0
    return run(args.inputs, args.out, args.base_url, args.model)


if __name__ == '__main__':
    raise SystemExit(main())
