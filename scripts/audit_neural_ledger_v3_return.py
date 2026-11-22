"""Audit returned whole-step partitions without changing outputs or invoking a GPU."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import shutil
import tempfile

from scripts.run_stage_claim_pipeline import read, save
from scripts.audit_neural_ledger_v2_return import CachedProcessor
from segmenter.claim_extraction import validate_response
from segmenter.claim_sft_backend import messages_for, convert_partial
from segmenter import stage_claim_summary as core
from segmenter.ledger import sha, encoded
from segmenter.online_ledger import OnlineLedger
from remote_pilots.openhands_smoke.neural_ledger_acceptance import successful_timings, distribution


def content(claim):
    return {k: claim[k] for k in ('text', 'basis', 'evidence_refs')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def review_new_claims(rows):
    """Fixed AI citation review for this frozen return; never repairs model text."""
    notes = {
        ('swe', '00000001.json'): [
            ('supported', 'The two plan actions show the status change and reproduction remaining todo.'),
            ('supported', 'Cited source includes def as_expr and its surrounding implementation/comments.'),
            ('supported', 'Cited source includes def ring and related constructors.'),
            ('partial', 'Only the view action is cited. Prefer requested a view; completed viewing is not established by this citation alone.'),
            ('supported', 'The exact supplied-symbol expression is printed in the cited observation.'),
            ('partial', 'The excerpt shows _sort_gens and _analyze_gens. Coefficient analysis is not established by this excerpt.'),
            ('partial', 'Only a view action is cited. Completed viewing is not established; preserve the actual lowercase polyutils.py path.'),
            ('supported', 'The excerpt includes parallel_dict_from_expr and related helpers; file naming capitalization is imprecise.'),
        ],
        ('terminal', '00000001.json'): [
            ('supported', 'The displayed head output contains the header and five rows; ten is the requested maximum, not the observed line count.'),
            ('supported', 'The observation explicitly reports Python 3.12.3 and pip3 command not found.'),
            ('partial', 'The cited observation only prints /usr/bin/apt-get. It does not itself establish negative availability for apk and yum.'),
        ],
        ('terminal', '00000002.json'): [
            ('supported', 'The cited output contains Setting up lines for all three named venv packages.'),
            ('partial', 'The cited observation ends during a pyarrow download and contains no installation completion. Later uncited output cannot repair this claim citation.'),
            ('supported', 'The cited output includes Installing collected packages and Successfully installed.'),
            ('supported', 'The exact interpreter path and Python version are printed.'),
        ],
        ('terminal', '00000003.json'): [
            ('supported', 'The tool reports File created successfully at the specified path; this does not validate script correctness.'),
        ],
    }
    require(sha(encoded(rows)) == 'aea5a732c5dd0508b7cff7523fcb322ce8436b0ae645190377e4ff2eb2f20681',
            'New output snapshot requires a new semantic review')
    result = []
    for row in rows:
        index = int(row['claim']['local_id'][1:])-1
        judgment, note = notes[(row['session'], row['job'])][index]
        result.append(dict(row, claim_sha256=sha(encoded(content(row['claim']))),
            reviewer='AI', judgment=judgment, review_note=note))
    return dict(method='AI review against cited excerpts; no human accuracy claim, no comprehensive fact-recall annotation.',
        judgments=dict(Counter(r['judgment'] for r in result)), rows=result)


def main():
    root = Path('neural_ledger_acceptance_v3')
    pack = Path('pilot/neural_ledger_acceptance_v1')
    out = Path('out/neural_ledger_v3_return_audit')
    out.mkdir(parents=True, exist_ok=True)
    config, protocol = read(root/'config.json'), read(pack/'protocol.json')
    require(config['protocol_sha256'] == sha((pack/'protocol.json').read_bytes()), 'Protocol changed')
    restored = Path(tempfile.mkdtemp(prefix='restore-', dir=out))
    sessions, records, new_claims, part_metrics = [], [], [], []
    prior_count = 0
    for session in protocol['sessions']:
        name = session['session_id']
        directory = root/name
        original = [json.loads(line) for line in (pack/session['path']).read_text(encoding='utf-8').splitlines()]
        native = [r['event'] for r in (read(p) for p in sorted((directory/'native_events').glob('*.json'))) if r.get('kind') == 'event']
        require(original == native, f'{name}: event stream changed')
        jobs = {p.name: read(p) for p in (directory/'stage_jobs').glob('*.json')}
        batches = {p.name: read(p) for p in (directory/'claim_batches').glob('*.json')}
        require(set(jobs) == set(batches), f'{name}: missing/extra receipts')
        for filename, batch in sorted(batches.items()):
            job = jobs[filename]
            require(batch['input_sha256'] == sha(encoded(job)), 'Input fingerprint mismatch')
            require(batch['evidence_review_identity'] == config['processor'], 'Processor fingerprint mismatch')
            audit = directory/'review_backend/extraction'/Path(filename).stem
            state, request, attempt = read(audit/'status.json'), read(audit/'request.json'), read(audit/'attempt_1.json')
            require(state['attempts'] == 1 and len(list(audit.glob('attempt_*.json'))) == 1, 'Repeated extraction')
            details = batch['extraction_details']
            raw_parts = attempt['raw_output']
            require(len(raw_parts) == len(details['parts']), 'Part count mismatch')
            covered, rebuilt, rejected = [], [], []
            for raw, part in zip(raw_parts, details['parts']):
                require({k: v for k, v in raw.items() if k not in ('raw_output', 'model_request')} == part, 'Part metadata changed')
                child = deepcopy(request)
                child['stage'] = part['stage']
                start, end = part['stage']['start_step_id'], part['stage']['end_step_id']
                child['evidence'] = [e for e in request['evidence'] if start <= e['source_step_id'] <= end]
                covered.extend(range(start, end+1))
                row, messages = messages_for(child)
                require(messages == raw['model_request'], 'Partition input or seal cutoff changed')
                require(part['status'] == 'generated', 'Generation failed: inspect new failure')
                metrics = part['metrics']
                require(metrics['input_tokens'] + 512 <= 8192 and metrics['output_tokens'] < 512, 'Budget violation')
                converted, invalid = convert_partial(core.strict_json(raw['raw_output']), row)
                rebuilt.extend(converted['claims'])
                rejected.extend(dict(r, part_index=part['part_index']) for r in invalid)
                part_metrics.append(metrics)
            full = list(range(job['stage']['start_step_id'], job['stage']['end_step_id']+1))
            require(sorted(covered + details['oversized_steps']) == full, 'Missing or repeated steps')
            require(rejected == details['rejected_claims'], 'Rejected output changed')
            require(not details['duplicate_claims'], 'New deduplication case requires review')
            require([content(c) for c in rebuilt] == [content(c) for c in batch['claims']], 'Claim text/source/citation changed')
            response = dict(claims=[{k: c[k] for k in ('local_id', 'kind', 'text', 'basis', 'objects', 'conditions', 'unknowns', 'evidence_refs')} for c in batch['claims']])
            validate_response(response, request)
            complete = not details['oversized_steps'] and not rejected
            expected_status = 'ready' if complete else 'partial' if batch['claims'] else 'evidence_only'
            require(details['complete'] == complete and batch['status'] == expected_status, 'Completeness/status mismatch')
            retained = []
            if not complete:
                retained = [e for e in job['evidence'] if e['field'] == 'observation']
                require(batch['review_status'] == 'needs_review' and batch['retained_observations'] == retained, 'Incomplete evidence omitted')
                require(all(sha(e['text'].encode()) == e['content_sha256'] for e in retained), 'Observation hash mismatch')
            for signal in batch['observation_signals']:
                ev = next(e for e in job['evidence'] if e['record_id'] == signal['record_id'])
                require(ev['text'][signal['start']:signal['end']] == signal['quote'] and signal['available_after_seq'] == job['detected_at_seq'], 'Signal provenance mismatch')
            old = read(Path('neural_ledger_acceptance_v2')/name/'claim_batches'/filename)
            if old['claims']:
                require([content(c) for c in old['claims']] == [content(c) for c in batch['claims']], 'Previously reviewed claims changed')
                prior_count += len(old['claims'])
            else:
                new_claims.extend(dict(session=name, job=filename, claim=c) for c in batch['claims'])
            records.append(dict(session=name, job=filename, status=batch['status'], claims=len(batch['claims']),
                parts=len(raw_parts), rejected_claims=len(rejected), oversized_steps=details['oversized_steps'],
                retained_observations=len(retained)))
        copy = restored/name
        shutil.copytree(directory, copy)
        before = {p.name: p.read_bytes() for p in (copy/'claim_batches').glob('*.json')}
        online = read(copy/'online.json')
        runtime = OnlineLedger(copy, name, background=False, claim_processor=CachedProcessor(config['processor']), input_mode=online['input_mode'])
        runtime.drain()
        runtime.close()
        require(before == {p.name: p.read_bytes() for p in (copy/'claim_batches').glob('*.json')}, 'Cached recovery changed receipts')
        sessions.append(dict(session=name, statuses=dict(Counter(b['status'] for b in batches.values())),
            cached_recovery_passed=True,
            full_timings=successful_timings(directory, {n for n, b in batches.items() if b['status'] == 'ready'}),
            partial_timings=successful_timings(directory, {n for n, b in batches.items() if b['status'] == 'partial'})))
    review = review_new_claims(new_claims)
    result = dict(sessions=sessions, jobs=len(records), receipts=len(records), statuses=dict(Counter(r['status'] for r in records)),
        claims=prior_count+len(new_claims), claims_matching_v2=prior_count, new_claims=len(new_claims),
        new_claim_semantic_review=dict(method=review['method'], judgments=review['judgments']),
        generation_parts=len(part_metrics), rejected_claims=sum(r['rejected_claims'] for r in records),
        retained_observations=sum(r['retained_observations'] for r in records),
        oversized_steps=[dict(session=r['session'], job=r['job'], steps=r['oversized_steps']) for r in records if r['oversized_steps']],
        generation_seconds_per_part=distribution([m['generation_seconds'] for m in part_metrics]),
        peak_gpu_GiB=max(m['peak_gpu_bytes'] for m in part_metrics)/1024**3,
        scope='Three frozen native event streams replayed at 0.05s/event with default phase rules. Structural completeness is not semantic correctness. Local recovery reuses cached results; no forced GPU crash tested.')
    save(out/'summary.json', result)
    save(out/'batches.json', records)
    save(out/'new_claims.json', new_claims)
    save(out/'new_claim_ai_review.json', review)
    print(json.dumps({k: v for k, v in result.items() if k != 'sessions'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
