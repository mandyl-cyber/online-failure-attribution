"""Run local claim auditing, causal sealing controls and alarm diagnostics.

Usage: python -m scripts.run_stage_claim_pipeline
Only reads frozen model outputs. Never executes commands found in evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from segmenter.delayed_seal import DelayedSeal
from scripts.claim_sidecar_review_v1 import JUDGMENTS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def evidence_map(request):
    return {chunk['id']: chunk['text'] for unit in request['units']
            for field in ('action', 'observation') for record in unit.get(field, [])
            for chunk in record['chunks']}


def audit_claims(root, out):
    pack = root / 'pilot/claim_sidecar_input_ablation_v1'
    returned = root / 'claim_sidecar_input_ablation_v1'
    protocol = read(pack / 'protocol.json')
    config = read(returned / 'config.json')
    rows = {}
    for mode in ('original', 'sidecar_v2'):
        path = pack / mode / 'dev.jsonl'
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        if fingerprint != protocol['file_hashes'][mode] or fingerprint != config['input_hashes'][mode]:
            raise ValueError('Frozen input hash mismatch: ' + mode)
        rows[mode] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    output_path = returned / 'paired_outputs.json'
    if hashlib.sha256(output_path.read_bytes()).hexdigest() != '6339c9a24d37b66382e60f6811cc6fbdbfd60f497cb5291fd0af44b9a92281ed':
        raise ValueError('Reviewed output snapshot changed. Re-review annotations before continuing.')
    outputs = read(output_path)
    by_id = {r['job_id']: r for r in outputs}
    if len(by_id) != len(outputs) or set(by_id) != {r['job_id'] for r in rows['original']}:
        raise ValueError('Missing or duplicate returned pair')
    historical = {r['job_id']: r for r in read(root / 'eval/claim_sft_7b_reviewed_v2_audit/semantic_review.json')}
    packet = []
    for index, (original, variant) in enumerate(zip(rows['original'], rows['sidecar_v2'])):
        if original['job_id'] != variant['job_id'] or original['completion'] != variant['completion']:
            raise ValueError('Pair identity or reference changed')
        pair = by_id[original['job_id']]
        if pair['reference'] != original['completion'][0]['content']:
            raise ValueError('Returned reference mismatch')
        requests = {m: json.loads(r['prompt'][1]['content']) for m, r in [('original', original), ('sidecar_v2', variant)]}
        stripped = [{k: v for k, v in q.items() if k not in ('observed_facts', 'observed_facts_version')} for q in requests.values()]
        if stripped[0] != stripped[1]:
            raise ValueError('Raw source input changed')
        evidence = evidence_map(requests['original'])
        item = dict(index=index, job_id=pair['job_id'], group=original['group'], evidence=evidence, arms={})
        for mode in ('original', 'sidecar_v2'):
            result = pair[mode]
            value = json.loads(result['text'])
            claims = value.get('claims') if isinstance(value, dict) else None
            valid = isinstance(claims, list) and set(value) == {'claims'}
            annotated = []
            for claim in claims or []:
                ids = claim.get('evidence_ids', [])
                valid = valid and set(claim) == {'text', 'source', 'evidence_ids'} and bool(ids)
                valid = valid and isinstance(claim.get('text'), str) and bool(claim['text'].strip())
                valid = valid and claim.get('source') in ('agent_assertion', 'tool_observation')
                valid = valid and len(ids) == len(set(ids)) and all(e in evidence for e in ids)
                previous = historical.get(pair['job_id'], {}).get('adapter', {}).get('claims', [])
                matches = [c for c in previous if all(c.get(k) == claim.get(k) for k in ('text', 'source', 'evidence_ids'))]
                judgment = {k: matches[0][k] for k in ('support', 'note', 'severe', 'error_types')} if matches else None
                origin = 'prior_AI_exact_claim_match' if matches else 'pending_AI_review'
                key = (index, len(annotated))
                if mode == 'sidecar_v2' and not matches and key in JUDGMENTS:
                    support, note, severe = JUDGMENTS[key]
                    judgment = dict(support=support, note=note, severe=severe, error_types=[])
                    origin = 'AI_evidence_review_2026_11_21'
                annotated.append(dict(claim=claim, judgment=judgment,
                                      review_origin=origin))
            item['arms'][mode] = dict(schema_and_ids_valid=bool(valid), hit_token_limit=result['hit_token_limit'],
                                     seconds=result['seconds'], claims=annotated)
        item['changed'] = pair['original']['text'] != pair['sidecar_v2']['text']
        packet.append(item)
    save(out / 'claim_review_packet.json', packet)
    summary = dict(pairs=len(packet), changed_pairs=sum(p['changed'] for p in packet), arms={})
    for mode in ('original', 'sidecar_v2'):
        claims = [c for p in packet for c in p['arms'][mode]['claims']]
        summary['arms'][mode] = dict(schema_and_ids_valid=sum(p['arms'][mode]['schema_and_ids_valid'] for p in packet),
            claims=len(claims), prior_exact_AI_reviews=sum(c['review_origin'] == 'prior_AI_exact_claim_match' for c in claims),
            pending_claim_reviews=sum(c['judgment'] is None for c in claims),
            fully_supported_claims=sum(c['judgment'] is not None and c['judgment']['support'] == 'supported' for c in claims),
            outputs_with_severe_errors=sum(any(c['judgment'] and c['judgment']['severe'] for c in p['arms'][mode]['claims']) for p in packet),
            token_limit_hits=sum(p['arms'][mode]['hit_token_limit'] for p in packet))
    save(out / 'claim_audit.json', summary)
    return summary


def correct_adjacent_review(root, out):
    review = read(root / 'out/adjacent_claim_pair_v1/ai_review.json')
    for item in review['point_coverage']:
        if item['group_id'] == 'x004':
            item['adjacent'] = dict(p1='partial', p2='unsupported')
            item['note'] = 'Adjacent preserves count/skips but omits explicit OK/exit0. Its second claim promotes sed source inspection to testing.'
        if item['group_id'] == 'x011':
            item['adjacent']['p1'] = 'missed'
            item['note'] = 'Adjacent omits Ctrl+C and exit130. Exploration is retained.'
    counts = {}
    for item in review['point_coverage']:
        for arm in ('original', 'adjacent'):
            for value in item[arm].values():
                key = 'yes' if value == 'yes_as_task_assertion' else value
                counts[key] = counts.get(key, 0) + 1
    review['complete_pair_counts'].update(fully_covered=counts.get('yes', 0),
        partially_covered=counts.get('partial', 0), missed=counts.get('missed', 0),
        unsupported=counts.get('unsupported', 0))
    review['pairwise_conclusion'] = 'Calls decrease on six complete pairs. Adjacent summary omits a recorded interruption and promotes source viewing to testing. Semantic improvement remains unestablished.'
    review['correction_note'] = 'Corrected x004 and x011 against returned claim text and source excerpts on 2026-11-21. Original review preserved.'
    save(out / 'adjacent_review_corrected.json', review)
    return dict(status='corrected_AI_review', point_assessments=counts)


def replay_seals(root, out):
    base = root / 'eval/temporal_boundary_v1/external_candidates'
    records = read(base / 'records.json')
    predictions = read(base / 'predictions.json')['fold0']['original']
    results = []
    for row in records:
        cuts = {e['boundary_step_id'] for e in predictions[row['traj_id']]}
        arms = {}
        for delay in (0, 1):
            state = DelayedSeal(delay=delay)
            events = []
            for t in row['ids']:
                decisions = {b: b in cuts for b in range(max(2, t-delay), t+1)}
                events.extend(state.feed(t, decisions))
            events.extend(state.finish())
            covered = [i for e in events for i in range(e['start_step_id'], e['end_step_id']+1)]
            if covered != row['ids']:
                raise ValueError('Stage coverage mismatch')
            arms[str(delay)] = events
        results.append(dict(traj_id=row['traj_id'], arms=arms))
    save(out / 'seal_wait_control.json', results)
    return dict(trajectories=len(results), steps=sum(len(r['ids']) for r in records),
                status='completed_wait_only_control',
                adaptive_revision_status='pending_causal_policy_and_new_claim_generation',
                interpretation='D1 uses identical frozen candidates. It measures waiting behavior and exact step coverage. No accuracy gain is inferred.')


def alarm_candidates(root, out):
    """Flag plan-only verification assertions for review, without deciding risk."""
    rows = [json.loads(line) for line in (root / 'pilot/claim_sidecar_input_ablation_v1/original/dev.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    candidates = []
    for row in rows:
        request = json.loads(row['prompt'][1]['content'])
        for unit in request['units']:
            for record in unit.get('action', []):
                for chunk in record['chunks']:
                    if chunk is not record['chunks'][0]:
                        continue
                    try:
                        action = json.loads(''.join(c['text'] for c in record['chunks']))
                    except ValueError:
                        continue
                    if not isinstance(action, dict):
                        continue
                    for task in action.get('task_list', []):
                        notes = task.get('notes', '')
                        if 'verified' in notes.lower() or 'test passes' in notes.lower():
                            candidates.append(dict(job_id=row['job_id'], group=row['group'], evidence_id=chunk['id'],
                                availability_seq=record['availability_seq'], assertion=notes,
                                status='needs_same_object_failure_and_retest_check', reviewed=False))
    save(out / 'alarm_candidates.json', candidates)
    return dict(candidates=len(candidates), status='candidate_collection_only',
                evaluated_alarm_recall=None, false_alarms_per_trajectory=None,
                blocker='Requires complete chronological same-object failure/retest evidence and reviewed risk labels.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--out', type=Path, default=Path('out/stage_claim_pipeline_v1'))
    parser.add_argument('--adaptive-replay', action='store_true', help='Run neural D0/D1 rescoring and prepare new paired claim jobs')
    parser.add_argument('--local-windows', action='store_true', help='Prepare common-boundary local windows from the frozen v1 replay')
    args = parser.parse_args()
    root = args.root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    state = dict(stages={}, status='running')
    if args.local_windows:
        subprocess.run([sys.executable, '-m', 'scripts.prepare_local_seal_windows'], cwd=root, check=True)
        subprocess.run([sys.executable, '-m', 'remote_pilots.openhands_smoke.adaptive_seal_claims',
                        'check-inputs', '--inputs', 'pilot/adaptive_seal_local_v2'], cwd=root, check=True)
        state['stages']['local_windows'] = read(root/'pilot/adaptive_seal_local_v2/report.json')
    if args.adaptive_replay:
        encoder_python = root / '.venv-encoder/Scripts/python.exe'
        python = str(encoder_python) if encoder_python.exists() else sys.executable
        subprocess.run([python, '-m', 'scripts.prepare_adaptive_seal_claims'], cwd=root, check=True)
        subprocess.run([python, '-m', 'scripts.prepare_adaptive_fact_review'], cwd=root, check=True)
        subprocess.run([python, '-m', 'scripts.review_adaptive_critical_facts'], cwd=root, check=True)
        subprocess.run([python, '-m', 'remote_pilots.openhands_smoke.adaptive_seal_claims', 'check-inputs'], cwd=root, check=True)
        state['stages']['adaptive_replay'] = read(root / 'pilot/adaptive_seal_claim_v1/report.json')
        state['stages']['source_fact_review'] = dict(source_inventory=read(root / 'pilot/adaptive_seal_claim_v1/fact_review_summary.json'),
            reviewed_critical_facts=len(read(root / 'pilot/adaptive_seal_claim_v1/critical_facts.json')['facts']),
            remote_generation='pending_remote_execution', semantic_coverage='selected_AI_facts_only')
    for name, function in [('claim_audit', audit_claims), ('adjacent_review_correction', correct_adjacent_review), ('seal_replay', replay_seals), ('alarm_candidates', alarm_candidates)]:
        try:
            state['stages'][name] = function(root, args.out)
        except Exception as exc:
            state['status'] = 'failed'
            state['failed_stage'] = name
            state['error'] = f'{type(exc).__name__}: {exc}'
            save(args.out / 'summary.json', state)
            raise
        save(args.out / 'summary.json', state)
        print(name, json.dumps(state['stages'][name], ensure_ascii=False), flush=True)
    if (root/'adaptive_seal_claim_v1/summary.json').exists():
        subprocess.run([sys.executable, '-m', 'scripts.audit_adaptive_claim_return',
                        '--out', str(args.out.resolve()/'adaptive_return')], cwd=root, check=True)
        state['stages']['adaptive_return'] = read(args.out/'adaptive_return/summary.json')
    if (root/'adaptive_seal_local_v2/summary.json').exists():
        subprocess.run([sys.executable, '-m', 'scripts.audit_local_seal_return',
                        '--out', str(args.out.resolve()/'local_return')], cwd=root, check=True)
        state['stages']['local_return'] = read(args.out/'local_return/summary.json')
        subprocess.run([sys.executable, '-m', 'scripts.replay_claim_evidence_review'], cwd=root, check=True)
        state['stages']['claim_evidence_review'] = read(root/'out/claim_evidence_review_v1/summary.json')
    state['status'] = 'local_diagnostics_complete_with_research_tasks_pending'
    save(args.out / 'summary.json', state)


if __name__ == '__main__':
    main()
