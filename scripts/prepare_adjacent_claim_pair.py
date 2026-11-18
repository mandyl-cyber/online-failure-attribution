"""Freeze paired claim inputs for fold0 stage disagreements on new SWE tasks."""
import hashlib
import json
from pathlib import Path

from segmenter.ledger import Ledger
from segmenter import stage_claim_summary as core


SOURCE = Path('eval/temporal_boundary_v1/external_candidates')
OUT = Path('pilot/adjacent_claim_pair_v1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranges(events, last):
    cuts = [e['boundary_step_id'] for e in events]
    return [(i+1, start, end-1) for i, (start, end) in
            enumerate(zip([1, *cuts], [*cuts, last+1]))]


def source_job(ledger, task, arm, number, start, end, last):
    ops = [ledger.steps[i] for i in range(start, end+1)]
    evidence = [e for op in ops for e in op['evidence']]
    assert len(evidence) == 2*(end-start+1)
    return dict(job_id=f'{ledger.run_id}-{arm}-{number}', run_id=ledger.run_id,
                stage=dict(stage_id=number, start_step_id=start, end_step_id=end,
                           sealed=True, reason='predicted_boundary'),
                detected_at_seq=end+1 if end < last else end,
                incomplete=False, task_context=None, evidence=evidence)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = json.loads((SOURCE/'records.json').read_text(encoding='utf-8'))
    pred = json.loads((SOURCE/'predictions.json').read_text(encoding='utf-8'))['fold0']
    pair_groups, requests = [], []
    for record in rows:
        tid = record['traj_id']
        original = pred['original'][tid]
        adjacent = pred['adjacent'][tid]
        old_cuts = {e['boundary_step_id'] for e in original}
        new_cuts = {e['boundary_step_id'] for e in adjacent}
        if old_cuts == new_cuts:
            continue
        last = record['ids'][-1]
        changed = sorted(old_cuts ^ new_cuts)
        common = sorted({1, last+1} | (old_cuts & new_cuts))
        left = max(x for x in common if x < changed[0])
        right = min(x for x in common if x > changed[-1])
        envelope = [left, right-1]
        ledger = Ledger(SOURCE/'claim_input_probe/journals'/record['key'],
                        f'claim-input-{record["key"]}', defer_claims=True)
        assert list(ledger.steps) == record['ids']
        group = dict(group_id=record['key'], traj_id=tid, task=record['task'],
                     envelope=envelope, old_cuts=sorted(old_cuts),
                     new_cuts=sorted(new_cuts), gold_boundaries=record['boundaries'],
                     arms={})
        for arm, events in [('original', original), ('adjacent', adjacent)]:
            selected = [(number, start, end) for number, start, end in ranges(events, last)
                        if left <= start and end < right]
            assert [i for _, start, end in selected for i in range(start, end+1)] == list(range(left, right))
            ids = []
            for number, start, end in selected:
                row = core.build(source_job(ledger, record['task'], arm, number, start, end, last))
                core.verify_row(row)
                ids.append(row['job_id'])
                requests.append(row)
            group['arms'][arm] = ids
        pair_groups.append(group)
    if len(pair_groups) != 9:
        raise ValueError(f'Expected 9 changed tasks, got {len(pair_groups)}')
    if len({r['job_id'] for r in requests}) != len(requests):
        raise ValueError('Duplicate request ID')
    OUT.mkdir(parents=True)
    (OUT/'requests.jsonl').write_bytes(b''.join(core.encoded(r)+b'\n' for r in requests))
    (OUT/'groups.json').write_bytes(json.dumps(pair_groups, ensure_ascii=False, indent=2).encode('utf-8') + b'\n')
    protocol = dict(version=1, experiment='adjacent-stage-claim-pair',
                    scope='9 fold0 disagreements among 37 previously unused SWE tasks',
                    model='Qwen2.5-32B-Instruct-AWQ (same pinned remote weights for both arms)',
                    claim_protocol=core.PROTOCOL,
                    instruction_sha256=core.digest(core.INSTRUCTION.encode()),
                    schema_sha256=core.digest(core.encoded(core.SCHEMA)),
                    request_sha256=sha(OUT/'requests.jsonl'), groups_sha256=sha(OUT/'groups.json'),
                    boundary_protocol_sha256=sha(SOURCE/'protocol.json'),
                    boundary_result_sha256=sha(SOURCE/'results.json'),
                    calls=len(requests), groups=len(pair_groups),
                    limits=dict(context_tokens=8192, output_tokens=2048,
                                max_request_bytes=100000, max_response_bytes=1000000),
                    selection='All fold0 changed tasks; minimal span between common boundaries enclosing changes. No claim-output selection.',
                    comparison='Same model/prompt/decoding. Compare fixed union of source steps, grouped into original vs adjacent sealed stages.',
                    evaluation=['per-arm schema/provenance success including failures',
                                'claim-level support against cited evidence, AI review clearly labeled',
                                'key fact coverage over same union, reviewed before viewing outputs',
                                'unsupported completion/verification claims',
                                'calls, tokens, request latency, stage seal availability'],
                    caveats=['No human gold semantic claim labels in this pack.',
                             'The nine groups share a previously evaluated boundary model selection.',
                             'Oversize groups must be excluded symmetrically after tokenizer preflight.',
                             'A single merged request can hit context limit when two old stages fit separately.'])
    (OUT/'protocol.json').write_bytes(json.dumps(protocol, ensure_ascii=False, indent=2).encode('utf-8') + b'\n')
    print(json.dumps({'groups':len(pair_groups),'requests':len(requests),
                      'bytes':(OUT/'requests.jsonl').stat().st_size,
                      'per_group':[(g['task'],len(g['arms']['original']),len(g['arms']['adjacent']),g['envelope'])
                                   for g in pair_groups]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
