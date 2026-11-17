"""Freeze a small diagnostic set; no teacher outputs are reference labels."""
import json
from pathlib import Path
from segmenter import stage_claim_summary as core


def main():
    root = Path('pilot/swe_claim_sft_v1')
    out = Path('pilot/seal_claim_diagnostic_v1')
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    rows = {}
    for split in ('train', 'dev'):
        for line in (root/f'{split}_inputs.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            core.verify_row(row)
            rows[row['job_id']] = row
    groups = sorted({m['group'] for m in manifest if m['split'] in ('train', 'dev')})
    selected = []
    for i, group in enumerate(groups):
        options = sorted([m for m in manifest if m['group'] == group], key=lambda m:(m['input_bytes'], m['job_id']))
        # Rotate short/median/long among task groups, without looking at outputs.
        selected.append(options[[0, len(options)//2, len(options)-1][i % 3]])
    prompts = dict(baseline=core.INSTRUCTION, evidence_first=core.INSTRUCTION + '\n'
        'Select supporting evidence before composing each claim. For every selected point, '
        'identify whether it is an intention, an attempted action, or an observed result. '
        'Write one narrowly supported proposition, preserving that status explicitly in text. '
        'Include the action citation when the observation alone does not identify the object or scope. '
        'Do not output intermediate reasoning or extra fields. Before finishing, check coverage of '
        'material changes, verification outcomes and unresolved blockers actually visible in this stage.')
    requests, review = [], []
    for m in selected:
        row = rows[m['job_id']]
        for variant, instruction in prompts.items():
            requests.append(dict(job_id=row['job_id'], variant=variant, input_sha256=row['input_sha256'],
                messages=[dict(role='system', content=instruction),
                          dict(role='user', content=core.encoded(row['request']).decode())]))
        review.append(dict(job_id=row['job_id'], reviewed=False, reviewer_type=None,
                           input_complete=None, required_points=[], excluded_inferences=[],
                           claim_judgments=[], note='Reference points must be written before inspecting candidates.'))
    out.mkdir(parents=True, exist_ok=False)
    def save(name, obj):
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save('selection.json',selected)
    save('review.json',review)
    (out/'inputs.jsonl').write_bytes(b''.join(core.encoded(rows[m['job_id']])+b'\n' for m in selected))
    (out/'requests.jsonl').write_bytes(b''.join(core.encoded(r)+b'\n' for r in requests))
    old=json.loads(Path('eval/boundary_confirmation/summary.json').read_text(encoding='utf-8'))
    save('prior_confirmation.json',dict(metrics={k:{n:v[n] for n in ('tol1_eventual','tol1_D0','tol1_D1','tol1_D2')}
             for k,v in old['metrics'].items()},comparisons=old['comparisons'],
             source='eval/boundary_confirmation/summary.json',
             source_sha256=core.digest(Path('eval/boundary_confirmation/summary.json').read_bytes())))
    save('protocol.json',dict(status='prepared_only_not_run',stages=len(selected),requests=len(requests),
         instruction_sha256={k:core.digest(v.encode()) for k,v in prompts.items()},
         requests_sha256=core.digest((out/'requests.jsonl').read_bytes()),
         selection='one short/median/long stage per train/dev task group, rotating sorted group order',
         limits=dict(temperature=0,max_output_tokens=2048,no_silent_truncation=True),
         evaluation=['structural_validity','severe_factual_errors','source_correctness','citation_sufficiency',
                     'required_point_coverage','output_tokens','latency'],
         caveats=['Seen-task diagnostic only; no independent holdout claim.',
                  'Gold stage inputs isolate extraction; online-stage robustness remains untested.',
                  'AI review is not human gold; reference points and judgments are pending.',
                  'Prompt experiment only, not a guarantee of semantic verification.']))
    print(json.dumps(dict(stages=len(selected),requests=len(requests),out=str(out))))


if __name__ == '__main__':
    main()
