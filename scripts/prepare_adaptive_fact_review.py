"""Freeze source-first review points and stage membership before generation."""
import hashlib
import re
from pathlib import Path

from scripts.run_stage_claim_pipeline import read, save
from segmenter import stage_claim_summary as core


SIGNALS = {
    'failure_or_interruption': r'(?i)Traceback|NotImplementedError|ModuleNotFoundError|command not found|No module named|No replacement was performed|KeyboardInterrupt|\bFAILED\b|exit code (?:[1-9]\d*)',
    'verification_result': r'(?im)^Ran \d+ tests?|^OK(?:\s|$)|\d+ passed|\d+ failed|^All good|^Pickle OK|^EXIT:\s*\d+',
    'edit_receipt': r'(?i)has been edited|has been created|file created|file written',
    'agent_status_assertion': r'(?i)verified|tests? pass|fix implemented|completed|"status"\s*:\s*"done"',
}


def main():
    root = Path('pilot/adaptive_seal_claim_v1')
    groups = read(root / 'groups.json')
    rows = {r['job_id']: r for r in map(core.strict_json, (root / 'requests.jsonl').read_text(encoding='utf-8').splitlines())}
    packet = []
    for group in groups:
        arm_records = {}
        memberships = {}
        for arm, jobs in group['arms'].items():
            evidence = {}
            memberships[arm] = []
            covered = []
            for job_id in jobs:
                row = rows[job_id]
                core.verify_row(row)
                request = row['request']
                stage = request['stage']
                covered.extend(range(stage['start_step_id'], stage['end_step_id']+1))
                memberships[arm].append(dict(job_id=job_id, stage=stage, cutoff=request['input_cutoff_seq']))
                for record in row['source_snapshot']['evidence']:
                    key = (record['source_step_id'], record['field'])
                    if key in evidence:
                        raise ValueError('Duplicate evidence membership')
                    evidence[key] = record
            if covered != list(range(group['envelope'][0], group['envelope'][1]+1)):
                raise ValueError('Envelope coverage mismatch')
            arm_records[arm] = evidence
        if arm_records['immediate'] != arm_records['adaptive']:
            raise ValueError('Paired source evidence changed')
        points = []
        for (step, field), record in sorted(arm_records['immediate'].items()):
            text = record['text']
            categories = [name for name, expression in SIGNALS.items()
                          if re.search(expression, text) and (field == 'observation' or name == 'agent_status_assertion')]
            if not categories:
                continue
            memberships_for_point = {arm: next(s for s in stages if s['stage']['start_step_id'] <= step <= s['stage']['end_step_id'])
                                     for arm, stages in memberships.items()}
            points.append(dict(point_id=f'{group["group_id"]}:S{step}:{field}', source_step_id=step,
                field=field, categories=categories, source_text=text,
                source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                membership=memberships_for_point, reviewed=False, reviewer_type=None,
                final_fact=None, coverage=dict(immediate=None, adaptive=None)))
        packet.append(dict(group_id=group['group_id'], task=group['task'], envelope=group['envelope'],
            source_equal=True, stages=memberships, candidate_points=points,
            reviewer_notes='', reviewed=False))
    save(root / 'fact_review.json', dict(scope='Source-first rule-selected candidates across all 18 disagreement tasks',
        exhaustive=False, semantics='Exact source excerpts. Categories require AI or human review, particularly code listings and agent assertions.',
        groups=packet))
    summary = dict(groups=len(packet), source_equal_groups=sum(g['source_equal'] for g in packet),
        candidate_points=sum(len(g['candidate_points']) for g in packet),
        reviewed_groups=0, evidence_coverage='identical raw records and complete step ranges in both arms',
        status='source_candidate_inventory_complete_semantic_review_pending')
    save(root / 'fact_review_summary.json', summary)
    protocol = read(root / 'protocol.json')
    protocol.update(groups_sha256=hashlib.sha256((root/'groups.json').read_bytes()).hexdigest(),
                    fact_review_sha256=hashlib.sha256((root/'fact_review.json').read_bytes()).hexdigest())
    save(root / 'protocol.json', protocol)
    print(summary)


if __name__ == '__main__':
    main()
