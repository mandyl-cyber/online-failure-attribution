"""Audit frozen adaptive claim returns and record a focused AI fact review."""
import argparse
import hashlib
from pathlib import Path

from scripts.run_stage_claim_pipeline import read, save
from segmenter import stage_claim_summary as core


# Source-first facts, reviewed against returned text and cited chunks. Selective AI review.
FACT_REVIEW = {
    ('x008',34): ('covered','partial','Both retain runner failure and exit1. Adaptive omits test_sqlite name.'),
    ('x009',37): ('partial','partial','Original only says suite runs. Adaptive gives 20 passing tests. Both omit /testbed/django scope.'),
    ('x009',39): ('partial','partial','Original gives seven tests without explicit pass. Adaptive says seven pass. Both omit runner path.'),
    ('x012',45): ('partial','partial','Original retains missing-runner cause without127. Adaptive retains127 without explicit command-not-found cause.'),
    ('x012',46): ('partial','partial','Original retains missing pytest module without1. Adaptive retains1 without missing-module cause.'),
    ('x015',36): ('partial','partial','Both retain failed import. Original names settings. Adaptive names ImproperlyConfigured and exit1. Neither states all selected details.'),
    ('x015',43): ('partial','partial','Original says tests and exit0. Adaptive adds461. Both omit selected runner scope/skips/expected failures.'),
    ('x015',45): ('unsupported_citation','partial','Original claim gives109 but cites S44/S46/S47 rather than S45. Adaptive cites S45 and gives109/exit0, omitting scope.'),
    ('x015',48): ('partial','partial','Original gives461/11 skipped/2 expected failures. Adaptive gives461/exit0. Both omit runner path.'),
    ('x021',26): ('partial','partial','Both preserve AssertionError. Neither explains the registry becoming empty or preserves exit1 explicitly.'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--returned', type=Path, default=Path('adaptive_seal_claim_v1'))
    parser.add_argument('--inputs', type=Path, default=Path('pilot/adaptive_seal_claim_v1'))
    parser.add_argument('--out', type=Path, default=Path('out/adaptive_seal_claim_return_v1'))
    args = parser.parse_args()
    snapshot = hashlib.sha256()
    for path in sorted(args.returned.glob('*.json')):
        snapshot.update(path.name.encode())
        snapshot.update(path.read_bytes())
    if snapshot.hexdigest() != '7c86e6c8fab1252f3d0414857766cee742030f7f3a4eee26da4fea97e47303c3':
        raise ValueError('AI-reviewed return snapshot changed. Review new outputs before applying annotations.')
    protocol = read(args.inputs/'protocol.json')
    config = read(args.returned/'config.json')
    if config['protocol'] != protocol:
        raise ValueError('Returned protocol mismatch')
    for name,key in [('requests.jsonl','requests_sha256'),('groups.json','groups_sha256'),
                     ('fact_review.json','fact_review_sha256'),('critical_facts.json','critical_facts_sha256')]:
        if hashlib.sha256((args.inputs/name).read_bytes()).hexdigest() != protocol[key]:
            raise ValueError('Input pack changed: '+name)
    rows = {r['job_id']:r for r in map(core.strict_json,(args.inputs/'requests.jsonl').read_text(encoding='utf-8').splitlines())}
    groups = read(args.inputs/'groups.json')
    preflight = read(args.returned/'preflight.json')
    totals = {arm: dict(attempted=0,ready=0,claims=0,seconds=0.) for arm in ('immediate','adaptive')}
    complete_totals = {arm: dict(attempted=0,claims=0,seconds=0.) for arm in totals}
    inventory = []
    for group in groups:
        job_ids = [j for ids in group['arms'].values() for j in ids]
        eligible = all(preflight['prompt_tokens'][j]+protocol['max_new_tokens'] <= protocol['max_length'] for j in job_ids)
        if eligible != preflight['groups'][group['group_id']]:
            raise ValueError('Eligibility mismatch')
        result = dict(group_id=group['group_id'],eligible=eligible,arms={})
        if not eligible:
            inventory.append(result)
            continue
        for arm,ids in group['arms'].items():
            results = []
            for job_id in ids:
                row = rows[job_id]
                core.verify_row(row)
                r = read(args.returned/(job_id+'.json'))
                if r['job_id'] != job_id:
                    raise ValueError('Result identity mismatch')
                if r['status']=='ready':
                    validated = core.validate(core.strict_json(r['text']),row)
                    if validated != r['claims'] or r['token_limit']:
                        raise ValueError('Ready response revalidation failed')
                results.append(r)
                target = totals[arm]
                target['attempted']+=1
                target['ready']+=r['status']=='ready'
                target['claims']+=len(r.get('claims',[]))
                target['seconds']+=r['seconds']
            result['arms'][arm] = results
        result['complete_pair'] = all(r['status']=='ready' for rs in result['arms'].values() for r in rs)
        if result['complete_pair']:
            for arm,rs in result['arms'].items():
                complete_totals[arm]['attempted']+=len(rs)
                complete_totals[arm]['claims']+=sum(len(r['claims']) for r in rs)
                complete_totals[arm]['seconds']+=sum(r['seconds'] for r in rs)
        inventory.append(result)
    complete = {g['group_id'] for g in inventory if g.get('complete_pair')}
    review = []
    for fact in read(args.inputs/'critical_facts.json')['facts']:
        key=(fact['group_id'],fact['source_step_id'])
        if fact['group_id'] not in complete:
            continue
        original,adaptive,note=FACT_REVIEW[key]
        review.append(dict(fact=fact,immediate=original,adaptive=adaptive,note=note,
                           reviewed=True,reviewer_type='AI',scope='selected_fact_coverage_and_citation'))
    summary = dict(total_groups=len(groups),eligible_groups=sum(g['eligible'] for g in inventory),
        complete_pairs=sorted(complete),complete_pair_count=len(complete),all_eligible_requests=totals,
        comparable_complete_pairs=complete_totals,reviewed_selected_facts=len(review),
        caveat='AI review of selected facts. No full claim-level support rate or independent test accuracy.',
        integrity='Frozen metadata and all ready outputs revalidated. Runner did not record per-request input hashes or generation token counts.',
        reviewed_return_sha256=snapshot.hexdigest(),
        decision='No consistent semantic benefit established. Retain as exploratory diagnostics.')
    save(args.out/'inventory.json',inventory)
    save(args.out/'fact_review.json',review)
    save(args.out/'summary.json',summary)
    print(summary)


if __name__=='__main__':
    main()
