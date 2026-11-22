"""Apply optional evidence checks to frozen local-window generation results."""
from pathlib import Path
from collections import Counter

from scripts.run_stage_claim_pipeline import read, save
from segmenter import stage_claim_summary as core
from segmenter.claim_evidence_review import review


def main():
    inputs = Path('pilot/adaptive_seal_local_v2')
    returned = Path('adaptive_seal_local_v2')
    out = Path('out/claim_evidence_review_v1')
    preflight = read(returned/'preflight.json')
    rows = {r['job_id']:r for r in map(core.strict_json,(inputs/'requests.jsonl').read_text(encoding='utf-8').splitlines())}
    results=[]
    failures=[]
    for group in read(inputs/'groups.json'):
        if not preflight['groups'][group['group_id']]:
            continue
        for arm,ids in group['arms'].items():
            for job_id in ids:
                result=read(returned/(job_id+'.json'))
                if result['status']!='ready':
                    failures.append(dict(job_id=job_id,error=result.get('error')))
                    continue
                item=review(core.strict_json(result['text']),rows[job_id])
                item.update(group_id=group['group_id'],arm=arm)
                results.append(item)
    save(out/'reviewed_jobs.json',results)
    save(out/'failed_jobs.json',failures)
    summary=dict(jobs=len(results),failed_jobs=len(failures),
        claims=sum(len(r['reviewed_claims']) for r in results),
        flagged_claims=sum(c['status']=='needs_review' for r in results for c in r['reviewed_claims']),
        observation_signals=sum(len(r['observation_signals']) for r in results),
        uncited_signals=sum(len(r['uncited_signal_ids']) for r in results),
        issue_counts=dict(Counter(i['code'] for r in results for c in r['reviewed_claims'] for i in c['issues'])),
        scope='Narrow citation checks and exact source signals. Semantic accuracy unmeasured.')
    save(out/'summary.json',summary)
    print(summary)


if __name__=='__main__':
    main()
