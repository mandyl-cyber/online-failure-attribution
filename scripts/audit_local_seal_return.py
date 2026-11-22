"""Revalidate local-window generation and record source-first AI fact checks."""
import argparse
import hashlib
from pathlib import Path

from scripts.run_stage_claim_pipeline import read, save
from scripts.audit_adaptive_claim_return import FACT_REVIEW
from segmenter import stage_claim_summary as core


LOCAL_REVIEW = {
    ('x027',25): ('partial','covered','Both retain failed replacement. Adaptive explicitly retains old_str mismatch as cause.'),
    ('x027',27): ('partial','covered','Original says neither version replaced. Adaptive also retains old_str mismatch.'),
    ('x027',29): ('covered','covered','Both cite the failed attempts and retain no replacement.'),
    ('x027',30): ('covered','covered','Both cite the failed attempts and retain no replacement.'),
    ('x031',19): ('missed','partial','Immediate omits S19 failure and compresses attempts with later edit. Adaptive explicitly cites S19 no replacement, omitting its exact cause.'),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--returned',type=Path,default=Path('adaptive_seal_local_v2'))
    parser.add_argument('--out',type=Path,default=Path('out/adaptive_seal_local_return_v2'))
    args=parser.parse_args()
    pack=Path('pilot/adaptive_seal_local_v2')
    config=read(args.returned/'config.json')
    protocol=read(pack/'protocol.json')
    if config['protocol']!=protocol:
        raise ValueError('Protocol mismatch')
    snapshot=hashlib.sha256()
    for path in sorted(args.returned.glob('*.json')):
        snapshot.update(path.name.encode())
        snapshot.update(path.read_bytes())
    if snapshot.hexdigest()!='ef148dcdf98be7162b24e66312a0aaa953959b66507cb0c973eecdd8dc3e809c':
        raise ValueError('Reviewed return changed. Re-review before applying AI annotations.')
    for name,key in [('requests.jsonl','requests_sha256'),('groups.json','groups_sha256'),
                     ('fact_review.json','fact_review_sha256'),('critical_facts.json','critical_facts_sha256')]:
        if hashlib.sha256((pack/name).read_bytes()).hexdigest()!=protocol[key]:
            raise ValueError('Frozen input changed')
    rows={r['job_id']:r for r in map(core.strict_json,(pack/'requests.jsonl').read_text(encoding='utf-8').splitlines())}
    groups=read(pack/'groups.json')
    preflight=read(args.returned/'preflight.json')
    old_config=read(Path('adaptive_seal_claim_v1/config.json'))
    if old_config['adapter_sha256']!=config['adapter_sha256']:
        raise ValueError('Adapter changed')
    inventory=[]
    totals={arm:dict(requests=0,claims=0,seconds=0.) for arm in ('immediate','adaptive')}
    reused,new,ready=0,0,0
    for group in groups:
        ids=[j for members in group['arms'].values() for j in members]
        eligible=all(preflight['prompt_tokens'][j]+protocol['max_new_tokens']<=protocol['max_length'] for j in ids)
        if eligible!=preflight['groups'][group['group_id']]:
            raise ValueError('Preflight eligibility mismatch')
        item=dict(group_id=group['group_id'],parent_group_id=group['parent_group_id'],eligible=eligible,arms={})
        if not eligible:
            inventory.append(item)
            continue
        for arm,members in group['arms'].items():
            results=[]
            for job_id in members:
                row=rows[job_id]
                core.verify_row(row)
                result=read(args.returned/(job_id+'.json'))
                if result['job_id']!=job_id:
                    raise ValueError('Returned job mismatch')
                if result['status']=='ready':
                    if core.validate(core.strict_json(result['text']),row)!=result['claims'] or result['token_limit']:
                        raise ValueError('Ready validation failed')
                    ready+=1
                if 'reused_from' in result:
                    previous=Path('adaptive_seal_claim_v1')/(job_id+'.json')
                    if hashlib.sha256(previous.read_bytes()).hexdigest()!=result['reused_result_sha256']:
                        raise ValueError('Reuse hash mismatch')
                    old=read(previous)
                    if any(result[k]!=v for k,v in old.items()):
                        raise ValueError('Reused result content mismatch')
                    reused+=1
                else:
                    if result['input_sha256']!=row['input_sha256'] or result['prompt_tokens']!=preflight['prompt_tokens'][job_id]:
                        raise ValueError('Fresh generation input mismatch')
                    new+=1
                results.append(result)
            item['arms'][arm]=results
        item['complete']=all(r['status']=='ready' for rs in item['arms'].values() for r in rs)
        if item['complete']:
            for arm,results in item['arms'].items():
                totals[arm]['requests']+=len(results)
                totals[arm]['claims']+=sum(len(r['claims']) for r in results)
                totals[arm]['seconds']+=sum(r['seconds'] for r in results)
        inventory.append(item)
    complete={g['group_id'] for g in inventory if g.get('complete')}
    points=[]
    judgments=FACT_REVIEW.copy()
    judgments.update(LOCAL_REVIEW)
    for fact in read(pack/'critical_facts.json')['facts']:
        if fact['group_id'] not in complete:
            continue
        original,adaptive,note=judgments[(fact['parent_group_id'],fact['source_step_id'])]
        points.append(dict(fact=fact,reviewer_type='AI',reviewed=True,immediate=original,adaptive=adaptive,note=note))
    summary=dict(windows=len(groups),eligible_windows=sum(g['eligible'] for g in inventory),
        complete_windows=len(complete),complete_parent_tasks=len({g['parent_group_id'] for g in inventory if g.get('complete')}),
        reused_requests=reused,new_requests=new,ready=ready,failed=reused+new-ready,
        complete_pair_totals=totals,selected_facts_reviewed=len(points),
        snapshot_sha256=snapshot.hexdigest(),
        conclusion='Local grouping improves evaluation coverage. Selected facts show individual benefits and omissions. Overall adaptive policy benefit remains unestablished.',
        limitations=['Windows within a task are correlated.','AI review covers selected facts, not all claims.',
                     'Cost totals mix reused v1 timings and six fresh requests.','Previously evaluated tasks, no independent test claim.'])
    save(args.out/'inventory.json',inventory)
    save(args.out/'fact_review.json',points)
    save(args.out/'summary.json',summary)
    save(args.out/'new_output_findings.json',[
        dict(group_id='x027_s25_30',arm='adaptive',job_id='7d20e3c06f4555bdeee589754b29d216de4819e62855ef8f55172c557185de00',claim_index=1,
             finding='The claim says docinfo_re has a space after the colon. The cited S26 line has no such space. This is an unsupported literal-code detail.'),
        dict(group_id='x031_s19_25',arm='immediate',job_id='03df60776331400fd09066ad4b065c2a1c031b5601e7c7fca41ae7d0a86458f1',claim_index=0,
             finding='Claim omits failed S19 replacement and cites action requests for a completed change. S25 separately confirms a later edit, so failure and later success must remain distinct.')])
    print(summary)


if __name__=='__main__':
    main()
