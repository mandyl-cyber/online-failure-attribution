"""Freeze a candidate-stratified, Qwen-blind Codex semantic audit sample."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .purpose_pilot_remote_v2 import jobs, read, verify_package

QUOTAS={'ACQUIRE':20,'CHECK':15,'UNKNOWN':10,'CONSTRUCT':8,
        'DELIVER':7,'OPERATE':3,'ENABLE':1}
SEED='swe-v2-codex-audit-2026-11-16'
TRAIN_QUOTAS={'ACQUIRE':30,'CHECK':30,'UNKNOWN':20,'CONSTRUCT':20,
              'DELIVER':10,'OPERATE':7,'ENABLE':5}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--profile',choices=('dev','train'),default='dev')
    a=p.parse_args()
    if a.out.exists(): raise SystemExit('Refusing overwrite: '+str(a.out))
    verify_package(a.package)
    review=read(a.review/'review_pending.json')['labels']
    alljobs={j['request_id']:j for j in jobs(a.package)}
    if set(alljobs)!=set(review): raise ValueError('Review/package ID mismatch')
    quotas=TRAIN_QUOTAS if a.profile=='train' else QUOTAS
    seed=SEED+(':train' if a.profile=='train' else '')
    rng=random.Random(seed)
    strata={label:sorted(i for i,x in review.items() if x['candidate']['purpose']==label)
            for label in quotas}
    selected=[]
    for label,quota in quotas.items():
        pool=strata[label]
        chosen=rng.sample(pool,min(quota,len(pool)))
        selected.extend(dict(request_id=i,stratum=label,inclusion_probability=min(quota,len(pool))/len(pool),
                             sampling='probability') for i in chosen)
    already={x['request_id'] for x in selected}
    priority=sorted(i for i,x in review.items()
                    if x['priority_reason']=='check_before_candidate_construction' and i not in already)
    for i in rng.sample(priority,min(8 if a.profile=='train' else 6,len(priority))):
        selected.append(dict(request_id=i,stratum='priority_check',inclusion_probability=None,
                             sampling='targeted'))
    rng.shuffle(selected)
    a.out.mkdir(parents=True)
    with (a.out/'blind_packets.jsonl').open('w',encoding='utf-8') as f:
        for item in selected:
            job=alljobs[item['request_id']]
            f.write(json.dumps(job,ensure_ascii=False)+'\n')
    candidate_sha256={x['request_id']:hashlib.sha256(json.dumps(dict(
        input_sha256=review[x['request_id']]['input_sha256'],
        candidate=review[x['request_id']]['candidate']),ensure_ascii=False,
        sort_keys=True,separators=(',',':')).encode()).hexdigest() for x in selected}
    (a.out/'selection.json').write_text(json.dumps(dict(seed=seed,quota=quotas,
        stratum_sizes={k:len(v) for k,v in strata.items()},items=selected,
        candidate_sha256=candidate_sha256,
        package_manifest_sha256=hashlib.sha256((a.package/'package_manifest.json').read_bytes()).hexdigest(),
        review_sha256=hashlib.sha256((a.review/'review_pending.json').read_bytes()).hexdigest()),indent=2)+'\n')
    (a.out/'ai_labels.json').write_text(json.dumps(dict(reviewer_kind='ai',reviewer_id='codex-blind-v2',
        blind_packets_sha256=hashlib.sha256((a.out/'blind_packets.jsonl').read_bytes()).hexdigest(),
        labels={x['request_id']:dict(label=None,evidence_id=None,note='',reviewed=False) for x in selected}),indent=2)+'\n')
    print(json.dumps(dict(total=len(selected),sampling=dict(Counter(x['sampling'] for x in selected)),
                          strata={k:len(v) for k,v in strata.items()})))


if __name__=='__main__': main()
