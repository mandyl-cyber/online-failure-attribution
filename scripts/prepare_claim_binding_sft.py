"""Build a versioned SFT candidate; diagnostic tasks become training, not test."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from scripts.run_stage_claim_pipeline import read,save
from scripts.audit_neural_ledger_v3_return import require
from segmenter.ledger import sha,encoded
from segmenter import stage_claim_summary as core
from segmenter.claim_extraction import build_request
from segmenter.claim_sft_backend import messages_for,convert_partial


def verify_package(path):
    protocol=read(path/'protocol.json')
    for split,digest in protocol['data_hashes'].items():
        require(sha((path/f'{split}.jsonl').read_bytes())==digest,'Data hash changed: '+split)
    rows={s:[json.loads(x) for x in (path/f'{s}.jsonl').read_text(encoding='utf-8').splitlines()] for s in ('train','dev')}
    require(not {r['group'] for r in rows['train']} & {r['group'] for r in rows['dev']},'Task-group overlap')
    aliases=protocol['canonical_group_aliases']
    regression=[json.loads(x) for x in (path/'regression.jsonl').read_text(encoding='utf-8').splitlines()]
    canonical={s:{aliases[r['group']] for r in values} for s,values in dict(rows,regression=regression).items()}
    require(not canonical['train'] & (canonical['dev']|canonical['regression']),'Canonical evaluation task overlaps training')
    require(protocol['training_eligible'] is True and protocol['independent_acceptance'] is False,'Wrong usage protocol')
    return rows,protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('pilot/claim_binding_sft_v1'))
    parser.add_argument('--tokenizer-path',type=Path,default=Path('claim_sft_7b_reviewed_v2/adapter'))
    args=parser.parse_args()
    source=Path('pilot/claim_sft_reviewed_v4_scopefix')
    frozen=Path('pilot/claim_prospective_eval_v1')
    facts=read(frozen/'references.json')['facts']
    train=[json.loads(x) for x in (source/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(train)==100,'Legacy train changed')
    records=[json.loads(x) for x in (frozen/'stage_jobs.jsonl').read_text(encoding='utf-8').splitlines()]
    require(sha((frozen/'references.json').read_bytes())==read(frozen/'protocol.json')['reference_sha256'],'Frozen reference drift')
    provenance=[]
    for item in records:
        job=item['job']
        request=build_request(job)
        request['_claim_policy']='v4'
        row,prompt=messages_for(request)
        selected=[f for f in facts if f['sealed_job_id']==job['job_id']]
        require(selected,'Empty supervised target')
        claims,bindings=[],[]
        for fact in selected:
            primary=[eid for eid,ref in row['evidence_map'].items() if ref['record_id']==fact['record_id']]
            require(primary,'No primary evidence for fact')
            context=[eid for eid,ref in row['evidence_map'].items() if ref['source_step_id']==fact['step_id']
                     and ref['field']=='action'] if fact['basis']=='tool_observation' else []
            ids=primary+[eid for eid in context if eid not in primary]
            claim=dict(evidence_ids=ids,source=fact['basis'],text=fact['text'])
            response,rejected=convert_partial(dict(claims=[claim]),row,enforce_result_scope=True)
            require(not rejected and response['claims'],'Invalid reference: '+str(rejected))
            require(all(row['evidence_map'][eid]['field']==fact['source_field'] for eid in primary),'Wrong evidence role')
            claims.append(claim)
            bindings.append(dict(fact_id=fact['fact_id'],primary_evidence_ids=primary,context_evidence_ids=context,
                                 category=fact['category'],basis=fact['basis']))
        candidate=dict(job_id='binding-v1:'+job['job_id'],group='authored-task:'+job['run_id'],
            parent_stage_key=job['job_id'],prompt=prompt,
            completion=[dict(role='assistant',content=core.encoded(dict(claims=claims)).decode())],
            reviewer_type='AI',training_eligible=True,origin='retired_prospective_diagnostic',
            references_frozen_before_original_model_outputs=True)
        train.append(candidate)
        provenance.append(dict(job_id=candidate['job_id'],session=item['session'],original_job_id=job['job_id'],
            input_sha256=item['input_sha256'],bindings=bindings,usage='training_not_acceptance'))
    regression=Path('pilot/claim_evidence_standard_v1/reference_candidates.jsonl').read_bytes()
    audited=read(Path('pilot/claim_sft_next_v1/task_provenance.json'))
    aliases={r['source_group']:r['canonical_group'] for s in ('train_tasks','dev_tasks') for r in audited[s]}
    for row in train[100:]:aliases[row['group']]=row['group']
    for session in audited['known_development_sessions']:
        aliases['development-session:'+session['session']]=session['canonical_group'] or 'authored-task:legacy-ranges-smoke'
    payloads=dict(train=b''.join(core.encoded(r)+b'\n' for r in train),dev=(source/'dev.jsonl').read_bytes(),regression=regression)
    args.out.mkdir(parents=True,exist_ok=True)
    for split,payload in payloads.items():
        path=args.out/f'{split}.jsonl'
        if path.exists():require(path.read_bytes()==payload,'Candidate data changed; version the dataset')
        else:path.write_bytes(payload)
    from transformers import AutoTokenizer
    from remote_pilots.openhands_smoke.claim_sft_train import encode_reference
    tokenizer=AutoTokenizer.from_pretrained(str(args.tokenizer_path),local_files_only=True)
    checks=[]
    for split,payload in payloads.items():
        for r in [json.loads(x) for x in payload.decode().splitlines()]:
            item=encode_reference(tokenizer,r)
            p=item['labels'].count(-100)
            n=len(item['input_ids'])-p
            require(p+512<=8192 and n<=512 and len(item['input_ids'])<=8192,'Budget exceeded: '+r['job_id'])
            checks.append(dict(split=split,job_id=r['job_id'],prompt_tokens=p,reference_tokens=n,eligible=True))
    protocol=dict(protocol='claim-binding-sft-v1',training_eligible=True,reviewer_kind='AI',
        independent_acceptance=False,training_started=False,train=len(train),dev=38,regression=11,
        legacy_train=100,targeted_native_stages=len(provenance),targeted_facts=sum(len(r['bindings']) for r in provenance),
        legacy_reference_revision='Includes prior corrections in train windows 9,39,42,55; effects not isolated from native-stage augmentation.',
        role_change='The four opened prospective tasks are retired into training diagnostics. Original first-pass evaluation and frozen files remain unchanged; they cannot establish candidate generalization.',
        dev_bytes_unchanged=True,regression_usage='Previously used development controls, not blind acceptance',
        data_hashes={s:sha(v) for s,v in payloads.items()},
        canonical_group_aliases=aliases,
        original_reference_sha256=sha((frozen/'references.json').read_bytes()),
        token_budget=dict(max_length=8192,max_new_tokens=512),
        scope='Same output schema, prompts and 7B family. Bind observed outcomes to observation IDs first, retain action context and specific conditions. No new inference module or RL.')
    save(args.out/'protocol.json',protocol)
    save(args.out/'evidence_bindings.json',provenance)
    save(args.out/'token_preflight.json',checks)
    verify_package(args.out)
    print(dict(train=len(train),dev=38,regression=11,new_native_facts=protocol['targeted_facts'],
               budget_passed=len(checks),max_reference_tokens=max(c['reference_tokens'] for c in checks),training_started=False))


if __name__=='__main__':main()
