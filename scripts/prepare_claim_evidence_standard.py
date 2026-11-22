"""Prepare AI-reviewed development references from v4 failures; never train."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from scripts.run_stage_claim_pipeline import read,save
from scripts.audit_neural_ledger_v3_return import require,content
from segmenter.claim_sft_backend import messages_for,convert_partial
from segmenter import stage_claim_summary as core
from segmenter import stage_claim_facts as facts
from segmenter.ledger import sha,encoded


SPECS=[
    ('swe','00000001','c6','source_is_not_a_diff',
     'The rings.py excerpt rebuilds Add expressions by recursively mapping over expr.args.',
     'tool_observation',7,'observation',['elif expr.is_Add:','reduce(add, list(map(_rebuild, expr.args)))']),
    ('swe','00000001','c9','object_and_event_identity',
     'The polyutils.py excerpt defines parallel_dict_from_expr and _parallel_dict_from_expr.',
     'tool_observation',10,'observation',['def parallel_dict_from_expr(','def _parallel_dict_from_expr(']),
    ('swe','00000005','c1','commit_message_is_intent',
     'The commit message says it changes PolyElement.as_expr to respect user-supplied symbols.',
     'agent_assertion',20,'observation',['rings: respect user-supplied symbols']),
    ('swe','00000006','c1','plan_status_is_agent_assertion',
     'The agent task list marks reproduction and verification done; its notes report pytest was unavailable and verification used a direct script.',
     'agent_assertion',21,'action',['"id":"phase2"','"id":"phase7"','Could not run pytest; environment lacks pytest command.']),
    ('terminal','00000001','c3','short_circuit_leaves_unknown',
     'The chained availability probe prints /usr/bin/apt-get; this output does not establish yum availability.',
     'tool_observation',5,'observation',['/usr/bin/apt-get']),
    ('terminal','00000002','c1','package_is_not_interpreter',
     'The package manager reports setting up python3-setuptools-whl, python3-pip-whl and python3.12-venv.',
     'tool_observation',10,'observation',['Setting up python3-setuptools-whl','Setting up python3-pip-whl','Setting up python3.12-venv']),
    ('terminal','00000002','c2','progress_is_not_completion',
     'The pip output shows the PyArrow download at 24.3/47.7 MB; this cited output does not show installation completion.',
     'tool_observation',11,'observation',['Downloading pyarrow-','24.3/47.7 MB']),
    ('live_smoke','00000004','c2','edit_request_is_not_receipt',
     'The agent requests replacing start < merged[-1][1] with start <= merged[-1][1].',
     'agent_assertion',8,'action',['start < merged[-1][1]','start <= merged[-1][1]']),
]


def load_part(root,session,job,index):
    directory=root/session/'review_backend/extraction'/job
    request=read(directory/'request.json')
    raw=read(directory/'attempt_1.json')['raw_output'][index]
    require(raw['part_index']==index,'Part index changed')
    child=deepcopy(request)
    child['_claim_policy']='v4'
    child['stage']=raw['stage']
    start,end=raw['stage']['start_step_id'],raw['stage']['end_step_id']
    child['evidence']=[e for e in request['evidence'] if start<=e['source_step_id']<=end]
    if raw['page_evidence_ids'] is not None:
        child['_claim_page_ids']=raw['page_evidence_ids']
    row,prompt=messages_for(child)
    require(prompt==raw['model_request'],'Frozen runtime prompt changed')
    visible=child.get('_claim_page_ids',list(row['evidence_map']))
    return row,prompt,visible,core.strict_json(raw['raw_output'])


def select_ids(row,visible,step,field,needles):
    """Select only original chunks overlapping literal anchors on this page."""
    records=[r for r in row['source_snapshot']['evidence'] if r['source_step_id']==step and r['field']==field]
    require(len(records)==1,'Ambiguous reference record')
    record=records[0]
    selected=set()
    for needle in needles:
        start=record['text'].index(needle)
        end=start+len(needle)
        refs=[eid for eid,ref in row['evidence_map'].items() if ref['record_id']==record['record_id']
              and ref['start']<end and ref['end']>start]
        require(refs and set(refs)<=set(visible),'Anchor uses evidence outside the visible page')
        selected.update(refs)
    return [eid for eid in row['evidence_map'] if eid in selected]


def validate_target(response,row,visible):
    converted,rejected=convert_partial(response,row,allowed_ids=visible,enforce_result_scope=True)
    require(not rejected,'Reference failed native validation: '+str(rejected))
    for claim in response['claims']:
        error=facts.guard_claim(claim,row['evidence_map'],facts.extract(row['source_snapshot'],row['evidence_map']))
        require(error is None,'Reference failed observed-fact guard: '+str(error))
    return converted


def build(root):
    issues=read(Path('pilot/neural_ledger_acceptance_v1/v4_semantic_issues.json'))
    issue_map={(r['session'],Path(r['job']).stem,r['claim']['local_id']):r for r in issues}
    require(len(issue_map)==8,'Expected the eight reviewed v4 issues')
    groups,cases={},[]
    def group(session,job,index,control=False):
        key=(session,job,index)
        if key not in groups:
            row,prompt,visible,response=load_part(root,*key)
            groups[key]=dict(row=row,prompt=prompt,visible=visible,original=response,
                             replacements={},additions=[],case_ids=[],control=control)
        return groups[key]
    for session,job,cid,family,text,basis,step,field,needles in SPECS:
        issue=issue_map[(session,job,cid)]
        batch=read(root/session/'claim_batches'/f'{job}.json')
        original=next(c for c in batch['claims'] if c['local_id']==cid)
        require(sha(encoded(content(original)))==issue['claim_sha256'],'Reviewed claim changed')
        directory=root/session/'review_backend/extraction'/job
        parts=read(directory/'attempt_1.json')['raw_output']
        matches=[(p['part_index'],c) for p in parts for c in core.strict_json(p['raw_output'])['claims'] if c['text']==original['text']]
        require(len(matches)==1,'Ambiguous original generated claim')
        index,bad=matches[0]
        item=group(session,job,index)
        ids=select_ids(item['row'],item['visible'],step,field,needles)
        if family=='short_circuit_leaves_unknown':
            ids=select_ids(item['row'],item['visible'],step,'action',['which apt-get && which apk && which yum'])+ids
        reference=dict(text=text,source=basis,evidence_ids=ids)
        validate_target(dict(claims=[reference]),item['row'],item['visible'])
        case_id=f'v4-{session}-{job}-{cid}'
        item['replacements'][bad['text']]=reference
        item['case_ids'].append(case_id)
        cases.append(dict(case_id=case_id,family=family,session=session,job=job,part_index=index,
            bad_claim=bad,reference_claim=reference,reviewer_kind='AI',reviewed=True,training_eligible=False,
            review_note=issue['review_note'],
            evidence_refs=[item['row']['evidence_map'][eid] for eid in ids]))
    failures=[
        (7,'externally_managed_environment','error: externally-managed-environment',
         'The pip command reports an externally-managed-environment error.'),
        (9,'venv_creation_failed','The virtual environment was not created successfully because ensurepip is not\navailable.',
         'Virtual environment creation fails because ensurepip is not available.'),
    ]
    item=group('terminal','00000002',0)
    for step,family,anchor,text in failures:
        ids=select_ids(item['row'],item['visible'],step,'observation',[anchor])
        reference=dict(text=text,source='tool_observation',evidence_ids=ids)
        case_id=f'v4-terminal-00000002-{family}'
        item['additions'].append(reference)
        item['case_ids'].append(case_id)
        cases.append(dict(case_id=case_id,family=family,session='terminal',job='00000002',part_index=0,
            bad_claim=None,reference_claim=reference,reviewer_kind='AI',reviewed=True,training_eligible=False,
            review_note='Critical failure missing from accepted model claims; literal evidence is present within the original seal cutoff.',
            evidence_refs=[item['row']['evidence_map'][eid] for eid in ids]))
    for key in [('swe','00000003',0),('terminal','00000004',0),('live_smoke','00000003',0)]:
        group(*key,control=True)
    rows=[]
    for (session,job,index),item in sorted(groups.items()):
        claims,discarded=[],[]
        for original in item['original']['claims']:
            if original['text'] in item['replacements']:
                claims.append(item['replacements'][original['text']])
            else:
                accepted,rejected=convert_partial(dict(claims=[original]),item['row'],allowed_ids=item['visible'],enforce_result_scope=True)
                if rejected:
                    discarded.extend(rejected)
                else:
                    require(accepted['claims'],'Unexpected empty conversion')
                    claims.append(original)
        claims.extend(item['additions'])
        target=dict(claims=claims)
        validate_target(target,item['row'],item['visible'])
        rows.append(dict(job_id=f'evidence-v1-{session}-{job}-part{index}',group=f'development-session:{session}',
            usage='development_only',training_eligible=False,reviewer_kind='AI',
            source=dict(session=session,job=job,part_index=index,seal_cutoff=item['row']['request']['input_cutoff_seq'],
                        prompt_sha256=sha(encoded(item['prompt'])),visible_evidence_ids=item['visible']),
            case_ids=item['case_ids'],unchanged_control=item['control'],
            discarded_invalid_outputs=discarded,prompt=item['prompt'],
            completion=[dict(role='assistant',content=core.encoded(target).decode())]))
    summary=dict(protocol='claim-evidence-standard-v1',reviewer_kind='AI',cases=len(cases),
        corrected_claims=8,added_critical_failures=2,reference_inputs=len(rows),
        corrected_inputs=sum(not r['unchanged_control'] for r in rows),unchanged_controls=sum(r['unchanged_control'] for r in rows),
        task_groups='Three known engineering sessions; canonical benchmark task-group identity not established.',
        training_started=False,training_eligible=False,independent_acceptance=False,
        preserved_runtime_prompts=True,full_part_targets=True,
        target_scope='Preserve accepted unaffected claims, correct the eight reviewed claims, add two selected failures. Not exhaustive stage-fact gold.',
        issue_snapshot_sha256=sha(encoded(issues)),
        data_sha256={name:sha(encoded(value)) for name,value in [('cases',cases),('references',rows)]})
    return cases,rows,summary


def token_preflight(rows,tokenizer_path):
    from transformers import AutoTokenizer
    from remote_pilots.openhands_smoke.claim_sft_train import encode_reference
    tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_path),local_files_only=True)
    checks=[]
    for row in rows:
        item=encode_reference(tokenizer,row)
        prompt_tokens=item['labels'].count(-100)
        reference_tokens=len(item['input_ids'])-prompt_tokens
        fits=prompt_tokens+512<=8192 and reference_tokens<=512 and len(item['input_ids'])<=8192
        require(fits,'Reference exceeds frozen input/output budgets: '+row['job_id'])
        checks.append(dict(job_id=row['job_id'],prompt_tokens=prompt_tokens,reference_tokens=reference_tokens,fits=True))
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--returns',type=Path,default=Path('neural_ledger_acceptance_v4'))
    parser.add_argument('--out',type=Path,default=Path('pilot/claim_evidence_standard_v1'))
    parser.add_argument('--tokenizer-path',type=Path,default=Path('claim_sft_7b_reviewed_v2/adapter'))
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    cases,rows,summary=build(args.returns)
    checks=token_preflight(rows,args.tokenizer_path)
    summary['tokenizer_preflight']=dict(inputs=len(checks),eligible=sum(c['fits'] for c in checks),max_reference_tokens=max(c['reference_tokens'] for c in checks))
    if args.check:
        existing=[json.loads(line) for line in (args.out/'reference_candidates.jsonl').read_text(encoding='utf-8').splitlines()]
        require(read(args.out/'cases.json')==cases and existing==rows,'Prepared case/reference changed')
        require(read(args.out/'protocol.json')==summary and read(args.out/'token_preflight.json')==checks,'Protocol/preflight changed')
    else:
        args.out.mkdir(parents=True,exist_ok=True)
        save(args.out/'cases.json',cases)
        save(args.out/'protocol.json',summary)
        save(args.out/'token_preflight.json',checks)
        (args.out/'reference_candidates.jsonl').write_bytes(b''.join(core.encoded(r)+b'\n' for r in rows))
    print(summary)


if __name__=='__main__':
    main()
