"""Length preflight and paired claim requests. No truncation or automatic retry."""
import argparse
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
import time
import urllib.request
from remote_pilots.openhands_smoke import claims_v3 as io
from segmenter import stage_claim_summary as core

ROOT=Path('pilot/seal_claim_diagnostic_v3')


def load(root):
    protocol=io.read(root/'protocol.json')
    for name,key in [('inputs.jsonl','input_sha256'),('requests.jsonl','requests_sha256')]:
        if core.digest((root/name).read_bytes())!=protocol[key]: raise ValueError('pack drift: '+name)
    rows=io.rows_at(root/'inputs.jsonl'); requests=io.rows_at(root/'requests.jsonl')
    byid={r['job_id']:r for r in rows}
    if len(byid)!=len(rows): raise ValueError('duplicate inputs')
    for row in rows: core.verify_row(row)
    variants=protocol.get('variants',['baseline','evidence_first'])
    if len(variants)!=2 or len(set(variants))!=2: raise ValueError('expected two distinct arms')
    seen=set()
    for r in requests:
        key=(r['job_id'],r['variant'])
        if key in seen: raise ValueError('duplicate request')
        seen.add(key)
        row=byid[r['job_id']]
        expected_input=row['request']
        if protocol.get('input_transform')=='source_roles_v1' and r['variant']=='source_typed':
            from segmenter.claim_source_input import render
            expected_input=render(row)
        if (r['input_sha256']!=row['input_sha256'] or
            r['messages'][1]!=dict(role='user',content=core.encoded(expected_input).decode()) or
            r['messages'][0]['role']!='system' or len(r['messages'])!=2 or
            core.digest(r['messages'][0]['content'].encode())!=protocol['instruction_sha256'][r['variant']]):
            raise ValueError('request does not match frozen input/instruction')
    if seen!={(j,v) for j in byid for v in variants}: raise ValueError('unpaired requests')
    return protocol,byid,requests


def count_tokens(tokenizer,messages):
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
    if isinstance(ids,Mapping): ids=ids['input_ids']
    if hasattr(ids,'tolist'): ids=ids.tolist()
    if isinstance(ids,list) and len(ids)==1 and isinstance(ids[0],list): ids=ids[0]
    if not isinstance(ids,list) or not ids or any(type(i) is not int for i in ids):
        raise ValueError('unexpected tokenizer output')
    return len(ids)


def input_issues(row):
    issues=[]
    if row['request']['completeness']['upstream_incomplete']: issues.append('upstream_incomplete')
    for unit in row['request']['units']:
        for field in ('action','observation'):
            text=''.join(c['text'] for p in unit[field] for c in p['chunks'])
            if not text.strip(): issues.append(f"{unit['step_id']}:{field}:missing_or_blank")
    return issues


def preflight(root,model_path,context,out):
    from transformers import AutoTokenizer
    protocol,rows,requests=load(root)
    config=io.read(model_path/'config.json')
    if context<=2048 or context>config['max_position_embeddings']: raise ValueError('invalid context budget')
    tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True)
    checks=[]
    for r in requests:
        tokens=count_tokens(tokenizer,r['messages'])
        checks.append(dict(job_id=r['job_id'],variant=r['variant'],prompt_tokens=tokens,
                           fits=tokens+2048<=context,input_issues=input_issues(rows[r['job_id']])))
    pairs={j:all(x['fits'] and not x['input_issues'] for x in checks if x['job_id']==j) for j in rows}
    metadata=dict(model_path=str(model_path.resolve()),context_limit=context,output_tokens=2048,
        request_sha256=protocol['requests_sha256'],input_sha256=protocol['input_sha256'],
        tokenizer_files={p.name:core.digest(p.read_bytes()) for p in model_path.iterdir()
                         if p.is_file() and (p.name.startswith('tokenizer') or p.name in ('config.json','special_tokens_map.json','chat_template.jinja'))},
        checks=checks,pair_eligible=pairs,eligible_pairs=sum(pairs.values()),
        note='Tokenizer-only preflight; server must use the same tokenizer/template and context limit. No inference performed.')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'preflight.json').exists(): raise ValueError('preflight exists; use a fresh directory')
    io.save(out/'preflight.json',metadata)
    print(dict(total_pairs=len(pairs),eligible_pairs=sum(pairs.values()),report=str(out/'preflight.json')))


def check_service(base_url,model):
    """Fail before reserving jobs; model discovery does not prove generation health."""
    try:
        with urllib.request.urlopen(base_url.rstrip('/')+'/models',timeout=10) as response:
            raw=response.read(100001)
        if len(raw)>100000: raise ValueError('oversize model listing')
        listing=core.strict_json(raw.decode())
        if model not in {item['id'] for item in listing['data']}:
            raise ValueError('requested model not advertised: '+model)
    except Exception as exc:
        raise RuntimeError('Model service unavailable or incompatible at '+base_url+
                           '; start the 7B service before running. No jobs reserved. '+str(exc)) from exc


def run(root,out,check_path,base_url,model):
    protocol,rows,requests=load(root); pre=io.read(check_path)
    if pre['request_sha256']!=protocol['requests_sha256'] or pre['input_sha256']!=protocol['input_sha256']:
        raise ValueError('preflight drift')
    reviews=io.read(root/'review.json'); reference={r['job_id']:r for r in reviews}
    if len(reference)!=len(reviews) or set(reference)!=set(rows): raise ValueError('review membership mismatch')
    for review in reviews:
        if review['reviewed'] is True:
            for point in review['required_points']:
                core.validate(dict(claims=[{k:point[k] for k in ('text','source','evidence_ids')}]),rows[review['job_id']])
    config=dict(model=model,base_url=base_url,preflight_sha256=core.digest(check_path.read_bytes()),
                requests_sha256=protocol['requests_sha256'],review_sha256=core.digest((root/'review.json').read_bytes()),
                runner_sha256=core.digest(Path(__file__).read_bytes().replace(b'\r\n',b'\n')))
    if any(pre['pair_eligible'][r['job_id']] and reference[r['job_id']]['reviewed'] is True
           and reference[r['job_id']]['input_complete'] is True
           and reference[r['job_id']]['reviewer_type'] in ('AI','human')
           and not (out/(r['job_id']+'_'+r['variant']+'.json')).exists() for r in requests):
        check_service(base_url,model)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists() and io.read(out/'config.json')!=config: raise ValueError('run config changed')
    io.save(out/'config.json',config)
    for name in ('inputs.jsonl','requests.jsonl','review.json','protocol.json','selection.json','repairs.json'):
        dest=out/name
        if dest.exists() and dest.read_bytes()!=(root/name).read_bytes(): raise ValueError('saved pack drift')
        dest.write_bytes((root/name).read_bytes())
    io.save(out/'preflight.json',pre)
    for index,r in enumerate(requests):
        # Counterbalance order within tasks; no new random subset selection.
        if index%2: continue
        pair=requests[index:index+2]
        if (index//2)%2: pair=list(reversed(pair))
        for r in pair:
            path=out/(r['job_id']+'_'+r['variant']+'.json')
            if path.exists(): continue
            review=reference[r['job_id']]
            status='reserved'
            if not pre['pair_eligible'][r['job_id']]: status='blocked_input'
            elif not (review['reviewed'] is True and review['input_complete'] is True and review['reviewer_type'] in ('AI','human')):
                status='pending_reference'
            result=dict(job_id=r['job_id'],variant=r['variant'],status=status,semantic_quality='not_evaluated')
            with path.open('x',encoding='utf-8') as f: f.write(core.encoded(result).decode())
            if status!='reserved': continue
            body=dict(model=model,messages=r['messages'],temperature=0,max_tokens=2048)
            result['request']=body;started=time.perf_counter()
            try:
                req=urllib.request.Request(base_url.rstrip('/')+'/chat/completions',data=core.encoded(body),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=300) as response: raw=response.read(1000001)
                result['raw_response_text']=raw.decode('utf-8',errors='replace')
                if len(raw)>1000000: raise ValueError('oversize response')
                reply=core.strict_json(raw.decode());result['raw_response']=reply
                expected=next(x['prompt_tokens'] for x in pre['checks'] if x['job_id']==r['job_id'] and x['variant']==r['variant'])
                actual=reply.get('usage',{}).get('prompt_tokens')
                if actual!=expected: raise ValueError(f'server/tokenizer prompt mismatch: expected {expected}, got {actual}')
                choice=reply['choices'][0]
                if choice['finish_reason']!='stop': raise ValueError('generation not complete')
                result['claims']=core.validate(core.strict_json(choice['message']['content']),rows[r['job_id']])
                result['status']='ready'
            except Exception as exc:
                result.update(status='failed',error=type(exc).__name__+': '+str(exc))
            result['seconds']=time.perf_counter()-started;io.save(path,result)
            print(r['variant'],result['status'],flush=True)
    counts=Counter(io.read(out/(r['job_id']+'_'+r['variant']+'.json'))['status'] for r in requests)
    io.save(out/'summary.json',dict(counts=dict(counts),total_requests=len(requests),semantic_quality='pending AI review',
        caution='Only reviewed, token-eligible pairs called; report selected-stage count and skip coverage, not population accuracy.'))
    print(dict(counts))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['preflight','run'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model-path',type=Path);p.add_argument('--context-limit',type=int,default=32768)
    p.add_argument('--preflight',type=Path);p.add_argument('--base-url',default='http://127.0.0.1:30001/v1')
    p.add_argument('--model',default='qwen-local');a=p.parse_args()
    if a.command=='preflight':
        if a.model_path is None: p.error('--model-path required')
        preflight(a.root,a.model_path,a.context_limit,a.out)
    else:
        if a.preflight is None: p.error('--preflight required')
        run(a.root,a.out,a.preflight,a.base_url,a.model)
