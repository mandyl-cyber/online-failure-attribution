"""Remote local-Qwen runner. One reserved attempt/job, resumable, never executes evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def check_schema(value,schema):
    """Only the type/enum/object/array/string subset used by this fixed contract."""
    kinds=schema.get('type',[]);kinds=[kinds] if isinstance(kinds,str) else kinds
    matches={'object':isinstance(value,dict),'array':isinstance(value,list),
             'string':isinstance(value,str),'integer':type(value)==int,'null':value is None}
    if kinds and not any(matches.get(k,False) for k in kinds):raise ValueError('schema type')
    if 'enum' in schema and value not in schema['enum']:raise ValueError('schema enum')
    if isinstance(value,dict):
        if not set(schema.get('required',[]))<=set(value):raise ValueError('missing schema fields')
        props=schema.get('properties',{})
        if schema.get('additionalProperties') is False and set(value)-set(props):raise ValueError('extra schema fields')
        for k,v in value.items():
            if k in props:check_schema(v,props[k])
    if isinstance(value,list):
        if not schema.get('minItems',0)<=len(value)<=schema.get('maxItems',float('inf')):raise ValueError('schema array length')
        for v in value:check_schema(v,schema.get('items',{}))
    if isinstance(value,str) and not schema.get('minLength',0)<=len(value)<=schema.get('maxLength',float('inf')):
        raise ValueError('schema string length')
    if type(value)==int and value<schema.get('minimum',float('-inf')):raise ValueError('schema minimum')


def validate(response,request):
    check_schema(response,request['response_schema'])
    known={e['record_id']:e for e in request['evidence']};ids=set()
    for c in response['claims']:
        if c['local_id'] in ids:raise ValueError('duplicate claim ID')
        ids.add(c['local_id'])
        for ref in c['evidence_refs']:
            e=known[ref['evidence_id']];a,b=ref['start'],ref['end']
            if type(a)!=int or type(b)!=int or not 0<=a<b<=len(e['text']):raise ValueError('invalid span')
            if e['text'][a:b]!=ref['quote']:raise ValueError('quote mismatch')
        if c['basis']=='tool_observation' and not any(known[x['evidence_id']]['field']=='observation' for x in c['evidence_refs']):
            raise ValueError('missing observation reference')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base-url',default='http://127.0.0.1:30001/v1')
    ap.add_argument('--out',default='out/swe_ledger_sensitivity_v1');args=ap.parse_args()
    root=Path('pilot/swe_ledger_sensitivity_v1');raw=(root/'requests.jsonl').read_bytes()
    rows=[json.loads(x) for x in raw.decode('utf-8').splitlines()]
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    config=dict(input_sha256=hashlib.sha256(raw).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),base_url=args.base_url,model='qwen-local',
                temperature=0,max_tokens=2048,attempts=1)
    cp=out/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Configuration changed; use a new output directory')
    cp.write_text(json.dumps(config,indent=2),encoding='utf-8')
    with urllib.request.urlopen(args.base_url.rstrip('/')+'/models',timeout=10) as response:
        models=json.load(response)
    if not any(m['id']=='qwen-local' for m in models.get('data',[])):raise ValueError('qwen-local unavailable')
    for i,row in enumerate(rows,1):
        dest=out/(row['job_id']+'.json')
        if dest.exists():continue
        result=dict(job_id=row['job_id'],status='reserved',semantic_quality='not_evaluated')
        dest.write_text(json.dumps(result),encoding='utf-8')
        started=time.perf_counter();req=row['request']
        try:
            body=dict(model='qwen-local',temperature=0,max_tokens=2048,
                messages=[dict(role='system',content=req['instruction']),dict(role='user',content=json.dumps(req,ensure_ascii=False))])
            request=urllib.request.Request(args.base_url.rstrip('/')+'/chat/completions',
                data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=240) as response:reply=json.load(response)
            result['raw_response']=reply
            if reply['choices'][0]['finish_reason']!='stop':raise ValueError('Incomplete completion')
            claims=json.loads(reply['choices'][0]['message']['content'])
            validate(claims,req);result.update(status='valid',claims=claims['claims'])
        except Exception as exc:result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        result['seconds']=time.perf_counter()-started
        temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(dest)
        print(f'{i}/{len(rows)} {result["status"]}',flush=True)
    manifest=json.loads((root/'remote_manifest.json').read_text(encoding='utf-8'))
    counts={arm:dict(valid=0,failed=0,reserved=0,missing=0,preflight_failed=0,claims=0) for arm in ('gold','structure','rule')}
    for job in manifest['jobs']:
        c=counts[job['arm']]
        if job['preflight_error']:c['preflight_failed']+=1;continue
        path=out/(job['job_id']+'.json')
        if not path.exists():c['missing']+=1;continue
        row=json.loads(path.read_text(encoding='utf-8'));c[row['status']]+=1
        if row['status']=='valid':c['claims']+=len(row['claims'])
    (out/'summary.json').write_text(json.dumps(dict(counts=counts,semantic_quality='not_evaluated',
        dependency_quality='not_evaluated',attribution='not_evaluated'),indent=2),encoding='utf-8')
    print('Results:',out,'; valid means schema/provenance only, not semantic correctness.')


if __name__=='__main__':main()
