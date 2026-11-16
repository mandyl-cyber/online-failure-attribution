"""Standalone causal, bounded-prefix Qwen purpose client (protocol v2)."""
import argparse,hashlib,json,os,time,urllib.request,urllib.error,zipfile
from pathlib import Path

LABELS=['ACQUIRE','ENABLE','CONSTRUCT','OPERATE','CHECK','DELIVER','UNKNOWN']
FIELDS={'purpose','evidence_id','rationale'}
LIMITS={'task':5000,'current_action':2200,'current_observation':3000,
        'current_public_intent':1000,'previous_action':600,
        'previous_observation':350,'previous_public_intent':250}


def canonical(obj): return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def sha(value): return hashlib.sha256(value.encode('utf-8')).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,obj):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8'); temp.replace(path)


def clipped(value,limit):
    value=str(value or '')
    if len(value)<=limit: return value,False
    head=int(limit*.7); tail=limit-head
    return value[:head]+f'\n[... {len(value)-limit} CHARACTERS OMITTED ...]\n'+value[-tail:],True


def packet_for(trajectory,j):
    steps=trajectory['steps']; current=steps[j]
    packet=dict(trajectory_id=trajectory['trajectory_id'],current_step_id=current['idx'],
                policy='Task, first two and last six completed steps; text is head/tail clipped with explicit omission counts.',
                evidence=[])
    def add(id,value,limit,step_id=None):
        if not str(value or '').strip(): return
        excerpt,truncated=clipped(value,limit)
        packet['evidence'].append(dict(id=id,step_id=step_id,text=excerpt,
                                       truncated=truncated,original_characters=len(str(value))))
    add('TASK',trajectory['task'],LIMITS['task'],0)
    previous=sorted(set(range(min(2,j)))|set(range(max(0,j-6),j)))
    for i in previous:
        row=steps[i]; prefix=f"P{row['idx']}"
        add(prefix+'_ACTION',row['action'],LIMITS['previous_action'],row['idx'])
        add(prefix+'_OBSERVATION',row['observation'],LIMITS['previous_observation'],row['idx'])
        add(prefix+'_INTENT',row['public_intent'],LIMITS['previous_public_intent'],row['idx'])
        packet['evidence'].append(dict(id=prefix+'_META',step_id=row['idx'],tool=row['tool'],exit_code=row['exit_code'],
                                       source_input_truncated=row['input_truncated']))
    add('C_ACTION',current['action'],LIMITS['current_action'],current['idx'])
    add('C_OBSERVATION',current['observation'],LIMITS['current_observation'],current['idx'])
    add('C_INTENT',current['public_intent'],LIMITS['current_public_intent'],current['idx'])
    packet['evidence'].append(dict(id='C_META',step_id=current['idx'],tool=current['tool'],exit_code=current['exit_code'],
                                   source_input_truncated=current['input_truncated']))
    return packet


def jobs(root):
    for line in (root/'trajectories.jsonl').read_text(encoding='utf-8').splitlines():
        trajectory=json.loads(line)
        for j,row in enumerate(trajectory['steps']):
            packet=packet_for(trajectory,j)
            yield dict(request_id=f"{trajectory['trajectory_id']}_s{row['idx']:05d}",input_sha256=sha(canonical(packet)),packet=packet)


def validate(answer,packet):
    if not isinstance(answer,dict) or set(answer)!=FIELDS: raise ValueError('Wrong output fields')
    if answer['purpose'] not in LABELS: raise ValueError('Unknown purpose')
    if answer['evidence_id'] not in {e['id'] for e in packet['evidence'] if e['id'].startswith('C_')}:
        raise ValueError('Evidence ID must identify an available current-step field')
    if not isinstance(answer['rationale'],str) or not 1<=len(answer['rationale'])<=240:
        raise ValueError('Rationale must be 1-240 characters')
    return answer


def control_answer(packet):
    """Explicitly label empty task-tracking updates as control steps."""
    fields={e['id']:e for e in packet['evidence'] if e['id'].startswith('C_')}
    meta=fields['C_META']
    observation=fields.get('C_OBSERVATION',{}).get('text','')
    if meta['tool']=='task_tracking' and 'C_ACTION' not in fields and not fields.get('C_INTENT') \
            and observation.startswith('Task list has been updated with '):
        return dict(purpose='UNKNOWN',evidence_id='C_OBSERVATION',
                    rationale='Only a task-tracking list update is visible; no task work purpose is supported.')
    return None


def verify_package(root):
    manifest=read(root/'package_manifest.json')
    for name,digest in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest: raise ValueError('Package hash mismatch: '+name)
    return manifest


def audit(root,out):
    results=[]; good=0; missing=0; invalid=0
    run=read(out/'run_manifest.json') if (out/'run_manifest.json').exists() else None
    for job in jobs(root):
        path=out/'responses'/f"{job['request_id']}.json"
        if not path.exists(): missing+=1; results.append(dict(request_id=job['request_id'],status='missing')); continue
        try:
            record=read(path)
            if run is None or record['identity']!=run['identity']: raise ValueError('Run identity mismatch')
            if record['input_sha256']!=job['input_sha256']: raise ValueError('Input hash mismatch')
            if record['request_id']!=job['request_id']: raise ValueError('Request id mismatch')
            validate(record['answer'],job['packet']); good+=1
            results.append(dict(request_id=job['request_id'],status='valid'))
        except (ValueError,KeyError,TypeError) as e:
            invalid+=1; results.append(dict(request_id=job['request_id'],status='invalid',error=str(e)))
    summary=dict(valid=good,missing=missing,invalid=invalid,complete=not(missing or invalid),semantic_accuracy_evaluated=False,results=results)
    save(out/'audit.json',summary)
    print(canonical({k:v for k,v in summary.items() if k!='results'}))
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['inspect','preflight','run','validate','bundle'])
    ap.add_argument('--package',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--out',type=Path,default=Path('qwen_results'))
    ap.add_argument('--base-url',default='http://127.0.0.1:8000/v1')
    ap.add_argument('--model'); ap.add_argument('--tokenizer',help='Local tokenizer directory matching served weights/template')
    ap.add_argument('--context-tokens',type=int); ap.add_argument('--max-output-tokens',type=int,default=256)
    ap.add_argument('--margin-tokens',type=int,default=128)
    ap.add_argument('--chat-template-kwargs',default='{}')
    ap.add_argument('--limit',type=int); ap.add_argument('--timeout',type=int,default=180)
    ap.add_argument('--request-ids-file',type=Path,help='One request ID per line; run a fixed representative subset')
    ap.add_argument('--attempts',type=int,default=3)
    args=ap.parse_args(); root=args.package; manifest=verify_package(root)
    alljobs=list(jobs(root)); prompt=(root/'SYSTEM_PROMPT.txt').read_text(encoding='utf-8')
    if args.command=='inspect':
        print(canonical(dict(trajectories=manifest['trajectories'],requests=len(alljobs),files_verified=True,contains_gold=False))); return
    args.out.mkdir(parents=True,exist_ok=True)
    if args.command in ('validate','bundle'):
        result=audit(root,args.out)
        if args.command=='bundle':
            target=args.out.parent/(args.out.name+'_return.zip')
            with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
                for p in sorted(args.out.rglob('*')):
                    if p.is_file() and p.suffix!='.tmp': z.write(p,p.relative_to(args.out))
            print(str(target))
        return
    if not args.tokenizer or not args.context_tokens: ap.error('--tokenizer and actual --context-tokens are required')
    if args.command=='run' and not args.model: ap.error('--model is the actual served model ID')
    if args.context_tokens<=args.max_output_tokens+args.margin_tokens or args.max_output_tokens<1 or args.margin_tokens<0 or args.attempts<1: ap.error('Invalid budgets/attempts')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
    template_kwargs=json.loads(args.chat_template_kwargs)
    if not isinstance(template_kwargs,dict): ap.error('template kwargs must be an object')
    selected=alljobs[:args.limit] if args.limit else alljobs
    if args.request_ids_file:
        wanted=[x.strip() for x in args.request_ids_file.read_text().splitlines() if x.strip()]
        if len(wanted)!=len(set(wanted)): ap.error('Duplicate request IDs')
        by_id={x['request_id']:x for x in alljobs}
        if set(wanted)-set(by_id): ap.error('Unknown request IDs')
        selected=[by_id[x] for x in wanted]
    def messages(job): return [dict(role='system',content=prompt),dict(role='user',content=canonical(job['packet']))]
    counts=[]
    for job in selected:
        ids=tokenizer.apply_chat_template(messages(job),tokenize=True,add_generation_prompt=True,**template_kwargs)
        token_ids=ids['input_ids'] if hasattr(ids,'input_ids') else ids
        counts.append(dict(request_id=job['request_id'],input_tokens=len(token_ids),fits=len(token_ids)+args.max_output_tokens+args.margin_tokens<=args.context_tokens))
    preflight=dict(total=len(counts),fits=sum(x['fits'] for x in counts),max_input_tokens=max((x['input_tokens'] for x in counts),default=0),context_tokens=args.context_tokens,output_tokens=args.max_output_tokens,margin=args.margin_tokens,requests=counts)
    save(args.out/'preflight.json',preflight)
    print(canonical({k:v for k,v in preflight.items() if k!='requests'}),flush=True)
    if args.command=='preflight': return
    config=dict(package_sha256=sha(canonical(manifest)),prompt_sha256=sha(prompt),model=args.model,base_url=args.base_url,
        tokenizer=str(args.tokenizer),tokenizer_vocab_sha256=sha(canonical(tokenizer.get_vocab())),chat_template_sha256=sha(str(tokenizer.chat_template)),
        chat_template_kwargs=template_kwargs,context_tokens=args.context_tokens,max_output_tokens=args.max_output_tokens,margin_tokens=args.margin_tokens,temperature=0)
    identity=sha(canonical(config)); runpath=args.out/'run_manifest.json'
    if runpath.exists() and read(runpath)['identity']!=identity: raise ValueError('Run configuration changed; choose a new --out')
    save(runpath,dict(identity=identity,config=config,note='Machine candidate labels; not gold. Server model revision/quantization must be supplied in ENVIRONMENT.txt.'))
    (args.out/'responses').mkdir(exist_ok=True); (args.out/'errors').mkdir(exist_ok=True)
    for job,count in zip(selected,counts):
        destination=args.out/'responses'/f"{job['request_id']}.json"
        if destination.exists():
            old=read(destination)
            if old['identity']!=identity or old['input_sha256']!=job['input_sha256']: raise ValueError('Resume identity mismatch')
            validate(old['answer'],job['packet']); continue
        error_path=args.out/'errors'/f"{job['request_id']}.json"
        if not count['fits']:
            save(error_path,dict(status='context_overflow',**count)); continue
        rule_answer=control_answer(job['packet'])
        if rule_answer is not None:
            validate(rule_answer,job['packet'])
            save(destination,dict(request_id=job['request_id'],input_sha256=job['input_sha256'],identity=identity,
                                  answer=rule_answer,label_source='deterministic_control_rule',usage=None,server_model=None,
                                  seconds=0,attempt=0))
            print(job['request_id'],'valid deterministic_control_rule',flush=True)
            continue
        body=dict(model=args.model,messages=messages(job),temperature=0,max_tokens=args.max_output_tokens,stream=False)
        if template_kwargs: body['chat_template_kwargs']=template_kwargs
        headers={'Content-Type':'application/json'}
        if os.environ.get('QWEN_API_KEY'): headers['Authorization']='Bearer '+os.environ['QWEN_API_KEY']
        failures=[]
        for attempt in range(args.attempts):
            started=time.perf_counter(); content=None
            try:
                request=urllib.request.Request(args.base_url.rstrip('/')+'/chat/completions',data=json.dumps(body).encode(),headers=headers)
                with urllib.request.urlopen(request,timeout=args.timeout) as response: raw=json.load(response)
                choice=raw['choices'][0]; content=choice['message']['content']
                if choice.get('finish_reason')!='stop': raise ValueError('Output did not finish normally: '+str(choice.get('finish_reason')))
                parsed=json.loads(content)
                normalized=False
                current_action=next((x.get('text','') for x in job['packet']['evidence'] if x['id']=='C_ACTION'),'')
                if isinstance(parsed,dict) and parsed.get('purpose')=='SEARCH' and \
                        any(word in current_action.lower() for word in ('grep ','rg ','search ','find ')):
                    parsed=dict(parsed,purpose='ACQUIRE')
                    normalized=True
                answer=validate(parsed,job['packet'])
                save(destination,dict(request_id=job['request_id'],input_sha256=job['input_sha256'],identity=identity,answer=answer,
                    label_source='qwen_search_alias' if normalized else 'qwen',raw_answer=content if normalized else None,
                    review_priority=normalized,usage=raw.get('usage'),server_model=raw.get('model'),seconds=time.perf_counter()-started,attempt=attempt+1))
                print(job['request_id'],'valid',flush=True); break
            except (ValueError,KeyError,TypeError,urllib.error.URLError,TimeoutError) as e:
                failures.append(dict(attempt=attempt+1,error=str(e),raw_content=content))
                save(error_path,dict(request_id=job['request_id'],status='failed',failures=failures))
                print(job['request_id'],'failed',str(e),flush=True)
                if attempt+1<args.attempts:
                    if isinstance(content,str) and content:
                        available=[x['id'] for x in job['packet']['evidence'] if x['id'].startswith('C_')]
                        body['messages']=messages(job)+[dict(role='assistant',content=content),
                            dict(role='user',content='Your previous output failed validation: '+str(e)[:200]+
                                 '. Correct it now. Return ONLY one complete JSON object with exactly purpose, evidence_id, rationale. '+
                                 'purpose must be one of '+', '.join(LABELS)+'. evidence_id must be one of '+', '.join(available)+
                                 '. Use the CURRENT step, not a previous step.')]
                    time.sleep(1)
    audit(root,args.out)


if __name__=='__main__': main()
