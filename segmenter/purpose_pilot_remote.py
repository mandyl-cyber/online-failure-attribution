"""Standalone Python 3.10+ remote Qwen pilot client and strict response validator."""
import argparse,hashlib,json,os,time,urllib.request,urllib.error,zipfile
from pathlib import Path

LABELS=['ACQUIRE','ENABLE','CONSTRUCT','OPERATE','CHECK','DELIVER','UNKNOWN']
FIELDS={'purpose','confidence','abstain_reason','alternatives','evidence','rationale'}


def canonical(obj): return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def sha(value): return hashlib.sha256(value.encode('utf-8')).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,obj):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8'); temp.replace(path)


def jobs(root):
    for line in (root/'trajectories.jsonl').read_text(encoding='utf-8').splitlines():
        trajectory=json.loads(line)
        for j,row in enumerate(trajectory['steps']):
            packet=dict(trajectory_id=trajectory['trajectory_id'],task=trajectory['task'],current_step_id=row['idx'],steps=trajectory['steps'][:j+1])
            yield dict(request_id=f"{trajectory['trajectory_id']}_s{row['idx']:05d}",input_sha256=sha(canonical(packet)),packet=packet)


def validate(answer,packet):
    if not isinstance(answer,dict) or set(answer)!=FIELDS: raise ValueError('Wrong output fields')
    if answer['purpose'] not in LABELS: raise ValueError('Unknown purpose')
    if answer['confidence'] not in ('high','medium','low'): raise ValueError('Invalid confidence')
    reasons=('NONE','CONTROL_STEP','MIXED_PURPOSE','INSUFFICIENT_CONTEXT','CONFLICTING_EVIDENCE')
    if answer['abstain_reason'] not in reasons: raise ValueError('Invalid abstain reason')
    unknown=answer['purpose']=='UNKNOWN'
    if unknown != (answer['abstain_reason']!='NONE'): raise ValueError('Abstention mismatch')
    if unknown and answer['confidence']=='high': raise ValueError('Unknown cannot claim high confidence')
    alt=answer['alternatives']
    if not isinstance(alt,list) or len(alt)>2 or any(not isinstance(x,str) or x not in LABELS[:-1] for x in alt) or len(set(alt))!=len(alt) or answer['purpose'] in alt: raise ValueError('Invalid alternatives')
    if not isinstance(answer['rationale'],str) or not 1<=len(answer['rationale'])<=600: raise ValueError('Invalid concise rationale')
    evidence=answer['evidence']
    if not isinstance(evidence,list) or not 1<=len(evidence)<=3: raise ValueError('Need 1-3 evidence references')
    steps={s['idx']:s for s in packet['steps']}; current=False
    for ref in evidence:
        if not isinstance(ref,dict) or set(ref)!={'step_id','field','quote'}: raise ValueError('Invalid evidence fields')
        i=ref['step_id']; field=ref['field']; quote=ref['quote']
        if isinstance(i,bool) or not isinstance(i,int): raise ValueError('Invalid step id')
        if i==0 and field=='task': text=packet['task']
        elif i in steps and field in ('action','observation','public_intent'): text=steps[i][field]
        else: raise ValueError('Future/foreign evidence')
        if not isinstance(quote,str) or not 1<=len(quote)<=300 or quote not in text: raise ValueError('Quote must exactly occur in referenced evidence')
        current |= i==packet['current_step_id']
    if not current: raise ValueError('At least one quote must refer to the current step')
    return answer


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
    ap.add_argument('--context-tokens',type=int); ap.add_argument('--max-output-tokens',type=int,default=1200)
    ap.add_argument('--margin-tokens',type=int,default=128)
    ap.add_argument('--chat-template-kwargs',default='{}')
    ap.add_argument('--limit',type=int); ap.add_argument('--timeout',type=int,default=180)
    ap.add_argument('--attempts',type=int,default=2)
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
    def messages(job): return [dict(role='system',content=prompt),dict(role='user',content=canonical(job['packet']))]
    counts=[]
    for job in selected:
        ids=tokenizer.apply_chat_template(messages(job),tokenize=True,add_generation_prompt=True,**template_kwargs)
        counts.append(dict(request_id=job['request_id'],input_tokens=len(ids),fits=len(ids)+args.max_output_tokens+args.margin_tokens<=args.context_tokens))
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
        body=dict(model=args.model,messages=messages(job),temperature=0,max_tokens=args.max_output_tokens,stream=False)
        if template_kwargs: body['chat_template_kwargs']=template_kwargs
        headers={'Content-Type':'application/json'}
        if os.environ.get('QWEN_API_KEY'): headers['Authorization']='Bearer '+os.environ['QWEN_API_KEY']
        for attempt in range(args.attempts):
            started=time.perf_counter(); content=None
            try:
                request=urllib.request.Request(args.base_url.rstrip('/')+'/chat/completions',data=json.dumps(body).encode(),headers=headers)
                with urllib.request.urlopen(request,timeout=args.timeout) as response: raw=json.load(response)
                choice=raw['choices'][0]; content=choice['message']['content']
                if choice.get('finish_reason')!='stop': raise ValueError('Output did not finish normally: '+str(choice.get('finish_reason')))
                answer=validate(json.loads(content),job['packet'])
                save(destination,dict(request_id=job['request_id'],input_sha256=job['input_sha256'],identity=identity,answer=answer,
                    usage=raw.get('usage'),server_model=raw.get('model'),seconds=time.perf_counter()-started,attempt=attempt+1))
                print(job['request_id'],'valid',flush=True); break
            except (ValueError,KeyError,TypeError,urllib.error.URLError,TimeoutError) as e:
                save(error_path,dict(request_id=job['request_id'],status='failed',error=str(e),attempt=attempt+1,raw_content=content))
                print(job['request_id'],'failed',str(e),flush=True)
                if attempt+1<args.attempts: time.sleep(1)
    audit(root,args.out)


if __name__=='__main__': main()
