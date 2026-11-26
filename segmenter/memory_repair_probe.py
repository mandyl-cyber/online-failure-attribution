"""Frozen output-format, provenance-wording and new-evidence diagnostics."""
import argparse
from copy import deepcopy
from pathlib import Path
import time
from segmenter import memory_mechanism_probe as m
from segmenter import stage_claim_ablation as a

ROOT = Path('pilot/memory_repair_probe_v1')
LIMIT = '\nCite at most TWO original records. Keep limitations within 20 words. Do not list all records.'


def replace_ids(value, mapping):
    if isinstance(value, str):
        return mapping.get(value,value)
    if isinstance(value,list):
        return [replace_ids(v,mapping) for v in value]
    if isinstance(value,dict):
        return {k:replace_ids(v,mapping) for k,v in value.items()}
    return value


def prompt(job):
    units=deepcopy(job['units'])
    ids=sorted(a.visible_ids(units))
    aliases={v:f'E{i+1}' for i,v in enumerate(ids)} if job['short_ids'] else {v:v for v in ids}
    units=replace_ids(units,aliases)
    if job['task']=='support':
        messages=m.messages_for(job['case'],units,False)
    else:
        messages=a.messages_for_query(job['query'],units)
    if job['limited']:
        messages[0]['content']+=LIMIT
    return messages,{v:k for k,v in aliases.items()}


def prepare():
    require=a.require
    require(not any((ROOT/n).exists() for n in ('jobs.json','labels.ai.json','protocol.json')),
            'Output exists; do not overwrite frozen data')
    cases,packets,_=m.load_inputs(Path('pilot/memory_mechanism_probe_v1'))
    labels={g['case_id']:g for g in a.read(Path('pilot/memory_mechanism_probe_v1/labels.ai.json'))}
    jobs,gold=[],[]
    def add(group,arm,case_id,units,label,**kwargs):
        key=a.digest(a.encoded([group,arm,case_id]))[:24]
        jobs.append(dict(job_id=key,group=group,arm=arm,task='support',units=deepcopy(units),
                         short_ids=False,limited=False,**kwargs))
        gold.append(dict(label,job_id=key,case_id=case_id))
        return jobs[-1]
    for c in cases:
        units=m.units_for(packets[c['case_id']],'learned_claim')
        for arm in ('original','limit_only','short_and_limit'):
            j=add('format',arm,c['case_id'],units,labels[c['case_id']],case=c)
            j.update(limited=arm!='original',short_ids=arm=='short_and_limit')
    c=next(c for c in cases if c['case_id']=='config_overrides:prefix15:commit_support')
    for arm in ('original_wording','agent_wording','explicit_source_wording'):
        units=m.units_for(packets[c['case_id']],'learned_claim')
        matches=[cl for u in units for cl in u.get('claims',[]) if cl.get('basis')=='agent_assertion'
                 and cl.get('source_steps')==[15]]
        require(len(matches)==1,'Ambiguous source assertion')
        cl=matches[0]
        require(cl['text'].startswith('The commit reports '),'Source text changed')
        if arm=='agent_wording':
            cl['text']=cl['text'].replace('The commit reports ','The agent reports ',1)
        if arm=='explicit_source_wording':
            cl['text']='Agent statement, not a tool confirmation: '+cl['text'].replace('The commit reports ','The agent reports ',1)
        j=add('provenance',arm,c['case_id'],units,labels[c['case_id']],case=c)
        j.update(short_ids=True,limited=True)
    data=a.read(Path('pilot/stage_claim_ablation_v1/inputs.json'))
    oldgold={g['query_id']:g for g in a.read(Path('pilot/stage_claim_ablation_v1/labels.json'))}
    outputs={p.stem:a.read(p) for p in Path('stage_claim_ablation_v1/claims').glob('*.json')}
    require(len(outputs)==27,'Complete original returned claims required for preparation')
    for q in data['queries']:
        units=a.representation(data,q,'stage_claim',outputs,'representation',.05)
        s=next(s for s in data['sessions'] if s['session']==q['session'])
        records=[r for r in s['records'] if r['source_step_id']==q['target_step']]
        target=next(r for r in records if r['field']==q['target_field'])
        raw={r['evidence_id'] for u in units for r in u.get('records',[])}
        if target['record_id'] not in a.visible_ids(units) or target['record_id'] in raw or q['target_step']>q['prefix_step']-2:
            continue
        require(all(r['source_step_id']<=q['prefix_step'] and r['availability_seq']<=q['prefix_step'] for r in records),'Future source')
        for arm in ('claims_only_for_target','target_raw_restored'):
            selected=deepcopy(units)
            if arm=='target_raw_restored':
                selected.append(a.raw_unit([r for r in records if r['record_id'] not in raw]))
            j=add('retrieval',arm,q['query_id'],selected,oldgold[q['query_id']],query=q)
            j.update(task='category',short_ids=True,limited=True)
    a.save(ROOT/'jobs.json',jobs)
    a.save(ROOT/'labels.ai.json',gold)
    a.save(ROOT/'protocol.json',dict(version='memory-repair-v1',jobs=len(jobs),
        files={n:a.digest((ROOT/n).read_bytes()) for n in ('jobs.json','labels.ai.json')},
        scope='Posthoc project-seen diagnostic; nonblind AI labels, not online alarm efficacy',
        format='Original vs citation limit vs short IDs+limit; same inputs, 256 output tokens',
        provenance='One commit-support case, wording only; raw source and basis unchanged',
        retrieval='32 historical targets naturally absent from recent raw buffer; target selection oracle given by query, not deployable retrieval; paired action included; no future source',
        short_ids='Reversible mapping to original IDs, saved with each answer',model_calls=0))
    print(dict(jobs=len(jobs),model_calls=0))


def load():
    p=a.read(ROOT/'protocol.json')
    for n,h in p['files'].items():
        a.require(a.digest((ROOT/n).read_bytes())==h,'Frozen data changed')
    return a.read(ROOT/'jobs.json'),a.read(ROOT/'labels.ai.json'),p


def parse(text,job,mapping):
    visible=list(mapping)
    value=m.parse(text,visible,False) if job['task']=='support' else a.parse_answer(text,visible)
    if value['valid']:
        if job['limited'] and len(value['evidence_ids'])>2:
            return dict(valid=False,error='citation_limit_exceeded')
        if job['limited'] and job['task']=='support' and len(value['limitations'].split())>20:
            return dict(valid=False,error='limitations_limit_exceeded')
        value['evidence_ids']=[mapping[i] for i in value['evidence_ids']]
    return value


def summarize(rows,gold):
    labels={g['job_id']:g for g in gold}
    groups={}
    for r in rows:
        v,g=r['answer'],labels[r['job_id']]
        key=r['group']+'/'+r['arm']
        d=groups.setdefault(key,dict(evaluated=0,valid=0,correct=0,status_correct=0,support_correct=0,unsupported_acceptances=0,input_tokens=0,seconds=0.,errors={}))
        d['evaluated']+=1
        d['valid']+=v['valid']
        if r['group']=='retrieval':
            d['correct']+=v['valid'] and v.get('category')==g['category']
        else:
            s=v['valid'] and v.get('latest_run_status')==g['latest_run_status']
            c=v['valid'] and v.get('assertion_support')==g['assertion_support']
            d['status_correct']+=s; d['support_correct']+=c; d['correct']+=s and c
            d['unsupported_acceptances']+=v['valid'] and g['assertion_support']=='not_established' and v.get('assertion_support')=='supported'
        if not v['valid']:
            e=v.get('error','invalid'); d['errors'][e]=d['errors'].get(e,0)+1
        d['input_tokens']+=r['input_tokens']; d['seconds']+=r['seconds']
    return dict(groups=groups,scope='Separate diagnostic groups; never pool accuracies; citation membership is not semantic entailment')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','check','preflight','run'))
    p.add_argument('--model-path',type=Path)
    p.add_argument('--training-run',type=Path,default=Path('out/claim_sft_7b_reviewed_v2'))
    p.add_argument('--out',type=Path,default=Path('out/memory_repair_probe_v1'))
    p.add_argument('--evidence',action='store_true',help='Run short-ID source and autonomous retrieval diagnostics')
    args=p.parse_args()
    if args.command=='prepare':
        prepare(); return
    jobs,gold,protocol=load()
    active_root=ROOT
    prompt_fn=prompt
    summary_fn=summarize
    if args.evidence:
        from segmenter import memory_evidence_probe as evidence
        jobs,gold,protocol=evidence.load()
        active_root=evidence.ROOT
        prompt_fn=evidence.prompt
        summary_fn=evidence.summarize
        if args.out==Path('out/memory_repair_probe_v1'):
            args.out=Path('out/memory_evidence_probe_v1')
    if args.command=='check':
        print(dict(jobs=len(jobs),model_calls=0)); return
    a.require(args.model_path is not None,'--model-path required')
    a.require(Path(a.read(args.training_run/'config.json')['model_path']).resolve()==args.model_path.resolve(),'Use original base')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(args.model_path),local_files_only=True)
    def tokenize(messages):
        return tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
    adapter=args.training_run/'adapter'
    files=[args.model_path/'config.json',adapter/'adapter_config.json',*sorted(adapter.glob('adapter_model.*'))]
    a.require(len(files)>=3,'Adapter missing')
    config=dict(protocol=a.digest((active_root/'protocol.json').read_bytes()),code=a.digest(Path(__file__).read_bytes()),
        dependencies={n:a.digest((Path(__file__).parent/n).read_bytes()) for n in ('memory_mechanism_probe.py','stage_claim_ablation.py','stage_claim_summary.py')},
        model_path=str(args.model_path.resolve()),files={f.name:a.digest(f.read_bytes()) for f in files},max_input=7936,max_output=256,reader='same NF4 base, adapter disabled')
    if args.evidence: config['evidence_code']=a.digest(Path(evidence.__file__).read_bytes())
    if (args.out/'config.json').exists():
        a.require(a.read(args.out/'config.json')==config,'Config changed: use new --out')
    a.save(args.out/'config.json',config)
    checks=[dict(job_id=j['job_id'],tokens=len(tokenize(prompt_fn(j)[0]))) for j in jobs]
    if args.evidence:
        checks.extend(dict(job_id=j['job_id'],phase='lookup',tokens=len(tokenize(evidence.lookup_prompt(j)[0])))
                      for j in jobs if j['arm']=='model_lookup')
    a.save(args.out/'preflight.json',dict(checks=checks,model_calls=0))
    a.require(all(c['tokens']<=7936 for c in checks),'Over budget')
    if args.command=='preflight':
        print(dict(jobs=len(jobs),max_tokens=max(c['tokens'] for c in checks))); return
    import torch
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig
    from peft import PeftModel,prepare_model_for_kbit_training
    a.require(torch.cuda.is_available(),'CUDA required')
    started=time.perf_counter()
    base=AutoModelForCausalLM.from_pretrained(str(args.model_path),local_files_only=True,dtype=torch.bfloat16,device_map={'':0},quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=False)
    model=PeftModel.from_pretrained(base,str(adapter),local_files_only=True).eval()
    model.config.use_cache=True
    a.save(args.out/'initializations'/(str(time.time_ns())+'.json'),dict(seconds=time.perf_counter()-started))
    rows=[]
    def generate_messages(messages,remaining):
        ids=tokenize(messages)
        a.require(len(ids)<=remaining,'cumulative_input_budget_exceeded')
        tensor=torch.tensor([ids],device=model.device)
        torch.cuda.synchronize(); started=time.perf_counter()
        with model.disable_adapter(),torch.inference_mode():
            output=model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),do_sample=False,max_new_tokens=256,pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        torch.cuda.synchronize(); seconds=time.perf_counter()-started
        tokens=output[0,len(ids):].tolist()
        return tokenizer.decode(tokens,skip_special_tokens=True),len(tokens),len(ids),seconds
    # Interleave arms within each case; all group contrasts are frozen in jobs.json.
    for j in jobs:
        messages,mapping=prompt_fn(j); ids=tokenize(messages)
        key=a.digest(a.encoded([j,config,messages]))
        path=args.out/'answers'/(j['job_id']+'.json')
        if path.exists():
            r=a.read(path); a.require(r['input_sha256']==key,'Cache changed')
        elif args.evidence:
            r=evidence.evaluate(j,generate_messages)
            r['input_sha256']=key
            a.save(path,r)
        else:
            tensor=torch.tensor([ids],device=model.device)
            torch.cuda.synchronize(); started=time.perf_counter()
            with model.disable_adapter(),torch.inference_mode():
                output=model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),do_sample=False,max_new_tokens=256,pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
            torch.cuda.synchronize(); seconds=time.perf_counter()-started
            tokens=output[0,len(ids):].tolist(); text=tokenizer.decode(tokens,skip_special_tokens=True)
            answer=parse(text,j,mapping)
            if len(tokens)>=256: answer=dict(valid=False,error='output_limit_reached')
            r=dict(job_id=j['job_id'],group=j['group'],arm=j['arm'],messages=messages,id_mapping=mapping,text=text,answer=answer,input_tokens=len(ids),output_tokens=len(tokens),seconds=seconds,input_sha256=key)
            a.save(path,r)
        rows.append(r); a.save(args.out/'summary.json',summary_fn(rows,gold))
        print(dict(completed=len(rows),expected=len(jobs)),flush=True)


if __name__=='__main__': main()
