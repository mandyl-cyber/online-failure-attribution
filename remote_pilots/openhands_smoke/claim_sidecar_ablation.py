"""Matched original/v2-sidecar inference with the same frozen7B adapter."""
import argparse
import hashlib
import json
import time
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['preflight','run'])
    p.add_argument('--inputs',type=Path,default=Path('pilot/claim_sidecar_input_ablation_v1'))
    p.add_argument('--training-run',type=Path,default=Path('out/claim_sft_7b_reviewed_v2'))
    p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('out/claim_sidecar_input_ablation_v1'))
    a=p.parse_args()
    protocol=json.loads((a.inputs/'protocol.json').read_text(encoding='utf-8'))
    records={}
    for mode in ('original','sidecar_v2'):
        path=a.inputs/mode/'dev.jsonl'
        if sha(path)!=protocol['file_hashes'][mode]:raise ValueError('Frozen inputs changed')
        records[mode]=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
    original=records['original'];variant=records['sidecar_v2']
    assert len(original)==len(variant)==protocol['examples']==38
    for x,y in zip(original,variant):
        assert x['job_id']==y['job_id'] and x['completion']==y['completion']
        assert x['group']==y['group'] and x['prompt'][0]==y['prompt'][0]
        q0=json.loads(x['prompt'][1]['content']);q1=json.loads(y['prompt'][1]['content'])
        assert {k:v for k,v in q0.items() if k not in ('observed_facts','observed_facts_version')}=={k:v for k,v in q1.items() if k not in ('observed_facts','observed_facts_version')}
    trained=json.loads((a.training_run/'config.json').read_text(encoding='utf-8'))
    if Path(trained['model_path']).resolve()!=a.model_path.resolve():raise ValueError('Use the original7B training model path')
    assert trained['data_hashes']['dev']==protocol['file_hashes']['original']
    adapter=a.training_run/'adapter'
    weights=sorted(adapter.glob('adapter_model.*'))
    if not weights or not (adapter/'adapter_config.json').exists():raise ValueError('Completed training adapter required')
    fingerprints={x.name:sha(x) for x in [adapter/'adapter_config.json',*weights]}
    from transformers import AutoTokenizer,AutoConfig,set_seed
    from remote_pilots.openhands_smoke.claim_sft_train import chat_ids
    if getattr(AutoConfig.from_pretrained(str(a.model_path),local_files_only=True),'quantization_config',None):raise ValueError('Original7B weights required, not AWQ')
    tokenizer=AutoTokenizer.from_pretrained(str(a.model_path),local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    encoded={mode:[chat_ids(tokenizer,r['prompt'],True) for r in rows] for mode,rows in records.items()}
    limit=trained['max_length'];new_tokens=trained['max_new_tokens']
    preflight=[dict(job_id=r['job_id'],prompt_tokens={m:len(encoded[m][i]) for m in records},
        eligible=all(len(encoded[m][i])+new_tokens<=limit for m in records)) for i,r in enumerate(original)]
    versions={}
    for name in ('torch','transformers','peft','bitsandbytes'):
        try:versions[name]=version(name)
        except PackageNotFoundError:versions[name]=None
    config=dict(input_hashes=protocol['file_hashes'],adapter_hashes=fingerprints,model_path=str(a.model_path.resolve()),versions=versions,
        max_length=limit,max_new_tokens=new_tokens,seed=trained['seed'],do_sample=False,
        condition_order='alternate per pair',quantization='same NF4 double quantization as training')
    if (a.out/'config.json').exists() and json.loads((a.out/'config.json').read_text(encoding='utf-8'))!=config:raise ValueError('Run configuration drift; choose a new output directory')
    save(a.out/'config.json',config);save(a.out/'preflight.json',preflight)
    eligible=sum(x['eligible'] for x in preflight)
    print(dict(pairs=38,eligible=eligible,excluded_both_conditions=38-eligible),flush=True)
    if a.command=='preflight':return
    if not eligible:raise ValueError('No matched pairs fit; no truncation allowed')
    import torch
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig
    from peft import PeftModel,prepare_model_for_kbit_training
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if torch.cuda.mem_get_info()[0]<20*1024**3:raise RuntimeError('Reserve at least20GiB free GPU memory; no processes are stopped automatically')
    set_seed(trained['seed'])
    model=AutoModelForCausalLM.from_pretrained(str(a.model_path),local_files_only=True,dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),device_map={'':0})
    # Reproduce the frozen nonquantized-parameter casting used by the training
    # runner, without enabling gradients/checkpointing for this inference task.
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
    model=PeftModel.from_pretrained(model,str(adapter),local_files_only=True).eval()
    model.config.use_cache=True
    outputs=json.loads((a.out/'paired_outputs.json').read_text(encoding='utf-8')) if (a.out/'paired_outputs.json').exists() else []
    by_id={r['job_id']:r for r in outputs}
    for i,row in enumerate(original):
        if not preflight[i]['eligible']:continue
        pair=by_id.get(row['job_id'],dict(job_id=row['job_id'],group=row['group'],parent_stage_key=row['parent_stage_key'],reference=row['completion'][0]['content']))
        assert pair['reference']==row['completion'][0]['content']
        for mode in (('original','sidecar_v2') if i%2==0 else ('sidecar_v2','original')):
            if mode in pair:continue
            tensor=torch.tensor([encoded[mode][i]],device=model.device)
            torch.cuda.synchronize();started=time.perf_counter()
            with torch.inference_mode():
                generated=model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),max_new_tokens=new_tokens,do_sample=False,pad_token_id=tokenizer.pad_token_id)
            completion=generated[0,len(encoded[mode][i]):].tolist()
            torch.cuda.synchronize()
            pair[mode]=dict(text=tokenizer.decode(completion,skip_special_tokens=True),seconds=time.perf_counter()-started,generated_tokens=len(completion),hit_token_limit=len(completion)>=new_tokens)
            by_id[row['job_id']]=pair
            outputs=[by_id[r['job_id']] for r in original if r['job_id'] in by_id]
            save(a.out/'paired_outputs.json',outputs)
        print(f'{sum(all(m in r for m in records) for r in outputs)}/{eligible} paired outputs saved',flush=True)
    summary=dict(total_pairs=38,eligible_pairs=eligible,completed_pairs=sum(all(m in r for m in records) for r in outputs),
        excluded_both_conditions=38-eligible,semantic_review='pending',model_training=False,
        note='Matched eligible subset only; no truncation, no label changes, no evidence of quality improvement until review',
        hardware=torch.cuda.get_device_name(0),peak_gpu_bytes=torch.cuda.max_memory_allocated())
    save(a.out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
