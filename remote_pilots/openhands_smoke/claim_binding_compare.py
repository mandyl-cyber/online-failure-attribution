"""Matched old-SFT versus new-SFT inference, shared base, identical prompts/budgets."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from scripts.prepare_claim_binding_sft import verify_package
from scripts.run_stage_claim_pipeline import read,save
from segmenter import stage_claim_summary as core
from segmenter.ledger import sha,encoded


def structural_check(text,prompt):
    packet=core.strict_json(prompt[1]['content'])
    mapping={c['id']:field for unit in packet['units'] for field in ('action','observation')
             for parent in unit[field] for c in parent['chunks']}
    try:
        response=core.strict_json(text)
        if set(response)!={'claims'} or not isinstance(response['claims'],list):raise ValueError('Expected claims object')
        for claim in response['claims']:
            if set(claim)!={'text','source','evidence_ids'} or not isinstance(claim['text'],str) or not claim['text'].strip():
                raise ValueError('Invalid claim fields')
            ids=claim['evidence_ids']
            if not isinstance(ids,list) or not ids or any(not isinstance(e,str) or e not in mapping for e in ids) or len(set(ids))!=len(ids):
                raise ValueError('Unknown/duplicate evidence IDs')
            if claim['source'] not in ('agent_assertion','tool_observation'):raise ValueError('Invalid source')
            if claim['source']=='tool_observation' and not any(mapping[e]=='observation' for e in ids):
                raise ValueError('Observation claim has no observation citation')
        return dict(valid=True,claims=len(response['claims']),semantic_correctness='not_scored')
    except (ValueError,TypeError,KeyError) as error:
        return dict(valid=False,error=str(error),semantic_correctness='not_scored')


def adapter_hash(path):
    files=[path/'adapter_config.json',*sorted(path.glob('adapter_model.*'))]
    if len(files)<2:raise FileNotFoundError('Adapter missing: '+str(path))
    return sha(encoded({p.name:sha(p.read_bytes()) for p in files}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','run'])
    parser.add_argument('--data',type=Path,default=Path('pilot/claim_binding_sft_v1'))
    parser.add_argument('--model-path',type=Path)
    parser.add_argument('--baseline-run',type=Path,default=Path('out/claim_sft_7b_reviewed_v2'))
    parser.add_argument('--candidate-run',type=Path,default=Path('out/claim_binding_sft_v1'))
    parser.add_argument('--out',type=Path,default=Path('out/claim_binding_compare_v1'))
    args=parser.parse_args()
    _,protocol=verify_package(args.data)
    rows=[]
    for split in ('dev','regression'):
        rows.extend(dict(row=json.loads(line),split=split) for line in (args.data/f'{split}.jsonl').read_text(encoding='utf-8').splitlines())
    if args.command=='check':
        print(dict(train=protocol['train'],dev=38,known_regression=11,inference_inputs=len(rows),model_calls=0))
        return
    if args.model_path is None:parser.error('--model-path required')
    for run in (args.baseline_run,args.candidate_run):
        if Path(read(run/'config.json')['model_path']).resolve()!=args.model_path.resolve():
            raise ValueError('Adapters must use the same original base model')
    candidate_config=read(args.candidate_run/'config.json')
    if any(candidate_config['data_hashes'][s]!=protocol['data_hashes'][s] for s in ('train','dev')):
        raise ValueError('Candidate was not trained with the declared data')
    if read(args.baseline_run/'config.json')['data_hashes']['dev']!=protocol['data_hashes']['dev']:
        raise ValueError('Baseline development inputs differ')
    hashes={name:adapter_hash(run/'adapter') for name,run in [('baseline',args.baseline_run),('candidate',args.candidate_run)]}
    if hashes['baseline']==hashes['candidate']:raise ValueError('Baseline and candidate adapter hashes are identical')
    config=dict(data_hashes=protocol['data_hashes'],adapter_hashes=hashes,model_path=str(args.model_path.resolve()),
        max_length=8192,max_new_tokens=512,do_sample=False,order='alternate_by_input',training=False,
        evaluation_scope='Task-disjoint known development and regression only; no independent acceptance claim')
    if (args.out/'config.json').exists() and read(args.out/'config.json')!=config:
        raise ValueError('Comparison configuration changed; use a new output directory')
    save(args.out/'config.json',config)
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
    from peft import PeftModel,prepare_model_for_kbit_training
    from .claim_sft_train import chat_ids
    tokenizer=AutoTokenizer.from_pretrained(str(args.model_path),local_files_only=True)
    started=time.perf_counter()
    base=AutoModelForCausalLM.from_pretrained(str(args.model_path),local_files_only=True,dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),device_map={'':0})
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=False)
    model=PeftModel.from_pretrained(base,str(args.baseline_run/'adapter'),adapter_name='baseline',local_files_only=True)
    model.load_adapter(str(args.candidate_run/'adapter'),adapter_name='candidate',local_files_only=True)
    model.eval()
    model.config.use_cache=True
    load_seconds=time.perf_counter()-started
    pairs=[]
    for index,item in enumerate(rows):
        row=item['row'];path=args.out/'pairs'/f'{index+1:04d}.json'
        fingerprint=sha(encoded(row['prompt']))
        if path.exists():
            pair=read(path)
            if pair['input_sha256']!=fingerprint or pair['job_id']!=row['job_id']:raise ValueError('Cached input changed')
            pairs.append(pair)
            continue
        ids=chat_ids(tokenizer,row['prompt'],True)
        if len(ids)+512>8192:raise ValueError('Input budget exceeded; no truncation')
        tensor=torch.tensor([ids],device=model.device)
        pair=dict(job_id=row['job_id'],group=row['group'],split=item['split'],input_sha256=fingerprint,
            prompt=row['prompt'],reference=row['completion'][0]['content'],arms={})
        for arm in (('baseline','candidate') if index%2==0 else ('candidate','baseline')):
            model.set_adapter(arm)
            torch.cuda.synchronize();started=time.perf_counter()
            with torch.inference_mode():
                output=model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),max_new_tokens=512,
                    do_sample=False,pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            torch.cuda.synchronize()
            tokens=output[0,len(ids):].tolist()
            text=tokenizer.decode(tokens,skip_special_tokens=True)
            pair['arms'][arm]=dict(text=text,generated_tokens=len(tokens),hit_token_limit=len(tokens)>=512,
                seconds=time.perf_counter()-started,structural_check=structural_check(text,row['prompt']))
        save(path,pair)
        pairs.append(pair)
        print(dict(done=index+1,total=len(rows),split=item['split']),flush=True)
    summary=dict(pairs=len(pairs),splits=dict(Counter(p['split'] for p in pairs)),
        structural_valid={arm:sum(p['arms'][arm]['structural_check']['valid'] for p in pairs) for arm in ('baseline','candidate')},
        peak_gpu_GiB=torch.cuda.max_memory_allocated()/1024**3,model_load_seconds=load_seconds,
        semantic_review='pending_AI_paired_review',independent_acceptance=False,
        scope=config['evaluation_scope'])
    save(args.out/'summary.json',summary)
    print(summary)


if __name__=='__main__':main()
