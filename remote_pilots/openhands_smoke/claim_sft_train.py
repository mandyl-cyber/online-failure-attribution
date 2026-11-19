"""Token preflight, completion-only QLoRA training and paired base/adapter generation."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def encode_reference(tokenizer, row):
    prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(row["prompt"]+row["completion"], tokenize=True,
                                        add_generation_prompt=False)
    if full[:len(prompt)] != prompt or len(full) <= len(prompt):
        raise ValueError("Chat template does not preserve the assistant prompt prefix")
    return dict(input_ids=full, attention_mask=[1]*len(full),
                labels=[-100]*len(prompt)+full[len(prompt):])


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["preflight","train"])
    parser.add_argument("--data",type=Path,default=Path("pilot/claim_sft_reviewed_v2"))
    parser.add_argument("--model-path",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=Path("out/claim_sft_7b_v1"))
    parser.add_argument("--max-length",type=int,default=8192)
    parser.add_argument("--max-new-tokens",type=int,default=1024)
    parser.add_argument("--epochs",type=float,default=2)
    parser.add_argument("--seed",type=int,default=41)
    args=parser.parse_args()
    from transformers import AutoConfig, AutoTokenizer, set_seed
    set_seed(args.seed)
    model_config=AutoConfig.from_pretrained(str(args.model_path),local_files_only=True)
    if getattr(model_config,"quantization_config",None):
        raise ValueError("Use original pretrained weights for QLoRA, not an AWQ/GPTQ checkpoint")
    tokenizer=AutoTokenizer.from_pretrained(str(args.model_path),local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    raw={split:read_rows(args.data/f"{split}.jsonl") for split in ("train","dev")}
    if {r["group"] for r in raw["train"]} & {r["group"] for r in raw["dev"]}:
        raise ValueError("Train/development task overlap")
    config=dict(model_path=str(args.model_path.resolve()),max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,epochs=args.epochs,seed=args.seed,
        data_hashes={s:hashlib.sha256((args.data/f"{s}.jsonl").read_bytes()).hexdigest() for s in raw},
        lora=dict(rank=16,alpha=32,dropout=0.05,target_modules="all-linear"),
        learning_rate=1e-4,gradient_accumulation_steps=8,quantization="nf4 double quantization")
    if (args.out/"config.json").exists() and json.loads((args.out/"config.json").read_text())!=config:
        raise ValueError("Configuration changed; choose another output directory")
    args.out.mkdir(parents=True,exist_ok=True)
    save(args.out/"config.json",config)
    encoded={}
    eligible={}
    checks=[]
    for split, records in raw.items():
        encoded[split]=[]
        eligible[split]=[]
        for row in records:
            item=encode_reference(tokenizer,row)
            prompt_length=item["labels"].count(-100)
            fits=(len(item["input_ids"])<=args.max_length
                  and prompt_length+args.max_new_tokens<=args.max_length
                  and len(item["input_ids"])-prompt_length<=args.max_new_tokens)
            checks.append(dict(job_id=row["job_id"],split=split,tokens=len(item["input_ids"]),
                               prompt_tokens=prompt_length,
                               reference_tokens=len(item["input_ids"])-prompt_length,eligible=fits))
            if fits:
                encoded[split].append(item)
                eligible[split].append(row)
    save(args.out/"preflight.json",checks)
    print({s:len(encoded[s]) for s in encoded},flush=True)
    if args.command=="preflight":
        return
    if not encoded["train"] or not encoded["dev"]:
        raise ValueError("No eligible training or development examples")
    if (args.out/"adapter").exists():
        raise ValueError("Adapter output already exists; use a new output directory")
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    started=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(str(args.model_path),local_files_only=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),
        device_map={"":0},dtype=torch.bfloat16)
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,
                                          target_modules="all-linear",task_type="CAUSAL_LM"))
    model.config.use_cache=False
    def collate(batch):
        length=max(len(row["input_ids"]) for row in batch)
        return {key:torch.tensor([row[key]+[padding]*(length-len(row[key])) for row in batch])
                for key,padding in (("input_ids",tokenizer.pad_token_id),("attention_mask",0),("labels",-100))}
    training=TrainingArguments(output_dir=str(args.out/"checkpoints"),num_train_epochs=args.epochs,
        per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=8,
        learning_rate=1e-4,bf16=True,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False},logging_steps=1,
        save_strategy="epoch",save_total_limit=2,eval_strategy="epoch",
        report_to="none",seed=args.seed,remove_unused_columns=False,prediction_loss_only=True)
    trainer=Trainer(model=model,args=training,train_dataset=encoded["train"],
        eval_dataset=encoded["dev"],data_collator=collate)
    trained=trainer.train()
    trainer.save_model(str(args.out/"adapter"))
    tokenizer.save_pretrained(str(args.out/"adapter"))
    save(args.out/"training_metrics.json",dict(metrics=trained.metrics,
        seconds=time.perf_counter()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        versions={name:__import__(name).__version__ for name in ("torch","transformers","peft","bitsandbytes")}))
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache=True
    outputs=[]
    for row in eligible["dev"]:
        ids=tokenizer.apply_chat_template(row["prompt"],tokenize=True,add_generation_prompt=True)
        tensor=torch.tensor([ids],device=model.device)
        pair=dict(job_id=row["job_id"],group=row["group"],
                  parent_stage_key=row.get("parent_stage_key",row["job_id"]),
                  reference=row["completion"][0]["content"])
        for mode in ("base","adapter"):
            context=model.disable_adapter() if mode=="base" else nullcontext()
            before=time.perf_counter()
            with context,torch.inference_mode():
                generated=model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),
                    max_new_tokens=args.max_new_tokens,do_sample=False,pad_token_id=tokenizer.pad_token_id)
            completion=generated[0,len(ids):].tolist()
            pair[mode]=dict(text=tokenizer.decode(completion,skip_special_tokens=True),
                generated_tokens=len(completion),seconds=time.perf_counter()-before,
                hit_token_limit=len(completion)>=args.max_new_tokens)
        outputs.append(pair)
        save(args.out/"paired_dev_outputs.json",outputs)
        review=[dict(job_id=p["job_id"],group=p["group"],reference=p["reference"],
            base=p["base"],adapter=p["adapter"],reviewed=False,
            base_support_rate=None,adapter_support_rate=None,
            base_key_fact_coverage=None,adapter_key_fact_coverage=None,
            base_severe_errors=[],adapter_severe_errors=[],note="")
            for p in outputs]
        save(args.out/"paired_dev_review.json",review)
    save(args.out/"summary.json",dict(train=len(encoded["train"]),dev=len(encoded["dev"]),
        paired_outputs=len(outputs),semantic_review="pending",hardware=torch.cuda.get_device_name(0)))


if __name__=="__main__":
    main()
