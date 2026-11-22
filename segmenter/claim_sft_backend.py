"""The trained 7B short-claim protocol bridged to stage extraction records."""
from pathlib import Path
import time

from . import stage_claim_summary as core
from . import stage_claim_facts as facts


def messages_for(request):
    source=dict(job_id=request['job_id'],run_id=request['run_id'],stage=request['stage'],
                detected_at_seq=request['detected_at_seq'],incomplete=request['incomplete'],
                task_context=request['task_context'],evidence=request['evidence'])
    row=core.build(source)
    packet=dict(row['request'],observed_facts=facts.extract(source,row['evidence_map']),
                observed_facts_version=facts.VERSION)
    return row,[dict(role='system',content=core.INSTRUCTION),dict(role='user',content=core.encoded(packet).decode())]


def convert_response(response,row):
    validated=core.validate(response,row)
    claims=[]
    for i,claim in enumerate(validated,1):
        claims.append(dict(local_id=f'c{i}',kind='decision' if claim['source']=='agent_assertion' else 'conclusion',
            text=claim['text'],basis=claim['source'],objects=[],conditions=[],unknowns=['semantic_entailment'],
            evidence_refs=[dict(evidence_id=r['record_id'],start=r['start'],end=r['end'],quote=r['quote'])
                           for r in claim['evidence_refs']]))
    return dict(claims=claims)


class ClaimSFTBackend:
    mode='local_7b_sft'

    def __init__(self,model_path,adapter_path,max_length=8192,max_new_tokens=512):
        self.model_path=Path(model_path).resolve(strict=True)
        self.adapter_path=Path(adapter_path).resolve(strict=True)
        self.max_length,self.max_new_tokens=max_length,max_new_tokens
        if not 0<max_new_tokens<max_length:
            raise ValueError('Invalid token budget')
        files=[self.model_path/'config.json',self.adapter_path/'adapter_config.json',*sorted(self.adapter_path.glob('adapter_model.*'))]
        if len(files)<3:
            raise FileNotFoundError('Adapter weights missing')
        self.identity=core.digest(core.encoded(dict(version='claim-sft-ledger-v1',base=str(self.model_path),
            files={p.name:core.digest(p.read_bytes()) for p in files},instruction=core.INSTRUCTION,
            max_length=max_length,max_new_tokens=max_new_tokens)))
        self.model=self.tokenizer=None
        self.metrics={}

    def extract(self,request):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
        from peft import PeftModel,prepare_model_for_kbit_training
        from remote_pilots.openhands_smoke.claim_sft_train import chat_ids
        self.metrics,self.raw_output={},None
        row,messages=messages_for(request)
        self.model_request=messages
        if self.tokenizer is None:
            self.tokenizer=AutoTokenizer.from_pretrained(str(self.model_path),local_files_only=True)
        ids=chat_ids(self.tokenizer,messages,True)
        if len(ids)+self.max_new_tokens>self.max_length:
            raise ValueError(f'token_budget_exceeded: {len(ids)}+{self.max_new_tokens}>{self.max_length}')
        loaded_now=self.model is None
        if loaded_now:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required for the 7B backend')
            started=time.perf_counter()
            model=AutoModelForCausalLM.from_pretrained(str(self.model_path),local_files_only=True,dtype=torch.bfloat16,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),device_map={'':0})
            model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
            self.model=PeftModel.from_pretrained(model,str(self.adapter_path),local_files_only=True).eval()
            self.model.config.use_cache=True
            self.load_seconds=time.perf_counter()-started
        if len(ids)+self.max_new_tokens>self.model.config.max_position_embeddings:
            raise ValueError('Model context budget exceeded')
        tensor=torch.tensor([ids],device=self.model.device)
        torch.cuda.synchronize()
        started=time.perf_counter()
        with torch.inference_mode():
            result=self.model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),do_sample=False,
                max_new_tokens=self.max_new_tokens,pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        torch.cuda.synchronize()
        tokens=result[0,len(ids):].tolist()
        self.raw_output=self.tokenizer.decode(tokens,skip_special_tokens=True)
        self.metrics=dict(input_tokens=len(ids),output_tokens=len(tokens),generation_seconds=time.perf_counter()-started,
            model_load_seconds=self.load_seconds if loaded_now else 0.,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        if len(tokens)>=self.max_new_tokens:
            raise ValueError('Generation output budget reached')
        return convert_response(core.strict_json(self.raw_output),row)
