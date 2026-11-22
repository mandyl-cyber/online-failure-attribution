"""The trained 7B short-claim protocol bridged to stage extraction records."""
from pathlib import Path
import time
from copy import deepcopy

from . import stage_claim_summary as core
from . import stage_claim_facts as facts
from .claim_evidence_review import result_scope_issues

RESULT_POLICY = '''Distinguish requested actions, progress and completed results.
A view action supports requested a view, not a claim that viewing completed.
Download progress does not establish package installation completion. Preserve
failed setup and unresolved results. Only cited observations support completion.'''
PAGE_POLICY = '''This is one page of a larger step. Original chunk IDs and offsets
are unchanged. Other pages and paired action context may be absent. Do not infer
the outcome of the full step from this page. Cite only IDs actually shown here.'''


def messages_for(request):
    source=dict(job_id=request['job_id'],run_id=request['run_id'],stage=request['stage'],
                detected_at_seq=request['detected_at_seq'],incomplete=request['incomplete'],
                task_context=request['task_context'],evidence=request['evidence'])
    row=core.build(source)
    packet=dict(row['request'],observed_facts=facts.extract(source,row['evidence_map']),
                observed_facts_version=facts.VERSION)
    instruction=core.INSTRUCTION
    if request.get('_claim_policy')=='v4':
        instruction+='\n'+RESULT_POLICY
    if '_claim_page_ids' in request:
        ids=request['_claim_page_ids']
        if not ids or len(ids)!=len(set(ids)) or any(i not in row['evidence_map'] for i in ids):
            raise ValueError('Invalid page evidence IDs')
        packet=deepcopy(packet)
        for unit in packet['units']:
            for field in ('action','observation'):
                parents=[]
                for parent in unit[field]:
                    parent['chunks']=[c for c in parent['chunks'] if c['id'] in ids]
                    if parent['chunks']:
                        parents.append(parent)
                unit[field]=parents
            unit['missing_fields']=[f for f in ('action','observation') if not unit[f]]
        # Full-record derived facts can expose results outside the visible page.
        packet['observed_facts']=[]
        packet['page']=dict(visible_evidence_ids=ids,total_chunks=len(row['evidence_map']),
                            partial_step_context=True)
        instruction+='\n'+PAGE_POLICY
    return row,[dict(role='system',content=instruction),dict(role='user',content=core.encoded(packet).decode())]


def convert_response(response,row):
    validated=core.validate(response,row)
    claims=[]
    for i,claim in enumerate(validated,1):
        claims.append(dict(local_id=f'c{i}',kind='decision' if claim['source']=='agent_assertion' else 'conclusion',
            text=claim['text'],basis=claim['source'],objects=[],conditions=[],unknowns=['semantic_entailment'],
            evidence_refs=[dict(evidence_id=r['record_id'],start=r['start'],end=r['end'],quote=r['quote'])
                           for r in claim['evidence_refs']]))
    return dict(claims=claims)


def convert_partial(response,row,allowed_ids=None,enforce_result_scope=False):
    if not isinstance(response,dict) or set(response)!={'claims'} or not isinstance(response['claims'],list):
        raise ValueError('expected claims object')
    accepted,rejected=[],[]
    for index,claim in enumerate(response['claims']):
        try:
            converted=convert_response(dict(claims=[claim]),row)['claims']
            if allowed_ids is not None and any(e not in allowed_ids for e in claim['evidence_ids']):
                raise ValueError('evidence ID not visible on this page')
            if enforce_result_scope:
                refs=[row['evidence_map'][eid] for eid in claim['evidence_ids']]
                quoted='\n'.join(r['quote'] for r in refs if r['field']=='observation')
                issues=result_scope_issues(claim['text'],quoted,has_observation=any(r['field']=='observation' for r in refs))
                if issues:
                    raise ValueError('result_scope: '+','.join(i['code'] for i in issues))
            accepted.extend(converted)
        except (ValueError,KeyError,TypeError) as error:
            rejected.append(dict(claim_index=index,raw_claim=deepcopy(claim),error=str(error)))
    for index,claim in enumerate(accepted,1):
        claim['local_id']=f'c{index}'
    return dict(claims=accepted),rejected


def partition_requests(request,fits):
    """Bisect whole-step ranges, preserving evidence records and seal cutoff."""
    parts,oversized=[],[]
    def visit(start,end):
        child=deepcopy(request)
        child['stage'].update(start_step_id=start,end_step_id=end)
        child['evidence']=[e for e in request['evidence'] if start<=e['source_step_id']<=end]
        if fits(child):
            parts.append(child)
        elif start==end:
            oversized.append(start)
        else:
            middle=(start+end)//2
            visit(start,middle)
            visit(middle+1,end)
    visit(request['stage']['start_step_id'],request['stage']['end_step_id'])
    return parts,oversized


def partition_paged_requests(request,fits):
    """Keep whole-step partitions, paging only single steps that still exceed budget.

    Pages select original chunks; native evidence is never sliced, rehashed or
    renumbered. Every original chunk is either visible once or an explicit gap.
    """
    parts,oversized=partition_requests(request,fits)
    gaps=[]
    for sid in oversized:
        child=deepcopy(request)
        child['stage'].update(start_step_id=sid,end_step_id=sid)
        child['evidence']=[e for e in request['evidence'] if e['source_step_id']==sid]
        ids=list(messages_for(child)[0]['evidence_map'])
        def visit(selected):
            page=deepcopy(child)
            page['_claim_page_ids']=selected
            if selected and fits(page):
                parts.append(page)
            elif len(selected)>1:
                middle=len(selected)//2
                visit(selected[:middle])
                visit(selected[middle:])
            else:
                gaps.append(dict(step_id=sid,evidence_ids=selected,reason='minimum_page_exceeds_budget'))
        visit(ids)
    parts.sort(key=lambda p:(p['stage']['start_step_id'],
        list(messages_for(p)[0]['evidence_map']).index(p['_claim_page_ids'][0]) if '_claim_page_ids' in p else -1))
    return parts,gaps


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
        self.identity=core.digest(core.encoded(dict(version='claim-sft-ledger-v4-original-chunk-pages',base=str(self.model_path),
            files={p.name:core.digest(p.read_bytes()) for p in files},instruction=core.INSTRUCTION,
            result_policy=RESULT_POLICY,page_policy=PAGE_POLICY,
            max_length=max_length,max_new_tokens=max_new_tokens)))
        self.model=self.tokenizer=None
        self.metrics={}

    @staticmethod
    def retryable_error(error):
        # Generation is deterministic for the same request. Invalid budgets,
        # JSON and provenance require a changed input or a review decision.
        return not isinstance(error,(ValueError,KeyError,TypeError))

    def extract(self,request):
        self.extraction_details=None
        self.metrics,self.raw_output,self.model_request={},None,None
        from transformers import AutoTokenizer
        from remote_pilots.openhands_smoke.claim_sft_train import chat_ids
        if self.tokenizer is None:
            self.tokenizer=AutoTokenizer.from_pretrained(str(self.model_path),local_files_only=True)
        def fits(child):
            _,messages=messages_for(child)
            return len(chat_ids(self.tokenizer,messages,True))+self.max_new_tokens<=self.max_length
        request=deepcopy(request)
        request['_claim_policy']='v4'
        parts,gaps=partition_paged_requests(request,fits)
        oversized=sorted({g['step_id'] for g in gaps})
        claims,records,rejections=[],[],[]
        for index,child in enumerate(parts):
            self.metrics,self.raw_output,self.model_request={},None,None
            try:
                response=self._generate(child)
                claims.extend(response['claims'])
                rejections.extend(dict(r,part_index=index) for r in self.part_rejections)
                record=dict(part_index=index,stage=child['stage'],status='generated',rejected_claims=len(self.part_rejections))
            except Exception as error:
                record=dict(part_index=index,stage=child['stage'],status='failed',error=str(error))
            record.update(raw_output=self.raw_output,metrics=self.metrics,model_request=self.model_request)
            record['page_evidence_ids']=child.get('_claim_page_ids')
            records.append(record)
        unique,duplicates,seen=[],[],set()
        for claim in claims:
            fingerprint=core.digest(core.encoded({k:v for k,v in claim.items() if k!='local_id'}))
            if fingerprint in seen:
                duplicates.append(claim)
            else:
                seen.add(fingerprint)
                unique.append(claim)
        claims=unique
        # The native contract caps the batch at64. Excess claims remain explicit.
        for claim in claims[64:]:
            rejections.append(dict(raw_claim=claim,error='parent_batch_claim_limit'))
        claims=claims[:64]
        for index,claim in enumerate(claims,1):
            claim['local_id']=f'c{index}'
        self.extraction_details=dict(parts=[{k:v for k,v in r.items() if k not in ('raw_output','model_request')} for r in records],
            oversized_steps=oversized,unprocessed_pages=gaps,rejected_claims=rejections,duplicate_claims=duplicates,input_complete=not gaps,
            complete=not oversized and not rejections and all(r['status']=='generated' for r in records))
        self.raw_output=records
        self.model_request=None
        self.metrics=dict(parts=len(records),generation_seconds=sum(r['metrics'].get('generation_seconds',0.) for r in records),
            peak_gpu_bytes=max((r['metrics'].get('peak_gpu_bytes',0) for r in records),default=0))
        return dict(claims=claims)

    def _generate(self,request):
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
        response,self.part_rejections=convert_partial(core.strict_json(self.raw_output),row,
            allowed_ids=request.get('_claim_page_ids'),enforce_result_scope=True)
        return response
