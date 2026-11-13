"""Offline local Qwen backend; weights are reused for all stage requests."""
import json
from pathlib import Path
import time
from copy import deepcopy

from .ledger import encoded, sha
from .claim_extraction import ExtractionError


def resolve_quotes(response, request):
    result = deepcopy(response)
    known = {e['record_id']: e['text'] for e in request['evidence']}
    for claim in result['claims']:
        for ref in claim['evidence_refs']:
            if set(ref) != {'evidence_id', 'quote'}:
                raise ExtractionError('Expected evidence_id and quote only')
            source = known.get(ref['evidence_id'])
            quote = ref['quote']
            if source is None or not isinstance(quote, str) or not quote:
                raise ExtractionError('Unknown evidence or empty quote')
            start = source.find(quote)
            if start < 0 or source.find(quote, start + 1) >= 0:
                raise ExtractionError('Quote absent or ambiguous in cited evidence')
            ref.update(start=start, end=start + len(quote))
    return result


class QwenBackend:
    mode = 'local_qwen'

    def __init__(self, model_path, max_input_tokens=24576, max_new_tokens=4096, max_seconds=120):
        if min(max_input_tokens, max_new_tokens, max_seconds) <= 0:
            raise ValueError('Budgets must be positive')
        self.path = str(Path(model_path).resolve(strict=True))
        self.input_limit, self.output_limit, self.seconds = max_input_tokens, max_new_tokens, max_seconds
        self.identity = 'qwen-quote-v2-' + sha(encoded([self.path, max_input_tokens, max_new_tokens, max_seconds]))
        self.model = self.tokenizer = None
        self.metrics = {}

    def extract(self, request):
        self.metrics, self.raw_output, self.model_request = {}, None, None
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise ExtractionError('CUDA unavailable; run in a GPU-enabled environment')
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.path, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
        packet = deepcopy(request)
        packet['backend_mode'] = self.mode
        packet['instruction'] = request['instruction'].replace('and exact Unicode character spans', 'and exact verbatim quotations')
        packet['instruction'] += (' Return at most 6 concise claims. Each evidence reference contains only evidence_id and quote; '
                                  'copy a short unique substring exactly. Do not output start/end offsets or Markdown fences. '
                                  'Use only kind and basis values listed in the schema.')
        items = packet['response_schema']['properties']['claims']
        items['maxItems'] = 6
        refs = items['items']['properties']['evidence_refs']['items']
        refs['required'] = ['evidence_id', 'quote']
        for key in ('start', 'end'):
            del refs['properties'][key]
        messages = [dict(role='system', content=packet['instruction']),
                    dict(role='user', content=json.dumps(packet, ensure_ascii=False))]
        self.model_request = messages
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                    return_tensors='pt', return_dict=True)
        count = inputs['input_ids'].shape[-1]
        if count > self.input_limit or count + self.output_limit > self.model.config.max_position_embeddings:
            raise ExtractionError('token_budget_exceeded; no truncation performed')
        inputs = inputs.to('cuda')
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.output_limit,
                                         max_time=self.seconds)
        torch.cuda.synchronize()
        tokens = output[0, count:]
        self.raw_output = self.tokenizer.decode(tokens, skip_special_tokens=True)
        self.metrics = dict(input_tokens=count, output_tokens=len(tokens),
            seconds=time.monotonic()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            timeout_policy='cooperative generation max_time; not a hard process deadline')
        if self.metrics['seconds'] > self.seconds:
            raise ExtractionError('generation_time_budget_exceeded')
        return resolve_quotes(json.loads(self.raw_output), request)
