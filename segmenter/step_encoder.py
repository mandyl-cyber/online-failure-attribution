"""Pinned local step encoder, deterministic budgets, reusable frozen embeddings."""
import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET','1')
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModel, AutoTokenizer
from .encoder_data import bounded

MODELS = {'minilm':'sentence-transformers/all-MiniLM-L6-v2',
          'modernbert':'answerdotai/ModernBERT-base'}
KEY_LINE = re.compile(r'\b(passed|failed|error|exception|traceback|assert|warning|success|collected|test result)\b',re.I)
PACK_VERSION = 1


def pack_step(row, tokenizer, policy='headtail', length=256):
    """One immutable step only. Labels, thought and neighboring steps are ignored."""
    special = tokenizer.num_special_tokens_to_add(False)
    header = tokenizer.encode(f"TOOL {row['tool']} EXIT {row['exit_code']} ACTION",add_special_tokens=False)[:24-special]
    action = tokenizer.encode(bounded(row['action'],8192),add_special_tokens=False,truncation=False)
    action_budget = min(80,len(action))
    action_ids = action if len(action)<=action_budget else action[:action_budget//2]+action[-(action_budget-action_budget//2):]
    marker = tokenizer.encode(' OBSERVATION ',add_special_tokens=False)
    obs_budget = length-special-len(header)-len(action_ids)-len(marker)
    obs = bounded(row['observation'])
    obs_ids = tokenizer.encode(obs,add_special_tokens=False,truncation=False)
    if policy=='keylines' and len(obs_ids)>obs_budget:
        # Tokenize lines independently so selected positions have an unambiguous
        # source order. Deduplicate overlaps between head, tail and signal lines.
        line_ids, key_positions = [], []
        for line in obs.splitlines(keepends=True):
            part = tokenizer.encode(line,add_special_tokens=False)
            if KEY_LINE.search(line):
                key_positions.extend(range(len(line_ids),len(line_ids)+len(part)))
            line_ids.extend(part)
        selected = set(key_positions[:obs_budget//2])
        for i in range(len(line_ids)):
            for position in (i,len(line_ids)-1-i):
                if len(selected)<obs_budget:
                    selected.add(position)
            if len(selected)>=min(obs_budget,len(line_ids)):
                break
        kept = [line_ids[i] for i in sorted(selected)]
    else:
        kept = obs_ids if len(obs_ids)<=obs_budget else obs_ids[:obs_budget//2]+obs_ids[-(obs_budget-obs_budget//2):]
    ids = tokenizer.build_inputs_with_special_tokens(header+action_ids+marker+kept)
    assert len(ids)<=length
    return ids, dict(action_tokens=len(action),observation_tokens=len(obs_ids),tokens=len(ids),
                     action_truncated=len(action)>len(action_ids),observation_truncated=len(obs_ids)>len(kept))


def download(name):
    repo = MODELS[name]
    out = Path('vendor/encoder_models')/name
    metadata = out/'pinned_revision.json'
    if metadata.exists():
        return out
    revision = HfApi().model_info(repo).sha
    snapshot_download(repo_id=repo,revision=revision,local_dir=out,
                      cache_dir='vendor/encoder_hf_cache',
                      allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json',
                                      'special_tokens_map.json','vocab.txt','merges.txt','vocab.json',
                                      'sentence_bert_config.json','1_Pooling/config.json','README.md','LICENSE*'])
    metadata.write_text(json.dumps(dict(repo=repo,revision=revision),indent=2),encoding='utf-8')
    return out


class StepEncoder:
    def __init__(self, path, device='cpu', dtype='float32', attention='eager'):
        self.path = str(path)
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model = AutoModel.from_pretrained(path,local_files_only=True,attn_implementation=attention,
                                               torch_dtype=getattr(torch,dtype)).to(device).eval()
        # ModernBERT's optional compilation is not assumed on native Windows.
        if hasattr(self.model.config,'reference_compile'):
            self.model.config.reference_compile = False

    def embed(self, token_rows):
        batch = self.tokenizer.pad({'input_ids':token_rows},padding=True,return_tensors='pt')
        batch = {k:v.to(self.device) for k,v in batch.items() if k in ('input_ids','attention_mask')}
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state.float()
            mask = batch['attention_mask'].unsqueeze(-1)
            pooled = (hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
            return F.normalize(pooled,p=2,dim=1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',choices=list(MODELS),default='minilm')
    ap.add_argument('--text',default='eval/boundary_encoder/text_v1')
    ap.add_argument('--out',default='eval/boundary_encoder/minilm_headtail')
    ap.add_argument('--policy',choices=['headtail','keylines'],default='headtail')
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--download-only',action='store_true')
    ap.add_argument('--attention',choices=['eager','sdpa'],default='sdpa')
    a = ap.parse_args()
    if a.device=='cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; configure environment or explicitly choose --device cpu')
    path = download(a.model)
    if a.download_only:
        print(path)
        return
    out = Path(a.out)
    if out.exists():
        raise SystemExit('Output exists; use a fresh path')
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    encoder = StepEncoder(path,a.device,'float16' if a.device=='cuda' else 'float32',a.attention)
    text_path = Path(a.text)/'steps.jsonl'
    trajectories = [json.loads(line) for line in text_path.read_text(encoding='utf-8').splitlines()]
    rows, index, stats = [],[],[]
    start = time.perf_counter()
    for trajectory in trajectories:
        for r in trajectory['rows']:
            ids, audit = pack_step(r,encoder.tokenizer,a.policy)
            rows.append(ids); stats.append(audit)
            index.append(dict(traj_id=trajectory['traj_id'],idx=r['idx']))
    preprocess_seconds = time.perf_counter()-start
    start = time.perf_counter()
    embeddings = []
    for i in range(0,len(rows),a.batch_size):
        embeddings.append(encoder.embed(rows[i:i+a.batch_size]))
        if i%(a.batch_size*40)==0:
            print(f'{a.model}: {i}/{len(rows)} steps encoded',flush=True)
    arr = np.concatenate(embeddings).astype(np.float32)
    np.save(out/'embeddings.npy',arr)
    (out/'index.json').write_text(json.dumps(index),encoding='utf-8')
    # Ragged token lists allow later supervised fine-tuning with identical preprocessing.
    (out/'tokens.json').write_text(json.dumps(rows),encoding='utf-8')
    (out/'packed_texts.json').write_text(json.dumps(encoder.tokenizer.batch_decode(rows,skip_special_tokens=True)),encoding='utf-8')
    (out/'input_audit.json').write_text(json.dumps(stats),encoding='utf-8')
    audit = dict(model=a.model,model_path=str(path),revision=json.loads((path/'pinned_revision.json').read_text()),
                 parameters=sum(p.numel() for p in encoder.model.parameters()),shape=list(arr.shape),
                 policy=a.policy,pack_version=PACK_VERSION,max_tokens=256,device=a.device,dtype=encoder.dtype,
                 attention=a.attention,
                 preprocessing_seconds=preprocess_seconds,encoding_seconds=time.perf_counter()-start,
                 peak_cuda_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else None,
                 text_sha256=hashlib.sha256(text_path.read_bytes()).hexdigest(),
                 action_truncated_steps=sum(r['action_truncated'] for r in stats),
                 observation_truncated_steps=sum(r['observation_truncated'] for r in stats),
                 frozen_pretrained=True,labels_used=False)
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
