"""Batch-1 CPU/GPU latency and cache equivalence. No training or label access."""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from .step_encoder import StepEncoder, pack_step


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',default='eval/boundary_encoder/minilm_headtail')
    ap.add_argument('--text',default='eval/boundary_encoder/text_v1')
    ap.add_argument('--out',default='eval/boundary_encoder/minilm_runtime.json')
    ap.add_argument('--device',choices=['all','cpu','cuda'],default='all')
    a=ap.parse_args()
    cache=Path(a.cache)
    audit=json.loads((cache/'audit.json').read_text())
    idx=json.loads((cache/'index.json').read_text())
    lookup={(r['traj_id'],r['idx']):i for i,r in enumerate(idx)}
    cached=np.load(cache/'embeddings.npy')
    ts=[json.loads(s) for s in (Path(a.text)/'steps.jsonl').read_text(encoding='utf-8').splitlines()]
    all_rows=[(t['traj_id'],r) for t in ts for r in t['rows']]
    # Uniformly cover the corpus without using labels, including short and long inputs.
    selected=[all_rows[i] for i in np.linspace(0,len(all_rows)-1,128,dtype=int)]
    result={}
    for device,dtype in [('cpu','float32'),('cuda','float16')]:
        if a.device not in ('all',device):
            continue
        if device=='cuda' and not torch.cuda.is_available():
            continue
        torch.set_num_threads(4)
        started=time.perf_counter()
        enc=StepEncoder(audit['model_path'],device,dtype,audit.get('attention','eager'))
        load_seconds=time.perf_counter()-started
        for _,r in selected[:8]:
            ids,_=pack_step(r,enc.tokenizer,audit['policy']); enc.embed([ids])
        total,pre,forward,similarity=[],[],[],[]
        for tid,r in selected:
            start=time.perf_counter()
            ids,_=pack_step(r,enc.tokenizer,audit['policy'])
            ready=time.perf_counter()
            vector=enc.embed([ids])[0] # .cpu() synchronizes GPU before timing ends
            end=time.perf_counter()
            total.append((end-start)*1000);pre.append((ready-start)*1000);forward.append((end-ready)*1000)
            similarity.append(float(vector@cached[lookup[(tid,r['idx'])]]))
        def q(xs):
            return dict(zip(('p50','p95','p99'),map(float,np.quantile(xs,[.5,.95,.99]))))
        result[device]=dict(dtype=dtype,threads=4,samples=len(selected),load_seconds=load_seconds,
                            preprocess_ms=q(pre),forward_pool_transfer_ms=q(forward),total_ms=q(total),
                            min_cosine_with_cached=float(min(similarity)),
                            max_cuda_allocated=torch.cuda.max_memory_allocated() if device=='cuda' else None)
        del enc
        if device=='cuda': torch.cuda.empty_cache()
    report=dict(model=audit['revision'],hardware=platform.processor(),torch=torch.__version__,
                batch_size=1,warmup=8,results=result,
                scope='128 representative steps, 4 CPU threads; includes tokenization, pooling and transfer; excludes classification head, event I/O and queueing; not concurrent-load or real-agent benchmark')
    Path(a.out).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
