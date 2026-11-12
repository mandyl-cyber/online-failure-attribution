"""Sequential, logged continuation of the preregistered encoder experiment.

Waits for the separately running frozen ModernBERT encoding, then executes the
remaining local experiments. No external evaluation or deployment promotion.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='eval/boundary_encoder/continuation')
    ap.add_argument('--wait-hours',type=float,default=4.)
    a=ap.parse_args()
    out=Path(a.out)
    if out.exists(): raise SystemExit('Output exists; choose a fresh orchestration directory')
    out.mkdir(parents=True)
    state=dict(status='waiting_for_frozen_encoding',completed=[],running=None)
    def save():
        state['updated_unix']=time.time()
        p=out/'status.json.tmp'
        p.write_text(json.dumps(state,indent=2),encoding='utf-8')
        p.replace(out/'status.json')
    save()
    prerequisite=Path('eval/boundary_encoder/modernbert_headtail/audit.json')
    deadline=time.monotonic()+a.wait_hours*3600
    while not prerequisite.exists():
        if time.monotonic()>deadline:
            state.update(status='failed',error='Frozen encoding did not finish before wait timeout');save()
            raise SystemExit(state['error'])
        time.sleep(10)
    steps=[
        ('modernbert_cv',['segmenter.train_encoder_boundary','--cache','eval/boundary_encoder/modernbert_headtail','--out','eval/boundary_encoder/modernbert_cv']),
        ('minilm_runtime',['segmenter.bench_step_encoder','--out','eval/boundary_encoder/minilm_runtime_isolated.json']),
        ('modernbert_runtime',['segmenter.bench_step_encoder','--cache','eval/boundary_encoder/modernbert_headtail','--out','eval/boundary_encoder/modernbert_runtime.json']),
        ('prefix_cache',['segmenter.finetune_encoder_boundary','prepare']),
        ('finetune_resource_probe',['segmenter.finetune_encoder_boundary','train','--fold','0','--probe-batches','8','--out','eval/boundary_encoder/modernbert_finetune_probe']),
        ('modernbert_finetune',['segmenter.finetune_encoder_boundary','train']),
    ]
    for name,args in steps:
        command=[sys.executable,'-u','-m']+args
        state.update(status='running',running=name,command=command);save()
        print('Starting '+name,flush=True)
        with (out/(name+'.log')).open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=process.pid;save()
            code=process.wait()
        if code:
            state.update(status='failed',returncode=code);save()
            raise SystemExit(f'{name} failed; inspect {out/(name+".log")}')
        state['completed'].append(name)
        state.update(running=None,child_pid=None);save()
    state.update(status='complete');save()
    print('Frozen comparisons, isolated runtime and five-fold partial adaptation complete.',flush=True)


if __name__=='__main__': main()
