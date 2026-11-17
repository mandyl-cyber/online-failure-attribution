"""Rebuild SWE features from repaired raw inputs, preserving frozen task folds."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .adapters.openhands import load_trajectory
from .bench import iter_bench
from .boundary_features import BoundaryFeatures
from .rules import Rules
from .temporal_boundary import StepEncoder, VERSION


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frozen',default='eval/boundary_supervised/data_fixed')
    ap.add_argument('--manifest',default='codetrace_bench/bench_manifest.verified.jsonl')
    ap.add_argument('--artifacts',default='codetrace_bench')
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    root=Path(args.out); root.mkdir(parents=True,exist_ok=False)
    frozen=Path(args.frozen)
    records=json.loads((frozen/'records.json').read_text(encoding='utf-8'))
    selected={r['traj_id']:r for r in records if r['source']=='SWE'}
    labels=json.loads((frozen/'labels.json').read_text(encoding='utf-8'))
    rules=Rules()
    started=time.perf_counter(); output=[]; arrays={}; failures=[]
    for e in iter_bench(args.manifest,args.artifacts,traj_ids=set(selected)):
        tid=e['traj_id']; old=selected[tid]
        try:
            if e.get('error'): raise ValueError(e['error'])
            steps,meta=load_trajectory(e['traj_path'],rules,base=e['base'],expect_steps=e['step_count'])
            if len(steps)>len(old['ids']):
                suffix=steps[len(old['ids']):]
                if meta['format']!='completions' or any(s.tool_class!='FINISH' for s in suffix):
                    raise ValueError('Unexplained suffix')
                steps=steps[:len(old['ids'])]
            if [s.idx for s in steps]!=old['ids']: raise ValueError('Step ID alignment changed')
            mapping=[dict(idx=s.idx,raw_ref=s.raw_ref) for s in steps]
            if mapping!=old['mapping']: raise ValueError('Native event mapping changed')
            encoder=StepEncoder(); base=BoundaryFeatures(4)
            key=f't{len(output):04d}'
            arrays[key]=np.stack([encoder.feed(s) for s in steps])
            rows=[dict(idx=s.idx,**base.feed(s)) for s in steps]
            source=Path(meta['source'])
            if not source.is_file(): raise ValueError('Expected SWE completion source file')
            output.append(dict(traj_id=tid,key=key,ids=old['ids'],fold=old['fold'],group=old['group'],
                task=old['task'],source='SWE',boundaries=labels[tid]['boundaries'],baseline_rows=rows,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),raw_source=str(source),
                repaired_step_sha256=hashlib.sha256(json.dumps([s.to_dict() for s in steps],sort_keys=True).encode()).hexdigest()))
        except Exception as exc:
            failures.append(dict(traj_id=tid,error=str(exc)))
        if (len(output)+len(failures))%25==0:
            print(f'prepared={len(output)} failed={len(failures)}',flush=True)
    missing=set(selected)-{r['traj_id'] for r in output}-{r['traj_id'] for r in failures}
    failures.extend(dict(traj_id=t,error='Missing from manifest') for t in sorted(missing))
    (root/'audit.json').write_text(json.dumps(dict(expected=len(selected),prepared=len(output),failures=failures,
        version=VERSION,seconds=time.perf_counter()-started,rules=rules.raw,
        frozen_records_sha256=hashlib.sha256((frozen/'records.json').read_bytes()).hexdigest(),
        alignment='Offline benchmark normalization; source identities checked; not native stream timing',
        scope='Previously inspected SWE development tasks, not pristine holdout'),ensure_ascii=False,indent=2),encoding='utf-8')
    if failures: raise RuntimeError('Preparation incomplete; inspect audit.json; no training dataset published')
    # Detect repaired-content duplicates crossing frozen folds, rather than silently regrouping.
    groups={}; contents={}
    for r in output:
        for table,key in ((groups,r['group']),(contents,r['repaired_step_sha256'])):
            if key in table and table[key]!=r['fold']: raise ValueError('Cross-fold identity collision')
            table[key]=r['fold']
    np.savez_compressed(root/'vectors.npz',**arrays)
    (root/'records.json').write_text(json.dumps(output,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(trajectories=len(output),steps=sum(len(r['ids']) for r in output),out=str(root))))


if __name__=='__main__':main()
