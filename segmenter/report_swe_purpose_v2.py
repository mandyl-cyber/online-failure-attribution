"""Verify complete v2 candidate runs and summarize provenance without semantic scoring."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .prepare_purpose_round2 import write
from .purpose_pilot_remote_v2 import jobs, read, validate, verify_package


def summarize(package,run_dir):
    manifest=verify_package(package)
    run=read(run_dir/'run_manifest.json')
    missing=[]; invalid=[]; sources=Counter(); labels=Counter(); retries=0; output_tokens=0; seconds=0.0
    for job in jobs(package):
        path=run_dir/'responses'/(job['request_id']+'.json')
        if not path.exists(): missing.append(job['request_id']); continue
        try:
            r=read(path)
            if (r['request_id'],r['input_sha256'],r['identity'])!=(job['request_id'],job['input_sha256'],run['identity']):
                raise ValueError('Run/input identity mismatch')
            answer=validate(r['answer'],job['packet'])
            sources[r.get('label_source','qwen')]+=1
            labels[answer['purpose']]+=1
            retries+=max(0,r.get('attempt',1)-1)
            output_tokens+=(r.get('usage') or {}).get('completion_tokens',0)
            seconds+=r.get('seconds',0)
        except (ValueError,KeyError,TypeError) as e:
            invalid.append(dict(request_id=job['request_id'],error=str(e)))
    total=manifest['requests']
    return dict(total=total,valid=total-len(missing)-len(invalid),missing=missing,invalid=invalid,
                complete=not missing and not invalid,label_sources=dict(sources),purposes=dict(labels),
                successful_response_retries=retries,completion_tokens=output_tokens,
                model_response_seconds=round(seconds,2),semantic_accuracy_evaluated=False,
                run_identity=run['identity'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    result=summarize(a.package,a.run)
    if a.out: write(a.out,result)
    print(json.dumps({k:v if k not in ('missing','invalid') else len(v) for k,v in result.items()},ensure_ascii=False))
    if not result['complete']: raise SystemExit(2)


if __name__=='__main__': main()
