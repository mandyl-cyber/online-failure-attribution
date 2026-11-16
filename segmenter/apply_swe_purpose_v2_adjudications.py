"""Fill explicitly reviewed invalid v2 candidates, preserving model errors and AI provenance."""
import argparse
import hashlib
from pathlib import Path

from .purpose_pilot_remote_v2 import jobs, read, save, validate, verify_package


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--decisions',type=Path,required=True)
    a=p.parse_args()
    verify_package(a.package)
    run=read(a.run/'run_manifest.json')
    decisions=read(a.decisions)
    by_id={j['request_id']:j for j in jobs(a.package)}
    if set(decisions)-set(by_id): raise ValueError('Unknown adjudication request ID')
    for request_id,decision in decisions.items():
        job=by_id[request_id]
        destination=a.run/'responses'/(request_id+'.json')
        if destination.exists(): raise ValueError('Refusing to overwrite response: '+request_id)
        error_path=a.run/'errors'/(request_id+'.json')
        error=read(error_path)
        if error.get('status')!='failed' or not error.get('failures'):
            raise ValueError('No failed model attempts: '+request_id)
        if set(decision)!={'answer','review_note'} or not decision['review_note']:
            raise ValueError('Decision needs answer and review note')
        answer=validate(decision['answer'],job['packet'])
        save(destination,dict(request_id=request_id,input_sha256=job['input_sha256'],identity=run['identity'],
             answer=answer,label_source='codex_ai_adjudication',review_priority=True,
             review_note=decision['review_note'],failed_attempts_sha256=hashlib.sha256(error_path.read_bytes()).hexdigest(),
             usage=None,server_model=None,seconds=0,attempt=0))
        print(request_id,answer['purpose'])


if __name__=='__main__': main()
