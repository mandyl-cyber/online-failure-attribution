"""Collect two blind AI passes into a pending review table; never auto-approve."""
import argparse
from pathlib import Path
from .prepare_purpose_round2 import read, write
from .purpose_pilot_remote import jobs, validate, verify_package


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--first',type=Path,required=True)
    p.add_argument('--second',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit('Refusing to overwrite review')
    if a.first.resolve()==a.second.resolve():
        raise ValueError('Need separate pass directories')
    verify_package(a.package)
    manifests = [read(root/'run_manifest.json') for root in (a.first,a.second)]
    entries = {}
    for job in jobs(a.package):
        answers, errors = [], []
        for root,manifest in zip((a.first,a.second),manifests):
            try:
                r = read(root/'responses'/f"{job['request_id']}.json")
                if r['identity']!=manifest['identity'] or r['input_sha256']!=job['input_sha256'] or r['request_id']!=job['request_id']:
                    raise ValueError('Identity mismatch')
                answers.append(validate(r['answer'],job['packet']))
                errors.append(None)
            except (OSError,ValueError,KeyError,TypeError) as e:
                answers.append(None)
                errors.append(str(e))
        agree = all(answers) and answers[0]['purpose']==answers[1]['purpose']
        entries[job['request_id']] = dict(input_sha256=job['input_sha256'],candidates=answers,
              errors=errors,agreement=bool(agree),final_label=answers[0]['purpose'] if agree else None,
              reviewed=False,review_note='')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    write(a.out,dict(reviewer_kind='ai',reviewer_id=None,
          note='Agreement is not semantic validation. All entries require review; identical models/prompts are correlated, not independent annotators.',
          runs=manifests,labels=entries))


if __name__=='__main__':
    main()
