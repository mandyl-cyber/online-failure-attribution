"""Atomically publish stage batches with observation signals and review status."""
from copy import deepcopy
import json
from pathlib import Path

from .ledger import atomic, encoded, sha
from . import stage_claim_summary as core
from .claim_evidence_review import VERSION, citation_issues, review


class ReviewedClaimProcessor:
    def __init__(self, processor=None):
        if processor is None:
            from .online_ledger import process_job
            processor = process_job
        self.processor = processor
        self.identity = sha(encoded(dict(version=VERSION,backend=getattr(processor,'identity','rules-v1'))))

    def __call__(self, job_path, result_dir):
        job_path = Path(job_path)
        job = json.loads(job_path.read_text(encoding='utf-8'))
        fingerprint = sha(encoded(job))
        target = Path(result_dir)/job_path.name
        if target.exists():
            previous = json.loads(target.read_text(encoding='utf-8'))
            if previous.get('input_sha256')!=fingerprint or previous.get('evidence_review_identity')!=self.identity:
                raise ValueError('Reviewed batch input or processor changed')
            return previous
        # Backend output and its recovery journal persist independently. Final
        # publication includes annotations and observations in one atomic batch.
        staging = Path(result_dir).parent/'review_backend'/'claim_batches'
        staging.mkdir(parents=True,exist_ok=True)
        batch = self.processor(job_path,staging)
        if not batch or batch.get('status')!='ready':
            return batch
        if batch.get('input_sha256')!=fingerprint:
            raise ValueError('Backend returned a different stage input')
        result = deepcopy(batch)
        row = core.build(job)
        signals = review(dict(claims=[]),row)['observation_signals']
        known = {e['record_id']:e for e in job['evidence']}
        checked = []
        for index,claim in enumerate(result['claims']):
            refs = claim.get('evidence_refs')
            if refs is None:
                refs = [dict(evidence_id=eid,start=0,end=len(known[eid]['text']),quote=known[eid]['text'])
                        for eid in claim.get('evidence_ids',[]) if known[eid]['text']]
            for ref in refs:
                ev = known[ref['evidence_id']]
                if type(ref['start']) is not int or type(ref['end']) is not int or not 0<=ref['start']<ref['end']<=len(ev['text']) or ev['text'][ref['start']:ref['end']]!=ref['quote']:
                    raise ValueError('Invalid backend evidence span')
            quoted = '\n'.join(r['quote'] for r in refs if known[r['evidence_id']]['field']=='observation')
            observed = claim.get('basis')=='tool_observation' or claim.get('epistemic_status')=='observed'
            issues = citation_issues(claim['text'],quoted) if observed else []
            if observed and not quoted:
                issues.append(dict(code='no_nonempty_observation_reference'))
            claim['evidence_review'] = dict(version=VERSION,issues=issues,
                status='needs_review' if issues else 'narrow_checks_passed')
            if issues:
                claim['semantic_status']='needs_review'
            key = claim.get('entry_id',claim.get('claim_id',f'{job["job_id"]}:c{index+1}'))
            checked.append(dict(claim_id=key,refs=refs))
        for signal in signals:
            signal.update(stage_id=job['stage']['stage_id'],generated_from_job=job['job_id'],
                          available_after_seq=job['detected_at_seq'])
            signal['citing_claim_ids'] = [c['claim_id'] for c in checked if any(
                r['evidence_id']==signal['record_id'] and r['start']<signal['end'] and r['end']>signal['start']
                for r in c['refs'])]
        result.update(evidence_review_identity=self.identity,observation_signals=signals,
                      uncited_signal_ids=[s['signal_id'] for s in signals if not s['citing_claim_ids']])
        atomic(target,result)
        return result
