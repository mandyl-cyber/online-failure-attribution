"""Check real tokenizer page budgets, exact coverage and frozen v3 guard effects."""
import argparse
from collections import Counter
from pathlib import Path

from scripts.run_stage_claim_pipeline import read, save
from segmenter.claim_sft_backend import messages_for, partition_paged_requests, convert_partial
from segmenter.claim_evidence_review import review
from segmenter import stage_claim_summary as core


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-path',type=Path,default=Path('claim_sft_7b_reviewed_v2/adapter'))
    parser.add_argument('--returns',type=Path,default=Path('neural_ledger_acceptance_v3'))
    parser.add_argument('--out',type=Path,default=Path('out/neural_claim_v4_preflight'))
    args=parser.parse_args()
    from transformers import AutoTokenizer
    from remote_pilots.openhands_smoke.claim_sft_train import chat_ids
    tokenizer=AutoTokenizer.from_pretrained(str(args.tokenizer_path),local_files_only=True)
    reports=[]
    for session in ('swe','terminal','live_smoke'):
        for path in sorted((args.returns/session/'review_backend/extraction').glob('*/request.json')):
            request=read(path)
            request['_claim_policy']='v4'
            def count(child):
                return len(chat_ids(tokenizer,messages_for(child)[1],True))
            parts,gaps=partition_paged_requests(request,lambda child:count(child)+512<=8192)
            expected=list(messages_for(request)[0]['evidence_map'])
            seen=[]
            detail=[]
            for child in parts:
                row,_=messages_for(child)
                ids=child.get('_claim_page_ids',list(row['evidence_map']))
                seen.extend(ids)
                tokens=count(child)
                if tokens+512>8192:
                    raise ValueError('Page budget violation')
                detail.append(dict(stage=child['stage'],page_evidence_ids=child.get('_claim_page_ids'),input_tokens=tokens))
            uncovered=[i for gap in gaps for i in gap['evidence_ids']]
            if Counter(seen+uncovered)!=Counter(expected):
                raise ValueError('Missing or repeated evidence chunks')
            signals=review(dict(claims=[]),messages_for(request)[0])['observation_signals']
            replay=[]
            for raw in read(path.parent/'attempt_1.json')['raw_output']:
                child=dict(request,stage=raw['stage'])
                child['evidence']=[e for e in request['evidence'] if raw['stage']['start_step_id']<=e['source_step_id']<=raw['stage']['end_step_id']]
                row,_=messages_for(child)
                old,old_rejected=convert_partial(core.strict_json(raw['raw_output']),row)
                new,rejected=convert_partial(core.strict_json(raw['raw_output']),row,enforce_result_scope=True)
                replay.append(dict(previous_accepted=len(old['claims']),guarded_accepted=len(new['claims']),
                    previous_rejected=len(old_rejected),rejected=rejected))
            reports.append(dict(session=session,job=path.parent.name,parts=detail,gaps=gaps,
                signals=[dict(kind=s['kind'],source_step_id=s['source_step_id'],record_id=s['record_id'],
                              start=s['start'],end=s['end'],quote=s['quote']) for s in signals],frozen_guard_replay=replay))
    if len(reports)!=19:
        raise ValueError('Expected all 19 frozen stages')
    summary=dict(stages=len(reports),generation_parts=sum(len(r['parts']) for r in reports),
        paged_parts=sum(p['page_evidence_ids'] is not None for r in reports for p in r['parts']),
        unresolved_gaps=sum(len(r['gaps']) for r in reports),
        added_rejections=sum(p['previous_accepted']-p['guarded_accepted'] for r in reports for p in r['frozen_guard_replay']),
        signal_counts=dict(Counter(s['kind'] for r in reports for s in r['signals'])),
        scope='CPU actual-tokenizer preflight and replayed v3 responses only; new v4 model outputs and latency not measured.')
    save(args.out/'parts.json',reports)
    save(args.out/'summary.json',summary)
    print(summary)
    if summary['unresolved_gaps']:
        raise SystemExit('Unresolved minimum-page budget gaps; inspect parts.json before GPU execution.')


if __name__=='__main__':
    main()
