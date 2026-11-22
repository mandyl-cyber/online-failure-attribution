"""Token-only partition preflight and replay of per-claim validation."""
import argparse
from pathlib import Path

from scripts.run_stage_claim_pipeline import read,save
from segmenter.claim_sft_backend import messages_for,partition_requests,convert_partial
from segmenter import stage_claim_summary as core


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-path',type=Path,default=Path('claim_sft_7b_reviewed_v2/adapter'))
    parser.add_argument('--out',type=Path,default=Path('out/neural_claim_repair_preflight_v3'))
    args=parser.parse_args()
    from transformers import AutoTokenizer
    from remote_pilots.openhands_smoke.claim_sft_train import chat_ids
    tokenizer=AutoTokenizer.from_pretrained(str(args.tokenizer_path),local_files_only=True)
    reports=[]
    for session in ('swe','terminal','live_smoke'):
        directory=Path('neural_ledger_acceptance_v2')/session/'review_backend/extraction'
        for path in sorted(directory.glob('*/request.json')):
            request=read(path)
            def count(child):
                return len(chat_ids(tokenizer,messages_for(child)[1],True))
            parts,oversized=partition_requests(request,lambda child:count(child)+512<=8192)
            covered=[i for p in parts for i in range(p['stage']['start_step_id'],p['stage']['end_step_id']+1)]
            expected=list(range(request['stage']['start_step_id'],request['stage']['end_step_id']+1))
            if sorted(covered+oversized)!=expected or len(set(covered+oversized))!=len(expected):
                raise ValueError('Partition coverage error')
            expected_records=[r for r in request['evidence'] if r['source_step_id'] not in oversized]
            if [r for p in parts for r in p['evidence']]!=expected_records:
                raise ValueError('Partition evidence changed')
            attempt=read(path.parent/'attempt_1.json')
            raw=attempt.get('raw_output')
            replay=None
            if isinstance(raw,str):
                try:
                    converted,rejected=convert_partial(core.strict_json(raw),messages_for(request)[0])
                    replay=dict(accepted=len(converted['claims']),rejected=rejected)
                except (ValueError,KeyError,TypeError) as error:
                    replay=dict(error=str(error))
            reports.append(dict(session=session,job=path.parent.name,parent_input_tokens=count(request),
                parts=[dict(start=p['stage']['start_step_id'],end=p['stage']['end_step_id'],input_tokens=count(p)) for p in parts],
                oversized_steps=oversized,old_raw_claim_validation=replay))
    summary=dict(stages=len(reports),generation_parts=sum(len(r['parts']) for r in reports),
        oversized_single_steps=[dict(session=r['session'],job=r['job'],steps=r['oversized_steps']) for r in reports if r['oversized_steps']],
        formerly_oversized_stages=[r for r in reports if r['parent_input_tokens']+512>8192],
        note='CPU tokenizer and frozen-response validation only. New chunk summaries require GPU generation and semantic review.')
    save(args.out/'parts.json',reports)
    save(args.out/'summary.json',summary)
    print(summary)


if __name__=='__main__':
    main()
