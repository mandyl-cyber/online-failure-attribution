"""Prepare same-model input-role ablation on eight already reviewed stages."""
from pathlib import Path
from segmenter import stage_claim_summary as core
from segmenter.claim_source_input import render
from remote_pilots.openhands_smoke import seal_claim_diagnostic as runner
from remote_pilots.openhands_smoke import claims_v3 as io


def main():
    parent=runner.ROOT;out=Path('pilot/claim_source_input_v1')
    _,rows,_=runner.load(parent)
    reviews=[r for r in io.read(parent/'review.json') if r['reviewed'] is True]
    selected=[rows[r['job_id']] for r in reviews];ids={r['job_id'] for r in selected}
    requests=[]
    for row in selected:
        for variant,data in [('baseline',row['request']),('source_typed',render(row))]:
            requests.append(dict(job_id=row['job_id'],variant=variant,input_sha256=row['input_sha256'],
                messages=[dict(role='system',content=core.INSTRUCTION),dict(role='user',content=core.encoded(data).decode())]))
    out.mkdir(parents=True,exist_ok=False)
    for name,values in [('inputs.jsonl',selected),('requests.jsonl',requests)]:
        (out/name).write_bytes(b''.join(core.encoded(r)+b'\n' for r in values))
    io.save(out/'review.json',reviews)
    io.save(out/'selection.json',[m for m in io.read(parent/'selection.json') if m['job_id'] in ids])
    io.save(out/'repairs.json',io.read(parent/'repairs.json'))
    io.save(out/'protocol.json',dict(status='prepared_not_run',stages=8,requests=16,
        variants=['baseline','source_typed'],input_transform='source_roles_v1',
        input_sha256=core.digest((out/'inputs.jsonl').read_bytes()),
        requests_sha256=core.digest((out/'requests.jsonl').read_bytes()),
        instruction_sha256={v:core.digest(core.INSTRUCTION.encode()) for v in ['baseline','source_typed']},
        limits=dict(max_output_tokens=2048,temperature=0,no_silent_truncation=True),
        changed='Input source-role annotations and explicit interpretation policy only.',
        unchanged=['all original evidence text/IDs','task background','system instruction','output schema','reference points'],
        caveats=['Seen short-stage diagnostic; designed after inspecting failures, not independent validation.',
                 'Annotations are format heuristics; they do not prove claim truth.',
                 'Input-side instructions and source hints are a combined representation change, not isolated tagging effect.',
                 'Baseline must rerun on same service; no comparison of different model sizes.']))
    (out/'.gitattributes').write_text('*.json text eol=lf\n*.jsonl text eol=lf\n',encoding='utf-8')
    runner.load(out)
    print('Prepared 8 paired stages at',out)


if __name__=='__main__': main()
