"""Versioned recovery of blank public plan actions using exact native call IDs."""
from copy import deepcopy
from pathlib import Path
from segmenter import stage_claim_summary as core
from remote_pilots.openhands_smoke import claims_v3 as io


def full_calls(obj,wanted):
    found={}; stack=[obj]
    while stack:
        value=stack.pop()
        if isinstance(value,dict):
            fn=value.get('function')
            if isinstance(value.get('id'),str) and value['id'] in wanted and isinstance(fn,dict) and fn.get('name') in ('task_tracker','task_tracking'):
                args=fn.get('arguments')
                if isinstance(args,str): args=core.strict_json(args)
                if not isinstance(args,dict): raise ValueError('invalid public arguments')
                if value['id'] in found and found[value['id']]!=args: raise ValueError('conflicting native call')
                found[value['id']]=args
            stack.extend(value.values())
        elif isinstance(value,list): stack.extend(value)
    return found


def main():
    root=Path('pilot/seal_claim_diagnostic_v1')
    out=Path('pilot/seal_claim_diagnostic_v3')
    records={r['traj_id']:r for r in io.read(Path('eval/boundary_supervised/data_fixed/records.json'))}
    sources={s['traj_id']:s for s in io.read(Path('eval/plan_payload_audit/report.json'))['sources']}
    rows=io.rows_at(root/'inputs.jsonl'); repaired=[]; changes=[]; identity={}
    for row in rows:
        core.verify_row(row)
        source=deepcopy(row['source_snapshot'])
        blank=[e for e in source['evidence'] if e['field']=='action' and not e['text'].strip()]
        if blank:
            record=records[row['run_id']]
            mapping={r['idx']:r['raw_ref']['action_id'] for r in record['mapping']}
            path=Path(record['raw_source']); blob=path.read_bytes()
            if core.digest(blob)!=sources[row['run_id']]['sha256']:
                raise ValueError('native source changed')
            calls=full_calls(core.strict_json(blob.decode('utf-8')), {mapping[e['source_step_id']] for e in blank})
            for e in blank:
                call=mapping[e['source_step_id']]
                if call not in calls: raise ValueError('cannot recover exact public call')
                text=core.encoded(calls[call]).decode()
                changes.append(dict(old_job_id=row['job_id'],run_id=row['run_id'],step=e['source_step_id'],
                    native_call_id=call,source_sha256=core.digest(blob),old_sha256=e['content_sha256'],
                    new_sha256=core.digest(text.encode()),payload=calls[call]))
                e.update(text=text,content_sha256=core.digest(text.encode()))
        fixed=core.build(source,row['chunk_chars']); repaired.append(fixed); identity[row['job_id']]=fixed
    out.mkdir(parents=True,exist_ok=False)
    (out/'inputs.jsonl').write_bytes(b''.join(core.encoded(r)+b'\n' for r in repaired))
    requests=[]
    for r in io.rows_at(root/'requests.jsonl'):
        new=deepcopy(r); row=identity[r['job_id']]
        new.update(job_id=row['job_id'],input_sha256=row['input_sha256'])
        new['messages'][1]['content']=core.encoded(row['request']).decode()
        requests.append(new)
    (out/'requests.jsonl').write_bytes(b''.join(core.encoded(r)+b'\n' for r in requests))
    selection=io.read(root/'selection.json')
    for m in selection:
        fixed=identity[m['job_id']];m.update(job_id=fixed['job_id'],input_bytes=len(core.encoded(fixed['request'])))
        m['input_provenance']='v1 plus exact native public plan recovery; unchanged observations'
    io.save(out/'selection.json',selection)
    reviews=io.read(root/'review.json')
    for r in reviews:
        r.update(job_id=identity[r['job_id']]['job_id'],reviewed=False,reviewer_type=None,
                 input_complete=None,required_points=[],excluded_inferences=[],claim_judgments=[])
    io.save(out/'review.json',reviews)
    io.save(out/'repairs.json',changes)
    protocol=io.read(root/'protocol.json')
    protocol.update(status='repaired_inputs_prepared_not_run',version=3,
        input_sha256=core.digest((out/'inputs.jsonl').read_bytes()),
        requests_sha256=core.digest((out/'requests.jsonl').read_bytes()),
        repair_count=len(changes),repaired_stages=len({c['old_job_id'] for c in changes}),
        parent_requests_sha256=core.digest((root/'requests.jsonl').read_bytes()),
        repair_basis='exact native action ID; verified raw source hash; full public tool arguments including notes; no thought')
    io.save(out/'protocol.json',protocol)
    print(dict(repaired_actions=len(changes),repaired_stages=protocol['repaired_stages'],out=str(out)))


if __name__=='__main__': main()
