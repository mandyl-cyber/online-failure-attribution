"""Fixed 16-case, three-partition structural probe plus remote extraction inputs."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from segmenter.adapters.openhands import load_trajectory
from segmenter.claim_extraction import LocalRuleBackend, build_request, validate_response, ExtractionError
from segmenter.ledger import encoded, sha
from segmenter.prepare_downstream import validate_ranges


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def ranges(ids,starts):
    starts=sorted({ids[0],*starts})
    result=[dict(stage_id=j+1,start_step_id=a,end_step_id=starts[j+1]-1 if j+1<len(starts) else ids[-1]) for j,a in enumerate(starts)]
    validate_ranges(result,ids)
    return result


def main():
    out=Path('pilot/swe_ledger_sensitivity_v1');out.mkdir(exist_ok=False)
    records={r['traj_id']:r for r in read('eval/boundary_supervised/data_fixed/records.json')}
    keys=read('pilot/swe_relation_balanced_v2/answer_key.json')
    labels=read('eval/boundary_supervised/data_fixed/labels.json')
    preds=read('pilot/swe_only_boundary_v1/predictions.json')['swe_train_swe_dev']
    chosen=sorted((k for k in keys if records[k['traj_id']]['n_steps']<=40),
                  key=lambda k:sha(k['traj_id'].encode()))[:3]
    assert len(chosen)==3
    remote_ids={k['traj_id'] for k in chosen}
    save(out/'protocol.json',dict(trajectories=16,selection='Frozen balanced-v2 trajectories; not representative or independent.',
        arms=['gold','structure','rule'],input_bytes_per_job=64000,request_bytes_per_trajectory=512000,
        local_backend='Copies up to 64 literal observation hints; no semantic inference.',
        remote_selection='Three SHA256-first tasks of length <=40 in the same fixed set; stage-only extraction.',
        remote_model='qwen-local',temperature=0,max_output_tokens=2048,attempts=1,
        clock='Retrospective partition sensitivity; source rule detected_at times not used. Not streaming latency.',
        no_claims=['semantic accuracy','dependency quality','failure attribution accuracy'],
        contract_sha256=sha(Path('segmenter/claim_extraction.py').read_bytes())))
    totals=defaultdict(Counter);details=[];requests=[];jobs=[];backend=LocalRuleBackend()
    for k in keys:
        tid=k['traj_id'];r=records[tid];path=Path(r['raw_source'])
        steps,_=load_trajectory(path.parent,base=r['ids'][0],expect_steps=r['n_steps'])
        steps=steps[:r['n_steps']];h=hashlib.sha256()
        for s in steps:h.update(json.dumps([s.action,s.observation],ensure_ascii=False).encode())
        assert h.hexdigest()==r['content_sha256'] and [s.idx for s in steps]==r['ids']
        ev={};hints={}
        for s in steps:
            ev[s.idx]=[dict(record_id=f'{tid}:{s.idx}:{field}',run_id=tid,source_step_id=s.idx,
                field=field,text=value,content_sha256=sha(value.encode()),availability_seq=s.idx)
                for field,value in [('action',s.action),('observation',s.observation)] if value]
            obs=next((e for e in ev[s.idx] if e['field']=='observation'),None)
            hints[s.idx]=dict(kind='conclusion',text=obs['text'][:160],epistemic_status='observed',evidence_ids=[obs['record_id']]) if obs else None
        partitions=dict(gold=labels[tid]['stages'],structure=ranges(r['ids'],[e['boundary_step_id'] for e in preds[tid]]),
                        rule=ranges(r['ids'],[e['boundary_step_id'] for e in r['rules_online']['phase']]))
        for arm,stages in partitions.items():
            validate_ranges(stages,r['ids']);spent=0;remote_spent=0;refs=[];texts=[];accepted=set();errors=Counter()
            stats=Counter(trajectories=1,stages=len(stages),observations=sum(bool(x) for x in hints.values()),
                          available_evidence=sum(map(len,ev.values())))
            for stage in stages:
                a,b=stage['start_step_id'],stage['end_step_id'];evidence=[x for i in range(a,b+1) for x in ev[i]]
                hs=[hints[i] for i in range(a,b+1) if hints[i]]
                job=dict(job_id=sha(f'{arm}:{tid}:{a}:{b}'.encode()),run_id=tid,stage=stage,evidence=evidence,
                         candidates=hs[:64],detected_at_seq=b,incomplete=False,task_context=None)
                try:
                    req=build_request(job,64000);n=len(encoded(req))
                    if spent+n>512000:raise ExtractionError('trajectory_budget_exceeded')
                    spent+=n;response=backend.extract(req);validate_response(response,req)
                    stats['ready_jobs']+=1;stats['claims']+=len(response['claims'])
                    accepted.update(e['record_id'] for e in evidence)
                    for c in response['claims']:
                        texts.append(c['text']);refs.extend(x['evidence_id'] for x in c['evidence_refs'])
                except ExtractionError as exc:errors[str(exc)]+=1;stats['failed_jobs']+=1
                if tid in remote_ids:
                    job['candidates']=[]
                    error=None
                    try:
                        req=build_request(job,64000);req['backend_mode']='remote_semantic_pilot'
                        n=len(encoded(req))
                        if remote_spent+n>512000:raise ExtractionError('trajectory_budget_exceeded')
                        remote_spent+=n
                        requests.append(dict(job_id=job['job_id'],request=req))
                    except ExtractionError as exc:error=str(exc)
                    jobs.append(dict(job_id=job['job_id'],traj_id=tid,arm=arm,stage=stage,preflight_error=error))
            stats.update(request_bytes=spent,accepted_evidence=len(accepted),cited_observations=len(set(refs)),
                         duplicate_refs=len(refs)-len(set(refs)),duplicate_texts=len(texts)-len(set(texts)))
            details.append(dict(traj_id=tid,arm=arm,counts=dict(stats),errors=dict(errors)))
            totals[arm].update(stats)
    save(out/'local_report.json',dict(complete=True,semantic_quality='not_evaluated',dependencies='not_evaluated',
        attribution='not_evaluated',network_calls=0,metrics={a:dict(c) for a,c in totals.items()},details=details))
    save(out/'remote_manifest.json',dict(tasks=sorted(remote_ids),jobs=jobs,request_count=len(requests),
        max_completion_tokens=len(requests)*2048,warning='No semantic/attribution results yet. Failed jobs remain visible.'))
    # Gold arm labels are only in manifest, never in model request.
    requests.sort(key=lambda x:x['job_id'])
    (out/'requests.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in requests),encoding='utf-8')
    print(json.dumps(dict(metrics={a:dict(c) for a,c in totals.items()},remote_requests=len(requests),remote_jobs=len(jobs)),indent=2))


if __name__=='__main__':main()
