"""Read-only census of local task metadata and available annotation schemas."""
import argparse
import hashlib
import json
import statistics
from collections import Counter,defaultdict
from pathlib import Path


def source(row):
    raw=str(row.get('source_relpath') or '').replace('\\','/')
    annotation=str(row.get('annotation_relpath') or '').replace('\\','/')
    if raw.startswith('swe_raw/') or annotation.startswith('merged_cleaned'):
        return 'SWE'
    if annotation.startswith('agent_failure_analysis/'):
        return 'Terminal-Bench'
    return 'unknown'


def describe(rows):
    stage_keys=Counter();incorrect_keys=Counter();annotated_step_keys=Counter();label_values=Counter()
    lengths=[];incorrect=set();unuseful=set();invalid=[]
    for r in rows:
        ids=[]
        for s in r['stages']:
            stage_keys.update(s.keys())
            span=list(range(s['start_step_id'],s['end_step_id']+1))
            ids+=span;lengths.append(len(span))
        if not ids or ids!=list(range(ids[0],ids[0]+r['step_count'])):invalid.append(r['traj_id'])
        for s in r['incorrect_stages']:
            incorrect_keys.update(s.keys())
            incorrect.update((r['traj_id'],x) for x in s.get('incorrect_step_ids',[]))
            unuseful.update((r['traj_id'],x) for x in s.get('unuseful_step_ids',[]))
            for x in s.get('steps',[]):
                if isinstance(x,dict):
                    annotated_step_keys.update(x.keys())
                    label_values.update(x.get('labels',[]))
    categories=defaultdict(list)
    for r in rows:categories[r['category']].append(r)
    return dict(trajectories=len(rows),unique_task_names=len({r['task_name'] for r in rows}),
        sources=dict(Counter(source(r) for r in rows)),agents=dict(Counter(r['agent'] for r in rows)),
        categories={c:dict(trajectories=len(rs),tasks=len({r['task_name'] for r in rs}),
                      sources=dict(Counter(source(r) for r in rs)),examples=list(dict.fromkeys(r['task_name'] for r in rs))[:3])
                    for c,rs in sorted(categories.items(),key=lambda x:-len(x[1]))},
        source_categories={s:dict(Counter(r['category'] for r in rows if source(r)==s)) for s in sorted({source(r) for r in rows})},
        difficulty=dict(Counter(r['difficulty'] for r in rows)),solved=dict(Counter(str(r['solved']) for r in rows)),
        steps=sum(r['step_count'] for r in rows),step_median=statistics.median(r['step_count'] for r in rows),
        stages=len(lengths),stage_length_median=statistics.median(lengths),
        stage_fields=dict(stage_keys),incorrect_stage_fields=dict(incorrect_keys),annotated_step_fields=dict(annotated_step_keys),
        step_label_values=dict(label_values),
        incorrect_steps=len(incorrect),unuseful_steps=len(unuseful),both_flags=len(incorrect&unuseful),
        stage_coverage_invalid=invalid,
        missing_raw_paths=sum(not r.get('source_relpath') for r in rows),
        existing_annotation_paths=sum((Path('codetrace_bench')/r['annotation_relpath']).exists() for r in rows),
        existing_archives=sum(bool(r.get('artifact_path')) and (Path('codetrace_bench')/r['artifact_path']).exists() for r in rows))


def longrca():
    datasets={};hashes={};examples={}
    for split in ('full','mini'):
        buckets=defaultdict(list);digests={};root_keys=Counter();history_keys=Counter();invalid=[];roles=Counter()
        for p in sorted((Path('longrca_bench/data')/split).rglob('*.json')):
            x=json.loads(p.read_text(encoding='utf-8'))
            root_keys.update(x.keys());h=x['history'];history_keys.update(k for step in h for k in step)
            ids=[s['step'] for s in h]
            if ids!=list(range(len(h))) or x['mistake_step'] not in ids:invalid.append(str(p))
            roles.update({s.get('name','') for s in h})
            source_name=p.parent.name
            buckets[source_name].append(dict(id=x['question_ID'],length=len(h),root=x['mistake_step'],reason=bool(x.get('mistake_reason'))))
            digests[x['question_ID']]=hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            if split=='full' and source_name not in examples:
                user=next((s for s in h if s.get('role')=='user'),h[0])
                examples[source_name]=dict(path=str(p),question_ID=x['question_ID'],task_excerpt=user.get('content','')[:2000],
                    root_step=x['mistake_step'],responsible_role=x['mistake_agent'],root_reason_excerpt=x.get('mistake_reason','')[:1000])
        datasets[split]=dict(trajectories=sum(map(len,buckets.values())),sources={s:dict(trajectories=len(rs),
            history_records=sum(r['length'] for r in rs),median_records=statistics.median(r['length'] for r in rs),
            min_records=min(r['length'] for r in rs),max_records=max(r['length'] for r in rs),
            root_rationale_present=sum(r['reason'] for r in rs)) for s,rs in buckets.items()},
            top_level_fields=dict(root_keys),history_fields=dict(history_keys),invalid_step_ids=invalid,
            workflow_role_trajectory_counts=dict(roles))
        hashes[split]=digests
    datasets['overlap']=dict(mini_ids_in_full=len(set(hashes['mini'])&set(hashes['full'])),
        differing_content=sum(hashes['full'].get(k)!=v for k,v in hashes['mini'].items()))
    datasets['examples']=examples
    return datasets


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='eval/dataset_task_audit')
    ap.add_argument('--refresh',action='store_true',help='Recompute derived reports; does not modify source data')
    a=ap.parse_args();out=Path(a.out)
    if out.exists() and not a.refresh:raise SystemExit('Output exists')
    out.mkdir(parents=True,exist_ok=a.refresh)
    manifests={};report={}
    for split in ('full','verified'):
        path=Path('codetrace_bench')/f'bench_manifest.{split}.jsonl'
        rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
        manifests[split]=rows;report[split]=describe(rows)
        report[split]['manifest_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    current=json.loads(Path('eval/boundary_supervised/data_fixed/records.json').read_text(encoding='utf-8'))
    tids={r['traj_id'] for r in current}
    report['current_461']=describe([r for r in manifests['verified'] if r['traj_id'] in tids])
    report['verified_openhands']=describe([r for r in manifests['verified'] if r['agent']=='OpenHands'])
    full={r['traj_id']:r for r in manifests['full']}
    report['overlap']=dict(verified_in_full=sum(r['traj_id'] in full for r in manifests['verified']),
        differing_stages=sum(r['stages']!=full[r['traj_id']]['stages'] for r in manifests['verified'] if r['traj_id'] in full))
    report['source_inference']='SWE from swe_raw or merged_cleaned annotation prefix; Terminal-Bench from agent_failure_analysis annotation prefix. Missing source paths do not imply unknown task source.'
    (out/'codetrace.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lr=longrca()
    (out/'longrca.json').write_text(json.dumps(lr,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:{z:v[z] for z in ('trajectories','sources','solved')} for k,v in report.items() if isinstance(v,dict) and 'trajectories' in v},indent=2))
    print(json.dumps({k:v['sources'] for k,v in lr.items() if k in ('full','mini')},indent=2))


if __name__=='__main__':main()
