"""Inspect stratified archive member names and possible annotation JSON, without extraction."""
import hashlib
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path
import zstandard
from .audit_dataset_tasks import source


def main():
    base=Path('codetrace_bench');out=Path('eval/dataset_task_audit');out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(s) for s in (base/'bench_manifest.full.jsonl').read_text(encoding='utf-8').splitlines()]
    buckets=defaultdict(list)
    for r in rows:
        p=base/(r.get('artifact_path') or '_missing_')
        if p.is_file():buckets[(source(r),r['agent'])].append(r)
    reports=[]
    for group,rs in sorted(buckets.items()):
        # Deterministic hash sampling: not chosen by annotation contents or outcome.
        chosen=sorted(rs,key=lambda r:hashlib.sha256(('41:'+r['traj_id']).encode()).hexdigest())[:2]
        for r in chosen:
            p=base/r['artifact_path'];members=0;candidates=[];structures=[];prefixes=set()
            with p.open('rb') as fh,zstandard.ZstdDecompressor().stream_reader(fh) as z,tarfile.open(fileobj=z,mode='r|') as tf:
                for m in tf:
                    if not m.isfile():continue
                    members+=1;name=m.name.replace('\\','/');filename=name.rsplit('/',1)[-1]
                    prefixes.add(name.split('/')[0])
                    if (not re.search(r'(annot|stage|label|split|ground.?truth|gold)',filename,re.I)
                        and not re.fullmatch(r'step_\d+\.jsonl',filename) and filename!='steps.json'):continue
                    if not name.endswith(('.json','.jsonl','.yaml','.yml')):continue
                    candidates.append(dict(name=name,bytes=m.size))
                    if name.endswith(('.json','.jsonl')) and m.size<=10_000_000:
                        f=tf.extractfile(m)
                        try:value=json.loads(f.read().decode('utf-8-sig'))
                        except (ValueError,UnicodeError):continue
                        stack=[value];keys=set();semantic=[]
                        while stack:
                            x=stack.pop()
                            if isinstance(x,dict):
                                keys.update(x.keys())
                                for k,v in x.items():
                                    if re.search(r'(stage|phase).*(type|label|name)|(type|label|name).*(stage|phase)',k,re.I):
                                        semantic.append(dict(field=k,value=str(v)[:250]))
                                stack.extend(x.values())
                            elif isinstance(x,list):stack.extend(x)
                        structures.append(dict(name=name,keys=sorted(keys),semantic_candidates=semantic[:20]))
            reports.append(dict(traj_id=r['traj_id'],source=group[0],agent=group[1],archive=str(p),
                annotation_relpath=r['annotation_relpath'],annotation_path_exists=(base/r['annotation_relpath']).exists(),
                members=members,top_level_prefixes=sorted(prefixes),candidate_files=candidates,json_structures=structures))
            print(f'{len(reports)} inspected: {group}, {members} files, {len(candidates)} annotation-name candidates',flush=True)
    report=dict(sample_method='two existing archives per source/agent stratum, SHA256(41:traj_id) ascending',
        sampled=len(reports),results=reports,
        limitation='Absence in sampled raw-artifact archives does not prove absence in every archive or upstream annotation repository; semantic key candidates require provenance verification.')
    (out/'archive_annotations.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
