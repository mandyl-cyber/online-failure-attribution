"""Audit parent-stage coverage and evidence preservation of returned partitions."""
import json
import argparse
from collections import Counter,defaultdict
from pathlib import Path
from remote_pilots.openhands_smoke.claim_sft_teacher import rows
from segmenter import stage_claim_summary_v5 as core


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("claim_sft_teacher_partition_v1"))
    parser.add_argument("--inputs",type=Path,default=Path("pilot/swe_claim_sft_v1"))
    parser.add_argument("--out",type=Path,default=Path("pilot/claim_sft_partition_return_v1"))
    args=parser.parse_args()
    root=args.root
    originals={r["job_id"]:(m,r) for m,r in rows(args.inputs)}
    aliases={m["parent_job_id"]:jid for jid,(m,r) in originals.items() if m.get("parent_job_id")}
    partitions={x["parent_job_id"]:x for x in read(root/"partitions.json")}
    children=defaultdict(list)
    counts=Counter()
    failures=[]
    total_seconds=0.
    for line in (root/"effective_inputs.jsonl").read_text(encoding="utf-8").splitlines():
        entry=json.loads(line)
        meta,row=entry["metadata"],entry["row"]
        core.verify_row(row)
        parent=meta.get("parent_job_id",row["job_id"])
        parent=aliases.get(parent,parent)
        expected=originals[parent][1]
        lookup={e["record_id"]:e for e in expected["source_snapshot"]["evidence"]}
        for evidence in row["source_snapshot"]["evidence"]:
            if lookup.get(evidence["record_id"])!=evidence:
                raise ValueError("Partition evidence drift")
        result=read(root/(row["job_id"]+".json"))
        if result["input_sha256"]!=row["input_sha256"] or result["group"]!=meta["group"]:
            raise ValueError("Result provenance drift")
        counts[result["status"]]+=1
        counts["claims"]+=len(result.get("claims",[]))
        total_seconds+=result.get("seconds",0)
        if result["status"]=="failed":
            failures.append(dict(job_id=row["job_id"],parent_job_id=parent,error=result.get("error")))
        children[parent].append((row,result))
    parents=[]
    for parent,(meta,original) in originals.items():
        found=children[parent]
        seen=[e["record_id"] for r,v in found for e in r["source_snapshot"]["evidence"]]
        if len(seen)!=len(set(seen)):
            raise ValueError("Duplicate partition coverage")
        expected={e["record_id"] for e in original["source_snapshot"]["evidence"]}
        missing=expected-set(seen)
        oversized=partitions.get(parent,{}).get("oversized_steps",[])
        allowed={e["record_id"] for e in original["source_snapshot"]["evidence"]
                 if e["source_step_id"] in {x["step_id"] for x in oversized}}
        if meta.get("known_input_gap"):
            allowed=expected
        if missing!=allowed:
            raise ValueError("Unexplained missing evidence")
        complete=not missing
        all_ready=complete and bool(found) and all(v["status"] in ("ready","ready_empty") for r,v in found)
        parents.append(dict(job_id=parent,split=meta["split"],complete_input=complete,
            all_parts_ready=all_ready,children=len(found),missing_records=len(missing),
            oversized_steps=[x["step_id"] for x in oversized]))
    report=dict(parents=len(parents),complete_input=sum(x["complete_input"] for x in parents),
        all_parts_ready=sum(x["all_parts_ready"] for x in parents),
        child_counts=dict(counts),request_seconds=round(total_seconds,3),
        failures=failures,parent_details=parents,
        semantic_review="pending; schema success does not establish correctness")
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    (out/"audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("failures","parent_details")}))


if __name__=="__main__":
    main()
