"""Build the remaining parent-stage packets from repaired source snapshots."""
import json
from pathlib import Path
from collections import defaultdict
from remote_pilots.openhands_smoke.claim_sft_teacher import rows
from scripts.export_claim_sft_seed import REFERENCES
from segmenter import stage_claim_summary as base


def main():
    root=Path("claim_sft_teacher_repaired_v1")
    originals={r["job_id"]:(m,r) for m,r in rows(Path("pilot/swe_claim_sft_repaired_v1"))}
    aliases={m["parent_job_id"]:j for j,(m,r) in originals.items() if m.get("parent_job_id")}
    audit=json.loads(Path("pilot/claim_sft_repaired_return_v1/audit.json").read_text(encoding="utf-8"))
    ready={r["job_id"] for r in audit["parent_details"] if r["all_parts_ready"]}
    children=defaultdict(list)
    for line in (root/"effective_inputs.jsonl").read_text(encoding="utf-8").splitlines():
        entry=json.loads(line)
        row=entry["row"]
        parent=entry["metadata"].get("parent_job_id",row["job_id"])
        parent=aliases.get(parent,parent)
        children[parent].append(dict(row=row,result=json.loads(
            (root/(row["job_id"]+".json")).read_text(encoding="utf-8"))))
    packets=[]
    for parent,(meta,row) in originals.items():
        if parent not in ready or any(c["row"]["job_id"][:8] in REFERENCES for c in children[parent]):
            continue
        packets.append(dict(parent_job_id=parent,metadata=meta,source=row["source_snapshot"],
                            children=children[parent]))
    assert len(packets)==37
    out=Path("pilot/claim_sft_remaining_v1")
    out.mkdir(parents=True,exist_ok=True)
    (out/"packets.json").write_bytes(base.encoded(packets)+b"\n")
    print(dict(parents=len(packets),steps=sum(p["source"]["stage"]["end_step_id"]-
          p["source"]["stage"]["start_step_id"]+1 for p in packets)))


if __name__=="__main__":
    main()
