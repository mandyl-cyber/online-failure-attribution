"""Audit blank fields and source clipping before declaring semantic coverage."""
import json
from pathlib import Path
from collections import Counter


def main():
    root=Path("pilot/claim_sft_partition_semantics_v1")
    packets=json.loads((root/"parent_packets.json").read_text(encoding="utf-8"))
    details=[]
    for packet in packets:
        issues=[]
        for child in packet["children"]:
            for record in child["source_snapshot"]["evidence"]:
                text=record["text"]
                if not text.strip():
                    issues.append(dict(record_id=record["record_id"],step=record["source_step_id"],
                                       field=record["field"],type="blank_source_field"))
                if "<response clipped>" in text or "Due to the max output limit" in text:
                    issues.append(dict(record_id=record["record_id"],step=record["source_step_id"],
                                       field=record["field"],type="source_clipping_marker"))
        details.append(dict(parent_job_id=packet["parent_job_id"],issues=issues,
                            completeness="has_known_gaps" if issues else "no_detected_gap"))
    summary=dict(parents=len(details),with_known_gaps=sum(bool(x["issues"]) for x in details),
        no_detected_gap=sum(not x["issues"] for x in details),
        issue_counts=dict(Counter(i["type"] for x in details for i in x["issues"])),
        caveat="Absence of these markers is not proof of native-source completeness",
        details=details)
    (root/"input_completeness.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print({k:v for k,v in summary.items() if k!="details"})


if __name__=="__main__":
    main()
