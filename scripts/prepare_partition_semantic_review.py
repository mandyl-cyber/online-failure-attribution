"""Prepare parent-stage review packets and evidence-grounded failure corrections."""
import json
from pathlib import Path
from collections import defaultdict
from segmenter import stage_claim_summary as core

ROOT=Path("claim_sft_teacher_partition_v1")
OUT=Path("pilot/claim_sft_partition_semantics_v1")
CORRECTIONS={
"ad9b":[("tool_observation","The tool acknowledged adding data[\"labels\"] = gt_proc to image_processing_sam.py.",[82]),
        ("agent_assertion","The task list marks final review done while explicitly stating tests cannot be run.",[83])],
"f835":[("tool_observation","The displayed test_apt.py source contains package expansion checks covering trivial inputs, version wildcards and package-name wildcards.",[48])],
"c62e":[("agent_assertion","The task list claims useAutocomplete grouping was fixed and unit tests pass except one described as unrelated; no test execution is shown in this window.",[33])],
"cb53":[("tool_observation","The inline UnsignedIntegerCoder probe exited with code 0. Its four printed dtype results were uint8, int8, uint8 and int8 respectively.",[17])],
"e123":[("tool_observation","The initial pytest attempt reports a SyntaxError while collecting test_bytes_param.py.",[23]),
        ("tool_observation","The test file was rewritten; the subsequent pytest output reports 2 passed and EXIT:0.",[25,26])],
"abb6":[("tool_observation","The reproduction script ran sylow_subgroup(2) for DihedralGroup(18) and DihedralGroup(50), printed order 4 for both, and exited with code 0.",[19])],
"a5e3":[("tool_observation","The sed command displayed lines 240 to 590 of links.ts.",[13])],
}


def main():
    audit=json.loads(Path("pilot/claim_sft_partition_return_v1/audit.json").read_text())
    ready={p["job_id"] for p in audit["parent_details"] if p["all_parts_ready"]}
    by_parent=defaultdict(list)
    corrections=[]
    for line in (ROOT/"effective_inputs.jsonl").read_text(encoding="utf-8").splitlines():
        entry=json.loads(line)
        row=entry["row"]
        result=json.loads((ROOT/(row["job_id"]+".json")).read_text(encoding="utf-8"))
        parent=entry["metadata"].get("parent_job_id",row["job_id"])
        by_parent[parent].append(dict(job_id=row["job_id"],source_snapshot=row["source_snapshot"],
            evidence_map=row["evidence_map"],candidate_claims=result.get("claims",[])))
        prefix=row["job_id"][:4]
        if prefix in CORRECTIONS:
            claims=[]
            for basis,text,steps in CORRECTIONS[prefix]:
                ids=[eid for eid,ref in row["evidence_map"].items()
                     if ref["source_step_id"] in steps and
                     ref["field"]==("action" if basis=="agent_assertion" else "observation")]
                claims.append(dict(text=text,source=basis,evidence_ids=ids))
            core.validate({"claims":claims},row["base_row"])
            corrections.append(dict(job_id=row["job_id"],reviewer_type="AI",
                reviewed=True,scope="Targeted failure correction, not exhaustive stage coverage",
                final_response={"claims":claims},original_error=result.get("error")))
    OUT.mkdir(parents=True,exist_ok=True)
    packets=[dict(parent_job_id=parent,children=children,reviewed=False,
        final_response=None,coverage_reviewed=False,reviewer_type=None)
        for parent,children in by_parent.items() if parent in ready]
    assert len(packets)==85 and len(corrections)==7
    (OUT/"parent_packets.json").write_bytes(core.encoded(packets)+b"\n")
    (OUT/"failure_corrections.json").write_bytes(core.encoded(corrections)+b"\n")
    print(dict(parent_packets=len(packets),reviewed_parents=0,targeted_failure_corrections=len(corrections)))


if __name__=="__main__":
    main()
