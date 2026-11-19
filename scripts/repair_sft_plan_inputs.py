"""Recover empty plan actions from native call IDs into a versioned SFT input pack."""
from copy import deepcopy
import json
from pathlib import Path
from remote_pilots.openhands_smoke.repair_seal_claim_inputs import full_calls
from segmenter import stage_claim_summary as core


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    root=Path("pilot/swe_claim_sft_v1")
    out=Path("pilot/swe_claim_sft_repaired_v1")
    records={r["traj_id"]:r for r in load(Path("eval/boundary_supervised/data_fixed/records.json"))}
    audited={s["traj_id"]:s for s in load(Path("eval/plan_payload_audit/report.json"))["sources"]}
    protocol=load(root/"protocol.json")
    manifest=load(root/"manifest.json")
    repaired={}
    changes=[]
    unresolved=[]
    for split in ("train","dev"):
        for line in (root/f"{split}_inputs.jsonl").read_text(encoding="utf-8").splitlines():
            row=core.strict_json(line)
            core.verify_row(row)
            source=deepcopy(row["source_snapshot"])
            blanks=[e for e in source["evidence"] if e["field"]=="action" and not e["text"].strip()]
            if blanks:
                try:
                    record=records[row["run_id"]]
                    blob=Path(record["raw_source"]).read_bytes()
                    fingerprint=core.digest(blob)
                    if fingerprint!=audited[row["run_id"]]["sha256"]:
                        raise ValueError("Native source hash changed")
                    mapping={r["idx"]:r["raw_ref"]["action_id"] for r in record["mapping"]}
                    calls=full_calls(core.strict_json(blob.decode("utf-8")),
                                     {mapping[e["source_step_id"]] for e in blanks})
                    for evidence in blanks:
                        call_id=mapping[evidence["source_step_id"]]
                        if call_id not in calls:
                            raise ValueError("Native public plan call unavailable")
                        text=core.encoded(calls[call_id]).decode("utf-8")
                        evidence.update(text=text,content_sha256=core.digest(text.encode("utf-8")))
                        changes.append(dict(old_job_id=row["job_id"],step=evidence["source_step_id"],
                                            native_call_id=call_id,source_sha256=fingerprint))
                except (KeyError,OSError,ValueError) as exc:
                    unresolved.append(dict(job_id=row["job_id"],error=str(exc)))
            repaired[row["job_id"]]=core.build(source,row["chunk_chars"])
    out.mkdir(parents=True,exist_ok=True)
    new_manifest=[]
    for item in manifest:
        if item["split"] not in ("train","dev"):
            continue
        updated=deepcopy(item)
        row=repaired[item["job_id"]]
        updated.update(job_id=row["job_id"],input_bytes=len(core.encoded(row["request"])),
                       parent_job_id=item["job_id"],
                       known_input_gap=any(not e["text"].strip() for e in row["source_snapshot"]["evidence"]))
        new_manifest.append(updated)
    for split in ("train","dev"):
        old_ids=[m["job_id"] for m in manifest if m["split"]==split]
        (out/f"{split}_inputs.jsonl").write_bytes(b"".join(core.encoded(repaired[j])+b"\n" for j in old_ids))
    (out/"manifest.json").write_bytes(core.encoded(new_manifest)+b"\n")
    protocol.update(manifest_sha256=core.digest(core.encoded(new_manifest)),
        input_sha256=core.digest(core.encoded([repaired[m["job_id"]] for m in manifest if m["split"] in ("train","dev")])),
        status="repaired_source_requires_new_generation_and_review")
    (out/"protocol.json").write_bytes(core.encoded(protocol)+b"\n")
    report=dict(repaired_actions=len(changes),repaired_stages=len({x["old_job_id"] for x in changes}),
                unresolved=unresolved,changes=changes)
    (out/"repairs.json").write_bytes(core.encoded(report)+b"\n")
    print(dict(repaired_actions=report["repaired_actions"],repaired_stages=report["repaired_stages"],
               unresolved=len(unresolved)))


if __name__=="__main__":
    main()
