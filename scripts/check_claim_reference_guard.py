"""Check evidence-reviewed targets against the current runtime claim validator."""
import json
from pathlib import Path
from segmenter import stage_claim_summary_v5 as core


def main():
    root=Path("claim_sft_teacher_repaired_v1")
    inputs={}
    for line in (root/"effective_inputs.jsonl").read_text(encoding="utf-8").splitlines():
        entry=json.loads(line)
        row=core.build(entry["row"]["source_snapshot"],entry["row"]["chunk_chars"])
        inputs[row["job_id"]]=row
    conflicts=[]
    passed=0
    for split in ("train","dev"):
        path=Path("pilot/claim_sft_reviewed_v2")/f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            record=json.loads(line)
            try:
                core.validate(json.loads(record["completion"][0]["content"]),inputs[record["job_id"]])
                passed+=1
            except (ValueError,KeyError,TypeError) as exc:
                conflicts.append(dict(job_id=record["job_id"],error=str(exc)))
    report=dict(passed=passed,conflicts=conflicts)
    path=Path("pilot/claim_sft_reviewed_v2/guard_consistency.json")
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(report)


if __name__=="__main__":
    main()
