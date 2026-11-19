"""Recheck saved completions under the current guard without new generation."""
import json
from pathlib import Path
from collections import Counter
from segmenter import stage_claim_summary as base
from segmenter import stage_claim_summary_v5 as core


def main():
    root=Path("claim_sft_teacher_repaired_v1")
    output=[]
    counts=Counter()
    for line in (root/"effective_inputs.jsonl").read_text(encoding="utf-8").splitlines():
        entry=json.loads(line)
        row=entry["row"]
        # Fact extraction has changed. Rebuild against immutable source evidence.
        current=core.build(row["source_snapshot"],row["chunk_chars"])
        result=json.loads((root/(row["job_id"]+".json")).read_text(encoding="utf-8"))
        item=dict(original_job_id=row["job_id"],original_status=result["status"],
                  original_error=result.get("error"),reviewed=False)
        try:
            raw=base.strict_json(result["raw_response"]["choices"][0]["message"]["content"])
            checked=core.validate_partial(raw,current)
            item["recheck"]=checked
            counts["retained_claims"]+=len(checked["claims"])
            counts["rejected_claims"]+=len(checked["rejected_claims"])
            counts["requests_with_rejections"]+=bool(checked["rejected_claims"])
            counts["requests_with_missing_critical"]+=bool(checked["missing_critical_fact_ids"])
        except (ValueError,KeyError,TypeError) as exc:
            item["error"]=str(exc)
            counts["unparseable"]+=1
        output.append(item)
    out=Path("pilot/claim_sft_repaired_return_v1")
    out.mkdir(parents=True,exist_ok=True)
    (out/"guard_recheck.json").write_text(json.dumps(dict(counts=dict(counts),details=output),
        ensure_ascii=False,indent=2),encoding="utf-8")
    print(dict(counts))


if __name__=="__main__":
    main()
