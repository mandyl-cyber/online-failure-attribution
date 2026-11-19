"""Merge reviewed short stages and reviewed child projections without task leakage."""
import hashlib
import json
from pathlib import Path
from scripts.export_claim_sft_seed import main as export_seed
from scripts.review_long_claim_references import main as review_long
from segmenter import stage_claim_summary as core


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    export_seed()
    review_long()
    seed=Path("pilot/claim_sft_seed_v1")
    long=Path("pilot/claim_sft_long_review_v1")
    out=Path("pilot/claim_sft_reviewed_v2")
    records={split:read_rows(seed/f"{split}.jsonl") for split in ("train","dev")}
    for row in read_rows(long/"projected_examples.jsonl"):
        records[row["split"]].append(row)
    all_ids=[r["job_id"] for rows in records.values() for r in rows]
    parents={r["parent_stage_key"] for rows in records.values() for r in rows}
    if len(parents)!=78:
        raise ValueError("Parent-stage membership mismatch")
    if len(all_ids)!=len(set(all_ids)):
        raise ValueError("Duplicate reviewed request ID")
    groups={s:{r["group"] for r in rows} for s,rows in records.items()}
    if groups["train"]&groups["dev"]:
        raise ValueError("Task-group leakage")
    for split,rows in records.items():
        for row in rows:
            request=core.strict_json(row["prompt"][1]["content"])
            response=core.strict_json(row["completion"][0]["content"])
            ids={chunk["id"]:unit["step_id"] for unit in request["units"]
                 for field in ("action","observation") for parent in unit[field] for chunk in parent["chunks"]}
            if not response["claims"]:
                raise ValueError("Empty reviewed target")
            for claim in response["claims"]:
                if not claim["evidence_ids"] or not set(claim["evidence_ids"])<=ids.keys():
                    raise ValueError("Citation outside the projected input")
    out.mkdir(parents=True,exist_ok=True)
    for split,rows in records.items():
        (out/f"{split}.jsonl").write_bytes(b"".join(core.encoded(row)+b"\n" for row in rows))
    short_reviews=json.loads((seed/"ai_review.json").read_text(encoding="utf-8"))
    long_reviews=json.loads((long/"ai_review.json").read_text(encoding="utf-8"))
    summary=dict(reviewed_parent_stages=len(short_reviews)+len(long_reviews),
        short_stages=len(short_reviews),long_stages=len(long_reviews),
        examples={s:len(v) for s,v in records.items()},
        parents_by_split={s:len({r["parent_stage_key"] for r in rows}) for s,rows in records.items()},
        task_groups={s:len(v) for s,v in groups.items()},
        input_hashes={s:hashlib.sha256((out/f"{s}.jsonl").read_bytes()).hexdigest() for s in records},
        reviewer_type="AI",pending_selected_parent_reviews=0,
        scope="Key-fact references from supplied evidence; no independent human accuracy estimate",
        training_started=False,tokenizer_preflight="pending target tokenizer")
    (out/"protocol.json").write_bytes(core.encoded(summary)+b"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
