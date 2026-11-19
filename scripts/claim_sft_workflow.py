"""Run local reference export and dataset gates, then emit a reproducible status."""
import hashlib
import json
from pathlib import Path
from scripts.export_claim_sft_seed import main as export


def main():
    export()
    root=Path("pilot/claim_sft_seed_v1")
    rows={split:[json.loads(line) for line in (root/f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
          for split in ("train","dev")}
    groups={split:{r["group"] for r in records} for split,records in rows.items()}
    if groups["train"]&groups["dev"]:
        raise ValueError("Task split overlap")
    for split,records in rows.items():
        ids=[r["job_id"] for r in records]
        if len(ids)!=len(set(ids)):
            raise ValueError("Duplicate training IDs")
        for row in records:
            if row["reviewer_type"]!="AI":
                raise ValueError("Unexpected reviewer provenance")
            response=json.loads(row["completion"][0]["content"])
            request=json.loads(row["prompt"][1]["content"])
            allowed={chunk["id"] for unit in request["units"]
                     for field in ("action","observation") for parent in unit[field] for chunk in parent["chunks"]}
            for claim in response["claims"]:
                if not claim["evidence_ids"] or not set(claim["evidence_ids"])<=allowed:
                    raise ValueError("Reference citation drift")
    summary=json.loads((root/"summary.json").read_text())
    manifest=dict(counts=summary["counts"],task_groups={s:len(v) for s,v in groups.items()},
        hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                for name in ("train.jsonl","dev.jsonl","ai_review.json")},
        local_gates="passed",pending_parent_review=summary["pending_parent_review"],
        tokenizer_preflight="pending target-model tokenizer",gpu_training="pending",
        paired_generation="pending",semantic_acceptance="pending",
        reviewer_type="AI",selection="short-stage evidence-reviewed subset")
    (root/"workflow_status.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
