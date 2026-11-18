"""Verify returned teacher candidates against reconstructed task-disjoint inputs."""
import json
from collections import Counter
from pathlib import Path
from remote_pilots.openhands_smoke.claim_sft_teacher import rows
from segmenter import stage_claim_summary as core
from segmenter import stage_claim_summary_v5 as v5


def main():
    root = Path("claim_sft_teacher_v1")
    inputs = rows(Path("pilot/swe_claim_sft_v1"))
    by_id = {r["job_id"]:(m,r) for m,r in inputs}
    preflight = json.loads((root/"preflight.json").read_text(encoding="utf-8"))
    if len(preflight)!=len(by_id) or {r["job_id"] for r in preflight}!=set(by_id):
        raise ValueError("Preflight membership mismatch")
    counts = Counter()
    details = []
    for check in preflight:
        meta,row = by_id[check["job_id"]]
        counts[f'{meta["split"]}:eligible={check["eligible"]}'] += 1
        if not check["eligible"]:
            continue
        result = json.loads((root/(row["job_id"]+".json")).read_text(encoding="utf-8"))
        if result["input_sha256"]!=row["input_sha256"] or result["group"]!=meta["group"]:
            raise ValueError("Result provenance mismatch")
        item = dict(job_id=row["job_id"],split=meta["split"],group=meta["group"],
                    status=result["status"],error=result.get("error"),reviewed=False)
        if result.get("raw_response"):
            raw = core.strict_json(result["raw_response"]["choices"][0]["message"]["content"])
            item["raw_claims"] = raw["claims"]
            try:
                validated = v5.validate_partial(raw,row)
                item["partial_check"] = validated
                counts["claims_retained_after_partial_check"] += len(validated["claims"])
                counts["claims_rejected_after_partial_check"] += len(validated["rejected_claims"])
                counts["requests_missing_critical"] += bool(validated["missing_critical_fact_ids"])
            except (ValueError,KeyError,TypeError) as exc:
                item["recheck_error"] = str(exc)
        counts[result["status"]] += 1
        details.append(item)
    out = Path("pilot/claim_sft_teacher_return_v1")
    out.mkdir(parents=True,exist_ok=True)
    (out/"audit.json").write_text(json.dumps(dict(counts=dict(counts),details=details,
        semantic_review="pending",scope="Provenance and validator audit"),ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(dict(counts),ensure_ascii=False))


if __name__=="__main__":
    main()
