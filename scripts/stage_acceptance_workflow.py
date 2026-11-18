"""CPU workflow for frozen boundary predictions and stage-claim regression cases."""
import json
from collections import Counter
from pathlib import Path
from segmenter.eval_online import matched_pairs
from scripts import audit_stage_claim_facts_return as claims


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    records = read("eval/temporal_boundary_v1/data/records.json")
    predictions = read("temporal_boundary_train_v1/predictions.json")
    report = {}
    for mode, runs in predictions.items():
        for deadline in (0, 1, 2):
            counts = Counter()
            for record in records:
                gold = record["boundaries"]
                cuts = runs[record["traj_id"]]
                pairs = matched_pairs(cuts, gold, 1, deadline)
                used = {i for i, _ in pairs}
                hit = {gold[j] for _, j in pairs}
                counts.update(tp=len(pairs), fp=len(cuts)-len(pairs), fn=len(gold)-len(pairs))
                for i, cut in enumerate(cuts):
                    pos = cut["boundary_step_id"]
                    if i not in used and any(abs(pos-g) <= 1 for g in hit):
                        counts["extra_near_matched_gold"] += 1
                    if i not in used and all(abs(pos-g) > 1 for g in gold):
                        counts["far_from_gold"] += 1
                    counts["delayed_predictions"] += cut["detected_at_step_id"] > pos
            t, f, n = (counts[k] for k in ("tp","fp","fn"))
            report[f"{mode}_D{deadline}"] = dict(counts, precision=t/max(1,t+f),
                recall=t/max(1,t+n), micro_f1=2*t/max(1,2*t+f+n))
    claims.main()
    out = Path("out/stage_acceptance_workflow")
    out.mkdir(parents=True, exist_ok=True)
    result = dict(trajectories=len(records), boundary=report,
                  claim_regression=read(claims.OUT/"mechanical_summary.json"),
                  independent_claim_semantics="pending independent evidence review")
    (out/"report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for key, row in report.items():
        print(key, round(row["micro_f1"],4), row["fp"], row.get("extra_near_matched_gold",0),
              row.get("far_from_gold",0), row.get("delayed_predictions",0))


if __name__ == "__main__":
    main()
