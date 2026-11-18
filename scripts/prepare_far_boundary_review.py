"""Freeze a deterministic sample of far false cuts with causal and later evidence."""
import json
import random
from pathlib import Path
from segmenter.eval_online import matched_pairs


def main():
    records = json.loads(Path("eval/temporal_boundary_v1/data/records.json").read_text(encoding="utf-8"))
    predictions = json.loads(Path("temporal_boundary_train_v1/predictions.json").read_text(encoding="utf-8"))
    cases = []
    for rec in records:
        cuts = predictions["immediate"][rec["traj_id"]]
        paired = {i for i, _ in matched_pairs(cuts, rec["boundaries"], 1, 0)}
        for i, cut in enumerate(cuts):
            step = cut["boundary_step_id"]
            if i in paired or any(abs(step-g) <= 1 for g in rec["boundaries"]):
                continue
            position = rec["ids"].index(step)
            cases.append(dict(traj_id=rec["traj_id"], task=rec["task"], step=step,
                causal_steps=rec["baseline_rows"][max(0,position-3):position+1],
                later_for_retrospective_review=rec["baseline_rows"][position+1:position+4],
                gold_boundaries=rec["boundaries"],
                review=dict(reviewed=False, prefix_supports_boundary=None,
                            retrospective_supports_boundary=None, reason=None)))
    sample = random.Random(20261119).sample(cases, min(40,len(cases)))
    out = Path("pilot/far_boundary_review_v1")
    out.mkdir(parents=True,exist_ok=True)
    target = out/"packets.json"
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        expected = [(r["traj_id"],r["step"]) for r in sample]
        if [(r["traj_id"],r["step"]) for r in existing] != expected:
            raise ValueError("Existing review selection differs")
    else:
        target.write_text(json.dumps(sample,ensure_ascii=False,indent=2),encoding="utf-8")
    print(dict(population=len(cases),sample=len(sample),reviewed=0))


if __name__ == "__main__":
    main()
