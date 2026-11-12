"""Freeze development IDs and input provenance; never calls APIs or trains models."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(config_path, out):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ds = config["dataset"]
    records_path, manifest_path = Path(ds["records"]), Path(ds["manifest"])
    records = json.loads(records_path.read_text(encoding="utf-8"))
    unknown_sources = set(ds["sources"]) - {r["source"] for r in records}
    if unknown_sources:
        raise ValueError(f"Configured sources missing: {unknown_sources}; available: {sorted({r['source'] for r in records})}")
    manifest = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["traj_id"] in manifest:
            raise ValueError("Duplicate manifest trajectory")
        manifest[row["traj_id"]] = row
    if len({r["traj_id"] for r in records}) != len(records):
        raise ValueError("Duplicate eligible trajectory")
    selected, used_groups = [], set()
    n = ds["per_stratum"]
    for source in ds["sources"]:
        for solved in ds["solved_strata"]:
            pool = [r for r in records if r["source"] == source
                    and manifest[r["traj_id"]].get("solved") is solved]
            pool.sort(key=lambda r: (r["n_steps"], r["traj_id"]))
            if len(pool) < n:
                raise ValueError(f"Insufficient stratum: {source}/{solved}: {len(pool)}")
            for bucket in range(n):
                candidates = pool[len(pool)*bucket//n:len(pool)*(bucket+1)//n]
                def key(r):
                    return hashlib.sha256(f'{config["seed"]}|{r["group"]}|{r["traj_id"]}'.encode()).hexdigest()
                candidates.sort(key=key)
                chosen = next((r for r in candidates if r["group"] not in used_groups), None)
                if chosen is None:
                    raise ValueError(f"No unique task in bucket {source}/{solved}/{bucket}")
                used_groups.add(chosen["group"])
                gold = manifest[chosen["traj_id"]]
                ids = chosen["ids"]
                covered = [i for s in gold["stages"]
                           for i in range(s["start_step_id"], s["end_step_id"]+1)]
                if ids != covered or len(ids) != chosen["n_steps"]:
                    raise ValueError("Stage coverage mismatch: " + chosen["traj_id"])
                raw = Path(chosen["raw_source"])
                if not raw.is_file():
                    raise ValueError("Missing raw source: " + str(raw))
                selected.append(dict(traj_id=chosen["traj_id"], task_group=chosen["group"],
                    source=source, solved=solved, n_steps=chosen["n_steps"],
                    length_bucket=bucket, raw_source=str(raw), raw_sha256=digest(raw),
                    legacy_content_sha256=chosen["content_sha256"],
                    purpose="development_metadata_not_model_input"))
    if len(selected) != n*len(ds["sources"])*len(ds["solved_strata"]):
        raise ValueError("Unexpected sample count")
    plan_path = Path("eval/plan_payload_audit/report.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["counts"]["unrecovered_empty_steps"] != 0:
        raise ValueError("Unrecovered public plan steps")
    provenance = {str(p): digest(p) for p in [config_path, records_path, manifest_path, plan_path,
        Path("eval/plan_payload_audit/payloads.jsonl")]}
    report = dict(status="prepared_not_executed", trajectories=len(selected),
        task_groups=len(used_groups), seed=config["seed"], input_sha256=provenance,
        plan_audit_counts=plan["counts"],
        checks=["unique_trajectory_ids", "unique_selected_task_groups", "exact_stage_coverage",
                "raw_sources_exist_and_hashed", "four_equal_source_outcome_strata"],
        blockers=["public-plan-v2 runtime input not implemented", "ledger writer not implemented",
                  "API provider/model/prompt/cost cap/data release scope unset",
                  "task text evidence export and cross-dataset overlap audit pending"],
        network_calls=0, training_started=False,
        limitations="Previously inspected development data; coverage checks do not prove raw alignment or semantic fidelity.")
    out.mkdir(parents=True, exist_ok=False)
    (out/"development_manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in selected), encoding="utf-8")
    (out/"preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (out/"experiment.json").write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "trajectories": len(selected), "status": report["status"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("docs/ledger/experiment_v1.json"))
    parser.add_argument("--out", type=Path, default=Path("eval/ledger_protocol_v1"))
    args = parser.parse_args()
    prepare(args.config, args.out)


if __name__ == "__main__":
    main()
