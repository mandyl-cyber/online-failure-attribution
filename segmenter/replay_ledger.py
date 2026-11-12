"""Replay the frozen development set into a new local ledger directory."""
import argparse
import json
from pathlib import Path

from .adapters.openhands import load_trajectory, convert_events, messages_to_events
from .rules import default_rules
from .audit_plan_payloads import find_calls
from .ledger import Ledger, atomic, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("eval/ledger_protocol_v1/development_manifest.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval/ledger_local_v1"))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Output exists; choose a fresh directory")
    chosen = [json.loads(s) for s in args.manifest.read_text(encoding="utf-8").splitlines()]
    records = {r["traj_id"]: r for r in json.loads(Path("eval/boundary_supervised/data_fixed/records.json").read_text(encoding="utf-8"))}
    gold = {r["traj_id"]: r for r in map(json.loads, Path("codetrace_bench/bench_manifest.verified.jsonl").read_text(encoding="utf-8").splitlines())}
    args.out.mkdir(parents=True)
    report = dict(status="running", trajectories=[], network_calls=0, input_version="public-plan-v2",
                  selection_sha256=sha(args.manifest.read_bytes()),
                  limitations=["offline oracle stage assignment", "no semantic extraction or inferred dependency edges",
                               "task text export not implemented", "single writer only", "not a live deployment benchmark"])
    for item in chosen:
        tid = item["traj_id"]
        r = records[tid]
        source = Path(r["raw_source"])
        blob = source.read_bytes()
        if sha(blob) != item["raw_sha256"]:
            raise ValueError("Raw source drift: " + tid)
        if r["format"] == "completions":
            raw = json.loads(blob.decode("utf-8"))
            messages = list(raw["messages"]) + [dict(raw["response"]["choices"][0]["message"], role="assistant")]
            steps, _ = convert_events(messages_to_events(messages, fold_text_turns=True), default_rules(), base=r["ids"][0])
            meta = {"format": "completions"}
        else:
            steps, meta = load_trajectory(source, base=r["ids"][0], expect_steps=r["n_steps"])
        if len(steps) > r["n_steps"]:
            if meta["format"] != "completions" or any(s.tool_class != "FINISH" for s in steps[r["n_steps"]:]):
                raise ValueError("Unexpected extra steps")
            steps = steps[:r["n_steps"]]
        if [s.idx for s in steps] != r["ids"]:
            raise ValueError("Native step alignment mismatch")
        legacy = b"".join(json.dumps([s.action, s.observation], ensure_ascii=False).encode() for s in steps)
        if sha(legacy) != r["content_sha256"]:
            raise ValueError("Legacy adapter content drift")
        wanted = {s.raw_ref.get("action_id") for s in steps if s.tool == "task_tracking"}
        if wanted and r["format"] != "completions":
            raise ValueError("Native plan extraction not yet implemented for this format")
        calls = find_calls(json.loads(blob.decode("utf-8")), wanted) if wanted else {}
        directory = args.out / sha(tid.encode())[:24]
        ledger = Ledger(directory, tid)
        # Retain an immutable raw source copy, outside the model-facing evidence API.
        (directory/"source.bin").write_bytes(blob)
        atomic(directory/"provenance.json", dict(source=str(source), sha256=sha(blob),
               raw_copy="source.bin", legacy_content_sha256=r["content_sha256"],
               raw_access="audit_only_may_contain_thought_or_future_records"))
        repaired = 0
        for step in steps:
            call_id = step.raw_ref.get("action_id")
            payload = calls.get(call_id) if step.tool == "task_tracking" else None
            ledger.feed(step, payload, call_id if payload is not None else None)
            repaired += payload is not None
        ledger.assign_stages(gold[tid]["stages"], origin="gold_oracle")
        ledger.export()
        # Verify recovery and every evidence pointer, including hashes and cutoff.
        recovered = Ledger(directory, tid)
        assert recovered.operations == ledger.operations
        assert recovered.get_evidence(list(recovered.evidence), len(steps)) == list(ledger.evidence.values())
        row = dict(traj_id=tid, directory=str(directory), steps=len(steps), repaired_plans=repaired,
                   entries=sum(len(op["entries"]) for op in ledger.steps.values()), evidence=len(ledger.evidence))
        report["trajectories"].append(row)
        atomic(args.out/"report.json", report)
        print(f'{len(report["trajectories"])}/{len(chosen)} steps={len(steps)} plans={repaired}', flush=True)
    report["status"] = "complete_local_prototype"
    report["totals"] = {key: sum(r[key] for r in report["trajectories"]) for key in ("steps", "repaired_plans", "entries", "evidence")}
    atomic(args.out/"report.json", report)
    print(json.dumps(report["totals"]))


if __name__ == "__main__":
    main()
