"""Two real native-stream smoke checks; no gold stages or label-based alignment."""
import argparse
import json
from pathlib import Path
import time

from .adapters.openhands import messages_to_events
from .ledger import atomic, sha
from .online_ledger import OnlineLedger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="eval/online_ledger_smoke_v1")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("Use a new output directory")
    selected = [json.loads(line) for line in Path("eval/ledger_protocol_v1/development_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    chosen = [next(row for row in selected if row["source"] == source) for source in ("SWE", "Terminal-Bench")]
    out.mkdir(parents=True)
    report = dict(status="running", mode="real_log_causal_replay_with_restart_not_live_agent",
                  gold_stages_used=False, offline_message_folding=False, API_calls=0, runs=[])
    for item in chosen:
        path = Path(item["raw_source"])
        blob = path.read_bytes()
        if sha(blob) != item["raw_sha256"]:
            raise ValueError("Raw source hash mismatch")
        if item["source"] == "SWE":
            raw = json.loads(blob.decode("utf-8"))
            messages = list(raw["messages"]) + [dict(raw["response"]["choices"][0]["message"], role="assistant")]
            events = messages_to_events(messages, fold_text_turns=False)
        else:
            events = [json.loads(line) for line in blob.decode("utf-8").splitlines() if line.strip()]
        directory = out/item["source"]
        runtime = OnlineLedger(directory, item["traj_id"])
        timings = []
        midpoint = len(events)//2
        restarted = False
        for pos, event in enumerate(events):
            if runtime.finished:
                break
            if pos == midpoint:
                runtime.close()
                runtime = OnlineLedger(directory, item["traj_id"])
                restarted = True
                # Source readers restart at byte zero; prior events must be idempotent.
                for old in events[:pos]:
                    assert runtime.ingest(old) is False
            started = time.perf_counter()
            runtime.ingest(event)
            timings.append((time.perf_counter()-started)*1000)
        runtime.drain()
        runtime.close()
        batches = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(runtime.results.glob("*.json"))]
        for batch in batches:
            refs = [ref for claim in batch["claims"] for ref in claim["evidence_ids"]]
            runtime.ledger.get_evidence(refs, batch["detected_at_seq"])
        claims = [claim for batch in batches for claim in batch["claims"]]
        assert len({c["entry_id"] for c in claims}) == len(claims)
        report["runs"].append(dict(source=item["source"], trajectory=item["traj_id"],
            events=len(runtime.seen), completed_steps=len(runtime.segmenter.steps),
            closed_stages=len(batches), claims=len(claims), restarted=restarted,
            terminal=runtime.terminal_kind, open_stages=sum(not s["sealed"] for s in runtime.segmenter.segmenter.snapshot()),
            median_ingest_ms=sorted(timings)[len(timings)//2], max_ingest_ms=max(timings),
            all_claim_evidence_resolved=True))
        atomic(out/"report.json", report)
        print(json.dumps(report["runs"][-1]), flush=True)
    report["status"] = "complete"
    report["limitations"] = ["only two logs", "rule extraction, no semantic claim-quality metric",
                              "timing includes local disk and background writer contention, not deployment SLO"]
    atomic(out/"report.json", report)


if __name__ == "__main__":
    main()
