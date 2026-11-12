"""Live OpenHands events -> causal stages -> durable jobs -> claim batches.

No gold spans, offline folding, model training or API. Single event producer.
Run `follow` on an append-only JSONL file or OpenHands events directory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import time

from .adapters.openhands import OpenHandsStreamAdapter
from .audit_plan_payloads import public_payload
from .ledger import Ledger, atomic, encoded, repair_step, sha
from .rules import default_rules
from .run_online import OnlineSegmenter, _iter_live_events


def generate_claim_batch(job):
    """Extractor seam: stage-only immutable evidence in, zero or more claims out.

    Rule candidates remain conservative assertions/observations, not semantic
    conclusions. An API extractor can replace this function later.
    """
    claims = []
    for candidate in job["candidates"]:
        claim = deepcopy(candidate)
        claim.update(stage_id=job["stage"]["stage_id"],
                     generated_from_job=job["job_id"],
                     available_after_seq=job["detected_at_seq"],
                     stage_incomplete=job["incomplete"],
                     node_type="claim", extractor="rules-v1")
        claims.append(claim)
    return dict(job_id=job["job_id"], input_sha256=sha(encoded(job)),
                stage=job["stage"], detected_at_seq=job["detected_at_seq"],
                incomplete=job["incomplete"], claims=claims, edges=[],
                status="ready", completed_at_unix=time.time())


def process_job(job_path, result_dir):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    target = Path(result_dir)/Path(job_path).name
    fingerprint = sha(encoded(job))
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        if previous["input_sha256"] != fingerprint:
            raise ValueError("Committed claim job conflicts with new input")
        return previous
    result = generate_claim_batch(job)
    # One atomic batch = all claim nodes from this stage are visible together.
    atomic(target, result)
    return result


class OnlineLedger:
    def __init__(self, directory, run_id, rules=None, background=True, claim_processor=None):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rules = rules or default_rules()
        self.run_id = run_id
        self.processor = claim_processor or process_job
        self.raw = self.root/"native_events"
        self.jobs = self.root/"stage_jobs"
        self.results = self.root/"claim_batches"
        for directory in (self.raw, self.jobs, self.results):
            directory.mkdir(exist_ok=True)
        config = dict(run_id=run_id, rules=self.rules.raw, mode="phase",
                      input_version="public-plan-v2", event_ids_required=True)
        if claim_processor is not None:
            config["claim_processor"] = claim_processor.identity
        config_path = self.root/"online.json"
        if config_path.exists():
            if json.loads(config_path.read_text(encoding="utf-8")) != config:
                raise ValueError("Online run/configuration mismatch")
        else:
            atomic(config_path, config)
        self.ledger = Ledger(self.root/"evidence_store", run_id, defer_claims=True)
        self.adapter = OpenHandsStreamAdapter(self.rules, fold_agent_messages=False)
        self.segmenter = OnlineSegmenter(self.rules, modes=["phase"])
        self.seen = {}
        self.actions = {}
        self.finished = False
        self.terminal_kind = None
        self.pool = None
        self.futures = []
        for path in sorted(self.raw.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if path.name != f"{len(self.seen)+1:08d}.json":
                raise ValueError("Noncontiguous event journal")
            self._consume(record)
        if background:
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="claim-writer")
            for path in sorted(self.jobs.glob("*.json")):
                self._submit(path)
        self._snapshot()

    def _submit(self, path):
        if self.pool is not None:
            self.futures.append(self.pool.submit(self.processor, path, self.results))

    def _queue(self, stage, detected_at_seq, incomplete=False):
        job_id = sha(encoded([self.run_id, "stage", stage["stage_id"]]))[:32]
        selected = [op for idx, op in self.ledger.steps.items()
                    if stage["start_step_id"] <= idx <= stage["end_step_id"]]
        job = dict(job_id=job_id, run_id=self.run_id, stage=deepcopy(stage),
                   detected_at_seq=detected_at_seq, incomplete=incomplete,
                   extractor="rules-v1", candidates=[c for op in selected for c in op["buffered_candidates"]],
                   evidence=[e for op in selected for e in op["evidence"]])
        job["task_context"] = ({"text": self.adapter.task, "source": "observed_user_message"}
                               if self.adapter.task else None)
        path = self.jobs/f'{stage["stage_id"]:08d}.json'
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != job:
                raise ValueError("Stage job changed after sealing")
        else:
            atomic(path, job)
            self._submit(path)

    def _consume(self, record):
        event = record["event"]
        key = encoded(event["id"]).decode()
        self.seen[key] = record["sha256"]
        if record["kind"] == "abort":
            self._finish(incomplete=True)
            return
        if "action" in event and "observation" not in event:
            self.actions[event["id"]] = event
        for step in self.adapter.feed(event):
            current = self.actions.get(step.raw_ref.get("action_id"), {})
            payload = public_payload(current.get("args") or {}) if step.tool == "task_tracking" else None
            fixed = repair_step(step, payload, step.raw_ref.get("action_id") if payload else None)
            self.ledger.feed(step, payload, step.raw_ref.get("action_id") if payload else None)
            self.segmenter.feed(fixed)
            for stage in self.segmenter.segmenter.stages:
                # Only newly sealed stages; old jobs retain their original confirmation time.
                path = self.jobs/f'{stage["stage_id"]:08d}.json'
                if not path.exists():
                    self._queue(stage, len(self.segmenter.steps))
            if step.tool == "finish":
                self._finish()

    def _finish(self, incomplete=False):
        if self.finished:
            return
        self.segmenter.finish()
        for stage in self.segmenter.segmenter.stages:
            if not (self.jobs/f'{stage["stage_id"]:08d}.json').exists():
                self._queue(stage, len(self.segmenter.steps), incomplete)
        self.finished = True
        self.terminal_kind = "interrupted" if incomplete else "finished"

    def _snapshot(self):
        atomic(self.root/"stages.json", dict(stages=self.segmenter.segmenter.snapshot(),
            observed_steps=len(self.segmenter.steps), terminal=self.terminal_kind,
            pending_action_ids=list(self.adapter._pending),
            task=self.adapter.task, task_source="observed_user_message"))

    def ingest(self, event):
        if not isinstance(event, dict) or not isinstance(event.get("id"), (str, int)):
            raise ValueError("Stable native event ID required")
        key, fingerprint = encoded(event["id"]).decode(), sha(encoded(event))
        if key in self.seen:
            if self.seen[key] != fingerprint:
                raise ValueError("Conflicting native event ID")
            return False
        if self.finished:
            raise ValueError("Run already closed")
        if "observation" in event and event.get("cause") is not None and event["cause"] not in self.actions:
            raise ValueError("Observation precedes its action; ordered source required")
        record = dict(kind="event", event=deepcopy(event), sha256=fingerprint)
        path = self.raw/f"{len(self.seen)+1:08d}.json"
        atomic(path, record)  # durable input first; recovery completes interrupted processing
        self._consume(record)
        self._snapshot()
        self.check_worker()
        return True

    def abort(self):
        """Explicit permanent interruption. Idle/pause must NOT call this."""
        if self.finished:
            return
        event = {"id": "ledger:explicit-abort"}
        key = encoded(event["id"]).decode()
        if key in self.seen:
            raise ValueError("Reserved abort event ID collision")
        record = dict(kind="abort", event=event, sha256=sha(encoded(event)))
        atomic(self.raw/f"{len(self.seen)+1:08d}.json", record)
        self._consume(record)
        self._snapshot()

    def check_worker(self):
        pending = []
        for future in self.futures:
            if future.done():
                future.result()
            else:
                pending.append(future)
        self.futures = pending

    def drain(self):
        if self.pool:
            for future in self.futures:
                future.result()
            self.futures.clear()
        else:
            for path in sorted(self.jobs.glob("*.json")):
                self.processor(path, self.results)

    def close(self):
        """Stop this reader without sealing the open stage or flushing pending actions."""
        if self.pool:
            self.pool.shutdown(wait=True)
        self.check_worker()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="append-only native OpenHands JSONL or events directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--poll", type=float, default=0.2)
    parser.add_argument("--idle", type=float, default=600)
    parser.add_argument("--abort", action="store_true", help="Explicitly close as incomplete after reading")
    parser.add_argument("--claim-extractor", choices=["rules", "dry-run"], default="rules")
    args = parser.parse_args()
    processor = None
    if args.claim_extractor == "dry-run":
        from .claim_extraction import StageProcessor
        processor = StageProcessor()
    runtime = OnlineLedger(args.out, args.run_id, claim_processor=processor)
    try:
        for event in _iter_live_events(Path(args.path), args.poll, args.idle):
            if runtime.ingest(event):
                print(json.dumps(dict(native_event_id=event["id"], steps=len(runtime.segmenter.steps),
                      sealed_stages=len(runtime.segmenter.segmenter.stages))), flush=True)
            if runtime.finished:
                break
        if args.abort:
            runtime.abort()
    except KeyboardInterrupt:
        print("Paused; open stage and pending actions preserved.", flush=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
