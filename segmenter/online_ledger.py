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
from .input_gate import stage_check, dispatch, blocked
from .input_timeline import InputTimeline, validate_event


def generate_claim_batch(job):
    """Extractor seam: stage-only immutable evidence in, zero or more claims out.

    Rule candidates remain conservative assertions/observations, not semantic
    conclusions. An API extractor can replace this function later.
    """
    if blocked(job):
        raise ValueError('blocked_input: known incomplete stage')
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
    def __init__(self, directory, run_id, rules=None, background=True, claim_processor=None, boundary_model=None, input_mode='live'):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rules = rules or default_rules()
        self.run_id = run_id
        self.timeline = InputTimeline(self.root/'timing', input_mode)
        self.recovering = True
        self.processor = claim_processor or process_job
        self.raw = self.root/"native_events"
        self.jobs = self.root/"stage_jobs"
        self.results = self.root/"claim_batches"
        for directory in (self.raw, self.jobs, self.results):
            directory.mkdir(exist_ok=True)
        config = dict(run_id=run_id, rules=self.rules.raw, mode="phase",
                      input_version="event-integrity-v4-timing", event_ids_required=True, input_mode=input_mode)
        self.boundary = None
        self.boundary_events = []
        if boundary_model is not None:
            model_path = Path(boundary_model).resolve(strict=True)
            from .boundary_model import OnlineBoundaryPredictor
            self.boundary = OnlineBoundaryPredictor.load(model_path)
            config.update(mode="supervised", boundary_model=str(model_path),
                          boundary_sha256=sha(model_path.read_bytes()),
                          boundary_input="adapter-v2-repaired-plan", boundary_feature_version=self.boundary.model.feature_version)
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
        self.input_checks = {}
        self.finished = False
        self.terminal_kind = None
        self.pool = None
        self.futures = []
        for path in sorted(self.raw.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if path.name != f"{len(self.seen)+1:08d}.json":
                raise ValueError("Noncontiguous event journal")
            self._consume(record)
        self.recovering = False
        if background:
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="claim-writer")
            for path in sorted(self.jobs.glob("*.json")):
                self._submit(path)
        self._snapshot()

    def _submit(self, path):
        if self.pool is not None:
            self.futures.append(self.pool.submit(self._process_timed, path))

    def _process_timed(self, path):
        if (self.results/path.name).exists():
            result = dispatch(self.processor, path, self.results)
            self.timeline.emit('claim_reused', job=path.name)
            return result
        queued_path = self.timeline.root/('queue-'+path.name)
        queued = json.loads(queued_path.read_text(encoding='utf-8')) if queued_path.exists() else None
        start = self.timeline.emit('claim_started', job=path.name)
        self.timeline.emit('claim_queue_wait', job=path.name, queue_ms=self.timeline.elapsed(queued, start))
        try:
            result = dispatch(self.processor, path, self.results)
            self.timeline.emit('claim_completed', job=path.name,
                processing_ms=self.timeline.elapsed(start, self.timeline.stamp()))
            return result
        except Exception as exc:
            self.timeline.emit('claim_failed', job=path.name, error=type(exc).__name__)
            raise

    def _queue(self, stage, detected_at_seq, incomplete=False):
        job_id = sha(encoded([self.run_id, "stage", stage["stage_id"]]))[:32]
        selected = [op for idx, op in self.ledger.steps.items()
                    if stage["start_step_id"] <= idx <= stage["end_step_id"]]
        job = dict(job_id=job_id, run_id=self.run_id, stage=deepcopy(stage),
                   detected_at_seq=detected_at_seq, incomplete=incomplete,
                   extractor="rules-v1", candidates=[c for op in selected for c in op["buffered_candidates"]],
                   evidence=[e for op in selected for e in op["evidence"]])
        job['input_integrity']=stage_check(stage,self.input_checks,incomplete)
        job["task_context"] = ({"text": self.adapter.task, "source": "observed_user_message"}
                               if self.adapter.task else None)
        path = self.jobs/f'{stage["stage_id"]:08d}.json'
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != job:
                raise ValueError("Stage job changed after sealing")
        else:
            atomic(path, job)
            queued = self.timeline.emit('stage_sealed_claim_enqueued', job=path.name,
                               recovery=self.recovering, detected_at_step=detected_at_seq)
            atomic(self.timeline.root/('queue-'+path.name), queued)
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
        validate_event(self.adapter, event)
        self.timeline.receive(event, recovery=self.recovering)
        for step in self.adapter.feed(event):
            self.timeline.ready(step, recovery=self.recovering)
            self.input_checks[step.idx]=deepcopy(step.attrs.get('input_integrity'))
            # Adapter v2 already preserves full public arguments, including notes.
            payload = None
            fixed = repair_step(step, payload, step.raw_ref.get("action_id") if payload else None)
            self.ledger.feed(step, payload, step.raw_ref.get("action_id") if payload else None)
            if self.boundary is not None:
                # Legacy models include action_len and possibly text: do not feed v2 repair.
                prediction = self.boundary.feed(step)
                self.boundary_events.append({k: v for k, v in prediction.items() if k != "emitted_at"})
            self.segmenter.feed(fixed)
            for stage in self._selected_stages():
                if not stage["sealed"]:
                    continue
                # Only newly sealed stages; old jobs retain their original confirmation time.
                path = self.jobs/f'{stage["stage_id"]:08d}.json'
                if not path.exists():
                    self._queue(stage, len(self.segmenter.steps))
            if step.tool == "finish":
                self._finish()
            self._snapshot()
            self.timeline.published(step.idx, recovery=self.recovering)

    def _finish(self, incomplete=False):
        if self.finished:
            return
        if self.boundary is not None and hasattr(self.boundary, 'finish'):
            for event in self.boundary.finish(incomplete=incomplete):
                self.boundary_events.append(dict(event, status='committed'))
        self.segmenter.finish()
        for stage in self._selected_stages(final=True):
            if not (self.jobs/f'{stage["stage_id"]:08d}.json').exists():
                self._queue(stage, len(self.segmenter.steps), incomplete)
        self.finished = True
        self.terminal_kind = "interrupted" if incomplete else "finished"

    def _selected_stages(self, final=False):
        if self.boundary is not None:
            return self.boundary.snapshot(final=final or self.finished)
        return self.segmenter.segmenter.snapshot()

    def _snapshot(self):
        atomic(self.root/"stages.json", dict(stages=self._selected_stages(),
            boundary_mode="supervised" if self.boundary is not None else "phase",
            boundary_events=self.boundary_events,
            observed_steps=len(self.segmenter.steps), terminal=self.terminal_kind,
            pending_action_ids=list(self.adapter._pending),
            task=self.adapter.task, task_source="observed_user_message"))
        self.timeline.emit('backlog', recovery=self.recovering,
            pending_actions=len(self.adapter._pending),
            uncommitted_claim_jobs=sum(not (self.results/p.name).exists() for p in self.jobs.glob('*.json')))

    def ingest(self, event):
        if not isinstance(event, dict) or not isinstance(event.get("id"), (str, int)):
            raise ValueError("Stable native event ID required")
        key, fingerprint = encoded(event["id"]).decode(), sha(encoded(event))
        if key in self.seen:
            if self.seen[key] != fingerprint:
                raise ValueError("Conflicting native event ID")
            return False
        lifecycle_only=('action' not in event and event.get('observation') in ('agent_state_changed','null','recall'))
        if self.finished and not lifecycle_only:
            raise ValueError("Run already closed")
        validate_event(self.adapter, event)
        self.timeline.receive(event)
        if not lifecycle_only and "observation" in event and event.get("cause") is not None and event["cause"] not in self.actions:
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
                self._process_timed(path)
        self._snapshot()

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
    parser.add_argument('--input-mode', choices=['live', 'replay'], default='live')
    parser.add_argument("--abort", action="store_true", help="Explicitly close as incomplete after reading")
    parser.add_argument("--claim-extractor", choices=["rules", "dry-run"], default="rules")
    parser.add_argument('--claim-evidence-review', action='store_true', help='Publish exact failure signals and flag unsupported numeric citations')
    parser.add_argument("--boundary-model", help="Trusted local joblib boundary model; keeps legacy model input")
    args = parser.parse_args()
    processor = None
    if args.claim_extractor == "dry-run":
        from .claim_extraction import StageProcessor
        processor = StageProcessor()
    if args.claim_evidence_review:
        from .reviewed_claim_processor import ReviewedClaimProcessor
        processor = ReviewedClaimProcessor(processor)
    runtime = OnlineLedger(args.out, args.run_id, claim_processor=processor, boundary_model=args.boundary_model, input_mode=args.input_mode)
    try:
        for event in _iter_live_events(Path(args.path), args.poll, args.idle):
            if runtime.ingest(event):
                print(json.dumps(dict(native_event_id=event["id"], steps=len(runtime.segmenter.steps),
                      sealed_stages=sum(s['sealed'] for s in runtime._selected_stages()))), flush=True)
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
