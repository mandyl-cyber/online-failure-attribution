"""Local ledger prototype. Single writer, atomic per-operation commits; no API."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from .audit_plan_payloads import public_payload
from .events import E, EventDetector
from .schema import Step

VERSION = "public-plan-v2"


def encoded(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def atomic(path, obj):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        fh.write(encoded(obj))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def repair_step(step, payload=None, native_call_id=None):
    """Separate version; never mutate legacy Step or accept a different native call."""
    result = deepcopy(step)
    result.thought = ""
    if payload is None and step.tool == 'task_tracking' and step.attrs.get('input_integrity',{}).get('version') == 2:
        public = step.attrs.get('public_action_args')
        if not isinstance(public,dict) or json.loads(step.action)!=public or 'thought' in public:
            raise ValueError('Adapter public plan integrity mismatch')
        result.action=encoded(public).decode('utf-8')
        result.attrs['public_plan']=deepcopy(public)
        return result
    if payload is not None:
        if step.tool != "task_tracking" or native_call_id is None or native_call_id != step.raw_ref.get("action_id"):
            raise ValueError("Plan payload does not match current native call")
        clean = public_payload(payload)
        if clean is None:
            raise ValueError("Missing public task_list")
        result.action = encoded(clean).decode("utf-8")
        result.attrs["public_plan"] = clean
    elif step.tool == "task_tracking":
        try:
            query = json.loads(step.action)
        except (ValueError, TypeError) as exc:
            raise ValueError("Plan requires exact-call payload or structured query") from exc
        if not isinstance(query, dict) or not isinstance(query.get("command"), str) or "task_list" in query:
            raise ValueError("Unrecognized plan query; exact-call payload required")
        result.action = encoded({"command": query["command"]}).decode("utf-8")
    return result


class Ledger:
    """Evidence immutable after commit; restart rebuilds detector from committed steps.

    The journal is authoritative; exported JSONL views may be regenerated. Not safe
    for concurrent writers. Each completed action+observation is one transaction.
    """

    def __init__(self, directory, run_id, defer_claims=False):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.journal = self.root / "journal"
        self.journal.mkdir(exist_ok=True)
        self.run_id = run_id
        self.defer_claims = defer_claims
        meta = {"schema_version": "ledger-prototype-v0.1", "run_id": run_id,
                "input_version": VERSION, "step_unit": "action_observation",
                "writer_mode": "single_writer", "network": False}
        if defer_claims:
            meta["claim_commit_policy"] = "stage_close"
        path = self.root / "run.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != meta:
                raise ValueError("Run identity/version mismatch")
        else:
            atomic(path, meta)
        self.operations = []
        self.steps = {}
        self.detector = EventDetector()
        self.evidence = {}
        for path in sorted(self.journal.glob("*.json")):
            operation = json.loads(path.read_text(encoding="utf-8"))
            if path.name != f"{len(self.operations)+1:08d}.json":
                raise ValueError("Noncontiguous journal")
            self._accept(operation, replay=True)

    def _id(self, *parts):
        return sha(encoded([self.run_id, *parts]))[:32]

    def _accept(self, operation, replay=False):
        self.operations.append(operation)
        if operation["kind"] == "step":
            st = Step.from_dict(operation["step"])
            self.steps[st.idx] = operation
            if replay:
                self.detector.detect(st)
            for ev in operation["evidence"]:
                self.evidence[ev["record_id"]] = ev

    def _commit(self, operation):
        path = self.journal / f"{len(self.operations)+1:08d}.json"
        if path.exists():
            raise ValueError("Concurrent writer or journal conflict")
        atomic(path, operation)
        self._accept(operation)

    def feed(self, step, payload=None, native_call_id=None):
        st = repair_step(step, payload, native_call_id)
        # Explicit whitelist: attrs may contain offline annotations in other adapters.
        clean = {k: getattr(st, k) for k in (
            "idx", "role", "action", "observation", "tool_class", "tool",
            "exit_code", "touched_paths", "written_paths", "raw_ref")}
        clean["attrs"] = {"public_plan": st.attrs["public_plan"]} if "public_plan" in st.attrs else {}
        st = Step.from_dict(clean)
        fingerprint = sha(encoded(clean))
        if st.idx in self.steps:
            old = self.steps[st.idx]
            if old["fingerprint"] != fingerprint:
                raise ValueError("Conflicting duplicate step")
            return old
        if self.steps and st.idx <= max(self.steps):
            raise ValueError("Out-of-order step")
        seq = len(self.steps)+1
        evidence = []
        for field in ("action", "observation"):
            text = clean[field]
            evidence.append(dict(record_id=self._id(st.idx, field), run_id=self.run_id,
                source_step_id=st.idx, field=field, span=[0, len(text)],
                content_sha256=sha(text.encode("utf-8")), text=text,
                raw_ref=st.raw_ref, availability_seq=seq,
                representation="repaired_public_plan" if field == "action" and payload is not None else "adapter_field_untruncated"))
        # Detector state is committed only after the disk transaction succeeds.
        detector = deepcopy(self.detector)
        events = detector.detect(st)
        kinds = {e.type for e in events}
        entries = []
        def add(kind, text, status, **attrs):
            entries.append(dict(record_id=self._id(st.idx, kind), entry_id=self._id(st.idx, kind),
                run_id=self.run_id, revision=1, observed_at_seq=seq, kind=kind,
                text=text, source_step_ids=[st.idx], evidence_ids=[e["record_id"] for e in evidence],
                epistemic_status=status, adoption_status="unknown", attrs=attrs,
                artifact_refs=[dict(path=p, version_id=None, version_basis="unknown", environment_id=None)
                               for p in st.touched_paths], producer="deterministic-prototype-v0.1"))
        if E.STATE_CHANGE in kinds:
            add("state_change", "Rule-detected state-changing action; resulting content not inspected.",
                "inferred", written_paths=st.written_paths)
        if E.VERIFY_RUN in kinds:
            ev = next(e for e in events if e.type == E.VERIFY_RUN)
            add("verification", "Observed verification command output; patch correctness not established.",
                "observed", exit_code=st.exit_code, rule_result=ev.attrs.get("result"),
                verified_artifact_version=None, environment_id=None)
        if "public_plan" in st.attrs:
            add("decision", "Agent public task plan (not proof of execution).", "asserted",
                public_plan=st.attrs["public_plan"])
        if st.tool == "finish":
            add("conclusion", st.action, "asserted")
        operation = dict(kind="step", observed_at_seq=seq, fingerprint=fingerprint,
            step=clean, evidence=evidence, entries=entries,
            events=[e.to_dict() for e in events], edges=[])
        if self.defer_claims:
            operation["buffered_candidates"] = operation["entries"]
            operation["entries"] = []
        self._commit(operation)
        self.detector = detector
        return operation

    def assign_stages(self, stages, origin="gold_oracle"):
        if origin not in ("gold_oracle", "rule", "model"):
            raise ValueError("Unknown stage origin")
        assigned = {}
        stage_ids = set()
        for stage in stages:
            sid = stage["stage_id"]
            if sid in stage_ids:
                raise ValueError("Duplicate stage ID")
            stage_ids.add(sid)
            lo, hi = stage["start_step_id"], stage["end_step_id"]
            if lo > hi:
                raise ValueError("Reversed stage")
            for idx in range(lo, hi+1):
                if idx not in self.steps or idx in assigned:
                    raise ValueError("Future, missing, or overlapping stage step")
                assigned[idx] = sid
        if set(assigned) != set(self.steps):
            raise ValueError("Stage snapshot must cover observed steps")
        previous = [o for o in self.operations if o["kind"] == "stages"]
        if previous and previous[-1]["stages"] == stages and previous[-1]["origin"] == origin:
            return previous[-1]
        operation = dict(kind="stages", stages=deepcopy(stages), origin=origin,
            revision=len(previous)+1, observed_at_seq=len(self.steps),
            memberships=[dict(entry_id=e["entry_id"], stage_id=assigned[idx])
                         for idx, op in self.steps.items() for e in op["entries"]])
        self._commit(operation)
        return operation

    def get_evidence(self, ids, cutoff_seq=None):
        result = []
        for eid in ids:
            ev = self.evidence[eid]
            if cutoff_seq is not None and ev["availability_seq"] > cutoff_seq:
                raise ValueError("Evidence exceeds online cutoff")
            if sha(ev["text"].encode("utf-8")) != ev["content_sha256"]:
                raise ValueError("Evidence checksum mismatch")
            result.append(deepcopy(ev))
        return result

    def export(self):
        for name in ("evidence", "entries", "edges"):
            records = [r for op in self.operations if op["kind"] == "step" for r in op[name]]
            path = self.root / f"{name}.jsonl"
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(b"".join(encoded(r)+b"\n" for r in records))
            os.replace(tmp, path)
        atomic(self.root/"stage_snapshots.json", [o for o in self.operations if o["kind"] == "stages"])
