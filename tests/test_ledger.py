import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import unittest
from segmenter.ledger import Ledger, repair_step
from segmenter.schema import Step


class LedgerTests(unittest.TestCase):
    def test_public_plan_and_native_identity(self):
        step = Step(1, tool="task_tracking", thought="private", raw_ref={"action_id": "a"})
        payload = {"task_list": [{"title": "Test", "status": "todo", "thought": "secret"}], "gold": 9}
        fixed = repair_step(step, payload, "a")
        self.assertEqual(step.action, "")
        self.assertEqual(fixed.thought, "")
        self.assertNotIn("secret", fixed.action)
        self.assertNotIn("gold", fixed.action)
        with self.assertRaises(ValueError): repair_step(step, payload, "future")
        with self.assertRaises(ValueError): repair_step(step)
        query = Step(2, tool="task_tracking", action='{"command":"view","thought":"secret"}')
        self.assertEqual(repair_step(query).action, '{"command":"view"}')

    def test_recovery_idempotency_and_conflict(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = Ledger(d, "run")
            step = Step(1, action="pytest", observation="1 passed", tool="run", tool_class="VERIFY", exit_code=0)
            first = ledger.feed(step)
            self.assertEqual(first, ledger.feed(step))
            self.assertEqual(len(ledger.operations), 1)
            recovered = Ledger(d, "run")
            self.assertEqual(first, recovered.feed(step))
            next_step = Step(2, action="cat a.py", tool="run", tool_class="READ")
            expected = recovered.feed(next_step)
            # Separate uninterrupted execution gives identical second-step output.
            with tempfile.TemporaryDirectory() as other:
                continuous = Ledger(other, "run")
                continuous.feed(step)
                self.assertEqual(continuous.feed(next_step), expected)
            step.observation = "different"
            with self.assertRaises(ValueError): recovered.feed(step)
            verification = first["entries"][0]
            self.assertEqual(verification["kind"], "verification")
            self.assertIsNone(verification["attrs"]["verified_artifact_version"])
            self.assertEqual(first["edges"], [])

    def test_prefix_evidence_and_stage_revision(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = Ledger(d, "run")
            one = ledger.feed(Step(1, action="pytest", tool_class="VERIFY", observation="1 passed", exit_code=0))
            ids = [e["record_id"] for e in one["evidence"]]
            original = ledger.get_evidence(ids, 1)
            ledger.feed(Step(2, action="different future"))
            self.assertEqual(original, ledger.get_evidence(ids, 1))
            future = ledger.steps[2]["evidence"][0]["record_id"]
            with self.assertRaises(ValueError): ledger.get_evidence([future], 1)
            with self.assertRaises(KeyError): ledger.get_evidence(["foreign-id"])
            a = ledger.assign_stages([dict(stage_id=1, start_step_id=1, end_step_id=2)])
            b = ledger.assign_stages([dict(stage_id=1, start_step_id=1, end_step_id=1), dict(stage_id=2, start_step_id=2, end_step_id=2)])
            self.assertEqual({x["entry_id"] for x in a["memberships"]}, {x["entry_id"] for x in b["memberships"]})
            self.assertEqual(b["revision"], 2)
            ledger.assign_stages(b["stages"])
            self.assertEqual(len(ledger.operations), 4)
            with self.assertRaises(ValueError): ledger.assign_stages([dict(stage_id=1, start_step_id=1, end_step_id=3)])
            ledger.evidence[ids[0]]["text"] = "tampered"
            with self.assertRaises(ValueError): ledger.get_evidence(ids)


if __name__ == "__main__":
    unittest.main()
