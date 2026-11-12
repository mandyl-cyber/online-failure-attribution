import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segmenter.online_ledger import OnlineLedger
from segmenter.run_online import _iter_live_events


def action(i, command="pytest"):
    return {"id": i, "source": "agent", "action": "run", "args": {"command": command}}


def observation(i, cause, content="1 passed in 0.01s"):
    return {"id": i, "source": "environment", "observation": "run", "cause": cause,
            "content": content, "extras": {"exit_code": 0}}


def finish(i):
    return {"id": i, "source": "agent", "action": "finish", "args": {"final_thought": "Done"}}


class OnlineTests(unittest.TestCase):
    def test_boundary_before_finish_seals_only_previous_steps(self):
        with tempfile.TemporaryDirectory() as d:
            runtime = OnlineLedger(d, "run", background=False)
            runtime.ingest(action(1, "cat a.py"))
            runtime.ingest(observation(2, 1, "file content"))
            runtime.ingest(action(3))
            runtime.ingest(observation(4, 3))
            # Verification transition can require several observations under rules.
            runtime.ingest(action(5))
            runtime.ingest(observation(6, 5))
            self.assertFalse(runtime.finished)
            self.assertTrue(list(runtime.jobs.glob("*.json")))
            job = json.loads(next(runtime.jobs.glob("*.json")).read_text(encoding="utf-8"))
            self.assertLess(job["stage"]["end_step_id"], len(runtime.segmenter.steps))
            self.assertTrue(all(e["source_step_id"] <= job["stage"]["end_step_id"] for e in job["evidence"]))
            runtime.drain()
            runtime.close()

    def test_stage_close_multiple_claims_and_no_early_commit(self):
        with tempfile.TemporaryDirectory() as d:
            runtime = OnlineLedger(d, "run", background=False)
            runtime.ingest(action(1))
            self.assertEqual(len(runtime.ledger.steps), 0)
            runtime.ingest(observation(2, 1))
            runtime.ingest(action(3))
            runtime.ingest(observation(4, 3))
            self.assertFalse(list(runtime.results.glob("*.json")))
            self.assertFalse(list(runtime.jobs.glob("*.json")))
            self.assertTrue(all(not op["entries"] for op in runtime.ledger.steps.values()))
            runtime.ingest(finish(5))
            self.assertTrue(runtime.finished)
            self.assertTrue(list(runtime.jobs.glob("*.json")))
            runtime.drain()
            batch = json.loads(next(runtime.results.glob("*.json")).read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(batch["claims"]), 2)
            self.assertFalse(batch["incomplete"])
            self.assertEqual(batch["edges"], [])
            before = next(runtime.results.glob("*.json")).read_bytes()
            runtime.drain()
            self.assertEqual(before, next(runtime.results.glob("*.json")).read_bytes())
            runtime.close()

    def test_pending_action_restart_duplicate_conflict_and_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            runtime = OnlineLedger(d, "run", background=False)
            runtime.ingest(action(10))
            runtime.close()
            recovered = OnlineLedger(d, "run", background=False)
            self.assertIn(10, recovered.adapter._pending)
            self.assertFalse(recovered.ingest(action(10)))
            with self.assertRaises(ValueError): recovered.ingest(action(10, "different"))
            recovered.ingest(observation(11, 10))
            prefix = (recovered.ledger.journal/"00000001.json").read_bytes()
            recovered.ingest(action(12, "cat something.py"))
            recovered.ingest(observation(13, 12, "future content"))
            self.assertEqual(prefix, (recovered.ledger.journal/"00000001.json").read_bytes())
            recovered.close()
            self.assertFalse(recovered.finished)
            second = OnlineLedger(d, "run", background=False)
            self.assertEqual(len(second.ledger.steps), 2)
            second.ingest(finish(14))
            second.drain()
            self.assertEqual(second.terminal_kind, "finished")
            second.close()

    def test_public_plan_current_event_and_abort(self):
        with tempfile.TemporaryDirectory() as d:
            runtime = OnlineLedger(d, "run", background=False)
            event = {"id": 1, "action": "task_tracking", "source": "agent",
                     "args": {"command": "plan", "thought": "secret",
                              "task_list": [{"title": "Test", "status": "todo"}, {"title": "Fix", "status": "todo"}]}}
            runtime.ingest(event)
            runtime.ingest({"id": 2, "observation": "task_tracking", "cause": 1, "content": "saved"})
            self.assertIn("Test", runtime.ledger.steps[1]["step"]["action"])
            self.assertNotIn("secret", json.dumps(runtime.ledger.steps[1]))
            runtime.ingest(action(3))  # unfinished action must not become a completed step
            runtime.abort()
            runtime.drain()
            self.assertEqual(len(runtime.ledger.steps), 1)
            batch = json.loads(next(runtime.results.glob("*.json")).read_text(encoding="utf-8"))
            self.assertTrue(batch["incomplete"])
            self.assertEqual(runtime.terminal_kind, "interrupted")
            runtime.close()

    def test_actual_append_only_file_partial_line_and_background_worker(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d)/"events.jsonl"
            source.touch()
            runtime = OnlineLedger(Path(d)/"out", "live", background=True)
            errors = []
            def consume():
                try:
                    for ev in _iter_live_events(source, .01, 3):
                        runtime.ingest(ev)
                        if runtime.finished:
                            break
                except Exception as exc:
                    errors.append(exc)
            reader = threading.Thread(target=consume)
            reader.start()
            with source.open("a", encoding="utf-8") as fh:
                line = json.dumps(action(1))
                fh.write(line[:len(line)//2]); fh.flush()
                time.sleep(.05)
                self.assertEqual(len(runtime.seen), 0)
                fh.write(line[len(line)//2:]+"\n"); fh.flush()
                for ev in (observation(2, 1), action(3), observation(4, 3), finish(5)):
                    fh.write(json.dumps(ev)+"\n"); fh.flush()
                    time.sleep(.02)
            reader.join(timeout=5)
            self.assertFalse(reader.is_alive())
            self.assertEqual(errors, [])
            runtime.drain()
            self.assertTrue(runtime.finished)
            self.assertEqual(len(runtime.ledger.steps), 3)
            self.assertTrue(list(runtime.results.glob("*.json")))
            runtime.close()


if __name__ == "__main__":
    unittest.main()
