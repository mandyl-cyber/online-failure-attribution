"""合成一条 OpenHands 事件流，复现设计文档"一个时间步的生命周期"，并检查各部件。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmenter.adapters.openhands import convert_events, load_trajectory, messages_to_events  # noqa: E402
from segmenter.eval_segmenter import evaluate, match_boundaries  # noqa: E402
from segmenter.events import E  # noqa: E402
from segmenter.rules import default_rules  # noqa: E402
from segmenter.run_online import run_steps  # noqa: E402
from segmenter.shell import classify_command  # noqa: E402

_id = [0]


def act(kind, thought="", **args):
    _id[0] += 1
    return {"id": _id[0], "source": "agent", "action": kind, "args": {"thought": thought, **args}}


def obs(content, exit_code=None, kind="run"):
    _id[0] += 1
    ex = {"metadata": {"exit_code": exit_code}} if exit_code is not None else {}
    return {"id": _id[0], "cause": _id[0] - 1, "observation": kind, "content": content, "extras": ex}


def make_events():
    _id[0] = 0
    ev = [{"id": 0, "source": "user", "action": "message", "args": {"content": "Fix date parsing with timezones."}}]
    ev += [act("run", "Let me look at the repo.", command="ls -la && git status"), obs("src tests", 0)]            # 1
    ev += [act("run", command="grep -rn 'parse_date' src/"), obs("src/dates.py:10:def parse_date(s):", 0)]         # 2
    ev += [act("read", path="/repo/src/dates.py"), obs("def parse_date(s): ...", kind="read")]                     # 3
    ev += [act("edit", "I will fix parse_date to handle offsets.", path="/repo/src/dates.py", command="str_replace",
               old_str="def parse_date(s):\n    return a", new_str="def parse_date(s):\n    return b"),
           obs("The file /repo/src/dates.py has been edited.", kind="edit")]                                      # 4 -> n1 state
    ev += [act("run", command="cd /repo && python -m pytest tests/test_dates.py -q"),
           obs("F..\nFAILED tests/test_dates.py::test_tz - AssertionError\n1 failed, 2 passed in 0.12s", 1)]        # 5 fail
    ev += [act("edit", path="/repo/src/dates.py", command="str_replace", old_str="return b", new_str="return c"),
           obs("The file /repo/src/dates.py has been edited.", kind="edit")]                                      # 6 repair, same node
    ev += [act("run", command="python -m pytest tests/test_dates.py -q"), obs("...\n3 passed in 0.10s", 0)]        # 7 pass
    ev += [act("edit", "Now update the formatter, which calls parse_date.", path="/repo/src/fmt.py",
               command="str_replace", old_str="x = 1", new_str="x = 2"),
           obs("The file /repo/src/fmt.py has been edited.", kind="edit")]                                        # 8 VERIFY_PASS(for 7)+STATE+REF
    ev += [act("run", command="python -m pytest -q"), obs("=========== 12 passed in 1.02s ===========", 0)]        # 9
    ev += [act("finish", final_thought="The issue is fixed. All tests pass.")]                                    # 10
    return ev


def test_shell():
    r = default_rules()
    assert classify_command('genisoimage -o /tmp/a.iso -R /tmp/input', r).tool_class == 'EXEC_WRITE'
    assert classify_command('genisoimage --help', r).tool_class != 'EXEC_WRITE'
    assert classify_command("cd /repo && python -m pytest tests/ -q 2>&1 | tail -5", r).tool_class == "VERIFY"
    assert classify_command("sed -i 's/a/b/' src/x.py", r).written == ["src/x.py"]
    i = classify_command("cat > /repo/a.py <<'EOF'\nimport os; rm -rf /\nEOF", r)
    assert i.tool_class == "EXEC_WRITE" and i.written == ["/repo/a.py"] and not i.irreversible
    assert classify_command("grep -rn foo . > /dev/null; ls", r).tool_class == "SEARCH"
    assert classify_command("git push origin main", r).irreversible
    assert not classify_command("rm -rf build/", r).irreversible
    assert classify_command("rm -rf src/", r).irreversible
    assert classify_command("pip install -e . && pytest", r).classes >= {"install", "verify"}
    assert classify_command("git checkout -- src/x.py", r).undo
    assert classify_command("python --version", r).classes == {"env"}
    assert classify_command('echo "a > b" | grep a', r).tool_class == "SEARCH"


def test_lifecycle(tmp_path=None):
    steps, meta = convert_events(make_events())
    assert meta["task"].startswith("Fix date") and [s.idx for s in steps] == list(range(1, 11))
    assert [s.tool_class for s in steps] == ["READ", "SEARCH", "READ", "EDIT", "VERIFY", "EDIT", "VERIFY", "EDIT", "VERIFY", "FINISH"]
    res = run_steps(steps, out_dir=tmp_path, modes=["node", "phase"])
    rows = {r["idx"]: r for r in res["rows"]}
    assert rows[4]["events"] == [E.STATE_CHANGE] and rows[4]["node"] == "n1"
    assert rows[5]["detail"][0]["result"] == "fail"
    assert rows[6]["node"] == "n1" and not rows[6]["closed_node"]          # 修复循环不封口
    assert rows[7]["detail"][0]["result"] == "pass" and E.VERIFY_PASS not in rows[7]["events"]
    r8 = rows[8]                                                            # 生命周期第 6-7 步
    assert set(r8["events"]) == {E.VERIFY_PASS, E.STATE_CHANGE, E.REFERENCE}
    assert r8["for_idx"] == 7 and r8["closed_node"] == ["n1"] and r8["node"] == "n2" and r8["refs"] == ["n1"]
    assert r8["trigger"] and r8["sync"] and r8["sealed_stage"] == [1]
    assert set(rows[10]["events"]) == {E.VERIFY_PASS, E.HANDOFF} and rows[10]["closed_node"] == ["n2", "n3"]
    nodes = {n["id"]: n for n in res["tree"]["nodes"]}
    assert nodes["n1"]["span"] == [1, 7] and nodes["n1"]["relied_on_by"] == ["n2"] and nodes["n1"]["symbols"] == ["parse_date"]
    assert [(s["start_step_id"], s["end_step_id"]) for s in res["stages"]["node"]] == [(1, 7), (8, 10)]
    assert [(s["start_step_id"], s["label"]) for s in res["stages"]["phase"]][:3] == [(1, "INSPECT"), (4, "PATCH"), (5, "VERIFY")]
    types = {c["type"] for c in res["claims"]}
    assert types >= {"test_result", "plan_step", "completion"}
    if tmp_path:
        for f in ("steps.jsonl", "events.jsonl", "claims.jsonl", "tree.json", "stage_ranges.json"):
            assert (Path(tmp_path) / f).stat().st_size > 0


def test_revert_and_failed_edit():
    _id[0] = 0
    ev = [act("edit", path="a.py", command="str_replace", old_str="x = 1", new_str="x = 2"), obs("edited", kind="edit"),
          act("run", command="pytest -q"), obs("1 passed in 0.1s", 0),
          act("edit", path="a.py", command="str_replace", old_str="nope", new_str="y"),
          obs("ERROR:\nNo replacement was performed, old_str `nope` did not appear verbatim in a.py.", kind="edit"),
          act("edit", "That approach did not work, let me try a different approach.", path="a.py", command="str_replace",
              old_str="x = 2", new_str="x = 1"), obs("edited", kind="edit")]
    res = run_steps(convert_events(ev)[0])
    rows = res["rows"]
    assert rows[2]["events"] == [E.VERIFY_PASS, E.EXPLORE]                  # 失败的编辑不算改状态
    assert E.PLAN_SWITCH in rows[3]["events"]
    assert [e for e in res["tree"]["edges"] if e["type"] == "reverts"][0]["dst"] == "n1"


def test_formats_and_eval(tmp_path=None):
    import tempfile
    tmp = Path(tmp_path or tempfile.mkdtemp())
    (tmp / "t1").mkdir(exist_ok=True)
    (tmp / "t1" / "traj.json").write_text(json.dumps({"history": make_events()}))
    steps, meta = load_trajectory(tmp / "t1")
    assert meta["format"] == "events" and len(steps) == 10
    msgs = [{"role": "user", "content": "task"},
            {"role": "assistant", "content": "look", "tool_calls": [{"id": "c1", "function": {"name": "execute_bash", "arguments": json.dumps({"command": "pytest -q"})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "2 passed in 0.1s\n[The command completed with exit code 0.]"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "function": {"name": "finish", "arguments": "{\"message\": \"done\"}"}}]}]
    s2, _ = convert_events(messages_to_events(msgs))
    assert [s.tool_class for s in s2] == ["VERIFY", "FINISH"] and s2[0].exit_code == 0 and s2[0].thought == "look"
    (tmp / "steps.json").write_text(json.dumps([{"step_id": 0, "action": "ls", "observation": "a", "thinking": "t"},
                                                 {"step_id": 1, "action": "sed -i s/a/b/ a.py", "observation": ""}]))
    s3, m3 = load_trajectory(tmp / "steps.json")
    assert m3["native_ids"] and [s.idx for s in s3] == [0, 1] and s3[1].tool_class == "EXEC_WRITE"

    assert match_boundaries([4, 8, 20], [5, 8, 12], 1) == 2
    entry = {"traj_id": "t1", "traj_path": str(tmp / "t1"), "base": 1, "step_count": 10, "solved": False,
             "stages": [{"stage_id": 1, "start_step_id": 1, "end_step_id": 3}, {"stage_id": 2, "start_step_id": 4, "end_step_id": 7},
                        {"stage_id": 3, "start_step_id": 8, "end_step_id": 10}],
             "incorrect_steps": [4, 6], "instances": [4]}
    out = evaluate(iter([entry]), default_rules(), ["node", "phase"])["summary"]
    assert out["boundary"]["node"]["precision"] == 1.0 and out["boundary"]["node"]["recall"] == 0.5
    assert out["error_landing"]["state_change_rate"] == 1.0
    assert out["window_proxy"]["any"]["median"] == 4 and out["window_proxy"]["targeted"]["median"] == 4
    assert out["event_recall"]["incorrect_steps"]["recall"] == 0.5        # 步 6 之后 3 步内有，步 4 没有
    assert out["event_recall_curve"]["3"]["instances"]["hits"] == 0
    assert out["event_recall_curve"]["6"]["instances"]["hits"] == 1
    assert out["by_source"]["unknown"]["n_traj"] == 1
    from segmenter.eval_segmenter import source_family
    assert source_family({"source_relpath": "swe_raw/openhands__verified/task"}) == "SWE"
    assert source_family({"source_relpath": "openhands/model/task"}) == "Terminal-Bench"
    assert source_family({}) == "unknown"


if __name__ == "__main__":
    test_shell(); test_lifecycle(); test_revert_and_failed_edit(); test_formats_and_eval()
    print("ok")
