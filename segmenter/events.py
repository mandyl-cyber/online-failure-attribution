"""事件检测：统一 step (+ 树快照) -> 事件集合。正则 + 工具类型，零解码。

STATE_CHANGE / EXPLORE / VERIFY_RUN / VERIFY_PASS / REFERENCE / HANDOFF / IRREVERSIBLE / PLAN_SWITCH / DRIFT

- VERIFY_PASS 是补发的：第 t 步验证通过，第 t+1 步（第一个非纯验证步）到来时发出，for_idx = t。
  它表示"验证结果被接受、agent 往下走了"，所以天然晚一步。
- REFERENCE 依赖树（要知道更早的节点改过什么），由 detect_references 在树更新之后算。
- DRIFT 第一轮不发（rules.yaml: events.drift.enabled）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .rules import Rules, default_rules
from .schema import Step, ToolClass


class E:
    STATE_CHANGE = "STATE_CHANGE"
    EXPLORE = "EXPLORE"
    VERIFY_RUN = "VERIFY_RUN"
    VERIFY_PASS = "VERIFY_PASS"
    REFERENCE = "REFERENCE"
    HANDOFF = "HANDOFF"
    IRREVERSIBLE = "IRREVERSIBLE"
    PLAN_SWITCH = "PLAN_SWITCH"
    DRIFT = "DRIFT"

    ALL = [STATE_CHANGE, EXPLORE, VERIFY_RUN, VERIFY_PASS, REFERENCE, HANDOFF, IRREVERSIBLE, PLAN_SWITCH, DRIFT]
    # "接受"的结构事件：t_lock 的操作化定义取其中最早的一个
    STRUCTURAL = {REFERENCE, HANDOFF, VERIFY_PASS, IRREVERSIBLE}


@dataclass
class Event:
    type: str
    idx: int
    for_idx: Optional[int] = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"type": self.type, "idx": self.idx}
        if self.for_idx is not None:
            d["for_idx"] = self.for_idx
        if self.attrs:
            d.update(self.attrs)
        return d


_PYTEST_SUMMARY = re.compile(r"(?m)^=+ (.*?\b\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)\b.*?) in [\d.]+ ?s.*=+\s*$")
_PYTEST_SHORT = re.compile(r"(?m)^((?:\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)(?:, )?)+) in [\d.]+ ?s")
_UNITTEST = re.compile(r"(?m)^Ran (\d+) tests? in [\d.]+s\s*\n+\s*(OK|FAILED)[^\n]*")
_JEST = re.compile(r"(?m)^Tests:\s+(.*)$")
_GO = re.compile(r"(?m)^(ok|FAIL)\s+\S+(\s+[\d.]+s)?")
_CARGO = re.compile(r"test result: (ok|FAILED)\. (\d+) passed; (\d+) failed")
_COUNT = re.compile(r"(\d+) (passed|failed|errors?)")


def verify_result(step: Step, rules: Rules) -> tuple[str, str]:
    """返回 (pass|fail|unknown, 汇总行)。汇总行 > exit code > 通用正则。"""
    out = step.observation or ""
    tail = out[-6000:]
    m = None
    for m in _PYTEST_SUMMARY.finditer(tail):
        pass
    if m is None:
        for m in _PYTEST_SHORT.finditer(tail):
            pass
    if m is not None:
        counts = {k.rstrip("s") if k.startswith("error") else k: int(n) for n, k in _COUNT.findall(m.group(1))}
        bad = counts.get("failed", 0) + counts.get("error", 0)
        if bad:
            return "fail", m.group(1).strip()
        if counts.get("passed", 0):
            return "pass", m.group(1).strip()
    m = _UNITTEST.search(tail)
    if m:
        return ("pass" if m.group(2) == "OK" else "fail"), re.sub(r"\s+", " ", m.group(0)).strip()
    m = _CARGO.search(tail)
    if m:
        return ("pass" if m.group(1) == "ok" and m.group(3) == "0" else "fail"), m.group(0)
    m = _JEST.search(tail)
    if m:
        return ("fail" if "failed" in m.group(1) else "pass"), m.group(0).strip()
    gos = _GO.findall(tail)
    if gos:
        return ("fail" if any(g[0] == "FAIL" for g in gos) else "pass"), f"go: {len(gos)} packages"
    if step.exit_code is not None:
        return ("pass" if step.exit_code == 0 else "fail"), f"exit code {step.exit_code}"
    if step.attrs.get("error_observation"):
        return "fail", "error observation"
    if any(p.search(tail) for p in rules.verify_fail):
        return "fail", ""
    if any(p.search(tail) for p in rules.verify_pass):
        return "pass", ""
    return "unknown", ""


class EventDetector:
    def __init__(self, rules: Rules | None = None):
        self.rules = rules or default_rules()
        self._pending_pass: Optional[int] = None   # 最近一次"通过且尚未被放行"的验证步
        self._seen_state = False                   # 轨迹里是否已经出现过状态改变

    def detect(self, step: Step) -> list[Event]:
        r, a, tc, i = self.rules, step.attrs, step.tool_class, step.idx
        classes = set(a.get("cmd_classes", []))
        evs: list[Event] = []

        is_verify = tc == ToolClass.VERIFY or "verify" in classes
        probe = False
        if not is_verify and "probe" in classes and self._seen_state and not a.get('suppress_probe_verify'):
            is_verify = probe = True               # 改过东西之后跑内联脚本 / 脚本文件 = 验证
        is_state = tc in ToolClass.STATE or "state" in classes or "install" in classes
        failed_edit = False
        if tc == ToolClass.EDIT:
            head = (step.observation or "")[:600]
            failed_edit = bool(a.get("error_observation")) or any(p.search(head) for p in r.edit_failed)
            is_state = not failed_edit
            if failed_edit:                      # 没写成：不算写过这个文件
                step.written_paths = []
                a["failed_edit"] = True

        # 1. 补发 VERIFY_PASS：上一个通过的验证之后，agent 做了别的事
        if self._pending_pass is not None and (is_state or not is_verify):
            evs.append(Event(E.VERIFY_PASS, i, for_idx=self._pending_pass))
            self._pending_pass = None

        # 2. PLAN_SWITCH（看 thought）
        for p in r.plan_switch:
            m = p.search(step.thought or "")
            if m:
                evs.append(Event(E.PLAN_SWITCH, i, attrs={"match": m.group(0)[:80]}))
                break

        if a.get("handoff_before"):      # 上一回合 agent 只说了话没调工具（想交卷），被要求继续
            evs.append(Event(E.HANDOFF, i, attrs={"kind": "message", "before": True}))
        # 3. 本步动作
        if a.get("irreversible"):
            evs.append(Event(E.IRREVERSIBLE, i, attrs={"cmd": a["irreversible"]}))
            is_state = True
        if is_state:
            self._seen_state = True
            kind = "env" if (tc == ToolClass.INSTALL or "install" in classes) and not step.written_paths else "file"
            attrs = {"kind": kind, "paths": step.written_paths[:20]}
            if a.get("undo"):
                attrs["undo"] = True
            evs.append(Event(E.STATE_CHANGE, i, attrs=attrs))
        if is_verify:
            res, summary = verify_result(step, r)
            evs.append(Event(E.VERIFY_RUN, i, attrs={"result": res, "kind": "probe" if probe else a.get("verify_kind", "test"),
                                                      "cmd": a.get("verify_cmd", ""), "summary": summary}))
            self._pending_pass = i if res == "pass" else None
        handoff = (tc in (ToolClass.FINISH, ToolClass.DELEGATE) or a.get("submit")
                   or (tc == ToolClass.MESSAGE and step.role != "user"
                       and (a.get("wait_for_response") or not r.message_requires_wait)))
        if handoff:
            evs.append(Event(E.HANDOFF, i, attrs={"kind": step.tool or tc}))
        if not (is_state or is_verify or handoff) and step.role != "user":
            attrs = {"failed_state_change": True} if failed_edit else {}
            evs.append(Event(E.EXPLORE, i, attrs=attrs))
        # 4. DRIFT：第二轮（一次 embedding）接在这里
        return evs

    def detect_references(self, step: Step, tree) -> Optional[Event]:
        """本步是否把更早节点的产物当前提用了。必须在 tree.update(step) 之后调用。"""
        refs = tree.find_refs(step)
        if not refs:
            return None
        return Event(E.REFERENCE, step.idx, attrs={
            "refs": list(dict.fromkeys(n for n, _, _ in refs)),
            "via": [{"node": n, "kind": k, "token": t} for n, k, t in refs],
        })
