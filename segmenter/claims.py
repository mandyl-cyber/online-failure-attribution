"""候选 claim：step -> test_result / env_assertion / plan_step / completion 四类候选。

正则 + 简单句法，只负责"捞出来"，不判断真假；真伪和 σ 由账本写入器（M4）决定。
每条候选带来源字段和字符 span，账本可以按指针回读原文。
"""
from __future__ import annotations

import re
from typing import Optional

from .events import E, Event
from .rules import Rules, default_rules
from .schema import Step

_SENT = re.compile(r"(?:[^.!?。！？\n]|[.!?](?=\S))+[.!?。！？]*")
_FAILED_LINE = re.compile(r"(?m)^(?:FAILED|ERROR) (\S+)(?: - (.*))?$")


def sentences(text: str):
    """(句子, 起, 止)。按句末标点和换行切，够用就行。"""
    for m in _SENT.finditer(text or ""):
        s = m.group(0).strip()
        if len(s) >= 4:
            a = m.start() + (len(m.group(0)) - len(m.group(0).lstrip()))
            yield s, a, a + len(s)


class ClaimExtractor:
    def __init__(self, rules: Rules | None = None):
        self.rules = rules or default_rules()
        self.max_per_step = int(self.rules.claims.get("max_per_step", 8))
        self.max_len = int(self.rules.claims.get("max_len", 240))

    def _mk(self, step: Step, n: int, typ: str, text: str, source: str, span, node: Optional[str]) -> dict:
        return {"id": f"k{step.idx}.{n}", "type": typ, "text": text[: self.max_len], "step_idx": step.idx,
                "source": source, "span": list(span) if span else None, "node": node}

    def extract(self, step: Step, events: list[Event], node: Optional[str] = None) -> list[dict]:
        r, out, seen = self.rules, [], set()

        def add(typ, text, source, span=None):
            text = re.sub(r"\s+", " ", text).strip()
            if text and (typ, text) not in seen and len(out) < self.max_per_step:
                seen.add((typ, text))
                out.append(self._mk(step, len(out) + 1, typ, text, source, span, node))

        obs = step.observation or ""
        # ---- test_result：验证步的汇总行 + 失败用例 ----
        for ev in events:
            if ev.type == E.VERIFY_RUN:
                target = " ".join(p for p in step.touched_paths[:3]) or ev.attrs.get("cmd", "")[:80]
                summary = ev.attrs.get("summary") or ev.attrs.get("result")
                add("test_result", f"{target}: {summary} [{ev.attrs.get('result')}]", "tool_output")
                for m in list(_FAILED_LINE.finditer(obs[-6000:]))[:3]:
                    add("test_result", m.group(0), "tool_output")
        # ---- env_assertion：工具输出里的版本 / 缺失 / 安装结果 ----
        head = obs[:3000] + ("\n" + obs[-3000:] if len(obs) > 3000 else "")
        for p in r.cl_env_obs:
            for m in list(p.finditer(head))[:2]:
                add("env_assertion", m.group(0), "tool_output", (m.start(), m.end()) if len(obs) <= 3000 else None)
        # ---- thought / 交接消息里的三类 ----
        fields = [("thought", step.thought)]
        if any(e.type == E.HANDOFF for e in events):
            fields.append(("message", step.action))
        for source, text in fields:
            for s, a, b in sentences(text):
                if any(p.search(s) for p in r.cl_completion) and not r.cl_completion_neg.search(s):
                    add("completion", s, source, (a, b))
                elif any(p.search(s) for p in r.cl_env_thought):
                    add("env_assertion", s, source, (a, b))
                elif any(p.search(s) for p in r.cl_plan):
                    add("plan_step", s, source, (a, b))
        return out
