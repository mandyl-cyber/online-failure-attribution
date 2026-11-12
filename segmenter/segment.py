"""切分：事件 + 树 -> stage_ranges.json（懒封口）。

两种模式（rules.yaml: segment.mode）
  node   stage 边界 = 节点封口点。VERIFY_RUN 之后的第一步先记为候选边界，
         等节点真的封口（下一步改别处 / 验证放行 / 交接）才落成 stage；
         短于 min_stage_len 的不封，并进下一个 stage。
  phase  按活动相位切（ENV / INSTALL / INSPECT / PATCH / VERIFY，对应 CodeTraceBench 的人标 stage 类型）。
         相位变化先记候选，新相位连续 >= phase_min_run 步才封口，否则当作抖动吸收。

输出格式与 CodeTraceBench 的 stages 字段一致：
  [{stage_id, start_step_id, end_step_id, label, sealed, nodes}, ...]
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from .events import E, Event
from .rules import Rules, default_rules
from .schema import Step, ToolClass

PHASES = ["ENV", "INSTALL", "INSPECT", "PATCH", "VERIFY"]


def step_phase(step: Step, types: set) -> Optional[str]:
    """一步属于哪个相位；think / message 等中性步返回 None（继承前一步）。"""
    classes = set(step.attrs.get("cmd_classes", []))
    if E.STATE_CHANGE in types:
        if step.tool_class == ToolClass.INSTALL or (classes & {"install"} and not step.written_paths):
            return "INSTALL"
        return "PATCH"
    if E.VERIFY_RUN in types:
        return "VERIFY"
    if step.tool_class in (ToolClass.THINK, ToolClass.MESSAGE, ToolClass.PLAN, ToolClass.FINISH,
                           ToolClass.DELEGATE, ToolClass.OTHER):
        return None
    if "noop" in classes:
        return None
    if classes and classes <= {"env"}:
        return "ENV"
    return "INSPECT"


class Segmenter:
    def __init__(self, rules: Rules | None = None, mode: str | None = None):
        self.rules = rules or default_rules()
        cfg = self.rules.segment
        self.mode = mode or cfg.get("mode", "node")
        if self.mode not in ("node", "phase"):
            raise ValueError(f"segment.mode 只能是 node / phase，得到 {self.mode}")
        self.min_len = int(cfg.get("min_stage_len", 2))
        self.pull_back = bool(cfg.get("phase_pull_back_neutral", True))
        self.min_run = {k: int(v) for k, v in cfg.get("phase_min_run", {}).items()}
        self.stages: list[dict] = []
        self.candidates: list[int] = []          # 候选边界（新 stage 的起始步）
        self._open_start: Optional[int] = None
        self._last_idx: Optional[int] = None
        self._phases: dict[int, Optional[str]] = {}
        self._nodes: dict[int, str] = {}
        # phase 模式的状态
        self._stage_phase: Optional[str] = None
        self._cand: Optional[int] = None
        self._cand_phase: Optional[str] = None
        self._cand_run = 0
        self._prev_verify = False

    # ---------------------------------------------------------------
    def _label(self, a: int, b: int) -> str:
        c = Counter(p for i, p in self._phases.items() if a <= i <= b and p)
        return c.most_common(1)[0][0] if c else "OTHER"

    def _seal(self, end_idx: int) -> dict:
        a = self._open_start
        st = {"stage_id": len(self.stages) + 1, "start_step_id": a, "end_step_id": end_idx,
              "label": self._label(a, end_idx), "sealed": True,
              "nodes": list(dict.fromkeys(n for i, n in sorted(self._nodes.items()) if a <= i <= end_idx))}
        self.stages.append(st)
        self._open_start = end_idx + 1
        return st

    def update(self, step: Step, events: list[Event], tree_update: dict) -> list[dict]:
        """喂一步，返回本步新封口的 stage（通常 0 或 1 个）。"""
        i = step.idx
        types = {e.type for e in events}
        sealed: list[dict] = []
        if self._open_start is None:
            self._open_start = i
        prev = self._last_idx
        ph = step_phase(step, types)
        self._phases[i] = ph
        self._nodes[i] = tree_update["node"]

        if self.mode == "node":
            if self._prev_verify and E.VERIFY_RUN not in types:
                self.candidates.append(i)                    # VERIFY_RUN 之后的第一步：候选，暂不封
            # 本步之前封口的节点 -> stage 在 prev 结束
            pre = [c for c in tree_update["closed"] if c[1] != "handoff"]
            if pre and prev is not None and prev - self._open_start + 1 >= self.min_len:
                sealed.append(self._seal(prev))
            self._last_idx = i
            # 本步之后封口（HANDOFF）
            if any(c[1] == "handoff" for c in tree_update["closed"]) and i - self._open_start + 1 >= self.min_len:
                sealed.append(self._seal(i))
        else:
            eff = ph or self._stage_phase            # 中性步继承
            if self._stage_phase is None:
                self._stage_phase = eff
            elif eff != self._stage_phase and ph is not None:
                if self._cand is None or ph != self._cand_phase:
                    # A different candidate phase cannot inherit the previous onset.
                    self._cand = i
                    self.candidates.append(i)
                    self._cand_phase, self._cand_run = ph, 0
                self._cand_run += 1
                if self._cand_run >= self.min_run.get(ph, 1):
                    start = self._cand
                    if self.pull_back:           # 切点前紧挨着的中性步归新 stage
                        while start - 1 > self._open_start and self._phases.get(start - 1, "x") is None:
                            start -= 1
                    sealed.append(self._seal(start - 1))
                    self._stage_phase, self._cand = ph, None
            elif self._cand is not None and ph is not None:
                self._cand = None                    # 回到原相位：抖动，丢掉候选
                self._cand_phase, self._cand_run = None, 0
            self._last_idx = i
        self._prev_verify = E.VERIFY_RUN in types
        return sealed

    def finish(self) -> list[dict]:
        if self._open_start is not None and self._last_idx is not None and self._last_idx >= self._open_start:
            if self.stages and self._last_idx - self._open_start + 1 < self.min_len and self.mode == "node":
                # 尾巴太短：并进上一个 stage
                last = self.stages[-1]
                last["end_step_id"] = self._last_idx
                last["nodes"] = list(dict.fromkeys(last["nodes"] + [self._nodes[i] for i in sorted(self._nodes) if i > last["start_step_id"]]))
                self._open_start = self._last_idx + 1
                return []
            return [self._seal(self._last_idx)]
        return []

    def snapshot(self) -> list[dict]:
        """已封口的 stage + 当前未封口的尾巴（sealed = false）。"""
        out = list(self.stages)
        if self._open_start is not None and self._last_idx is not None and self._last_idx >= self._open_start:
            out.append({"stage_id": len(out) + 1, "start_step_id": self._open_start, "end_step_id": self._last_idx,
                        "label": self._label(self._open_start, self._last_idx), "sealed": False,
                        "nodes": list(dict.fromkeys(n for i, n in sorted(self._nodes.items()) if i >= self._open_start))})
        return out


def cut_points(stages: list[dict]) -> list[int]:
    """stage 列表 -> 切点（除第一个以外每个 stage 的起始步）。"""
    starts = sorted(int(s["start_step_id"]) for s in stages)
    return starts[1:]
