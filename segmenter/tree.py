"""状态树：事件 -> 节点、挂载的探索 / 验证步、relied_on_by 边、reverts 边、封口信号。

节点 = 一次"状态改变单元"：若干前导探索步 + 一批改状态步 + 挂在下面的验证步。
封口后的下一步立即开新节点；新节点在出现 STATE_CHANGE 之前是"仅探索"节点。

封口规则
  封口在本步之前（本步进新节点）：
    VERIFY_PASS            验证被放行                      reason = verify_pass
    PLAN_SWITCH            节点已有改动、agent 换方案       reason = plan_switch
    IRREVERSIBLE           不可逆动作前强制封口             reason = pre_irreversible
    STATE_CHANGE           节点上次改动后已验证过           reason = verified
                           （验证失败且改同一批文件 = 修复循环，不封，除非 close_on_failed_verify）
    STATE_CHANGE           隔了 >= explore_gap 步探索且文件不相交   reason = new_target
    STATE_CHANGE           merge_consecutive_edits = false  reason = per_step
    任意                   节点超过 max_node_steps          reason = max_len
  封口在本步之后：
    HANDOFF                交接强制封口                     reason = handoff
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional

from .events import E, Event
from .rules import Rules, default_rules
from .schema import Step

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_PATH_IN_TEXT = re.compile(r"[\w./\-]*\w\.[A-Za-z][A-Za-z0-9]{0,4}\b")


def _h(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", "replace")).hexdigest()[:12]


def paths_match(a: str, b: str) -> bool:
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


@dataclass
class Node:
    id: str
    start_idx: int
    end_idx: int
    state_steps: list = field(default_factory=list)
    explore_steps: list = field(default_factory=list)
    verify_steps: list = field(default_factory=list)
    other_steps: list = field(default_factory=list)
    touched_paths: list = field(default_factory=list)   # 本节点写过的路径
    symbols: list = field(default_factory=list)         # 本节点改动里出现的 def / class 名
    last_verify: Optional[str] = None                   # pass / fail / unknown
    last_verify_idx: Optional[int] = None
    last_state_idx: Optional[int] = None
    relied_on_by: list = field(default_factory=list)    # 依赖本节点的后继节点
    relies_on: list = field(default_factory=list)
    reverts: list = field(default_factory=list)         # 本节点撤销了哪些节点
    irreversible: bool = False
    closed: bool = False
    close_reason: Optional[str] = None
    _edits: list = field(default_factory=list, repr=False)   # (path, hash(old), hash(new))

    @property
    def kind(self) -> str:
        return "state" if self.state_steps else "explore"

    @property
    def n_steps(self) -> int:
        return self.end_idx - self.start_idx + 1

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "span": [self.start_idx, self.end_idx],
            "state_steps": self.state_steps, "explore_steps": self.explore_steps,
            "verify_steps": self.verify_steps, "other_steps": self.other_steps,
            "touched_paths": self.touched_paths, "symbols": self.symbols,
            "last_verify": self.last_verify, "relied_on_by": self.relied_on_by,
            "relies_on": self.relies_on, "reverts": self.reverts,
            "dep_count": len(self.relied_on_by), "irreversible": self.irreversible,
            "closed": self.closed, "close_reason": self.close_reason,
        }


class StateTree:
    def __init__(self, rules: Rules | None = None):
        self.rules = rules or default_rules()
        self.cfg = self.rules.tree
        self.nodes: list[Node] = []
        self.by_id: dict[str, Node] = {}
        self.current: Optional[Node] = None
        self.edges: list[dict] = []
        self.step_node: dict[int, str] = {}
        self._sym_index: dict[str, list[str]] = {}

    # ------------------------------------------------------------ 节点开合
    def _open(self, idx: int) -> Node:
        n = Node(id=f"n{len(self.nodes) + 1}", start_idx=idx, end_idx=idx)
        self.nodes.append(n)
        self.by_id[n.id] = n
        self.current = n
        return n

    def _close(self, reason: str) -> Optional[str]:
        n = self.current
        if n is None:
            return None
        n.closed, n.close_reason = True, reason
        self.current = None
        return n.id

    def _pre_close_reason(self, step: Step, types: set) -> Optional[str]:
        n = self.current
        if n is None:
            return None
        has_state = bool(n.state_steps)
        if E.VERIFY_PASS in types and (has_state or self.cfg.get("verify_pass_closes_empty_node", True)):
            return "verify_pass"
        if E.PLAN_SWITCH in types and has_state:
            return "plan_switch"
        if E.IRREVERSIBLE in types:
            return "pre_irreversible"
        if E.STATE_CHANGE in types and has_state:
            if not self.cfg.get("merge_consecutive_edits", True):
                return "per_step"
            overlap = any(paths_match(a, b) for a in step.written_paths for b in n.touched_paths)
            last_state = n.last_state_idx if n.last_state_idx is not None else step.idx
            verified = n.last_verify_idx is not None and n.last_verify_idx > last_state
            if verified:
                repair = n.last_verify == "fail" and (overlap or not step.written_paths)
                if not repair or self.cfg.get("close_on_failed_verify", False):
                    return "verified"
            else:
                gap = step.idx - last_state - 1
                if gap >= int(self.cfg.get("explore_gap", 3)) and step.written_paths and n.touched_paths and not overlap:
                    return "new_target"
        if n.n_steps >= int(self.cfg.get("max_node_steps", 40)):
            return "max_len"
        return None

    # ------------------------------------------------------------ 主入口
    def update(self, step: Step, events: list[Event]) -> dict:
        """喂一步。返回 {node, closed: [(node_id, reason)], opened}。"""
        types = {e.type for e in events}
        closed: list = []
        opened = None
        handoff_before = any(e.type == E.HANDOFF and e.attrs.get("before") for e in events)
        handoff_after = any(e.type == E.HANDOFF and not e.attrs.get("before") for e in events)

        reason = "handoff_before" if (handoff_before and self.current is not None) else self._pre_close_reason(step, types)
        if reason:
            closed.append((self._close(reason), reason))
        if self.current is None:
            opened = self._open(step.idx).id
        n = self.current
        n.end_idx = step.idx
        self.step_node[step.idx] = n.id

        if E.STATE_CHANGE in types:
            n.state_steps.append(step.idx)
            n.last_state_idx = step.idx
            for p in step.written_paths:
                if p not in n.touched_paths:
                    n.touched_paths.append(p)
            for s in step.attrs.get("symbols", []):
                if s not in n.symbols:
                    n.symbols.append(s)
                    self._sym_index.setdefault(s, []).append(n.id)
            if E.IRREVERSIBLE in types:
                n.irreversible = True
            self._link_reverts(step, n)
        if E.VERIFY_RUN in types:
            n.verify_steps.append(step.idx)
            ev = next(e for e in events if e.type == E.VERIFY_RUN)
            n.last_verify, n.last_verify_idx = ev.attrs.get("result"), step.idx
        if E.EXPLORE in types:
            n.explore_steps.append(step.idx)
        if not types & {E.STATE_CHANGE, E.VERIFY_RUN, E.EXPLORE}:
            n.other_steps.append(step.idx)

        if handoff_after:
            closed.append((self._close("handoff"), "handoff"))
        return {"node": n.id, "closed": closed, "opened": opened}

    def finish(self) -> Optional[str]:
        return self._close("end")

    # ------------------------------------------------------------ 边
    def _add_edge(self, src: str, dst: str, typ: str, idx: int, **kw) -> bool:
        s, d = self.by_id[src], self.by_id[dst]
        if typ == "relied_on_by":
            if src in d.relied_on_by:
                return False
            d.relied_on_by.append(src)
            s.relies_on.append(dst)
        else:
            if dst in s.reverts:
                return False
            s.reverts.append(dst)
        self.edges.append({"type": typ, "src": src, "dst": dst, "idx": idx, **kw})
        return True

    def _link_reverts(self, step: Step, n: Node) -> None:
        old, new = step.attrs.get("edit_old"), step.attrs.get("edit_new")
        path = step.written_paths[0] if step.written_paths else None
        target = None
        if path and old is not None and new is not None and (old or new):
            ho, hn = _h(old), _h(new)
            for m in reversed(self.nodes):          # 逆编辑：new/old 恰好对调
                if m is not n and any(paths_match(path, p) and o == hn and nw == ho for p, o, nw in m._edits):
                    target = m
                    break
            n._edits.append((path, ho, hn))
        if target is None and step.attrs.get("undo"):
            for m in reversed(self.nodes):
                if m is n or not m.state_steps:
                    continue
                if not step.written_paths or any(paths_match(a, b) for a in step.written_paths for b in m.touched_paths):
                    target = m
                    break
        if target is not None:
            self._add_edge(n.id, target.id, "reverts", step.idx)

    def find_refs(self, step: Step) -> list[tuple[str, str, str]]:
        """本步引用了哪些更早的节点：[(node_id, kind, token)]。不含本步所在节点。"""
        cfg = self.rules.reference
        sources = set(cfg.get("sources", []))
        cur = self.step_node.get(step.idx)
        out: list[tuple[str, str, str]] = []
        seen = set()

        def add(nid, kind, tok):
            if nid != cur and (nid, kind) not in seen:
                seen.add((nid, kind))
                out.append((nid, kind, tok))

        def latest_writer(path: str) -> Optional[str]:
            for m in reversed(self.nodes):
                if m.id != cur and any(paths_match(path, p) for p in m.touched_paths):
                    return m.id
            return None

        text = (step.thought or "") + "\n" + (step.action or "")[:4000]
        if "thought_symbol" in sources and self._sym_index:
            minlen = int(cfg.get("min_symbol_len", 4))
            for tok in set(_IDENT.findall(text)):
                if len(tok) >= minlen and tok not in self.rules.stop_symbols and tok in self._sym_index:
                    for nid in reversed(self._sym_index[tok]):
                        if nid != cur:
                            add(nid, "symbol", tok)
                            break
        if "thought_path" in sources:
            for tok in set(_PATH_IN_TEXT.findall(step.thought or "")):
                base = tok.rsplit("/", 1)[-1]
                if base in self.rules.stop_basenames and "/" not in tok:
                    continue
                nid = latest_writer(tok.lstrip("./"))
                if nid:
                    add(nid, "path_mention", tok)
        if "write_path" in sources:
            for p in step.written_paths:
                nid = latest_writer(p)
                if nid:
                    add(nid, "path_write", p)
        return out

    def add_references(self, step: Step, ev: Event) -> dict:
        """把 REFERENCE 事件落成 relied_on_by 边。返回 {node_id: 被依赖数}。"""
        cur = self.step_node[step.idx]
        deps = {}
        for v in ev.attrs.get("via", []):
            self._add_edge(cur, v["node"], "relied_on_by", step.idx, kind=v["kind"], token=v["token"])
        for nid in ev.attrs.get("refs", []):
            deps[nid] = len(self.by_id[nid].relied_on_by)
        return deps

    # ------------------------------------------------------------ 序列化
    def to_dict(self) -> dict:
        return {"nodes": [n.to_dict() for n in self.nodes],
                "edges": self.edges,
                "current": self.current.id if self.current else None}
