"""转换器：OpenHands event stream（及其几种落盘形态）-> 统一 step。

支持的输入
  1. OpenHands 事件流：[{id, source, action, args, ...}, {id, cause, observation, content, extras}, ...]
     - 单个 .json（list，或 {"history"|"trajectory"|"events": [...]}）、.jsonl、或 events/ 目录（0.json, 1.json ...）
     - 旧格式 history: [[action, observation], ...] 也认
  2. OpenAI chat messages（SWE-rebench / OpenHands function calling 存储格式）：
     assistant.tool_calls + role=tool
  3. CodeTracer 归一化后的 steps.json：[{step_id, action, observation, thinking, tool_type}, ...]
     —— 目录里有它就优先用，保证步号与 CodeTraceBench 标注一致

在线用法：OpenHandsStreamAdapter().feed(event) -> [Step, ...]；一步 = 一个 agent 动作 + 它的观察。
"""
from __future__ import annotations

import json
import re
import hashlib
from copy import deepcopy
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Optional

from ..rules import Rules, default_rules
from ..schema import Step, ToolClass
from ..shell import classify_command, norm_path

# 不构成 step 的动作
SKIP_ACTIONS = {"system", "recall", "condensation", "condensation_request", "change_agent_state",
                "null", "pause", "resume", "initialize", "loop_recovery"}
# 没有对应观察、到了就可以直接出 step 的动作
NO_OBS_ACTIONS = {"message", "finish", "reject"}
VIEW_COMMANDS = {"view", "open", "goto", "scroll_down", "scroll_up", "find_file", "search_file", "search_dir"}
_EXIT_RE = re.compile(r"(?:exit code|exit_code)[\s:=]*(-?\d+)", re.I)
_DEF_RE = re.compile(r"(?m)^[ \t]*(?:async\s+)?(?:def|class|function|func|fn|interface|struct)\s+([A-Za-z_]\w*)")

# function calling 工具名 -> OpenHands 动作名
TOOL_NAME_MAP = {
    "execute_bash": "run", "bash": "run", "terminal": "run", "run": "run", "execute_command": "run", "shell": "run",
    "str_replace_editor": "edit", "file_editor": "edit", "str_replace_based_edit_tool": "edit",
    "edit_file": "edit", "edit": "edit", "write_file": "write", "create_file": "write",
    "read_file": "read", "view_file": "read",
    "execute_ipython_cell": "run_ipython", "ipython": "run_ipython",
    "think": "think", "finish": "finish", "submit": "finish",
    "browser": "browse_interactive", "web_read": "browse", "fetch": "browse",
    "task_tracker": "task_tracking", "delegate": "delegate",
}


def _s(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        if x and all(isinstance(p,dict) and p.get('type') in ('text','input_text','output_text')
                     and isinstance(p.get('text'),str) for p in x):
            return '\n'.join(p['text'] for p in x)
        return json.dumps(x,ensure_ascii=False)
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def _response_text(ev: dict) -> str:
    """args.thought 为空时，从 tool_call_metadata 里捞 assistant 的正文。"""
    try:
        msg = ev["tool_call_metadata"]["model_response"]["choices"][0]["message"]
        return _s(msg.get("content")) or _s(msg.get("reasoning_content"))
    except Exception:
        return ""


def _is_action(ev: dict) -> bool:
    return isinstance(ev.get("action"), str) and "observation" not in ev


def _is_observation(ev: dict) -> bool:
    return isinstance(ev.get("observation"), str)


def build_step(action_ev: dict, obs_ev: Optional[dict], idx: int, rules: Rules) -> Step:
    kind = action_ev.get("action", "")
    args = action_ev.get("args") or {}
    thought = _s(args.get("thought")) or _response_text(action_ev)
    st = Step(idx=idx, role=action_ev.get("source") or "agent", thought=thought, tool=kind)
    st.raw_ref = {"action_id": action_ev.get("id"), "observation_id": (obs_ev or {}).get("id")}
    a = st.attrs
    # Keep public arguments alongside the human-readable action rendering.
    # This is evidence, not instructions or a semantic success assertion.
    a['public_action_args'] = deepcopy({k:v for k,v in args.items()
                                       if k not in ('thought','_handoff_before')})
    if kind == 'run':
        a['is_input'] = args.get('is_input')
        a['terminal_session'] = args.get('session_id', args.get('terminal_id', 'default'))
    if args.get("_handoff_before"):
        a["handoff_before"] = True

    if kind == "run":
        cmd = _s(args.get("command"))
        st.action = cmd
        info = classify_command(cmd, rules)
        st.tool_class = info.tool_class
        st.touched_paths, st.written_paths = info.paths, info.written
        a["cmd_classes"] = sorted(info.classes)
        if info.irreversible:
            a["irreversible"] = info.irreversible
        if info.verify_kind:
            a["verify_kind"], a["verify_cmd"] = info.verify_kind, info.verify_cmd
        if info.undo:
            a["undo"] = True
        if any(p.search(cmd.strip()) for p in rules.submit_commands):
            a["submit"] = True
    elif kind == "run_ipython":
        code = _s(args.get("code"))
        st.action = code
        if rules.sh_ipython_write.search(code):
            st.tool_class = ToolClass.EXEC_WRITE
            a["cmd_classes"] = ["state"]
        else:
            st.tool_class = ToolClass.EXEC
            a["cmd_classes"] = ["probe"]
    elif kind in ("edit", "write"):
        path = norm_path(_s(args.get("path")))
        sub = _s(args.get("command")) or ("create" if kind == "write" else "edit")
        old = _s(args.get("old_str"))
        new = _s(args.get("new_str")) or _s(args.get("file_text")) or _s(args.get("content"))
        st.touched_paths = [path] if path else []
        if sub in VIEW_COMMANDS:
            st.tool_class = ToolClass.READ
            st.action = f"{sub} {path}" + (f" {args.get('view_range')}" if args.get("view_range") else "")
        else:
            st.tool_class = ToolClass.EDIT
            st.written_paths = list(st.touched_paths)
            st.action = f"{sub} {path}" + (f"\n--- old\n{old}" if old else "") + (f"\n+++ new\n{new}" if new else "")
            a["edit_cmd"] = sub
            a["edit_old"], a["edit_new"] = old, new
            a["symbols"] = sorted(set(_DEF_RE.findall(old) + _DEF_RE.findall(new)))
            if sub == "undo_edit":
                a["undo"] = True
    elif kind == "read":
        path = norm_path(_s(args.get("path")))
        st.tool_class, st.action, st.touched_paths = ToolClass.READ, f"read {path}", [path] if path else []
    elif kind in ("browse", "browse_interactive"):
        st.tool_class = ToolClass.BROWSE
        st.action = _s(args.get("url")) or _s(args.get("browser_actions"))
    elif kind == "think":
        st.tool_class = ToolClass.THINK
        st.action = ""
    elif kind == "message":
        st.tool_class = ToolClass.MESSAGE
        st.action = _s(args.get("content")) or _s(action_ev.get("message"))
        a["wait_for_response"] = bool(args.get("wait_for_response"))
    elif kind in ("finish", "reject"):
        st.tool_class = ToolClass.FINISH
        st.action = (_s(args.get("final_thought")) or _s(args.get("message")) or _s(args.get("outputs"))
                     or _s(action_ev.get("message")))
    elif kind == "delegate":
        st.tool_class = ToolClass.DELEGATE
        st.action = f"delegate {_s(args.get('agent'))}: {_s(args.get('inputs'))}"
    elif kind == "task_tracking":
        st.tool_class = ToolClass.PLAN
        st.action = _s({k:v for k,v in args.items() if k not in ('thought','_handoff_before')})
    else:
        st.tool_class = ToolClass.EXEC if kind in ("mcp", "call_tool_mcp") else ToolClass.OTHER
        st.action = _s(args) if args else _s(action_ev.get("message"))

    if obs_ev is not None:
        st.observation = _s(obs_ev.get("content"))
        ex = obs_ev.get("extras") or {}
        code = (ex.get("metadata") or {}).get("exit_code", ex.get("exit_code")) if isinstance(ex, dict) else None
        if isinstance(code, int) and code != -1:   # OpenHands 用 -1 表示未知 / 超时
            st.exit_code = code
        if obs_ev.get("observation") == "error":
            a["error_observation"] = True
    if st.exit_code is None and st.tool in ("run", "run_ipython") and st.observation:
        m = _EXIT_RE.search(st.observation[-300:])
        if m and m.group(1) != "-1":
            st.exit_code = int(m.group(1))
    return st


class OpenHandsStreamAdapter:
    """在线转换器：逐个喂事件，按动作顺序吐出完整的 step。"""

    def __init__(self, rules: Rules | None = None, base: int = 1, include_user: bool = False,
                 fold_agent_messages: bool = False):
        """fold_agent_messages: agent 的纯文字消息不单独成步（CodeTraceBench 的步号不数它们），
        正文并进下一步的 thought，并给下一步打 handoff_before。在线使用时保持 False。"""
        self.rules = rules or default_rules()
        self.next_idx = base
        self.include_user = include_user
        self.fold_agent_messages = fold_agent_messages
        self.folded_messages = 0
        self._carry: Optional[str] = None
        self.task: str = ""
        self._pending: "OrderedDict[Any, list]" = OrderedDict()   # action id -> [action_ev, obs_ev, done]
        self._auto = 0
        self._seen = {}
        self._sequence = 0
        self._arrivals = {}
        self.diagnostics = []

    def _emit_ready(self, force: bool = False) -> list[Step]:
        out = []
        while self._pending:
            key, (act, obs, done) = next(iter(self._pending.items()))
            if not (done or force):
                break
            self._pending.popitem(last=False)
            step=build_step(act, obs, self.next_idx, self.rules)
            arrival=self._arrivals.pop(key)
            flags=[]
            if obs is None and act.get('action') not in NO_OBS_ACTIONS:
                flags.append('missing_observation_at_flush')
            if obs is not None:
                extras=obs.get('extras') or {}
                if obs.get('truncated') is True or (isinstance(extras,dict) and extras.get('truncated') is True):
                    flags.append('upstream_truncation_declared')
                if re.search(r'(?i)(output (?:has been )?truncated|\[.*?truncated.*?\])',step.observation):
                    flags.append('possible_truncation_marker')
            step.attrs['input_integrity']=dict(version=2,action_received_seq=arrival['action'],
                observation_received_seq=arrival.get('observation'),available_at_seq=self._sequence,
                sequence_basis='adapter feed order, not native time or step id',
                pairing=arrival.get('pairing','not_required' if act.get('action') in NO_OBS_ACTIONS else 'missing'),
                issues=flags,observation_complete=False if flags else None,
                action_event_sha256=arrival['action_hash'],observation_event_sha256=arrival.get('observation_hash'))
            out.append(step)
            self.next_idx += 1
        return out

    def feed(self, ev: dict) -> list[Step]:
        if not isinstance(ev, dict):
            return []
        self._sequence+=1
        digest=hashlib.sha256(json.dumps(ev,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        event_id=ev.get('id')
        if event_id is not None:
            identity=(type(event_id).__name__,str(event_id))
            if identity in self._seen:
                if self._seen[identity]!=digest:
                    self.diagnostics.append(dict(kind='conflicting_duplicate',event_id=event_id,received_seq=self._sequence))
                    raise ValueError('Conflicting duplicate event ID: '+str(event_id))
                self.diagnostics.append(dict(kind='duplicate_event',event_id=event_id,received_seq=self._sequence))
                return []
            self._seen[identity]=digest
        ev=deepcopy(ev)
        if _is_action(ev):
            kind, src = ev["action"], ev.get("source")
            if kind in SKIP_ACTIONS:
                return []
            if src == "user" or (src is None and kind == "message" and not self.task):
                if kind == "message" and not self.task:
                    self.task = _s((ev.get("args") or {}).get("content")) or _s(ev.get("message"))
                if not self.include_user:
                    return []
            if self.fold_agent_messages and src != "user":
                if kind == "message":
                    self.folded_messages += 1
                    if self._carry is None:          # 连续多条只留第一条
                        self._carry = (_s((ev.get("args") or {}).get("content")) or _s(ev.get("message")))[:2000]
                    return []
                if self._carry is not None:
                    args = dict(ev.get("args") or {})
                    args["thought"] = (self._carry + "\n" + _s(args.get("thought"))).strip()
                    args["_handoff_before"] = True
                    ev = dict(ev, args=args)
                    self._carry = None
            key = ev.get("id")
            if key is None:
                self._auto += 1
                key = ('adapter_auto', self._auto)
            self._pending[key] = [ev, None, kind in NO_OBS_ACTIONS]
            self._arrivals[key]=dict(action=self._sequence,action_hash=digest)
            return self._emit_ready()
        if _is_observation(ev):
            if ev.get('observation') in ('agent_state_changed', 'null', 'recall'):
                return []  # lifecycle notification, never a tool result
            cause = ev.get("cause")
            slot = self._pending.get(cause)
            pair_key=cause
            pairing='exact_cause'
            if slot is None and cause is None:   # 没有 cause：配给最早一个还没观察的动作
                candidates=[(k,s) for k,s in self._pending.items() if not s[2]]
                if len(candidates)>1:
                    self.diagnostics.append(dict(kind='ambiguous_observation',event_id=event_id,received_seq=self._sequence))
                    raise ValueError('Ambiguous observation without cause; multiple pending actions')
                if candidates:
                    pair_key,slot=candidates[0];pairing='single_pending_fallback'
            if slot is None:
                self.diagnostics.append(dict(kind='unmatched_observation',event_id=event_id,cause=cause,received_seq=self._sequence))
                return []                        # 环境自发的观察（agent_state_changed 等）
            if slot[2]:
                self.diagnostics.append(dict(kind='multiple_observations',event_id=event_id,cause=cause,received_seq=self._sequence))
                raise ValueError('Multiple observations for one action require an explicit streaming adapter')
            slot[1], slot[2] = ev, True
            self._arrivals[pair_key].update(observation=self._sequence,observation_hash=digest,pairing=pairing)
            return self._emit_ready()
        return []

    def flush(self) -> list[Step]:
        return self._emit_ready(force=True)


# ---------------------------------------------------------------- 批量 / 落盘形态

def messages_to_events(messages: list[dict], fold_text_turns: bool = False) -> list[dict]:
    """OpenAI chat messages -> 合成的 OpenHands 事件。

    fold_text_turns: 没有工具调用的 assistant 回合不单独成步（CodeTraceBench 的步号只数"工具调用 + 结果"），
    它的正文并进下一步的 thought，并给下一步打上 handoff_before（agent 曾经想交卷、被要求继续）。
    """
    evs, n = [], 0
    carry = None
    for m in messages:
        role = m.get("role")
        if role == "user":
            evs.append({"id": f"m{n}", "source": "user", "action": "message", "args": {"content": _s(m.get("content"))}})
        elif role == "assistant":
            text = _s(m.get("content"))
            calls = m.get("tool_calls") or []
            if not calls:
                if fold_text_turns:
                    carry = (carry + "\n" if carry else "") + text
                else:
                    evs.append({"id": f"m{n}", "source": "agent", "action": "message", "args": {"content": text}})
            for j, tc in enumerate(calls):
                fn = tc.get("function") or tc
                try:
                    args = json.loads(fn.get("arguments") or "{}") if isinstance(fn.get("arguments"), str) else dict(fn.get("arguments") or {})
                except Exception:
                    args = {"command": _s(fn.get("arguments"))}
                name = fn.get("name", "")
                kind = TOOL_NAME_MAP.get(name, name)
                if kind == "run_ipython" and "code" not in args:
                    args["code"] = _s(args.get("command"))
                if kind == "edit" and args.get("command") in VIEW_COMMANDS:
                    pass  # build_step 会把 view 归成 READ
                if j == 0 and text and not args.get("thought"):
                    args["thought"] = text
                if j == 0 and carry is not None:
                    args["thought"] = (carry + "\n" + _s(args.get("thought"))).strip()
                    args["_handoff_before"] = True
                    carry = None
                evs.append({"id": tc.get("id") or f"m{n}.{j}", "source": "agent", "action": kind, "args": args})
        elif role == "tool":
            evs.append({"id": f"m{n}", "cause": m.get("tool_call_id"), "observation": "run", "content": _s(m.get("content"))})
        n += 1
    return evs


def steps_json_to_steps(records: list[dict], rules: Rules) -> list[Step]:
    """CodeTracer StepRecord -> 统一 step（保留原 step_id）。"""
    out = []
    for i, r in enumerate(records):
        act = r.get("action")
        args, kind = None, "run"
        if isinstance(act, str) and act.lstrip().startswith("{"):
            try:
                act = json.loads(act)
            except Exception:
                pass
        if isinstance(act, dict):
            if isinstance(act.get("action"), str) and isinstance(act.get("args"), dict):
                kind, args = act["action"], act["args"]
            else:
                name = act.get("name") or act.get("tool") or (act.get("function") or {}).get("name") or ""
                args = act.get("arguments") or act.get("args") or (act.get("function") or {}).get("arguments") or act
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"command": args}
                kind = TOOL_NAME_MAP.get(name) or ("edit" if "path" in args and "command" in args else "run")
        else:
            args = {"command": _s(act)}
            tt = (r.get("tool_type") or "").lower()
            if tt in ("think", "message", "finish"):
                kind, args = tt, {"content": _s(act), "thought": _s(act)}
        args = dict(args)
        args.setdefault("thought", _s(r.get("thinking") or r.get("thought")))
        ev = {"id": r.get("step_id", i), "source": "agent", "action": kind, "args": args}
        obs = {"observation": "run", "content": _s(r.get("observation"))} if r.get("observation") is not None else None
        st = build_step(ev, obs, int(r.get("step_id", i)), rules)
        if r.get("tool_type"):
            st.attrs["src_tool_type"] = r["tool_type"]
        out.append(st)
    return out


def convert_events(events: Iterable[dict], rules: Rules | None = None, base: int = 1,
                   include_user: bool = False, fold_agent_messages: bool = False) -> tuple[list[Step], dict]:
    ad = OpenHandsStreamAdapter(rules, base=base, include_user=include_user, fold_agent_messages=fold_agent_messages)
    steps: list[Step] = []
    for ev in events:
        steps += ad.feed(ev)
    steps += ad.flush()
    return steps, {"task": ad.task, "folded_messages": ad.folded_messages}


def _flatten(xs: list) -> list:
    out = []
    for x in xs:
        if isinstance(x, (list, tuple)):
            out += _flatten(list(x))
        elif isinstance(x, dict):
            out.append(x)
    return out


def _load_json_any(p: Path) -> Any:
    text = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    return json.loads(text)


def _detect(obj: Any) -> tuple[str, list]:
    """返回 (格式名, 记录列表)。格式名：events / messages / steps / unknown"""
    if isinstance(obj, dict):
        for k in ("history", "trajectory", "events", "messages", "steps"):
            if isinstance(obj.get(k), list):
                return _detect(obj[k])
        return "unknown", []
    if not isinstance(obj, list) or not obj:
        return "unknown", []
    flat = _flatten(obj)
    if not flat:
        return "unknown", []
    head = flat[:50]
    if any(_is_action(e) or _is_observation(e) for e in head) and any("args" in e or "content" in e or "extras" in e for e in head):
        return "events", flat
    if all("role" in e for e in head):
        return "messages", flat
    if any("step_id" in e for e in head) and any("action" in e for e in head):
        return "steps", flat
    return "unknown", flat


_COMPLETION_NAME = re.compile(r"-(\d+\.\d+)\.json$")


def _completion_files(d: Path) -> list[tuple[float, Path]]:
    out = []
    for p in d.iterdir():
        m = _COMPLETION_NAME.search(p.name)
        if m and p.is_file():
            out.append((float(m.group(1)), p))
    return sorted(out)


def load_completion_logs(d: Path, expect_steps: Optional[int] = None) -> tuple[list[dict], dict]:
    """OpenHands 的 LLM completion 日志目录（CodeTraceBench 的 swe_raw/openhands__*）-> chat messages。

    每次 LLM 调用落一个 <model>-<时间戳>.json：{messages: 到这次调用为止的全部对话, response: 这次的回复}。
    同一个任务目录里可能混着两次运行（文件数约为步数的两倍），按"第一个工具调用的 id"分开；
    每次运行取 messages 最长的那个文件，再接上它的 response 就是完整轨迹。
    多次运行时：优先选工具调用数等于 expect_steps（manifest 的 step_count）的那次，否则选最后结束的。
    """
    # 日志是累积的（后一次调用包含前面全部对话），所以每次运行里最大的文件就是它的终态。
    # 按文件大小从大到小读：命中 expect_steps 立即停；没给 expect_steps 时连续 3 个文件都属于已见过的运行就停。
    # 这样每条轨迹只读几个文件，而不是全部几十上百个。
    files = sorted(_completion_files(d), key=lambda x: -x[1].stat().st_size)
    runs: dict[str, dict] = {}
    repeats = 0
    for ts, p in files:
        try:
            j = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            msgs = j["messages"]
            resp = j["response"]["choices"][0]["message"]
        except Exception:
            continue
        first = next((m["tool_calls"][0].get("id") for m in msgs
                      if m.get("role") == "assistant" and m.get("tool_calls")), None)
        if first is None:
            first = ((resp.get("tool_calls") or [{}])[0]).get("id") or f"notool:{p.name}"
        n_tool = sum(1 for m in msgs if m.get("role") == "tool")
        r = runs.get(first)
        if r is None:
            runs[first] = {"n_tool": n_tool, "msgs": msgs, "resp": resp, "file": p, "last_ts": ts}
            repeats = 0
            if expect_steps is not None and n_tool == expect_steps:
                break
        else:
            repeats += 1
            if n_tool > r["n_tool"]:
                r.update(n_tool=n_tool, msgs=msgs, resp=resp, file=p, last_ts=ts)
            if repeats >= 3 and expect_steps is None:   # 有 expect_steps 而还没命中：继续找，直到读完
                break
    runs = {k: r for k, r in runs.items() if "msgs" in r}
    if not runs:
        raise ValueError(f"{d} 里没有可用的 completion 日志")
    cands = sorted(runs.values(), key=lambda r: r["last_ts"])
    pick = next((r for r in reversed(cands) if expect_steps is not None and r["n_tool"] == expect_steps), cands[-1])
    messages = list(pick["msgs"]) + [dict(pick["resp"], role="assistant")]
    return messages, {"n_core": pick["n_tool"], "n_runs": len(runs), "source": str(pick["file"])}


def find_trajectory_file(d: Path) -> Optional[Path]:
    """在解包后的轨迹目录里找轨迹文件：steps.json > events/ > 最像事件流的 json。"""
    for p in sorted(d.rglob("steps.json")):
        return p
    if (d / "events.jsonl").is_file():           # bench._extract_stream 合并出来的事件流
        return d / "events.jsonl"
    for ev_dir in sorted(x for x in d.rglob("events") if x.is_dir()):
        if any(ev_dir.glob("*.json")):
            return ev_dir
    best, best_n = None, 0
    for p in sorted(list(d.rglob("*.json")) + list(d.rglob("*.jsonl"))):
        try:
            if p.stat().st_size > 200 * 1024 * 1024:
                continue
            fmt, recs = _detect(_load_json_any(p))
        except Exception:
            continue
        if fmt != "unknown" and len(recs) > best_n:
            best, best_n = p, len(recs)
    return best


def _completion_dir(p: Path) -> Optional[Path]:
    """p 或它下面（解包后的 artifact）是否是 completion 日志目录。"""
    if not p.is_dir():
        return None
    if _completion_files(p):
        return p
    for sub in sorted(x for x in p.rglob("*") if x.is_dir()):
        if sub.name != "events" and _completion_files(sub):
            return sub
    return None


def load_trajectory(path: str | Path, rules: Rules | None = None, base: int = 1,
                    include_user: bool = False, expect_steps: Optional[int] = None) -> tuple[list[Step], dict]:
    """读一条落盘轨迹，返回 (steps, meta)。

    meta: {format, source, task, native_ids, n_core}
    n_core = 与数据集标注步号对齐的步数（completion 日志格式下，末尾的 finish 是多出来的一步，不在标注范围内）。
    """
    rules = rules or default_rules()
    p = Path(path)
    cdir = None
    if p.is_dir() and p.name != "events" and not any(x.is_dir() for x in p.rglob("events")):
        cdir = _completion_dir(p)
    if cdir is not None:
        messages, m = load_completion_logs(cdir, expect_steps)
        steps, m2 = convert_events(messages_to_events(messages, fold_text_turns=True), rules, base=base)
        meta = {"format": "completions", "native_ids": False, **m, **m2}
        return steps, meta
    if p.is_dir() and p.name != "events":
        f = find_trajectory_file(p)
        if f is None:
            raise FileNotFoundError(f"{p} 下没有找到可识别的轨迹文件")
        p = f
    if p.is_dir():   # events/ 目录：一个事件一个文件
        files = sorted(p.glob("*.json"), key=lambda x: (int(x.stem) if x.stem.isdigit() else 1 << 60, x.name))
        obj: Any = [json.loads(x.read_text(encoding="utf-8", errors="replace")) for x in files]
    else:
        obj = _load_json_any(p)
    fmt, recs = _detect(obj)
    meta = {"format": fmt, "source": str(p), "task": "", "native_ids": fmt == "steps"}
    if fmt == "steps":
        steps = steps_json_to_steps(recs, rules)
        meta["n_core"] = len(steps)
        return steps, meta
    if fmt == "messages":
        recs = messages_to_events(recs)
    elif fmt != "events":
        raise ValueError(f"认不出的轨迹格式：{p}（顶层类型 {type(obj).__name__}）")
    steps, m = convert_events(recs, rules, base=base, include_user=include_user)
    if expect_steps is not None and len(steps) != expect_steps and fmt == "events":
        # 步数对不上：CodeTraceBench 的标注不把 agent 的纯文字消息算作一步，折叠后再试
        steps2, m2 = convert_events(recs, rules, base=base, include_user=include_user, fold_agent_messages=True)
        if len(steps2) == expect_steps:
            steps, m = steps2, m2
    meta.update(m)
    meta["n_core"] = len(steps)
    return steps, meta
