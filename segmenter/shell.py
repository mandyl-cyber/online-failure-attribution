"""shell 命令分类：一行命令 -> 子命令 -> 类别集合 / tool_class / 触及的路径。纯正则。"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Optional

from .rules import Rules
from .schema import ToolClass

_HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2\b", re.S)
_HEREDOC_OPEN = re.compile(r"<<-?\s*['\"]?\w+['\"]?[^\n]*\n.*", re.S)
_ASSIGN = re.compile(r"^[A-Za-z_]\w*=(?:\"[^\"]*\"|'[^']*'|\S*)\s+")
_QUOTED = re.compile(r"'[^']*'|\"(?:\\.|[^\"\\])*\"")
_REDIRECT = re.compile(r"(?<![\d&<>])(?:>>?|&>)\s*([^\s&|;<>()]+)")
_PATHLIKE = re.compile(r"^[\w.@~+\-/*]+$")
_ABS_HEAD = re.compile(r"^/(?:[\w.@+\-]+/)+(?=[\w.@+\-]+(\s|$))")
_GIT_C = re.compile(r"^git\s+(?:-C\s+\S+\s+|--no-pager\s+|-c\s+\S+\s+)+")
_PATCH_FILE = re.compile(r"(?m)^\*\*\* (?:Update|Add|Delete) File: (\S+)")
_SCRIPT_FIRST = {"sed", "awk", "perl", "grep", "egrep", "fgrep", "rg", "ag", "ack"}
_EXT = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,5}$")


@dataclass
class ShellInfo:
    classes: set = field(default_factory=set)   # irreversible / verify / install / state / env / search / read / exec
    tool_class: str = ToolClass.EXEC
    irreversible: Optional[str] = None
    verify_kind: Optional[str] = None           # test / repro / build
    verify_cmd: Optional[str] = None
    undo: bool = False
    paths: list = field(default_factory=list)
    written: list = field(default_factory=list)


def split_segments(cmd: str) -> list[str]:
    """按 && || ; | 换行 切开，尊重引号；heredoc 正文先剥掉。"""
    cmd = _HEREDOC.sub(" <<HEREDOC\n", cmd)
    cmd = _HEREDOC_OPEN.sub(" <<HEREDOC", cmd)
    segs, buf, q, i, n = [], [], None, 0, len(cmd)
    while i < n:
        c = cmd[i]
        if q:
            buf.append(c)
            if c == "\\" and q == '"' and i + 1 < n:
                buf.append(cmd[i + 1]); i += 1
            elif c == q:
                q = None
        elif c in "'\"":
            q = c; buf.append(c)
        elif c == "\\" and i + 1 < n:
            if cmd[i + 1] == "\n":
                buf.append(" ")
            else:
                buf.append(c); buf.append(cmd[i + 1])
            i += 1
        elif c in ";\n" or (c == "&" and cmd[i:i + 2] == "&&") or c == "|":
            if c == "|" and buf and buf[-1] == ">":      # >| clobber
                buf.append(c)
            else:
                segs.append("".join(buf)); buf = []
                if cmd[i:i + 2] in ("&&", "||", "|&"):
                    i += 1
        else:
            buf.append(c)
        i += 1
    segs.append("".join(buf))
    return [s for s in (x.strip().lstrip("({ ").strip() for x in segs) if s]


def _strip_prefix(seg: str, rules: Rules) -> str:
    prev = None
    while prev != seg:
        prev = seg
        seg = _ASSIGN.sub("", seg, count=1)
        seg = rules.sh_wrappers.sub("", seg, count=1)
    seg = _ABS_HEAD.sub("", seg, count=1)          # /opt/venv/bin/pip -> pip
    seg = _GIT_C.sub("git ", seg, count=1)         # git -C <dir> status -> git status
    return seg.strip()


def _tokens(seg: str) -> list[str]:
    try:
        return shlex.split(seg, posix=True)
    except ValueError:
        return seg.split()


def norm_path(p: str) -> str:
    p = p.strip().strip("'\"`,;:()")
    p = p.split("::")[0]
    while p.startswith("./"):
        p = p[2:]
    return p.rstrip("/") if len(p) > 1 else p


def path_tokens(seg: str) -> list[str]:
    out = []
    toks = _tokens(seg)
    head = toks[0].rsplit("/", 1)[-1] if toks else ""
    if head in ("echo", "printf"):
        return out
    skip_script = head in _SCRIPT_FIRST          # 第一个非选项参数是脚本 / 模式，不是路径
    for t in toks[1:]:
        if t.startswith("-"):
            continue
        if skip_script:
            skip_script = False
            continue
        if t.startswith("-") or "://" in t or "=" in t or not _PATHLIKE.match(t):
            continue
        t = norm_path(t)
        if t and ("/" in t or _EXT.search(t)) and t not in (".", "..", "/"):
            out.append(t)
    return out


def _any(pats, s) -> bool:
    return any(p.search(s) for p in pats)


def classify_command(cmd: str, rules: Rules) -> ShellInfo:
    info = ShellInfo()
    cmd = cmd or ""
    if rules.sh_noop.match(cmd.strip()):
        info.classes.add("noop")
        info.tool_class = ToolClass.OTHER
        return info
    has_heredoc = "<<" in cmd
    for raw in split_segments(cmd):
        seg = _strip_prefix(raw, rules)
        if not seg:
            continue
        paths = path_tokens(seg)
        unq = _QUOTED.sub('""', seg)
        redirects = [norm_path(m.group(1)) for m in _REDIRECT.finditer(unq)
                     if not rules.sh_redirect_exempt.search(m.group(1))]
        cls = None
        for p in rules.sh_irreversible:
            if p.search(seg) and not (seg.startswith("rm") and rules.sh_irreversible_exempt.search(seg)):
                info.irreversible = info.irreversible or seg[:200]
                cls = "state"
                info.classes.add("irreversible")
                break
        if cls is None:
            for kind, pats in rules.sh_verify.items():
                if _any(pats, seg):
                    cls = "verify"
                    # test > repro > build：一行里多条验证命令取最强的
                    rank = {"test": 3, "repro": 2, "build": 1}
                    if rank.get(kind, 0) > rank.get(info.verify_kind or "", 0):
                        info.verify_kind, info.verify_cmd = kind, seg[:200]
                    break
        if cls is None and _any(rules.sh_install, seg):
            cls = "install"
        if cls is None and _any(rules.sh_state, seg):
            cls = "state"
        if cls is None:
            if _any(rules.sh_env, seg):
                cls = "env"
            elif _any(rules.sh_probe, seg):
                cls = "probe"
                if has_heredoc and "<<" in seg and rules.sh_ipython_write.search(cmd):
                    info.classes.add("state")          # 内联脚本里写文件
            elif _any(rules.sh_search, seg):
                cls = "search"
            elif _any(rules.sh_read, seg):
                cls = "read"
            else:
                cls = "exec"
        if cls == "state":
            if seg.startswith(("applypatch", "apply_patch")):
                paths = paths + [norm_path(x) for x in _PATCH_FILE.findall(cmd)]
            info.written += paths
            if _any(rules.sh_undo, seg):
                info.undo = True
        if redirects:
            info.classes.add("state")
            info.written += redirects
            paths = paths + redirects
        info.classes.add(cls)
        info.paths += paths

    c = info.classes
    if "state" in c:
        info.tool_class = ToolClass.EXEC_WRITE
    elif "install" in c:
        info.tool_class = ToolClass.INSTALL
    elif "verify" in c:
        info.tool_class = ToolClass.VERIFY
    elif "probe" in c or "exec" in c:
        info.tool_class = rules.raw["shell"].get("unknown_exec_as", ToolClass.EXEC)
    elif "search" in c:
        info.tool_class = ToolClass.SEARCH
    elif "read" in c or "env" in c:
        info.tool_class = ToolClass.READ
    else:
        info.tool_class = rules.raw["shell"].get("unknown_exec_as", ToolClass.EXEC)
    info.paths = list(dict.fromkeys(info.paths))
    info.written = list(dict.fromkeys(info.written))
    return info
