"""统一 step —— 划分器所有下游部件的唯一输入格式。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


class ToolClass:
    EDIT = "EDIT"              # 编辑器类工具写文件（str_replace / create / insert / write）
    EXEC_WRITE = "EXEC_WRITE"  # shell 里改状态（sed -i、重定向、mv、git checkout ...）
    INSTALL = "INSTALL"        # 改环境（pip install ...）
    VERIFY = "VERIFY"          # 测试 / 复现脚本 / 构建 / lint
    READ = "READ"              # 读文件、看环境
    SEARCH = "SEARCH"          # grep / find
    EXEC = "EXEC"              # 认不出的执行，默认当探索
    BROWSE = "BROWSE"
    THINK = "THINK"
    PLAN = "PLAN"              # task_tracking
    MESSAGE = "MESSAGE"        # agent 对用户说话
    DELEGATE = "DELEGATE"
    FINISH = "FINISH"
    OTHER = "OTHER"

    STATE = {EDIT, EXEC_WRITE, INSTALL}


@dataclass
class Step:
    idx: int                       # 步号，与数据集标注的 step_id 对齐
    role: str = "agent"
    thought: str = ""
    action: str = ""               # 动作原文（命令 / 编辑内容的文本渲染）
    observation: str = ""
    tool_class: str = ToolClass.OTHER
    touched_paths: list[str] = field(default_factory=list)
    exit_code: Optional[int] = None
    tool: str = ""                 # 原始工具名（run / edit / read ...）
    written_paths: list[str] = field(default_factory=list)  # touched_paths 中被写的子集
    attrs: dict[str, Any] = field(default_factory=dict)     # 规则抽出的附加信息
    raw_ref: dict[str, Any] = field(default_factory=dict)   # 回读原文的指针（事件 id）

    def to_dict(self) -> dict:
        d = asdict(self)
        # edit 的 old/new 已经在 action 文本里，不重复落盘
        d["attrs"] = {k: v for k, v in self.attrs.items() if k not in ("edit_old", "edit_new")}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})
