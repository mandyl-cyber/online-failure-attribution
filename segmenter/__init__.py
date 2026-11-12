"""划分器（M1）：纯规则、零解码的在线状态树构建器。

转换器 -> 事件检测 -> 状态树 -> 切分 -> 候选 claim
各部件只通过文件格式耦合：steps.jsonl / events.jsonl / tree.json / stage_ranges.json / claims.jsonl
"""

from .schema import Step, ToolClass  # noqa: F401

__version__ = "0.1.0"
