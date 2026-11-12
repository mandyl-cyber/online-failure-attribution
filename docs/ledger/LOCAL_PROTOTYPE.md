# 本地增量账本原型

2026-11-13。实现：`segmenter/ledger.py`；真实数据回放：`segmenter/replay_ledger.py`。无 API、无新增训练、无语义关系推断；默认 OnlineSegmenter 和旧输入缓存未改动。

## 已实现

- `repair_step` 返回独立 public-plan-v2 输入，不修改原 Step；按当前原生调用 ID 恢复 command/task_list 的 id/title/status。查询动作仅保留 command；thought 不进入模型可读证据。
- `Ledger.feed` 按已完成步骤立即记录完整动作、观察和规则事件，不等阶段结束。记录公开计划、状态变化候选、验证和结束自述；尚不抽取语义假设或未解决事项。
- 每个操作保存为单独原子替换的 journal 文件，写入前 flush/fsync；重启从已提交日志重建事件检测状态。临时文件不是已提交记录。限定单写者，不承诺网络文件系统或断电下的目录元数据持久性。
- 同步号同内容幂等，冲突内容/乱序步骤报错。当前身份依赖经过审计的稳定步号，尚未实现在线乱序原生事件去重和动作—观察缓冲恢复。
- Evidence 保存完整适配器字段文本、字段名、SHA256、原始引用和可用步骤顺序。`get_evidence` 校验摘要，支持前缀 cutoff，拒绝不存在或未来证据。
- `assign_stages` 记录已观察步骤的完整归属快照及修订，不改变账本项 ID。此次使用人工区间，明确标为 gold_oracle；尚未实现在线边界事件接线和开放阶段生命周期。
- 验证项记录退出码及规则结果，但对象版本和环境未知，不支持“补丁正确”的结论。状态变化是 inferred，不冒充读取过文件内容。`edges.jsonl` 当前为空。

每条轨迹有 `run.json`、`journal/`、`evidence.jsonl`、`entries.jsonl`、`edges.jsonl`、`stage_snapshots.json`、`provenance.json` 和 `source.bin`。journal 是权威记录，其余视图可重建。

`source.bin` 是原始日志审计副本，可能含 thought 和未来记录；不提供给模型可读的 `get_evidence`。未来 API 客户端只能通过白名单接口读取证据，不能把整个输出目录直接上传。Evidence 中的计划 action 是恢复后的公开字段表示，已显式标注，不冒充原始字节。

## 运行与验证

```powershell
python tests/test_ledger.py
python tests/test_plan_payloads.py
python -m segmenter.replay_ledger --out eval/ledger_local_v1_new
```

回放固定开发清单，先验证原文哈希、原始步号以及旧 action/observation 指纹，再生成新版本；不使用经过裁剪和路径归一化的编码器文本缓存。completion 日志使用已审计的原始文件，按原规则折叠纯文字回合；该折叠仅限离线数据对齐。

最终产物目录：`eval/ledger_local_v1_final/`，以其中 `report.json` 的 status/totals 为准。最初调试目录 `ledger_local_v1`、`ledger_local_v1_checked` 不完整；`ledger_local_v1_complete` 为查询白名单补强前的回放，均不作为最终结果。脚本拒绝覆盖已有目录。

测试覆盖计划白名单与调用隔离、重复/冲突步骤、连续与重启执行等价、前缀证据隔离、跨阶段重分配不改变条目 ID、证据篡改检测。真实开发回放还逐条检查恢复后的操作记录相等，并回查全部证据。

## 与完整协议的差距

这是接口的可运行子集，不是完整 M4：尚缺任务原文/验收约束抽取、语义假设与结论整理、对象内容版本、支持/依赖边、异步整理队列、在线 StageUpdate 接线、检索预算及 API 归因器。当前直接构造 Step 的调用者必须提供已完成且按序的步骤，不允许把 oracle 运行称为真实在线部署。

下一步先接任务原文与对象/环境证据，再实现有出处的关系候选和归因 dry-run 输入包；在这些信息不足前，不为追求图规模自动补边。
