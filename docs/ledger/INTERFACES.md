# 在线阶段、账本与归因接口 v0.1

日期：2026-11-12。状态：待实现的接口约定；不是现有运行时已支持的功能。

## 范围与兼容

第一版覆盖代码修改、构建/环境配置和终端任务。在线只消费已经到达的动作与观察；首版归因在执行结束后调用。现有 `segmenter/schema.py:Step`、`stage_ranges.json`、`tree.json`、`claims.jsonl` 保持兼容，新协议放在独立的 `ledger_v1/` 输出目录。旧状态树节点不是 stage，也不是 claim，不能直接改名作为新账本。

标准文件：`run.json`、`evidence.jsonl`、`stage_updates.jsonl`、`entries.jsonl`、`edges.jsonl`、`attribution.json`。前五种事件记录采用追加语义，快照只是可重建的视图。

## 通用身份与时间

- 每条记录带 `schema_version`、`run_id`、`record_id`、`revision`、`observed_at_seq`、`producer`。producer 含实现/模型版本、输入协议版本、配置哈希。
- `seq` 为本次轨迹内单调递增的处理顺序；原始 `source_step_id` 独立保留，不强制改变原数据的 0/1 起点。
- CodeTraceBench 的动作—观察步骤与 LongRCA 的消息记录保持不同 `step_unit`，不能隐式混合或拆分。
- `record_id` 从轨迹、原生事件 ID 和记录类型稳定生成，不包含阶段编号。修订增加 revision，不能覆盖历史。
- 重复源事件及相同内容幂等；同 ID 不同内容报冲突。缺少原生 ID 时按原文件定位和内容摘要建立映射，不能仅按内容去重重复执行。
- 每个派生记录所有输入的到达顺序必须不晚于 `observed_at_seq`。来源时间、事件生效步骤和实际确认时间分别记录。

## 对象与必需字段

| 对象 | 字段 | 约束 |
| --- | --- | --- |
| Run / TaskProfile | dataset、trajectory_id、step_unit、input_version、task_text_refs、deliverables、success_contract、profile_source | 任务画像只用当时可见说明；未知用 null；每个语义字段可回查来源 |
| Evidence | source_step_id、raw_locator、content_sha256、field、span、availability_seq | raw_locator 指向保留的不可变原文副本；span 为解码后字段的 Unicode 字符区间；文件位置与内容摘要共同核验 |
| StageUpdate | stage_id、start_seq、end_seq、status、boundary_seq、detected_at_seq、activity_type、objective、evidence_ids、origin | 区间闭区间；开放阶段 end_seq 为当前已观察末尾；类型/目标可未知；origin 区分 gold_oracle/rule/model |
| LedgerEntry | entry_id、kind、text、source_step_ids、evidence_ids、artifact_refs、epistemic_status、adoption_status | kind 为 hypothesis/decision/state_change/verification/conclusion/open_issue；语义不支持时不生成 |
| Membership | entry_id、stage_id、revision、observed_at_seq | 阶段归属为独立可修订视图；跨阶段条目可以多归属；不能改 entry_id |
| ArtifactRef | object_id、path、version_id、version_basis、environment_id | 内容未获取时版本可未知；“第几次编辑事件”只是事件版本，不冒充真实文件哈希 |
| Edge | edge_id、relation、src_entry_id、dst_entry_id、evidence_ids、status、scope、producer、observed_at_seq | 只连接存在的账本项；status 为 candidate/supported/rejected；supported 表示有记录依据，不表示已证明因果 |
| Attribution | run_id、mode、status、error_step_ids、root_candidates、responsible_role、evidence_ids、limitations、usage | status 为 attributed/no_supported_failure/insufficient_evidence；错误集合与根因候选独立；不强迫选一个根因 |

`epistemic_status`：asserted / observed / inferred / supported / contradicted / unknown。观察到“测试退出码 0”不自动生成 supported 的“补丁正确”。`adoption_status`：tentative / relied_on / unknown，与真假独立。

根因候选字段：原生 step ID、entry ID、解释、证据 ID、传播路径和定性置信度；置信度默认未校准。模型必须允许返回空候选及证据不足。

## 边的方向和限制

| relation | 方向 src → dst | 允许依据 |
| --- | --- | --- |
| supports | 证据对应账本项 → 被支持判断 | 验证对象、版本和条件匹配；未知条件必须缩小结论范围 |
| depends_on | 使用者 → 被使用的前提/产物 | 显式引用、交接或有记录的产物消费 |
| contradicts | 新观察/判断 → 被反驳判断 | 同一对象及适用条件下的冲突 |
| supersedes | 新状态/决策 → 被替代状态/决策 | 有证据的替代 |
| reverts | 回退动作 → 被撤销状态变化 | 明确回退记录；仅文件名相同不足以建边 |

同路径、相邻时间、相似文本只能生成 candidate。旧 `relies_on` / `relied_on_by` 属启发式关系，迁移时保留来源并降为 candidate，不能视作因果金标。图允许跨阶段连接；回溯必须记录 visited，不能假定依赖图一定无环。

## 增量生命周期与恢复

1. 原始动作/观察可先独立持久化；配对后提交统一 Step，保留各自原始 ID。
2. 证据及确定性状态记录立即追加，不等待阶段封口或 API。
3. 划分器追加 StageUpdate 与 Membership；延迟确认需要同时保留边界位置和确认时刻。
4. 阶段整理通过持久化队列异步执行，作业绑定输入快照和幂等键；结果记录实际可用时间，不回填为早已知道。
5. 后续修改对象时保留旧验证，但不沿用其对新版本的支持结论。反驳追加新记录，不删除旧判断。
6. 检查点仅在持久化完成后推进。恢复时重放事件并去重；尾部未完成 JSON 行等待补全，明确结束事件与空闲超时分开。
7. 在线运行不读取人工阶段。oracle 实验的人工区间是离线上限条件，不能报告为在线行为。

## 归因证据接口

`get_evidence(run_id, evidence_ids, token_limit)` 仅返回本轨迹允许读取的原文、ID、摘要核验及截断状态。查询与返回计入成本。模型不可访问 gold 标签目录、其他轨迹、任意本机路径或执行命令。轨迹内容作为待分析数据，不能成为归因器的新指令。

失败后归因可以读取完整已结束记录；若未来扩展在线诊断，检索必须以 cutoff_seq 限制前缀。任何输入截断、缺失证据和失败调用均写入运行记录，不能悄悄丢弃样本。
