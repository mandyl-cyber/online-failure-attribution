# 输入审计与实施门禁

日期：2026-11-12。本次复核现有代码和已落盘审计，不重跑全量训练、不修改原始数据。

## 已确认

| 项目 | 证据 | 对本轮的约束 |
| --- | --- | --- |
| 当前严格对齐集合 461 条 | `eval/boundary_supervised/data_fixed/records.json` | 第一轮开发样本仅从此集合选择，不扩大覆盖声明 |
| 计划工具 1,649 步，其中 1,378 步 action 空白，涉及 270 条轨迹 | `eval/plan_payload_audit/report.json` | 文本丢失是转换缺陷；旧语义实验不能作为模型能力上限 |
| 1,378 个空白步骤全部按原生调用 ID 恢复 | 同目录 `payloads.jsonl`，unrecovered_empty_steps=0 | 已有 sidecar，不等于默认转换器已修复 |
| full/verified 和 full/mini 存在包含关系 | `eval/dataset_task_audit/codetrace.json`、`longrca.json` | 不能作为独立训练和测试集合相加 |
| 本地 stage 主要提供区间，错误标签为 incorrect/unuseful | 同上 | 不能把阶段编号解释成五类语义标签 |
| 运行时先事件检测、树更新，再阶段和候选 claim | `segmenter/run_online.py:OnlineSegmenter.feed` | 可复用检测，但目前同线程；新增 API 不能阻塞 feed |
| 现有树按状态变化聚合，候选 claim 尚非完整账本 | `segmenter/tree.py`、`segmenter/claims.py` | 新账本作为独立模块接入 |

## 编码前必须落实

- [ ] 计划字段使用新输入版本：白名单 command/task_list 的 id/title/status；按原调用 ID 绑定当前步，禁止取累计日志中的未来计划。
- [ ] 新版本独立目录、缓存键和转换器版本；旧特征、折分、模型输入不静默覆盖。对比旧/新转换的步号、工具、观察，只允许预期的计划字段变化。
- [ ] 每条轨迹固化原始记录摘要、转换后摘要、步号映射、排除原因。当前缓存一致性不等于原始字段保真。
- [ ] 数据输入与评分标签分目录。输入只允许任务原文、动作、观察、原生角色及可用元数据；thought 第一版不加入。solved、incorrect、mistake_step/reason/agent 只能用于选样或评分。
- [ ] 原始日志里真实出现的运行反馈可保留，但在线前缀不能读取未来反馈；数据集附加的离线评价不可混入。
- [ ] API 前确认任务原文确实可读取；缺失则显式排除并报告，不从根因解释反推任务。
- [ ] 成功轨迹可能含错误；solved=true 不等于 error_step_ids 必须为空。没有“已修复”金标时不自动声称已测修复识别准确率。
- [ ] LongRCA 原生消息步号保持不变；其未提供人工阶段，不参加 gold-stage 对照。
- [ ] 跨数据集重复任务核查尚未完成；完成前 LongRCA 只能称迁移探索，不能称独立无泄漏测试。

## 首轮开发集

由 `python -m segmenter.prepare_ledger_protocol` 确定性生成 40 条 CodeTraceBench 开发轨迹：SWE/Terminal × 成功/失败四格，每格 10 条，每个任务组最多一条。每格按长度排序分成 10 个分位桶，每桶通过 seed=41 的 SHA256 顺序选一个未用任务组。不足则报错，不自动放宽。

该集合取自已多次分析的数据，只用于接口、提示和账本开发，不是全新测试集。采样文件包含结果分层元数据，禁止直接作为模型输入。程序只读取已有文件、写入新的实验目录，不解包、不调用网络、不训练。

最终评测需要另行锁定协议和合格任务组清单，排除开发任务及已知重复任务；对过去接触情况如实记录，不能通过重新命名集合制造未见测试。
