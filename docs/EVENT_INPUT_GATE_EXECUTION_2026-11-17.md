# 真实事件回放与阶段输入门槛执行结果

## 回放审计

只读取本地回传 OpenHands ZIP 中的原生 events JSON，不解压执行任何任务命令。按原生事件文件数字 ID 顺序回放；这不是实时网络延迟测试。

| 历史运行 | 事件 | 转出步骤 | 精确 cause 配对 | flush 前待处理动作 |
|---|---:|---:|---:|---:|
| run-8a3wc93e | 46 | 20 | 20 | 0 |
| run-arxu6xkh | 46 | 20 | 20 | 0 |
| run-v7gm_ugu | 36 | 15 | 15 | 0 |

合计 128 个事件、55 步。三条均无转换异常，没有命中缺失或截断标记。没有标记不等于上游绝对未截断；这三条 smoke 任务不能代表全数据集。完整诊断与每步身份/可见顺序在 `eval/live_input_audit_v1/report.json`。

## 已接入的门槛

新模块 `segmenter/input_gate.py`。OnlineLedger 在阶段封口创建作业时，按阶段步骤收集入口完整性记录。阶段中断、完整性记录缺失、缺失观察、声明截断或疑似截断均使作业成为 blocked。unknown completeness 不伪装成已证明完整，未发现已知缺陷只称 eligible。

同步 drain 和后台 worker 均先经过 dispatch 门槛。被阻止的作业生成 `status=blocked_input`、零 claim、明确原因的批次，不调用 processor。它不同于正常摘要返回零条 claim。相同输入重复处理保持一致，冲突结果拒绝覆盖。

`claim_extraction.build_request` 同时检查门槛，防止直接通过旧抽取入口绕开已标记的缺陷。旧离线作业缺少 input_integrity 时仍作为 legacy 数据按既有检查处理；没有回溯宣称旧数据完整，也没有重写旧实验。

## 计划字段衔接修复

回归发现旧 ledger.repair_step 仅接受外部精确调用 payload，不能接受 adapter v2 已保留的完整计划。新增 adapter v2 专用分支，检查 action 与 public_action_args 一致，再保留全部公开参数和 notes。OnlineLedger 不再重新应用丢弃 notes 的旧白名单。

新 OnlineLedger 输入版本为 event-integrity-v3，旧运行目录会因配置不一致被拒绝，必须新建目录。边界模型输入元数据也明确为修复后的 adapter，而不再声称仍是 legacy 输入。旧边界分数不能直接当作新输入版本的性能。

## 验证

- 输入门槛 4 项测试通过：缺失/截断判定、阻止 backend 与重复处理、真实 ledger 接入与恢复、计划 notes 进入账本。
- 既有 OnlineLedger 5 项、claim_extraction 4 项、Ledger 3 项测试通过。
- 故障测试验证截断 stage 不触发抽取，恢复后结果一致。真实三条回放未含已标记截断，不能用它们证明截断检出召回率。

## 尚未覆盖

这次接的是已有在线 ledger 作业通路，不是把远程 Qwen v3 试验直接部署为在线摘要服务。多段工具观察、无标记上游截断、完整实时延迟压测、跨服务队列恢复仍需另测。接收事件序号与 ledger operation seq 是不同时间坐标，分别保存，不混称真实时间。

本轮无需 GPU、没有模型调用或训练。代码及报告已在本地完成；修改涉及会话前已有未提交工作的 online_ledger/claim_extraction 文件，未将这些文件的其他既有改动擅自一并推送。
