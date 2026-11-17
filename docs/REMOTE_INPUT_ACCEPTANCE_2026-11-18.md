# 远程验收：输入完整性、恢复与延迟

## 范围与结果标记

先验收输入和结构划分旁路，再单独测神经 claim。测试报告分别标为 PASS / FAIL / INSUFFICIENT（样本或证据不足），不能把代理任务失败等同输入失败，也不能把 schema 正确当作 claim 语义正确。

本轮没有更换默认划分模型。所有数字仅适用于本次提交、规则/模型版本、机器和输入模式。跨机器源 timestamp 不参与耗时相减；replay、恢复阶段和跨进程不可比耗时排除，保留 null 与样本数。

## 1. 准备与本地式故障验收

使用之前 shadow 依赖安装成功的 Python 环境（锁定 numpy 版本要求 Python >=3.12）。不要改动正在提供 Qwen 服务的环境。

```bash
cd ~/online-failure-attribution
git pull --ff-only
git rev-parse HEAD
python --version
python -m pip install -r remote_pilots/openhands_smoke/shadow-requirements.txt
python -m unittest discover -s tests -p 'test_input_timeline.py'
python -m unittest discover -s tests -p 'test_durable_input.py'
python -m unittest discover -s tests -p 'test_online_ledger.py'
python -m unittest discover -s tests -p 'test_live_shadow.py'
python -m scripts.validate_input_timeline --out out/input_timeline_remote_v1
```

输出目录必须是新的；重跑改 v2/v3，不覆盖上次结果。合成测试必须成功退出、9 steps、restart_with_pending=true；它不替代真实 agent 验收。

## 2. 实际 OpenHands 旁路

确认本地模型服务可用后：

```bash
curl --max-time 5 http://127.0.0.1:30001/v1/models
python remote_pilots/openhands_smoke/run.py --prompt-mode action-guided --shadow --quiet --collect
# 用本次 RUN_DIR 的真实值替换下面路径，不使用“猜测最新目录”方式。
RUN=out/openhands_live_smoke/run-实际目录名
python -m segmenter.timeline_report "$RUN/shadow/input_timing" --out "$RUN/shadow/latency_report.json"
python -m remote_pilots.openhands_smoke.collect "$RUN"
```

第二次 collect 会把首次收集后生成的 latency_report，以及 input_timing/receipts 和 timeline 一起带上。只需返回最后打印的诊断文件。运行结束再出报告，避免读取半行。即使 agent 修复任务失败，也保留日志供判断输入链路是否正常。

真实恢复检查：在另一次仍运行的 agent 上，单独启动 online_ledger 跟读该 session 的 events 目录（见时间轴实现文档）；确认 stages.json 的 pending_action_ids 非空时 Ctrl+C 停止监测器，保持 agent 继续运行，再用同一命令、同一 out/run-id 恢复。不要停止 Qwen 服务或删运行目录。若没有捕捉到 pending，标记该项 INSUFFICIENT，不宣称验证通过。旧 ledger 输出版本不兼容，必须新建本轮输出目录。

## 3. 完整性与时间标准

硬性通过标准（任何一项违反均 FAIL）：

- 支持协议内的完成动作：无漏步、无重复 step、无错误 cause 配对；与结束后原生事件核对 ID 集合和动作/观察引用。
- 恢复前已经发布的证据和 claim batch 内容不改变、不重复录入；pending 在结果到达后只完成一次。
- 首次 receipt 恢复前后不变；恢复记录明确标记。新完整结果不得使用未来事件；无观察的动作不得被标记为已完成。
- 同进程有效耗时非负；跨进程或 replay 耗时不伪装成 live。已知缺失/截断保留标记，受影响 stage 不当作可用语义抽取输入。
- 明确不支持的流式/孤立结果应清晰报错，不能悄悄丢弃后继续声称完整。

暂定工程性能目标（不是文献结论或已达到的承诺）：

| 指标 | 首轮目标/解释 |
|---|---|
| step_ready → 决策发布 | p95 ≤200 ms、p99 ≤500 ms；包含本地处理和落盘，区别于纯模型计算 |
| observation_received → step_ready | 报 p50/p95/p99；前序动作未结束造成的等待单独查原因，不能全部算适配器计算 |
| action_received → step_ready | 报告但不设固定毫秒门槛，主要含工具运行和等待 |
| pending/claim 积压 | 跟踪整个运行；上游停止且正常完成后应收敛，不能持续增长而报告低延迟 |

先收集至少 3 个真实 session、合计至少 100 个完成工具 step，再给首轮性能结论。p99 样本仍少，保留每次最大值和逐条记录；不足则 INSUFFICIENT。至少一次 pending 中断恢复。短 smoke 只证明接通，长期稳定性另需至少 30 分钟持续运行；未做则明确未验收。

## 4. Qwen claim 测量单独进行

当前 smoke shadow 不生成神经 claim，online_ledger 默认也是规则处理器。以上 latency_report **不能代表 Qwen 速度**。先完成输入验收，再连接实际 Qwen StageProcessor；本次不提供不存在的 online_ledger --qwen 参数。

对实际 Qwen 至少记录 20 个已封口 stage：模型/量化、上下文长度、输入/输出 token、首轮及重试次数、排队/处理/总完成耗时、成功/失败/blocked_input 数、GPU 峰值显存。冷启动单独报告，重试不能从耗时中删去。按 stage 长度分组，报告 p50/p95 和样本量；20 个样本的 p99 不作稳定结论。

吞吐验收看同一真实轨迹中 claim 产生速率与处理速率、队列趋势；平均处理时间小于平均 stage 间隔只是必要参考，不保证峰值不积压。在确定业务可接受的警报陈旧程度之前，不捏造固定的 Qwen 秒数门槛。语义质量另审证据支持、来源、遗漏和幻觉，本轮时序通过不等于质量通过。

## 返回材料

提交号、Python/依赖版本、运行命令、是否 live/replay、每个 session 的诊断包；合成验收目录；ledger 恢复运行目录（native_events、evidence_store、stage_jobs、claim_batches、timing、stages.json、online.json）。不要上传模型权重、API key 或整个 Conda 环境。结论分开填写：输入完整性、恢复、性能、长期运行、Qwen 速度、claim 质量。
