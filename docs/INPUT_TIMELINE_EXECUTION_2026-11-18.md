# 输入时间轴与在线验收

## 实现

三个入口 run_online follow、online_ledger、remote_pilots/openhands_smoke/shadow 现在共用 validate_event 和 InputTimeline。保留各入口已有的持久化/恢复流程，不复制另一套账本事件日志。

严格配对：稳定事件 ID、明确 cause、每动作一个最终结果；通知不参与配对。输入模式分 live/replay。receipt 保留源 timestamp 和首次接收 UTC 纳秒、单调时间及进程 UUID；事件重放不覆盖首次 receipt。时间元数据不写入 Step 或不可变证据，避免恢复时改变账本哈希。

阶段：首次接收、step_ready、boundary_published（每步决策发布，包括不切分）、stage_sealed_claim_enqueued、claim_started、claim_completed/failed/reused。补充 pending 动作和未完成 claim job 数量，供观察积压。在线账本在 stages.json 发布后记录决策发布时间；shadow 同样在 snapshot 写入后记录。follow 在该步输出后记录，阶段快照仍遵循原有写入策略。

耗时：action_to_ready 含工具等待及前序阻塞；observation_to_ready 含本地持久化、配对和前序阻塞；ready_to_publish 含下游处理与落盘；queue_ms 与 processing_ms 分开。它们不是纯分类器运算时间。stage 封口目前记时间事件，尚没有独立“候选边界首次提出”事件，所以不能直接据此计算 D0/D1/D2 的主动等待时间。

只对同进程、live、非恢复的时间计算差值。跨进程为 null；重放被排除。源 timestamp 原样保存，不计算跨机器时钟差。首次接收指读取器拿到完整可解析事件的时刻，不代表写入第一字节或源端开始执行。已有历史文件使用 --input-mode replay；live 模式下启动积压也只能解释为观察器接收之后的耗时。

## 本地验收

`python -m scripts.validate_input_timeline --out <新目录>` 启动真实 JSONL 追加线程、分块写行，消费过程中在 pending 动作处重建账本，测试去重，最终产生 9 个 step 并完成规则 claim。结果为合成在线文件验收，不是实际 OpenHands 或 Qwen 延迟实验。

初次结果 `eval/input_timeline_v1/report.json`：9 step，pending 重启通过。观察到 ready_to_publish 中位 8.33 ms，p95 43.88 ms（n=9）；样本极小且含 Windows 文件写入，不作性能承诺。claim processing 样本仅 2 个，规则处理，不能推断神经抽取延迟。

第二次 `eval/input_timeline_v2/report.json` 增加积压采样，9 step 的 ready_to_publish 中位 8.22 ms、p95 13.56 ms。两次波动说明不能用这个小样本给出延迟承诺。最终补充 drain 后的积压快照，避免结果已落盘但最后一个采样仍显示待处理。

测试覆盖首次 receipt 持久性、跨进程不相减、replay/recovery 不入统计、严格配对、账本恢复、重复 claim 结果复用；并回归 durable_input、online_ledger、live_shadow、input_gate、event_input_integrity。

## 远程指令（代码同步后）

已有运行的 smoke 入口带 --shadow 会自动使用新计时；请新开运行目录，保留 input_timing 子目录。

```bash
python remote_pilots/openhands_smoke/run.py --prompt-mode action-guided --shadow --quiet --collect
# RUN 替换为本次输出目录；待 writer 停止后生成报告：
python -m segmenter.timeline_report "$RUN/shadow/input_timing" --out "$RUN/shadow/latency_report.json"
```

如单独验证账本事件读取，可启动以下命令跟读正在追加的源，Ctrl+C 暂停后用完全相同的命令恢复：

```bash
python -m segmenter.online_ledger "$EVENTS" --out out/ledger_timing_v1 --run-id ledger-timing-v1 --input-mode live
python -m segmenter.timeline_report out/ledger_timing_v1/timing --out out/ledger_timing_v1/latency_report.json
```

默认规则 claim，不会调用 Qwen。验收先检查重复/丢失为零、pending 恢复正确，再观察 p50/p95/p99、样本数与积压。不要把预存轨迹标成 live。online_ledger 输入版本已提升，旧目录配置不兼容，使用新目录。

## 尚未完成的验收

未在远程机器完成实际 OpenHands 中断恢复和长时间积压测试；未测 Qwen 排队和语义抽取耗时；未做跨机时钟同步。单写者约束、O(N) 重放、源轮转、多结果流等上一版限制仍在。收到真实在线报告后才能判断是否达到延迟目标。本轮没有训练、调阈值或声称边界 F1 提升。
