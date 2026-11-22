# 神经划分接入账本：规则／GRU 双路验收（2026-11-23）

神经策略已接入实际账本验收入口，使用同一批事件、同一个 claim 处理器进行对照。`neural_ledger_acceptance` 新增 `--boundary-model`；不传时保留历史规则路径。新 `boundary_ledger_rollout` 默认运行规则和神经两路，候选失败不会静默回退成规则。全局默认替换仍待真实 7B 下游验收，不再以未达到 F1 0.70 为理由搁置接入。

## 固定候选

`artifacts/boundary_gru_adjacent_v1/model.json` 是部署入口，配套 `weights.pt` 共 190081 参数、764223 字节；CPU 推断，不逐步调用 LLM。基础阈值 0.55、相邻阈值 0.65、max_delay=0。权重 SHA 为 `7094f9c1c2f49d1a4f55d7faa4d43ac2ebdcc9ae399f8e2ffd950c05411619fc`。

它来自此前选定的 fold0 即时 checkpoint，不是本轮挑出的最好折，也不是全数据重训模型。已有 SWE 研究中的五折策略结果不能直接当作这个单文件在新会话上的准确率。模型来源和 JSON 哈希见同目录 `provenance.json`。

本轮把此前只在本地的相邻阈值实现一并版本化；JSON 神经模型直接加载 TemporalPredictor，避免无关的 LR/scikit-learn 依赖。神经耗时写入时间轴，不混入持久化决策快照；恢复仍重放原生事件，不能改变历史封口任务。输入模型或策略变更要求新输出目录。

## 已完成的本地验证

对四份已知合成会话、124 条事件运行双路回放：两路均覆盖 49 个归一化 step（其中 46 个工具 step），无阶段空洞、重叠或空段。阶段快照、封口任务与发布文件在恢复后保持字节一致；均保留 2 个开放末段，不补造 finish。

| 本地 CPU／规则 claim 干跑 | 规则 | 神经 |
| --- | ---: | ---: |
| 封口任务 | 26 | 15 |
| 开放末段 | 2 | 2 |
| step 就绪→决策发布 p50 | 15.91 ms | 18.74 ms |
| step 就绪→决策发布 p99 | 108.91 ms | 67.38 ms |

神经特征＋推断共 49 次，中位 1.14 ms、p95 7.22 ms、p99／最大 55.93 ms，包含首次调用的峰值。结果来自本地 Windows 磁盘、后台规则 claim 和恢复测试，不能视为远程实时 SLO，亦不能据两次小样本回放宣称 p99 改善。

任务变少不等于误切变少：这些会话没有金标 stage。实际 claim 的语义、排队与生成成本尚未测量，干跑的 ready 也不能算 7B 摘要成功。两路共 41 个任务均通过真实 CPU tokenizer 检查，每任务一块，最大输入 3700 token、无缺口，详见 `token_preflight.json`。36 项相关测试通过，包括真实候选加载、无 LR 依赖、恢复和模型变更阻断。

`local_acceptance.json` 保留两路阶段长度、恢复状态及每条既有事实的新阶段归属。事实正文没有随切分变化而改写；归属只表示输入可用性，不等于模型语义覆盖。两路都显式设置 PyTorch CPU threads=1。相同 claim 处理器身份由脚本核对。

## 远程 7B 端到端对照

等正在运行的 SFT／adapter 对照结束后，再执行本轮 GPU 回放，避免同时抢显存。**本轮两路固定使用旧 `out/claim_sft_7b_reviewed_v2/adapter`**，不同时改变划分器与 claim 模型；这也不会重切刚准备的 SFT 数据。

```bash
cd ~/online-failure-attribution
git pull --ff-only
conda activate agentforesight
python -m remote_pilots.openhands_smoke.boundary_ledger_rollout check
python -u -m remote_pilots.openhands_smoke.boundary_ledger_rollout run \
  --model-path /home/sjtu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --out out/boundary_ledger_rollout_v1
```

默认事件间隔 0.05 秒，仍是受控回放；规则与神经分别运行、分别保存日志与恢复证据。神经模型在 CPU，GPU 只用于固定的 7B 摘要。日志分别为 `rules.log`、`neural.log`，汇总为 `comparison.json`。回传 `out/boundary_ledger_rollout_v1/` 整个目录，失败记录也保留。

## 替换门槛和限制

1. 功能门槛：完整 step 覆盖、时序正确、封口不变、恢复不重复、模型身份不漂移。本地已通过，远程需再次核验。
2. 下游门槛：对同一证据事实比较两路摘要，关键事实完整覆盖不退化、错误完成声明不增加；结构失败和开放部分分别计数，不以 citation overlap 自动打语义分。
3. 成本门槛：报告实际 7B 生成块、失败与重试、排队和发布长尾；同时报告 CPU 模型耗时。不能用阶段数变化直接代替成本收益。

通过上述门槛再切新运行流程的默认策略；旧会话始终沿用原配置。若效果退化则保留明确的神经候选入口和规则对照，不宣称默认已替换。本轮没有新的边界 F1 或 failure alarm 结果。四份合成会话现在都是已研究的开发材料，不能据此作独立泛化结论。
