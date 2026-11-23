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
# 远程回传验收（2026-11-24）

已收到根目录 `boundary_ledger_rollout_v1/`。两路固定使用同一旧 7B adapter；本轮没有收到新 SFT adapter 对照结果。复现检查：

```powershell
python -m scripts.audit_boundary_ledger_return
python -m scripts.review_boundary_ledger_return
```

工程检查通过：同一批 4 个原生会话、49 个归一化 step；输入哈希、模型身份、引用、逐块生成与发布结果一致，封口无空洞或重叠，中断恢复未重复发布。两路均保留 2 个开放末段，6 条开放事实（其中 5 条关键事实）不算抽取遗漏。所有生成均一次完成，无重试、分页缺口或拒绝项。ready 只表示结构与来源检查通过。

| 指标 | 规则 | 固定 GRU 候选 |
| --- | ---: | ---: |
| 封口任务／实际生成块 | 26 | 15 |
| 发布 claim | 39 | 28 |
| 累计生成时间 | 41.86 s | 35.46 s |
| 单块生成中位数 | 1.55 s | 2.13 s |
| 峰值 GPU 分配内存 | 8.47 GiB | 9.03 GiB |
| AI 判定：支持／部分支持／不支持 | 25／8／6 | 17／8／3 |
| 同一 40 条封口事实：完整／部分／遗漏 | 17／19／4 | 17／15／8 |
| 其中 28 条关键事实：完整／部分／遗漏 | 10／16／2 | 11／13／4 |
| 20 条负面结果的粗粒度保留 | 20 | 17 |

生成块减少 42.3%，累计生成时间减少 15.3%，但单块更慢、峰值显存增加，不能推断统一的延迟收益。例如 config_overrides 的排队到发布中位数从 5.68 s 增至 6.77 s。GRU 在单 CPU 线程运行，各会话特征加推理中位数为 0.48–0.55 ms、全部最大 0.80 ms。step 就绪到划分发布全部最大值为规则 11.88 ms、神经 10.41 ms；这些是 0.05 s 间隔的受控回放数据，不是新实时 SLO 验收。

**下游语义门槛未过，暂不替换全局默认。**已有改善是将 copy-before-update 与随后四测试通过放在一起后，明确保留了复制行为。退化集中在 prefix_totals：第二次替换失败和成功编辑缺少对应结果引用；实际测试失败被写成“没有通过”；一次测试的 pass/fail/pass 被写成 fail/pass/fail 的虚构经过。另有把已有三行注释当作本次新增三行、把输入字典被修改误述成 x 等于 z 等错误。规则摘要也有错误，保留默认并不表示规则已合格。

本次为非盲 AI 阅读，不是人工质量评测；粗粒度失败保留也不保证原因和时序正确。事实仍使用输出产生前冻结的同一份参考，未按新切分改写；规则 39 条输出逐字匹配上轮结果，因此沿用原判定。神经 28 条逐条核对引用，正确子句可以覆盖事实，而同条 claim 的其他子句仍可判错。四会话属于已研究的合成开发任务，没有 stage 金标，本轮不产生新的边界 F1 或独立泛化结论。

下一步先接收既定 SFT 新旧 adapter 对照，重点检查同一步动作／结果绑定、测试用例与调用次数、修改前后差异及失败信息保留；该对照若通过，再冻结相同 GRU 候选复测端到端。暂不据这四条开发轨迹调整切分阈值。后续默认替换仍要求关键事实覆盖不退化、错误完成声明不增加，再报告真实排队成本。

版本化结果：`return_summary.json`、`return_semantic_issues.json`、`return_fact_coverage.json`；全部 claim 的 AI 复核及工程详情位于 `out/boundary_ledger_return_audit_v1/`。原始回传与冻结参考未修改。
