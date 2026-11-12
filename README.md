# 划分器（M1 / 第一步）

## 在线阶段—账本—归因协议

新增[阶段 claim 抽取协议与 dry-run](docs/ledger/CLAIM_EXTRACTION.md)：在线入口加 `--claim-extractor dry-run`，生成带严格证据引用的请求、规则模拟输出、校验记录和 claim 批次。支持持久化重试预算及缓存；无 API 调用，语义质量尚未评估。

最新入口：[在线阶段封口与多 claim 入账](docs/ledger/ONLINE_STAGE_CLAIMS.md)。`python -m segmenter.online_ledger <events.jsonl或events目录> --out out/live_claims --run-id <会话ID>` 接收实时原生事件，用 phase 规则封口并后台生成 claim 批次；支持暂停重启，尚无语义 API 抽取与关系边。

已实现无 API 的[本地增量账本原型](docs/ledger/LOCAL_PROTOTYPE.md)：独立计划字段修复、逐步证据落盘、恢复与阶段归属修订。入口 `python -m segmenter.replay_ledger --out <新目录>`；支持/依赖边和归因器仍未实现。

新一轮实施入口：[接口规范](docs/ledger/INTERFACES.md)、[输入审计](docs/ledger/INPUT_AUDIT.md)、[实现清单](docs/ledger/IMPLEMENTATION.md)、[实验配置](docs/ledger/experiment_v1.json)。这些是待实现协议，不代表账本与 API 归因已上线。
`python -m segmenter.prepare_ledger_protocol` 仅固化 40 条开发样本及输入哈希到 `eval/ledger_protocol_v1/`，不训练、不调用 API；输出目录已存在时拒绝覆盖。

## 执行上下文实验与 M2 材料

默认行为保持不变。`--set execution_context.enabled=true` 启用实验模块：
保留 OpenHands `is_input` 和 session id，在明确输入标志或同 session 上一步
末尾等待 login/password/qemu 提示且当前无退出码时，将输入从 shell 命令类别降为 OTHER。
该判断仅使用前缀；没有 session id 时使用默认 session，多终端混流需补齐元数据。
匹配到脚本观察中的 saved/wrote 路径时，写入 `attrs.reported_writes`（含原文 span），
并抑制该 probe 自动变成验证。输出自报写入不等于实际文件变化，因此不直接生成
STATE_CHANGE，也不推断任意脚本文件的内部逻辑。显式测试命令仍按原验证规则处理。
开关默认关闭，需人工审计其召回与误判；实验输出在 `eval/verified_context/`。

M2 操作规范在 [pilot/README.md](pilot/README.md)。
`python -m segmenter.pilot` 生成 40 条 CodeTraceBench 完整轨迹、两份空白模板，
以及隔离的协调者候选文件。LongRCA 编码域另 40 条已通过
`python -m segmenter.pilot_longrca` 生成到 `pilot/m2_longrca/`，保留原生消息步号。
两套材料共 80 条，真实双人标注尚未开始；LongRCA 规则事件候选尚不支持。

## M1 开发集审计

`python -m segmenter.audit_m1 --out eval/m1_audit_new` 生成任务级固定哈希拆分、
27 组 phase 阈值扫描、开发集多切边界上下文及 EXEC 错误步清单。
可用 `--rules eval/m1_audit/baseline_rules.json` 复现原始规则。
首次扫描结果在 `eval/m1_audit/summary.json`：Terminal-Bench 开发集 145 条、
留出集 40 条；仅开发集选中的 VERIFY=2 在留出集 F1 从 0.434 降至 0.424，
因此没有修改默认 phase 阈值。该拆分是本轮调参留出集，数据此前已经用于总体分析，
不应称为从未接触过的最终测试集。

89 个 EXEC 错误步中 45 个是已识别 probe，44 个为其他执行。
开发集实例显示交互终端输入、脚本副作用及 SQL/下载命令均需更细的语义处理，
不能只按命令头强行归为改状态。此次仅补充带明确 `-o` 输出的 ISO 创建命令；
全量复验写入 `eval/verified_command_audit/`。
`dev_extra_boundaries.json` 是距离任一金标边界超过 1 步的候选样本，
不是一对一匹配意义下所有假阳性的完整枚举。

## 2026-11-12 评估更新

`eval_segmenter run` 现在默认同时报告 k=3、6、10（并包含 `--recall-k` 指定值），
在 `summary.json` 的 `by_source` 中分别汇总 SWE 与 Terminal-Bench，输出
`event_recall_curve.csv` 和 `event_recall_curve.png`。按 manifest 的
`source_relpath` 前缀 `swe_raw/`、`openhands/` 识别当前 OpenHands 子集来源；
其他路径保留为 unknown，不根据模型或是否有 thought 推断。
来源汇总仅包含成功对齐的轨迹，全局 skipped 保留排除计数。
召回分母分别报告错误实例和 incorrect 步，默认不含错误当步；窗口分布仅统计闭合样本。
本轮结果写入 `eval/verified_by_source/`，旧结果保留在 `eval/verified/`。
下文 45 条抽样及“全量未跑”是早期记录，已由 498 条全量结果取代。

接口参考：[数据集卡片](https://huggingface.co/datasets/NJU-LINK/CodeTraceBench/blob/main/README.md)
与 [CodeTracer](https://github.com/NJU-LINK/CodeTracer)。stage 导出保留官方三个字段；
下游比较使用 tracebench profile 的 `codetracer_labels.json`，逐轨迹 F1 后取宏平均。
仅字段兼容不等于已验证 CodeTracer 完整运行接入；本次未调用诊断模型。

总体技术设计"第一步：划分器"的实现：不训任何模型，纯规则、零解码的在线树构建器 + 打分脚本。
第一轮只用正则和工具类型；DRIFT 和 embedding 留了开关（`rules.yaml: events.drift.enabled`），没有实现。

```
segmenter/
  adapters/openhands.py   转换器：OpenHands 事件流 / chat messages / CodeTracer steps.json -> 统一 step
  rules.yaml              全部规则和阈值（命令分类、事件、树、切分、候选 claim、触发）
  shell.py                shell 命令分类（一行命令 -> 子命令 -> 类别、读写路径）
  events.py               事件检测：STATE_CHANGE / EXPLORE / VERIFY_RUN / VERIFY_PASS / REFERENCE / HANDOFF / IRREVERSIBLE / PLAN_SWITCH
  tree.py                 状态树：节点、挂载步、relied_on_by 边、reverts 边、封口信号
  segment.py              切分：stage_ranges.json（懒封口），node / phase 两种模式
  claims.py               候选 claim：test_result / env_assertion / plan_step / completion
  run_online.py           在线主循环；replay（回放）/ follow（跟读事件流）/ bench（批量）
  bench.py                CodeTraceBench 读取；inspect = 打印一条轨迹的全部字段
  eval_segmenter.py       边界 F1、错误步落点、窗口代理分布、事件召回、下游 F1 差
tests/test_lifecycle.py   合成轨迹，复现设计文档"一个时间步的生命周期"
```

依赖：Python ≥ 3.9、`pyyaml`；解 `.tar.zst` 要 `zstandard`（或系统 `zstd`）；出图要 `matplotlib`；读 parquet manifest 要 `pandas` + `pyarrow`（用 .jsonl manifest 则不需要）。

## 按逐日计划怎么用

```bash
pip install pyyaml zstandard matplotlib
python -m pytest -q tests                      # 先确认本地能跑

# 第 1 天：下数据，打印一条轨迹的全部字段
huggingface-cli download NJU-LINK/CodeTraceBench --repo-type dataset --local-dir ./codetrace_bench
python -m segmenter.bench inspect --manifest codetrace_bench/bench_manifest.verified.jsonl --artifacts codetrace_bench

# 第 2–4 天：回放单条轨迹看事件 / 树 / 切分
python -m segmenter.run_online replay codetrace_bench/_extracted/<traj_id> --out out/<traj_id> --print

# 第 5 天：前三个数字 + 事件召回（先 verified 调规则，再 full 报数）
python -m segmenter.eval_segmenter run --manifest codetrace_bench/bench_manifest.verified.jsonl \
    --artifacts codetrace_bench --out eval/verified --modes node,phase --export-stages eval/verified/stage_ranges
#   调参不用改文件： --set tree.explore_gap=2 --set segment.min_stage_len=3

# 第 6–7 天：CodeTracer 分别喂人标 stage 和 eval/verified/stage_ranges 跑完后
python -m segmenter.eval_segmenter downstream --manifest ... --gold-run outputs_gold --rule-run outputs_rule

# 接在正在跑的 OpenHands 上（events/ 目录或 .jsonl）
python -m segmenter.run_online follow <session>/events --out out/live
```

代码里直接用：

```python
from segmenter.adapters.openhands import OpenHandsStreamAdapter
from segmenter.run_online import OnlineSegmenter

adapter, seg = OpenHandsStreamAdapter(), OnlineSegmenter(out_dir="out/live")
for event in openhands_event_stream:
    for step in adapter.feed(event):
        row = seg.feed(step)          # events.jsonl 的一行；row["trigger"] / row["sync"] 是归因器触发位
seg.finish()
```

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `steps.jsonl` | 统一 step：idx, role, thought, action, observation, tool_class, touched_paths（另有 written_paths, exit_code, attrs, raw_ref） |
| `events.jsonl` | 每步一行事件向量：events, refs, for_idx, node, closed_node, close_reason, sealed_stage, dep, trigger, sync, detail |
| `tree.json` | nodes（span、state/explore/verify 步、touched_paths、symbols、last_verify、relied_on_by、reverts、close_reason）+ edges |
| `stage_ranges.json` | `[{stage_id, start_step_id, end_step_id, label, sealed, nodes}]`，前三个键与 CodeTraceBench 的 `stages` 一致 |
| `claims.jsonl` | 候选 claim：id, type, text, step_idx, source, span, node |

## 设计文档没写死、这里做了选择的地方

划分器的详细技术文档不在这个文件夹里，下面几条是按总体设计推出来的，都能在 `rules.yaml` 里改：

1. **节点粒度**：一个节点 = 前导探索 + 一批改状态步 + 挂着的验证步。验证失败后在同一批文件上继续改算同一节点（修复循环）；验证被放行、换方案、换到不相交的文件、交接、不可逆动作前都封口。`tree.merge_consecutive_edits: false` 可退回"每个改状态步一个节点"。
2. **VERIFY_PASS**：验证通过后第一个非验证步到来时补发，`for_idx` 指向那次验证。
3. **REFERENCE**：thought / 动作里提到更早节点改过的符号（def / class 名）或文件，或本步又写了更早节点改过的文件。不含同一节点内的引用。
4. **切分两种模式**：`node`（设计文档的写法：边界 = 节点封口，VERIFY_RUN 后先记候选）和 `phase`（按 环境检查 / 安装 / 排查 / 打补丁 / 验证 相位切，贴近 CodeTraceBench 人标 stage 的定义）。`eval_segmenter` 默认两个都报。
5. **窗口代理**报两个口径：任意 VERIFY_PASS / REFERENCE，以及指向错误步所在节点的（targeted）。错误实例 = 每个出错 stage 的第一个 incorrect 步。
6. **触发位**里"σ → consequential"那一条要等账本写入器（M4），现在只算 HANDOFF / VERIFY_PASS / IRREVERSIBLE / 被依赖 ≥ k。

## CodeTraceBench 的实际结构（下载后核对过）

OpenHands 子集在 verified 里有 520 条，原始轨迹有两种落盘形态，转换器都认：

| 来源 | 条数 | 形态 | 步号怎么对齐 |
| --- | --- | --- | --- |
| SWE 系（`swe_raw/openhands__{verified,poly,pro}/<任务>/`） | 313 | 每次 LLM 调用一个 `<model>-<时间戳>.json`（`messages` + `response`） | 一步 = 一次工具调用 + 它的结果；没有工具调用的 assistant 回合和末尾的 finish 不占步号 |
| Terminal-Bench 系（只在 `bench_artifacts/full/*.tar.zst` 里） | 199 | 标准 OpenHands `events/` 目录 | 一步 = 一个 agent 动作（含 finish） |

另有 8 条 manifest 里没有路径，会被跳过。同一个 SWE 任务目录里可能混着两次运行的日志，读取时按 manifest 的 `step_count` 选对应的那次。
Terminal-Bench 那一半需要 `pip install zstandard` 才能解包。

## 在 45 条抽样上的初步结果（不是正式数字）

从 verified 的 OpenHands 子集里每种来源抽了几条，共 45 条、1653 步，步号全部对齐。只有 19 个错误实例，结论很粗：

| 数字 | 结果 | 门禁 |
| --- | --- | --- |
| 边界 F1（phase 模式） | 0.52（node 模式 0.32；知道 stage 数的等距切分 0.46） | ≥ 0.70 |
| 错误步落点 | 65% 落在改状态步 | 预期 > 70% |
| 窗口代理 | 中位 6 步，全部 ≥ 3 步，42% 到轨迹结束都没有接受事件 | 对照 FaaP 中位 1 |
| 事件召回（3 步内） | 0.16 | ≥ 0.60 |

## 还没验证的事

监督划分接续实现与最新结果见 [segmenter/BOUNDARY_SUPERVISED.md](segmenter/BOUNDARY_SUPERVISED.md)：五折严格对齐的 461 条轨迹上最终边界 F1=0.5599；119 条外部任务轨迹上为 0.5811。支持 `--boundary-model` 可选在线接入，未设为默认，尚未达到 0.70 或完成真实长时部署。下列历史全量基线保留作为规则研究记录，不能与不同样本上的新模型直接比较。

- **已完成全量评估**：520 条中 498 条进入统计，14 条步号不匹配、8 条缺路径。最新基线见 `eval/verified_command_audit/summary.json`；上面的 45 条结果仅为历史抽样记录。phase F1 约 0.4866，k=3 代理事件召回约 0.4115，仍未达到原门禁。
- **AgentRx 的事件召回**：没有 AgentRx 的转换器。`eval_segmenter run --gold-jsonl` 接受通用 gold（`{traj_id, traj_path, key_steps}`），轨迹是 OpenHands 事件流 / chat messages 格式即可直接用。
- 下游影响（数字 4）：40 对输入已生成并通过官方 CodeTracer 接口校验，一对完成本地提示词 dry-run；尚无模型诊断结果。运行入口和限制见 [downstream/README.md](downstream/README.md)。Docker Linux 引擎未启动，模型/API 和预算待配置。
- 完整接续进展见 [HANDOFF.md](HANDOFF.md) 第 8 节。M2 材料已准备，但用户暂不标注，不能据此认定真实接受时刻已得到验证。

小编码器方案的执行记录见 [segmenter/ENCODER_EXPERIMENT.md](segmenter/ENCODER_EXPERIMENT.md)：MiniLM 和 ModernBERT 冻结五折均已完成，暂未证明语义编码带来稳定收益；耗时对照与受控微调正在串行执行，实时状态见 `eval/boundary_encoder/continuation/status.json`。尚未接入编码器在线 backend，也未替换现有模型。

后续即时/延迟确认对照与错误审计见 [segmenter/CONFIRMATION_AUDIT.md](segmenter/CONFIRMATION_AUDIT.md)：不建议统一等待一两步；发现并独立恢复了 1,378 个因转换器而丢失内容的公开计划动作。一次轻量输入修复探测 F1=0.5653，仍未部署。原协议微调继续运行，旧转换输入缺失须纳入结果解释。

任务与标注的全量审计及下一版结构设计见 [数据集任务标注审计与stage账本设计.md](数据集任务标注审计与stage账本设计.md)：区分任务领域、工作目标、阶段、状态节点和 claim；目前有阶段边界监督，尚未在本地找到论文五类阶段的逐步类别标注。任务画像与阶段账本方案仍是待验证提案。
