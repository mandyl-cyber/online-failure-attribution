# Stage 结束时的关键事实保护：实现与远程验收

## 目的与位置

在线系统仍按“事件输入 → 轻量 stage 划分 → 延迟封口 → stage 结束后生成多条 claim → 后续 failure alarm”运行。此版只改封口后的 claim 输入与写入校验，不增加逐步模型调用，也不替换当前在线默认链路。

现有 Qwen32 邻接 stage 摘要试验中，曾把 `sed` 阅读测试源码表述为运行测试，也曾在合并摘要时漏掉 `Ctrl+C`。计划清单里的“测试通过”、0 个测试却 `OK`、缺少测试运行器等也需要明确保留其证据性质。

## 实现

- `segmenter/stage_claim_facts.py`：从封口 stage 已配对的 action/observation 中提取保守的观察事实；每条事实记录 step、原始证据 ID、退出码、测试数量与可用的 chunk ID。它不推断补丁正确性。
- `segmenter/stage_claim_summary_v4.py`：沿用 v3 的原始证据、schema 和来源校验，在请求中增加 `observed_facts`；生成后拒绝无观察依据的高风险测试成功表述，并记录遗漏的关键失败／中断事实。原始响应由调用方保留。
- `remote_pilots/openhands_smoke/stage_claim_facts_v1.py`：复用冻结的 9 组、28 个 v3 stage source snapshot，同一 Qwen2.5-32B-Instruct-AWQ 权重、temperature 0、2048 输出 token；全组上下文预检，超限时整组排除。输出保存输入/代码哈希、权重哈希、模型原始响应、用时、用量、失败原因和显存快照。

守卫仅拦截可明确判错的表达；它不是通用的文本蕴含判别器，也不会保证 claim 覆盖所有重要事实。`missing_critical_fact_ids` 是待审查标记，不会偷偷补写 claim。

## 本地回顾性诊断

对上一轮 28 个旧请求构造 v4 输入并回放旧响应：旧版可写入 21 个，新守卫拦下 1 个（Django 12406 邻接摘要把 `sed` 阅读表述为测试），20 个通过；通过者中 1 个（Django 15382 邻接摘要）仍漏掉 Ctrl+C 事实。原始结果见 `pilot/stage_claim_facts_v1_probe/report.json`。这是旧输出的规则兼容检查，不能作为 v4 模型质量或总体准确率。

本地验证：`python -m unittest tests.test_stage_claim_facts -q` 为 6/6；冻结输入 `check` 为 9 组、28 个调用；脚本通过 `py_compile`。

## 远程运行与回传

在已配置模型和 vLLM 的远程 Linux 仓库中：

```bash
git pull --ff-only
conda run -n agentforesight python -m remote_pilots.openhands_smoke.stage_claim_facts_v1 check
conda run -n agentforesight python -m remote_pilots.openhands_smoke.stage_claim_facts_v1 run \
  --inputs pilot/adjacent_claim_pair_v1 \
  --out out/stage_claim_facts_v1 \
  --model-path vendor/models/Qwen2.5-32B-Instruct-AWQ \
  --vllm vendor/.venv-vllm/bin/vllm
```

运行返回非零时仍先保留 `out/stage_claim_facts_v1`；脚本会记录逐请求失败。回传该目录中的 `summary.json`、`preflight.json`、`mapping.json`、`config.json`、所有生成的请求结果 JSON 及 `server.log`，无需上传权重。应按任务组核对新旧在共同 eligible 集上的真实性、关键事实覆盖、错误断言、有效 claim 数和生成耗时。仅靠 schema 通过率或 claim 数量不得宣布改进。

若 32B 的事实侧栏仍不能稳定覆盖中断与测试失败，下一步优先改 stage 输入的证据保留和针对性监督数据；不应直接把提示词试验当作后训练成效。上线前仍需在真实事件流中检验封口、排队、重试和恢复。
