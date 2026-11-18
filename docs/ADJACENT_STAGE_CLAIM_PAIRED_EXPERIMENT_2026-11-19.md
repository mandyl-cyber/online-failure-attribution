# 相邻切点策略对语义 claim 的配对实验

状态：**输入、参考和远程运行器已准备；Qwen 生成尚未执行。**本实验接续 [`ONLINE_STAGE_ADJACENT_BOUNDARY_EXPERIMENT_2026-11-19.md`](ONLINE_STAGE_ADJACENT_BOUNDARY_EXPERIMENT_2026-11-19.md)，检查减少重复切点是否真的改善阶段摘要，而非仅改善边界 F1。

## 固定设计

- 使用此前未进入边界训练/选型的 37 条 OpenHands SWE 任务中，fold0 预测切分发生变化的全部 **9 个任务**。每个任务取两策略差异周围最近的共同边界，形成同一个步骤并集；原策略与相邻策略只改变阶段分组，不改变输入步骤或模型。
- 共有 **28 个 stage 请求**：原策略 18、相邻策略 10。输入在 `pilot/adjacent_claim_pair_v1/requests.jsonl`，配对关系在 `groups.json`。每个请求保留完整、可定位的动作与观察证据；没有用金标 stage 构造模型输入。
- 固定现有 `stage-claim-summary-v3` 的提示、schema、温度 0 和单次输出预算 2048 token。两组使用同一份、已在远程验证过的 Qwen2.5-32B-Instruct-AWQ 权重。上下文上限 8192 token，最大并发 1。每组任一请求超长，则整个任务组排除；不截断、不只留一个 arm。
- 模型输出前，AI 从原始证据中整理了 **20 个非穷尽参考点**及禁止推断，见 `reference_points.json`。它们不是人工金标；审阅时必须同时检查输出是否虚构结果、是否遗漏关键失败/验证事实、引用是否真能支持陈述。结构通过率和 claim 条数不等于摘要质量。
- 记录每组完整输出和失败、claim 支持与参考点覆盖、调用次数、token、耗时，以及封口时间变化。由于原策略一个并集通常要两次模型调用，而新策略通常一次，成本按组比较，不能按单请求中位耗时冒充等成本。

冻结协议在 `pilot/adjacent_claim_pair_v1/protocol.json`，保存请求、分组、AI 参考、边界协议和结果的哈希。远程运行器在 `remote_pilots/openhands_smoke/adjacent_claim_pair.py`；原始返回和未打分复核包保存在 `out/adjacent_claim_pair_v1/`。

## 远程硬件与运行

先前该远程机器的 RTX 5090 32 GB 已完成同一 32B AWQ、8192 token、单并发实验；服务就绪显存约 30.2 GB，请求后约 30.8 GB。此次**不需要新硬件或训练**，但模型服务会占用几乎全部显存；同一时刻不能让其他大 GPU 任务占显存。28 个请求是上限，实际调用数取决于 token 预检。模型请求本身预计数分钟，权重校验与服务启动另计，预留 20–30 分钟较稳妥。

在 Linux 远程仓库根目录，用之前运行 32B 的 `agentforesight` 环境：

```bash
git pull --ff-only
conda run -n agentforesight python -m remote_pilots.openhands_smoke.adjacent_claim_pair check
conda run -n agentforesight python -m remote_pilots.openhands_smoke.adjacent_claim_pair run \
  --inputs pilot/adjacent_claim_pair_v1 \
  --out out/adjacent_claim_pair_v1 \
  --model-path vendor/models/Qwen2.5-32B-Instruct-AWQ \
  --vllm vendor/.venv-vllm/bin/vllm
```

运行器核验固定权重哈希，在本地端口 30032 启动**自己的** vLLM 进程，结束后仅停止这一进程。预检结果 `preflight.json` 会先落盘；若无可运行配对组，模型不会启动。逐请求 JSON 保留原始响应、schema/引用检查、token 和耗时。返回整个 `out/adjacent_claim_pair_v1/` 目录；若只通过 Git 同步，应先打包或按项目现有上传流程同步，避免把模型权重和服务日志提交到仓库。

## 回传验收与下一步

1. 核对配置、权重、输入哈希；逐组报告纳入/排除和全部请求的成功/失败分母。
2. 盲于策略优劣，逐 claim 审核原引用是否完整支持，逐参考点记 `covered/partial/missed`，尤其核查计划、提交说明、0 测试、缺 pytest、Matplotlib 的 `NotImplementedError` 等危险升级。
3. 按同一任务步骤并集比较两边的可用关键信息、严重误陈述、封口后可用时刻和总调用成本；如果两边都失败，记录双败，不强行选胜者。
4. 这 9 组是基于已检查的 fold0 切分差异选出的诊断样本，不能推断总体 claim 提升。只有语义质量与延迟同时可接受，再扩大到新的真实事件流并考虑替换在线策略。
