# 一步自适应封口与7B摘要配对实验

本地执行入口为 `python -m scripts.run_stage_claim_pipeline --adaptive-replay`。同一冻结 fold0 权重在37条已有SWE轨迹上分别使用即时策略和置信度不足时最多等一步的策略。策略参数在本轮回放前固定，尚未在独立任务上验证。

本轮即时策略D0 F1为0.4745，自适应为0.4060。在共同D1时限下为0.5144与0.5054。自适应发布277个边界，即时发布265个。按切点集合比较，自适应撤销8个原切点、增加20个切点。18条任务的阶段范围变化，共准备92个摘要请求。该版未表现出边界收益。

远程先执行token预检，使用训练时相同的7B基座及已训练adapter。任一组内任一请求超限，将整组排除，保留同一证据窗口的配对关系。两臂都附加v1 observed_facts，采用相同提示、NF4加载方式和512 token输出预算。请求哈希固定在protocol.json。GPU生成后的语义质量需要回传复核。

```bash
cd ~/online-failure-attribution
conda activate agentforesight
QWEN7B_DIR=/home/sjtu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
python -m remote_pilots.openhands_smoke.adaptive_seal_claims preflight --model-path "$QWEN7B_DIR"
python -m remote_pilots.openhands_smoke.adaptive_seal_claims run --model-path "$QWEN7B_DIR"
```

回传 `out/adaptive_seal_claim_v1/`。现有边界结果偏负，生成任务主要用于诊断撤销/新增边界是否保留失败证据。远程预检和生成尚未执行。

2026-11-22 已完成18组同源证据与阶段覆盖核对，得到200个规则候选片段。`critical_facts.json` 在生成前固定23条AI复核的关键执行事实，覆盖12组，另外6组仍需要补工作目的及编辑语义审查。该清单属于定向事实集合。`fact_review.json` 保留完整来源、候选和两臂阶段归属，尚未把全部200项标为已审。

远程预检前可先执行 `python -m remote_pilots.openhands_smoke.adaptive_seal_claims check-inputs`，该入口无需加载模型。本地已通过18组、92条请求的哈希和证据引用检查。

验收先按完整配对组汇总。需要同时报告整组排除和生成失败，随后逐条对照23条预先事实填写两臂覆盖。生成调用数、token和耗时按同组比较。质量决策使用事实保留、无证据断言及延迟，当前尚无新claim生成结果。
