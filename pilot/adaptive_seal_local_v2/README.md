# 局部共同边界窗口实验

本包沿用v1的边界预测、stage范围、封口时刻、原始证据和请求。每个任务拆成相邻共同边界之间的窗口，仅保留内部切点有变化的区间。每个窗口内，两种策略覆盖相同步骤和原始记录。78条请求通过了来源校验。

18个任务得到25个局部窗口。依据同一输入在v1回传中的token计数，13个窗口满足8192 token预算，涉及41条请求。远程预检需要再次确认。35条已有输出可复用，只有6条需要首次生成。复用保留旧失败记录，且会检查adapter、提示、预算和完整请求内容相同。

这次重新分组减少了长窗口对其他局部窗口的连带排除。单个stage输入的长度保持原值，较长的stage仍会超限。25个窗口来自18个已分析任务，按任务聚合才可用于配对不确定性估计。

生成前的23条定向AI事实中，20条落在这些局部变化窗口，已保留来源哈希及父任务ID。剩余事实位于被排除的无切分变化区间。fact_review.json同时保留规则候选，尚待逐条语义复核。新一轮结果可评价局部变化窗口对关键事实的影响。

## 远程执行

先同步仓库。在已有agentforesight环境运行以下命令，默认训练目录为out/claim_sft_7b_reviewed_v2。reuse-results指向上轮远程原始结果目录。

```bash
cd ~/online-failure-attribution
git pull --ff-only
conda activate agentforesight
QWEN7B_DIR=/home/sjtu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
python -m remote_pilots.openhands_smoke.adaptive_seal_claims check-inputs --inputs pilot/adaptive_seal_local_v2
python -m remote_pilots.openhands_smoke.adaptive_seal_claims preflight --inputs pilot/adaptive_seal_local_v2 --out out/adaptive_seal_local_v2 --model-path "$QWEN7B_DIR"
python -m remote_pilots.openhands_smoke.adaptive_seal_claims run --inputs pilot/adaptive_seal_local_v2 --out out/adaptive_seal_local_v2 --model-path "$QWEN7B_DIR" --reuse-results out/adaptive_seal_claim_v1
```

预计预检显示13/25个窗口、41条请求，实际以tokenizer结果为准。执行时复用35条原结果，模型仅生成6条缺失请求。回传out/adaptive_seal_local_v2/。

验收同时报告排除窗口、两个已知失败、剩余生成失败、同一父任务下的多个窗口及事实覆盖。累计耗时会混合历史复用结果和新生成，不能直接视为一次在线运行的延迟。

本地复算为 `python -m scripts.prepare_local_seal_windows`。独立的check-inputs入口已通过25窗口、78请求检查。远程GPU生成尚未执行。
