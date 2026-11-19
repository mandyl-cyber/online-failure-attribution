# SFT 复核中的源输入修复

对 85 个父 stage 的进一步内容检查发现，50 个包含共 76 个空白证据字段。此前“完整输入”表示分块运输覆盖了输入包已有记录。它没有保证输入包自身恢复了原生动作内容。该口径需要在后续报告中明确区分。

已复用原生调用恢复逻辑，根据动作 ID 从源日志取回公共计划参数，并校验已记录的源文件 SHA。148 个训练／开发 stage 中，77 个恢复了共 117 条计划动作。另 5 个 stage 的指定公共调用无法找到，记录在 repairs.json。新包路径为 pilot/swe_claim_sft_repaired_v1，已有输入与回传结果保留。

已知空白字段通过 manifest 的 known_input_gap 标记。运行脚本在生成前排除这些记录。恢复后原始证据和输入哈希发生变化，需要重新生成候选并复核。旧输出仍可作为已知错误回归材料。

```bash
git pull --ff-only
conda run -n agentforesight python -m remote_pilots.openhands_smoke.claim_sft_teacher check --inputs pilot/swe_claim_sft_repaired_v1
conda run -n agentforesight python -m remote_pilots.openhands_smoke.claim_sft_teacher run --inputs pilot/swe_claim_sft_repaired_v1 --partition-long --out out/claim_sft_teacher_repaired_v1
```

回传 out/claim_sft_teacher_repaired_v1。当前仍有单步超长的覆盖限制，参见 partitions.json。新的候选经过来源与关键事实复核后，才可导出 SFT 目标。本轮尚未导出训练数据或启动 LoRA。对无缺口短 stage 的定向证据阅读已进行，全面 85 组最终答案审核尚未完成。
