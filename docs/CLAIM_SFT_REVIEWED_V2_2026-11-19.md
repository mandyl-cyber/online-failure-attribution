# 78 个父阶段的 AI 参考与训练导出

## 本轮完成

补完此前 37 个长或多分块父阶段的关键事实参考，与既有 41 个短阶段合并。新数据位于 pilot/claim_sft_reviewed_v2，训练 100 条来自 63 个父阶段、18 个任务组；开发 38 条来自 15 个父阶段、5 个任务组。输出共 240 条参考 claim，180 条 tool_observation、60 条 agent_assertion。任务组间无交集。

37 个长阶段参考位于 pilot/claim_sft_long_review_v1/ai_review.json，按完整证据引用投影为 97 个子块目标。41 个短阶段参考仍在 claim_sft_seed_v1。短阶段目录的 pending=37 表示该子集自身的覆盖范围，当前合并版本协议记录全部选定 78 个父阶段已完成关键事实参考。

## 核对与修正

- 将计划说明和提交描述明确写成 agent 自述，保留尚未验证或环境受限的状态。
- 保留 grep 错误但管道退出0、Python 缺模块但外层退出0、zero-test 与成功用例的区别。
- 将“脚本确认修复”收紧为该脚本实际显示的输出与范围。
- 开发组 timeout 搜索中实际找到 Galaxy 测试，修正教师“test 中完全没有 timeout”的过度概括。
- 保留测试动作失败、字符串替换失败与后续写入成功的顺序。
- 聚合父阶段关键事实，子块答案只使用其自身包含的完整引用。检查发现并修正了跨块引用造成的丢失。

## 导出门禁

scripts.claim_sft_workflow 重建并检查全部导出。门禁检查重复样本 ID、父阶段成员、任务组隔离、引用是否落在当前输入、每个子块是否有目标，以及每条父参考是否进入至少一个子块目标。全部通过，文件 SHA 已写入 protocol.json 与 workflow_status.json。训练回归测试共13项通过。

全部138条参考又经过当前运行时 validate 校验，无守卫冲突，结果在 guard_consistency.json。该项验证引用与规则兼容性。本批为 AI 编写的关键事实参考，exhaustive_semantic_coverage=false 保留了全面语义覆盖尚未独立验证的限制。选择范围为教师输入完整且所有子请求通过的78个父阶段，不能代表全部148个父阶段，也不能据此报告模型语义准确率。

## 远程下一步

训练入口默认使用合并版本。需要原版 Qwen 7B Instruct 权重目录，并先运行真实 tokenizer 预检。通过后运行 QLoRA 与同底座配对生成。当前 GPU 训练、配对生成和独立语义验收尚未执行。

```bash
git pull --ff-only
conda run -n agentforesight python -m remote_pilots.openhands_smoke.claim_sft_train preflight --model-path MODEL_DIR --out out/claim_sft_7b_reviewed_v2
conda run -n agentforesight python -m remote_pilots.openhands_smoke.claim_sft_train train --model-path MODEL_DIR --out out/claim_sft_7b_reviewed_v2
```

MODEL_DIR 使用实际原版模型路径。参考 token 超出输出预算或总输入超限均明确排除，实际合格数量以 preflight.json 为准。评估应在父阶段和任务组层面汇总，不能把138个相关子块当作138个独立任务。
