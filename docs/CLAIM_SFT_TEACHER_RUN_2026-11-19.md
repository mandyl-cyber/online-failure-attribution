# Claim SFT 教师候选生成

已有输入共训练 116、开发 32、探索性测试 64 个 stage。当前最终目标均待复核。该脚本只运行训练与开发输入，先验证来源哈希与任务隔离，再使用现有 Qwen32 AWQ 生成候选答案。超出 8192 token 总预算的输入记录为排除，不截断。生成结果保存原始响应、引用校验结果和未审核复核表。

远程执行以下命令，使用原有模型与 vLLM 环境。

```bash
git pull --ff-only
conda run -n agentforesight python -m remote_pilots.openhands_smoke.claim_sft_teacher check
conda run -n agentforesight python -m remote_pilots.openhands_smoke.claim_sft_teacher run
```

回传 out/claim_sft_teacher_v1 全目录。失败请求仍保存原因。已有输出支持同配置续跑，已有 review.json 保留。回传后逐条核对来源、数字、验证范围、关键失败覆盖，修正答案后导出 SFT。已有阶段为离线数据集区间，训练后还需在预测 stage 上验证。

本机 check 已验证 148 个请求重建与训练／开发任务隔离。实际 token 预检及生成需要远程模型 tokenizer 与 GPU。本轮尚未启动 LoRA，当前生成候选的用时应由实际请求记录统计。
