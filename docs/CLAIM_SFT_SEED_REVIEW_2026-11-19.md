# 首批 Claim SFT 参考答案

从恢复后的原始 stage 证据中，已逐条阅读并编写 41 个较短阶段的 AI 参考答案。沿用原任务分组，训练 35 条、开发 6 条，组间无交集。源输入、输出格式和全部引用通过校验。

答案保留测试阻塞、异常、退出码、修改内容和实际脚本结果。计划中的完成状态与验证报告明确归属 agent，提交说明单独标注为 agent 对改动的描述。典型案例包括 FITSDiff 修复前异常及修复后自比较、call_command 定向探针、AppRegistryNotReady、缺 pytest、Unsigned 解码的具体 dtype 和数值、带临时文件清理的提交。

导出文件为 pilot/claim_sft_seed_v1/train.jsonl、dev.jsonl，逐条复核答案在 ai_review.json。脚本 scripts/export_claim_sft_seed.py 可重建全部文件。参考为 AI 复核答案，没有人工审核。样本按证据可完整阅读的短阶段选择，代表性和长输入覆盖有限。原 78 个全部生成通过的父 stage 中尚余 37 个未完成最终参考，尤其需要检查多块之间的遗漏与重复。

新增覆盖包括退出码为零但正文报错的 Node 复现、缺 pytest、IAM 配置警告分支、Kinesis DependsOn 双用例、SymPy 子群探针、有限非实坐标与无穷值处理，以及带 UserWarning 的 geometry pytest 输出。所有成功结论限定到实际显示的命令范围。

当前训练数据规模较小，适合先核对数据及训练流程。尚未启动 LoRA，也没有模型提升结果。
