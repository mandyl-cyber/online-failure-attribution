# 32B AWQ 阶段摘要能力对照

这不是目的标签/边界实验。复用已成功运行的 Qwen2.5-32B-Instruct-AWQ 权重和 vLLM 环境，只比较6个完整阶段＋7个短诊断窗口的 claim 输出，共13次生成调用。

## 远程运行

确认原来的7B服务已停止、GPU有足够空闲空间。脚本不会停止其他服务，也不安装或下载依赖。

```bash
cd ~/online-failure-attribution
git pull --ff-only
/home/sjtu/miniconda3/envs/agentforesight/bin/python -m remote_pilots.openhands_smoke.claims_qwen32 run
```

默认复用：

- 权重：`vendor/models/Qwen2.5-32B-Instruct-AWQ/`
- vLLM：`vendor/.venv-vllm/bin/vllm`
- 端口：30032，模型服务名：`qwen32-awq-local`
- 上下文8192、输出预算2048、temperature=0、单并发、float16激活、显存比例0.92

路径不同可用 `--model-path`、`--vllm`、`--port` 覆盖。有服务占用端口时直接停止本次启动，不复用不明服务。13次模型请求之外会进行权重校验、tokenizer预检和服务启动，因此总墙钟时间不等于模型耗时合计。

先逐个核验五个权重SHA-256，再用实际本地tokenizer检查所有请求的输入加输出预算。任一超限则保存 `preflight.json` 并停止，不裁剪、不降输出预算。注意原目的实验输出预算为256；本轮为2048，不能套用先前预检结果。

预检兼容 token 列表、BatchEncoding/input_ids 映射和单批张量返回，不以映射字段数充当 token 数。如果旧版本报 `unexpected tokenizer output`，拉取修复后用 `--out out/swe_claims_qwen32_v1_tokenfix` 重跑，保留原失败目录；旧目录的代码指纹不同，不能直接复用。

即使部分摘要格式/引用失败，也继续完成另一组。结束后自动停止本脚本启动的服务进程组，保留日志；其他服务不受影响。不使用逐步LLM分段，不更改默认划分器或账本。

## 回传

传回完整文件夹：`out/swe_claims_qwen32_v1/`。

包含 stages/short 原始请求响应及证据、模型配置和权重校验、token预检、启动命令与服务日志、GPU阶段快照、两组机械统计 `comparison.json`、待审的 `paired_review.json`。GPU快照不是连续采样峰值测量。

部分失败时退出码为1，已有结果仍完整保存；不要因此删除目录重跑。相同命令保留现有作业，不自动重试 failed/reserved，不覆盖审核表。配置变更必须使用新的 `--out`。异常中断后先检查服务是否已退出；SIGKILL或机器断电无法保证清理。

## 对照口径

原始7B历史输出已冻结在 `pilot/swe_claims_qwen32_comparison_v1/baseline_7b.jsonl`，没有修补旧JSON或引用。模型输入、v3提示、schema、温度与输出预算保持一致；模型权重、量化与推理后端不同，因此不能声称是纯参数量消融。

`comparison.json`只报告协议状态、条数和请求耗时，不自动给语义正确率。回传后逐条核验：所有子断言是否被所引证据支持；计划/假设与执行是否混淆；关键条件、失败和验证范围是否保留；冗余和幻觉是否减少。没有生成新标签金标，也不复用目的标签一致率评价摘要。

13条都是已分析的开发材料，不是独立验收集；短窗口与完整阶段存在重复证据，不能视作13个独立样本。强模型改善后仍需要新任务验证；失败则优先定位输入/协议问题，不盲目增加模型或训练规模。
