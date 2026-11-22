# 前瞻会话：冻结参考与固定抽取器验收（2026-11-23）

输入来自已采集的 4 个新合成任务，124 条原生事件。先复核原生记录，再冻结本包；没有生成或查看这批任务的 claim 模型输出。它们不是 CodeTraceBench 任务。

## 冻结内容与口径

- `references.json`：46 条 AI 证据事实，包括 33 条关键结果、13 条上下文。每条记录原文、哈希、原生事件 ID、字面锚点、来源和可用时间。agent 的自述保留为 agent_assertion，不当作独立验证。
- `stage_jobs.jsonl`：按当前默认 phase 规则形成的 26 个封口任务。它们是固定生成输入，不是人工 stage 金标。
- 四份事件 JSONL：包括 finish 后状态通知，不包含后验 `verification.log` 或作者参考解。
- `token_preflight.json`：26 个生成块均通过真实 Qwen tokenizer 预检，最大输入 2434 token；总预算 8192，输出预留 512，无输入缺口。
- `protocol.json`：参考与输入哈希。构建脚本禁止静默覆盖已变化的冻结文件；修正需要新版本，不能看过输出后改答案。

40 条事实位于封口阶段，其中关键结果 28 条；6 条事实仍在开放末段，其中关键结果 5 条。两份异常退出会话没有补造 finish。评估应分别报告封口任务的覆盖、全部关键结果的覆盖和开放部分，不能把未触发生成算成成功，也不能隐藏该部分。

注释是有范围的 AI 参考，不是人工金标或完整阶段真值。上下文事实不要求摘要逐项列出；重复失败可以合并，但要保留正确对象、时间、范围和引用。此处没有边界 F1 或 failure alarm 标签。

## 本地验证

`python -m scripts.prepare_claim_prospective_eval` 禁用抽取，只回放原生事件形成封口任务、保存手工编写的 AI 事实并做 tokenizer 预检。`python -m remote_pilots.openhands_smoke.claim_prospective_eval check` 核对哈希、原生配对、参考锚点和封口归属。相关 46 项测试通过。

本轮修复一个原生输入兼容问题：正常 finish 后的已知生命周期通知允许持久化；新动作和工具结果仍被拒绝。新增测试确认封口任务字节不变、恢复后不会重开阶段。后验独立测试只用于任务结果审计，不进入在线参考或模型输入。

## 远程运行

仍使用原 7B base 与 `out/claim_sft_7b_reviewed_v2/adapter`，固定 v4 抽取实现、提示和预算，不训练。当前比较固定 SFT 抽取器与冻结参考，没有新增 base-vs-SFT 两臂实验。

先在采集用 Qwen 工具服务终端 Ctrl+C 释放显存。此脚本直接加载模型，不需要 HTTP 服务；现有 RTX 5090 即可。

```bash
cd ~/online-failure-attribution
git pull --ff-only
conda activate agentforesight
python -m remote_pilots.openhands_smoke.claim_prospective_eval check
python -u -m remote_pilots.openhands_smoke.claim_prospective_eval run \
  --model-path /home/sjtu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --out out/claim_prospective_eval_v1
```

生成器只接收封口任务，参考事实不传给模型。结果保留原始输出、拒绝项和单独的缓存标记；结构 ready／partial 不等于语义评分。回传 `out/claim_prospective_eval_v1/` 整个目录后，再逐条核查引用支持、错误完成声明、关键失败和恢复覆盖。

本入口是固定输入的抽取器验收，不测在线队列延迟、划分准确率或报警性能。四个小型合成任务只支持工程结论，不代表 benchmark 泛化。本包不进入训练；之后若用它调方案，下一轮必须另取验收任务。
