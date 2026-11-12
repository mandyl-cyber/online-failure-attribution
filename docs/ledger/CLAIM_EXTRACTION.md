# 阶段 claim 抽取 dry-run

2026-11-13。模块 `segmenter/claim_extraction.py` 已接入在线后台队列。此版本只有本地规则 backend，不调用 API，也不代表语义抽取质量已经验证。

## 运行

```powershell
python -m segmenter.online_ledger <events.jsonl或events目录> --out out/live_claims_dry --run-id session-001 --claim-extractor dry-run
python -m segmenter.claim_extraction --jobs <已有stage_jobs目录> --out <独立输出目录>
python tests/test_claim_extraction.py
```

默认 `--claim-extractor rules` 保持旧规则输出；dry-run 使用新协议。在线运行目录固定抽取器配置，不能中途切换。独立 dry-run 目录可以重跑同样输入复用结果，输入或抽取配置变化必须换目录。

## 输入与输出

封口时冻结任务上下文（实际收到的用户消息，缺失则 null）、阶段区间、确认时刻、原文证据和规则候选提示。原始路径、thought、标签和 candidate attrs 不进入请求；任务/日志文本仅作为数据。历史作业缺少任务文本时不从最终日志补写。

请求包含版本化指令和 JSON Schema。每条 claim 输出 local_id、kind、text、basis、objects、conditions、unknowns、evidence_refs；每个引用包含 evidence_id、Unicode 字符起止位置及逐字 quote。对象版本和环境允许 null，允许零条 claim。

校验器检查字段类型和额外字段、唯一 ID、枚举、数量、证据归属、阶段范围、确认时间、证据 SHA256、字符区间与逐字引文。它不检查语义蕴含，不证明断言、对象版本或适用条件正确。规则 dry-run 仅使用已有候选及短证据摘录，不是 LLM 语义抽取。

## 执行与预算

- `StageProcessor` 为可替换后台 callable，backend 接口为 `extract(request)`。
- request/status/通过校验的 response/validation 保存在 `extraction/<stage>/`；claim 批次通过校验后才原子录入。
- 最多两次尝试，调用前先持久化计数；重启不重置预算。失败不提交 claim，返回 failed 状态，在线接收可继续。
- 缓存身份包含作业、协议、提示、Schema、backend 身份和预算。失败作业仍保留审计；输入配置冲突会报错。
- 默认输入 1,000,000 字节，输出 128,000 字节，最多 64 条 claim。超限显式失败，不静默裁剪。
- 字节限制不等于 token 或费用预算。尚无远程传输、API 超时取消、美元费用控制、自动退避或全局并发限流；不能直接把任意远程 backend 当成已完成预算管控的 API 实现。

## 验证

`eval/claim_dry_run_v1/` 保存 SWE 6 个阶段、Terminal-Bench 7 个阶段的 dry-run：13 个作业 ready、0 failed、18 条 claim，network_calls=0，semantic_correctness=not_evaluated。抽取测试验证输入校验、伪造引用/引文、缓存、持久化重试上限及在线后台接线；原有五项在线行为测试通过。

下一步需要确定服务商、模型版本、外发范围及真实费用上限，再实现远程传输和语义质量实验。支持/依赖边、跨阶段去重修订和最终失败归因尚未实现。
