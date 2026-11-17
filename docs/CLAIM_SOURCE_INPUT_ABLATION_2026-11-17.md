# Claim 输入来源区分实验

## 目的与当前状态

上轮实际输出表明，模型会把任务背景误当当前阶段证据、把计划更新回执误当测试结果。此次不训练模型、不调整阶段边界，准备同一 7B 的输入表示对照。8 个已审核短阶段、16 条请求已生成，9 项测试通过，远程推理尚未运行。

实现：`segmenter/claim_source_input.py`；输入包：`pilot/claim_source_input_v1/`。原始证据内容、分块、引用 ID、任务背景、系统提示、输出 schema、参考信息点全部保持不变。baseline 使用原输入，source_typed 增加 source_policy 和每步 source_roles。政策说明与来源提示共同改变输入，不能把结果归功于纯标签本身。

## 输入约束

- task_context 是用户提供的理解背景，不是已执行结果，不能借用阶段 ID 为背景中的示例作证。
- 可解析为 command=plan、task_list 数组的动作标为 agent_plan，保留所有原始状态和 notes。done 是 agent 自报完成，不等于测试证实。
- 同一步符合固定回执格式的观察标为 plan_update_receipt，只证明更新和条数，不证明计划里陈述的代码行为。
- 普通 action 表示操作请求；普通 observation 仍需阅读内容，不能把输出中回显的命令、源码或作者描述全部当运行事实。
- action 包含 git commit 时提示可能嵌入 commit_message；这是文本启发式，不是提交成功检测，更不证明改动正确。
- 证据不清的记录不强行细分。当前归一化包没有完整原生工具身份，来源提示不能宣称是可靠的原生工具类型分类器。

评价继续使用冻结 AI 参考：格式有效、引用支持、来源归属、状态准确性、阻塞和验证范围覆盖，以及输出长度/耗时。重点检查上轮失败案例但不只报告改善案例。不可使用格式通过率替代语义正确率。

这是查看上轮失败后设计的已见样本消融，不是盲测。先确认机制是否有帮助；如果改善，再完成其他阶段参考和独立任务验证。没有 SFT 训练集或训练结果产生。

## 远程执行

在已安装 transformers 的环境、仓库根目录运行；先同步 GitHub。即使旧输入预检通过，新输入增加了说明，也要重新计 token。

```bash
cd ~/online-failure-attribution
git pull --ff-only
conda activate agentforesight
python -m remote_pilots.openhands_smoke.seal_claim_diagnostic preflight \
  --root pilot/claim_source_input_v1 \
  --model-path /home/sjtu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --context-limit 32768 \
  --out out/claim_source_input_preflight_v1
```

确认原 7B 服务仍在 30001 端口运行（相同 tokenizer、模板与 32768 上下文）：

```bash
curl --fail --max-time 10 http://127.0.0.1:30001/v1/models
python -m remote_pilots.openhands_smoke.seal_claim_diagnostic run \
  --root pilot/claim_source_input_v1 \
  --preflight out/claim_source_input_preflight_v1/preflight.json \
  --out out/claim_source_input_run_v1
```

若 curl 失败，先按 `remote_pilots/openhands_smoke/SEAL_CLAIM_DIAGNOSTIC.md` 启动服务。最多 16 次调用，基线也重新运行；不采用自动修复或重试。回传整个 `out/claim_source_input_run_v1/`，包含预检、冻结输入、参考和响应。新实验沿用现有硬件，不要求新模型或训练。
