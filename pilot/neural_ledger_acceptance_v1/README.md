# 7B神经摘要封口发布验收

固定三份可用真实原生事件存档，共138个事件。它们分别来自SWE、Terminal-Bench和先前的OpenHands真实运行。该选择用于工程验收，失败、成功复验、计划与结果冲突等语义场景仍需逐条复核。回放按原有顺序到达，未完成阶段继续保持开放。

本轮使用同一7B基座及已训练adapter。后台单worker复用模型，沿用训练的短claim提示和证据分块ID，再确定性转换成账本所需的原始记录ID、偏移与quote。观察摘要映射为conclusion类型，Agent声明映射为decision类型，两者仍保持原basis且语义待审。默认8192上下文及512输出token，超限和结构失败保留诊断，最多两次尝试。

本地规则backend预演已完成64个完成step、19个封口任务，全部发布成功，恢复后批次字节一致。神经生成、显存、语义质量及延迟尚待远程验收。

本次固定在线默认phase规则划分，先测神经抽取与账本发布接口。未复用最近结果偏负的自适应封口策略。输出不用于比较监督划分器的边界F1。

```bash
cd ~/online-failure-attribution
git pull --ff-only
conda activate agentforesight
QWEN7B_DIR=/home/sjtu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
python -m remote_pilots.openhands_smoke.neural_ledger_acceptance check-inputs
python -u -m remote_pilots.openhands_smoke.neural_ledger_acceptance run --model-path "$QWEN7B_DIR"
```

回传out/neural_ledger_acceptance_v1/整个目录。summary.json逐会话记录step、封口任务、发布批次、失败及尝试次数、恢复一致性、信号与待审claim、模型生成和显存指标。review_backend/extraction保存原始请求、输出和失败。claim_batches保存正式发布结果，timing保存队列和发布日志。

验收要求每个封口任务有可追溯成功批次或明确失败记录，已发布批次恢复不变，引用校验通过，开放末段无隐式封口。语义逐stage检查失败保留、错误完成断言和证据支持，另行填写复核表。

队列、处理和排队至发布耗时使用相同观察进程的单调时钟，回放间隔默认0.05秒。它反映控制回放的负载，和Agent真实工具耗时不同。该间隔是本轮工程选择，可通过--interval调整，新值应使用新输出目录。中途恢复为优雅等待worker结束后的重启，尚未模拟GPU生成中强制杀进程。模型冷启动包含在首次处理耗时内，同时单列model_load_seconds。
