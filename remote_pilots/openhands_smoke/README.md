# OpenHands 0.59 实时事件试跑

前置：Linux 上 Docker 可用、两个固定摘要镜像已下载、Qwen 工具服务监听 127.0.0.1:30001，并通过容器内 LiteLLM probe。

完整 CodeAct 请求实测输入已达 16,306 tokens，原服务 16,384 总预算无法再容纳 1,024 输出。启动 Qwen 服务时必须显式使用 `--context-limit 32768`，与本控制器配置对齐；修改控制器不会自动修改已运行的服务。服务启动会打印实际有效上限（仍取模型配置上限与参数的较小值）。保持超限拒绝，不静默截断；后续历史继续增长仍可能超限，此次只验证短任务。

在原 Qwen 服务终端 Ctrl+C 后重启（不要另起第二个模型实例）：

```bash
conda activate agentforesight
cd ~/online-failure-attribution
python remote_pilots/qwen_tool_server/server.py \
  --model-path /home/sjtu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --model-name qwen-local --port 30001 --context-limit 32768
```

先同步代码，再重启服务；等待启动成功后在另一个终端运行以下试跑命令。

```bash
cd ~/online-failure-attribution
git pull --ff-only
python remote_pilots/openhands_smoke/run.py
```

只需宿主机 Python 标准库。保持 Qwen 服务终端运行。每次创建独立的 `out/openhands_live_smoke/run-*`，不覆盖之前的结果。不挂载研究仓库到 agent 工作目录。控制器需要 Docker socket 来创建 runtime，runtime 仅配置新的工作目录挂载。

该任务是人为构造的区间合并修复，不是 SWE-bench 样本，也不是边界准确率评测。限制 20 次迭代，关闭浏览器、Jupyter、MCP 和计划工具。真实任务仍需后续单独测试。

结果包含：

- `controller.log`：完整运行日志。
- `state/store/`：OpenHands 原生逐事件持久化；最终 `result.json` 列出实际 events 目录，运行中可用 `find out/openhands_live_smoke -type d -name events` 查找。
- `state/completions/`、`state/trajectory.json`：调用日志及结束后的轨迹。
- `workspace/`：agent 修改后的代码和可见测试。
- `verification.log`：另一个无网络容器使用原始测试副本验证修改结果。
- `state/controller_result.json`：agent 状态；`result.json`：进程、验证退出码及事件目录。

退出码为零要求 agent finished、原始测试通过且找到事件目录。仍应检查事件确实包含命令执行/观察、编辑和验证步骤，不能把测试通过当作实时划分验证通过。原生事件可以随后交给划分器旁路跟读；本脚本尚未启动划分器。

已依据固定版本配置代码编写；本地 Windows 只能检查脚本语法及初始测试确实失败，完整 Docker/GPU 验证需远程执行。

接口依据：[controller](https://github.com/OpenHands/OpenHands/blob/0.59.0/openhands/core/main.py)、[sandbox](https://github.com/OpenHands/OpenHands/blob/0.59.0/openhands/core/config/sandbox_config.py)、[LLM](https://github.com/OpenHands/OpenHands/blob/0.59.0/openhands/core/config/llm_config.py)。

## CPU 结构划分器旁路

在仓库根目录创建独立环境（不改 Qwen 服务环境）：

```bash
python -m venv .venv-shadow
.venv-shadow/bin/python -m pip install -r remote_pilots/openhands_smoke/shadow-requirements.txt
.venv-shadow/bin/python remote_pilots/openhands_smoke/run.py --prompt-mode action-guided --shadow --quiet --collect
```

冻结的 SWE 结构 LR 文件 `swe.joblib` 为本仓库可信模型，校验值与依赖版本见 `swe.json`；4 步窗口、无文本编码器、阈值 0.30。旁路就绪后才启动控制器。agent 失败仍会收集旁路输出；`shadow_exit_code=0` 表示旁路正常退出，不代表 agent 修复成功或边界准确。

每个原生事件到达后按序配对动作与观察，对完成的步骤预测，0.2 秒轮询。输出为 `shadow/stages.json`（边界分数、阶段、待配对动作、来源哈希）及 `shadow/timing.jsonl`（预测器耗时，不是端到端延迟）。不发模型请求，不影响 agent，不生成语义账本。收到 error/stopped/rejected 时末阶段标为 interrupted。暂停/普通停止观察不强制补齐未完成动作。

重新观察同一运行时重放不可变源事件恢复状态，检查哈希并避免重复步骤；要求保留源 events，不支持两个观察进程同时写同一输出。恢复耗时随历史增长，尚未做长会话压力测试。已有目录可单独运行：

```bash
.venv-shadow/bin/python -m remote_pilots.openhands_smoke.shadow out/openhands_live_smoke/run-arxu6xkh --once
```

`--once` 明确标记 recorded_replay。2026-11-16 本地两条真实失败轨迹回放：第一条 15 步/1 阶段，第二条 20 步/5 阶段，重启后边界与阶段完全一致；待配对动作恢复及内容冲突检测测试通过。这些结果没有人工阶段金标，不计算准确率；真正实时旁路仍待上述远程命令验证。

## 提示对照与诊断收集

操作提示对照（同一模型、工具、初始代码、测试和 20 次迭代预算，只追加操作纪律，不提供解题代码）：

```bash
python remote_pilots/openhands_smoke/run.py --prompt-mode action-guided --quiet --collect
```

`baseline` 是默认原提示；两种模式均支持 `--quiet --collect`。完整日志仍保存到文件，结束后自动打包并打印 `DIAGNOSTICS=` 路径，无需复制长日志。Qwen 服务保持 32K 配置，无需为此次提示对照重启。`result.json` 记录实现是否改变、可见测试是否保持原样，修复正确性由原始测试副本独立检查；发生改动不等于正确修复。单次对照只能用于诊断，不作为模型或划分准确率结论。

无需重跑任务，指定已有运行目录：

```bash
python remote_pilots/openhands_smoke/collect.py out/openhands_live_smoke/run-v7gm_ugu
```

终端只输出 zip 路径、大小及收集错误数。将该文件传回即可。包含完整原生事件、可读事件汇总、控制器日志、模型调用 JSON、最终轨迹和小型测试代码；不包含模型权重、Docker 镜像或序列化 session pickle。读取失败会记录在压缩包的 `collection_summary.json`，不会静默忽略。保留原目录，每次使用独立输出文件名。
