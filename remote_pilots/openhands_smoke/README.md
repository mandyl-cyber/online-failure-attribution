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
