# Qwen目的标注：GitHub同步入口

本目录已包含20条轨迹、735个前缀请求；不用zip，不需要原始数据集。模型每次只接收当前及历史步骤，不接收gold边界。输出是机器候选，不是人工真值。

在仓库根目录git pull后，进入本目录。需要已有Qwen指令模型的Chat Completions兼容服务、与其匹配的本地tokenizer和Python3.10+；使用服务环境已有transformers。服务无需公网暴露。填入真实参数：

```bash
cd remote_pilots/qwen_purpose_v1
export MODEL_ID='实际服务模型ID'
export TOKENIZER_DIR='/实际模型或tokenizer目录'
export CONTEXT_TOKENS='实际服务上下文上限的整数'
export BASE_URL='http://127.0.0.1:8000/v1'
# 可选：export PYTHON_BIN=/实际环境/bin/python
# 仅服务和模板支持时：export CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
python remote.py inspect
bash run_pilot.sh preflight
bash run_pilot.sh run --limit 5
bash run_pilot.sh run
bash run_pilot.sh validate
cp ENVIRONMENT_TEMPLATE.txt results/ENVIRONMENT.txt
# 填写ENVIRONMENT.txt，记录权重版本、量化、GPU、依赖版本和实际命令，不写凭据。
git add results
git commit -m "Add Qwen purpose pilot candidate labels"
git push
```

先检查preflight与前5条试跑，再运行全量。最长前缀约25.7万字符，token数以匹配tokenizer测量为准。超限记context_overflow，禁止服务端/客户端静默截断；大量超限时先同步results/preflight.json，不强行执行。输出最多1200token，预留128token模板余量。若需要鉴权，设置QWEN_API_KEY环境变量，不提交密钥。

原模型精确名称、服务ID、tokenizer、聊天模板、thinking和上下文配置必须一致。未提供固定显存要求；7B不保证任意上下文都能运行。脚本不安装或启动服务，不下载模型。

run支持同配置断点续跑，已验证请求不重复调用。若模型/模板/预算变化，请设置新的RESULTS_DIR再运行，不混用旧结果。部分失败可以提交results，包括errors、audit、preflight、run_manifest和responses；不要只交汇总。历史error文件可能保留已重试成功的记录，以audit与responses为准。

标注定义与反例见SYSTEM_PROMPT.txt，字段见OUTPUT_SCHEMA.json。纯JSON，1–3处精确证据，至少引用当前步骤；未知可弃权。不得把原始整条轨迹直接发给模型一次标完，也不得累积模型回答或额外加入原stage/失败标签。不要使用bundle命令，本次通过GitHub同步结果。

本地阶段真值未包含；同步回来后再独立检查语义及阶段边界。未实际测试你的远程Qwen，先做5条试跑。
