# Qwen2.5 工具调用服务探针

原 TrajDebug 服务未传递 tools、未解析 tool_calls，且始终返回 stop。本目录独立实现最小非流式工具协议，不覆盖旧文件，不执行模型生成的工具。

已本地验证 5 项解析测试；尚待远程 GPU 上验证。参考 Qwen 官方 tokenizer 模板：
https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/tokenizer_config.json

支持 tools、auto/none、历史 tool_calls 参数转换、tool 消息回传，按真实 EOS 区分 stop/tool_calls/length。超预算报错，不静默截断；生成的工具参数只做 JSON 对象及函数名检查，调用方仍须做 schema 验证。不支持 streaming、强制调用、response_format 或自定义 stop；会明确拒绝。不能视为完整 OpenAI API 或 OpenHands 兼容服务。

先通过 GitHub 或文件同步将本目录放到远程仓库，确认文件存在。停止**原 Qwen 服务终端**（Ctrl+C），释放其 GPU 占用；不要停止 Docker/RustDesk。然后：

```bash
cd ~/online-failure-attribution
conda activate agentforesight
python remote_pilots/qwen_tool_server/server.py \
  --model-path /home/sjtu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --model-name qwen-local --port 30001 --context-limit 16384
```

另一个终端：

```bash
cd ~/online-failure-attribution
python remote_pilots/qwen_tool_server/probe.py
```

更新版支持 `max_completion_tokens` 和文本 content parts；两个 token 上限字段冲突时拒绝请求。使用已拉取的 OpenHands 0.59 镜像中实际 LiteLLM 客户端验证：

```bash
docker run --rm --network host \
  -v "$PWD/remote_pilots/qwen_tool_server:/probe:ro" \
  --entrypoint python \
  ghcr.io/all-hands-ai/openhands@sha256:00968de77a7b36546413e1f3b5c6b4e2c387d19ade54de7926e10f5b3d265fa6 \
  /probe/probe.py --client litellm
```

从仓库根目录运行，不挂载 Docker socket。此测试通过也只验证镜像客户端请求，不代表已经运行 agent 控制器或沙箱任务。

保存两轮响应。只有末尾 PASS 才说明本次模型完成了工具请求和回传，仍须另查 OpenHands 客户端是否要求 streaming/强制调用。服务只监听 localhost；Docker 中如何访问宿主机需在 agent 部署时另外配置。
