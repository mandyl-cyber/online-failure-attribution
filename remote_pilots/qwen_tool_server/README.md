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

保存两轮响应。只有末尾 PASS 才说明本次模型完成了工具请求和回传，仍须另查 OpenHands 客户端是否要求 streaming/强制调用。服务只监听 localhost；Docker 中如何访问宿主机需在 agent 部署时另外配置。
