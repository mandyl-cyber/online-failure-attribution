# 证据优先摘要对照：远程预检与首批运行

当前使用 `pilot/seal_claim_diagnostic_v3/`。v1 有 12/24 个阶段包含空白动作，已从哈希核验的原始日志按 native call ID 恢复 16 个公开工具调用。v2 是中间白名单恢复版本；v3 保留完整公开参数，包括 notes。旧包没有覆盖，模型提示与输出格式没有改变。

目前 8/24 个阶段完成输出前 AI 参考审核，其余 16 个仍待审核。8 个中包含计划、源代码编辑、提交和针对性验证；它们偏短，不代表全部 stage 的质量。不把这些参考称为人工金标，也没有启动训练。`review.json` 的 input_complete 仅表示在提供的归一化输入中未发现空白/缺失，不保证原始工具输出未曾截断。

## 第一步：仅 tokenizer 预检

在仓库根目录、已有 transformers 的环境运行。无需启动模型或占用 GPU。模型路径使用此前 7B 本地缓存；若迁移过缓存请改成实际路径。

```bash
cd ~/online-failure-attribution
python -m remote_pilots.openhands_smoke.seal_claim_diagnostic preflight \
  --model-path /home/sjtu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --context-limit 32768 \
  --out out/seal_claim_preflight_v3
```

请回传 `out/seal_claim_preflight_v3/preflight.json`。它记录全部 48 条请求的精确 prompt token 数、2048 输出预算、配对可容纳状态、tokenizer 文件哈希。32768 必须与服务实际上下文限制一致；不能仅因模型配置允许就假设服务已配置。不会在线下载 tokenizer，也不静默截断。

## 第二步：首批已审核配对推理

首次回传的 `seal_claim_run_v3` 中 16 次请求均为 Connection refused，零模型响应；输入包与冻结版本一致。此运行只反映服务不可达，不能评估摘要。更新后的脚本在创建任务结果前访问 `/v1/models` 并检查模型名，失败立即退出。该检查仅确认服务发现，不保证随后生成成功。

如服务尚未启动，在独立终端运行并保持终端打开：

```bash
conda activate agentforesight
cd ~/online-failure-attribution
python remote_pilots/qwen_tool_server/server.py \
  --model-path /home/sjtu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --model-name qwen-local --port 30001 --context-limit 32768
```

等服务启动后，在另一个终端执行 `curl --fail --max-time 10 http://127.0.0.1:30001/v1/models`，确认存在 qwen-local。保留原失败目录，重跑时使用新的 `--out out/seal_claim_run_v3_retry1`，预检文件可复用。旧失败结果不会被覆盖或自动重试。

只有确认现有 7B 服务使用相同 tokenizer、模板和上下文预算后执行。服务 URL 沿用 30001。实际最多 16 次模型调用，具体取决于两个提示是否都能容纳；剩余请求记为 pending_reference 或 blocked_input，不算模型推理失败。

```bash
python -m remote_pilots.openhands_smoke.seal_claim_diagnostic run \
  --preflight out/seal_claim_preflight_v3/preflight.json \
  --out out/seal_claim_run_v3
```

回传整个 `out/seal_claim_run_v3/`。包含输入、修复来源、参考审核、原始请求/响应、耗时和状态；usage token 数与预检不一致会判失败以识别服务模板差异。ready 只代表结构、引用规则和预算检查，不代表事实正确。结果回来后还需按 reference_points（本包字段为 required_points）做逐条语义评价。

每个请求仅尝试一次，已有结果（包括 reserved/failed）不自动重跑。补完剩余审核、更换模型或更改模板后，使用新的输出目录，不混入首批结果。先保留所有 24 个 stage 的分母和跳过原因，再报告实际推理子集结果。

## 本地验证

```bash
python -m unittest discover -s tests -p test_seal_claim_diagnostic.py
```

预检和修复均为离线准备，不说明延迟封口已经接入在线账本。封口集成与后续 16 个长阶段参考审核仍是待完成项。
