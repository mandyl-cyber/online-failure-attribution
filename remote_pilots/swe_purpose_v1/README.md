# SWE 连续步骤目的标注 v1

> 2026-11-16 状态更新：输入包已在 GitHub 分支中。本机按本页执行的预检发现训练 375/935、开发 97/350 超出 32,768 token；训练首五步只有 2/5 通过严格校验。**暂停下方完整标注命令**，先按 `docs/SWE_PURPOSE_PREFLIGHT_AND_TRIAL_2026-11-16.md` 修订并冻结新版压缩/短引文协议；不要复用 v1 输出目录混跑。下方命令保留为 v1 历史复现说明。

已准备训练 25 条轨迹 / 935 步、开发 10 条 / 350 步。每条标全部步骤；训练与开发任务组不重叠。不需要 ZIP，也不需要远程下载原数据集。远程仓库根目录可读取输入包。

## 环境

使用现有 Qwen 指令模型、匹配的本地 tokenizer 和 Python 3.10+ / transformers。脚本连接远程机器本机的 Chat Completions 兼容推理服务，不调用付费外部 API，也不负责安装或启动服务。需要确认服务模型 ID、权重目录、实际上下文窗口和可用显存；不要根据模型名称猜上下文上限。

```bash
export MODEL_ID='填写服务实际返回的模型ID'
export TOKENIZER_DIR='/填写本地权重或tokenizer目录'
export CONTEXT_TOKENS='填写服务实际上下文上限整数'
export BASE_URL='http://127.0.0.1:8000/v1'

python remote_pilots/swe_purpose_v1/train/remote.py inspect
python remote_pilots/swe_purpose_v1/dev/remote.py inspect

for SPLIT in train dev; do
  python remote_pilots/swe_purpose_v1/$SPLIT/remote.py preflight \
    --tokenizer "$TOKENIZER_DIR" --context-tokens "$CONTEXT_TOKENS" \
    --out pilot/swe_purpose_qwen_v1/$SPLIT/first
done
```

先查看两个 preflight.json 的 fits / total。输入包含完整历史前缀，因此仍可能超过模型上下文；预处理没有截断不代表 tokenizer 检查一定通过。若大量超限，先回传报告，冻结新的压缩协议后再跑，不能静默删历史或把超限当 UNKNOWN。

先试跑训练集 5 步：

```bash
python remote_pilots/swe_purpose_v1/train/remote.py run \
  --model "$MODEL_ID" --tokenizer "$TOKENIZER_DIR" \
  --context-tokens "$CONTEXT_TOKENS" --base-url "$BASE_URL" \
  --out pilot/swe_purpose_qwen_v1/train/first --limit 5
```

确认 responses 中输出正常后运行完整第一轮（支持相同配置断点续跑）：

```bash
for SPLIT in train dev; do
  python remote_pilots/swe_purpose_v1/$SPLIT/remote.py run \
    --model "$MODEL_ID" --tokenizer "$TOKENIZER_DIR" \
    --context-tokens "$CONTEXT_TOKENS" --base-url "$BASE_URL" \
    --out pilot/swe_purpose_qwen_v1/$SPLIT/first
  python remote_pilots/swe_purpose_v1/$SPLIT/remote.py validate \
    --out pilot/swe_purpose_qwen_v1/$SPLIT/first
done
```

若模型需要关闭 thinking，preflight 和 run 都添加相同的 `--chat-template-kwargs '{"enable_thinking":false}'`，仅在模型及服务确实支持时使用。配置变化要换输出目录。

将 ENVIRONMENT_TEMPLATE.txt 填写为每个结果目录中的 ENVIRONMENT.txt，记录模型版本、精度、服务和硬件。通过 GitHub 同步整个 `pilot/swe_purpose_qwen_v1/`，不要只提交汇总；不要提交密钥。

## 复核与训练门槛

第一轮返回后进行另一轮 AI 盲复核：仍只看任务和当前前缀，不看初标结果、金标边界或未来。可以使用另一个已有模型按相同指令输出到 `second`；同模型同提示温度 0 重跑不能称为独立复核，不建议为此重复耗费推理。没有第二模型可先回传第一轮，再由接手 AI 复核。

两轮采用兼容结果格式时生成待复核表：

```bash
for SPLIT in train dev; do
  python -m segmenter.collect_swe_purpose \
    --package remote_pilots/swe_purpose_v1/$SPLIT \
    --first pilot/swe_purpose_qwen_v1/$SPLIT/first \
    --second pilot/swe_purpose_qwen_v1/$SPLIT/second \
    --out pilot/swe_purpose_qwen_v1/$SPLIT/review.json
done
```

所有 reviewed 初始均为 false，包括两轮一致的条目。AI 逐项核定 final_label、证据和分歧；缺失不能伪造。训练只使用训练侧已复核样本，开发集只评估/选型；目的模型按任务组交叉拟合生成边界训练特征。保留集本轮不标、不评估。AI 复核不是人工金标，格式通过也不代表语义正确。

包内整条轨迹仅供客户端存储，remote.py 每次只发送当前前缀，不要直接把整份 trajectories.jsonl 交给模型统一标注。远程输入没有阶段金标或错误标签。
