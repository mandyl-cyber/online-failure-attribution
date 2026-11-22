# Claim 证据绑定 SFT v1（2026-11-23）

本轮针对固定模型验收中出现的引用归因、条件改写和摘要过泛问题准备监督数据。保持 7B 基座、原输出 schema、v4 提示和 8192/512 预算，不加新推理模块，不做 RL。训练尚未在本机启动。

## 数据与角色变更

| 用途 | 数量 | 来源 |
| --- | ---: | --- |
| train | 126 | 原 100 条已审参考＋4 个已分析任务的 26 个封口阶段 |
| dev | 38 | 原开发文件逐字节不变 |
| regression | 11 | 先前已使用的纠错／对照输入，不进入训练 |

新增 26 个训练输入保持原 v4 模型提示，目标取自模型输出之前冻结的 40 条封口事实，不复制模型的错误回答。工具结果先引用 observation，再补同一步 action 作为条件上下文；agent 自述保留声明属性。目标覆盖整份输入的已审事实，不只替换一条错误句。6 条未封口事实不强行加入训练。

`evidence_bindings.json` 保存逐事实的主证据与上下文 ID；它用于审计，不是额外模型输入或新的分类头。所有标签来源为 AI 复核，不称人工金标。旧 100 条采用已有 scopefix 版本，包含此前 9、39、42、55 四个窗口的修正；因此本实验是一次联合数据修订，不能把效果单独归因于新增 26 条。

**四个前瞻任务已经开封、分析，现明确退役为训练诊断任务。**原始冻结包和首轮结果保持原样；本候选是派生版本。训练后不能再用那四个任务证明泛化，也不能沿用其“未参与训练”的旧身份。当前没有新的独立验收组。

规范任务映射保存在 protocol 的 aliases 中：原 group 通过既有 selection 和 manifest 追溯，不能直接当作 task_name 哈希。训练与 dev／regression 的规范任务无交集；dev 和 regression 均是已知开发材料，不是盲测。

## 已完成校验

175 份训练／评估参考全部通过实际 Qwen tokenizer 预算检查，最长目标 219 token。开发文件 SHA 与旧版相同。数据文件和来源哈希固定；训练入口会校验哈希与规范任务交集，明确标记 development-only 的数据禁止训练。任一超预算参考都会阻止 preflight／训练，不再静默筛掉后继续。

55 项相关测试通过。本地仅完成数据准备和验证，没有加载 GPU 模型训练，没有新质量分数。

## 训练及匹配对照

训练从原 Qwen2.5-7B-Instruct 基座新建 QLoRA adapter，不继续训练旧 adapter。参数维持 rank 16、alpha 32、dropout 0.05、all-linear、NF4、学习率 1e-4、2 epochs、seed 41；completion-only loss。

比较 baseline 是已部署旁路使用的旧 SFT adapter，不是未微调 base。训练后单独加载一次共享量化基座，分别切换旧／新 adapter，对相同 38 dev＋11 regression 输入重新生成，共 98 次调用。两臂都使用 512 输出预算，逐输入交替顺序；不会拿旧的 1024 预算输出直接比较。原训练脚本的 base-vs-adapter 生成用 `--skip-generation` 跳过，避免额外调用及基线混淆。

现有 RTX 5090 32 GiB 可沿用，无需新权重。先停止采集用 Qwen 服务和其他模型任务，释放显存。此前 100 条、2 epochs 的实测训练约 7.3 分钟；本次训练与配对生成暂估总计 15–30 分钟，非实测保证，实际耗时和峰值会落盘。

```bash
cd ~/online-failure-attribution
git pull --ff-only
conda activate agentforesight
QWEN7B_DIR=/home/sjtu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28

python -m remote_pilots.openhands_smoke.claim_sft_train preflight \
  --data pilot/claim_binding_sft_v1 --model-path "$QWEN7B_DIR" \
  --out out/claim_binding_sft_v1 --max-new-tokens 512

python -u -m remote_pilots.openhands_smoke.claim_sft_train train \
  --data pilot/claim_binding_sft_v1 --model-path "$QWEN7B_DIR" \
  --out out/claim_binding_sft_v1 --max-new-tokens 512 --skip-generation && \
python -u -m remote_pilots.openhands_smoke.claim_binding_compare run \
  --model-path "$QWEN7B_DIR"
```

不要重复覆盖已训练出的 adapter。训练完成、对照中断时，只重跑 compare 命令；已完整保存的配对结果会复用。baseline 默认为 `out/claim_sft_7b_reviewed_v2`，candidate 默认为 `out/claim_binding_sft_v1`。两者基座路径、数据身份和 adapter 哈希会核验。

回传 `out/claim_binding_sft_v1/` 与 `out/claim_binding_compare_v1/` 两个目录。前者含训练配置、metrics 和 adapter；后者含逐输入两臂原文、提示、引用结构检查、耗时和显存。adapter 大文件可沿用原文件回传方式；只同步 GitHub 时至少保留训练配置／metrics 与全部配对输出，不要把 `.safetensors` 强行推到 GitHub。

## 验收顺序

1. 确认实际训练 126、开发 38，配对输入 49，不静默丢弃失败或超限输出。
2. 按相同参考做逐输入 AI 配对复核：引用支持、错误完成声明、关键事实与负向结果覆盖、条件／变化范围。结构通过率和 loss 只作辅助，不是语义质量。
3. 将旧 dev 和已知 regression 分开报告；退役训练任务的拟合结果也另列，不当验收提升。
4. 只有开发／回归对照有明确语义收益后，才采新的独立任务并冻结参考做验收。当前不替换默认运行器或划分器。

本地重建：`python -m scripts.prepare_claim_binding_sft`（需要旧参考与原冻结包）；独立数据检查：`python -m remote_pilots.openhands_smoke.claim_binding_compare check`。本包包含完整训练／评估文件，远程训练不依赖本地未同步的旧候选目录。
