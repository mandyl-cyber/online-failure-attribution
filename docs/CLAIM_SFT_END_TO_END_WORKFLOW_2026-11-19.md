# Claim 数据复核、训练与验收工作流

## 目标和当前状态

阶段封口后生成有证据引用的多条 claim，并供账本和后续 failure alarm 使用。先固定现有划分器和输入协议，测量训练对抽取器的影响。

当前 148 个父 stage 中有 86 个完整进入教师生成，78 个的全部子请求通过当时校验。恢复了 77 个 stage 的 117 条原生计划动作。已据原始证据编写 41 个短阶段的 AI 参考，训练 35 条来自 18 个任务组，开发 6 条来自 4 个不同任务组。剩余 37 个多步骤或多块父 stage 仍待全面语义复核。当前短阶段种子集可以用于训练流程试跑，质量结论需要扩充覆盖和独立任务验证。

## 顺序与产物

| 步骤 | 执行内容 | 状态与产物 |
| --- | --- | --- |
| 1 | 原生证据恢复、分块覆盖与哈希核验 | 已执行，swe_claim_sft_repaired_v1 与 repaired_return 审计 |
| 2 | 按父 stage 审查来源、数值、失败覆盖与跨块重复，编写参考 | 41 个已完成，37 个待审，claim_sft_seed_v1/ai_review.json |
| 3 | 重建导出、引用与任务隔离检查、保存文件哈希 | 已执行，workflow_status.json |
| 4 | 目标 7B tokenizer 预检，记录每条训练总长度和生成预算 | 脚本已准备，待远程模型路径 |
| 5 | 7B QLoRA，回答 token 计算 loss | 脚本已准备，GPU 执行待验证 |
| 6 | 同输入、同预算的 base/adapter 配对生成和人工或 AI 证据复核 | 自动生成与复核表已准备，执行待验证 |
| 7 | 扩充参考集后重复评估，固定抽取器比较不同划分 | 待前述结果，报告延迟、调用量、关键事实和错误断言 |

复核到的关键案例包括计划声称测试通过、脚本报错但退出码为零、缺 pytest、pytest 的定向通过与范围、提交说明的来源归属、Unsigned dtype 数值和 SymPy 边界行为。AI 参考需要保留具体依据。所有复核标记均表明实际审核状态，未审数据保持待审。

## 本地入口

```powershell
python -m scripts.claim_sft_workflow
python -m unittest tests.test_claim_sft_train tests.test_stage_claim_facts tests.test_claim_input_partition -q
```

本轮数据门禁通过，13 个回归测试通过。回答 mask 验证仅对答案计 loss，遇到不兼容 chat template 显式失败。真实 7B tokenizer、CUDA、bitsandbytes 和 PEFT 组合仍需远程执行验证。

## 训练配置与远程入口

使用原版 Qwen 7B Instruct 本地目录。原版权重加载后通过 bitsandbytes 做 NF4 4-bit 量化，LoRA rank=16、alpha=32、dropout=0.05、all-linear，batch=1、累积8步、learning rate=1e-4、2 epochs、bf16 和 gradient checkpointing。该配置为流程试跑初始值。

现有 RTX 5090 32GB 是目标机器。32B AWQ 权重用于教师推理，训练脚本会拒绝预量化 AWQ/GPTQ 权重。需要兼容现有 CUDA 的 torch，以及 transformers、peft、accelerate、bitsandbytes。运行记录保存实际包版本和 GPU 峰值显存。

获得原版 7B 的实际路径后，将下方 MODEL_DIR 替换为其绝对路径。先释放占用 GPU 的教师服务，使用独立输出目录。

```bash
git pull --ff-only
conda run -n agentforesight python -m remote_pilots.openhands_smoke.claim_sft_train preflight --model-path MODEL_DIR --out out/claim_sft_7b_seed_v1
conda run -n agentforesight python -m remote_pilots.openhands_smoke.claim_sft_train train --model-path MODEL_DIR --out out/claim_sft_7b_seed_v1
```

预检逐条记录真实 token 数，超限样本明确排除，任务隔离在加载数据后检查。训练完成保存 adapter、训练指标、显存峰值，并对合格开发输入生成 base 与 adapter 输出。base 通过禁用同一冻结底座上的 adapter 实现，双方均使用相同的 4-bit 底座。paired_dev_review.json 提供逐条语义复核表。训练 loss 仅用于优化过程诊断。

当前数据只有 35 条训练样本，实际训练步数较少。训练时间和显存应由本次远程日志测量，当前未提供已验证的硬件耗时。脚本支持预检与训练分开执行，配置和数据哈希变化时要求新的输出目录，避免混淆结果。

## 验收口径

先记录 tokenizer 合格／排除数量及任务组，再审查 base/adapter 的证据支持率、关键事实保留率、严重错误成功断言和重复 claim。失败与超限均保留在父 stage 覆盖分母中。建议试验目标为关键事实保留率至少90%、证据支持率至少95%，预设风险样本中严重错误成功断言观察值为0，均报告实际条数与复核方式。

当前开发集属于已经研究过的任务，只适合开发对照。后续新任务冻结后再报告泛化收益。阶段划分验收继续单独保留即时边界 F1、precision、recall、短阶段召回和确认时间，最后检查关键失败事实是否及时入账。

## 技术依据

训练接口采用 Hugging Face Trainer 与 PEFT LoRA，量化和模型兼容性通过目标机器预检验证。参考 [Trainer 文档](https://huggingface.co/docs/transformers/main/en/main_classes/trainer) 与 [PEFT LoRA 文档](https://huggingface.co/docs/peft/main/package_reference/lora)。
