# 小编码器划分实验执行记录

2026-11-12。对应根目录《小编码器在线划分器调研与实施计划》。本文件区分已完成实验与仍在运行的实验；实时状态以 `eval/boundary_encoder/continuation/status.json` 为准。

后续审计修正：旧转换器遗漏了 1,378 个公开计划动作的任务列表内容，已按原始调用 ID 独立恢复。本文的指纹验证证明输入与旧转换口径一致，并不证明原日志所有字段都被保留。冻结/微调实验继续使用原锁定输入，结论仅限于这个输入条件。即时/延迟对照、缺陷证据与恢复探测见 [CONFIRMATION_AUDIT.md](CONFIRMATION_AUDIT.md)。

## 已完成及当前结论

已建立独立 `.venv-encoder`，安装 CUDA 版 PyTorch 并通过真实 GPU 张量运算。未修改原 Python 环境或系统电源设置。依赖见 `requirements-encoder.txt`，环境记录见 `eval/boundary_encoder/environment.json`。

重建原五折的 461 条轨迹、18,636 步，逐条核对动作与观察内容指纹，全部通过。未扩大样本，也未新增人工标注。输入排除 thought、标签、任务结果和未来步骤；每步独立编码至多 256 token，分类头仅使用当前及之前三步。冻结表示可在全部样本上预计算，所有有监督拟合、阈值与早停选择仍只使用各折训练/开发任务。

MiniLM 冻结表示和六种分类头的完整五折结果如下。F1 是边界指标，不是逐步分类 accuracy；“最终 ±1”允许位置容差，“D0”还要求不晚于金标步提交。

| 方法 | 最终 ±1 F1 | D0 F1 |
| --- | ---: | ---: |
| 上轮固定轻量模型 | 0.5599 | 0.5057 |
| 本轮结构 LR | 0.5576 | 0.5044 |
| 仅结构 MLP | 0.5645 | 0.5149 |
| 新输入模板 TF-IDF + 结构 | 0.5584 | 0.5070 |
| MiniLM 语义 LR | 0.5210 | 0.4656 |
| MiniLM + 结构 LR | 0.5618 | 0.5115 |
| MiniLM + 结构 MLP | 0.5598 | 0.5091 |
| 每折仅按开发 D0 选择分类头 | 0.5590 | 0.5078 |

最后一行相对上轮模型的 D0 增益，按任务/内容连通组 bootstrap 的 95% 区间为 **[-0.8054, +1.2546] 个百分点**，覆盖零。不同头的折外分数用于报告，不据此重新挑选部署模型。暂未发现支持增加编码器开销的稳定收益，当前在线默认行为未变。

按来源也没有明显改善：融合 LR 的 SWE 最终 F1/D0 为 0.5992/0.5462，Terminal-Bench 为 0.5170/0.4700；仅结构 MLP 对应 0.6041/0.5526、0.5167/0.4694。完整分组见 `eval/boundary_encoder/minilm_cv/summary.json`。

CPU 初测使用 4 线程、128 个均匀抽取步骤、batch=1、8 步预热：MiniLM 总耗时 p50=51.98 ms、p99=88.83 ms，包含分词、编码、池化，不含分类头和事件落盘；最小缓存余弦相似度 0.99999857。**测量时后台仍有 ModernBERT GPU 编码，不能当作独占环境或真实 agent 部署结果。** 续跑任务将另存独占运行的测量。现有轻量模型含落盘 p99≈5.94 ms 来自另一回放协议，二者不能当作严格配对加速比。

## ModernBERT 冻结对照（已完成）

18,636 步均已完成编码，实际参数量 149,014,272，revision 为 `8949b909ec900327062f0ebf497f51aef5e6f0c8`，文本指纹与 MiniLM 输入一致。冻结五折结果：

| 方法 | 最终 ±1 F1 | D0 F1 |
| --- | ---: | ---: |
| 仅语义 LR | 0.4824 | 0.4305 |
| ModernBERT + 结构 LR | 0.5633 | 0.5123 |
| ModernBERT + 结构 MLP | 0.5590 | 0.5055 |
| 每折按开发 D0 选择头 | 0.5670 | 0.5162 |

开发选择结果为第一折融合 LR，其余四折纯结构 MLP，因此最后一行不能全部归因为 ModernBERT 语义提升。相对上轮 LR，最终 F1 增益的任务组 bootstrap 95% 区间为 [-0.2875,+1.7379] 个百分点，D0 为 [+0.0742,+2.0275] 个百分点。及时性有小幅改善，但最终 F1 提升未达预设 3 个百分点，也没有稳定的语义收益证据。两种冻结编码器均不进入部署替换。

MiniLM 的独占实验测量也已完成（128 步、batch=1、预热 8 步）：CPU p50/p99=35.29/64.26 ms，GPU FP16 为 11.41/24.43 ms。仍只包含预处理、编码、池化和传输，不含分类头、落盘和排队；不是完整在线部署验收。见 `eval/boundary_encoder/minilm_runtime_isolated.json`，它取代前面的有后台负载初测作为本轮单步参考。

ModernBERT 同协议独占测量：CPU p50/p99=288.60/473.15 ms，GPU FP16 为 178.68/212.04 ms，见 `eval/boundary_encoder/modernbert_runtime.json`。这些是本机当前低频条件下的观测；不能用吞吐或模型参数量推算另一台机器的延迟。冻结前缀缓存任务已启动，首批通过与既有冻结向量的一致性检查；完整微调尚无结果。

## 后续正在执行

冻结对照完成后，续跑器依次执行以下步骤。微调质量结果尚未完成，不应将冻结探测的结论扩展为“微调无效”：

1. 同一五折、同一六种分类头的 ModernBERT 冻结对照（已完成）。
2. MiniLM 与 ModernBERT 的 CPU/GPU 单步耗时测量，避免多个实验同时占用 GPU（均已完成）。
3. 缓存 ModernBERT 前 21 个冻结层的逐 token 隐状态，保留逐步长度和输入版本校验；缓存可以断点续写。
4. 训练最后一层、最终归一化层及四步融合头。编码器学习率 1e-5，分类头 1e-3，每折最多 5 epochs，开发 D0 连续两次不提升早停；主 seed=41。前 8 个训练 batch 会记录资源与时间估计。

冻结前缀缓存不包含标签适配，可以跨折共享；最后一层、池化及历史向量在当前权重下重新计算，不能使用过期的最终向量代替微调。每折从同一公开预训练权重重新开始，外层测试不用于选 checkpoint 或阈值。

已验证拆层输出可重建原模型：CPU 测试余弦相似度 0.99999994，最后一层梯度存在且有限。训练脚本已通过语法检查，但完整五折微调验收须以实际产物为准。

GPU 运行期间观测到 P8、52 MHz、约 5 W，显著低频；原因尚未确定，未擅自更改设置。不能把本机当前测得的速度当作 RTX 4060 的普遍性能。当前后台任务有固定折数、epoch 上限和失败即停止机制；不会自动调用付费 API、测试外部任务或替换线上模型。

## 产物与运行

| 路径 | 内容 |
| --- | --- |
| `eval/boundary_encoder/protocol.json` | 预先锁定的实验矩阵与选择规则 |
| `eval/boundary_encoder/text_v1/` | 已核验的原始单步输入与审计 |
| `eval/boundary_encoder/minilm_headtail/` | 冻结向量、token、逐步截断审计 |
| `eval/boundary_encoder/minilm_cv/` | 已完成的五折模型、预测、汇总和来源拆分 |
| `eval/boundary_encoder/minilm_runtime_cpu.json` | 有后台编码负载的 CPU 初测 |
| `eval/boundary_encoder/continuation/status.json` | 后台续跑状态及当前子进程 PID |
| `eval/boundary_encoder/continuation/*.log` | 各个后续步骤的标准输出与错误 |
| `eval/boundary_encoder/modernbert_cv/` | 完成后生成的冻结五折结果 |
| `eval/boundary_encoder/modernbert_prefix/` | 完成后生成的冻结前缀缓存与进度 |
| `eval/boundary_encoder/modernbert_finetune/` | 微调曲线、资源探测、折模型及折外结果 |

查看状态，不要重复启动同一实验：

```powershell
Get-Content eval/boundary_encoder/continuation/status.json
Get-Content eval/boundary_encoder/continuation/modernbert_finetune.log -Tail 20
```

对应的独立命令如下。输出目录存在时训练器会拒绝覆盖，恢复冻结前缀缓存除外；续跑器正在执行时无需手动重复执行。

```powershell
.\.venv-encoder\Scripts\python.exe -m segmenter.train_encoder_boundary --cache eval/boundary_encoder/modernbert_headtail --out eval/boundary_encoder/modernbert_cv
.\.venv-encoder\Scripts\python.exe -m segmenter.bench_step_encoder --out eval/boundary_encoder/minilm_runtime_isolated.json
.\.venv-encoder\Scripts\python.exe -m segmenter.finetune_encoder_boundary prepare
.\.venv-encoder\Scripts\python.exe -m segmenter.finetune_encoder_boundary train
```

本轮测试已覆盖输入预算、标签/思想文本隔离、未来步骤扰动不影响历史表示、跨会话隔离，原有边界测试通过。尚未完成的验收包括微调质量结论、三种 seed 稳定性、ONNX/INT8、编码器在线 backend 的恢复/回退、真实 agent 长时旁路。它们不能因脚本已写好而计为完成；部署优化应在质量门槛通过后继续。旧 119 条外部轨迹本轮未用于选模型，t_lock 人工标注和 CodeTracer 下游调用也未启动。
