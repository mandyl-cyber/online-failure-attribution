# 小型时序边界模型：实现、训练与限制

远程推荐入口已整理为 `remote_pilots/temporal_boundary_v1/launch.py`，附带约 1.2 MB 已准备数据、两折 CPU 并行、每 epoch 断点及恢复。完整指令见同目录 README.md；无需再次重建数据。训练复用一次因果编码生成预测状态，等价性和恢复测试通过。优化后同机同折两轮 CPU 计时为 11.52 / 11.12 秒，损失与原两轮相同；旧计时保留供追溯，不能当远程用时承诺。

## 本次实现

主路线为 192 维固定哈希结构/计划特征 → 128 维投影 → 单层因果 GRU → 共享边界头。共 **190,081 参数**（0.190081M、0.000190081B），权重 FP32 约 0.76 MB，不含框架和运行内存。没有生成式解码、Qwen、隐式 thought 或 RL。

编码保留工具类别、路径重叠、退出码、观察测试信号、动作/观察长度和公开计划状态变化。原始文本未进入编码器；因此不能宣称模型已经理解子目标。没有引入七类目的模型或新的人工标签。

GRU 跨阶段连续更新，不因误切而清空。阶段记忆为投影向量的累计均值和长度；已确认边界更新阶段起点，候选最多缓存两个后续步骤，推断时不反复编码全历史。连续状态和阶段池化共同输入边界头。候选左侧、候选处、当前历史状态均来自已观察前缀。

Confidence 控制器按位置顺序处理：高/低分确定，灰区等待，最大两步后必须决定。不设强制最短 stage 长度。延迟模型和即时模型共享同一权重。这里的“置信”目前是未经校准的加权 BCE 分数，不能解释为真实正确率；校准验证仍需完整训练结果。

## 封口和恢复

`OnlineBoundaryPredictor.load` 接受新 `model.json`，继续支持旧 joblib。`run_online` 与 `online_ledger --boundary-model` 可加载新模型。正常结束调用 finish，按已经观察的前缀解决尾部候选，记录实际确认步；异常 abort 不强行确认未决边界，残余段标 incomplete。只在封口后创建 claim job，现有稳定任务 ID 保证重放不会重复录入。

模型状态靠持久原生事件重放恢复，不引入新的快照格式。候选 tensor 缓存有界，阶段列表和上游原始日志仍随轨迹增长；不能据此宣称整个系统恒定内存。训练使用完整单轨迹张量，极长轨迹可能增加内存，当前未做截断 BPTT。

## 数据与监督

已从修复后的原日志重建 **276 条 SWE、9,600 步**。每条新 step ID 和 native raw_ref 与冻结映射核对，全部成功；保留原 group/fold。写入原文件 SHA256、修复后轨迹指纹及准备审计。未把旧 features.jsonl 直接用于新模型。

准备入口 `segmenter.prepare_temporal`，产物 `eval/temporal_boundary_v1/data/`：audit.json、records.json、vectors.npz。records.json 含同组四步 LR 所需结构缓存，约 45.5 MB；vectors.npz 约 0.34 MB。不需要把 Qwen 权重带到训练机器。

这些是此前研究过的开发任务，五折测试不能重新命名为全新盲测。旧 gold 不变，完整轨迹只提供边界目标，前缀以外的事件不进入模型。基准转换可有离线步号对齐处理，不能用此训练数据证明原生实时任务的 F1。

### 训练和阶段状态

用 stage 起点生成二分类标签，首步排除。每个边界训练当前、后续一/两步可见的前缀，按候选出现次数归一权重。AdamW，lr=0.001，梯度裁剪 1，最多 15 epochs，开发损失连续 3 轮不改善早停。

实现将先前设计的“单独 OOF 老师生成阶段状态”简化为 **每轮当前模型在训练轨迹上的无梯度 rollout**：维护预测阶段起点，随后对多前缀边界做监督梯度更新。没有读取 gold 重置记忆，也没有训练额外老师。这是监督学习中的状态模拟，不是强化学习。训练 rollout 使用固定默认确认配置，部署阈值随后由开发集选，因此仍存在一定策略状态差异，需在正式结果中观察，不能声称完全消除了分布偏移。

## 必要对照，只有三条

1. 同组修复输入上的四步结构 LR。
2. 同一新网络，立即提交。
3. 同一新网络，置信度有限确认。

网络架构固定；开发集仅比较 0.25/0.40/0.55 三个分数阈值。即时选择 D0 F1，延迟方案选择共同 D1 F1。LR 开发阈值网格沿用轻量范围。测试集不参与训练、早停、阈值选择。输出精确和 ±1/±2、D0/D1/D2/eventual 指标，以及确认延迟步数。未自动替换在线默认模型；正式五折结果和外部冻结任务验证尚未执行。

## 测试与训练短跑

6 项新测试覆盖：候选同位置重评、有序到期确认、前缀稳定性、推断缓存上限、逐步/批量 GRU 一致、thought 不参与编码、结束不回填确认时刻、延迟阶段的账本恢复与幂等、完整小数据训练及 checkpoint 加载一致性。

真实数据 CPU 短跑：Intel Core Ultra 9 185H，2 线程，PyTorch 2.7.1+cu128，fold 0，训练 5,731 步/开发 1,881 步，epoch 用时 17.13 / 22.56 秒。输出 `eval/temporal_boundary_v1/cpu_benchmark/report.json`。只运行两轮计时，benchmark-only 权重不是部署候选，未读取测试结果、未证明质量提升。

## 硬件与时间

- 推荐先用 CPU：4–8 核、8 GB RAM 可作为准备配置，16 GB 更宽裕；这是工程建议，未实测内存峰值。远程 24 核/125 GB 完全够用。
- GPU 可选，现有 RTX 5090 32 GB 足够此模型；由于当前逐轨迹 rollout 存在细粒度调用及标量同步，不保证 GPU 比 CPU 快。不要为这个模型下载 7B/32B 权重或修改现有 Qwen 服务环境。
- 本机估算每折最多 15 轮约 5 分钟纯训练，另加开发选阈值和测试；单折预留约 5–10 分钟，五折约 25–50 分钟，早停可能明显缩短。远程先跑两轮 benchmark，再按实测估算。以上不是远程测量结果。
- 当前数据准备本地已完成。远程重建耗时取决于文件布局、缓存与解包，不计入上述训练时间。

## 指令

先使用已有安装 PyTorch、numpy、scikit-learn、joblib、PyYAML 的独立环境。无需运行 Qwen。

```bash
# 有原数据、冻结 records/labels 和 manifest 时重建；out 必须是新目录
python -m segmenter.prepare_temporal --out eval/temporal_boundary_v1/data

# 先测本机用时，不进行测试折选型
python -m segmenter.train_temporal --data eval/temporal_boundary_v1/data \
  --out eval/temporal_boundary_v1/cpu_benchmark --fold 0 --epochs 2 --benchmark --threads 2 --device cpu

# 完整五折，最多 15 轮，不做额外架构搜索
python -m segmenter.train_temporal --data eval/temporal_boundary_v1/data \
  --out eval/temporal_boundary_v1/full --fold all --epochs 15 --threads 2 --device cpu
```

每折产物：report.json、predictions.json、baseline.joblib、immediate/model.json、confidence/model.json 及各自 weights.pt。顶层 report.json 汇总三路结果。fold 模型只能用于相应隔离任务评价，尚未训练跨全部开发语料的最终部署模型。

```bash
# 新 run 输出目录，先旁路观察，模型 JSON 和相邻 weights.pt 必须一起保留
python -m segmenter.online_ledger "$EVENTS" --out out/temporal_shadow_v1 \
  --run-id temporal-shadow-v1 --input-mode live \
  --boundary-model eval/temporal_boundary_v1/full/fold0/confidence/model.json
```

正式训练完成前，不能引用 CPU benchmark-only 权重声称 F1 提升，也不能沿用旧 LR 的在线延迟作为新模型延迟。尚待后续：真实在线神经推断计时、置信校准、外部任务验证和 Qwen claim 速度。
