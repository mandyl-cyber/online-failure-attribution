# 有监督在线阶段边界：实施与结果

2026-11-12。已完成数据构建、候选规则修复、五折分组训练、独立任务评价和本地流式验证。未进行真实 agent 长时运行或 CodeTracer 模型诊断。

## 实现

- `stage_dataset.py`：读取现有 stage 起点生成标签，保留原步号和 raw_ref；任务名规范化及完全相同轨迹指纹组成隔离组，五折分组。
- `boundary_features.py`：仅当前及最近 3 步的工具类型、命令类别、退出码、路径重叠、修改间隔等；不读取 thought、gold stage、错误标签、最终结果或未来步骤。缓存长度有上限。
- `boundary_model.py`：稀疏 Logistic Regression；实验比较结构特征及可选 TF-IDF。最终候选为四步结构特征，C=1，阈值 0.225，无生成式模型、GPU 或 API 调用。
- `eval_online.py`：边界位置与实际确认时刻分别评分；晚到预测仍进入 precision 分母。D0 要求确认时间不晚于匹配的金标边界步，不等同于只检查预测自身是否零回看。
- `train_boundary.py`：五个外层评价折；每次取另一折作开发集，其余三折训练。配置、阈值只由开发折选择。五份折外结果合并；最终候选按各折选择最多的配置和对应阈值中位数在开发语料全集训练。
- `run_online.py --boundary-model`：可选接入，同步输出 `boundary_events.jsonl` 和 `supervised_stage_ranges.json`；原 `stage_ranges.json` 仍是规则输出。新模型未设为默认。

修复 `segment.py` 中候选阶段切换却保留旧起点的错误。回归测试使用“修改→短暂排查→验证”，修复前失败、修复后通过。修复会改变规则结果，所以新旧规则均保留同样本对照。

## 数据质量与实验范围

verified 的 520 条 OpenHands 中：8 条无路径、14 条既有步号异常，另有 37 条在声明的标注范围后仍出现非 FINISH 动作。本轮不假定这些额外步骤可以安全截断，保守排除，剩 461 条、18636 步、424 个任务/内容隔离组。它们是已研究过的开发语料，交叉验证不冒充全新测试集。

37 条新增排除只是“完整覆盖无法证实”，并不证明此前每条轨迹的前缀都错位。原因及 ID 见 data_fixed/audit.json。训练/评测均使用同一批严格检查后的样本，因此不能直接与此前 498 条上的 0.4866 作增益比较。

外部评价先排除整个 verified 中出现的任务名，再检查内容指纹：full 中 138 条候选，119 条通过对齐，3838 步、50 组；SWE 14、Terminal-Bench 104、来源未知 1。模型与阈值在外部评价前固定，结果未用于再次选择。任务名规范化与精确指纹不能保证识别所有语义近重复任务，这仍是泛化结论的限制。

离线转换依然可能使用 manifest 步数选择运行或折叠消息，这是 benchmark 适配流程；模型不使用该字段。当前结果证明归一化步骤上的因果划分效果，不能自动推广为原始在线事件的步号对齐已经解决。

## 结果

以下为 micro F1，容差 ±1 步，一对一匹配。所有值来自已落盘结果。

| 数据 | 方法 | 最终边界 F1 | 不晚于金标步确认的 F1（D0） |
| --- | --- | ---: | ---: |
| 开发语料五折折外（461 条） | 原规则 phase | 0.4898 | 0.3068 |
| 同上 | 修复规则 phase | 0.4780 | 0.3385 |
| 同上 | 训练集确定间隔的等距基线 | 0.4628 | 0.3338 |
| 同上 | 监督模型 | **0.5599** | **0.5057** |
| 外部任务（119 条） | 修复规则 phase | 0.4914 | 0.3592 |
| 同上 | 固定监督模型 | **0.5811** | **0.5203** |

规则修复使切点语义更符合实际候选开始时刻，D0 改善，但最终容差 F1 略降；不能将它单独声称为最终 F1 提升。

五折 D0 相对修复规则提升 16.72 个百分点，任务组 bootstrap 95% 区间为 [14.94,18.55]。外部 D0 提升约 16.11 点，区间为 [12.54,19.10]。这里的区间仅针对 D0，不能挪用于最终边界 F1。

外部最终边界 F1：SWE 0.6263→0.6720，Terminal 0.4790→0.5712；来源未知的 1 条为 0.8571→0.5000，未从总体中删掉。SWE 外部样本只有 14 条，不夸大稳定性。

工程推进线“D0 提升至少 5 点且置信区间为正”满足；原设计最终 F1≥0.70 尚未达到。阶段效果不等于 t_lock 事件召回，也不等于下游诊断或任务成功率提升。

## 流式与开销验证

`check_boundary_runtime.py` 在 10 条、460 步上使用各自未训练该任务的折模型，验证批处理概率与逐步推断一致；若干截断前缀得到相同的已提交边界；输出阶段完整覆盖。未训练模型不会在运行中读取标签。

| 本机单进程指标 | p50 | p99 |
| --- | ---: | ---: |
| 模型特征与推断 | 0.557 ms | 2.116 ms |
| 完整 feed，含规则、模型及本地落盘 | 1.262 ms | 5.941 ms |

上述 feed 计时不含日志读取和适配器。另用真实文件追加模拟 8 个事件、4 步，逐条拆成半行写入，5ms 轮询下从写入到 feed 完成的 p99 约 7.43ms。该样本太小，只能说明接线与半行读取可用，不能视为生产端到端时延保证。

未完成：真实 agent 长时旁路、多会话压力、重启断点恢复、迟到/重复事件系统性测试、峰值内存和超长观察压力测试。旧 follow 的默认轮询仍为 0.5 秒；仍会以写模式打开输出文件，因此不要对同一输出目录重启以期恢复。

## 使用与复现

训练依赖安装在仓库 `vendor/boundary_deps`，当前版本 scikit-learn 1.9.1、numpy 2.5.4；不改变规则默认依赖。序列化模型与这些版本配套；只加载本项目生成的受信任 joblib 文件。

```powershell
# 新目录构建当前修复规则的数据，不覆盖既有实验
python -m segmenter.stage_dataset --out eval/boundary_new/data
python -m segmenter.train_boundary --data eval/boundary_new/data --out eval/boundary_new/experiment

# 在一个新的输出目录旁路跟读正在运行的 OpenHands 会话
python -m segmenter.run_online follow <session>/events --out out/live_boundary_001 --boundary-model eval/boundary_supervised/experiment/candidate.joblib
```

训练脚本默认会读取本轮冻结的旧规则数据作为对照；若新数据集样本改变，需要显式传对应 `--legacy-data`，或者传不存在的目录以省略该对照，不能把不同样本硬拼。

产物：

- `eval/boundary_supervised/data/`：修复前基线与监督数据。
- `eval/boundary_supervised/data_fixed/`：修复后同样本、特征、标签、映射和分组。
- `eval/boundary_supervised/experiment/summary.json`：五折完整指标、各来源结果、配置选择及版本。
- `eval/boundary_supervised/experiment/external.json`：外部任务评价，不参与调参。
- `eval/boundary_supervised/experiment/runtime.json`：本地流式检查与计时。
- `eval/boundary_supervised/experiment/candidate.joblib`：可选接入模型；fold_*.joblib 为各评价折模型。

下一步应先解决真实事件流的幂等与恢复，再部署旁路会话；更高准确性实验必须另设开发协议，不能继续用这次外部结果反复挑参数。CodeTracer 下游对照仍需模型/API/预算和运行环境；旧的 40 对规则输入不能代表本轮模型。
