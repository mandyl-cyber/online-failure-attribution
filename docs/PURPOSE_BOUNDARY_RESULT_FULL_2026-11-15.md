# Qwen 目的特征：20 条轨迹完整开发集评估

此结果取代此前仅含 10 条 Terminal-Bench 的正向小样本结论。当前**不替换在线默认划分器**：融合方法在扩充数据后低于相同条件的结构基线，而且七选项目的分数尚无人类语义准确率验证，完整前缀推理也未满足在线时延要求。

## SWE 映射审计

试验包中的 SWE action/observation 经公开意图字段改写，不能与监督记录做完整内容 SHA256 匹配。仓库仍保存原轨迹编码时产生的 `eval/boundary_encoder/minilm_headtail/index.json` 和 `packed_texts.json`。新 `audited_mapping` 先做原有严格哈希匹配；仅对未匹配 SWE，要求来源和全部步号一致，再在独立保存的逐步文本中逐项核对 action/observation 的有界字母数字指纹。准入条件：至少 8 个可比较 action 且全部命中、至少 8 个 observation 且 ≥90% 命中、次优候选 action 命中率 <50%、原轨迹映射唯一。报告保留 `packed_step_verified` 状态，绝不称其为完整内容哈希。

10 条 SWE 的可比较 action 全部命中（44/44、27/27、23/23、34/34、17/17、20/20、30/30、21/21、24/24、33/33）；observation 合计 349/352 命中；次优候选最高 action 命中率约 5.9%。另 10 条 Terminal-Bench 仍为完整哈希匹配。逐项记录见 `eval/purpose_probe_evaluation_v2/mapping.json`，打包文本和索引 SHA256 固定在 `report.json`。由于原始 SWE 全内容仍缺失，此证据是强逐步对应证据，而非字节完全同一的证明。

## 重评结果

沿用事前协议：原任务组折、因果四步窗口、训练/开发/测试分离、折内选阈值；20 条共 735 步、113 个人标边界全部进入分母。Qwen 分数可用于 531 步（SWE 230/352、Terminal-Bench 301/383），超 32K 的 204 步保留为缺失。以下是即时 D0、容差 ±1 步的边界 F1：

| 方法 | F1 | 命中/预测/金标 |
| --- | ---: | ---: |
| 结构 | **0.4350** | 72/218/113 |
| 结构＋仅可用性 | 0.4237 | 68/208/113 |
| 结构＋目的分数 | 0.4011 | 76/266/113 |
| 仅目的分数 | 0.2778 | 80/463/113 |
| 目的变化直接切分 | 0.2569 | 28/105/113 |

融合相对结构 **−3.40 个百分点**；按 20 个任务组重抽 2,000 次的配对预测计数区间 **[−6.77,−0.01] 点**。SWE 中结构 0.4459、融合 0.4048；Terminal-Bench 中结构 0.4253、融合 0.3981。区间只重抽测试预测，未包含重训和阈值选择波动；这仍是已研究的开发集。先前 Terminal-Bench 单独训练得到的 +7.65 点与这里的同来源指标不同，是**训练集扩大后重训和重选阈值**的结果，不可作直接前后效果比较。上述重评没有达到事前 +1 点且区间下限 >0 的门槛。

复现：

```bash
python -m segmenter.evaluate_purpose_probe --probe eval/purpose_probe_v1 --data eval/boundary_supervised/data_fixed --packed eval/boundary_encoder/minilm_headtail --out eval/purpose_probe_evaluation_new
```

已生成两个独立标注者的 60 例盲审包，版本化副本在 `pilot/purpose_human_audit_v1/`（SWE/Terminal-Bench 各 30，单轨迹最多 4 例）。样本只含已得分步骤的截至当前步骤前缀，不含模型预测；`selection.json` 保存抽样协议与输入哈希。两人按 `remote_pilots/qwen_purpose_v1/SYSTEM_PROMPT.txt` 中七类定义分别填写 `annotator_A.json`、`annotator_B.json`，只看 `blind_packets.json`，不看 `selection.json` 中的模型类别计数或探测输出；有分歧时第三人只在 `adjudication.json` 填分歧 ID。之后在原探测结果仍在本机的环境运行：

```bash
python -m segmenter.purpose_human_audit score --out pilot/purpose_human_audit_v1
```

评分器会拒绝空标签、缺项、错误类别、未裁决分歧或更改过的盲包，并输出一致率、Cohen κ、总体/分来源准确率及 Wilson 95% 区间。**目前尚未收到任何人工标注，目的语义准确率仍未知。**既有边界金标不能代替目的类别金标。此前交接记载“暂不新增人工标注”；当前用户要求验证目的准确率，因此只完成可审计的标注准备，人工判断必须由人完成。`eval/` 在 `.gitignore` 中：完整模型分数、映射和逐折预测仍只在本机，仓库有代码、结果摘要和可移交盲审材料。

在线默认划分器保持现状。即使盲审目的类别准确率较高，也须先在独立任务组证明边界增益、解决 204 步上下文缺失和每步推理时延，再考虑在线切换；不能把已下降的开发集方法直接部署。
