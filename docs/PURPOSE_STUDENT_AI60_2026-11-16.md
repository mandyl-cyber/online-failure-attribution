# 用 AI 目的参考标签训练 CPU 小模型（试验版）

本实验回答两个问题：现有 60 条 AI 复核标签能否训练无需 Qwen 7B 的本地目的预测器，以及它的实测大小/时延。模型文件为 `artifacts/purpose_student_ai60_v1/model.json.gz`；部署推理代码为 `segmenter/purpose_student.py`，只用 Python 标准库，不加载 Qwen、PyTorch、scikit-learn 或 GPU。训练脚本 `segmenter/train_purpose_student.py` 依赖 scikit-learn。此模型是**可选旁路**，在线默认阶段划分器没有切换。

## 数据与防泄漏

- 训练参考是 `pilot/purpose_human_audit_v1/ai_review.json` 的 60 条 **AI 复核 AI 初稿**，不是人工金标。SWE/Terminal-Bench 各 30；标签数：`ACQUIRE` 28、`UNKNOWN` 12、`CHECK` 9、`CONSTRUCT` 7、`ENABLE` 3、`OPERATE` 1、`DELIVER` 0。模型**无法学到或预测 `DELIVER`**，不能当七类完备模型。
- 输入只含任务文本、当前动作与观察、前一步动作与观察、工具名；字段长度有界。训练与流式推理使用同一输入函数；不使用完整未来轨迹、金标阶段、Qwen 分数或仅在试验包中有而在线 Step 不稳定提供的 `public_intent`。超长前缀不再整体送入模型。
- 固定 5 折按轨迹 ID 分组，折间无同一轨迹，未按测试分数调参。训练集多数类基线也用每折训练标签决定。最后另用全部 60 条训练供旁路运行的模型；其 735 步预测是部署演示/分布检查，**不是 735 条有目的金标的评估**。

| 在 60 条 AI 参考上的轨迹留出预测 | 准确率 | 六个已出现类别的 macro F1 |
| --- | ---: | ---: |
| 小目的模型 | **0.5333**（32/60） | **0.3379** |
| 训练折多数类 | 0.4667（28/60） | 0.1061 |
| 原 Qwen 七选项（同 60 条的 AI 参考一致率） | 0.2333（14/60） | 未在本协议中计算 |

小模型在 `ENABLE`（3 条）和 `OPERATE`（1 条）上的留出召回均为 0；`DELIVER` 无训练例。这是类别覆盖不足的明确信号。与 Qwen 的数字也**不能解释为人工准确率提升**：标签来自 AI，样本由来源和 Qwen 类别分层抽取；小模型用这些标签训练，Qwen 没有，比较只说明这个学生在当前 AI 参考上更贴合。还没有用小模型目的预测重评边界 F1，不能声称划分效果改善。

分来源差异很大：SWE 留出为 **24/30（80.0%）**，Terminal-Bench 仅 **8/30（26.7%）**。总体 53.3% 掩盖了后者的失败；因此即便只作目的预测，这个试验模型也不适合直接上线。

## 模型体积与速度

最终模型是字符 3–5 元 TF-IDF（8,000 特征）加逻辑回归，压缩文件 **347,607 字节，约 340 KiB**。纯 Python 加载的当前对象约 **2.6 MB**（`tracemalloc`；加载峰值约 5.37 MB），一次加载约 33 ms。对试验包全部 735 个步骤串行 CPU 推理，中位 **1.32 ms/步**、p95 **1.74 ms/步**、最大 **2.07 ms/步**；这些是本机模型计算耗时，不含事件转码、磁盘 I/O 或完整在线服务。导出后的纯 Python 概率与训练时 scikit-learn 概率最大差约 `7.6e-15`。

在线旁路使用（已知任务文本保存为 `task.txt`）：

```bash
python -m segmenter.run_online replay <轨迹目录> \
  --purpose-model artifacts/purpose_student_ai60_v1/model.json.gz \
  --task-file task.txt --out out/purpose_shadow
```

输出 `purpose_events.jsonl`，同时每步事件行有 `purpose_prediction`；**不修改** `stage_ranges.json` 或边界决策。若原始事件流自带任务文本，可省略 `--task-file`；若没有，会明确报错。默认不加 `--purpose-model` 时，线上路径既不加载这个学生模型，也不调用 Qwen 7B。部署若显式采用此小模型，目的预测只需标准库和约 340 KiB 权重文件。

复训命令（本机用 scikit-learn 1.7.2、numpy 1.26.4）：

```bash
/home/sjtu/miniconda3/envs/RoboTwin/bin/python -m segmenter.train_purpose_student --out eval/purpose_student_new
```

`eval/purpose_student_v3/report.json`、5 折预测与 735 步旁路预测保留在本机且被 Git 忽略；GitHub 有代码、模型文件、方法和结果摘要。要考虑真正接入边界模型，还需要补全稀有类别、人工核查目的语义，并在不泄漏目的训练轨迹的独立任务组上比较边界收益；不能用当前 60 条 AI 标签与同 20 条边界测试集调参后声称独立验证。
