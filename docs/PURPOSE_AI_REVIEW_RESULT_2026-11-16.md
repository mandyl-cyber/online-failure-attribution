# 目的标签 AI 复核对照（2026-11-16）

`pilot/purpose_human_audit_v1/ai_review.json` 已填 60/60，`reviewer_id=codex-ai-review-2026-11-15`，备注逐条标明“AI复核”，相对初稿只改 1 条。**这是第二轮 AI 复核，不是用户或其他人的人工标注，也不能报告目的分类人工准确率。** `annotator_A.json` 目前只有 5 条非空，`annotator_B.json` 为空；不能运行双人金标评分。为保持盲审，含逐例 Qwen 预测的报告只保存在被 Git 忽略的 `eval/purpose_human_audit_ai_review/report.json`。

在分层抽取的 60 条样本上，Qwen 七选项目的预测与 AI 复核标签一致 **14/60（23.3%）**：SWE **3/30（10.0%）**，Terminal-Bench **11/30（36.7%）**。排除 AI 复核标签为 `UNKNOWN` 的 12 条后为 **13/48（27.1%）**。这只描述两个 AI 的一致情况；抽样按来源和 Qwen 类别分层，不能当作 735 步总体准确率。初稿与第二轮 AI 复核高度相关，不能视为两个独立标注者。

分歧最集中在 AI 复核的 `ACQUIRE` 被 Qwen 判成 `CONSTRUCT`（14 条）。Qwen 在 60 条中给出 `CONSTRUCT` 31 次，AI 复核给出 7 次；这一模式与此前融合模型多报切点一致，但**不能证明**误切由目的误判导致。还需要人对这些步骤和对应边界进行核对。AI 复核唯一修改的是 `T12_s00013`，从 `ACQUIRE` 改为 `CHECK`；该步发生在安装前、为排查 fasttext 导入错误而重试，按规范更像 `ACQUIRE`，应优先人工裁决。

下一步：用户若已亲自审过 60 条，请另存人工复核记录，并用真实人工标识与 `reviewer_kind=human` 重新评分；不要把当前 AI 记录直接改名为人工记录。若时间有限，先盲审 Qwen 与 AI 复核不一致的步骤，尤其是 `ACQUIRE→CONSTRUCT` 和 `T12_s00013`，再随机抽查一致案例。边界收益仍须在独立任务组评估，在线默认划分器不变。

复算命令：`python -m segmenter.purpose_human_audit score-assisted --out pilot/purpose_human_audit_v1`。评分器依据 `reviewer_kind` 写明参考标签来源，结果默认放入 `eval/`，不会污染盲审包。
