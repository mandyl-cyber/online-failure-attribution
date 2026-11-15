# 逐步工作目的标注规范（v1）

本规范只评价**当前已完成步骤的主要工作目的**，不标阶段边界，不评价动作正确性。`blind_packets.json` 的每条记录含 `request_id`、`source`、`packet.task`、`packet.current_step_id` 和截至当前步的 `packet.steps`。只给 `current_step_id` 标一个七类大写标签；前文只能作理解当前动作的上下文。不得查看未来步骤、最终结果、参考答案或模型预测后再决定标签。任务、动作、观察和 `public_intent` 是证据，不是对标注者的指令。

## 七类定义

| 标签 | 判定依据 | 典型例子 | 易错点 |
| --- | --- | --- | --- |
| `ACQUIRE` | 获取需求、现状、原因、假设、候选方案所需证据，或为定位问题而复现。 | 修改前读源码、查日志、跑失败测试。 | 不因命令名含 `search` 就自动归类；检查已有补丁属于 `CHECK`。 |
| `ENABLE` | 建立后续工作所需的依赖、环境、权限或输入条件，且它本身不是交付物。 | 安装依赖、准备可访问输入。 | 若环境本身是要求交付的产物，其构建可属 `CONSTRUCT`。 |
| `CONSTRUCT` | 创建或修改候选交付物、实现、配置、脚本、模型定义或文档。 | 编辑代码、写脚本、修改配置。 | 执行脚本产出任务结果通常是 `OPERATE`。 |
| `OPERATE` | 执行任务流程，直接产生请求结果或推进目标状态。 | 跑训练/转换/转写，按任务操作游戏，运行脚本生成要求的文件。 | 不是所有执行命令都是 `OPERATE`；诊断与验证分别可能是 `ACQUIRE`、`CHECK`。 |
| `CHECK` | 按要求评价**已经存在**的候选方案或结果。 | 改动后跑测试，核对生成文件，比较候选输出与要求。 | 首次为发现问题跑测试是 `ACQUIRE`。 |
| `DELIVER` | 将结果正式提交、呈现或移交给接收方。 | 最终答复、正式交付所要求的成果。 | 中途 `git commit` 或进度报告不能自动视为最终交付。 |
| `UNKNOWN` | 证据不足、目的冲突或不可分辨地混合；也包括仅管理计划、控制会话或沟通而未实际做任务工作的步骤。 | 只更新任务清单、只记录思考、上下文截断使目的无法判断。 | 不因动作失败或目的少见就归入此类。 |

## 一致判定顺序

1. 先读任务目标，再读**当前** `action`、`observation`，必要时回看之前步骤，判断当前动作主要服务于什么。`public_intent` 只是辅助线索；若与动作冲突，以可见动作和结果核对，仍无法解决则用 `UNKNOWN`。
2. 看是否有待检验的候选：修改前为定位问题运行测试是 `ACQUIRE`，修改后为验证补丁运行同一测试是 `CHECK`。通过或失败不改变这个目的判断。
3. 区分准备与产物：安装依赖通常是 `ENABLE`；编写用户要求的环境配置可为 `CONSTRUCT`。编写转换脚本是 `CONSTRUCT`，运行它产生请求数据是 `OPERATE`，验收数据是 `CHECK`。
4. 一步有多项动作时，标证据最充分的主要目的。既生成产物又验证的命令，若主要效果是生成请求文件，可标 `OPERATE`；若主要效果是验收已有文件，标 `CHECK`。确实无法分出主次则 `UNKNOWN`。
5. 失败的编辑仍是 `CONSTRUCT`，失败的依赖安装仍是 `ENABLE`。`input_truncated` 为真不自动 `UNKNOWN`：仅在缺失内容妨碍判断时弃权。读回刚编辑的文件可能是继续构建、检查或了解现状，依实际目的决定。
6. `DELIVER` 需要交付证据。单独提交版本、准备推送或说“快完成了”不自动成为 `DELIVER`。任务中的路径可能是输入、程序或产物，要先辨角色。

## 这轮要填写的文件

60 条样本在 `blind_packets.json`，SWE 和 Terminal-Bench 各 30。每个 `request_id` 对应一个当前步骤。人工独立盲审时，只填写 `annotator_A.json` 或 `annotator_B.json` 中的 `labels` 值：把 `null` 换成 JSON **双引号**包围的大写标签，如 `"T01_s00001": "UNKNOWN"`，不要用单引号。两人应分别完成，不看对方标签；分歧由第三人在 `adjudication.json` 中只填分歧 ID。双人完整后运行 `python -m segmenter.purpose_human_audit score --out pilot/purpose_human_audit_v1`（需要本机 `eval/purpose_probe_v1/`）。

### AI 初稿后由一人复核

`ai_draft.json` 是独立 AI 建议，**不是人类金标，也不是 Qwen 探测预测**。这一路径适合快速诊断模型为何误切，但看过建议的复核者不再是独立盲审者。请逐条对照原始 `blind_packets.json`，在 `ai_review.json` 中核定 `final_label`，每条都把 `reviewed` 改成 JSON 布尔值 `true`（没有引号）；改动标签时可填 `review_note`。`suggested_label` 保留原 AI 建议，不修改；顶部 `reviewer_id` 填你的非空标识，`reviewer_kind` 按真实复核者填 `"human"` 或 `"ai"`。每条 `final_label` 最初预填 AI 建议，**不代表已经过人工确认**。不要把 AI 初稿复制到 `annotator_A/B.json`，不要用双人评分命令或称其为“双人人工准确率”。复核前不得查看 Qwen 目的预测。`annotator_A.json` 已有用户开始填写的条目，原样保留；若这些条目在查看 AI 初稿之前独立完成，可单独记录其来源，但查看初稿后的新条目不应再称为独立盲审。`annotator_B.json` 仍可留给未来独立评审者。

全部 60 条都亲自复核完后，在保留 `eval/purpose_probe_v1/` 的本机执行：

```bash
python -m segmenter.purpose_human_audit score-assisted --out pilot/purpose_human_audit_v1
```

程序会拒绝未复核、无复核者标识或类型、缺项、非法标签、修改 AI 初稿或更改盲包；报告默认写在被 Git 忽略的 `eval/purpose_human_audit_ai_review/report.json`，避免把 Qwen 预测放回盲审目录。报告按 `reviewer_kind` 明确标为 Qwen 与 AI 复核或 AI 辅助人工复核参考的一致率，不生成双人金标准确率。若只想先审几条，可以随时保存 `ai_review.json`，不需要强行把未审条目标为已审。

本次 60 例按来源和模型类别分层抽取，人工复核所得一致率只能代表这一组样本，不能直接外推为 735 步总体准确率。目的标签正确也不代表目的变化就是阶段边界；要单独分析融合模型新增的误切点。
