# SWE 目的候选标签 v2：因果压缩、预检和试跑

日期：2026-11-16。v1 的完整前缀在 1,285 步中有 472 步超出本机 32,768 token 窗口，训练集首五步仅 2/5 通过格式校验。v1 包及原始结果保留，不与 v2 混用。

后续 200 条 Codex AI 盲审、轻量目的模型与边界对照见 `SWE_PURPOSE_AI_REVIEW_AND_BOUNDARY_2026-11-16.md`；主指标未见提升。

## 冻结输入与输出协议

新包在 `remote_pilots/swe_purpose_v2/{train,dev}`，复用 v1 已选的 25/10 条轨迹及 935/350 步；`package_manifest.json` 包括原始轨迹 SHA256 和包文件哈希。每个请求只含初始任务、当前完成步、此前**前两步及最近六步**。因此中间历史会被省略，判断所需上下文可能缺失。文本采用首尾截取，并写出被省略的字符数；任务最多原文 5,000 字符，当前 action/observation/intent 分别 2,200/3,000/1,000；历史对应 600/350/250。源数据已有的截断标记继续保留。包内没有金标边界、错误标签或未来步骤。

输出为单个 JSON 对象：`purpose`（七类之一）、可用的当前步 `evidence_id`、最多 240 字符的 `rationale`。证据 ID 指向输入中的原文摘录；模型不再复制长引文。验证器检查类别、字段与当前步证据 ID；**不能据此证明目的语义正确**。

纯 `task_tracking` 列表更新、action 和 intent 均为空时，以明确规则标 `UNKNOWN`，来源记为 `deterministic_control_rule`。Qwen 若把搜索命令输出为非法类别 `SEARCH`，客户端仅在当前 action 含搜索命令时暂映射为 `ACQUIRE`，保留原输出、来源 `qwen_search_alias`，并将 `review_priority` 设为 true；搜索也可能是在检查补丁，因此该映射必须重点复核。其他非法答复拒绝；最多三次尝试，失败后反馈具体校验错误和合法取值。规则、模型答复和重试来源均单独记录，不作为人工金标。

## 本机预检和试跑

使用本机 `Qwen/Qwen2.5-7B-Instruct` snapshot `a09a35458c702b33eeacc393d103063234e8bc28`，同目录 tokenizer，服务 ID `qwen-local`，上下文 32,768，输出预留 256 token 和 128 token 余量。修正了新版 transformers 的 `apply_chat_template` 返回 `BatchEncoding` 时误将键数当 token 数的问题，最终按 `input_ids` 长度计数。完整包预检：训练 **935/935** 可容纳，最长输入 **6,357 token**；开发 **350/350**，最长 **5,888 token**。逐请求数见 `pilot/swe_purpose_qwen_v2/{train,dev}/full_preflight/preflight.json`。

训练首五步 5/5 通过结构校验，但最初 `task_tracking` 步被模型误判为 `ACQUIRE`。跨轨迹初试训练 19/20、开发 9/10 有效；两步纯任务清单更新错误地引用了上一完成步。加入上面的明确控制步规则后，代表样本训练 **20/20**、开发 **10/10** 通过结构校验。完整初标开头又发现 Qwen 自造 `SEARCH` 类别，纠正重试在两例中只修复一例；因此增加来源可追踪的搜索别名处理。两例回归试跑均通过。所有中途协议的输出保留在独立试验目录，不混入正式目录。

抽查还发现训练 `TRAIN_001_s00015` 在修改代码之前运行复现脚本，Qwen 给 `CHECK`；按本研究的标签标准，初始复现应为 `ACQUIRE`。因此**结构通过率不是目的准确率**。全量初标仅作待复核候选；在独立 AI 复核及必要的人工抽审之前，不应用于正式目的准确率或边界增益结论，也不替换在线默认划分器。

## 复核与后续门槛

`segmenter.prepare_swe_purpose_v2_review` 读取完整首轮结果，分别输出 `blind_packets.jsonl` 与 `review_pending.json`。独立 AI 先只看盲审包，给出第二判断后再对照首轮候选。每条 `reviewed` 初始均为 false；优先复核搜索别名、纯控制步、尚未构建候选却标 `CHECK`、轨迹结束前标 `DELIVER` 的例子。机器候选和两个 AI 判断的一致也不等于人工准确率。只有训练侧完成复核的样本可参与小模型训练；开发侧只用于选型和边界比较，保留集不参与。小模型须按任务组交叉拟合产生训练侧在线特征，再在完全相同的边界协议和任务组划分下比较结构基线与结构＋目的的 D0 ±1 F1、误切计数和延迟。

复算完整性：

```bash
python -m segmenter.report_swe_purpose_v2 \
  --package remote_pilots/swe_purpose_v2/train \
  --run pilot/swe_purpose_qwen_v2/train/full_v2 \
  --out pilot/swe_purpose_qwen_v2/train/full_v2/summary.json
python -m segmenter.report_swe_purpose_v2 \
  --package remote_pilots/swe_purpose_v2/dev \
  --run pilot/swe_purpose_qwen_v2/dev/full_v2 \
  --out pilot/swe_purpose_qwen_v2/dev/full_v2/summary.json
```

## 完整初标结果

两包实际运行并经逐请求 ID、输入哈希、运行身份和答案结构复核：**训练 935/935、开发 350/350，有效合计 1,285/1,285；缺失 0、无效 0**。最终结果在 `pilot/swe_purpose_qwen_v2/{train,dev}/full_v2/`，逐包 `summary.json`、`audit.json`、`ENVIRONMENT.txt` 可复核。服务完成后已停止。模型训练和在线默认划分器均未改变。

| 来源 | 训练 | 开发 | 说明 |
| --- | ---: | ---: | --- |
| Qwen 原始合法输出 | 806 | 298 | 仅结构通过；语义未审 |
| 明确控制步规则 | 122 | 52 | 纯任务清单更新为 `UNKNOWN` |
| `SEARCH` 暂映射 | 3 | 0 | 原输出保留，优先复核 |
| Codex AI 裁定 | 4 | 0 | 三次仍非法的 `INSERT`、`VIEW`、`PLAN`；原错误保留 |

训练集四例裁定见 `train/invalid_adjudications.json`：两个 `PLAN` 任务清单更新暂标 `UNKNOWN`，一个代码插入标 `CONSTRUCT`，一个编辑后读回暂标 `CHECK`。后者仍有上下文解释空间，待独立复核。训练成功答复中有 6 次纠正重试，开发有 2 次；三次失败的四例不计为 Qwen 合法答复。

目的候选分布：训练 `ACQUIRE 466 / CHECK 249 / UNKNOWN 139 / CONSTRUCT 42 / DELIVER 20 / OPERATE 14 / ENABLE 5`；开发 `ACQUIRE 179 / CHECK 81 / UNKNOWN 56 / CONSTRUCT 23 / DELIVER 7 / OPERATE 3 / ENABLE 1`。类别严重不均衡，尤其 `ENABLE` 与 `OPERATE` 少，训练小模型前应检查任务组覆盖及稀有类召回。

盲审输入和待审表在 `pilot/swe_purpose_qwen_v2/{train,dev}/review_pending/`：**1,285 条全部 `reviewed=false`**。训练侧有 99 条“尚未构建候选却标 `CHECK`”提示，开发侧 18 条；这些是优先级线索，不是自动错误判断。尚无独立第二轮 AI 语义复核或人工准确率，因此本轮完成的是**机器候选的覆盖率**，不是目的准确率，更不是边界 F1 提升。
