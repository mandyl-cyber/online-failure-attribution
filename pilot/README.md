# M2 标注 pilot：工作协议 v0.1

目标：CodeTraceBench 40 条 + LongRCA 40 条，由两位人类标注者独立阅读。
本协议是项目操作化草案，未声称复现 FaaP 原始 codebook；正式标注前需小规模校准。
CodeTraceBench 包按来源 × 成功/失败分层各取 10 条，每个任务只选一个运行。
这是诊断性配额样本，不用于估计自然分布下的总体失败率。

## 文件与盲标流程

运行 `python -m segmenter.pilot --out pilot/m2_codetrace`。
已有输出目录会拒绝覆盖，避免擦掉人工标签。
`trajectories/` 包含完整规范化动作、观察、thought 和原文指针；
`A/`、`B/` 是两份独立空白 JSON 标注模板。`reviewer_only/` 仅供协调者使用，
包含成功标签、既有 incorrect 标签与规则候选。第一轮不要把这个目录发给标注者。

两人先在独立的少量校准样本上讨论定义，再冻结协议；正式 80 条独立标注，
不相互查看结果，不预先展示规则候选。完成后锁定原始 A/B 文件，另存仲裁标签。
最后才用规则候选核对覆盖率，避免“只在候选处找锁定”导致循环验证。

## 标注单位

每条轨迹可有零个或多个错误实例。不要强制把每个 incorrect 步当成独立错误，
也不要强制存在锁定。`errors` 填写实例列表；`error_template` 只是字段说明模板，
填写完成后设置 `status=complete`。干净轨迹填写空列表并说明判断。

- `claim`：具体错误结论或错误状态，写清内容，而非“代码有问题”。
- `t_err`：该错误首次引入的步号，须有动作或观察依据。
- `t_lock`：首次明确将这一错误当前提接受/使用的步号，可等于 t_err。
  仅出现同名文件、仅命令退出码为零，均不足以认定接受。
- `lock_type`：reference / handoff / verify_pass / irreversible；
  若只存在“继续构建”，填 continued_build，单独分析，不混进主定义。
- `lock_status`：observed / not_observed / uncertain。后两种 t_lock 留 null，
  不把轨迹终止步补作锁定点，不把无法确认称为没有锁定。
- `t_obs`：错误首次出现可观察证据的步号；没有证据则留 null。
- `outcome`：repaired / persistent_harmless / decisive / uncertain。
  最终任务失败不能自动证明每个错误都是 decisive。
- `evidence`：列出 step、field（thought/action/observation）、原文引句及解释，
  分别支持错误、接受与结果判断。`confidence` 为 high/medium/low。

验证通过后何时被接受必须看后续行动证据，不能照抄规则的补发位置。
交互终端里的密码、菜单输入可能不是 shell 命令；脚本说“saved”也不是独立文件系统证据。
若末尾多出一个无金标的 finish 步，可阅读并标注，但在 notes 记录它不属于已有 incorrect 标注范围。

## 分析计划（需完成双人标注后执行）

先按错误内容与证据对齐实例，报告未匹配实例数，不能只保留两人一致的实例。
主报告为锁定是否存在的 Cohen κ、两人均判 observed 时的精确步一致率、
±1 步一致率和绝对步差分布；步号差不能直接当普通类别 κ 的唯一结论。
缺失值与 uncertain 单独计数，κ 分母和排除数必须明确。
再按仲裁后的 t_lock 计算同一步/±1 步结构事件覆盖率，以及各事件类型的精确率。
按来源和 thought 是否可见分别报告。continued_build 仅作为扩展定义消融。
窗口未闭合实例须作为删失单列，不能只报闭合样本中位数而忽略其比例。

当前没有人类标签，因此不报告 κ 或 t_lock 覆盖率。

## LongRCA 编码域材料

`python -m segmenter.pilot_longrca --root longrca_bench --out pilot/m2_longrca`
从 full 的 SWE-bench Pro 与 Terminal-Bench 2 各选 20 条，按固定哈希排序、
以首条 user 任务全文去重；不混入 mini 的重复数据，也不使用根因标签选样。
full 本地包含 1,140 条，其中上述两来源分别有 128、42 条；全部为失败轨迹，
不能照搬 CodeTraceBench 的成功/失败配额。

LongRCA 的 idx 保留原生 history.step（包括用户消息，通常从 0 开始），
原角色 role、代理 name 与 content 全部保留，raw_message 可逐条无损回查。
标注 evidence.field 使用 content；不要把内容强行拆成 thought/action/observation。
两数据集步长定义不同，窗口长度须分开报告，不直接混合计算步数分布。
mistake_agent、mistake_step、mistake_reason 仅放在 reviewer_only，根因步不是 t_lock 金标。
该包没有运行 OpenHands 专用规则，candidates=null 表示尚不支持，不能解释为没有事件。
跨代理交接须按实际指令与后继使用证据判断，不能仅凭角色切换认定锁定。

CodeTraceBench 材料在 m2_codetrace/，LongRCA 在 m2_longrca/，合计 80 条；
两者 packet_id 分别用 p/l 前缀，A/B 各需标注 80 条。旧包的 status.json 保留
当时生成状态（LongRCA pending），当前总体状态以本节和新包 status.json 为准。
