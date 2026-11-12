# 总体技术设计：锁定感知的长程编码 agent 在线干预系统

Nov 11, 2026 · @Youval Benisti

## 定义与命题

本系统对正在运行的长程编码 agent 做在线失败干预。干预的目标不是错误发生的那一步 t\_err，而是错误**被系统接受、开始被当作前提使用**的那一刻 t\_lock。为了在线做到这一点，用一个零解码的划分器把轨迹实时组织成状态树和 claim 账本，让一个小归因器只读账本和当前段，在结构事件处判断刚被接受的结论是否可靠。

论文只证一句话：**对长程 agent，在线干预应该瞄准 t\_lock 而不是 t\_err**。三条证据线：标得稳（t\_lock 比单步根因更可标注，且能用结构事件逼近）、学得会（以窗口为目标训练的小模型优于以点为目标的同模型）、救得回（在窗口内干预提升成功率，在 t\_err 干预打断自愈，在 t\_lock 后干预无效）。划分器、账本、触发器都是为这句话服务的方法，不进贡献列表。

第一版只做编码 / 终端域，OpenHands 格式。

## 设计依据

| 事实 | 数据 | 导出的设计决定 |
| --- | --- | --- |
| 错误 ≠ 失败 | TrajDebug：失败轨迹均 7.6 个错误、仅 1 个关键、62% 自愈；71% 成功轨迹也有错误。TELBench：36.9% 成功轨迹含错误 span。MASC：弱检测器 + 每步纠正为负收益 | 不在 t\_err 报警；自愈区是训练里的显式负样本 |
| 长轨迹上单步根因标不稳，接受时刻标得稳 | LongRCA（中位 145 步）标注者精确步一致 40.3%，38.1% 相差 > 20 步；TRAVERSE / RootSE（30–50 步）κ 0.73–0.78；FaaP 时间点 κ ≥ 0.94；AgentRx “首个不可恢复步” κ 0.89 | 标注单位是窗口 (t\_err, t\_lock\] 而不是点；t\_lock 用结构事件定义而不是“还能不能救”的判断 |
| 接受有可观察的形状，且窗口存在 | FaaP：t\_lock 中位 = t\_err + 1，44% 的窗口 ≥ 3 步；锁定后 82% 继续运行、84% 的编造发生在锁定后；环境错误信号在成败轨迹间无差异（74% vs 72%），差异在锁定前是否响应（92% vs 37%） | 触发器是规则检出的结构事件（引用 / 交接 / 验证放行 / 不可逆）；干预点在窗口内 |
| 小模型读结构化输入够用，读全文不够 | Scout 4B 读固定格式轨迹超前沿模型；TRAIL 定位性能随长度负相关；TierMem 纯摘要丢 15–30% 关键信息；RCTA 摘要检索卡在 24% | 归因器读账本 + 当前段，不读前缀全文；账本是 claim 不是摘要；原文保留按指针回读 |
| 每步调用大模型不可行 | 前缀重读 O(T²)；Cadence 5.9 次/任务，OnTrack 每步 1 ms，StepFinder 零解码 | 每步只跑规则和一次 embedding；模型只在封口和触发时调 |

## 系统架构

&#91;embedded content: 系统架构 · 划分器每步、两个小模型按事件、前沿模型仅报警后\]

上行是每步都走的路径，全部零解码。中行两个小模型各有自己的触发条件：账本写入器跟着节点封口，归因器跟着结构事件。下行是干预路径：归因器报警则门暂停 agent 的下一个动作，前沿模型沿依赖边回读原文后决定放行、注入提醒还是阻断。所有部件只通过文件格式耦合（统一 step、events.jsonl、tree.json、ledger.json），可以分别开发和替换。

**同步与异步** —— 划分器和账本写入器始终异步，agent 不等它们。归因器在交接、验证放行、不可逆动作三种事件处同步（agent 在这些点本来就要停），其余触发异步，结论在下一步动作前到达。这是 FaaP “t\_lock 中位只晚一步”和“异步天然晚一步”之间的折中：窗口为 1 的情况靠同步点，窗口 ≥ 3 的 44% 靠异步。

## 模块定义

| 模块 | 输入 | 输出 | 实现 | 规模 | 调用时机 | 详细设计 |
| --- | --- | --- | --- | --- | --- | --- |
| 转换器 | OpenHands event stream / CodeTraceBench / SWE-rebench 存储格式 | 统一 step（idx, role, thought, action, observation, tool\_class, touched\_paths） | 规则 | — | 每步 | 划分器文档 §接口 |
| 事件检测 | 统一 step + 树快照 | 事件集合：STATE\_CHANGE / EXPLORE / VERIFY\_RUN / VERIFY\_PASS / REFERENCE / HANDOFF / IRREVERSIBLE / PLAN\_SWITCH / DRIFT | 正则 + 白名单 + 一次 embedding | 0.3B encoder | 每步 | 划分器文档 §事件规则 |
| 状态树 | 事件 | 节点（状态改变）、挂载的探索 / 验证步、relied\_on\_by 边、reverts 边、封口信号 | 规则 | — | 每步 | 划分器文档 §树 |
| 切分 | 事件 + 树 | stage\_ranges.json（懒封口） | 规则 | — | 每步 | 划分器文档 §切分 |
| 候选 claim | step | test\_result / env\_assertion / plan\_step / completion 四类候选 | 正则 + 简单句法 | — | 每步 | 划分器文档 §候选 claim |
| 账本写入器 | 刚封口节点的原文 + 其候选 claim + 现有账本 id 列表 | 账本增量：新 claim（文本、来源、τ、σ、引入 span）、被引用的旧 claim、σ 变化 | SFT，教师 = 前沿模型跑 DRIFT Claim Keeper prompt | 1.5–4B | 节点封口；HANDOFF / VERIFY\_PASS / IRREVERSIBLE 强制封口 | 待写 |
| 归因器 | 账本快照（固定列的表）+ 当前 stage 逐步原文 + 本步事件向量 | p\_t（锁定风险）、C\_t（涉及的 claim）、一句解释 | SFT + GRPO，窗口损失 | 4–8B | 事件向量命中：HANDOFF 或 VERIFY\_PASS 或 σ → consequential 或被依赖 ≥ k | 架构与训练文档 |
| L3 | C\_t + 沿 relied\_on\_by 边回读的原文 | 纠正建议或反事实重放结果 | prompt | 前沿 API | 仅报警后 | — |
| 动作前门 | 归因器结论 + L3 内容 + agent 拟执行的动作 | 放行 / 注入提醒 / 阻断 | OpenHands pre-action hook | — | 每个动作执行前；不可逆动作走独立白名单门 | — |

在线部分的显存：两个小模型 bf16 约 16 GB，一张 24 GB 卡。前期两个模型共用一个 4B 底座、两套 LoRA；消融时再试归因器换 8B。

## 一个时间步的生命周期

以 agent 第 t 步“跑测试，通过”为例。

1. 事件流送来第 t 步的 action 和 observation；转换器写统一 step，`tool_class = VERIFY`，`exit_code = 0`。
2. 事件检测：命中 VERIFY\_RUN，result = pass；候选 claim 抽出 `test_result: tests/test_x.py 3 passed`；REFERENCE 检到 thought 里提到了节点 n2 改过的 `parse_date`。
3. 树：VERIFY\_RUN 挂在当前节点 n4 下，`last_verify = pass`；加 n4 → n2 的 relied\_on\_by 边，n2 被依赖数升到 2。
4. 切分：VERIFY\_RUN 之后的第一步是候选边界，记下，暂不封口。
5. 事件向量写入 events.jsonl：`events = [VERIFY_RUN, REFERENCE]`，`refs = [n2]`。到这里全部零解码，几十毫秒。
6. 第 t+1 步到来，agent 打算改另一个文件。事件检测补发 VERIFY\_PASS（for\_idx = t），并命中 STATE\_CHANGE。
7. 树：n4 封口，`closed_node = n4`；开 n5。切分：在第 t+1 步前封口 stage。
8. 账本写入器被调用（异步）：读 n4 范围的原文和候选 claim，写入 claim `c7: “test_x 通过”, 来源 tool_output, σ = consequential`，并把 n2 对应的 claim c3 的 σ 从 tentative 升为 consequential（因为被验证放行且被引用）。
9. 触发判定：VERIFY\_PASS 是同步触发点，且 c3 的 σ 升到 consequential、n2 被依赖 ≥ k。归因器被调用，门暂停第 t+1 步的动作执行。
10. 归因器读账本快照（几十行）+ 当前 stage 原文 + 事件向量，输出 p = 0.71，C = {c3}，解释“c3 仅由一次局部测试支撑，测试未覆盖时区分支”。
11. p > θ，L3 被调用：沿 c3 的指针回读 n2 的 diff 和测试输出，给出提醒文本。门把提醒注入 agent 的下一条观察，放行。若 p ≤ θ，直接放行，不调 L3。

步骤 1–5 每步都发生；6–8 只在封口时；9–11 只在触发时。一条 64 步的 SWE-rebench 轨迹预期：账本写入器 10–20 次，归因器 5–10 次，L3 0–3 次。

## 训练目标与监督来源

**窗口标签** —— 每个错误实例 e 标 t\_err(e)、t\_lock(e)、t\_obs(e)、结局（repaired / persistent-harmless / decisive）。t\_lock 的操作化定义是“第一个把 e 的结论当前提的结构事件”：被引用、被交接、被验证放行、不可逆动作，取最早。每步的训练标签四态：窗口内（正）、窗口前含自愈区（负，权重大）、窗口后（负，权重小）、干净轨迹（负）。损失 = 带位置权重的交叉熵 + “窗口内至少报一次”的集合项。样本只在触发点采样，每条 5–10 个。

**三源监督**

| 源 | 数据 | t\_err | t\_lock | 规模 | 用在 |
| --- | --- | --- | --- | --- | --- |
| 人标 / LLM 标 + 人校 | CodeTraceBench 4,316、LongRCA 1,140 | incorrect 步 / 根因步（现有） | 用 FaaP 的 CODEBOOK 和 oracle prompt 让前沿模型标，人工抽验 400 条算 κ | \~5.5k 条 | 归因器 SFT 精调 + 评测 |
| 结构事件弱标签 | SWE-rebench 67k（35k 失败 / 32k 成功） | 不标，窗口左端 = 伪 t\_lock 往前 m 步 | 失败轨迹最后一次 VERIFY\_PASS / REFERENCE；成功轨迹全负 | 67k 条 | 归因器预训练；负样本主来源 |
| 反事实续跑 | SWE-rebench 可执行子集 300 条 | — | — | 300 条 × \~3 报警点 × 4 次 | GRPO 奖励 = 干预后成功率提升 ΔR；可选，不进第一篇的承诺 |

**账本写入器** —— 蒸馏。教师是前沿模型在 LongRCA 1,140 条和 SWE-rebench 2k 条上按 DRIFT Claim Keeper prompt 生成的账本，按节点切成增量样本；学生 1.5–4B SFT，输出 JSON。没有人标金标，验收靠下游：在 TELBench 上“喂账本”的 span F1 不低于“喂全文”的 95%；工作集大小曲线在 200 步内斜率趋近 0。

**归因器** —— 课程从短到长：AFTraj（11.5 轮）→ CodeTraceBench / SWE-rebench（20–64 步）→ LongRCA（145 步）。阶段 a：67k 弱标预训练；阶段 b：5.5k 有标精调；阶段 c：GRPO，奖励 = 窗口内 +1、自愈区 −1、窗口后 −0.3、成功轨迹 −1、C\_t 命中根因 claim +0.5。训练时划分器输出先用金标 stage（teacher forcing），再换自己的预测，两者差距就是划分器误差的传导量。

**指标** —— 主指标是窗口命中率对成功轨迹误报率的曲线；辅以自愈区误报、提前量、调用次数 / 输入 token；单步命中只作为与 AgentForesight / Scout / TrajAudit 对比的副指标。四条基线：同模型读前缀全文、同模型读 RCTA 式摘要、前沿模型读全文、纯规则（事件即报）。

## 与现有工作的位置

| 工作 | 他们做了什么 | 我们借什么 | 区别 |
| --- | --- | --- | --- |
| CodeTracer（NJU-LINK） | 离线状态树索引 + prompt 诊断，stage 来自人标，节点带 LLM 摘要 | 节点二分（探索 / 改状态）、stage\_ranges 格式、作为离线基线 | 树在线规则生成、stage 自切、节点内容是 claim、多 relied\_on\_by 边；划分器本身被评测 |
| DRIFT / TELBench（NJU-LINK） | 离线 claim 账本，四次全文 prompt，无训练 | ledger 字段、“探索 vs 承诺”的判定语言、Claim Keeper prompt 作蒸馏教师 | 账本在线增量维护、记“被依赖”而非“被检索”、b\_k 作触发点、小模型训练 |
| AgentForesight / Scout / AgentDebug | 前缀归因器，点标签，目标 t\_err | 小模型读结构化轨迹够用这个结论；两阶段 RL 配方 | 目标 t\_lock、窗口标签、输入是账本不是前缀 |
| Cadence / OnTrack | 决定什么时候看；非 LLM 每步信号 | 触发而非每步调用的思路 | 触发条件是结构事件而非健康度；回答“看什么”并给 claim 级解释 |
| FaaP | t\_err / t\_lock / t\_obs 标注与分布，无原文 | 标注协议、窗口先验 | 他们是分析，我们把 t\_lock 变成训练目标和在线触发 |
| TierMem / MEMIR / Graphiti | agent 自用记忆：分层、带出处、双时态 | 原文页 + 指针的分层；claim 作原子；失效不删 | 目标函数反过来：保留冲突、追踪依赖、不去重多源 |

一句话的定位：现有工作证明了结构化中间表示（状态树、claim 账本）让事后定位变得可行；我们问的是同样的结构在线增量维护、以接受时刻为目标，能不能让在线干预变得可行且可学。

## 第一步：划分器

不训任何模型，在 CodeTraceBench 的 OpenHands 子集上实现纯规则的在线树构建器并打分。一到两周。详细设计在划分器技术文档里，这里只定范围和验收。

**实现范围** —— `adapters/openhands.py`、`rules.yaml`、`events.py`、`tree.py`、`segment.py`、`claims.py`、`run_online.py`（含回放模式）、`eval_segmenter.py`。DRIFT 信号和 embedding 放在第二轮，第一轮只用正则和工具类型。

**要拿到的四个数字**

| 数字 | 定义 | 门禁 | 它回答什么 |
| --- | --- | --- | --- |
| 边界 F1 | 规则切点 vs 人标 stage 切点，±1 步 | ≥ 0.70 | 规则划分能不能用 |
| 错误步落点 | 人标 incorrect 步落在改状态节点的比例 | 预期 > 70%，不设硬门禁 | 树的节点定义对不对 |
| 窗口代理分布 | incorrect 步到之后第一个 VERIFY\_PASS / REFERENCE 的步数 | 报分布，与 FaaP（中位 1，44% ≥ 3）对照 | 编码域窗口存不存在、多长 |
| 下游影响 | CodeTracer 诊断 agent 喂人标 stage\_ranges vs 喂规则的，步级 F1 差 | ≤ 5 点 | 规则划分对归因有没有伤害 |

再加一个不进表但必须算的：**事件召回**——人标关键步后 3 步内是否有任一结构事件，CodeTraceBench 和 AgentRx 两集都 ≥ 0.60。这是整个设计最大的赌注（“接受的时刻伴随着可规则检出的动作”）的第一个答案。不过则划分器加小分类器，是改设计不是推翻。

**逐日** —— 第 1 天下数据、打印一条轨迹的全部字段；第 2–3 天转换器和事件检测；第 3–4 天树和切分；第 5 天前三个数字；第 6–7 天 CodeTracer 对照。先在 verified 上调规则，在 full 上报数。

**产出** —— 一个能接在 OpenHands 事件流上的划分器、论文的第一张表和第一张图（窗口代理分布）、下次组会的全部内容。

## 后续里程碑与关联文档

| 里程碑 | 内容 | 门禁 | 周 |
| --- | --- | --- | --- |
| M1 划分器 | 本文档第一步 | 边界 F1 ≥ 0.7，事件召回 ≥ 0.6 | 1–2 |
| M2 标注 pilot | CodeTraceBench 40 + LongRCA 40 条 t\_lock，两人标 | κ ≥ 0.8 | 2–3 |
| M3 数据统一 + 弱标签 | 七个来源转统一 schema；SWE-rebench 伪窗口 | 伪 t\_lock 在 AgentRx 上吸合 ≥ 0.6 | 3–5 |
| M4 账本写入器 | 蒸馏、SFT | TELBench 上喂账本 ≥ 喂全文的 95% | 5–6 |
| M5 归因器 | SFT 最小版 + 四基线，再加弱监督和账本输入 | 窗口命中比点损失 +10，比纯规则高 | 5–8 |
| M6 反事实续跑 | 100 条 × 3 位置 × 4 次 | 窗口内 ΔR 显著 > 0 | 8–10 |
| M7 写作 | — | — | 10–12 |

关联文档：划分器的技术设计（接口、规则、树、切分、评测、文件清单）；架构与训练方案（归因器的损失、阶段、算力）；两步训练可用数据与代码仓库；长程数据集一览（链接与标注方式）；行动指南（12 周路线与止损）。账本写入器和归因器的单独技术文档在 M1 完成后写，因为它们的输入格式要等划分器定稿。
