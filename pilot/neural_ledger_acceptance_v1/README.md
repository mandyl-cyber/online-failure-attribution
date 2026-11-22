# 7B神经摘要封口发布验收

固定三份可用真实原生事件存档，共138个事件。它们分别来自SWE、Terminal-Bench和先前的OpenHands真实运行。该选择用于工程验收，失败、成功复验、计划与结果冲突等语义场景仍需逐条复核。回放按原有顺序到达，未完成阶段继续保持开放。

本轮使用同一7B基座及已训练adapter。后台单worker复用模型，沿用训练的短claim提示和证据分块ID，再确定性转换成账本所需的原始记录ID、偏移与quote。观察摘要映射为conclusion类型，Agent声明映射为decision类型，两者仍保持原basis且语义待审。默认8192上下文及512输出token，超限和结构失败保留诊断，最多两次尝试。

本地规则backend预演已完成64个完成step、19个封口任务，全部发布成功，恢复后批次字节一致。神经生成、显存、语义质量及延迟尚待远程验收。

本次固定在线默认phase规则划分，先测神经抽取与账本发布接口。未复用最近结果偏负的自适应封口策略。输出不用于比较监督划分器的边界F1。

```bash
cd ~/online-failure-attribution
git pull --ff-only
conda activate agentforesight
QWEN7B_DIR=/home/sjtu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
python -m remote_pilots.openhands_smoke.neural_ledger_acceptance check-inputs
python -u -m remote_pilots.openhands_smoke.neural_ledger_acceptance run --model-path "$QWEN7B_DIR"
```

回传out/neural_ledger_acceptance_v1/整个目录。summary.json逐会话记录step、封口任务、发布批次、失败及尝试次数、恢复一致性、信号与待审claim、模型生成和显存指标。review_backend/extraction保存原始请求、输出和失败。claim_batches保存正式发布结果，timing保存队列和发布日志。

验收要求每个封口任务有可追溯成功批次或明确失败记录，已发布批次恢复不变，引用校验通过，开放末段无隐式封口。语义逐stage检查失败保留、错误完成断言和证据支持，另行填写复核表。

队列、处理和排队至发布耗时使用相同观察进程的单调时钟，回放间隔默认0.05秒。它反映控制回放的负载，和Agent真实工具耗时不同。该间隔是本轮工程选择，可通过--interval调整，新值应使用新输出目录。中途恢复为优雅等待worker结束后的重启，尚未模拟GPU生成中强制杀进程。模型冷启动包含在首次处理耗时内，同时单列model_load_seconds。

## GPU回传复核

三会话138个事件、64个完成step，共19个封口任务。SWE发布5/6，Terminal-Bench发布3/7，live_smoke发布6/6，合计14/19。五个失败中，两个stage输入为25855和9174 token，超过8192总预算，另外三个是观察型claim只引用了动作。失败均在日志中保留。两次相同请求的确定性重试重复了同样错误，需改成预算失败直接退出、引用错误使用明确修复策略或进入待审。

远程报告已发布批次恢复一致，本地核对了原生事件与固定输入一致、正式批次的输入和处理器指纹、精确引用、观察信号原文。对已发布24条claim做定向AI复核，23条获引用支持，一条输出中的“正确表达式”超出了仅打印结果所能直接证明的范围。另有一条commit-message目的描述的source标签需要保留Agent声明属性。该结果仅覆盖成功发布部分，不能当作完整任务的人工准确率或召回率。

单次成功生成的中位耗时1.61秒，最大8.34秒，进程累计GPU峰值约9.92GiB。原summary把失败处理完成事件也计入发布计时，并采用小样本向下取整的分位点，已在脚本修正。本地根据成功落盘批次去重、匹配时钟重算，SWE排队至发布中位11.86秒、最大19.20秒，Terminal-Bench为6.94/8.14秒，live_smoke为2.79/5.34秒。样本量分别只有5、3、6，p95/p99采用最近秩，基本对应最大值。回放每0.05秒送入一个事件形成的队列负载需要与真实agent节奏区分。

本轮证明7B已经走通原生事件、封口、生成、校验和持久化的部分任务，发布完整性仍受上下文和引用错误影响。下一步优先处理这五个失败及无效重试，并保证神经抽取失败时原始失败观察仍可供账本读取，再评估持续运行及报警。

复核脚本为 `python -m scripts.audit_neural_ledger_return`，结果写入out/neural_ledger_acceptance_return_v1/。原始远程回传保持原样，改正后的指标和AI记录单独落盘。

## 失败发布与无效重试修复

7B backend现在将预算、JSON和引用校验等确定性错误标为不可原样重试。StageProcessor会保存失败状态，重启后继续复用该状态，原始输出和错误诊断仍保留。临时运行异常继续使用已有尝试上限。该修改不自动更改claim的source或补猜证据引用。

最终抽取失败时，ReviewedClaimProcessor原子写入status=evidence_only的阶段记录，claims为空，review_status为needs_review，附抽取错误、尝试次数、诊断路径、该stage全部原始观察及已识别信号。信号带来源和封口后的可用时间。成功摘要仍使用status=ready。验收报告分别统计published_receipts、published_batches和evidence_only_batches，避免把仅证据发布计为摘要成功。

复用19个原始回传结果的回归中，得到19条阶段发布记录，其中14条摘要批次和5条仅证据批次。失败批次保留26条原始观察。确定性失败均只调用回归backend一次，恢复后没有重复调用，批次字节不变。这是保存结果的状态机回归，模型实际语义成功数仍为14/19，GPU性能尚未重测。

12项失败状态、发布恢复、既有抽取和计时测试通过。本地复算使用 `python -m scripts.replay_neural_failure_fixes --out out/neural_failure_fixes_v2`。新processor身份与旧会话不同，远程重新验收必须使用新目录。

```bash
git pull --ff-only
python -u -m remote_pilots.openhands_smoke.neural_ledger_acceptance run --model-path "$QWEN7B_DIR" --out out/neural_ledger_acceptance_v2
```

长stage仍可能无法在8192预算内生成摘要，目前的超限处理是明确失败并保留观察。要恢复该部分语义摘要，后续需要验证完整保留证据的分块抽取或更长上下文预算。

## v2远程回传验收

新一轮GPU运行确认19个封口任务均留下记录，其中14个有效摘要批次和5个仅证据批次。总尝试从上轮24次降到19次，所有失败stage均只尝试一次，状态为不可原样重试。五个仅证据批次完整保留26条观察，原文及哈希与stage输入一致。未发布任务为零。

本地独立核查了原生事件顺序、批次输入和处理器指纹、有效claim的精确引用、失败原始观察及信号偏移。在三会话副本上恢复、drain后，所有正式记录字节不变。恢复验证复用持久化结果，没有调用GPU，也未测试生成中强制崩溃。

14个有效摘要中的24条claim，其文本、basis及引用内容与v1一致，因此沿用上一轮针对相同内容的AI复核。有效摘要仍为14/19。剩余两个预算超限和三个引用类型错误需要继续改善，不能以19条发布记录计为19条有效摘要。

本轮成功生成中位1.62秒、最大8.34秒，GPU累计峰值9.92GiB。成功摘要排队至发布中位数为SWE11.86秒、Terminal-Bench4.73秒、live_smoke2.79秒，最大分别为19.22、5.94、5.39秒。相同控制回放中，Terminal-Bench中位数从6.94降到4.73秒。这是两次小样本运行的观测，尚不能推广为真实agent负载下的稳定收益。

验收结论为失败记录完整性、停止无效重试及缓存恢复已在这三份存档上通过。下一轮先处理长stage的保真分块和引用错误下有效claim的独立校验，分别统计语义摘要完整率、关键事实保留和排队时间。

复算命令为 `python -m scripts.audit_neural_ledger_v2_return`。报告在out/neural_ledger_v2_return_audit/summary.json，失败分项在failures.json。原始回传目录保持原样。

## v3整步分块与逐条claim校验

已实现以完整step为单位的token预算二分。每块保留动作、观察、记录ID、文本和原始封口时间。单步仍超限时记录oversized_steps，保留父stage原始观察。各块独立生成和校验，已通过的claim不因另一块或相邻claim失败而全部丢弃。无效claim及其原始标签、引用、错误写入rejected_claims。只对文本、basis、精确引用均相同的claim去重，丢弃的重复项保存在duplicate_claims。

父批次完整生成且所有claim校验通过时为ready。存在缺口但仍有有效claim时为partial，存在缺口且没有有效claim时为evidence_only。后两类保留完整原始观察，标needs_review。complete描述输入及结构校验完成度，语义质量仍需逐条复核。每批上限64条claim，超出项也保留明确拒绝原因。

CPU使用本地与远程同版本的tokenizer预检了19个stage，计划24个生成块。Terminal超限stage拆成6–9及10–13两个块，均符合8192预算。SWE超限stage可处理1–5、6、8、9、10五块，但第7步单独仍超限，缺口明确保留。旧的三个引用错误输出逐条重校验，可分别保留3、1、0条claim，同时隔离各1条错误项。该结果复用了旧模型输出，新分块摘要尚未GPU生成。

```bash
cd ~/online-failure-attribution
git pull --ff-only
conda activate agentforesight
QWEN7B_DIR=/home/sjtu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
python -u -m remote_pilots.openhands_smoke.neural_ledger_acceptance run --model-path "$QWEN7B_DIR" --out out/neural_ledger_acceptance_v3
```

回传out/neural_ledger_acceptance_v3/。分别验收完整批次、部分批次、仅证据批次、单步缺口、拒绝claim和分块尝试数，不能将部分完成计入完整摘要成功。summary增加partition_diagnostics和partial_batch_timings，原始模型输出按分块保存在attempt日志中。跨块事实关联和语义去重仍待评估。需要进一步处理单步大观察时，应使用保留原文偏移的分页方案。

## v3 回传验收（2026-11-22）

本地复核了 `neural_ledger_acceptance_v3/` 的实际 GPU 回传。19 个封口任务均有记录：14 个 `ready`、4 个 `partial`、1 个 `evidence_only`。相比 v2，四个原来只有证据的阶段现在保留下了部分 claim；**结构完整批次仍为 14/19，不能把 18/19 有 claim 称为完整摘要成功率。**

| 来源 | 完整批次 | 部分批次 | 仅证据 | 有效引用 claim 数 |
| --- | ---: | ---: | ---: | ---: |
| SWE | 5 | 1 | 0 | 20 |
| Terminal-Bench | 3 | 3 | 1 | 12 |
| live smoke | 6 | 0 | 0 | 8 |

核验通过：原生事件与固定输入一致，19 个任务各尝试一次；24 个生成块的实际请求与原始 step 范围、证据、封口截止一致，没有静默截断或修改引用。逐块重放转换后，接受与拒绝结果与正式批次一致。4 条无效 claim 被隔离，26 条父阶段原始观察完整保留。三会话副本重新打开并 drain 后，19 个发布文件字节不变。此次恢复复用缓存，没有测试 GPU 生成中强制崩溃。

单步超限尚未解决：SWE 首阶段的第 7 步仍超预算；其余五块生成了 8 条 claim，父批次明确标为 partial。Terminal 的长阶段已完整分块送入模型，但一条包含 `externally-managed-environment` 和 ensurepip 失败信息的 claim 因未知引用 ID 被拒绝。失败信息仍在原始观察和拒绝日志中，未被其余有效 claim 覆盖；该阶段也没有被现有观察信号规则捕获。**原始证据不丢失，不等于摘要保留了关键失败。**

40 条已接受 claim 中，24 条文本、basis 和引用与 v2 完全相同，沿用既有 AI 复核。对新增 16 条做引用范围内的 AI 复核，11 条获支持，5 条部分支持或措辞越界：主要是仅引用动作却表述成已完成、把生成器分析称为系数分析、仅凭 apt-get 路径推断其他包管理器不存在，以及下载中便声称安装完成。此结果不是人工准确率，也没有覆盖全部关键事实召回。原始输出未改写；这些 AI 判断只写在独立审计文件中。

分块增加了调用和排队成本。单块生成中位 2.05 秒、最大 8.37 秒，累计 GPU 分配峰值约 10.44 GiB。完整批次的“入队至发布”如下（部分批次另列在 JSON）：

| 来源 | n | 中位 / 秒 | 最大 / 秒 |
| --- | ---: | ---: | ---: |
| SWE | 5 | 12.85 | 30.57 |
| Terminal-Bench | 3 | 19.62 | 20.84 |
| live smoke | 6 | 2.79 | 5.37 |

SWE 首个部分批次处理约 29.35 秒。上述是每 0.05 秒送入事件、共享单模型串行 worker 的受控回放，不能代表真实 agent 节奏或划分器延迟。本实验仍使用默认 phase 规则，不是监督边界模型，也不提供新的边界 F1。

下一步顺序：先以保留记录 ID 和原文偏移的分页处理单步大观察；对被隔离的关键失败提供独立、可引用的结构事实，避免摘要只保留后续成功；再约束“动作请求／观察结果”和“进行中／完成”的措辞。验收仍分别记录完整、部分、仅证据和关键失败覆盖。当前不扩大训练或改线上默认划分器。

复算：`python -m scripts.audit_neural_ledger_v3_return`。结果为 `out/neural_ledger_v3_return_audit/summary.json`、`batches.json`、`new_claims.json` 和 `new_claim_ai_review.json`。脚本固定本轮新增 claim 的快照哈希，输出变动时要求重新复核；主 pipeline 已接入该审计入口。原始回传目录保持不变。

## v4：单步分页、失败信号与结果措辞（2026-11-22）

本轮保持 7B 权重、SFT adapter、8192 总预算和 512 输出预算，修复输入和校验链路，不重新训练。

1. **单步分页。**先按完整 step 分块，只有单步仍超限才按既有证据 chunk 二分分页。原始记录、哈希、chunk ID、原文偏移及封口截止不变；模型只能引用当前页可见 ID，不能引用同一步其他页。分页明确提示上下文不完整，不提供由整步推导、却不在当前页可见的 observed_facts。最小页仍超限时记录明确缺口，不截断证据。分页可能缺少配对动作，跨页语义整合仍待验证。
2. **关键失败独立保留。**新增 `externally_managed_environment` 和 `venv_creation_failed` 观察信号，随原子发布的批次保留原文、偏移、来源及可用时间。即使生成 claim 为空、部分失败或整批失败，信号仍可读取。它们不是模型生成的摘要，也不是 failure alarm；引用重叠只表示引用存在，不证明摘要已经表述了失败。
3. **完成状态约束。**补充提示，并对“仅动作引用却声称已完成查看”和“下载进度却声称安装完成”等明确模式逐条校验。7B backend 隔离不通过项，保留原文与错误，不替模型改写 source、引用或措辞。其他 backend 的同类问题标为 needs_review。该检查范围很窄，尚不能识别所有语义错误，如将生成器分析误称为系数分析。

本地真实 tokenizer 预检覆盖原来的 19 个阶段：26 个生成块，其中 2 页来自 SWE 第 7 步，输入为 4811 / 5394 token；每块输入加 512 输出预算均不超过 8192。全部原始证据 chunk 恰好覆盖一次，没有剩余缺口。用 v3 冻结输出回放校验，新增隔离 3 条 claim，不代表新提示下的模型表现。两类此前遗漏的环境失败分别命中 1 条。可同步的预检记录见 [v4_preflight.json](v4_preflight.json)。

33 项相关测试通过，包括不可见引用拒绝、分页精确偏移、原生发布和缓存恢复；三会话规则回放也通过 19 个封口任务与恢复检查。规则回放没有加载神经模型，不能用于声称 v4 摘要质量或速度已验收。v3 旧回传审计仍可复算。

远程继续使用现有 RTX 5090 和原来的训练 adapter，无需下载新权重。运行前确认当前目录是仓库；预检的 `--returns` 指向远程现有 v3 结果。使用全新 v4 输出目录，避免处理器身份变化与旧缓存冲突。

```bash
cd ~/online-failure-attribution
git pull --ff-only
conda activate agentforesight
QWEN7B_DIR=/home/sjtu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
python -m scripts.preflight_neural_claim_v4 --tokenizer-path "$QWEN7B_DIR" --returns out/neural_ledger_acceptance_v3 --out out/neural_claim_v4_preflight && \
python -u -m remote_pilots.openhands_smoke.neural_ledger_acceptance run --model-path "$QWEN7B_DIR" --out out/neural_ledger_acceptance_v4
```

回传 `out/neural_ledger_acceptance_v4/` 整个目录。验收分四层：

- 输入：原封口任务 19 个，逐页预算通过，原始 chunk 覆盖完整，无不可见引用通过校验；若最小页仍超限，必须明确报告缺口。
- 发布与恢复：19 个任务都有完整、部分或仅证据记录；已发布文件恢复不变。单独统计生成调用数、失败和隔离项，不把部分批次计为完整成功。
- 内容：复核新增和变动 claim 的引用支持；检查两类环境失败在独立信号中仍可读取，再单独判断摘要是否正确保留失败及后续恢复。信号存在不计为神经摘要命中。
- 成本：报告完整和部分批次的排队、处理与发布耗时，以及逐块生成时间和 GPU 峰值。v4 预计比 v3 多两次生成；在同样 0.05 秒回放负载下比较，不能用规则延迟替代模型耗时。

当前只完成本地工程验证，远程 v4 生成、AI 语义复核和成本测量待执行。在线 stage 划分器保持现状，本轮不提供新的边界 F1。

## v4 真实 GPU 回传验收（2026-11-22）

已复核 `neural_ledger_acceptance_v4/` 整个回传。**分页输入和关键失败独立发布通过本轮验收，claim 的语义稳定性仍未通过。** 此节替代上一节的“待远程执行”状态，不改变历史运行记录。

| 同批 19 个封口任务 | v3 | v4 |
| --- | ---: | ---: |
| 结构完整批次 ready | 14 | 16 |
| 部分批次 partial | 4 | 2 |
| 仅证据批次 evidence_only | 1 | 1 |
| 已接受 claim | 40 | 42 |
| 生成块 | 24 | 26 |
| 尚未送入模型的单步缺口 | SWE 第 7 步 | 无 |

16/19 只表示输入覆盖和结构校验完整，不能称为语义准确率。分来源为 SWE 6 ready、Terminal 4 ready＋2 partial＋1 evidence_only、live smoke 6 ready。SWE、Terminal、live 的 claim 数分别为 21、12、9。仍是默认 phase 规则切分，没有新的边界 F1 或 failure alarm 评估。

本地独立重建了 19 个模型请求，核对原生事件、封口任务与 v3 一致；逐块重建提示和可见 ID，重放 JSON 转换与校验，验证全量 chunk 恰好覆盖一次、偏移与引用原文一致，没有接受不可见引用。26 块均生成完成、未触达输出上限；SWE 第 7 步的两页实际输入为 4811 / 5394 token，与 CPU 预检一致。3 条引用错误仍被隔离，11 条部分／仅证据阶段的原始观察完整保留。三会话副本恢复并 drain 后，19 个发布文件字节不变；这仍是缓存恢复，不是生成中强制崩溃测试。

Terminal 第 2 阶段的 `externally_managed_environment` 和 `venv_creation_failed` 均发布了精确引用与封口后可用时间。两条信号都没有有效 claim 引用；原来包含这些失败的长 claim 仍因不存在的证据 ID 被拒绝。因此失败在账本可读信号中保住了，但**神经摘要仍遗漏这两条失败**，不能把信号命中算成摘要召回提升。另两个未解决的 Terminal 问题仍是只引用动作的观察型 claim。

对全部 42 条已发布 claim 做了非盲 AI 复核：34 条获引用支持、7 条部分支持、1 条不支持。支持范围允许借助配对动作解释 `has` 等短输出，但不允许借用未引用的其他阶段结果证明完成；agent_assertion 只核对其声明来源，不视为独立验证。这不是人工准确率，也没有穷举每个阶段应保留的事实；与 v3 抽审的样本和范围不同，不能据此比较语义准确率涨跌。仅 16 条文本、basis、引用与 v3 完全一致，已重新逐条记录本轮判断。

主要语义缺陷见 [v4_semantic_issues.json](v4_semantic_issues.json)：

- 单步第二页只有 rings.py 第 378–599 行源码，却生成“PolyElement.as_expr 方法 receipt 显示未改动原始源码”；没有前后对照，且 as_expr 定义不在该引用页内。
- 另一条把 polyutils.py 的源码查看称为 as_expr 方法 receipt，混淆对象与事件关系。
- 安装结果改写为 `reports ... installation success`，绕过仅识别 `installs/installed` 等模式的约束，但引用仍止于下载进度。上一轮的窄规则没有解决该语义问题。
- venv 包安装被概括为安装 Python 解释器；commit message 被表述成代码行为；计划更新 receipt 和 agent 回顾叙述被混作观察证据。
- `which apt-get && which apk && which yum` 会短路，不能由只打印 apt-get 推断 yum 不存在；仅引用编辑请求时也应明确是请求，而非已完成结果。

成本也有增加。v4 单块生成中位 2.29 秒、最大 8.42 秒，26 块合计约 86.56 秒，累计 GPU 分配峰值约 10.45 GiB。首个 SWE 阶段需 7 次生成，生成合计 31.20 秒、处理 36.23 秒。完整批次的入队至发布为：

| 来源 | v3 中位 / 最大（秒） | v4 中位 / 最大（秒） |
| --- | --- | --- |
| SWE | 12.85 / 30.57，n=5 | 15.35 / 37.52，n=6 |
| Terminal | 19.62 / 20.84，n=3 | 18.40 / 20.03，n=4 |
| live smoke | 2.79 / 5.37，n=6 | 2.93 / 6.32，n=6 |

两轮批次集合有变，不能把中位变化当作同任务性能收益。它们都是每 0.05 秒送入事件、共享串行模型的受控回放；不能代表 agent 真实节奏，也不是轻量划分器的延迟。部分批次计时另存 JSON。

下一步保留分页和独立失败信号，暂停扩大模型和追加完整回放。先把这 8 条问题及正确的失败保留样例冻结成开发检查集，明确“动作请求、源码观察、工具结果、agent 回顾”各自能支持的陈述。生成端优先尝试先选具体证据、再生成对应窄事实的输出约束，避免继续堆同义词正则；关键失败继续走独立带引用信号，不自动伪装成模型 claim。若后续做 SFT，将本轮问题作为训练／开发材料，另取未用于修改方案的新任务评估，不能把这三条熟悉轨迹改称独立验收集。

复算：先运行 `python -m scripts.audit_neural_ledger_v4_return`，再运行 `python -m scripts.review_neural_ledger_v4_return`。前者核验工程链路，后者保存固定快照的 AI 判断；输入快照变化会要求重新复核。结果在 `out/neural_ledger_v4_return_audit/`，主 pipeline 已接入两步。可同步汇总见 [v4_return_summary.json](v4_return_summary.json)。原始回传未改写，本轮没有训练或替换默认划分器。
