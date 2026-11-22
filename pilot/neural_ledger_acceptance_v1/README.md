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
