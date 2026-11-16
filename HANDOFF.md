
# 交接文档：划分器（M1）

## SWE Qwen 目的候选标签 v2（2026-11-16）

完整记录见 `docs/SWE_PURPOSE_V2_PROTOCOL_2026-11-16.md`。v2 对同一批 SWE 训练 25 条/935 步、开发 10 条/350 步使用明确的因果前缀压缩和三字段短 JSON。修复 tokenizer `BatchEncoding` 计数后，32,768 token 预检训练 **935/935**、开发 **350/350** 可容纳；跨轨迹试跑在明确控制步规则后为训练 20/20、开发 10/10 结构有效。完整初标已完成并逐条复核身份、输入哈希与结构：**训练 935/935、开发 350/350 有效，0 缺失、0 无效**。来源为 Qwen 原始合法输出 1,104、明确控制步规则 174、`SEARCH` 搜索别名 3、Codex AI 对持续非法输出的裁定 4；四条裁定及原始三次失败记录均保留。结果与环境在 `pilot/swe_purpose_qwen_v2/`，盲审输入和待审表已生成，**1,285 条仍全部未做独立 AI 语义复核**。已观察到修改前复现被 Qwen 误标 `CHECK`，故这些只是机器候选，不能当目的准确率或训练金标。新小模型训练、边界 F1 对照和在线默认替换尚未执行；下一步是独立 AI 盲审、对照首轮并裁定训练侧可信标签，然后按任务组交叉拟合训练轻量模型并评估边界。原 v1 数据和用户本地 `annotator_A.json` 改动保留。

## SWE Qwen 预检与五步试跑（2026-11-16）

见 `docs/SWE_PURPOSE_PREFLIGHT_AND_TRIAL_2026-11-16.md`。两个新远程包的哈希核验通过；使用本机已有 Qwen2.5-7B-Instruct snapshot 和同目录 tokenizer、服务实际模型 ID `qwen-local`、32,768 token 上限预检：训练 560/935 可容纳（375 超限），开发 253/350（97 超限），合计 472/1285 超限。训练首五步均可容纳；本机临时 Qwen 服务试跑后仅 2/5 通过严格 JSON/引用校验，另 3 步分别有错误证据 ID、未闭合 JSON、非法组合标签。服务已停止，完整 935＋350 标注及新模型训练**均未开始**。按远程包停止条件，先冻结新版前缀压缩和短引文协议，重新预检并另目录小样本试跑，再决定是否全量。`pilot/swe_purpose_qwen_v1/` 保存预检和原始五步结果；`annotator_A.json` 的用户本地改动继续保留。

## 当前 challenge 与下一步执行清单（2026-11-16）

完整可执行路线见 `docs/PURPOSE_CHALLENGES_AND_NEXT_STEPS_2026-11-16.md`。眼下的主瓶颈是目的标签无人工准确率、60 条 AI 参考严重类别失衡（`DELIVER` 0）、小目的模型 Terminal-Bench 轨迹留出仅 8/30，以及目的类别到阶段边界的关系未证实。优先冻结独立任务组评价集并补 Terminal-Bench/稀有类别可信标签，再训练比较轻量模型，最后单独验证边界增益及在线端到端时延。现有 CPU 模型只旁路输出；在线默认划分器保持原状。

## CPU 小目的模型试验（2026-11-16；不替换在线默认）

按用户要求，用现有 60 条 AI 复核标签训练了可在 CPU 上运行的目的学生模型，报告见 `docs/PURPOSE_STUDENT_AI60_2026-11-16.md`，权重 `artifacts/purpose_student_ai60_v1/model.json.gz`。按轨迹分组 5 折，对 **AI 参考**的准确率 32/60（53.3%），六个已出现类别 macro F1 0.3379；多数类 28/60。SWE 24/30、Terminal-Bench 仅 8/30，分来源差异大；`DELIVER` 无样本，`ENABLE`/`OPERATE` 留出召回为 0。纯 Python 推理不需要 Qwen、PyTorch、scikit-learn 或 GPU；模型文件 347,607 字节（约 340 KiB），735 步实测 CPU 中位 1.32 ms/步、p95 1.74 ms/步。`run_online.py --purpose-model ... --task-file ...` 只输出 `purpose_events.jsonl`，不参与默认切分。**没有人工语义准确率，也没有小模型目的特征的边界增益结果；不得把 53.3% 写成正式准确率或据此默认部署。**

## 使用现有 AI 参考分析误切（2026-11-16）

见 `docs/PURPOSE_AI_BOUNDARY_DIAGNOSIS_2026-11-16.md`。结构 218 切点/72 命中，融合 266/76；融合相对结构新增 74 个切点，其中 60 个按即时 D0 ±1 一对一规则未命中，净多 48 个预测只多命中 4 个，是 F1 下降的直接计数原因。60 条 AI 复核样本中仅 3 条恰好落在融合新增未命中切点，三条 Qwen 目的都与 AI 参考不同；另有 6 条融合未命中切点的目的却与 AI 参考一致。结论：目的误判是**可能原因之一**，但现有 AI 样本既非人工金标也非误切定向抽样，不能估计其贡献或据此重训/部署。可复算入口 `python -m segmenter.analyze_purpose_ai_review --out eval/purpose_ai_boundary_diagnostic_new`；在线默认保持原状。

## 目的标签 AI 复核结果（2026-11-16）

`ai_review.json` 现为 `codex-ai-review-2026-11-15` 完成的 **60/60 AI 复核**，不属于人工评审。Qwen 与 AI 复核标签一致 14/60（SWE 3/30、Terminal-Bench 11/30）；最集中分歧是 AI 的 `ACQUIRE` 被 Qwen 判成 `CONSTRUCT`（14 条）。报告见 `docs/PURPOSE_AI_REVIEW_RESULT_2026-11-16.md`，逐例结果仅在被 Git 忽略的 `eval/purpose_human_audit_ai_review/report.json`。这不是目的类别人工准确率，不能据此替换在线默认。评分器现要求标明 `reviewer_kind`，AI 对照报告默认写入 `eval/`，避免向盲审包泄露 Qwen 预测。`annotator_A.json` 的 5 条用户工作区标注未纳入提交，`annotator_B.json` 为空；人工核验仍是待办。

## 当前结论：目的特征全 20 条重评（2026-11-15；优先于下方 10 条历史结果）

2026-11-16 标注接续：七类目的标签的**完整人工判定规范**已保存到 `pilot/purpose_human_audit_v1/LABELING_GUIDE.md`。60 条逐步 AI 初稿及依据见同目录 `ai_draft.json`；用户要亲自审核的表为 `ai_review.json`：逐条对照 `blind_packets.json`，核定 `final_label`、把 `reviewed` 设为 `true`、填顶部 `reviewer_id`，含糊项可写 `review_note`。所有条目复核完后运行 `python -m segmenter.purpose_human_audit score-assisted --out pilot/purpose_human_audit_v1`。此结果只能称 **AI 辅助单人复核一致率**，不能称双人盲审准确率。`annotator_A.json` 已有用户开始独立填写的条目，工作区原样保留；`annotator_B.json` 仍空白。已看过 AI 建议后填写的条目不能混称独立盲审。

详见 `docs/PURPOSE_BOUNDARY_RESULT_FULL_2026-11-15.md`。通过原始编码步骤文本逐项核对，新增 10 条 SWE 的**有界指纹映射**（非完整内容哈希）；原 10 条 Terminal-Bench 仍用完整哈希。20 条、735 步、113 个人标边界均入评分，Qwen 目的分数 531 步可用、204 步因 32K 上限缺失。相同任务组折/因果特征/阈值协议下，即时 D0 ±1 F1：结构 **0.4350**，结构＋目的 **0.4011**，差 **−3.40 点**，20 任务组配对预测 bootstrap 区间 **[−6.77,−0.01] 点**。SWE 和 Terminal-Bench 分来源均下降。下方仅 10 条 Terminal-Bench 的 +7.65 点已被本轮扩样并重训结果取代，不能作为部署依据。

已实现 `segmenter/purpose_human_audit.py` 并将 60 例双人盲审材料放在版本化的 `pilot/purpose_human_audit_v1/`；两名独立人工标注者及分歧裁决尚未完成，**目的类别人工准确率未知**。评分器只接受完整人工标签，不能用模型标签冒充金标。完整前缀 Qwen 推理累计 756.1 秒，不满足每步毫秒级路径；**在线默认划分器保持原状**。`eval/` 被 Git 忽略，本机完整模型响应、映射和逐折预测未随仓库交付；GitHub 上的报告是结果摘要。

### 当前必须做的事（按顺序）

1. **完成人工目的核验。** 两名标注者分别只看 `pilot/purpose_human_audit_v1/blind_packets.json` 和七类定义，独立填写 `annotator_A.json`、`annotator_B.json`；第三人仅裁决分歧 ID。`UNKNOWN` 可用，不能根据模型预测填写。标完在保存本机 `eval/purpose_probe_v1/` 的环境执行 `python -m segmenter.purpose_human_audit score --out pilot/purpose_human_audit_v1`，记录一致率、分类准确率及分来源区间。目前此项**待人工输入**，不是代码失败。
2. **复核 SWE 原始对应关系。** 现有 10 条 SWE 仅通过独立打包文本的逐步有界指纹核对。若取得未经改写的原始 action/observation，补全内容 SHA256 再复评；否则所有报告继续明确称 `packed_step_verified`，不写成严格哈希匹配。
3. **做独立任务组的边界增益验证。** 当前 20 条是已分析开发集，融合 F1 下降 3.40 点。固定特征、阈值选择和任务组拆分规则后，在此前未用于选择方案的任务组重评结构/目的/可用性对照，报告覆盖率、分来源和配对区间；不要用当前 10 条正向子集重新调出收益。
4. **解决在线时延与长上下文缺失。** 本次 204/735 步无目的分数，完整前缀推理累计 756.1 秒。候选方案先离线比较固定短窗口、小模型或蒸馏，并计入所有缺失步及真实每步耗时；不能把离线 D0 信息时限当墙钟零时延。
5. **仅在门槛通过后考虑默认替换。** 至少需要目的语义核验、独立任务组边界增益、完整缺失策略和在线延迟预算。现阶段保留 `segmenter/run_online.py` / `segmenter/online_ledger.py` 的原默认路径。

复核命令：`python -m unittest discover -s tests -p test_purpose_probe.py`；`python -m segmenter.evaluate_purpose_probe --probe eval/purpose_probe_v1 --data eval/boundary_supervised/data_fixed --packed eval/boundary_encoder/minilm_headtail --out eval/purpose_probe_evaluation_new`。后者依赖本机被忽略的 `eval/` 数据和模型响应。接手者可从 `docs/PURPOSE_BOUNDARY_RESULT_FULL_2026-11-15.md` 看完整口径，下面“最新更新”保留的是历史 10 条子集结果。

## 最新更新：目的特征边界试验（2026-11-15）

完整事前方案和执行结果见 `docs/PURPOSE_BOUNDARY_EXECUTION.md`、`docs/PURPOSE_BOUNDARY_RESULT_2026-11-15.md`。Qwen2.5-7B 对 20 条轨迹的 735 个前缀完成独立七选项目的分数探测：531 得分，204 超 32K 上下文明确缺失；不是原 JSON 引文协议的合格标注，也没有人工目的金标。

只有 10 条 Terminal-Bench（383 步、59 个边界）通过完整内容哈希和步号与监督数据匹配。按原任务组折做结构/目的/融合/缺失对照：即时 D0 ±1 F1 为结构 0.3761、仅目的 0.3682、融合 0.4526、结构＋仅缺失标记 0.4023、目的切换直接切分 0.3172。融合相对结构 +7.65 点；10 任务组配对预测计数 bootstrap 区间 [+3.62,+11.70] 点。属于已分析开发集上的小样本信号，折人数 6/3/1，区间不计模型重训波动；无 SWE 金标结果，不能与全 461 条模型直接比较。保持在线默认模型不变。

新入口：`segmenter/purpose_probe.py`（本地 Qwen、单实例、前缀和显存/时延记录、断点重放）、`segmenter/evaluate_purpose_probe.py`（严格映射、折内阈值、配对比较）。`remote_pilots/qwen_purpose_v1/remote.py` 已修复现代 Transformers 的 token 数误计；原自由 JSON 五步试跑仅 1/5 通过，见其 `LOCAL_RUN_2026-11-14.md`。下一步先扩大可验证样本及改善目的标签质量/长前缀策略，再在独立任务组验证增益；不把七选项分数当目的语义正确率。

## 最新接手入口（2026-11-13；优先于下方历史记录）

### Linux 本地 Qwen 接入与实测更新（2026-11-13；优先于下方旧状态）

- 已找到并实际运行 `Qwen/Qwen2.5-7B-Instruct`。snapshot：`/home/sjtu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28`，四个权重分片约 15 GB，配置/tokenizer 可离线读取。运行环境：`/home/sjtu/miniconda3/envs/agentforesight/bin/python`，torch 2.14.1+cu130、transformers 5.19.0。RTX 5090 32 GB 在沙箱外正常可用；沙箱内 CUDA 不可见，不要据此判断驱动损坏。此次没有下载新模型。
- 新增 `segmenter/qwen_backend.py`，通过 `StageProcessor(QwenBackend(path))` 或抽取 CLI 的 `--qwen-model` 使用。BF16 离线加载并复用；输入上限 24576 token、输出上限 4096 token，超限不截输入；生成时间预算 120 秒为协作式 max_time，并非硬进程超时（加载也不受此限）。在线 CLI 仍未增加 Qwen 选项，当前完成的是封口作业批量抽取。
- v1 要求模型生成字符坐标：SWE 6/6 失败（5 个引用不匹配、1 个 JSON 不完整）。首次并行加载 Terminal-Bench 导致 OOM，是运行安排问题；随后独占重跑仍 7/7 校验失败。记录分别在 `eval/qwen_claims_v1/` 和 `eval/qwen_claims_serial_v1/`。
- v2 使用逐字引文协议：模型只给 evidence_id/quote，程序在指定证据内唯一匹配，计算 Unicode start/end；不存在、重复匹配、空引文和外来 ID 均拒绝。保留原始模型输出、实际模型提示、每次 token/耗时/峰值显存；校验失败的解析结果也落盘。提示最多 6 条简短 claim（当前通用最终校验仍允许 64 条，尚未强制这个更小上限）。
- v2 用一个模型实例顺序处理 13 个阶段，每阶段只尝试一次：SWE 3/6、Terminal-Bench 3/7 通过，共 25 条 claim；通过率 46.2%，推理合计 173.88 秒，PyTorch 峰值 allocated 18.52 GiB（非整卡总占用，不含加载耗时）。7 个失败包括：无证据 1、非法类别/字段值 2、引文不存在或歧义 2、JSON 截断 1、未知证据或空引文 1。报告 `eval/qwen_claims_v2/report.json`，逐次记录在各来源的 `extraction/*/attempt_1.json`。
- 4 项原抽取测试通过；额外运行 Unicode 唯一匹配、重复/缺失/空引文拒绝检查。抽查通过项仍有重复 claim、把读文件归为 state_change 等语义问题。46.2% 仅为结构与引用校验通过率，不能报告语义准确率、召回率或归因效果；25 条与规则 18 条也不能直接解释为覆盖提升。
- 下一步：分析失败原始输出，改进类别约束、证据选择、去重和输出长度控制；补持久化单元测试、可复现的单实例批处理入口、错误反馈重试及更完整版本指纹。当前 backend identity 仅含路径/预算及手工版本号，没有权重内容和提示内容哈希。再次调整须用新输出目录，保留旧结果，不绕过证据校验。先稳定抽取和语义质量，再接在线 CLI、对象版本关系与归因；不要重复启动两份 BF16 模型。

复跑单来源示例（在可访问 GPU 的环境执行，两个来源串行）：

```bash
/home/sjtu/miniconda3/envs/agentforesight/bin/python -m segmenter.claim_extraction \
  --jobs eval/online_ledger_smoke_v1/SWE/stage_jobs --out eval/qwen_claims_new/SWE \
  --qwen-model /home/sjtu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
```

注意：CLI 默认最多尝试两次，与 v2 单次实验预算不同。下方“未找到权重”“只有规则 backend”和原下一步 1–3 为本次执行之前的历史状态。

### 用户已确定的目标与粒度

实现在线阶段划分 → stage 结束触发账本生成 → 每阶段提取零到多条 claim，claim 为节点 → 建立阶段内/跨阶段支持、依赖、反驳、替代或回退关系 → 准确做 failure attribution。stage 不是固定的一节点；阶段内保存证据和候选，正式 claim 在封口后入账。每步划分不调用生成模型；阶段抽取异步执行。用户暂不新增人工标注。

最新偏好：不用外部 API，优先复用以前配置的本地 Qwen 7B。只找到 `F:\Yang\AgentForesight\outputs\outputs\qwen25_7b_coding_paper_test.log`，其中记录 `Qwen/Qwen2.5-7B-Instruct` 加载与实验产物输出。尚未找到当前权重位置、确认运行环境或成功启动；不能把旧日志当成当前已部署证明。当前项目和默认 Hugging Face 缓存未找到该权重，PATH 中未发现 ollama/gh。未下载新模型。

### 实现与证据

| 模块 | 入口 | 当前状态 |
| --- | --- | --- |
| 原划分器 | `segmenter/run_online.py`、`segment.py` | 规则 node/phase；旧入口可选监督模型。新账本入口目前固定 phase 规则 |
| 本地证据与缓冲 | `segmenter/ledger.py` | public-plan-v2 独立修复、步骤幂等、证据哈希和前缀回查、单写者重放 |
| 在线封口入账 | `segmenter/online_ledger.py` | 原生事件配对、阶段缓冲、持久化作业、后台批量 claim；暂停不封口，finish 正常封口，abort 标不完整 |
| 抽取协议 | `segmenter/claim_extraction.py` | 请求白名单、JSON Schema、引用/区间/引文校验、缓存和持久化尝试预算；只有 LocalRuleBackend，无模型推理 |
| 关系及归因 | 尚无完整实现 | edges 为空；跨阶段语义去重、claim 修订、支持/依赖图与最终归因器待实现 |

已验证：40 条离线人工阶段轨迹共 1,600 步、530 条早期规则账本项、3,200 条证据（`eval/ledger_local_v1_final/`）；此结果不是在线验证。随后两条真实日志不使用人工阶段、按在线顺序回放，中途重启：92 个原生事件、44 个完成步骤、13 个封口阶段、18 条规则 claim（`eval/online_ledger_smoke_v1/`）。真实 JSONL 增量追加/半行测试使用测试事件，不是活跃 agent 长期部署。

抽取 dry-run：`eval/claim_dry_run_v1/` 内 SWE 6 个阶段、Terminal-Bench 7 个阶段，全 ready、0 failed、18 条 claim。校验通过只表示结构与证据引用有效，semantic_correctness=not_evaluated，不是语义准确率。4 项抽取测试、5 项在线测试、3 项 ledger 测试及相关计划/生命周期检查此前通过。

历史划分结果：严格对齐 461 条的结构 LR 边界 F1 约 0.56、即时 D0 约 0.51；不是逐步准确率。冻结 MiniLM/ModernBERT 未证明稳定的大幅提升，未替换默认模型。另有早先启动的受控微调，本轮没有检查进程或最终结果，接手后先看 `eval/boundary_encoder/continuation/status.json` 与日志，不要重复启动。

### 优先执行的下一步

1. 只读追踪 AgentForesight 的推理代码、配置及旧日志，定位 Qwen 权重和 Python 环境；确认显存、量化方式及其他进程占用。找不到再问用户具体路径，不直接重下大模型。
2. 为 `StageProcessor` 接入本地 Qwen backend；固定权重/tokenizer/提示版本。现有预算是字节与尝试次数，不是 token、显存、超时预算；本地模型需补上下文和生成长度控制、加载复用及推理超时策略。
3. 先对已有 13 个阶段运行语义抽取，对比规则结果；检查遗漏、臆造、把测试通过扩大成补丁正确、引用保真，同时记录显存、输入/输出 token、耗时和失败率。
4. 补对象/版本/环境证据后建立有依据的关系。缺少 claim/edge 金标时不能报告语义准确率；不能自动把同路径或相邻时间当因果边。
5. 再接根因诊断和配对消融：原始轨迹、人工阶段账本、在线阶段账本、移除关系边。CodeTraceBench 错误步 F1 与 LongRCA 根因命中分别评分。

### 命令与约束

```powershell
python -m segmenter.online_ledger <events.jsonl或events目录> --out out/live_claims --run-id session-001 --claim-extractor dry-run
python -m segmenter.claim_extraction --jobs eval/online_ledger_smoke_v1/SWE/stage_jobs --out eval/claim_dry_run_new/SWE
python tests/test_claim_extraction.py
python tests/test_online_ledger.py
python tests/test_ledger.py
```

原生事件要求稳定 ID、有序且输入只追加。重启从头扫描并按 ID 去重；无持久化文件偏移、日志轮转、乱序处理或多写者支持。原始 native_events/source.bin 仅内部恢复审计，可能含 thought/未来记录；模型只能读冻结的白名单请求。不得使用离线消息折叠或人工阶段伪装在线效果。

CodeTraceBench full 3,316，verified 1,000 是其子集；当前严格对齐集合 461。LongRCA full 1,140、mini 200 是包含关系，编码/终端仅 170；原生消息步号不同于动作—观察步。阶段仅区间，未取得五类阶段逐步金标；无 claim/依赖/t_lock 金标。公开计划字段曾丢失 1,378 步，恢复采用独立版本，旧缓存/模型未替换。

详细说明：`docs/ledger/CLAIM_EXTRACTION.md`、`ONLINE_STAGE_CLAIMS.md`、`LOCAL_PROTOTYPE.md`、`INTERFACES.md`；数据审计见根目录 `数据集任务标注审计与stage账本设计.md`。早期接口文件是一部分目标约定，不能把所有字段当成已经实现。

### GitHub 交付状态

用户已要求推送 GitHub，明确目标为 `https://github.com/mandyl-cyber/online-failure-attribution.git`。本地已初始化 main，并设置 origin；推送前远程无分支。已添加 `.gitignore` 排除数据、权重、环境、vendor、eval 和标注轨迹等本地产物。代码仓库不包含这些产物，复现实验需另行准备数据。推送状态应以 Git 分支及远程引用核验为准。

---

以下为历史交接与实施记录，涉及“下一步”“尚未接在线”等旧描述时以上述最新入口为准。

## 当前交接补充：账本接口与开发协议（2026-11-13）

最新补充：`segmenter/claim_extraction.py` 实现阶段输入白名单、版本化输出 Schema、证据 ID/哈希/字符区间/逐字引用校验、可替换 backend、跨重启尝试预算和缓存；在线 `--claim-extractor dry-run` 接入后台处理。13 个真实日志阶段作业 dry-run 生成 18 条规则 claim，结果在 `eval/claim_dry_run_v1/`，全部 ready。详见 `docs/ledger/CLAIM_EXTRACTION.md`。无 API，预算为字节/尝试数，不冒充 token/费用预算；语义蕴含、关系边和归因效果未验证。

最新进展：按用户“stage 封口触发录入、一个 stage 多条 claim”修正并接通在线入口 `segmenter.online_ledger`。原生事件持久化恢复，公开计划直接从当前事件提取，不读 sidecar；phase 在线划分，未封口时仅缓冲证据/候选，封口后后台原子提交多 claim。暂停不封口，finish 正常封口，显式 abort 标不完整。详见 `docs/ledger/ONLINE_STAGE_CLAIMS.md`。5 项在线测试通过；两条真实日志因果回放共 92 个事件、44 个在线完成步骤、13 个阶段、18 条 claim，中途重启和证据回查通过，结果在 `eval/online_ledger_smoke_v1/`。仅规则抽取；没有语义去重、跨阶段修订、关系边或 API；不宣称真实 agent 长期部署和归因准确率已验证。下文“未接在线”的记录为较早原型状态。

后续执行已新增 `segmenter/ledger.py`、`segmenter/replay_ledger.py`、`tests/test_ledger.py`。详见 `docs/ledger/LOCAL_PROTOTYPE.md`。本地原型支持独立 public-plan-v2 恢复、逐步证据/账本写入、单写者原子提交与重放、证据哈希与前缀回查、阶段归属修订；没有语义支持/依赖边、API 或真实在线事件接线。最终开发回放在 `eval/ledger_local_v1_final/`，此前同名前缀目录为调试产物。不得把记录与回查验证解释为归因准确率。默认转换器、训练输入和既有模型未替换。

已完成 `docs/ledger/INTERFACES.md`、`INPUT_AUDIT.md`、`IMPLEMENTATION.md` 和 `experiment_v1.json`。它们规定稳定证据 ID、增量账本、阶段归属修订、关系方向、原文回查、oracle/在线区分与端到端预算；业务模块仍待实现。

新增 `python -m segmenter.prepare_ledger_protocol`，已生成 `eval/ledger_protocol_v1/`：40 条 CodeTraceBench 开发样本、40 个独立任务组，SWE/Terminal-Bench × 成功/失败各 10 条，按长度分桶和固定哈希抽样。检查阶段区间覆盖、原文件存在，记录配置、数据、原文及计划恢复产物哈希。属于此前已分析数据，不能声称独立未见测试。样本清单有 solved 元数据，只供协调与评分，禁止直接喂模型。

本轮无新训练或 API 调用，未修改旧转换器和模型输入。下一步实现版本化计划字段恢复、证据存储及本地增量账本原型；API provider/model/prompt/费用上限/数据外发范围尚未配置。此前训练进程本轮未启动、停止或核查，状态以其自身日志为准。

## 1. 目标和范围

按《总体技术设计：锁定感知的长程编码 agent 在线干预系统》实现「第一步：划分器」。它不训练任何模型，每来一步只用规则和工具类型做事件检测、建状态树、切 stage、捞候选 claim，并用打分脚本检验设计里的几个假设。

代码在 `F:\Yang\online_failure_attributor\segmenter\`，说明在同级的 `README.md`。你原来的 `classifier/events.py` 没有动。

## 2. 做了什么

### 2.1 代码结构

| 文件 | 作用 |
| --- | --- |
| `adapters/openhands.py` | 转换器：OpenHands 事件流、chat messages、LLM completion 日志、CodeTracer `steps.json` 统一成"一步 = 一个动作 + 它的观察" |
| `rules.yaml` / `rules.py` | 全部规则和阈值（命令分类、事件、树、切分、候选 claim、触发） |
| `shell.py` | shell 命令分类：切子命令、去前缀、归类、读写路径 |
| `events.py` | 事件检测：STATE_CHANGE / EXPLORE / VERIFY_RUN / VERIFY_PASS / REFERENCE / HANDOFF / IRREVERSIBLE / PLAN_SWITCH（DRIFT 未实现） |
| `tree.py` | 状态树：节点封口规则、relied_on_by 边、reverts 边 |
| `segment.py` | 切分，`node`（节点封口）和 `phase`（活动相位）两种模式 |
| `claims.py` | 候选 claim：test_result / env_assertion / plan_step / completion |
| `run_online.py` | 在线主循环；`replay` 回放、`follow` 跟读事件流、`bench` 批量 |
| `bench.py` | CodeTraceBench 读取与解包；`inspect` 打印一条轨迹的全部字段 |
| `eval_segmenter.py` | 打分：边界 F1、错误步落点、窗口代理、事件召回、等距切分参照、下游 F1 差 |
| `tests/test_lifecycle.py` | 4 个测试，复现设计文档"一个时间步的生命周期" |

每条轨迹输出 `steps.jsonl`、`events.jsonl`、`tree.json`、`stage_ranges.json`、`claims.jsonl`，下游部件只通过这些文件耦合。

### 2.2 对数据集的发现

数据集卡片没写原始轨迹的结构，下载后核对出来的是：

| 来源 | 条数 | 形态 | 步号规则 |
| --- | --- | --- | --- |
| SWE 系（`swe_raw/openhands__*`） | 313 | 每次 LLM 调用一个 JSON（对话历史 + 回复），GPT-5 跑的，没有 thought 文本 | 一步 = 一次工具调用 + 结果；纯文字回合和末尾 finish 不占步号 |
| Terminal-Bench 系（只在 `bench_artifacts/full/*.tar.zst`） | 199 | 标准 OpenHands `events/` 目录，Claude / DeepSeek 跑的，有 thought | 一步 = 一个 agent 动作；agent 的纯文字消息不占步号 |
| manifest 里没有路径 | 8 | — | 跳过 |

踩过的坑和处理：

- 同一个 SWE 任务目录混着两次运行，按 manifest 的 `step_count` 选对应那次；日志是累积的，只读最大的文件。
- 压缩包解出来路径超过 Windows 260 字符，改为只解需要的文件并摊平；一条轨迹的上千个事件小文件合并成一个 `events.jsonl`。解包缓存在 `codetrace_bench/_extracted`，完成标记是 `.ok2`。
- 有些轨迹结尾 agent 反复发"任务完成"被打回，转换器在步数对不上时自动折叠纯文字消息再对齐。**这个折叠只用于离线对齐，在线使用时保持关闭。**

### 2.3 规则按真实数据修过的地方

内联脚本（`python - <<'PY'`）和直接跑脚本文件在改动之后算验证；带绝对路径的命令、`git -C`、`applypatch`、`npm -s test`、空命令和 `C-c` 能正确归类；失败的编辑不算改状态；`sed`/`grep` 的第一个参数不当作路径。

## 3. 现在的成果

### 3.1 全量结果（verified 的 OpenHands 子集）

498 条进入统计（跳过 14 条步号对不上、8 条无路径），共 20595 步，每步处理中位 0.26 ms、p99 5.5 ms。

| 数字 | 结果 | 门禁 | 结论 |
| --- | --- | --- | --- |
| 边界 F1 | 0.487（phase）、0.382（node）；等距切分参照 0.455 | ≥ 0.70 | 没过，规则只比等距切分高约 3 个点 |
| 错误步落点 | 63.5% 在改状态步，8.6% 在验证步（n = 594） | 预期 > 70% | 略低 |
| 窗口代理 | 中位 4 步，66% ≥ 3 步，21% 没关上（n = 209）；指向错误所在节点的口径：中位 6、79% ≥ 3 | 对照 FaaP：中位 1、44% ≥ 3 | 窗口存在且比 FaaP 长 |
| 事件召回（3 步内） | 0.41（n = 209） | ≥ 0.60 | 没过 |
| 下游影响（数字 4） | 未做 | 差 ≤ 5 点 | 需要另跑 CodeTracer |

### 3.2 两半数据差别很大

下面是我用"全量减去只含 SWE 系的那次运行"推算的，**脚本没有直接按来源输出，数字是近似的**：

| | SWE 系（无 thought） | Terminal-Bench 系（有 thought） |
| --- | --- | --- |
| 事件召回 | 约 0.23（n = 90） | 约 0.55（n = 约 119） |
| 边界 F1（phase） | 0.54 | 约 0.44，低于等距参照；预测切点是人标的两倍多 |

### 3.3 能得出的判断

1. **窗口存在而且长。** 这是对"可以在窗口内干预"最有利的证据，可以作为论文的第一张图（`eval/verified/window_proxy.png`）。
2. **事件召回主要受限于有没有 thought 文本。** 有 thought 的一半已接近门禁，所以设计的赌注没被否定，但依赖 agent 输出推理。
3. **边界 F1 门禁对纯规则大概率够不着。** 人标 stage 是语义子目标，中位只有 5 步，没有任何一种步型转换的切点率超过 0.45。
4. **"3 步内"的召回定义与窗口中位 4–6 步矛盾**，应报成随 k 变化的曲线。

## 4. 已知问题和局限

- 错误实例、"接受时刻"都是代理：用人标的 incorrect 步当错误，用规则事件当接受，没有人标的 t_lock，所以事件召回回答的是代理问题。
- 划分器的详细技术文档不在文件夹里，节点粒度、VERIFY_PASS 的补发时机、REFERENCE 的判定都是我按总体设计推的，选择列在 `README.md`，阈值都在 `rules.yaml`。
- 认不出的命令（EXEC）占错误步的约 15%（89 个）。
- 14 条轨迹步号仍对不上（不到 3%），原因没查。
- Terminal-Bench 那一半切得太碎，规则还没针对它调过。
- 打分只覆盖 OpenHands；AgentRx 没有转换器。
- 我这边没法在你机器上直接运行（磁盘不足），所有全量数字都是你跑完贴回来的。

## 5. 还要做什么

按优先级：

1. **打分脚本加"按来源拆分"和"召回随 k 的曲线"**（k = 3、6、10；SWE 与 Terminal-Bench 分开列）。现在的拆分是推算的，要换成直接输出。
2. **调 Terminal-Bench 的切分**，重点是 `segment.phase_min_run` 和 `tree.explore_gap`，用 `--set key=value` 临时覆盖，不用改文件。
3. **补认不出的命令。** 把 EXEC 类里频次最高的命令列出来，补进 `rules.yaml` 后复跑，目标是错误步落点过 70%。
4. **决定要不要加第五种接受事件"在其上继续构建"**（出错改动之后不回退、继续在同一文件上编辑或运行）。会提高召回，但比 FaaP 的 t_lock 宽，我建议加但单独成一类、单独报数。
5. **第四个数字：下游影响。** 用 `eval_segmenter run --export-stages` 导出规则的 `stage_ranges.json`，让 CodeTracer 分别喂人标和规则切分，再用 `eval_segmenter downstream` 比较。这比边界 F1 更能说明划分好不好。
6. **边界 F1 门禁重新定**：建议改成"相对等距参照的提升"，或降为参考指标。
7. **M2 标注 pilot**（设计里的 40 + 40 条、两人标）：划分器检出的结构事件可以直接当候选给标注者，同时测"人认为的接受时刻有多少落在规则事件上"。这是回答赌注是否成立的真正依据。在此之前不建议加小分类器。
8. **后续里程碑接口：** 触发位里"σ → consequential"要等账本写入器（M4）；DRIFT 和 embedding 留了开关（`events.drift.enabled`），第二轮再做；AgentRx 需要写转换器，或用 `--gold-jsonl` 接入。
9. 查那 14 条步号对不上的轨迹（运行时会打印 `[skip] ... 转出 X 步，manifest 说 Y 步`）。

## 6. 怎么运行

```powershell
pip install pyyaml zstandard matplotlib
python -m pytest -q tests

# 看一条轨迹的全部字段
python -m segmenter.bench inspect --manifest codetrace_bench/bench_manifest.verified.jsonl --artifacts codetrace_bench

# 回放单条轨迹并逐步打印事件
python -m segmenter.run_online replay <轨迹目录> --out out/<id> --print

# 全量打分（约几分钟，有进度行；第一次要解 199 个压缩包）
python -m segmenter.eval_segmenter run --manifest codetrace_bench/bench_manifest.verified.jsonl --artifacts codetrace_bench --out eval/verified
#   --recall-k 6        改召回窗口
#   --set tree.explore_gap=2 --set segment.min_stage_len=3     临时覆盖规则
#   --export-stages eval/verified/stage_ranges                 导出给 CodeTracer
#   --dump                                                     每条轨迹的 events/tree/claims 也落盘

# 接在正在运行的 OpenHands 上
python -m segmenter.run_online follow <session>/events --out out/live
```

重跑前如果解包出过问题，先清缓存：`cmd /c rmdir /s /q codetrace_bench\_extracted`。

## 7. 待你决定的事

- 事件召回的 k 是保持设计里的 3，还是按实测窗口长度调大并报曲线（我建议两个都报）。
- 是否加"在其上继续构建"这一类弱接受事件。
- 边界 F1 的门禁是否保留。

## 8. 接续实施状态（2026-11-12）

本节为当前工作区核验结果；前文保留原始交接记录。

- 已实现按来源拆分和 k=3/6/10 召回曲线。全量基线见 eval/verified_command_audit/summary.json：498 条、20595 步，phase F1 约 0.4866，错误步落在改状态步约 63.64%，k=3 代理事件召回约 0.4115。仍未达到原门禁。
- Terminal-Bench 阈值扫描在开发集提升、留出集下降，未修改默认阈值。执行上下文实验也未改善总体指标，保持关闭。
- M2 两批各 40 条材料已准备（pilot/m2_codetrace、pilot/m2_longrca），尚未人工标注；用户暂不标注。不能据此声称 t_lock 已验证。
- 下游对照已生成 40 对输入：eval/downstream_pair_v2。官方 CodeTracer normalizer 和提示词注入校验通过；每对 steps/task/tree 完全相同，仅 stage 不同。eval/downstream_pair 是首次失败留下的不完整包，不使用。
- 官方源码提交为 2d302191dd07e7c0c2da6f7a5e9451c7cbb62d34；附带 LFS 数据未完整下载。纯空白动作通过本地适配器保护，未改动原始动作。
- 一对 gold/rule 样本已完成本地 --dry-run，见 eval/downstream_dry_run；没有 API 调用，没有下游 F1。
- 下游执行入口 downstream/run_one.py、容器定义与说明见 downstream/README.md。Docker 客户端存在，但 Linux 引擎连接失败；镜像尚未构建或验证。实际诊断还需要模型、API 配置和费用预算。密钥只在本机环境变量配置。
- 14 条步号异常报告见 eval/alignment_audit/report.json。没有统一的折叠或删 finish 修复方案；部分计数相等不能证明语义对齐，继续排除，不强行截断。
- 比较脚本报告 paired/expected，结果不齐时 gate_pass=null。当前未执行 80 次模型诊断，不能声称下游门禁通过。
- 实验只检验轨迹材料上的 stage 对照，不包含原始任务可执行沙箱，不是完整论文复现。

下一步：启动 Docker Linux 引擎并完成容器 dry-run；配置模型/API/预算后先运行一对真实诊断，再执行其余配对样本并报告完整覆盖率与 F1 差。M2 人工标注、M4 账本及在线干预效果仍待完成。

## 9. 阶段监督路线调整计划（2026-11-12）

按用户要求重新检索数据集、论文和划分标准，已保存 [划分器调整计划.md](划分器调整计划.md)。此为研究与实施计划，尚未训练新模型。当前优先级更新为利用已有阶段边界建立监督划分基线及因果评测；不再等待 t_lock 标注或 CodeTracer API 才推进划分器。

新审计 `python -m segmenter.audit_stage_labels` 输出 `eval/stage_label_audit/report.json`：verified 的 1000 条均在 full 中且边界相同；两者不能直接作为独立训练/测试集。manifest 有起止位置而无阶段语义名称；适合直接生成边界二分类标签。full 有两条阶段覆盖异常，与既有 14 条原始轨迹对齐异常分开处理。

纠正：CodeTracer 附录 A 的阶段规范是工作阶段连续区间，并非任意语义子目标；状态节点、工作阶段和 t_lock 需要分别建模与评测。默认规则未修改，部署与下游效果仍待验证。

## 10. 监督划分已实施（2026-11-12，接续更新）

已完成任务隔离数据构建、候选切点规则修复、轻量模型训练、严格前缀评分、外部任务评价及本地流式检查。详细说明与命令见 [segmenter/BOUNDARY_SUPERVISED.md](segmenter/BOUNDARY_SUPERVISED.md)。第 9 节“尚未训练、规则未修改”是当时状态，以本节为准。

五折使用 461 条严格覆盖轨迹：最终 ±1 边界 F1 从修复规则 0.4780 提高到 0.5599；不晚于金标步确认的 D0 F1 从 0.3385 提高到 0.5057。相比此前 498 条，新增保守排除 37 条含未解释非结束后缀动作的 SWE 轨迹，不能跨样本直接比较。

全 verified 任务之外的 119 条 full 轨迹上，固定模型最终 F1=0.5811，修复规则=0.4914；D0 为 0.5203 对 0.3592。模型不读 thought、不调用生成 API，最终候选是四步结构特征的 Logistic Regression。尚未达到 0.70。

在线入口支持显式 --boundary-model，另写 supervised_stage_ranges.json，不替换原规则默认输出。10 条共 460 步的回放中，含落盘 feed p99≈5.94ms；半行追加文件模拟通过。未完成真实 agent 长时部署、恢复机制和付费下游诊断。

## 11. 小编码器专项调研（2026-11-12）

按用户要求完成 [小编码器在线划分器调研与实施计划.md](小编码器在线划分器调研与实施计划.md)。主方案为独立编码当前 step、缓存短历史向量、融合结构特征的小分类头；先 MiniLM 冻结探测，再按收益试 ModernBERT 局部微调。在线不调用生成式 LLM。

本机有 RTX 4060 Laptop 8GB，但当前 torch 2.14.1+cpu 无 CUDA，GPU训练需独立环境准备。本次只做调研、数据/硬件核查和计划，没有下载模型、安装依赖或训练编码器。旧 119 条外部任务已有基线成绩，后续不能反复据此调参并称盲测。

## 12. 小编码器实验执行中（2026-11-12，后续更新）

第 11 节是调研时的历史状态。现已建立 `.venv-encoder` CUDA 环境，下载锁定 MiniLM/ModernBERT，完成 461 条、18,636 步输入指纹复核，以及 MiniLM 六种分类头的完整五折实验。

MiniLM+结构 LR 最终 ±1 F1=0.5618、D0=0.5115；仅结构 MLP=0.5645/0.5149。按开发折选择头的最终 F1=0.5590、D0=0.5078，相对旧模型 D0 增益的任务组 bootstrap 95% 区间为 [-0.8054,+1.2546] 个百分点，未证明稳定收益，未替换原在线模型。

ModernBERT 冻结编码与后续实验仍在执行。串行续跑状态见 `eval/boundary_encoder/continuation/status.json`，完成后会依次输出冻结对照、独占单步耗时、冻结前缀缓存及最后一层受控五折微调；失败即停止并记录日志。不要将等待中的实验视为已完成，不要重复启动。GPU 持续低频（52 MHz、约 5 W），未修改用户电源设置。

详细结果、文件接口、运行命令和未完成项见 [segmenter/ENCODER_EXPERIMENT.md](segmenter/ENCODER_EXPERIMENT.md)。仍未启动新的人工标注、外部任务调参、生成式 API 调用或真实 agent 部署。

后续进展：ModernBERT 冻结五折已完成。融合 LR 最终 F1/D0=0.5633/0.5123，开发选择头为 0.5670/0.5162，但其中四折选择了纯结构 MLP。最终 F1 增益置信区间覆盖零、未达到提升 3 点目标，不能认为语义模型已胜出。MiniLM 独占单步 p99：CPU 64.26 ms、GPU 24.43 ms，不含分类头及落盘。后续耗时测量和最后一层受控微调继续串行运行，当前阶段以 `continuation/status.json` 为准。

## 13. 延迟确认、错误审计与输入缺陷（2026-11-12）

已执行即时/看后续一步/两步的固定五折，以及只等待不看新证据的对照。最终 F1 为 0.5576/0.5703/0.5608；共同 D1 时限下即时模型 0.5576、看一步 0.5128，且 Terminal-Bench 下降，未启用统一延迟。误报 2,096 个中 2,077 个落在工具类别切换处；这是相关性诊断，不能自动判断边界是否当时可知或标签错误。

审计定位 `_s()` 将 task_list 字典误按 text 内容片段处理，导致 1,378 个计划动作变空（270 条轨迹）。已按原始 call ID 全量恢复至 `eval/plan_payload_audit/`，未改写旧缓存或默认适配器。旧“指纹一致”只证明新旧转换输出一致，不证明转换无字段损失。一次恢复计划状态特征的即时 LR 探测得到 F1=0.5653、D0=0.5083，改善有限，未发布新模型。

ModernBERT 前缀缓存已完成，原输入协议的最后一层微调正在训练，尚无完整五折结果。新增测试通过，详细证据、限制和产物见 [segmenter/CONFIRMATION_AUDIT.md](segmenter/CONFIRMATION_AUDIT.md)。未启动人工标注、外部任务调参或付费 API。

## 14. 先研究任务与标注，再定义 stage 和账本（2026-11-12）

按用户新要求完成 [数据集任务标注审计与stage账本设计.md](数据集任务标注审计与stage账本设计.md)，机器可读全量统计在 `eval/dataset_task_audit/`。CodeTraceBench full 3,316 条（SWE 646、Terminal 2,670），verified 1,000 全部是其子集；当前训练 461 条中 SWE 276、Terminal 185，含 23 个原领域类别。LongRCA full 1,140、mini 200 为包含关系；编码/终端部分仅 SWE-bench Pro 128 + Terminal-Bench 2 42 = 170 条。

核实了公开逐步 labels 只有 incorrect/unuseful；stage 对象只有编号与起止。论文五类阶段规范不等于本地已获得五分类金标。3,316 个原 annotation_relpath 均不在当前根目录；按六个来源/agent 组合抽查 12 个原始包，未发现已知命名的逐步阶段标注文件。上游读取代码支持 step_N.jsonl 的 llm_analysis.stage_label，但这些文件尚未取得；不能据接口推定标签已发布或已人工核验。

新设计提案：任务画像（领域、工作目标、产物/验收约束）约束阶段判断；以 stage 组织 claim 账本、按事件持续更新；跨阶段依赖、反驳和状态版本显式记录。任务 category、工作阶段、状态树节点、claim 和 t_lock 分开。没有新增模型实验或改写现有金标；原受控微调继续后台执行。完整账本写入器与语义阶段监督仍未完成。
