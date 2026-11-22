# Claim 后续数据与验收准备（2026-11-23）

本轮完成任务来源核定、证据案例库扩充和新会话采集准备，未启动 SFT／RL。**没有把重复整理的事实当新增训练监督，也没有把新任务定义当成已完成的独立评估。**

## 数据进展

- `case_bank.jsonl` 汇总 66 条证据案例：已有训练参考中的 56 条重点事实（14 个任务组，含 31 条失败、19 条有限正向结果、1 条警告、5 条执行路径范围），以及 v4 的 10 条开发纠错／遗漏案例。新增训练样本数仍为 0。
- `task_provenance.json` 沿旧 selection → 轨迹 ID → CodeTraceBench manifest → 规范任务名核定来源。旧 `group` 不能直接与任务名 SHA 比较；文件同时保存旧 group 和规范 group。100 个训练窗口属于 18 个任务，38 个开发窗口属于 5 个任务，规范任务没有交集。
- 三会话中，SWE 为 `sympy__sympy-24539`、Terminal 为 `csv-to-parquet`，均不在这 23 个训练／开发任务中；但两者已反复用于调试，仍只作开发检查。live smoke 是旧 merge_ranges 合成任务，不伪装成 benchmark task。
- 11 份已准备参考继续保留在 `pilot/claim_evidence_standard_v1/`，不混入旧训练集。旧训练参考也未改写。

既有来源筛查显示，大多数 SWE 任务已有项目使用记录，另一个未匹配任务的历史仍未知；“没有搜到”不能证明未使用。因此暂时没有可声称全新、已完成验收的 CodeTraceBench 测试组。

## 前瞻性采集

已在 `pilot/claim_prospective_tasks_v1/` 冻结 4 个新合成任务：保持首次出现顺序去重、保留末尾块的分块函数、带初始零的前缀和、保留 False/0/空字符串的配置覆盖。每个任务都在本地通过“缺陷实现失败、参考实现成功”的检查。参考解仅用于本地验证，不复制到 agent 工作区或任务提示；运行时只提供缺陷代码和可见测试。

这些是新合成工程任务，不是 CodeTraceBench，也不是 claim 金标。任务解决率和 claim 质量分别报告。先采集全部选中任务的原生事件，保留失败、循环及中断记录；再根据事件冻结 AI 证据参考；最后运行固定的 claim 模型对照，过程中不回头修改任务来提升成绩。4 个任务只能作小规模工程验收，不能代表 benchmark 泛化能力。

`run.py` 新增可选 `--task-bundle` 和 `--out-root`，默认 ranges 烟测保持原定义；诊断包现同时收集通用 Python 工作区文件与不可变检查文件。`claim_prospective.py` 仅采集日志，不调用 claim 模型，不训练。先检查 Docker、固定镜像和模型服务，再创建任务尝试记录。重跑时复用首轮记录，不覆盖失败尝试；若需基础设施修复后的重试，使用新输出目录并保留首轮，后续联合报告。

## 远程命令

仍用现有 RTX 5090 32 GiB、已有 OpenHands 镜像和本地 Qwen 7B，无需新权重。不要同时运行其他占显存的模型任务。

如果 `http://127.0.0.1:30001/v1/models` 尚无 `qwen-local`，在终端一启动已有工具服务（这里使用 32768 总上下文，避免原来的 16384 首轮超限问题）：

```bash
cd ~/online-failure-attribution
git pull --ff-only
conda activate agentforesight
python remote_pilots/qwen_tool_server/server.py \
  --model-path /home/sjtu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --model-name qwen-local --port 30001 --context-limit 32768
```

终端二先采第一项，确认基础设施可用后再继续四项；第二条 run 命令会复用第一项，不重新挑一个成功结果替代它：

```bash
cd ~/online-failure-attribution
conda activate agentforesight
python -m remote_pilots.openhands_smoke.claim_prospective check
python remote_pilots/qwen_tool_server/probe.py
python -u -m remote_pilots.openhands_smoke.claim_prospective run --limit 1
python -u -m remote_pilots.openhands_smoke.claim_prospective run --limit 4
```

回传 `out/claim_prospective_native_v1/` 整个目录，其中包含逐项 receipt、driver 日志、原生会话、代码检查与诊断包。不要只回传成功任务。单任务最多 20 个 agent iteration；耗时取决于模型实际工具调用与循环情况，不能按上轮 26 个摘要调用估算。

## 验收与训练顺序

1. 采集：4 个固定任务都有首轮记录，区分 agent 失败和基础设施失败；原生事件完整、检查文件未改动。
2. 冻结参考：尚不查看新 claim 输出，按证据标准记录真实发生的动作、失败、完成与恢复；允许 AI 复核，但必须注明来源、范围和引文。
3. 固定模型验收：报告引用支持、错误完成声明、关键失败覆盖、结构有效率、排队／生成成本。任务是否修好不是摘要的正确性标签。
4. 再安排 SFT：当前旧训练仍 100 条，不为凑数把开发纠错或 56 条既有事实重复塞入训练。新增材料须先完成规范任务映射、证据审查和实际 tokenizer 预检；训练、开发与前瞻验收按任务隔离。若之后使用本轮前瞻任务调方案，它们随即转为开发材料，下一轮必须再换验收任务。

本地复算：`python -m scripts.prepare_claim_next_stage`。它核对来源、生成案例库与任务定义，并在独立临时目录运行作者提供的测试；不加载 GPU 模型。当前相关 42 项回归测试通过，远程真实会话和语义验收仍待完成。
