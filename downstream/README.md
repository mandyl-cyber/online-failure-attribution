# CodeTracer stage 对照

准备脚本：`python -m segmenter.prepare_downstream --out eval/downstream_pair_v2`
输出已存在时不会覆盖。当前成功包是 `eval/downstream_pair_v2`；
`eval/downstream_pair` 是首次遇到官方空白动作错误时留下的未完成包，不能使用。

官方源码提交：2d302191dd07e7c0c2da6f7a5e9451c7cbb62d34。
源码克隆已取得，附带 LFS 数据下载被停止，未宣称整个上游数据检出完成。
官方分类器对纯空白动作发生 IndexError，本地 BlankSafeStore 仅保护空键；
原始 steps 文本不变，两个实验组使用相同保护和无学习缓存的官方树分类器。

## 已完成的离线验证

40 条来自既有 M2 分层样本（SWE/Terminal 各20，成功/失败各半）。
每条输出 gold/rule 两个输入目录，steps.json、task.md、tree.md 逐字节一致；
只有 stage_ranges.json 不同。rule 使用默认 phase 模式。
SWE 未被标注的末尾 finish 同时从两组排除。
官方 Normalizer 读取成功，stage 无重叠/遗漏覆盖全部评估步；
官方 ContextAssembler 实际注入 stage，忽略目录身份与 stage 文本后提示词完全一致。
输入不含 incorrect 标注、成功标签或原标注树。答案留在 evaluation_manifest.jsonl。

这是一项轨迹材料上的局部 stage 消融，不包含原任务沙箱，不代表完整论文复现。
40条包含成功轨迹，报告总体及失败子集时必须区分；宏F1遵循原脚本空真值记0的口径。
当前没有模型诊断输出，因此不能报告下游F1或门禁通过。

## 模型执行入口

使用 `downstream/run_one.py` 直接调用官方 TraceAgent，绕过 CLI 的全局记忆加载/写入；
memory=false、显式调用次数和费用估算上限，输出要求 complete 且输入未被改写。
未完成或结构不合法的标签重命名为 incomplete_labels.json，不能进入评分。
费用由上游静态价目表估算，不能当作供应商硬限额；未知模型定价需运行前确认。

代码提供 Dockerfile，但当前尚未构建或验证容器。2026-11-12 检查发现 Docker 客户端存在，Linux 引擎命名管道不存在，服务未就绪。正式运行需模型/API/预算和运行中的 Docker Linux 引擎。
一对 gold/rule 已通过本地 `--dry-run`，记录在 `eval/downstream_dry_run`；这仅验证提示词组装，没有调用模型或验证容器。
只挂载单条输入与该次空输出目录，不能把评估答案、整仓库或另一实验组挂载给 agent。
两组参数完全相同，每条启动新容器，交替 gold/rule 顺序，不能重用前次对话或记忆。

示意命令（Linux shell，路径和预算需替换）：

```bash
docker build -f downstream/Dockerfile -t m1-codetracer .
docker run --rm --cap-drop=ALL --security-opt=no-new-privileges \
  --memory=4g --cpus=2 --pids-limit=128 \
  -e CODETRACER_API_BASE -e CODETRACER_API_KEY \
  -v /absolute/input/one-trajectory:/input:ro \
  -v /absolute/outputs/gold/one-trajectory:/output \
  m1-codetracer --input /input --out /output --model MODEL --cost-limit USD --step-limit 60
```

先完成一对 smoke，再按已批准预算运行其余样本。80 次诊断总费用尚未发生。
本地 `--dry-run` 只校验提示，不构造 API 客户端。

## 比较

```bash
python -m segmenter.eval_segmenter downstream --manifest eval/downstream_pair_v2/evaluation_manifest.jsonl --gold-run outputs/gold --rule-run outputs/rule --out eval/downstream_pair_v2/comparison.json
```

缺少任意一组结果时不将轨迹当零分；报告 paired/expected 和缺失清单。
未全部配对时 gate_pass=null，防止只评估成功完成的子集而误称通过门禁。

## 步号排查

`python -m segmenter.audit_alignment` 输出 `eval/alignment_audit/report.json`。
14条均不能统一用折叠纯文本或去除 finish 修复：mteb-eval 有232次文字回复，
折叠后38步，manifest为30；部分样本仅run计数或去think后偶然等于金标长度。
这些仅是候选解释，不能据此证明逐步语义对应，因此继续排除。
需要原始逐步标注/官方对齐映射才能确认，不能按manifest长度截断。
