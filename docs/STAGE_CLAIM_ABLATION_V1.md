# Stage × Claim 四组消融：开发机制实验 v1

## 本轮回答什么

固定同一批原生轨迹、相同观察前缀、同一下游模型和8192 token总预算，比较：

| arm | stage | claim | 下游输入 |
|---|---|---|---|
| `raw` | 无 | 无 | 按时间排列的原始记录 |
| `stage_raw` | 有 | 无 | 按预测阶段组织的同源原文 |
| `fixed_claim` | 无 | 有 | 固定4步块生成的claim＋尚未发布的原文尾段 |
| `stage_claim` | 有 | 有 | 预测阶段生成的claim＋尚未发布的原文尾段 |

这不是新模型训练。抽取器用原7B adapter，下游判断器用**同一个NF4底座关闭adapter**，避免只有某一组更换判断模型。

本轮是四条已有原生任务上的**事件证据核查机制实验**，使用已有46项AI事实参考的类型标签，不是正式报警召回、误报、干预成功率，也不宣称新任务泛化。四条任务为 stable_unique、bounded_chunks、prefix_totals、config_overrides；部分任务后来已用于新的训练候选，明确按project-seen开发材料解释。原始adapter固定为最初 `claim_sft_7b_reviewed_v2/adapter`，不改用binding候选。

本地冻结及tokenizer预检：4会话、46查询、15个GRU阶段作业、12个固定块，合计27个生成部分、0输入缺口。每种时间口径184次下游判断；`both`为368次判断，另加首次27次claim抽取。已完成结果可断点复用。实际claim上下文在生成后再次按统一预算检查，所有被移除的完整单元均记录。

## 已固定的设计

- 有stage：默认冻结fold0即时GRU权重，阈值0.55，相邻门槛0.65。fold0预先指定，不按本批结果挑最好折。可另版本用phase基线。
- 无stage：固定4步计算块；只有原生finish才关闭最后不足4步的尾块。没有finish时保留开放尾段。
- 全部组只看当前prefix；每个参考步骤默认在其后4步核查，轨迹不足则使用已有最后步骤。问题只给步骤、字段，不给事实文本、分类答案或参考出处。
- 所有组使用同一批查询。原始证据不进入抽取后标签修补，标签从不进入生成提示。
- 保留最近2步原文。claim组额外保留尚未发布部分的原文；不使用独立失败信号增强某一组，也不在claim失败后偷偷补回封口原文。
- `raw`与`stage_raw`的原文无重复、无丢失，区别为阶段组织。后续模型预算不足时，按最旧完整输入单元移除；所有移除数量和ID均记录，失败不从分母删去。
- 同一底座和输出格式，确定性生成；按查询轮换四组顺序。实际提示、输出、token和耗时保存。
- 输入、参考、检查点、处理代码、底座配置和adapter指纹冻结；配置或输入改变时拒绝复用旧输出。底座权重不打包，不声称本入口重新计算了全部底座权重哈希。

## 两种时间口径

1. `representation`：同一观察前缀，使用已逻辑封口的claim，暂忽略计算等待。回答表示是否有用。
2. `queued`：每个策略、每个会话各设一个独立串行worker，按0.05秒/原生事件到达，用实测抽取服务时长模拟排队与发布。尚未发布内容以原文尾段呈现。回答这组受控队列假设下的表示可用性。

第二种是**队列仿真**，不是实际Agent持续运行延迟，也没有实测报警提前量。共享模型在两臂抽取之前统一加载，冷启动在initializations单独报告，不只计入首个组；队列使用暖模型的当次处理时长，复用缓存不把它变成零。恢复已完成抽取时仅重新加载模型，不增加一次隐藏生成。队列按各会话独立计算。

## 指标

主要统计事件类型正确率、按任务等权正确率，以及“类型正确且引用目标原始记录”的联合比例。类别包括零测试、源码观察、编辑失败、编辑回执、测试通过、一般probe、Agent声明、测试失败、运行异常；允许unknown，未知或非法输出仍进入分母。

联合指标仅检验原始记录归属，**不自动证明claim语义蕴含**。若需评价是否保留全部108式事实、错误成功断言或真正风险，还须另行语义复核和报警标签。46项事实不是46个独立任务，不对本批小样本作强显著性结论。

完整四组后输出四个差值：stage无claim、claim无stage、claim有stage、stage有claim，以及差中之差交互项。成本包括上游抽取和下游判断，不能只报判断器token而忽略claim生成。

## 本地准备（已提供冻结包后不用重做）

在仓库根目录：

```powershell
.\.venv-encoder\Scripts\python.exe -m segmenter.stage_claim_ablation prepare --out pilot/stage_claim_ablation_v1 --bundle out/stage_claim_ablation_v1_bundle.zip
python -m segmenter.stage_claim_ablation check --inputs pilot/stage_claim_ablation_v1
python -m unittest discover -s tests -p test_stage_claim_ablation.py
```

prepare拒绝覆盖现有目录。变更块大小、检查点、观察时刻或阶段策略必须另建版本。只读已有轨迹，不执行其中任何命令，不启动Agent，不调用生成模型。

当前本地Transformers 4.51.3对新版本保存的空 `extra_special_tokens` 列表有加载兼容问题；本地预算检查仅在加载adapter附带tokenizer时覆盖为空字典，不改文件、不改token词表或chat模板。远程仍必须使用实际底座tokenizer执行上述preflight，不能以本地结果跳过。

## 远程启动：GitHub拉取（本次推荐）

本次提交包含四组入口、测试、冻结输入、参考及小型GRU检查点；不含7B底座或LoRA权重。远程在已有仓库和训练环境中执行：

冻结目录单独固定JSON为LF行尾，保证Windows/Linux检出与归档后的字节哈希一致。

```bash
cd "$HOME/online-failure-attribution"
git pull --ff-only
conda activate agentforesight

QWEN7B_DIR=/home/sjtu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
TRAIN_RUN="$PWD/out/claim_sft_7b_reviewed_v2"

python -m segmenter.stage_claim_ablation check
python -m segmenter.stage_claim_ablation preflight --model-path "$QWEN7B_DIR" --training-run "$TRAIN_RUN" --out out/stage_claim_ablation_v1
python -u -m segmenter.stage_claim_ablation run --model-path "$QWEN7B_DIR" --training-run "$TRAIN_RUN" --out out/stage_claim_ablation_v1 --timing both
```

先确认预检 `input_gaps=0`。若git拉取有冲突，应保留本地修改并处理冲突，不使用reset清空。TRAIN_RUN必须是原版训练目录，含config.json和adapter/。下面的压缩包启动方式为备用。

## 远程启动：自包含包备用方式

将本地最终包 `out/stage_claim_ablation_v1_bundle_ready.zip` 上传到远程家目录。包内含必要Python源码、规则、冻结输入、参考和约0.8MB边界权重，不含7B权重或adapter。不需要git pull，不覆盖现有远程仓库；解压到新的实验目录。`bundle_manifest.json`保存每个文件哈希。

```bash
conda activate agentforesight
python -m zipfile -e "$HOME/stage_claim_ablation_v1_bundle_ready.zip" "$HOME/stage_claim_ablation_v1"
cd "$HOME/stage_claim_ablation_v1"

QWEN7B_DIR=/home/sjtu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
TRAIN_RUN="$HOME/online-failure-attribution/out/claim_sft_7b_reviewed_v2"

python -m segmenter.stage_claim_ablation check
python -m segmenter.stage_claim_ablation preflight --model-path "$QWEN7B_DIR" --training-run "$TRAIN_RUN" --out out/stage_claim_ablation_v1
```

先确认preflight的 `input_gaps=0`，再运行：

```bash
python -u -m segmenter.stage_claim_ablation run --model-path "$QWEN7B_DIR" --training-run "$TRAIN_RUN" --out out/stage_claim_ablation_v1 --timing both
```

若训练目录不在上述位置，只改TRAIN_RUN，指向原训练结果中同时含 `config.json` 和 `adapter/` 的目录。底座路径必须与原config一致；使用原版7B，不使用32B/AWQ。均为本地离线推理，不下载模型。确保GPU没有其他重负载任务，本命令不会停止任何进程。

可以先只跑表示层：`--timing representation`；随后相同输出目录改为 `--timing queued`补队列仿真。也可直接both。再次执行相同命令会复用完成的抽取与判断；配置改变必须换输出目录。异常中断时保留已落盘结果。

## 回传与审阅

回传整个 `out/stage_claim_ablation_v1/`：

- `preflight.json`：调用规模、token预算、分页与缺口。
- `claims/`：每个分块/阶段的抽取结果、原始输出、失败与服务时间。
- `answers/representation/`、`answers/queued/`：每组实际提示、输出、截断记录和判定。
- `summary-representation.json`、`summary-queued.json`：四组指标与差值，未完成时显式保留expected数量。
- `costs.json`：上游生成作业、状态和耗时。
- `initializations/`：每次启动的共享模型加载成本，避免归给最先运行的一组。
- `queue-simulation.json`：独立会话串行队列的开始与可用时刻。

```bash
tar -czf stage-claim-ablation-v1-results.tar.gz -C out stage_claim_ablation_v1
```

先看完整性和失败分母，再解释四组结果。下一轮需要加入风险／修复／依赖标签，以及未参与选型的新任务，才能回答正式预警问题。
