# Claim 抽取 v2：兼容性验收与后续实验

## 当前范围

本轮只修复语义抽取协议，暂不改划分器、不构建语义依赖图、不实现报警或自动干预。
v1 的原始 48 条响应与失败结论保持不变。v2 的格式成功不能回填为 v1 成功。

程序将原始证据按最多 1200 个 Unicode 字符分块，优先在换行处切分，完整覆盖输入，不丢弃尾部。
模型只生成 text、source、objects、evidence_ids。程序生成 claim ID、关联 step，并按目录还原证据范围和原文。
引用片段可定位不等于该片段支持陈述；重复原文也只通过指定 ID 定位，不模糊匹配。

每次最多六条 claim、2048 输出 token。六条是兼容性阶段的输出预算，不保证信息覆盖；后续必须检查遗漏。
不要求模型判断 kind、依赖、原因或最终成功。保留限定条件和否定信息在陈述内。
source 区分 agent_assertion 和 tool_observation；工具打印的代码不能自动等同运行事实。

## 冻结输入

- `pilot/swe_claims_v2/compat_requests.jsonl`：6 请求，按任务 ID 排序取前两任务，每个任务的 gold/structure/rule 各取最早可用阶段。
- `pilot/swe_claims_v2/requests.jsonl`：原 v1 可用输入的全部 48 请求，用统一新协议转换；暂不启动全量。
- `pilot/swe_claims_v2/protocol.json`：固定选样、源输入指纹、作业与原预检失败记录。

这些是历史开发任务上的接口检查，不是独立测试集，不用于选择最优分段方式。
gold 分段仅为兼容性覆盖，模型请求不包含实验组标签和最终结果。
源阶段的 detected_at_seq 是既有回顾性实验口径，不证明真实在线封口延迟。

## 远程执行

将本轮代码和 pilot 输入通过 GitHub 同步到远程后，在仓库根目录运行（Python 标准库，无新增依赖）：

```bash
cd ~/online-failure-attribution
git pull --ff-only
python -m remote_pilots.openhands_smoke.claims_v2 run
```

默认访问 `http://127.0.0.1:30001/v1`，模型名 `qwen-local`。
输出目录：`out/swe_claims_v2_compat/`。
每作业只尝试一次，运行前预留文件；续跑跳过已有文件，包括失败和 reserved，不会自动重试收费调用。
改变协议、输入、模型地址或代码后必须用新的输出目录，不能混合结果。
输入指纹使用解析后规范化 JSON，避免 Windows/Linux 文件换行影响，但保留字符串内部的原文。

请传回整个输出目录。包含：

- 每个作业的原始模型响应、耗时、错误或规范化 claim；
- `config.json`：输入和代码指纹；
- `summary.json`：格式与证据引用检查统计；
- `semantic_review.json`：原始片段、抽取结果和待审核字段。续跑不覆盖审核文件。

本轮本地六项测试使用模拟 HTTP 响应，不是 Qwen 推理结果。

## 验收门槛

先检查 6/6 完整返回且 schema/证据引用有效。空 claims 可以格式合法，但必须人工或 AI 复核是否漏提。
任何失败均保留原因，不能只报告成功子集。截断输出不补括号、不接受半条 claim。
通过格式检查后再逐条评估：

1. 陈述是否被所引证据支持，而非仅主题相关；
2. 是否把尝试、计划或 agent 自述写成已经发生的工具事实；
3. 是否遗漏否定、条件、测试范围、代码版本或不确定性；
4. 是否出现重要事实遗漏、重复或六条上限造成的覆盖损失；
5. 是否引入证据之外的成功判断、依赖或因果解释。

记录 reviewer_type=AI 或 human，不能把 AI 复核宣称为人工金标。
兼容性与语义检查通过后才考虑全部 48 条运行；此后仍不直接推断报警效果。

## 下一阶段（尚未实施）

先冻结按任务隔离的报警开发/测试集合与前缀评估标准；当前兼容性样本不能作为新测试集。
固定相同报警模型、调用时刻及预算，比较：最近原始 steps、历史 claim+当前 steps、后者加证据回查。
先旁路记录继续/需验证/报警/证据不足、可疑 steps 和证据，不自动回滚、不训练干预策略。
最终失败标签不能直接赋给每个前缀；AI 生成的前缀参考标签必须独立标记为代理评价。

## 本地复现

```powershell
python -m unittest discover -s tests -p test_claims_v2.py
# 输入已生成。重新生成必须选择尚不存在的目录：
python -m remote_pilots.openhands_smoke.claims_v2 prepare --out out/swe_claims_v2_rebuild
```
