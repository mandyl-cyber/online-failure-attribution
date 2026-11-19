# 固定7B adapter的辅助事实输入对照

目的：仅检验辅助事实的表达是否影响claim生成。两边使用同一已训练adapter、原dev38条参考、原系统提示及原始动作/观察；不训练、不使用v4_scopefix修订标签、不改变在线默认实现。

输入已冻结于 `pilot/claim_sidecar_input_ablation_v1/`。新增模块 `segmenter/stage_claim_facts_v2.py` 是显式启用的实验版本；原v1、v4/v5构建和既有输入哈希不变。

## 远程运行

沿用agentforesight环境、RTX5090和原7B基础权重，至少预留20GiB空闲显存。使用已完成训练的 `out/claim_sft_7b_reviewed_v2/adapter/`；不使用32B AWQ，也不启动模型服务。预估38组、76次生成约5–15分钟，实际以日志为准。

同步本轮代码和输入后执行：

```bash
cd ~/online-failure-attribution
conda activate agentforesight
QWEN7B_DIR=/home/sjtu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28

python -m remote_pilots.openhands_smoke.claim_sidecar_ablation preflight \
  --model-path "$QWEN7B_DIR"

python -u -m remote_pilots.openhands_smoke.claim_sidecar_ablation run \
  --model-path "$QWEN7B_DIR"
```

默认输出 `out/claim_sidecar_input_ablation_v1/`。返回该目录即可，不必再传权重。相同配置重新运行会跳过已保存的条件；输入、adapter或软件版本变化会拒绝复用同一目录。生成失败会退出并保留已完成结果。

## 预算及比较口径

- 采用原训练的8192总长度、1024最大输出、固定seed和确定性解码，实际值从训练config读取。
- 先用真实tokenizer核算两种输入。某一条件超预算，则整对不生成，不截断；`preflight.json`保存全部38对及两边token数。
- 只在共同可生成的样本上比较；若不足38对，必须报告缺失和分组范围，不能直接对照此前全38组指标。
- 对每对交替两种条件的先后顺序，记录同步后的GPU耗时、token数和触及生成上限标记。不存在从历史结果直接复用original条件造成的环境差异。
- `EXIT:n`保存为输出中报告的内部标记，不宣称已独立确认标记对应哪个子进程；`exit_code`仍是外层命令退出码。
- 报错和非零内部标记优先于整体通过判断；如果也有通过文本，标为混合结果。读取源码里的异常文本不作为执行异常；预期捕获错误不自动判成失败。

## 验收与回传

工程检查：输入/adapter哈希通过、无静默截断、每个有效pair两条件均有输出、原参考不变。质量检查待回传后按既有标准逐claim审查：引用支持、关键事实遗漏、严重错误、父stage/任务组汇总及耗时。格式合法和辅助事实回归测试通过都不代表神经claim质量提升。

`summary.json`的`semantic_review=pending`是预期状态。当前只完成实验准备与本地规则回归，尚无本轮模型收益结论。
