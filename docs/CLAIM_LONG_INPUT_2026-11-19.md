# 长输入处理与教师重跑

148 个 stage 请求中 94 个超出当前服务预算。服务上限为输入加输出 8192 token，预留输出 2048 后，系统提示与 stage 数据约只能使用 6144。实际输入包含完整计划、源码片段、命令及观察，最大达到 53291 token。chunk_chars=1200 只是证据编号粒度，所有 chunk 仍一次发送，因此不会减少总 token。

已增加 --partition-long。超限 stage 按连续完整 step 二分，并用实际 tokenizer 逐块预检。子请求完整保留动作与对应观察，保存父 stage、原始证据、有效请求快照和超限记录。单 step 超限仍记录为未覆盖，需要后续证据分页。分块输出作为局部候选，复核时应在父 stage 层面检查跨块遗漏与重复，训练／开发沿用父任务组。分块增加调用数和总推理量。

本地 9 项回归测试通过。远程执行后以 partitions.json 统计完整覆盖的父 stage，不能用子请求成功数代替父 stage 覆盖率。

```bash
git pull --ff-only
conda run -n agentforesight python -m remote_pilots.openhands_smoke.claim_sft_teacher run --partition-long --out out/claim_sft_teacher_partition_v1
```

回传 out/claim_sft_teacher_partition_v1。使用新目录以保留前轮配置和原始结果。手写 Python 脚本退出码为零的局部执行成功表述已获窄范围支持，整个套件通过仍需对应测试观察。
