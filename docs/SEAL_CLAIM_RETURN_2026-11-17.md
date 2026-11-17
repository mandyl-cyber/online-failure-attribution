# 首批摘要对照回传核验

回传目录：`seal_claim_run_v3/`。已逐文件核对 inputs、requests、review、protocol 与当前冻结 v3 包完全一致；按请求身份读取全部 48 个结果，汇总与 summary.json 相符。

| 状态 | 请求数 | 含义 |
|---|---:|---|
| failed | 16 | 全部为 `URLError: <urlopen error [Errno 111] Connection refused>` |
| pending_reference | 28 | 14 个阶段尚未审核，不调用 |
| blocked_input | 4 | 2 个阶段两种提示均超限，不调用 |

零 raw_response，未获得模型输出，不能计算或比较语义正确性。失败发生在访问 `http://127.0.0.1:30001/v1/chat/completions` 的连接阶段；可确认该地址当时不可达，无法仅凭这些记录确定服务未启动、退出或端口配置的具体原因。

预检为 32768 上下文、2048 输出预算，允许最大 prompt 30720 tokens。两个超限阶段的 baseline/evidence_first 分别为 35876/35963 与 35606/35693 tokens。没有静默截断。22/24 为输入预算资格，不是质量通过率。

修复：运行前检查 `/v1/models` 和目标模型名；不可达时立即停止，不产生一批已失败任务。新增连接失败与模型身份测试，合计 6 项测试通过。保留原始失败记录，启动同配置 7B 服务后另开 retry1 输出目录。命令见 `remote_pilots/openhands_smoke/SEAL_CLAIM_DIAGNOSTIC.md`。未变更提示、样本、参考答案或模型质量结论。
