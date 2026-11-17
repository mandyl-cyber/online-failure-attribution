# 在线输入协议与暂停恢复：首版实现

## 目标和参考

先固定可靠输入，再评估分段、claim 和 alarm。输入处理不调用模型。

- OpenHands SDK 将 append-only 日志和上下文 View 分开：https://raw.githubusercontent.com/OpenHands/software-agent-sdk/main/openhands-sdk/openhands/sdk/context/condenser/README.md
- Anthropic 将 session 持久日志和模型上下文分开：https://www.anthropic.com/research/managed-agents
- LangGraph 明确恢复、重试不自动保证外部副作用恰好一次：https://docs.langchain.com/oss/python/langgraph/fault-tolerance

这些是架构参考，不是对 OpenHands 0.59 接口兼容性的保证。

## 实现与支持范围

`segmenter.durable_input.DurableInput` 接收有稳定整数/字符串 ID 的原生事件；工具结果必须有精确 cause，且一个动作对应一个最终结果。结果可以按不同于动作的顺序到达，但结果先于对应动作不支持。明确标记为 partial/stream/is_final=false 的片段拒绝；未标记片段无法可靠识别。

每个接受事件先写 input_journal/event-*.json，flush/fsync 后原子替换，再交下游。重启按本地接收序重放，重建 pending、步号和去重状态。相同 ID 相同内容忽略，不同内容报错。规则、模型文件哈希、输入路径、任务和模式固定，不匹配拒绝恢复。

agent_state_changed、null、recall 是非工具结果通知：保留原始事件，不拿来填补 pending。recall 内容目前不投影到 step；如未来需要工作区上下文，应另定义带来源的上下文接口。

`segmenter.run_online follow` 必须提供 --out。空闲和 Ctrl+C 只暂停，不 flush 未完成动作，不结束阶段。动作 finish 或显式 --finalize 才结束；显式结束时缺观察的 step 保留缺失标记。已结束目录不重开。失败事件保存在 input_journal/rejected.json，状态在 follow_status.json。

```powershell
python -m segmenter.run_online follow <events目录或JSONL> --out out/input_live_v1
# 重启使用相同命令、相同目录；旧版无 journal 的输出需换新目录。
# 确认上游不会继续追加后，才显式结束：
python -m segmenter.run_online follow <events目录或JSONL> --out out/input_live_v1 --finalize
```

--finalize 在读到空闲结束后生效，不会立即中断仍在追加的输入。上游 error/stopped 不自动等同正常 finish，保留暂停状态供检查。

## 验收

2026-11-18：恢复测试 5 项、适配完整性 6 项、输入门禁 4 项、在线账本 5 项通过。覆盖 pending 跨重启、重复/冲突、乱序结果、无 cause、孤立结果、多个结果、部分 JSONL、通知过滤、配置冲突和显式结束。

三份返回日志：128 个事件全部接受、55 个 step、无 pending，持久化重放的 Step 字段全部相同。详见 `eval/durable_input_v1/report.json`；复跑 `python -m scripts.audit_durable_input`。它按历史文件 ID 排序，不能证明真实接收时序或长期在线延迟，也不是分段准确率结果。

## 限制与后续

1. 首版是单写者；不要让两个进程共用输出目录，没有实现跨进程锁。
2. 恢复为 O(历史事件数) 重放；跟读源从头扫描并按 ID 去重，尚无持久化源游标/增量状态快照。
3. follow 的 steps/events 等是重建输出，会在恢复时重写；不能作为外部恰好一次消费队列。不会重执行 agent 工具。claim API 调用和账本写入的幂等性仍由 online_ledger 的任务日志负责。
4. 这次接入的是 run_online follow。远程 shadow 和 online_ledger 尚未统一改用 DurableInput，不能宣称所有入口都具备同一恢复协议。
5. 不支持多结果流、跨 session 混合、源文件轮转/截断。目录文件在单次跟读中按已读文件名跳过，上游须保证不可变；重启重读时会检查重复 ID 内容。
6. fsync 文件与原子替换主要防进程中断，不宣称已验证断电、磁盘故障或网络文件系统恢复。
7. 无缺陷标记不代表输出完整；上游未声明的截断、多模态语义仍可能缺失。状态错误不能替代工具失败证据。
8. 当前没有引入新压缩器、神经模型或训练，也没有调整划分阈值。下步先统一生产入口，再做真实在线中断恢复验收、测量持久化成本；之后冻结输入版本重评下游。
