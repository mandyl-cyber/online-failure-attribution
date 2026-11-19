# 7B 训练预检的 tokenizer 返回类型兼容

远程原版 Qwen2.5-7B 四个权重分片存在，初次预检在回答 mask 构造时报告 assistant prompt prefix 不匹配。此前代码直接比较 apply_chat_template 返回值，隐含其为扁平整数列表的假设。

修复采用 return_dict=False，并复用已有 token_ids 归一化逻辑，处理列表、单条嵌套列表、带 input_ids 的映射及 tensor。训练输入和开发生成入口均使用同一归一化函数。归一化后仍严格检查 prompt 是否为完整对话 token 序列的前缀，避免把提示词误计入训练 loss。

本地15项回归通过，覆盖映射返回、嵌套序列、回答 mask 和真实不匹配拒绝。本地 Transformers 为4.51.3，远程此前报告版本为5.19.0。尚未用远程实际 tokenizer 确认故障已消除。若重跑仍失败，tokenizer_error.json 会记录样本、库版本、tokenizer 类及首个 token 差异位置。

重跑同一 preflight 输出目录即可，配置未改变。训练尚未启动，已有结果只涉及文件存在性检查。
