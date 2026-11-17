# SWE 小型时序划分器：远程一键训练

本包包含全部训练特征和标签（约 1.2 MB，276 条/9600 步），Git pull 后直接训练，不下载原始轨迹、Qwen 权重，不调用 API。

模型 190,081 参数；纯 CPU。默认两折并行、每折两个线程，适合现有 24 核/125 GB 主机。预计预留 15–35 分钟完成五折（基于旧本地单折短跑约 5 分钟及两路并行估算，未经远程实测）。GPU 不参与，训练期间可保留原 Qwen 服务，但共享 CPU 负载可能影响计时。

优化后本机两轮已测为 11.52 / 11.12 秒（训练 5731 步＋开发 1881 步），比先前短跑更快；仍保留上述保守时间预算，最终以远程日志为准。检查已包含 8 项模型/恢复测试和 1 项真实子进程五折启动/续跑测试。

## 执行

在已有 PyTorch、numpy、scipy、scikit-learn、joblib、PyYAML 的训练环境中：

```bash
cd ~/online-failure-attribution
git pull --ff-only
python remote_pilots/temporal_boundary_v1/launch.py --out out/temporal_boundary_train_v1
```

启动器验证数据哈希和环境，打印每折日志位置。若提示缺依赖，再在独立训练环境安装：

```bash
python -m pip install -r remote_pilots/temporal_boundary_v1/requirements.txt
```

不要求强制升级已经可用的环境。当前包只训练新模型，不加载之前由特定 sklearn 版本训练的模型；实际依赖版本写入 launch_protocol.json。

需要断开 SSH 时可用：

```bash
mkdir -p out
nohup python -u remote_pilots/temporal_boundary_v1/launch.py \
  --out out/temporal_boundary_train_v1 > out/temporal_boundary_train_v1.launch.log 2>&1 &
```

查看进度：

```bash
cat out/temporal_boundary_train_v1/progress.json
tail -n 5 out/temporal_boundary_train_v1/fold0.log
```

资源覆盖选项：`--workers 1 --threads 2`（串行，减少 CPU 竞争）；默认 `--workers 2 --threads 2`。只有改变并行折数 workers 可以继续同一目录；epochs/threads/数据/代码/依赖改变需要新目录，避免混合不同实验。

## 恢复

Ctrl+C 后等待启动器退出，再重复原命令。每个 epoch 原子保存权重、优化器、最佳权重、早停状态和随机状态。最多重做当前尚未完成的 epoch；完成折只校验并跳过。不要同时启动两个指向同一目录的训练，启动器有单写者锁。若整个主进程被强制杀死，先确认子训练进程也已停止，再恢复。

日志追加写入，不覆盖历史。最后生成 summary.json 才表示五折全部完成；单个 checkpoint 存在不代表训练完成。

## 训练内容与结果口径

- 原任务/内容分组五折不变：一折测试、下一折开发、其余训练。
- 网络固定；最多 15 轮，开发损失 3 轮不改善早停。
- 当前模型预测维护阶段状态，无 gold 重置，无额外老师/RL。
- 训练只计算一次因果编码，分离无梯度状态模拟与监督更新，减少重复 GRU 前向；有损失与梯度等价性测试。
- 三项对照：同组四步 LR、新模型即时提交、同权重置信度确认。只有少量开发阈值选择，没有额外架构搜索。
- 数据是历史研究过的 SWE 开发语料，不能称新盲测。分数阈值不代表已经校准的概率。
- 数据包仅删去了结构 LR 根本不使用的 baseline_rows.text，原数值/类别特征和神经输入不变，保留输入来源指纹。无训练集之外的原始轨迹依赖。

## 返回文件

训练完成后优先通过 Git 同步以下小文件，或把目录传回：

- `out/temporal_boundary_train_v1/summary.json`：五折汇总、各折对照。
- `out/temporal_boundary_train_v1/predictions.json`：逐轨迹预测。
- `launch_protocol.json`、`progress.json`、`fold0.log` 至 `fold4.log`。
- 各 `run_foldN/foldN/report.json` 与 `run_foldN/protocol.json`。

如需要在线模型，再同步各折 `immediate/`、`confidence/` 内的 model.json 和 weights.pt（必须成对）。resume.pt 仅用于本机续训，不必上传。没有默认自动 commit/push，也不会自动替换在线默认划分器。

完整设计及本地原始计时见 `docs/TEMPORAL_BOUNDARY_IMPLEMENTATION_2026-11-18.md`。本轮目标是验证新模型是否改善边界质量；运行成功不能作为准确率提升结论。
