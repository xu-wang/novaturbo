# WP3 开源条件生成模型筛选、冻结与最小适配说明

WP3 的完成标准是四类条件生成器通过同一数据接口完成训练、保存、加载和确定性采样。该阶段是功能 Gate，不要求 Generator 优于 Random。

## 最终冻结结果

| 角色 | 冻结实现 | 版本/commit | 许可证 | 本地适配 |
|---|---|---|---|---|
| CVAE | AntixK/PyTorch-VAE | `a6896b944c918dd7030e7d795a8c13e5c6345ec7` | Apache-2.0 | 保留重参数化与 reconstruction + beta-KL；CNN/离散标签替换为 5D/2D MLP |
| Conditional Flow | bayesiains/nflows | `0.14` / `64b856c081e5f07521b32be99da262e8338fbfe8` | MIT | 使用 `Flow`、条件 MAF、`log_prob` 和 `sample`，不实现 Jacobian |
| cINN | VLL-HD/FrEIA | `0.2` / `033f6fd6eb4d2ddbc98f746a9ef5f8c619587abe` | MIT | 使用 `SequenceINN` 和 `AllInOneBlock`，只提供 5D、2D condition 和 subnet 尺寸 |
| CFM | atong01/conditional-flow-matching | `1.0.7` / `3fd278f9ef2f02e17e107e5769130b6cb44803e2` | MIT | 使用 `ConditionalFlowMatcher` 的概率路径和 target velocity；本地仅提供条件速度 MLP |
| ODE | rtqichen/torchdiffeq | `0.2.5` | MIT | CFM 的固定 RK4 采样器，记录步数、NFE 和耗时 |

完整仓库、论文、示例路径、核心类、package hash、许可证 hash和允许修改范围见 `config/third_party_models.lock.yaml`。许可证副本及派生说明位于 `third_party/`。

Meta `flow_matching` 因 CC-BY-NC-4.0 限制仅登记为研究参考，不进入默认依赖。

## 代码结构

```text
src/ai/generators/
├── base.py               # ConditionalGenerator、训练外壳、checkpoint
├── data.py               # 公共 WP2 CSV/Dataset/DataLoader
├── cvae.py               # MLP-cVAE 最小适配
├── nflows_adapter.py     # nflows 薄 adapter
├── freia_adapter.py      # FrEIA cINN 薄 adapter
└── torchcfm_adapter.py   # TorchCFM + torchdiffeq 薄 adapter
```

上层统一调用：

```python
generator.fit(train_loader, val_loader)
result = generator.sample(condition, n_samples=32, seed=42, space="normalized")
generator.save(checkpoint)
restored = load_generator(checkpoint)
```

`GeneratedSamples` 明确保存 `values`、`space`、condition、seed 和模型特有诊断。单条件返回 `[n_samples, 5]`，批条件返回 `[batch, n_samples, 5]`。所有 adapter 均禁止静默裁剪越界样本。

## 数据一致性

- 四个模型读取同一 `generator_dataset_v1_valid.csv`、split、schema 和 train-only normalizer。
- 正式配置从 WP2 训练集固定抽取 20K、验证集固定抽取 5K，用于第一轮功能实现；抽样 seed 与模型 seed 分离。
- test split 和 target suite 不参与模型选择。
- checkpoint 内嵌模型配置、schema version、normalizer、训练历史、上游 package 版本和完整 upstream lock。

## 运行

安装锁定依赖：

```bash
python -m pip install -r requirements.txt
```

执行正式 WP3：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
conda run -n inverse_design_nova python scripts/run_wp3_generators.py \
  --config config/generator_wp3_full.yaml
```

完整训练产物位于 `data/generator_models/wp3_full_gpu6_v2/`，该目录被 Git 忽略。

## 验收内容

1. `upstream_smoke.csv`：每个上游核心的 loss、梯度、采样 shape/finite；FrEIA 额外检查 forward→inverse 误差。
2. `compatibility_report.md`：实际 package、CPU/GPU 可用性与 smoke 结论。
3. `seed_summary.csv`：每个模型、每个 seed 的 best validation loss、训练耗时、越界率、多样性和 checkpoint 一致性。
4. `best.pt` 与 `last.pt`：验证最优和最后一轮 checkpoint 均保留。
5. `sample_diagnostics.json`：重复率、越界率、维度标准差；CFM 额外记录 ODE NFE 和耗时。
6. `wp3_manifest.yaml`：数据/target hash、模型、seed、全部 checkpoint hash 和功能 Gate 状态。

2026-08-10 的正式训练已固定使用物理 GPU 6，进程内映射为逻辑 `cuda:0`。四个模型、两个训练种子均使用完整的 79,945 个训练样本和 9,992 个验证样本；详细收敛记录和下游结果见 `docs/阶段一完整训练与B0正式对比报告.md`。
