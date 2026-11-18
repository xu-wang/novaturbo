# NovaTurbo V2.1 权重补全与独立能力抽取 · 验收证据

> 生成时间：2026-11-18T03:05:45.552304+00:00（UTC）
> 本机：Windows + NVIDIA GeForce RTX 3090（物理索引 0，`CUDA_VISIBLE_DEVICES=0`）
> 运行环境：conda `novaturbo`，Python 3.13.14，torch 2.7.1+cu118

## 1. 生成器权重（CFM-4 bounded-sigmoid, seed 43）

- `data\generator_models\geometry19\wp19_23_cfm4_bounded_sigmoid_seed43_v1\baseline\best.pt`：存在，1668.0 KiB，sha256 `ce4299afe3c1957fb90c4614a13fc10828b251b9bce6420d663e7a3ca765db28`
- `data\generator_models\geometry19\wp19_23_cfm4_bounded_sigmoid_seed43_v1\baseline\normalizer.json`：存在，5.1 KiB，sha256 `2a0e898604d5a41abc835c0eee388358dccdfa12f4a654e102b48d768b7e3a7e`
- `data\generator_models\geometry19\wp19_23_cfm4_bounded_sigmoid_seed43_v1\baseline\last.pt`：存在，1668.0 KiB，sha256 `45fc26b402d0c341a67e1e93ccb9b1e25e550b5b81c94eebc6fc3936a68c99e0`
- `data\generator_models\geometry19\wp19_23_cfm4_bounded_sigmoid_seed43_v1\baseline\training_history.json`：存在，78.3 KiB，sha256 `5851f87eccc60ae7907c7712e1b5bdae273d85aa4a194c8c4d47e9130ffeb362`

| 项 | 值 |
|---|---|
| 状态 | `PASS` |
| 训练设备 | `cuda:0` |
| 完成 epoch | 156（best 136）|
| 早停 | True / early_stopping |
| 参数量 | 418835 |
| 显存峰值 | 101007872 B |
| 训练耗时 | 90.5 s |
| 源码积分一致 | True |
| checkpoint 重载一致 | True |
| 采样在界 | 128/128 |
| 样本去重率(1e-8) | 0.0 |

## 2. 性能代理权重（performance3_log, 三输出自然对数）

- `data\inverse_design\wp19_22_performance_surrogate_v2_ablation\formal\models\performance3_log\seed_43\weights_best_val.pt`：存在，583.2 KiB，sha256 `1a132c99d8fbfe3d14d79a480b53d93117b0491d5db23043ffe7469fd182cd3c`
- `data\inverse_design\wp19_22_performance_surrogate_v2_ablation\formal\models\performance3_log\seed_43\normalization.npz`：存在，1.2 KiB，sha256 `eb9352eb8499a237cee6048e773ad94baf76b77312b3ea3fb4e70203559fca6a`
- `data\inverse_design\wp19_22_performance_surrogate_v2_ablation\formal\models\performance3_log\seed_43\model_manifest.json`：存在，1.9 KiB，sha256 `881454a80205a05aadb4e77a9b889744f67d537ccce115202452b56a2c115e8f`
- `data\inverse_design\wp19_22_performance_surrogate_v2_ablation\formal\models\performance3_log\seed_43\training_metrics.json`：存在，12.2 KiB，sha256 `63f15d6d5d7384220324c490b5c84473c00564c5cb278e0bd48d28f7d01f5b1e`

- model_tag：`novaturbo_geometry19_performance3_log_seed43_v1`；output_transform：`natural_log`；device：`cuda:0`
- 完成 epoch：80（best 70）
- 数据划分：{'test': 3873, 'train': 15612, 'validation': 3938}

| 轴 | 测试集 p50 相对误差 | 测试集 p90 | 验证集 p50 | R²（test）|
|---|---:|---:|---:|---:|
| thrust_N | 10.0308% | 38.2388% | 10.0731% | 0.8971 |
| tsfc_kg_N_s | 4.7107% | 16.5831% | 4.9404% | 0.8699 |
| mass_kg | 1.6715% | 4.2142% | 1.7383% | 0.9910 |

## 3. V2.1 流水线端到端

- 运行目录：`D:\work\novaturbo\data\inverse_design\v21_runs\run2`
- 状态：`PASS`；终止阶段：`WP19-23C`；交付状态：`CLOSED_LOOP_DEVELOPMENT_DIAGNOSTIC_HIT_DELIVERED`
- 交付 design_id：`['a0ddaaf255604e3c519bf2d7289e252fe3c614222c7ee6815614ee299c5370b2']`
  - candidate 71：hit=True，rho=0.882298316364928，failed_axes=0，thrust=64.50058463444473，tsfc=6.46869922139107e-05，mass=2.28334390732512
- 聚合 manifest：存在（sha256 `69534dd187341e49eea1b3c0dab401842b930036922fb30c0f28b69e7e64a825`）

## 4. 独立能力包

- 目录：`D:\work\novaturbo\v21_capability`
- 模块数：54；mesh 支持：True
- 生成器权重：`data\generator_models\geometry19\wp19_23_cfm4_bounded_sigmoid_seed43_v1\baseline`
- 性能代理权重：`data\inverse_design\wp19_22_performance_surrogate_v2_ablation\formal\models\performance3_log\seed_43`
- 体积：3.3 MiB

## 5. 交付阻塞缺陷（已修复，一行）

交付快照 `src/visualization/geometry19_service.py::regenerate` 合并
`{**mesh_payload, **solve_payload}` 时，两个字典都含 `calibration_hash`：前者是可视化组合标定哈希
（与磁盘 `preview_mesh/manifest.json` 一致），后者是物理标定包哈希，后者覆盖前者，导致
`geometry19_delivery.py:204` 的 `Visualization calibration provenance mismatch` 必然触发。
修复：`solve_payload` 的键改名为 `physics_calibration_hash`，`calibration_hash` 保留可视化语义。
修复后 `test_geometry19_service / delivery / ui / geometry22_mesh_ui` 共 20 项通过，流水线端到端 PASS。

## 6. 独立能力包验证

| 检查 | 结果 |
|---|---|
| 隔离目录（仅依赖 numpy/scipy/torch/PyYAML/torchcfm/torchdiffeq）导入 | 通过 |
| 包内测试 `tests/test_bundle.py` | 6 passed（含 D-Flow 引导收敛性检查）|
| `predict` 参考设计 | 67.15 N / 7.153e-5 / 1.346 kg（物理真值 68.35 / 7.085e-5 / 1.329）|
| `evaluate` 参考设计 | 求解收敛，命中/约束裁决完整 |
| 引导式逆设计（64 N / 6.5e-5 / 2.26 kg） | `DIAGNOSTIC_HIT`，rho 0.8823，0 失配轴，43 次物理调用，23 s |
| 包内 `export_mesh` | 12 片 STL，诊断 passed |

## 7. 边界声明

- `formal_use_allowed=false`、`deployment_allowed=false`：物理为项目内低阶闭式模型，不等于 CFD、台架试验或制造验证。
- 两份权重均为**本地按仓库冻结配置重训**的产物，不是发布版 checkpoint；不得引用原仓库的命中率或对比结论。
- 命中口径为 `rho = max(相对误差 / 轴容差)`，容差 2.7451% / 0.8573% / 1.1707%，仅作开发诊断。
