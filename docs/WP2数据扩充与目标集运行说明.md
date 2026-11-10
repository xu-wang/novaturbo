# WP2 数据扩充、清洗与目标集运行说明

WP2 将 WP1 的固定 5D→2D schema 扩展到 100K 级低成本 Brayton 0D 数据，并冻结不接触训练 split 的正式测试目标。WP2 不修改物理求解器，也不改变 WP1 的逆问题定义。

## 运行

在仓库根目录执行：

```bash
python scripts/run_wp2_data_expansion.py --config config/generator_wp2.yaml
```

默认输出到 `data/generator_v1_100k/`。该目录被 Git 忽略，正式配置和实现代码仍受版本控制。

## 可恢复分片流程

1. 先使用固定 seed 生成一次全局 19D Latin Hypercube；再按 `shard_size` 切片，因此断点恢复不会改变全局分层。
2. 每个 `raw_shards/part-*.csv` 都有对应的 YAML 元数据，记录全局行号范围、seed、Git commit、配置/schema/边界哈希、列和 dtype、CSV 哈希。
3. 恢复时仅复用元数据契约和文件哈希都一致的完成分片；不完整或失配分片会重新计算。
4. 合并阶段要求全局 sample index 恰好覆盖 `[0, n_samples)`，并按 `raw_sample_id` 去重。

## 清洗与数据视图

清洗规则全部位于 `config/generator_wp2.yaml`。默认要求：原始物理解有效、5D 与 2D 条件均为有限值、推力和 TSFC 为正。无效行不会进入模型训练，但会完整保留在：

- `dataset_100000_raw.csv`
- `dataset_cleaning_ledger.csv`
- `invalid_samples.csv`
- `data_quality_report.yaml`

扩充后会重新生成 train/val/test split，并仅用新 train split 拟合 `normalizer_v1.json`。不会复用 WP1 的 10K 行号或 split。

## 覆盖与可行域产物

- `coverage_bins.csv`、`coverage_report.json`：19D 边界、分箱和空洞检查；整数维仅检查理论可占据分箱。
- `feasible_domain_report.json`：F/TSFC 分位数、相关性、Pareto 数量。
- `condition_density.npz`、`feasible_domain.png`：联合密度和 Pareto 可视化。
- `pareto_reference.csv`：由有效样本直接计算的非支配参考点。
- `parity_report.csv`：generator view 对 Brayton 0D 的逐行回代一致性。

## 冻结目标集

目标位于 `targets/`：

| 文件 | 内容 | 构造约束 |
|---|---|---|
| `targets_id.csv` | held-out test 的已知可行条件 | 保存精确 5D 前向解，不使用 train 样本 |
| `targets_interp.csv` | 局部插值目标 | 在 test 条件近邻对应的 5D 间插值，再调用 Brayton 0D 得到成对 F/TSFC |
| `targets_ood.csv` | 稀疏边界与不可行目标 | OOD 来自距 train 较远的 held-out 条件；不可行项来自显式域违规或联合边界外推 |
| `target_suite_v1.csv` | 统一目标套件 | OOD/不可行目标明确排除在普通 ID 误差之外 |

所有目标均固定 `target_id`、类型、预期状态、构造方法、来源 sample_id 和版本。实现禁止独立随机组合推力与 TSFC，并在冻结前检查目标来源 sample_id 及普通目标的精确条件均不与 train 重叠。

## 完成判据

- 原始合并数据 sample ID 唯一，行数与配置一致。
- 19D LHS 没有非预期空箱。
- 无效样本保留且 failure code 可统计。
- 新 split 互斥，normalizer 仅使用 train。
- ID、Interpolation、OOD-feasible-unknown、Infeasible 均已生成并版本化。
- 目标来源与 train 无交叉；不存在独立 F/TSFC 拼接。
- 所有有效 generator 样本通过 Brayton 0D parity。
