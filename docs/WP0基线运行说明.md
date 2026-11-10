# WP0 基线运行说明

WP0 使用现有 NovaTurbo 数据生成器、MLP surrogate 和 Random-NSGA-II，不修改三者的算法行为。运行器负责冻结配置、环境、随机种子、数据划分、Git 状态、评价成本和结果文件。

## 运行环境

当前工作区使用 Conda 环境 inverse_design_nova：

~~~bash
conda run -n inverse_design_nova python scripts/run_wp0_baseline.py \
  --config config/baseline_wp0.yaml
~~~

默认输出目录为：

~~~text
baseline_results/baseline_v1/
~~~

该目录被 Git 忽略。运行器禁止覆盖已有同名目录；需要重跑时应使用新的 run_name，保留原始基线。

## 默认正式配置

- 数据：10,000 个 LHS 样本，seed=42。
- MLP：现有 SurrogateConfig 架构，最多 200 epochs，patience=20。
- NSGA-II：population=100、generations=200、seed=42。
- 重复次数：2，用于比较 surrogate history、NSGA-II history 和 Pareto 结果。
- 物理评价次数：100 × (200 + 1) = 20,100 次/重复。

## 主要产物

~~~text
baseline_results/baseline_v1/
├── baseline_manifest.yaml
├── config.yaml
├── environment.json
├── pip_freeze.txt
├── source_manifest.json
├── git_diff.patch
├── run.log
├── report.md
├── data/
│   ├── dataset_10000.csv
│   └── dataset_summary.json
├── splits/
│   ├── split_indices.csv
│   └── split_summary.json
├── replicates/
│   ├── replicate_01/
│   │   ├── surrogate/
│   │   └── nsga2/
│   └── replicate_02/
│       ├── surrogate/
│       └── nsga2/
└── reproducibility_report.json
~~~

## 已冻结但暂不修正的行为

- surrogate 在切分前使用全部有效样本计算归一化统计量。
- surrogate 的 train/val/test 切分种子在原代码中固定为 42。
- OptimizationConfig 中的工程约束字段尚未参与可行性判定。
- 当前优化器会把名义整数变量作为连续变量初始化。
- 当前优化器依赖 NumPy 全局随机状态。

这些行为属于 WP0 对照基线。任何修正都应在后续独立提交和独立实验中进行，不能覆盖 WP0 结果。

