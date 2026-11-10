# WP1 Generator 数据视图运行说明

WP1 将 NovaTurbo 完整 19 维数据派生为第一版逆问题数据：

~~~text
[thrust_N, tsfc_kg_N_s]
        →
[compressor_pressure_ratio,
 compressor_efficiency,
 turbine_inlet_temp_K,
 turbine_efficiency,
 mass_flow_kg_s]
~~~

## 运行

先完成 WP0，确认以下数据存在：

~~~text
baseline_results/baseline_v1/data/dataset_10000.csv
~~~

然后执行：

~~~bash
conda run -n inverse_design_nova python scripts/build_generator_dataset_v1.py \
  --config config/generator_wp1.yaml
~~~

正式产物写入：

~~~text
data/generator_v1/
├── generator_dataset_v1_all.csv
├── generator_dataset_v1_valid.csv
├── invalid_samples.csv
├── generator_v1_schema.yaml
├── normalizer_v1.json
├── parity_report.csv
├── parity_failures.csv
├── parity_summary.json
├── dataset_card.md
├── wp1_manifest.yaml
├── environment.json
├── source_manifest.json
└── git_diff.patch

data/splits/
└── generator_v1_split.csv
~~~

运行器默认拒绝覆盖以上目录或 split 文件。

## 数据规则

- sample_id 由 dataset_version、稳定 source_id、source_row 和 5D 十六进制浮点表示计算 SHA-256。
- 无效样本保留在 all 视图和 invalid_samples.csv，不进入 valid 训练视图。
- 有效样本固定按 80%/10%/10% 划分 train/val/test。
- 5D 设计变量根据物理边界映射到 [-1,1]。
- 推力使用 train split 的 mean/std。
- TSFC 先取自然对数，再使用 train split 的 mean/std。
- 所有有效样本都会重新调用 Brayton 0D，验证 F/TSFC parity。
