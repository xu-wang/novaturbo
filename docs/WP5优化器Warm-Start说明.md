# WP5：改造 optimizer.py 支持 Warm-Start

## 1. 实现边界

WP5 只替换 NSGA-II 的 generation-0 种群构造入口。以下原有逻辑保持共用：

- `evaluate_design` 物理评价；
- tournament selection；
- SBX crossover；
- polynomial mutation；
- non-dominated sorting；
- crowding-distance selection；
- 原有约束和目标定义。

未传 `initial_genes` 或 `initializer` 时，优化器仍逐行调用全局 NumPy RNG 构造 19D 均匀随机种群，其随机数消耗顺序与 WP0 完全一致。

## 2. 优化器接口

```python
pareto, history = optimize(
    config=config,
    bounds=bounds,
    verbose=False,
    initial_genes=genes,      # ndarray 或 InitialPopulation
    initializer=initializer,  # callable(context)，与 initial_genes 二选一
)
```

`initial_genes` 和 `initializer` 同时提供时立即报错，不静默选择。

自定义 `initializer` 接收只读 `InitializationContext`：

- `param_names`：唯一且有序的 19D 字段名；
- `bounds_lo`、`bounds_hi`：与字段顺序一致的边界；
- `population_size`：目标种群规模。

它可以返回二维 `ndarray`，也可以返回带逐行 provenance 的 `InitialPopulation`。

## 3. 外部 19D 种群处理流程

| 步骤 | 输入 | 实现内容 | 输出/异常 |
|---:|---|---|---|
| 5.1 | `DesignSpaceBounds` | 生成唯一的 `param_names`、`bounds_lo`、`bounds_hi` | 固定 19D optimizer schema |
| 5.2 | external genes | 强制二维、19 列、float | shape 不正确立即失败 |
| 5.3 | external genes | 检查 NaN/inf | 记录位置并失败 |
| 5.4 | external genes | 检查上下界 | 默认 `reject`；显式实验可配置 `clip` 并统计裁剪值数 |
| 5.5 | external genes | 对三个 blade/injector 整数字段执行 `rint` 和边界投影 | 记录被投影值数 |
| 5.6 | normalized 19D | 按归一化 L∞ 距离稳定去重，保留首行 | 重复数量和源行索引 |
| 5.7 | unique genes | 超过 population 时保留输入前缀 | 稳定截取数量 |
| 5.8 | short genes | 使用独立 seed 的 Random 或 LHS 补齐 | 每行标记 `random_fill`/`lhs_fill` |
| 5.9 | complete genes | 创建带 `individual_id` 和 provenance 的 Individual | generation-0 population |
| 5.10 | external path | 推进与 WP0 初始化相同数量的全局 RNG draw | 后续遗传随机流与 Random 对齐 |

处理统计写入 `history["initialization"]`，初始矩阵和逐行来源分别写入：

```python
history["initial_population_genes"]
history["initial_population_provenance"]
```

## 4. 5D 到 19D 的按名适配

`build_warm_start_population` 不假设五个字段在 19D 数组中的位置。

| 步骤 | 实现内容 | 追溯信息 |
|---:|---|---|
| A1 | 从 generator schema 读取五个字段名 | `generated_fields` |
| A2 | 用 `background_seed` 和 Random/LHS 构造完整 19D 背景 | `background_row`、`background_method` |
| A3 | 校验 generator 5D finite、物理边界和重复 | 输入数、唯一数、重复源行 |
| A4 | 按上游 rank 顺序选取所需候选 | `generated_source_row`、`candidate_id`、`rank` |
| A5 | 只覆盖对应的五个命名字段 | 每行 `generated_fields` |
| A6 | 候选不足时保留原背景行 | `random_fill` |
| A7 | 验证未生成的 14D 与冻结背景逐元素相同 | `background_14d_preserved` |

Generator V1 的五个字段都是连续变量。因此 named adapter 不修改背景中的整数变量，确保 Random、Generator 和 Hybrid 使用同一份原始 14D 背景。任意外部 19D 数组仍按默认规则执行整数投影。

## 5. 三种初始化模式

| 模式 | generation-0 构造 | 优化器调用 |
|---|---|---|
| `random` | 原 WP0 19D 均匀随机种群 | 不传外部种群 |
| `generator` | 尽量用 Generator 5D 覆盖全部背景行，不足时保留背景并标记 `random_fill` | `initial_genes=InitialPopulation` |
| `hybrid` | 按 `round(population_size × generator_ratio)` 覆盖背景前缀 | `initial_genes=InitialPopulation` |

0%、30%、50%、70%、100% 比例均有精确构造测试。输入候选超过需要数量时，按已排序的输入前缀截取，不重新按模型类型分支。

## 6. 公平随机数处理

Random 初始化本身会消耗：

```text
population_size × 19
```

个全局 NumPy 随机数。外部种群若不做处理，虽然 optimizer seed 相同，后续 tournament、crossover 和 mutation 却会从不同的 RNG 位置开始。

默认 `warm_start_align_legacy_rng: true` 会在外部种群准备完成后消耗相同数量的占位随机数。因此相同 optimizer seed 下，Random 与 Warm-Start 的后续遗传随机流对齐，差异主要来自 generation-0 population。

## 7. 正式功能验收

运行：

```bash
conda run -n inverse_design_nova \
python scripts/run_wp5_warm_start.py --config config/optimizer_wp5.yaml
```

WP5 使用 WP4 冻结的 cVAE/ID ranked candidates，自动选择第一个至少具有配置数量 Brayton-valid 候选的目标，并执行：

- Random 0%；
- Hybrid 30%；
- Hybrid 50%；
- Hybrid 70%；
- Generator 100%。

默认功能验收采用较小的 `population=20, generations=3`，用于验证接口、映射、来源记录和完整 NSGA-II 路径，不用于方法性能结论。

默认输出：

```text
data/warm_start/wp5_v1_final/
├── build_config.yaml
├── input_ranked_candidates.csv
├── frozen_background_population.csv
├── runs/<run_id>/
│   ├── initial_genes.csv
│   ├── population_provenance.csv
│   ├── initialization.json
│   ├── pareto.csv
│   ├── history.csv
│   └── metrics.json
├── run_summary.csv
├── functional_gate.json
├── report.md
└── wp5_manifest.yaml
```

## 8. 测试与退出条件

```bash
conda run -n inverse_design_nova \
python -m pytest tests/test_optimizer_warm_start.py -q
```

测试覆盖：

- WP0 Random 初始矩阵逐元素回归；
- 合法 `(N,19)` 外部种群；
- 短种群可重复补齐和长种群稳定截取；
- NaN、inf、越界、重复和整数投影；
- `initial_genes`/`initializer` 互斥；
- downstream RNG 对齐；
- 5D 按字段名覆盖及其余 14D 不变；
- 0/30/50/70/100% 比例；
- Generator 不足时可追溯降级；
- Random、Generator、Hybrid 完整 NSGA-II；
- 小型端到端 artifact 和 functional gate。

WP5 通过只说明三种初始化模式功能正确、Random 回归兼容、19D 映射可追溯。它不要求 Warm-Start 优于 Random。
