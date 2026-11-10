# WP6：完整逆设计编排层

## 1. 实现目标

WP6 将 WP3、WP4、WP5 和原 NSGA-II 连接成单次、无人工中途选择的可恢复流程：

```text
TargetSpec + RunConfig
        ↓ compatibility validation
conditional generator
        ↓ normalized 5D candidates
inverse normalization + candidate validation
        ↓ physical 5D candidates
Brayton 0D round-trip
        ↓ ranked and diverse candidates
named 5D → frozen 19D background
        ↓ generation-0 population
original NSGA-II
        ↓
Pareto + history + evaluation ledger + cost ledger + manifest
```

Random 模式跳过 Generator 与预筛选，直接调用 WP0 原始初始化路径；Generator 和 Hybrid 使用完全相同的后续 NSGA-II。

## 2. 核心配置

### TargetSpec

TargetSpec 至少包含：

```yaml
target_version: wp6_target_v1
target_id: target-id
target_type: ID
expected_status: known_feasible
thrust_N: 108.6
tsfc_kg_N_s: 4.13e-05
tolerance_profile: round_trip_metrics_v1
```

`target_type` 支持 `ID`、`Interpolation`、`OOD-feasible-unknown` 和 `Infeasible`。无法进行条件变换的目标不会使流程崩溃，而是记录 `TARGET_OUT_OF_DOMAIN` 并将 Generator/Hybrid 安全降级到背景种群。

### RunConfig

RunConfig 冻结：

- `run_id`、`mode`；
- `model_id`、checkpoint；
- dataset、schema、normalizer、metrics config 及版本/hash；
- generator、optimizer、background seed；
- 生成数量与多样性选择数量；
- population、generator ratio 和背景方法；
- 完整 NSGA-II 配置；
- runtime device 和输出根目录。

Random 模式禁止配置 checkpoint；Generator/Hybrid 模式必须配置 checkpoint。为公平性，WP6 要求 `background seed == optimizer seed`。

## 3. 阶段流程表

| 阶段 | 实现内容 | 主要产物 | 退出检查 |
|---|---|---|---|
| `validated` | 校验 target、dataset version、schema、normalizer version/hash、metrics version、checkpoint model/schema/normalizer | `validated_context.json`、`derived_metrics.json` | 不兼容立即拒绝 |
| `generation_round_trip` | 采样、逆归一化、finite/bounds/duplicate、Brayton、误差、Top-K、多样性筛选 | `generator/*`、`round_trip/*` | 每个候选有状态；0D 次数记录 |
| `population` | 按名覆盖 5D→19D、构造 Generator/Hybrid/Random、候选不足时补齐 | `initial_genes.csv`、`population_provenance.csv` | 种群完整、来源逐行可追溯 |
| `optimization` | 调用原 `optimize`，记录 generation-0 和每代 offspring 的全部物理评价 | Pareto、history、`evaluation_ledger.csv` | ledger 数等于 `population × (generations+1)` |
| `summary` | 汇总生成、预筛选、适配和 NSGA 成本；写环境、Git、源码清单 | metrics、cost ledger、report、manifest | 两套成本账本平衡 |

## 4. 单命令运行

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
conda run -n inverse_design_nova \
python -m src.ai.inverse_design \
  --target-config config/inverse_design/targets/target_001.yaml \
  --run-config config/inverse_design/runs/cvae_generator.yaml
```

已有运行目录默认拒绝覆盖。恢复中断运行：

```bash
python -m src.ai.inverse_design \
  --target-config <same-target.yaml> \
  --run-config <same-run.yaml> \
  --resume
```

恢复前必须满足：

- manifest 格式受支持；
- run_id 和 target_id 相同；
- TargetSpec SHA256 相同；
- RunConfig SHA256 相同；
- 所有标记 completed 的阶段产物仍然存在。

任何条件不满足都会拒绝恢复。运行失败或显式中断时，已完成阶段及其产物保留，manifest 记录失败类型和当前阶段。

## 5. Artifact 结构

```text
data/inverse_design/<group>/<run_id>/
├── manifest.yaml
├── target.json
├── target.yaml
├── run_config.yaml
├── validated_context.json
├── derived_metrics.json
├── environment.json
├── git_state.json
├── source_manifest.json
├── generator/
│   ├── raw_candidates.csv
│   ├── checked_candidates.csv
│   └── sampling_metrics.json
├── round_trip/
│   ├── forward_results.csv
│   ├── ranked_candidates.csv
│   ├── failures.csv
│   └── metrics.json
├── population/
│   ├── initial_genes.csv
│   ├── population_provenance.csv
│   └── adapter_metrics.json
├── optimization/
│   ├── pareto.csv
│   ├── history.csv
│   ├── evaluation_ledger.csv
│   └── metrics.json
└── summary/
    ├── metrics.json
    ├── evaluation_cost_ledger.csv
    └── report.md
```

Pareto 行保留 individual provenance；evaluation ledger 包含 generation、initial/offspring、individual ID、parents、19D genes、objective 和 performance。

## 6. 成本记账

`summary/metrics.json` 分别报告：

- Generator sampling seconds；
- CFM ODE NFE；
- Round-Trip seconds；
- adapter seconds；
- optimizer seconds；
- generator prefilter 0D evaluation count；
- NSGA-II 0D evaluation count；
- end-to-end total 0D evaluation count；
- 当前执行时间和可跨恢复累计的阶段时间。

`evaluation_cost_ledger.csv` 的预筛选与 NSGA-II 两行之和必须等于 total；NSGA-II 行必须等于 `optimization/evaluation_ledger.csv` 行数。

## 7. G2 正式验收矩阵

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
conda run -n inverse_design_nova \
python scripts/run_wp6_pipeline.py \
  --config config/inverse_design/wp6_validation.yaml
```

矩阵包含：

| Run | 模式 | 模型 |
|---|---|---|
| `wp6_random` | Random | 无 |
| `wp6_cvae_generator` | Generator | cVAE |
| `wp6_nflows_hybrid` | Hybrid | nflows |
| `wp6_cinn_hybrid` | Hybrid | cINN |
| `wp6_cfm_hybrid` | Hybrid | CFM |

cVAE Generator 运行会用相同 target、checkpoint、配置和 seed 重复一次，并逐文件比较 raw candidates、ranked candidates、initial genes、population provenance、Pareto、history 和 evaluation ledger 的 SHA256。

默认验收采用 `population=12, generations=2`，用于验证管线功能，不用于方法性能结论。

## 8. 测试与 G2 判定

```bash
conda run -n inverse_design_nova python -m pytest tests/test_inverse_design.py -q
```

测试覆盖：

- TargetSpec/RunConfig 校验；
- 完整 Generator→Round-Trip→19D→NSGA-II；
- 候选不足的可追溯 Random fill；
- 预筛选/NSGA/总成本一致；
- population 阶段中断和恢复；
- 配置改变后拒绝恢复；
- checkpoint normalizer 不兼容拒绝；
- OOD 无法采样时安全降级；
- 相同输入和 seed 的核心产物逐文件一致；
- Pareto 和 evaluation ledger 来源追溯。

G2 通过要求四类模型使用同一编排器、三种初始化模式均完成、全部阶段和成本账本一致、provenance 完整、重复运行一致。G2 不要求 Generator 优于 Random。
