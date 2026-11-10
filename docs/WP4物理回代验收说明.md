# WP4：Physics-in-the-loop Round-Trip 验证

## 1. 实现目标

WP4 使用 NovaTurbo 的 `solve_brayton_cycle` 评价生成候选：

```text
[F_target, TSFC_target]
        ↓ condition normalization
frozen WP3 generator checkpoint
        ↓ K normalized 5D candidates
inverse normalization + finite/bounds/duplicate checks
        ↓ valid physical 5D candidates
EngineInputs → Brayton 0D
        ↓
target error + failure code + Top-K + diversity + abstention
```

评价对象是 `f(x_generated)` 与目标条件之间的误差，不计算生成设计与某个参考设计之间的 x-MSE。这样不会错误惩罚同一性能目标对应的多个可行设计。

## 2. 阶段流程表

| 阶段 | 输入 | 实现内容 | 主要输出 | 验收点 |
|---|---|---|---|---|
| WP4.0 指标冻结 | WP2 valid dataset、normalizer、metrics config | 对每个 test 条件在归一化条件空间搜索最近 train 条件，以冻结分位数误差派生成功阈值；训练域也只由 train split 派生 | `derived_metrics.json` | 阈值不读取 WP4 模型结果，不发生评价泄漏 |
| WP4.1 目标准备 | frozen target suite | 校验 target_id、target_type、expected_status、F、TSFC；条件标准化；不可做 log transform 的目标直接编码为 `TARGET_OUT_OF_DOMAIN` | `evaluated_targets.csv`、每目标 `target.json` | ID/Interpolation 与 OOD/Infeasible 保留原分层 |
| WP4.2 模型加载 | WP3 seed-42 checkpoints | 统一调用 `load_generator`；校验模型类型、checkpoint hash、embedded normalizer 与 WP4 normalizer 完全一致 | `model_loads.csv` | cVAE、nflows、cINN、CFM 使用同一入口 |
| WP4.3 候选生成 | normalized target、model、seed、K | 固定 target/model 派生 seed；调用 `sample(..., space="normalized")`；记录 shape、finite 和 generator diagnostics | `raw_candidates.csv` | 每个候选都可追溯到模型、目标和 seed |
| WP4.4 候选检查 | normalized/physical 5D | inverse transform；检查 NaN/inf、归一化和物理严格边界；以归一化 5D 的 L∞ 距离去重 | `checked_candidates.csv` | 边界值有效；非法候选不进入 Brayton |
| WP4.5 物理回代 | 合法物理 5D、schema | 按字段名构造 `EngineInputs`；逐候选调用 Brayton 0D；solver 异常候选级隔离 | `forward_results.csv`、`evaluation_ledger.csv` | 每个候选都有 forward 状态或明确失败码 |
| WP4.6 误差评价 | 回代 F/TSFC、目标 | 计算推力相对误差、TSFC 相对误差和加权总误差；按冻结阈值标记 success | candidate metrics | 零分母使用 versioned epsilon；只评价物理输出 |
| WP4.7 排序与筛选 | 全部候选指标 | 先按物理有效性、再按总误差稳定排序；在宽松误差池中先取 best-error，再用归一化 5D greedy max-min 增加多样性；候选不足时按配置逐级放宽 | `ranked_candidates.csv` | 选择理由为 `best_error` 或 `diversity_added`，放宽次数被记录 |
| WP4.8 Target 汇总 | ranked candidates | 计算 Top-1/5/10 的 effective K、success 和 best error；统计重复率、coverage、最近邻距离 | `metrics.json`、`target_summary.csv` | 有效候选不足 K 时不虚构候选 |
| WP4.9 OOD/不可行诊断 | target_type、expected_status、命中结果 | 单独记录 rejected、low confidence、false accept 和训练域成员关系 | abstention metrics、`failures.csv` | 不将拒绝目标混入普通 ID/Interpolation 平均误差 |
| WP4.10 G1 验收 | 所有模型和 target 结果 | 检查 checkpoint 加载、目标运行、Brayton 有效结果、failure code 注册完整性 | `functional_gate.json`、`report.md`、`wp4_manifest.yaml` | G1 仅为功能 Gate，不要求 Generator 优于 Random |

## 3. 指标和阈值来源

指标配置位于 `config/generator_wp4_metrics.yaml`。推力和 TSFC 的成功阈值按以下固定过程计算：

1. 仅从 WP2 `train` 和 `test` split 读取条件列。
2. 使用 WP1 冻结的 normalizer 将条件映射到相同归一化空间。
3. 对每个 test target 查找最近的 train condition。
4. 计算最近邻相对推力误差与相对 TSFC 误差。
5. 使用 metrics config 冻结的 90% 分位数，并施加其中记录的最小数值地板。
6. 加权总误差阈值由两个单项阈值及冻结权重直接计算。

运行产物中的 `derived_metrics.json` 保存数据文件 SHA256、split 数量、推导方法、原始分位数和最终阈值。因此阈值可以审计，也不会针对某个生成模型的 WP4 输出临时调整。

## 4. 失败编码

统一 evaluator 注册并输出以下失败码：

- `NON_FINITE`
- `OUT_OF_BOUNDS`
- `DUPLICATE`
- `BRAYTON_EXCEPTION`
- `INVALID_CYCLE`
- `NEGATIVE_OR_ZERO_THRUST`
- `NON_FINITE_TSFC`
- `TARGET_OUT_OF_DOMAIN`
- `NO_CANDIDATE_WITHIN_TOLERANCE`

候选级失败写入 forward ledger；目标没有任何候选命中阈值时，额外写入 target-level `NO_CANDIDATE_WITHIN_TOLERANCE`。异常不会被静默删除。

## 5. 正式运行

项目固定使用物理 GPU 6。`CUDA_VISIBLE_DEVICES=6` 后，进程内该卡映射为逻辑 `cuda:0`：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
conda run -n inverse_design_nova \
python scripts/run_wp4_round_trip.py --config config/generator_wp4.yaml
```

默认配置使用完整的 280 个 frozen targets、四个 seed-42 WP3 checkpoints，并为每个 model/target 生成 64 个候选。`maximum_targets_per_type` 只能用于明确标注的 smoke run；正式验收保持 `null`。

完整训练后另使用 `config/generator_wp4_full_seed42.yaml` 和 `config/generator_wp4_full_seed43.yaml` 对两个训练种子分别重跑完整 target suite；两个 G1 均通过。结果见 `docs/阶段一完整训练与B0正式对比报告.md`。

默认输出目录为：

```text
data/round_trip/wp4_v1_gpu6_final/
├── artifacts/round_trip/<model>__<target-id-prefix>/
│   ├── target.json
│   ├── raw_candidates.csv
│   ├── checked_candidates.csv
│   ├── forward_results.csv
│   ├── ranked_candidates.csv
│   ├── metrics.json
│   ├── failures.csv
│   └── report.md
├── derived_metrics.json
├── evaluation_ledger.csv
├── target_summary.csv
├── model_summary.csv
├── failures.csv
├── model_loads.csv
├── functional_gate.json
├── environment.json
├── git_state.json
├── source_manifest.json
├── report.md
└── wp4_manifest.yaml
```

输出目录默认拒绝覆盖，避免不同模型或阈值版本的结果混合。

## 6. 测试与 G1 判定

运行测试：

```bash
conda run -n inverse_design_nova python -m pytest tests/test_round_trip.py -q
```

测试覆盖已知设计近零误差、一对多解不使用 x-MSE、有效候选不足 Top-K、边界/NaN/重复/无效循环/solver 异常、多样性仅使用 Brayton 有效候选、OOD 单独拒绝，以及 WP2 阈值派生。

G1 通过要求：

- 四类 checkpoint 均能加载并核对 normalizer；
- 每个模型都有目标运行记录；
- 每个模型在普通目标上至少产生可进入 Brayton 的有效回代结果；
- 所有失败均使用已注册 failure code。

Top-K、误差、多样性、拒绝率和 false-accept 均作为第一轮诊断结果保存。它们不参与 G1 的通过判定，也不形成 Generator 相对 Random 的 Go/No-Go。
