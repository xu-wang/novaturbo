# tools/v21 — V2.1（Geometry19 19D）权重补全与能力抽取

本目录只放"把 V2.1 跑起来并抽出去"所必需的工具，不改动仓库原有流水线代码与冻结配置。

## 背景

交付包里缺少两类运行必需资产（`DELIVERY.md` 明确说明数据与权重被排除）：

| 资产 | 协议声明路径 | 训练入口 |
|---|---|---|
| CFM-4 bounded-sigmoid 生成器（19D，seed 43） | `data/generator_models/geometry19/wp19_23_cfm4_bounded_sigmoid_seed43_v1/baseline/{best.pt,normalizer.json}` | `scripts/train_wp19_23_geometry19_cfm4.py`（仓库自带） |
| `performance3_log` 性能代理（19D→3 输出，自然对数） | `data/inverse_design/wp19_22_performance_surrogate_v2_ablation/formal/models/performance3_log/seed_43/{weights_best_val.pt,normalization.npz,model_manifest.json}` | 仓库的 ablation 脚本依赖缺失的 S0 基线与 gate-H 池，故由本目录 `train_performance3_log.py` 以同一架构/数据/变换/早停策略重训 |

两者都依赖同一份训练数据：WP19-21 的 98,304 样本开发数据集（6 个 campaign、各 16,384，LHS 采样 + V2.1 物理求解）。

## 执行顺序

```powershell
$py = "C:\Users\QY\miniconda3\envs\novaturbo\python.exe"

# 1) 基础 campaign（16,384 样本，含 256 个预览网格）
& $py scripts\run_wp19_21_development_dataset.py

# 2) 另外 5 个 campaign + 合并（本工具用可配置并行度，协议默认只有 4）
& $py tools\v21\gen_dataset_fast.py --workers 14

# 3) 生成器（官方训练入口，bounded-sigmoid 配置）
$env:CUDA_VISIBLE_DEVICES = "0"
& $py scripts\train_wp19_23_geometry19_cfm4.py `
    --config config\experiments\geometry19_cfm4_bounded_sigmoid_seed43_v1.yaml

# 4) 性能代理（三输出，自然对数，seed 43）
& $py tools\v21\train_performance3_log.py --device cuda:0

# 5) 验收两份权重与物理链
& $py tools\v21\verify_v21.py --device cuda:0

# 6) V2.1 流水线端到端
$env:CUDA_VISIBLE_DEVICES = "0"
& $py scripts\run_wp19_23_pipeline.py `
    --thrust-N 60 --tsfc-kg-N-s 0.00006 --mass-kg 0.75 --device cuda:0

# 7) 抽取独立能力包
& $py tools\v21\build_bundle.py --output v21_capability --mesh --force
```

## 与协议的三处有意偏差（必须记录）
1. **GPU 编号**：`AGENTS.md` 要求物理 GPU 6，本机只有 1 张 RTX 3090（索引 0），
   所有 CUDA 运行使用 `CUDA_VISIBLE_DEVICES=0`，进程内 `cuda:0`。
2. **数据并行程数**：`config/physics/wp19_21_dataset_expansion_98304_v1.yaml` 冻结为 4 个
   worker；`gen_dataset_fast.py` 用 14 个并行进程生成其余 campaign。每个样本的 LHS 设计、
   种子与物理求解完全不变，只改变吞吐；合并 manifest 保留原 provenance 哈希，并额外记录
   `generation.workers` 与实际使用的驱动脚本。
3. **权重是本地重训产物**：不是发布版 checkpoint。生成器按仓库冻结配置（seed 43、400 epochs、
   patience 20、bounded offset-sigmoid）训练并通过仓库自带校验门；性能代理按
   `wp19_22_geometry19_surrogate_v1` 架构与 `performance3_log` 变换训练，manifest 里写明
   `training_note = "retrained from delivery-local data; not the published WP19-22 checkpoint"`。
   因此**不能**引用原仓库的命中率/对比结论作为本权重的指标。

## 交付阻塞缺陷与修复（必须记录）

交付快照里 `src/visualization/geometry19_service.py::regenerate` 把 `mesh_payload` 与
`solve_payload` 合并为响应时，两个字典都含 `calibration_hash`：前者是**可视化组合标定哈希**
（与磁盘 `preview_mesh/manifest.json` 一致），后者是**物理标定包哈希**。`{**mesh_payload,
**solve_payload}` 让后者覆盖前者，于是 F 阶段
`geometry19_delivery.py:204` 的 `"Visualization calibration provenance mismatch"` 必然触发，
V2.1 流水线无法完成交付。

修复（一行，语义不变）：把 `solve_payload` 里的键改名为 `physics_calibration_hash`，
`calibration_hash` 保留 `mesh_payload` 的可视化语义；物理哈希没有丢失，只是换了不冲突的名字。
已验证：`tests/test_geometry19_service.py`、`tests/test_geometry19_delivery.py`、
`tests/test_geometry19_ui.py`、`tests/test_geometry22_mesh_ui.py` 共 20 项通过，
V2.1 流水线端到端 PASS。

## 实测结果（2026-11-18，本机）

| 项 | 结果 |
|---|---|
| 训练数据 | 98,304 样本 / 23,423 收敛（23.83%）/ 0 重复 / 512-512 预览通过 |
| 生成器 | PASS，156 epochs（best 136，早停），418,835 参数，90.5 s，显存峰值 101 MB，采样 128/128 在界 |
| 性能代理 | 80 epochs（best 70），测试集 p50 相对误差：推力 10.03% / TSFC 4.71% / 质量 1.67%；R² 0.897 / 0.870 / 0.991 |
| 参考设计代理 vs 物理 | 推力 1.8% / TSFC 0.9% / 质量 1.3% |
| 流水线（64 N / 6.5e-5 / 2.26 kg） | PASS，C 阶段直接命中，rho 0.8823，交付 1 个设计，30.6 s |
| 独立能力包 | 3.3 MiB / 142 文件 / 52 模块；隔离目录 6 项测试通过；引导式逆设计 23 s 得到同一命中（rho 0.8823，43 次物理调用） |

## 边界

`formal_use_allowed=false`、`deployment_allowed=false`：物理是项目内低阶闭式模型，
不等于 CFD、台架试验或制造验证；代理只用于排序，不参与命中裁决。
