# NovaTurbo 本机（Windows）运行与使用指南

> 生成日期：2026-11-18。所有"可用/不可用"结论均在本机实际执行验证，不是照抄 README。
> 适用交付包：`turbo_v3_22d_7c88f776`（`V3-develop` 分支快照，git HEAD `58517c6 "turbo v3"`）。

## 1. 这个项目是什么

一个"微型涡喷发动机数字化设计系统"的代码交付包，包含四条能力线：

| 能力线 | 内容 | 入口 | 本机状态 |
|---|---|---|---|
| A. 参数化几何 + Brayton 热力循环 | 5 部件几何、站位热力分析、材料温度校核 | `app.py --geometry / --brayton` | ✅ 可用 |
| B. 代理模型 + NSGA-II + 启发式"逆设计建议" | 19 输入 MLP 代理、多目标优化、UI 快速反算 | `app.py --train/--optimize`、`POST /api/inverse_design` | ✅ 代理推理可用（权重在包内） |
| C. Geometry19 三目标逆设计流水线（V2.1 默认） | CFM-4 条件生成 → 代理排序 → V2.1 物理裁决 → 梯度增强 → NSGA-II → 交付 | `scripts/run_wp19_23_pipeline.py` | ❌ 缺生成器/代理权重 |
| D. V3 22D 开发线（V2.2 / V3） | 22D 设计、可视化网格、网页任务 | `--model "model V2.2"/"model V3"`、网页 V2.2/V3 模式 | ❌ 两级阻断（见 §5） |

外加一个 Flask + Three.js 三维查看器/工程仪表盘（`ui/`），是本机目前最能直接"用起来"的部分。

## 2. 本机环境（已验证）

```text
Conda 环境：C:\Users\QY\miniconda3\envs\novaturbo
Python：3.13.14
torch：2.7.1+cu118，torch.cuda.is_available() == True
GPU：NVIDIA GeForce RTX 3090（24 GB），全机仅 1 张，逻辑索引 0
numpy 2.5.2 / scipy 1.18.0 / flask 3.1.3 / flask-cors 6.0.5 / trimesh 5.0.0
cadquery 2.8.0 / CoolProp 8.0.0 / nflows 0.14 / FrEIA 0.2 / torchcfm 1.0.7 / torchdiffeq 0.2.5
前端依赖：ui/static/js/node_modules/ 已自带 three@0.15x、chart.js～4.x，无需 npm install
```

**不要**用系统自带的 `C:\Users\QY\AppData\Local\Python\pythoncore-3.14-64\python.exe`（Python 3.14，无任何依赖）。

`conda.exe` 不在 PATH，需用全路径或先 `& C:\Users\QY\miniconda3\shell\condabin\conda-hook.ps1`。

### GPU 默认设备的偏差声明

项目 `AGENTS.md` 要求默认物理 GPU `6`、进程内选 `cuda:0`。**本机只有 1 张卡，物理索引 0，不存在 GPU 6**（`nvidia-smi -i 6` 返回 "No devices were found"）。因此本机应改为：

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"   # 物理 0 号卡
# 进程内仍选 cuda:0
```

同时注意：网页 V2.2 页面把 `CUDA_VISIBLE_DEVICES='6'` 写死（`ui/server.py` 的 `_run_v22_ui_job`），在单卡机器上即使权重齐全也会因看不到设备而失败，需改成 `0`。

## 3. 可直接使用的命令（均已实测通过）

统一先设 python 别名：

```powershell
$py = "C:\Users\QY\miniconda3\envs\novaturbo\python.exe"
$env:TMP  = "D:\work\novaturbo\.tmp"   # pytest 需要，见 §6 备注
$env:TEMP = "D:\work\novaturbo\.tmp"
```

### 3.1 三维查看器（最推荐先跑这个）

```powershell
cd D:\work\novaturbo
& $py app.py --ui --port 5055
# 浏览器打开 http://127.0.0.1:5055
```

实测：`GET /` 200；`/api/engine`、`/api/validation`、`/api/pareto`、`/api/lattice/info`、`/api/blade_analysis`、`/api/tbc_analysis` 均 200。

页面里的"工程仪表盘 → V2.1 交互设计"可直接用：

- 改 19 维参数 → **校验并求解**（`POST /api/geometry19/solve`，实测 0.2 s，返回性能/约束/合法性）
- **生成并显示三维模型**（`POST /api/geometry19/regenerate`，实测 0.3 s，产出 12 片 preview STL 并在 Three.js 中显示）
- 切 V2.2/V3 模式需要权重与 Unix 专用模块（见 §5）

包内已带一份可直接查看的 19D 交付设计：

```text
design_id = 742712091b7bba185f0ee0b85d257fef309d573956b6828c6d650ced3c5b7061
路径 = exports/geometry19/<design_id>/{geometry19.json, derived_geometry.json, preview_mesh/*.stl}
接口 = GET /api/geometry19/design/<design_id>、GET /api/geometry19/components/<design_id>
```

### 3.2 命令行演示

```powershell
& $py app.py --geometry     # 整机几何摘要：总长 225 mm，最大直径 240 mm，质量 1866 g
& $py app.py --brayton      # 站位热力 + 材料校核：76.4 N，TSFC 37.2 g/(kN·s)，热效率 15.6%
& $py app.py --generate 60  # 数据集生成，60/60 合法，写入 data/generated/
```

### 3.3 22D 可视化网格（不需要任何模型权重）

```powershell
# 用 Flask 测试客户端或直接 POST 到运行中的服务
# 载荷：{"geometry22": {全部 22 个变量}}
# 22 个变量名与参考值取自 GET /api/geometry22/schema 的 design_variables[*].name/reference_value
```

实测 `POST /api/geometry22/mesh` 200，16.7 s 生成 12 片 STL + manifest，缓存到
`exports/geometry22_preview/mesh22/<mesh_id>/`，按 `mesh_id` 读取。这条路径只依赖几何与网格代码，
是当前最"完整"的一条可视化链路。

### 3.4 测试

```powershell
& $py -m pytest tests/test_brayton.py tests/test_geometry.py tests/test_geometry19_physics.py `
                 tests/test_geometry19_acceptance_policy.py -q     # 37 passed
& $py -m pytest tests/test_geometry19_mesh.py -q                    # 6 passed（约 90 s）
& $py -m pytest tests/test_geometry19_ui.py tests/test_geometry22_mesh_ui.py -q   # 6 passed
```

## 4. 关键架构（读代码前先看这三层）

```text
物理权威层   src/physics/        Geometry19 V2.1 低阶闭式模型、约束、质量模型
                ↑ 唯一裁决者（不会被代理模型取代）
搜索/生成层  src/ai/             CFM-4 生成器、D-Flow 引导、performance3_log 代理、
                                19D 梯度增强、warm-start NSGA-II、编排与账本
交付/展示层  src/visualization/  设计存储 Geometry19DesignStore、STL 网格、22D 网格、
                + ui/            作业管理、Flask 路由、Three.js 查看器
```

三条铁律（文档反复强调，改代码时必须遵守）：

1. `config/physics/wp19_23_inverse_closure_v1.yaml` 是模型/容差/预算的唯一机器可读权威；
2. `scripts/run_wp19_23_pipeline.py` 是唯一公开入口，阶段脚本只用于调试/审计/复现；
3. 代理模型只能"提候选、排序"，**不得**接受候选或宣布命中——一切由 V2.1 复算裁决。

命中度量：`rho = max(相对误差 / 各轴容差)`，`rho<=1` 等价于三轴全部命中；开发容差为
推力 2.7451%、TSFC 0.8573%、质量 1.1707%。

## 5. 已实测不可用的路径与精确原因

| 现象 | 精确原因 | 缺失对象 |
|---|---|---|
| `run_wp19_23_pipeline.py`（V2.1，默认）失败 | `Geometry19PipelineError: Generator checkpoint is missing` | `data/generator_models/geometry19/wp19_23_cfm4_bounded_sigmoid_seed43_v1/baseline/{best.pt,normalizer.json}` |
| 同上（即使补了生成器）还会缺 | 前向代理 `performance3_log` 权重 | `data/inverse_design/wp19_22_performance_surrogate_v2_ablation/formal/models/performance3_log/seed_43` |
| `--model "model V2.2"` | `ModuleNotFoundError: No module named 'fcntl'` | Windows 无 POSIX `fcntl`；`src/ai/geometry19_pipeline_v2_2.py` 顶层 import |
| `--model "model V3"` / 网页 V3 作业 | 同上 | `src/ai/geometry22_pipeline_v3_{routed,guided,cfm}.py`、`src/ai/geometry22_jobs.py` |
| `POST /api/geometry22/evaluate` 500 | `FileNotFoundError: docs/physics/V3_R0_22D_DEVELOPMENT_CONTRACT_20260923.yaml` | 交付时被排除的 1 份过程文档（代码硬编码引用） |
| `GET /api/geometry19/pipeline-result` 404 | 未配置已完成运行目录（需 `NOVATURBO_GEOMETRY19_PIPELINE_RESULT_ROOT` 指向含 `Complete_Manifest.json` 的 V3 运行根） | `data/validation/v3_22d_final_holdout_v1` 等 |
| `GET /api/geometry22/results` → `{"results":[]}` | 结果目录不存在 | 同上 |
| `GET /api/components` → `[]` | `exports/stl/` 不存在；包内 STL 在 `exports/` 根目录 | 需先调导出接口重建 `exports/stl/` |

统计口径：源码/脚本中引用的 250 个 `docs|data|config` 路径里，**101 个缺失**（100 个数据/模型资产 + 1 份文档）。
其中 32 个测试文件依赖 torch，其余多数依赖上述缺失数据集。因此"开箱跑通完整逆设计"不成立——
这与交付说明 `DELIVERY.md` 的声明一致（数据被刻意排除）。

## 6. 恢复到"完整可用"需要什么

1. **模型权重与数据集**（向交付方索取，或从原分支重训）：
   - `data/generator_models/geometry19/…`（CFM-4 bounded-sigmoid / baseline 的 `best.pt` + `normalizer.json`）
   - `data/inverse_design/wp19_22_performance_surrogate_v2_ablation/…`（`performance3_log` 三输出代理）
   - `data/inverse_design/wp19_22_gate_k_hit_aligned_surrogate_v1/…`（可选 `s0` 分支）
   - `data/validation/…`（V2.2/V3 目标集与结果，用于网页只读结果）
2. **1 份缺失文档**：`docs/physics/V3_R0_22D_DEVELOPMENT_CONTRACT_20260923.yaml`（22D 直接评价契约）。
3. **Windows 兼容改造**（若要跑 V2.2/V3）：把 17 个模块的 `import fcntl` 改成跨平台文件锁
   （如 `msvcrt.locking` 或 `portalocker`），并把 `CUDA_VISIBLE_DEVICES='6'` 参数化。
4. 备注：pytest 默认临时目录 `C:\Users\QY\AppData\Local\Temp\pytest-of-QY` 在本机被拒绝访问，
   设 `TMP`/`TEMP` 到工作区内即可（§3 已给出）。

## 7. 使用边界（必须原样保留）

- `physics_validation_status = not_completed`，`formal_use_allowed = false`，`deployment_allowed = false`；
- 物理是项目内低阶闭式模型，**不等于** CFD / 台架试验 / 制造验证；
- 当前命中只能称 `development_diagnostic_hit`，不能对外发布命中率；
- 网格是可视化表达，不是工程网格，不能作为制造依据；
- 默认模型是 V2.1；V3 22D 为开发分支能力，实现按用户决定暂停。

## 8. 文档索引（按阅读顺序）

1. `README.md` — 项目总览、特性表、Quick Start
2. `DELIVERY.md` — 交付范围、排除项、边界
3. `docs/physics/GEOMETRY19_CURRENT_PIPELINE.md` — 当前流水线（谁做什么、为什么这么分工）
4. `docs/physics/WP19_23_PIPELINE_ENTRYPOINT.md` — 入口约束与实现级语义
5. `docs/physics/NovaTurbo_GeometryAware_19D_Closed_LowOrder_Engine_Model_V2_1.md` — 物理模型规格
6. `docs/physics/GEOMETRY19_UI_CLI_CAPABILITY_MATRIX_20260921.md` — UI/CLI 功能对应表
7. `docs/physics/V3_22D_VISUAL_MESH_20260928.md` — 22D 网格映射与接口
8. `docs/WP0…WP6*.md` — 各阶段（数据、筛选、回代、warm-start、编排）运行说明
