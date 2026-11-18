# NovaTurbo V2.1 独立能力包（Geometry19 / 19 维）

把 NovaTurbo 的 **V2.1 逆设计能力**从原工程中抽出来，供另一个项目直接调用：

| 能力 | 说明 |
|---|---|
| `predict(design)` | 前向性能代理（`performance3_log`）：19 维设计 → 推力 / TSFC / 质量 |
| `evaluate(design)` | V2.1 闭式低阶物理（`geometry19_closed_v2_1`）：**唯一权威裁决** |
| `generate(target, n)` | CFM-4 bounded-sigmoid 条件生成器：三性能目标 → 19 维设计（`guided=True` 时启用 D-Flow 源引导）|
| `inverse_design(target)` | 生成（默认 D-Flow 引导）→ 代理排序 → V2.1 复核，首个严格命中即停 |

## 安装

```bash
pip install -r requirements.txt
```

依赖：`numpy`、`scipy`、`torch`、`PyYAML`、`torchcfm`、`torchdiffeq`。
若打包时启用了 `--mesh`，导出 STL 还需 `trimesh`、`scikit-image`。

## 目录

```text
v21_capability/
├── novaturbo_v21/
│   ├── api.py                  # 公开 API（本包唯一需要读的源码）
│   ├── __main__.py             # 命令行入口
│   └── _vendor/
│       ├── src/                # 从 NovaTurbo 抽取的依赖闭包（命名空间已改写）
│       ├── config/physics/     # 冻结的 schema / 标定 / 约束 / 协议
│       └── data/               # 两份权重（生成器 + 性能代理）
├── tests/test_bundle.py
├── example_usage.py
├── requirements.txt
└── bundle_manifest.json
```

把整个 `v21_capability/` 目录拷进目标项目即可，无需原仓库、无需联网。

## Python 用法

```python
from novaturbo_v21 import V21Capability

cap = V21Capability(device="cpu")          # 有 GPU 时用 "cuda:0"
info = cap.describe()                       # 19 个坐标名、边界、整数维、容差、模型标识

target = {"thrust_N": 60.0, "tsfc_kg_N_s": 6.7e-5, "mass_kg": 1.77}

prediction = cap.predict(cap.reference_design)       # {"thrust_N": ..., ...}
truth      = cap.evaluate(cap.reference_design)      # 物理结果 + 约束裁决
designs    = cap.generate(target, n_samples=256, seed=20260817)              # 纯 CFM 采样
guided     = cap.generate(target, n_samples=256, seed=20260817, guided=True) # D-Flow 引导
result     = cap.inverse_design(target, n_samples=256, seed=20260817)
```

`generate(..., guided=True)` 复刻流水线的 D-Flow：在冻结的 CFM-4 里对 latent source 做若干步
Adam（目标函数是代理模型的 hit-aligned 平滑 rho + 源位移先验 + 越界罚），再积分出设计。
默认超参来自协议 `dflow` 段（24 步、lr 0.01、位移上限 0.75、梯度裁剪 1.0）。
`inverse_design` 默认走引导路径，并在**首个严格命中且物理可接受**的候选处提前停止。

`inverse_design` 的返回结构：

```json
{
  "status": "DIAGNOSTIC_HIT | BEST_APPROXIMATION_NOT_HIT | NO_PHYSICS_VALID_CANDIDATE",
  "delivered": {
    "design": {"19 个坐标": "..."},
    "predicted": {"thrust_N": "...", "tsfc_kg_N_s": "...", "mass_kg": "..."},
    "performance": {"...": "V2.1 真值"},
    "hit": {"axis_ratios": {}, "rho": 0.68, "failed_axes": 0, "strict_hit": true},
    "physics_accepted": true
  },
  "ranking": [ ... ]
}
```

## 命令行

```bash
python -m novaturbo_v21 info
python -m novaturbo_v21 predict  --design design.json
python -m novaturbo_v21 evaluate --design design.json --target target.json
python -m novaturbo_v21 generate --target target.json --n 256 --seed 20260817
python -m novaturbo_v21 inverse  --target target.json --n 256 --budget 16
```

所有命令输出 JSON；`--design` / `--target` 接受文件路径或 `-`（标准输入）。

## 契约要点

- 设计坐标必须是 `describe()["feature_names"]` 里的 19 个名字，顺序无关；
- 三个整数维（叶片数、喷注器数）自动四舍五入到整数；
- `rho = max(各轴相对误差 / 各轴容差)`，`rho <= 1` 等价于三轴全部命中；冻结容差：
  推力 2.7451%、TSFC 0.8573%、质量 1.1707%（见 `describe()["relative_tolerances"]`）；
- `inverse_design` 是"能力级"实现：生成（含 D-Flow 引导）→ 代理排序 → V2.1 复核（默认验完所有保留候选，
  命中即停），**没有**原流水线的代理梯度增强（D 阶段）、warm-start NSGA-II（E 阶段）、审计账本和正式交付清单；
  需要那套审计与救援时仍应回到原工程的主流水线；
- 物理模型是项目内低阶闭式模型，`formal_use_allowed=false`：
  它不等于 CFD / 台架试验 / 制造验证，不能作为适航或制造依据。

## 权重来源

| 权重 | 路径 | 说明 |
|---|---|---|
| CFM-4 bounded-sigmoid 生成器 | `_vendor/data/generator_models/geometry19/wp19_23_cfm4_bounded_sigmoid_seed43_v1/baseline/` | `best.pt` + `normalizer.json`，seed 43 |
| `performance3_log` 性能代理 | `_vendor/data/inverse_design/wp19_22_performance_surrogate_v2_ablation/formal/models/performance3_log/seed_43/` | `weights_best_val.pt` + `normalization.npz` + `model_manifest.json` |

`bundle_manifest.json` 记录打包时的源仓库路径、协议文件、模块数量与权重相对路径。

## 重新打包

在原 NovaTurbo 仓库中执行：

```bash
python tools/v21/build_bundle.py --output v21_capability --mesh --force
```

## 许可

随原 NovaTurbo 工程（MIT）。第三方组件（torchcfm / torchdiffeq / nflows / FrEIA 等）
遵循其各自许可，详见原工程 `third_party/NOTICE.md`。
