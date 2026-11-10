# 交付说明 · NovaTurbo V3 22D 可视化网格代码快照

本包是 **NovaTurbo 逆向设计系统**在 `V3-develop` 分支最新功能性提交处的**纯代码交付**，供他人接手、阅读与继续开发使用。

---

## 1. 交付概要

| 项 | 值 |
|---|---|
| 来源提交 | `7c88f7764f51029a1070a0915fe1ddd94651e638` |
| 提交名称 | `feat(v3): 22D 参数驱动的可视化网格与三维查看器中文化` |
| 来源分支 | `V3-develop`（提交日期 2026-10-12） |
| 内容 | 代码 + 说明文档，共约 1000 个文件 / 7.5 MB |
| 不含 | 运行环境、数据集、模型权重、实验过程文档、`.git` 历史 |
| 完整性清单 | `MANIFEST.txt`（文件清单）、`SHA256SUMS`（校验和） |

> **重要**：本包从 `V3-develop` 导出，**不是**仓库默认分支 `main`。`main` 停在 2026-08-17 的 `07b74f76`，不含本次 22D 功能。请勿用 `main` 的代码替换本包内容。

> `README.md` 中 Quick Start 里的 `git clone https://github.com/bxf1001g/novaturbo.git` 是项目上游地址，与本交付包无关。**解压本包即可，无需 clone。**

---

## 2. 目录结构

```
turbo_v3_22d_7c88f776/
├── DELIVERY.md              ← 本文件（交付说明）
├── MANIFEST.txt             ← 文件清单（路径 / 字节数 / SHA256）
├── SHA256SUMS               ← 标准 sha256sum 格式校验和
├── README.md                ← 项目总览与 Quick Start（原仓库文件）
├── LICENSE / CONTRIBUTING.md / AGENTS.md
├── requirements.txt         ← Python 依赖声明（仅文本，非环境本身）
├── .gitignore
├── app.py                   ← 主入口
├── repo_thumbnail.png       ← README 引用的插图
├── src/                     ← 核心代码（237 文件）
│   ├── ai/  physics/  geometry/  visualization/  export/  cfd/
├── ui/                      ← Web 查看器（10 文件）
│   ├── server.py
│   ├── templates/viewer.html
│   └── static/{js,css}/
├── config/                  ← 配置与参数文件（335 文件）
├── scripts/                 ← 工具脚本（178 文件）
├── tests/                   ← 测试（209 文件）
├── third_party/             ← 第三方许可声明（6 文件）
└── docs/                    ← 精选说明文档（17 份，见下节）
```

---

## 3. 包含内容

| 类别 | 内容 | 数量 |
|---|---|---|
| 核心代码 | `src/` | 237 |
| Web 界面 | `ui/server.py`、模板、JS/CSS | 10 |
| 配置 | `config/`（YAML/JSON 参数、实验配置） | 335 |
| 工具脚本 | `scripts/` | 178 |
| 测试 | `tests/` | 209 |
| 第三方声明 | `third_party/NOTICE.md` 及各许可证文本 | 6 |
| 根目录文件 | `app.py`、`README.md`、`LICENSE`、`CONTRIBUTING.md`、`AGENTS.md`、`requirements.txt`、`.gitignore`、`repo_thumbnail.png` | 8 |
| 精选说明文档 | `docs/` 白名单 | 17 |
| 交付文档 | `DELIVERY.md`、`MANIFEST.txt`、`SHA256SUMS` | 3 |

**`docs/` 保留的 17 份说明文档**（均为规格、接口、契约或运行说明，非实验记录）：

- `docs/NovaTurbo默认逆设计交付策略.md`
- `docs/WP0基线运行说明.md` ～ `docs/WP6完整逆设计编排层说明.md`（7 份阶段运行说明）
- `docs/physics/NovaTurbo_GeometryAware_19D_Closed_LowOrder_Engine_Model_V2_1.md`（物理模型规格）
- `docs/physics/GEOMETRY19_CURRENT_PIPELINE.md`（流程说明）
- `docs/physics/GEOMETRY19_UI_CLI_CAPABILITY_MATRIX_20260921.md`（能力矩阵）
- `docs/physics/V22_MESH_EXPORT_CONTRACT_20260921.md`（网格导出接口契约）
- `docs/physics/geometry19_acceptance_policy_v1.md`、`geometry19_bounds_rationale_v1.md`、`geometry19_calibration_bootstrap_rationale_v1.md`（策略与依据）
- `docs/physics/WP19_23_PIPELINE_ENTRYPOINT.md`（流水线入口）
- `docs/physics/V3_22D_VISUAL_MESH_20260928.md`（本次 22D 可视化网格功能说明）

---

## 4. 明确排除的内容

| 排除项 | 规模 | 原因 |
|---|---|---|
| `data/` | 21,028 文件 / 3.68 GB | 数据集与生成物（用户要求排除数据） |
| `results/`、`reports/` | 134 文件 / 9.4 MB | 实验产物与过程报告 |
| `docs/` 其余文件 | 135 份 / 3.95 MB | 实验报告、审计、状态记录、阶段计划等过程文档 |
| 根目录 5 份中文文档 | 0.35 MB | 实施路径、执行计划、对话总结、前置分析、默认策略说明 |
| `install-gpu-torch.log` 及 `*.log`/`*.pt`/`*.pyc` | — | 环境日志、模型权重、字节码 |
| `node_modules/`、`venv/`、`.env/`、`__pycache__/`、`.git/` | — | 环境与版本历史 |

如需被排除的某份文档或数据，请向交付方单独索取；本包刻意保持精简。

---

## 5. 收件人环境准备

本包**不含运行环境**，需自行准备：

1. **Python 3.10+**
   ```bash
   cd turbo_v3_22d_7c88f776
   pip install -r requirements.txt
   ```
   依赖含 `torch`、`flask`、`cadquery`、`trimesh`、`pyvista`、`nflows`、`torchcfm` 等，见 `requirements.txt`。

2. **前端依赖（Web 查看器需要）**
   ```bash
   cd ui/static/js && npm install
   ```
   `three` (^0.152.0) 与 `chart.js` (^4.4.1)。`node_modules/` 未随包提供。

3. **数据（关键）**
   代码运行依赖 `data/` 下的设计数据集、训练模型与结果存储；本包**不含这些数据**，
   `data/` 目录需自行创建。按 `README.md` 与 `.gitignore` 注释，部分数据可由
   `app.py` 重新生成，例如：
   ```bash
   python app.py --generate 10000
   python app.py --train data/generated/dataset_10000.csv
   ```
   22D 结果查看接口还会读取 `data/` 下既有的结果目录，若缺失，相关接口只能返回空/错误。
   **因此本包开箱不能直接跑通完整逆设计流程**，但代码、配置、UI 与测试完整可读、可静态检查。

---

## 6. 启动与验证

```bash
# 启动 3D 查看器
python app.py --ui             # 浏览器打开 http://localhost:5000
python app.py --ui --port 8080 # 自定义端口

# 其它入口
python app.py --geometry       # 几何摘要
python app.py --brayton        # 布雷顿循环分析
python app.py --demo           # 完整流水线演示

# 验证本次功能相关测试（3 个文件，提交时记录 12 passed）
python -m pytest tests/test_geometry22_mesh_ui.py \
                  tests/test_geometry22_results_ui.py \
                  tests/test_geometry19_ui.py -q

# 浏览器端验收脚本
python scripts/validate_geometry19_v22_browser.py
```

> 上述 pytest / 浏览器脚本可能依赖 `data/` 中的结果存储，缺少数据时需按第 5 节补齐。

---

## 7. 本次功能：22D 参数驱动的可视化网格与查看器中文化

这是 `V3-develop` 分支在该阶段的收尾改动，把三维预览从 **19D 几何投影**换成 **22D 设计驱动的可视化网格**，并统一查看器界面语言。

**核心改动**

- 新增 `src/visualization/geometry22_mesh.py`：`Geometry22MeshStore` 按 design 发布可追溯网格，以 `mesh_id` + `complete_manifest.json` 封存身份、哈希与完整性；`mesh_scope` 由 `19d_geometry_projection_only` 变为 `22d_design_driven_visual_mesh`。
- 新 STL 实体由**第 20 维（扩压器面积比）**与**第 22 维（叶尖间隙）**驱动；第 21 维（一次空气份额）无冻结孔系映射，仅作流量元数据，不改变燃烧室实体。
- `src/visualization/geometry22_results.py` 改为消费新 store，移除 19D 投影与手写临时目录逻辑。

**新增服务端接口**（`ui/server.py`）

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/geometry22/mesh` | 为单个完整 22D 设计生成网格；入参只接受 `geometry22` 单键，越权/篡改分别返回 400/409 |
| `GET` | `/geometry22-mesh-files/<mesh_id>/<path:filename>` | 按 mesh_id 取回网格文件 |

**界面**

- `ui/templates/viewer.html`、`ui/static/css/viewer.css`：中文顶栏、"查看流程图与步骤说明"弹层（V3 反向设计流程 B–F）、"导出 STL"按钮。
- 新增 `ui/static/js/zh-cn.js`：**仅做展示层翻译**，API 键、枚举值、工程单位保持原文不变。
- `viewer.js`（导出当前可见组件为 STL）、`dashboard.js`、`geometry22-results.js`（22D 直接评价、网格按钮、revision/token 迟到响应隔离）同步改造。

**该提交涉及的 20 个文件**

```
docs/physics/GEOMETRY19_UI_CLI_CAPABILITY_MATRIX_20260921.md
docs/physics/V22_UI_BROWSER_ACCEPTANCE_20260922.md
docs/physics/V3_22D_DIRECT_UI_EVALUATION_20260928.md
docs/physics/V3_22D_VISUAL_MESH_20260928.md          (已随包保留)
docs/physics/V3_CURRENT_STATUS.md
docs/physics/V3_READONLY_VISUALIZATION_20260926.md
docs/physics/V3_UI_TARGET_JOBS_20260926.md
scripts/validate_geometry19_v22_browser.py
src/visualization/geometry22_mesh.py
src/visualization/geometry22_results.py
tests/test_geometry19_ui.py
tests/test_geometry22_mesh_ui.py
tests/test_geometry22_results_ui.py
ui/server.py
ui/static/css/viewer.css
ui/static/js/dashboard.js
ui/static/js/geometry22-results.js
ui/static/js/viewer.js
ui/static/js/zh-cn.js
ui/templates/viewer.html
```

> 其中 6 份 `docs/physics/*` 属实验/状态/验收记录，按"不打包过程文档"的要求已排除；代码与测试文件均已包含。

---

## 8. 边界与免责（务必阅读）

该提交自身明确声明，本包沿用这些边界：

- 物理关系仍为**开发模型**：`physics_validation_status=not_completed`、`formal_use_allowed=false`、`deployment_allowed=false`。
- 网格是**可视化表达**，不是完整 22D 工程网格，也不构成独立物理验证或工程交付依据。
- 系统**默认模型仍为 V2.1**，V3 22D 属于开发分支能力，不是正式发布版本。
- 本包不含数据与运行环境，不保证开箱即可完成端到端逆设计。

---

## 9. 完整性校验

```bash
# 校验所有文件哈希（SHA256SUMS 不含自身）
sha256sum -c SHA256SUMS

# MANIFEST.txt 为 TSV：路径 <TAB> 字节数 <TAB> SHA256
# 列出包内全部交付文件（不含 MANIFEST.txt 与 SHA256SUMS 自身）
```

若校验失败，说明传输过程中文件损坏或不完整，请重新获取本包。

---

## 10. 许可与第三方

- 项目许可见 `LICENSE`；贡献规范见 `CONTRIBUTING.md`。
- 第三方依赖的许可声明见 `third_party/NOTICE.md` 与 `third_party/licenses/`
  （FrEIA、nflows、TorchCFM、torchdiffeq、PyTorch-VAE）。
- Python 依赖清单见 `requirements.txt`，前端依赖见 `ui/static/js/package.json`。
