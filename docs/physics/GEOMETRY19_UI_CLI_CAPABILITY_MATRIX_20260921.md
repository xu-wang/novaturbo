# Geometry19 UI / CLI 功能对应表

日期：2026-09-27；补齐 V3 guided 能力。V3 实现暂停于 `ea1cfc5`，当前整理文档；统一状态见 [V3 当前状态](V3_CURRENT_STATUS.md)。

## V2.1 / V2.2 能力

| 功能 | 统一 CLI | Web UI | 说明 |
|---|---|---|---|
| 启动 V2.1 逆设计 | 支持，且为默认 | 不支持 | 使用 `run_wp19_23_pipeline.py` |
| 启动 V2.2 逆设计 | 支持，显式 `--model "model V2.2"` | 支持输入新三性能目标并异步启动 | 每个 UI 任务使用独立目录和 GPU 6 |
| 查看已完成 V2.2 结果 | 读取 summary/manifest | 新任务完成后自动加载；历史只读 API 保留 | UI 先复核运行目录和外部网格哈希 |
| 查看 preview/export STL | `--export-mesh` 生成 | 支持切换并加载 | 两级资产均来自已完成 pipeline |
| 查看源哈希与几何 ID | summary 中记录 | 支持 | 两个哈希域显式显示，不混为一体 |
| 修改19D并求解/重建 | 不属于交付结果读取 | 支持现有 V2.1 workspace | 会生成新的交互设计，不改写 pipeline 产物 |
| 旧 `/api/inverse_design` | 不适用 | 仍为随机/代理快速接口 | 不是 Geometry19 三目标统一 pipeline |
| 部分运行恢复 | V2.2 不支持 | 不支持 | 保留现场并停止 |

## V3 能力

| 功能 | 统一 CLI | Web UI | 说明 |
|---|---|---|---|
| 新目标逆设计 | 显式 V3 + `v3_guided_cfm_development` | 输入新目标、点击 Run V3 target 后异步提交 | 切入 V3 清空指标；切换本身不运行 |
| 选择历史 profile | fixture/routed/CFM/guided 分发 | 新作业固定 guided，历史按原身份读取 | CLI 省略 profile 是 fixture，默认模型仍 V2.1 |
| 完整 22D/F 结果 | summary/manifest | 支持完整核验后查看 | 软件命中、诊断有效、可行性、交付分开 |
| 3D | 当前 V3 不接受工程导出选项 | 12 STL 的 19D 投影预览 | 三新增物理变量不改变 STL，不是完整 22D 工程网格 |
| 部分运行恢复 | 原身份下复用已封存阶段 | 不提供恢复按钮 | 未封存阶段可能重算；源码变化拒绝旧身份 |
| 已完成历史读取 | 只读核验可无 CUDA | Saved V3 job / View job 或只读结果模式 | 完整性通过不代表当前代码匹配或可恢复 |
| 取消与异常子进程管理 | 无统一取消功能 | 未提供取消 | 异常退出需检查原子进程 |
| 旧仿真按钮 | 不适用 | V3 下关闭 | 不将旧物理仿真混为 V3 结果 |

V3 新作业使用物理 GPU 6 / cuda:0。完整流程见 [网页作业说明](V3_UI_TARGET_JOBS_20260926.md)。下列旧结果接口、preview/export 质量等级和外部网格核验说明针对 V2.2；V3 使用独立 geometry22 接口和投影缓存。

## 网页模式与 V2.2 使用

启动 UI：

```bash
python app.py --ui
```

进入 Geometry19 Design Workspace，选择 **V2.2 new design**。切换时清空旧模型与指标；输入新的推力、TSFC 和质量目标后点击 **Generate V2.2 design**。后端创建独立目录并在后台调用现有 V2.2 CLI。页面查询任务状态，完成后只加载本次结果；失败时显示错误并保持结果区空白。切换模式会阻止旧响应覆盖当前页面。任务状态写入各自的 `job.json`；服务重启后可查询已完成结果，未完成且未通过完成清单复核的任务会报告 `interrupted`，可提交新目标重新运行。

UI 顶部的 Physics model 选择器区分四种模式：

- **V2.1 interactive**：允许 Validate + solve 和 Generate + show in 3D；
- **V2.2 new design**：禁用 V2.1 交互求解，输入新目标并启动独立后台任务。
- **V3 · inverse design**：清空旧指标，等待新输入和显式提交，后台使用 guided profile。
- **V3 results · read only**：显式查看历史封存结果，不启动计算。

仅切换模式不会启动 GPU；点击生成后才启动新任务。

后端将依次校验：

1. `Complete_Manifest.json` 中的运行目录文件；
2. 对应质量等级的外部 mesh manifest；
3. manifest 所列每个 STL 的 SHA-256；
4. F 阶段账本和设计存储元数据。

校验通过后，接口返回本次任务的19D设计、F阶段性能/约束、源设计哈希、几何 `design_id` 及组件 URL。页面把19D值载入控件并直接替换 Three.js 当前模型。切换模式不会改写历史产物或改变 `formal_use_allowed`。

接口：

- `GET /api/geometry19/pipeline-result?quality=preview_mesh|export_mesh`
- `GET /geometry19-pipeline-files/<design_id>/<quality>/<filename>`
- `POST /api/geometry19/v2-2/jobs`
- `GET /api/geometry19/v2-2/jobs/<job_id>`
- `GET /api/geometry19/v2-2/jobs/<job_id>/result?quality=preview_mesh|export_mesh`

未配置结果目录返回404；运行或网格完整性失败返回409。

2026-09-22 实测：通过 UI 任务接口提交新目标，任务从 `queued` 到 `running` 再到 `complete`，结果接口返回 200；重启 Viewer 服务后，同一任务状态仍为 `complete`，结果 JSON 的 SHA-256 保持一致，组件 STL 仍可下载。Firefox 浏览器点击流程也已完成 7/7 项验收，包含模式切换清空、生成后显示、旧响应隔离与刷新后不读取历史结果；见 [V22_UI_BROWSER_ACCEPTANCE_20260922.md](V22_UI_BROWSER_ACCEPTANCE_20260922.md)。
