# V2.2 网格导出接入契约

日期：2026-09-21。适用入口：`scripts/run_wp19_23_pipeline.py --model "model V2.2"`。

本切片只补齐已交付 Geometry19 设计的几何资产发布，不改变 B–F 搜索、V2.2 物理判定、命中口径、模型选择或默认 V2.1 路由。

## 接口与行为

- 未指定 `--export-mesh` 时不生成网格，保持既有 V2.2 默认成本与产物。
- 指定 `--export-mesh` 时，仅在 F 独立复算通过后，为同一份完整 19D 映射生成 `preview_mesh` 和 `export_mesh` 两级 STL。
- `--storage-root` 可显式指定设计存储；省略时使用当前协议继承的 `artifact_protocol.mesh_storage_root`。
- 存储按 `<storage_root>/<design_id>/<quality>/` 隔离。已有完整的同哈希资产可复用；不静默覆盖不同内容。
- 网格发布只执行 Geometry19 到装配参数的确定性映射、网格生成和诊断，不再次运行物理求解；F 仍是唯一新增的交付复算调用。
- 任一级网格诊断失败时，pipeline 失败并保留现场，不把仅有物理解标记为完整导出成功。

## 身份与产物

搜索/物理账本的 `source_design_hash` 与几何存储的 `design_id` 属于不同哈希域，不强行等同。产物显式记录二者的映射，并通过完整 19D 值逐项相等来确认资产对应 F 阶段交付设计。

`export_mesh` 和解析后的 `storage_root` 进入 `Protocol_Lock.json` 身份。完成运行只能由相同身份的 `--resume` 核验复用。

成功导出后：

- `Pipeline_Summary.json` 记录 `mesh_export_integrated=true`、设计存储目录和两级网格摘要；
- `pipeline_manifest.json` 记录两级网格 manifest 的路径和 SHA-256；
- `Complete_Manifest.json` 继续覆盖运行目录内全部文件；外部设计存储通过嵌入的 manifest 哈希建立引用，不复制大型 STL 到运行目录。

## 本切片不包含

- STEP/B-Rep、制造级工艺包或打印准备；
- 部分 target 的阶段恢复或阶段重启；
- 把 V2.2 设为默认模型；
- 改变 `formal_use_allowed=false`；
- 将旧 `/api/inverse_design` 改造成统一 pipeline 前端。
