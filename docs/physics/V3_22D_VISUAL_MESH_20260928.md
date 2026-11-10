# V3 22D 参数驱动可视化网格

日期：2026-09-28。分支：`V3-develop`。本功能为完整 22D 设计生成可核验的三维 STL 预览，接入已封存 V3 结果、V3 作业结果和手动编辑页面；不修改 B–F 物理记录，不调用 V3 求解器，不提升正式用途许可。

## 参数到网格的映射

网格复用 `geometry19_to_assembly_params()` 和现有整机 STL 导出器，输入仍是完整 `Geometry22Design`。前 19 维按原装配规则生成实体与晶格；第 20 维 `diffuser_area_ratio` 覆盖装配配置，扩压器出口半径比为其平方根；第 22 维 `compressor_tip_clearance_ratio` 乘以压气机叶轮出口直径，得到壳体与叶轮之间的间隙毫米值。两者会改变 STL 实体及实体散列。第 22 维的这项映射只用于新 V3 可视化资产，不修改旧 19D STL 或把开发坐标包络认定为实际几何间隙约束。

第 21 维 `combustor_primary_air_fraction` 是主燃区空气质量份额，现有 R0 变量契约没有将它定义为燃烧室孔径、孔数或孔系分布。因此该值保存在完整 22D 设计身份及网格 manifest 的 `flow_only_parameters` 中，不强行改变燃烧室实体。只改变它会产生不同网格产物 ID，但实体散列可相同。这是**22D 设计驱动的可视化网格**，尚不是能完整表达流量分配与制造细节的工程网格。

原 22D schema 和历史选中设计 ID 保持原样；旧 19D 投影资产仍属历史基线，新网格使用独立 `geometry22_visual_mesh_v1` 身份与缓存目录。

## 接口与操作

| 场景 | 操作或接口 |
|---|---|
| 手动设计 | 选 **V3 · edit 22D design**，编辑全部 22 维后点击 **Generate 22D mesh**。可独立于 **Evaluate 22D design** 使用。 |
| 已存结果与作业 | 查看核验结果后点击 **Generate 22D mesh in 3D**。加载的性能仍来自归档 F 复算记录。 |
| 手动生成 API | `POST /api/geometry22/mesh`，JSON 请求体仅含完整 `geometry22` 映射。 |
| 手动 STL | `GET /geometry22-mesh-files/<mesh_id>/<filename>`。 |
| 结果 STL | 原 `geometry22-preview-files` 与 `geometry22-job-preview-files` 路由保留，读取时重新核验源结果和网格。 |

缓存位于 `NOVATURBO_GEOMETRY22_PREVIEW_ROOT/mesh22/<mesh_id>`，默认根目录为 `exports/geometry22_preview`。`mesh_id` 包含完整 22D 设计 ID、`preview_mesh` 质量档及相关源码/配置散列。产物含 12 个 STL、逐部件诊断、manifest 和封存散列清单。读取时核对文件清单、每片 STL 的散列、当前源码/配置身份和完整设计 ID；篡改或代码变化会拒绝旧缓存。用户修改手动表单后，页面清除旧网格，迟到的网格响应不能覆盖新输入。

参考设计网格已存于 `exports/geometry22_preview/mesh22/83a0696f0b582d017d17051011ebc0d797b80ec2e7b1d92f95ec83b98dd22651/`，包含 12 片 STL 与 manifest；它是本地生成缓存，不纳入 Git 源码提交。

预览质量使用现有几何诊断门槛。间隙修正物理关系和参数边界仍未独立标定；本功能不证明整机工程可行、制造可行或 R7 物理验证通过。`formal_use_allowed=false`、`deployment_allowed=false`。

## 验收

`tests/test_geometry22_mesh_ui.py` 覆盖两项新增几何映射、主燃区空气份额不改变实体、完整 22D 身份、文件篡改拦截和错误输入。连同 V3 结果、作业、直接物理评价及 V2 网页回归测试，共 34 项通过。参考设计真实生成 12 个 STL，网格诊断通过。真实网格差分检查显示：改变面积比时 `assembly`、`casing`、`full_engine_assembly`、`transitions` 改变；改变间隙比时这些部件和 `compressor` 改变；只改变空气份额时 STL 均不变，但 22D 网格 ID 改变。

另对已封存七中心结果中的 `003851992c2386bd0b4fd32b89578836a1aed24440402901e1f7e5559fe41821` 实际生成 22D 网格，12 片 STL 的诊断通过；源 F 记录未修改。
