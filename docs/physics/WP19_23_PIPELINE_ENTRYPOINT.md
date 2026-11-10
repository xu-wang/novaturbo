# WP19-23 统一全流程入口

> 当前pipeline的面向使用者说明、流程图、模块职责、配置、产物和能力边界统一见 `GEOMETRY19_CURRENT_PIPELINE.md`；本文保留为开发入口约束与实现级语义说明。

## 强制同步标记

`WP19_23_PIPELINE_SYNC_REQUIRED`

从本文件建立后，任何新增或修改 WP19-23 运行能力，都必须在同一次变更中同步接入：

- `scripts/run_wp19_23_pipeline.py`
- `src/ai/geometry19_pipeline.py`
- `tests/test_geometry19_pipeline.py`

不能只修改独立阶段脚本而让统一入口缺少该能力。

`scripts/run_wp19_23_pipeline.py` 是唯一公开全流程入口；`src/ai/geometry19_pipeline.py` 是其内部实现模块，其他 `run_wp19_23_*.py` 阶段脚本仅用于调试、审计和单阶段复现。

## 当前覆盖范围

更新日期：2026-09-27。统一入口按 `--model` 分发，默认 V2.1；V2.2 为独立局部 B–F，V3 为独立 22D profiles。下面的 direct/D-Flow、整数投影、二次幂保留、640 次 NSGA-II 和网格导出规则属于 V2.1，不适用于 V3。V3 实现暂停，最新使用及文档索引见 [V3 当前状态](V3_CURRENT_STATUS.md)。

统一入口接受已有目标 YAML，或直接接受 `[F*, TSFC*, Mass*]`，随后执行：

```text
WP19-23B 单次生成 direct 或 D-Flow-assisted 最终候选
→ WP19-23C Geometry19 V2.1
→ 命中则直接 WP19-23F
→ 否则 WP19-23D 19D refinement
→ 命中则直接 WP19-23F
→ 否则 WP19-23E warm-start NSGA-II
→ WP19-23F V2.1重算、preview mesh和UI JSON
```

若 WP19-23B 得到 `N_valid=0`，统一入口跳过无候选可算的 C/D，直接进入带独立标签的 `WP19-23E cold_start_lhs`，使用64人口和640次V2.1预算，然后进入F。

支持 `baseline` 与 `bounded-sigmoid` 两个生成器分支，支持 `--resume` 断点续跑。生成器变体的 model tag、checkpoint、normalizer、结构修改标志及默认变体全部由 `config/physics/wp19_23_inverse_closure_v1.yaml::models.generator` 唯一声明；公开入口和候选阶段不再维护硬编码的第二份模型路径。当前协议默认变体为 `bounded-sigmoid`，`baseline` 仅作为显式回归/消融分支。每次运行使用独立根目录和 `pipeline_contract.json`，恢复运行时拒绝改变目标、模型、D-Flow选择、设备或协议哈希。

前向代理也由同一协议的 `models.forward` 唯一声明。当前 `default_tag` 为 `performance3_log`：模型只输出推力、TSFC、质量三个自然对数量，运行时统一反变换到物理量；D-Flow、候选排序和D阶段梯度增强使用同一套 hit-aligned 性能目标，合法性与约束最终由V2.1判断，默认分支不加载也不使用旧可行性代理。历史48输出S0以 `variants.s0` 保留，可通过命令行 `--forward-model-tag s0` 单次选择，也可把配置文件的 `models.forward.default_tag` 改为 `s0` 作为部署级默认；S0分支继续加载其原有约束和可行性软损失。每次运行的 `pipeline_contract.json` 和阶段模型哈希都会记录最终解析的逻辑tag、模型tag、输出变换和职责范围。

`--sampling direct` 与 `--sampling dflow` 是互斥的生产模式；省略参数时读取协议中的 `generation.default_sampling_mode`，当前为 `dflow`。D-Flow 是冻结 CFM-4 内部的 source guidance；两种模式都只输出一组 256 个最终候选。三个整数维无条件采用 `numpy.rint` 最近整数投影（半整数按 nearest-even），不再按距整数 `0.25 count` 的历史容差拒绝候选。通过其余确定性约束的候选先由 forward surrogate 按 predicted `rho=max(e_i/tau_i)` 升序排列，以原始 `candidate_id` 作为稳定 tie-break，再保留不超过 `N_valid` 的最大二次幂 `N_keep`。WP19-23C 对每个保留候选只运行一次 V2.1。direct/D-Flow 真实物理配对选择只保留为离线审计，不进入生产交付池。

`pipeline_contract.json` 现在同时冻结解析后的 generator variant、model tag、checkpoint及其SHA-256、normalizer及其SHA-256。WP19-23B从实际加载的checkpoint读取model tag写入候选账本，统一入口会拒绝“协议选择的variant”和“实际checkpoint model tag”不一致的运行。

WP19-23E 根据可用 warm-start 数量把种群动态调整为 16、32 或 64，并在固定640次物理评价预算内计算可运行代数；不再使用固定40人口。

C/D/E/F 共用 `geometry19_hit_alignment_v1`：三轴相对误差分别除以各自验收容差，定义 `rho=max(r_F,r_TSFC,r_M)`，`rho<=1` 与三轴 AND 命中完全等价。D-Flow 与 D 阶段 surrogate-gradient 使用平滑最大轴比；真实候选接受、E 阶段选择和交付排序依次比较失败轴数量、`rho`、轴比和，再比较工程约束违反。历史 `total_true_error` 仅保留用于兼容报告，不再主导搜索。冻结的 `geometry19_constraint_dominance_v1` 继续提供工程违反量归一化及数值/物理域硬失败罚值。各阶段分别写出 `candidate_disposition_ledger.jsonl`，确保每个评价都有明确交付、转交或淘汰原因。

默认只生成 preview。显式传入 `--export-mesh` 或同义参数 `--export-stl` 时，F为每个交付设计额外生成高质量 `export_mesh/*.stl`、质量诊断、组件哈希和manifest，并把路径及组件信息写回UI JSON。

## V3 22D 显式入口与约束

```bash
rtk env CUDA_VISIBLE_DEVICES=6 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/wangjianyu/miniforge3/envs/inverse_design_nova/bin/python scripts/run_wp19_23_pipeline.py \
  --model 'model V3' --pipeline-profile v3_guided_cfm_development \
  --target /absolute/path/target.json --output-root /absolute/path/new_run --device cuda:0
```

该命令为恢复开发后的使用参考，不在文档更新时执行。网页新 V3 作业显式使用相同 profile。省略 V3 profile 仍是 fixture；`v3_routed_development` 为 Sobol，`v3_local_cfm_development` 为无引导 CFM，两者保留原协议。

guided 使用 55 类 gate、局部条件 CFM/D-Flow 和中心 Ridge；B 256 槽→最多 24 唯一设计，C 固定 8；C 无诊断域有效且三性能各 ±5% 命中时，D 最多 8 种子、24 步连续优化、8 次评价；C/D 均未命中时 E 评价 B 第 9–16 个，没有 NSGA-II。三个整数固定，局部连续半径 0.08。F 依整机可行→诊断域有效→其他候选优先级及误差/ID 选取，新 runtime 复算；无 rho 则空终态。

V3 不接受 V2.1 专属模型变体、前向模型选择、采样覆盖或工程网格导出；物理 GPU 6 对应 cuda:0，无 CPU 推理回退。阶段封存与身份锁支持严格 `--resume`，源码/配置/目标变化须新身份；未封存阶段可能重算并产生重复成本。完成结果只读核验使用 `scripts/verify_geometry19_pipeline_artifact.py <run_directory>`；不提升旧身份恢复许可。

V3 运行根含 `Protocol_Lock.json`、`target.json`、B–F 阶段文件、`Run_State.json`、`Pipeline_Summary.json`、`Complete_Manifest.json`，不使用下面 V2.1 的 manifest 名称或输出层级。网页展示完整 22D，网格仅为独立 19D 投影。当前实现/验证情况、默认输出路径和资产见 [V3 当前状态](V3_CURRENT_STATUS.md)及 [guided 配置](../../config/experiments/geometry22_pipeline_v3_guided_development_v1.yaml)。

## V2.1 调用示例

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
python scripts/run_wp19_23_pipeline.py \
  --thrust-N 16.240269568822608 \
  --tsfc-kg-N-s 0.0001583899410074742 \
  --mass-kg 2.010229994741361 \
  --model-variant bounded-sigmoid \
  --forward-model-tag performance3_log \
  --sampling dflow \
  --export-mesh \
  --device cuda:0
```

默认运行目录为：

```text
data/inverse_design/wp19_23_pipeline_v1/<target_id>/<model_variant>/<forward_model_tag>/
```

省略 `--forward-model-tag` 时使用配置中的 `performance3_log`。复现旧S0路径时使用：

```bash
python scripts/run_wp19_23_pipeline.py \
  --target <target.yaml> \
  --forward-model-tag s0 \
  --device cuda:0
```

运行成功后，根目录额外生成 `pipeline_manifest.json`。该聚合 manifest 按实际执行分支列出 B–F 阶段顺序、各阶段 manifest/summary 哈希，以及所有阶段必需文件的相对路径、SHA-256 和字节数；同时冻结 pipeline contract、target、protocol、constraint-dominance 和 Git commit 来源。

## 端到端验证

2026-08-28 已使用有界 sigmoid、D-Flow source guidance 和物理 GPU 6 完成新流程真实验证：`B-dflow → C → D → E → F`。B 为 `128→125→64`；C 恰好执行64次唯一 V2.1，38个候选物理有效且无命中；D 使用8个种子完成24次 V2.1；E 从40个可用 warm-start 动态选择64人口、9代预算上限，实际第8代停止，共576次逻辑评价并得到3个开发诊断命中；F 完成3/3重算、preview、UI JSON和design ID检查，`--resume` 通过。旧 `paired` 生产目录不能用于新入口恢复。

同日已对上述3个真实交付设计执行高质量导出验证：每个设计生成12个 `export_mesh` STL，3/3质量诊断、manifest和UI回写通过。cold-start分支已通过640次模拟V2.1预算的完整E阶段测试，并由pipeline测试确认调用路径严格为 `B→E-cold-start→F`。
