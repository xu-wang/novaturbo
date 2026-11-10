# NovaTurbo 默认逆设计交付策略

> 策略版本：`default_inverse_design_policy_v1`  
> 生效日期：2026-08-14  
> 状态：已于 2026-08-17 被高精度默认策略取代，仅作历史冻结记录  
> 当前默认方案见根目录 `完整高精度逆设计默认策略说明.md`。

本策略正式纠正并覆盖`docs/S2-WP6-G3方法验证正式报告.md`第4节中将“nflows普通目标精度基线”延伸为工程默认选择的表述。原报告受冻结哈希保护，继续作为实验统计记录保留；其G3数值和FAIL判定不变，但其中部署含义以本策略为准。

## 1. 默认方案

NovaTurbo的目标导向逆设计默认采用以下固定链路：

```text
[F*, TSFC*, Mass*]
        ↓
GPU-Sinkhorn CFM-4 seed43
        ↓
一次生成 N_raw=128
        ↓
剔除非有限、越界和重复候选
        ↓
N_keep = 2^floor(log2(N_valid))，不补采样
        ↓
真实Brayton 0D与质量回代
        ↓
命中容差？——是→直接交付并停止
        │
        否
        ↓
active-10D surrogate-gradient refinement
        ↓
仅接受真实0D误差确实改善的候选
        ↓
命中容差？——是→交付并停止
        │
        否
        ↓
CFM候选warm-start NSGA-II
        ↓
交付命中解；若预算耗尽仍未命中，则交付最佳近似解并明确标记未命中
```

规范入口：

- 稳定执行脚本：`scripts/run_inverse_design.py`；
- 默认组合配置：`config/inverse_design/default.yaml`；
- 默认策略：`config/inverse_design/policies/default_delivery_v1.yaml`；
- 默认生成器：`config/inverse_design/generators/cfm4_sinkhorn_seed43_v1.yaml`；
- 目标需求：每个任务一个`config/inverse_design/targets/**/*.yaml`文件，并通过`--target-config`显式传入；
- 默认生成器：GPU-Sinkhorn CFM-4 FiLM Residual，训练seed 43；
- 梯度surrogate：S2-WP3冻结forward surrogate；
- 候选协议：V2.1固定原始候选协议；
- 默认GPU：物理GPU 6，进程内逻辑设备`cuda:0`。

运行示例：

```bash
CUDA_VISIBLE_DEVICES=6 conda run -n inverse_design_nova \
  python scripts/run_inverse_design.py \
  --target-config config/inverse_design/targets/examples/user_request.yaml \
  --output-dir data/inverse_design/runs/example_user_request
```

旧`config/stage2/s2_wp5_cfm_default.yaml`和
`scripts/run_stage2_wp5_cfm.py`保持字节不变，仅用于Stage-II冻结结果复现。稳定入口把
policy、generator和target组合成同一份旧执行schema，再调用未修改的数值核心；因此不会引入
新的采样、筛选、验收或优化行为。

## 2. 阶段停止与成本规则

1. CFM直接阶段已经满足`minimum_hit_count=1`时，立即停止，不运行梯度或NSGA-II。
2. 直接阶段未命中且存在真实0D合法候选时，才运行有预算上限的active-10D梯度增强。
3. surrogate只提供梯度；候选必须经过真实0D复验，真实误差未改善则拒绝更新。
4. 梯度后仍未命中，才运行以当前CFM/梯度候选为warm-start的NSGA-II。
5. `N_valid=0`时不得补采样、复制候选或切换到nflows掩盖失败；运行必须保留为零候选失败。
6. 只有用户明确要求额外工程目标/Pareto集合时，才允许在已经直接命中的情况下继续执行多目标NSGA-II。
7. 网络生成、surrogate前向/反向、真实0D和NSGA-II评价分别记账。

## 3. nflows的角色

nflows保留为：

- 普通ID/Interpolation目标的研究精度基线；
- G3及后续方法研究的冻结对照；
- 消融、复现实验和误差上界诊断工具。

nflows不是：

- 默认交付生成器；
- CFM失败后的自动路由目标；
- 默认链中的隐式补采样方法；
- 用于覆盖或隐藏CFM失败run的回退方法。

## 4. G3结果与默认方案的关系

S2-WP6的G3正式结果为FAIL，其准确含义是：CFM没有同时满足“相对Screened Random和冻结nflows均取得统计稳定优势”的科研声明门槛。正式实验同时确认：

- CFM相对nflows的普通目标物理合法率提高27.12个百分点；
- 固定K多样性比为1.114，没有发生多样性塌缩；
- CFM相对Screened Random明显更准且首次命中更早；
- CFM在冻结困难目标上提供稳定合法起点，并通过梯度/NSGA-II实现3/4次最终命中；
- nflows普通目标直接误差更低，但在极端目标上出现零合法候选。

因此，G3结果限制的是“CFM全面优于nflows”的表述，而不是默认工程链的选择。工程默认更重视：

- 是否稳定提供合法可回代候选；
- 是否为困难目标提供可精修起点；
- 是否通过真实0D验收保持物理可信；
- 是否能够在直接命中时立即停止，并只为失败目标支付增强成本；
- 是否具有单一、可审计、无隐式路由的交付语义。

基于这些要求，CFM增强链保持默认方案。

## 5. 决策层级

当文档表述发生冲突时，按以下层级解释：

1. `config/inverse_design/policies/default_delivery_v1.yaml`定义当前默认交付策略；
2. `config/inverse_design/default.yaml`与generator、target配置共同定义稳定入口的可执行参数；
3. 本文定义工程语义和停止规则；
4. `docs/NovaTurbo_Current_Implementation_Audit.md`记录当前代码、配置、冻结产物与总结文档的实现对照；
5. `config/stage2/s2_wp5_cfm_default.yaml`定义冻结Stage-II复现参数；
6. `docs/S2-WP6-G3方法验证正式报告.md`保留科研统计结论，其中“nflows精度主模型”只表示研究精度基线，不表示默认部署生成器。

## 6. 变更规则

默认方案只有在以下条件同时满足时才能更换：

1. 新方案有明确、版本化的配置与checkpoint；
2. 使用未参与方案选择的冻结目标集完成验证；
3. 零候选、未命中和超预算run全部保留；
4. 合法覆盖、普通目标误差、困难目标救援和端到端成本均有证据；
5. 形成新的默认策略版本，不原地覆盖本策略的历史记录。
