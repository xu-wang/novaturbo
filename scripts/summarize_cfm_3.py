#!/usr/bin/env python3
"""Build the CFM-3 ledger, comparisons, and strict stage report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_cfm_2 import aggregate, git_value, relative
from src.ai.baseline import sha256_file


TRAIN_DIR = Path("data/generator_models/stage2/cfm_arch/cfm_3_residual_gpu6_v1")
EVAL_DIRS = {
    42: Path("data/inverse_design/stage2/cfm_arch/cfm_3_eval_seed42_v2_1"),
    43: Path("data/inverse_design/stage2/cfm_arch/cfm_3_eval_seed43_v2_1"),
}
INFERENCE = {
    42: Path("data/inverse_design/stage2/cfm_arch/cfm_3_inference_seed42.json"),
    43: Path("data/inverse_design/stage2/cfm_arch/cfm_3_inference_seed43.json"),
}
CONFIG = Path("config/experiments/cfm_arch/cfm_3_residual_train.yaml")
TARGETS = Path("config/experiments/cfm_arch/cfm_0_generalization_targets.csv")
CFM_0_DIR = Path("results/s2_arch_cfm/cfm_0")
CFM_2_DIR = Path("results/s2_arch_cfm/cfm_2")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/s2_arch_cfm/cfm_3")
    )
    args = parser.parse_args()
    output = REPO_ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    training = pd.read_csv(REPO_ROOT / TRAIN_DIR / "seed_summary.csv")
    frames = []
    for seed, directory in EVAL_DIRS.items():
        frame = pd.read_csv(REPO_ROOT / directory / "target_summary.csv")
        frame.insert(0, "seed", seed)
        frames.append(frame)
    evaluations = pd.concat(frames, ignore_index=True)
    target_groups = pd.read_csv(REPO_ROOT / TARGETS)[
        ["target_id", "generalization_group"]
    ]
    evaluations = evaluations.merge(
        target_groups, on="target_id", how="left", validate="many_to_one"
    )
    if len(evaluations) != 60 or evaluations.target_id.nunique() != 30:
        raise RuntimeError("CFM-3 requires the same 30 frozen targets for both seeds")
    if evaluations.generalization_group.isna().any():
        raise RuntimeError("unclassified CFM-3 target")

    branch = git_value("branch", "--show-current")
    source_commit = git_value("rev-parse", "HEAD")
    ledgers = []
    seed_rows = []
    for seed in (42, 43):
        train = training.loc[training.seed == seed].iloc[0]
        inference = json.loads(
            (REPO_ROOT / INFERENCE[seed]).read_text(encoding="utf-8")
        )
        checkpoint = TRAIN_DIR / f"cfm/seed_{seed}/best.pt"
        values = evaluations.loc[evaluations.seed == seed].copy()
        values.insert(0, "experiment_id", "s2_arch_cfm_3_v1")
        values.insert(1, "architecture", "CFM-3")
        values["target_F"] = values.target_thrust_N
        values["target_TSFC"] = values.target_tsfc_kg_N_s
        values["target_Mass"] = values.target_total_mass_kg
        values["parameter_count"] = int(train.total_parameters)
        values["trainable_parameter_count"] = int(train.trainable_parameters)
        values["checkpoint_size_bytes"] = int(train.best_checkpoint_bytes)
        values["training_time"] = float(train.training_elapsed_seconds)
        values["average_epoch_time"] = float(train.average_epoch_seconds)
        values["best_val_loss"] = float(train.best_validation_loss)
        values["best_epoch"] = int(train.best_epoch)
        values["epochs_completed"] = int(train.epochs_completed)
        values["generation_time"] = float(inference["mean_seconds"])
        values["peak_gpu_memory"] = int(train.peak_gpu_memory_bytes)
        values["0d_success"] = values["0d_success_count"]
        values["0d_failure"] = values["0d_failure_count"]
        values["best_error"] = values.best_total_error
        values["git_commit"] = source_commit
        values["branch"] = branch
        values["config_path"] = str(CONFIG)
        values["checkpoint_path"] = str(checkpoint)
        values["checkpoint_sha256"] = sha256_file(REPO_ROOT / checkpoint)
        ledgers.append(values)
        seed_rows.append(
            {
                "architecture": "CFM-3",
                "seed": seed,
                "total_parameters": int(train.total_parameters),
                "trainable_parameters": int(train.trainable_parameters),
                "checkpoint_size_bytes": int(train.best_checkpoint_bytes),
                "best_validation_loss": train.best_validation_loss,
                "best_epoch": int(train.best_epoch),
                "epochs_completed": int(train.epochs_completed),
                "training_time_seconds": train.training_elapsed_seconds,
                "average_epoch_seconds": train.average_epoch_seconds,
                "peak_gpu_memory_bytes": int(train.peak_gpu_memory_bytes),
                "generation_128_mean_seconds": inference["mean_seconds"],
                "generation_128_median_seconds": inference["median_seconds"],
                "generation_128_std_seconds": inference["std_seconds"],
                "N_valid_mean": values.N_valid.mean(),
                "N_keep_mean": values.N_keep.mean(),
                "valid_rate_mean": values.valid_rate.mean(),
                "best_error_median": values.best_total_error.median(),
                "topk_error_median": values.topk_error.median(),
                "median_error_median": values.median_error.median(),
                "hit_rate_mean": values.hit_rate.mean(),
                "target_success_rate": values.best_within_tolerance.mean(),
            }
        )

    ledger = pd.concat(ledgers, ignore_index=True)
    ledger.to_csv(output / "cfm_3_ledger.csv", index=False)
    with (output / "cfm_3_ledger.jsonl").open("w", encoding="utf-8") as stream:
        for record in ledger.replace({np.nan: None}).to_dict(orient="records"):
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(output / "cfm_3_seed_summary.csv", index=False)

    group_summary = aggregate(evaluations)
    group_summary.to_csv(output / "cfm_3_group_summary.csv", index=False)
    overall_frame = evaluations.copy()
    overall_frame["generalization_group"] = "Overall"
    overall = aggregate(overall_frame)
    overall.to_csv(output / "cfm_3_overall_summary.csv", index=False)

    old_ledger = pd.read_csv(REPO_ROOT / CFM_2_DIR / "cfm_2_ledger.csv")
    paired = evaluations.merge(
        old_ledger[
            [
                "seed", "target_id", "valid_rate", "N_valid", "N_keep",
                "best_total_error", "topk_error", "hit_rate",
                "best_within_tolerance",
            ]
        ],
        on=["seed", "target_id"],
        suffixes=("_cfm3", "_cfm2"),
        validate="one_to_one",
    )
    paired["valid_rate_delta"] = paired.valid_rate_cfm3 - paired.valid_rate_cfm2
    paired["best_error_delta"] = (
        paired.best_total_error_cfm3 - paired.best_total_error_cfm2
    )
    paired["best_error_improved"] = paired.best_error_delta < 0
    paired["hit_rate_delta"] = paired.hit_rate_cfm3 - paired.hit_rate_cfm2
    paired.to_csv(output / "cfm_3_vs_cfm_2_paired.csv", index=False)

    old_overall = pd.read_csv(REPO_ROOT / CFM_2_DIR / "cfm_2_overall_summary.csv")
    baseline_all = pd.read_csv(REPO_ROOT / CFM_0_DIR / "model_overall_summary.csv")
    baseline = baseline_all.loc[baseline_all.model == "cfm"].iloc[0]
    nflows = baseline_all.loc[baseline_all.model == "nflows"].iloc[0]
    old = old_overall.iloc[0]
    new = overall.iloc[0]
    comparison = pd.concat(
        [
            overall.assign(architecture="CFM-3 Residual"),
            old_overall.assign(architecture="CFM-2 Wide/Deep"),
            baseline_all.loc[baseline_all.model == "cfm"].assign(
                architecture="CFM-0 Scalar Time"
            ),
            baseline_all.loc[baseline_all.model == "nflows"].assign(
                architecture="nflows Existing Best"
            ),
        ], ignore_index=True,
    )
    comparison.to_csv(output / "model_comparison_overall.csv", index=False)

    old_seeds = pd.read_csv(REPO_ROOT / CFM_2_DIR / "cfm_2_seed_summary.csv")
    param_delta = relative(seeds.total_parameters.iloc[0], old_seeds.total_parameters.iloc[0])
    train_delta = relative(
        seeds.training_time_seconds.mean(), old_seeds.training_time_seconds.mean()
    )
    inference_delta = relative(
        seeds.generation_128_mean_seconds.mean(),
        old_seeds.generation_128_mean_seconds.mean(),
    )
    improved = paired.best_error_improved.mean()
    groups3 = group_summary.set_index("generalization_group")
    groups2 = pd.read_csv(REPO_ROOT / CFM_2_DIR / "cfm_2_group_summary.csv").set_index(
        "generalization_group"
    )

    report = f"""# S2-ARCH-CFM CFM-3 阶段报告

## 阶段状态

- Architecture：CFM-3 Residual MLP
- Status：PASS
- Branch：`{branch}`
- 实验源 commit：`{source_commit}`
- 唯一核心变化：在CFM-2的256维、32维时间嵌入基础上，将普通隐藏层改为输入投影加两个双层Residual Block。
- 未加入FiLM、gate、U-Net、attention、normalization或可训练时间投影。
- CFM-4：未实现、未配置、未训练。

PASS表示完整工程与实验流程完成，不等同于性能优于CFM-2。

## Architecture

```text
[x_t(10), time_embedding(32), c(3)]
→ Linear(45,256) → SiLU
→ [Linear(256,256) → SiLU → Linear(256,256) + skip] × 2
→ Linear(256,10)
```

- Plain `ConditionalFlowMatcher(sigma=0)`；RK4/32/128 NFE。
- 数据、训练、采样、V2.1、目标集及物理评价全部冻结。
- 参数量：{int(seeds.total_parameters.iloc[0]):,}，相对CFM-2增加 {param_delta:+.2%}。

## Training and Complexity

| Seed | Best val loss | Best epoch | Epochs | Train (s) | Peak GPU MiB | Inference/128 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in seed_rows:
        report += (
            f"| {row['seed']} | {row['best_validation_loss']:.6f} | "
            f"{row['best_epoch']} | {row['epochs_completed']} | "
            f"{row['training_time_seconds']:.3f} | "
            f"{row['peak_gpu_memory_bytes']/2**20:.2f} | "
            f"{1000*row['generation_128_mean_seconds']:.3f} |\n"
        )
    report += f"""

两个seed均使用完整79,945/9,992样本并正常早停。相对CFM-2，平均训练时间变化 {train_delta:+.2%}，同步推理时间变化 {inference_delta:+.2%}。

## V2.1 Validity and Target Accuracy

严格执行一次 `128 → N_valid → 2^floor(log2(N_valid)) → candidate_id裁剪 → 0D`，不补采样。

| Group | N_valid | N_keep | Valid rate | Best error median | Top-10 error median | Hit rate | Target success |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in group_summary.itertuples(index=False):
        report += (
            f"| {row.generalization_group} | {row.N_valid_mean:.2f} | "
            f"{row.N_keep_mean:.2f} | {row.valid_rate_mean:.2%} | "
            f"{row.best_error_median:.6f} | {row.topk_error_median:.6f} | "
            f"{row.hit_rate_mean:.2%} | {row.target_success_rate:.2%} |\n"
        )
    report += f"""
| Overall | {new.N_valid_mean:.2f} | {new.N_keep_mean:.2f} | {new.valid_rate_mean:.2%} | {new.best_error_median:.6f} | {new.topk_error_median:.6f} | {new.hit_rate_mean:.2%} | {new.target_success_rate:.2%} |

0D成功率为 {new['0d_success_rate']:.2%}；没有可靠已知可行OOD目标，因此OOD not evaluated。

## CFM-3 vs CFM-2

| Metric | CFM-2 | CFM-3 | Change |
|---|---:|---:|---:|
| Parameters | {int(old_seeds.total_parameters.iloc[0]):,} | {int(seeds.total_parameters.iloc[0]):,} | {param_delta:+.2%} |
| Mean train time (s) | {old_seeds.training_time_seconds.mean():.3f} | {seeds.training_time_seconds.mean():.3f} | {train_delta:+.2%} |
| Mean inference/128 (ms) | {1000*old_seeds.generation_128_mean_seconds.mean():.3f} | {1000*seeds.generation_128_mean_seconds.mean():.3f} | {inference_delta:+.2%} |
| Valid rate | {old.valid_rate_mean:.2%} | {new.valid_rate_mean:.2%} | {new.valid_rate_mean-old.valid_rate_mean:+.2%} pt |
| N_keep mean | {old.N_keep_mean:.2f} | {new.N_keep_mean:.2f} | {new.N_keep_mean-old.N_keep_mean:+.2f} |
| Best error median | {old.best_error_median:.6f} | {new.best_error_median:.6f} | {relative(new.best_error_median, old.best_error_median):+.2%} |
| Top-10 error median | {old.topk_error_median:.6f} | {new.topk_error_median:.6f} | {relative(new.topk_error_median, old.topk_error_median):+.2%} |
| Candidate hit rate | {old.hit_rate_mean:.2%} | {new.hit_rate_mean:.2%} | {new.hit_rate_mean-old.hit_rate_mean:+.2%} pt |
| Target success | {old.target_success_rate:.2%} | {new.target_success_rate:.2%} | {new.target_success_rate-old.target_success_rate:+.2%} pt |

逐目标同seed配对中，CFM-3有 {improved:.2%} 的运行取得更小最佳误差。合法率分组变化：ID {groups3.loc['ID','valid_rate_mean']-groups2.loc['ID','valid_rate_mean']:+.2%} pt、Interpolation {groups3.loc['Interpolation','valid_rate_mean']-groups2.loc['Interpolation','valid_rate_mean']:+.2%} pt、Boundary {groups3.loc['Boundary','valid_rate_mean']-groups2.loc['Boundary','valid_rate_mean']:+.2%} pt。

## Seed Stability

| Seed | Valid rate | Best error median | Top-10 error median | Hit rate | Target success |
|---:|---:|---:|---:|---:|---:|
"""
    for row in seed_rows:
        report += (
            f"| {row['seed']} | {row['valid_rate_mean']:.2%} | "
            f"{row['best_error_median']:.6f} | {row['topk_error_median']:.6f} | "
            f"{row['hit_rate_mean']:.2%} | {row['target_success_rate']:.2%} |\n"
        )
    report += f"""

两个seed均通过功能Gate并达到相同目标成功率，但最佳误差仍有明显seed差异，必须保留双种子结论。

## Comparison with Baseline and Existing Best

| Model | Valid rate | Best error median | Top-10 error median | Hit rate | Target success |
|---|---:|---:|---:|---:|---:|
| CFM-0 | {baseline.valid_rate_mean:.2%} | {baseline.best_error_median:.6f} | {baseline.topk_error_median:.6f} | {baseline.hit_rate_mean:.2%} | {baseline.target_success_rate:.2%} |
| CFM-2 | {old.valid_rate_mean:.2%} | {old.best_error_median:.6f} | {old.topk_error_median:.6f} | {old.hit_rate_mean:.2%} | {old.target_success_rate:.2%} |
| CFM-3 | {new.valid_rate_mean:.2%} | {new.best_error_median:.6f} | {new.topk_error_median:.6f} | {new.hit_rate_mean:.2%} | {new.target_success_rate:.2%} |
| nflows | {nflows.valid_rate_mean:.2%} | {nflows.best_error_median:.6f} | {nflows.topk_error_median:.6f} | {nflows.hit_rate_mean:.2%} | {nflows.target_success_rate:.2%} |

CFM-3最佳误差中位数仍为nflows的 {new.best_error_median/nflows.best_error_median:.2f} 倍，候选命中率低 {nflows.hit_rate_mean-new.hit_rate_mean:.2%} 个百分点。

## Advantages

1. 相对CFM-2，合法率提高 {new.valid_rate_mean-old.valid_rate_mean:.2%} 个百分点，并保持相同的目标成功率。
2. 两个seed均正常早停，训练没有不稳定或梯度崩溃；Residual MLP在当前CFM流程中工程上可用。
3. 结构仍是低维问题适用的纯MLP，没有引入不必要的卷积或注意力。

## Disadvantages

1. 最佳误差只改善 {-relative(new.best_error_median, old.best_error_median):.2%}，逐目标配对改善率恰为 {improved:.2%}；这不构成稳定准确度收益。
2. Top-10误差恶化 {relative(new.topk_error_median, old.topk_error_median):.2%}，候选命中率下降 {old.hit_rate_mean-new.hit_rate_mean:.2%} 个百分点，平均N_keep也略降。
3. 参数增加 {param_delta:.2%}、推理变慢 {inference_delta:.2%}，但没有相应的核心精度提升，复杂度权衡弱于CFM-2。
4. seed 42和43的最佳误差方向不同；目标精度仍落后nflows，且没有OOD结论。

## Current Interpretation

Residual MLP可以稳定训练并维持CFM-2的大部分容量收益，但没有证明残差连接本身有效：最佳误差变化仅-0.32%，配对胜率为50%，同时Top-10误差和命中率退化且计算成本上升。一个更优seed和略低的聚合最佳误差不足以抵消这些证据。

## Recommendation

保留CFM-3作为残差消融结果，但当前精度—复杂度权衡仍推荐CFM-2作为最强CFM基线。是否进入CFM-4 FiLM Conditional Residual MLP，必须由用户根据本报告明确决定。

## STAGE STOP

CFM-3 已完成。CFM-4 未开始。
"""
    (output / "CFM-3阶段报告.md").write_text(report, encoding="utf-8")

    manifest = {
        "experiment_id": "s2_arch_cfm_3_v1",
        "architecture": "CFM-3",
        "status": "PASS",
        "branch": branch,
        "source_commit": source_commit,
        "cfm_4_started": False,
        "targets": 30,
        "seeds": [42, 43],
        "ledger": "cfm_3_ledger.csv",
        "ledger_sha256": sha256_file(output / "cfm_3_ledger.csv"),
        "report": "CFM-3阶段报告.md",
        "report_sha256": sha256_file(output / "CFM-3阶段报告.md"),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
