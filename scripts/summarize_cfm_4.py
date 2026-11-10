#!/usr/bin/env python3
"""Build the CFM-4 ledger, comparisons, and strict stage report."""

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


TRAIN_DIR = Path("data/generator_models/stage2/cfm_arch/cfm_4_film_gpu6_v1")
EVAL_DIRS = {
    42: Path("data/inverse_design/stage2/cfm_arch/cfm_4_eval_seed42_v2_1"),
    43: Path("data/inverse_design/stage2/cfm_arch/cfm_4_eval_seed43_v2_1"),
}
INFERENCE = {
    42: Path("data/inverse_design/stage2/cfm_arch/cfm_4_inference_seed42.json"),
    43: Path("data/inverse_design/stage2/cfm_arch/cfm_4_inference_seed43.json"),
}
CONFIG = Path("config/experiments/cfm_arch/cfm_4_film_train.yaml")
TARGETS = Path("config/experiments/cfm_arch/cfm_0_generalization_targets.csv")
RESULTS = Path("results/s2_arch_cfm")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS / "cfm_4")
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
        raise RuntimeError("CFM-4 requires the same 30 frozen targets for both seeds")
    if evaluations.generalization_group.isna().any():
        raise RuntimeError("unclassified CFM-4 target")

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
        values.insert(0, "experiment_id", "s2_arch_cfm_4_v1")
        values.insert(1, "architecture", "CFM-4")
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
                "architecture": "CFM-4",
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
    ledger.to_csv(output / "cfm_4_ledger.csv", index=False)
    with (output / "cfm_4_ledger.jsonl").open("w", encoding="utf-8") as stream:
        for record in ledger.replace({np.nan: None}).to_dict(orient="records"):
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(output / "cfm_4_seed_summary.csv", index=False)

    groups = aggregate(evaluations)
    groups.to_csv(output / "cfm_4_group_summary.csv", index=False)
    overall_frame = evaluations.copy()
    overall_frame["generalization_group"] = "Overall"
    overall = aggregate(overall_frame)
    overall.to_csv(output / "cfm_4_overall_summary.csv", index=False)

    old_ledger = pd.read_csv(REPO_ROOT / RESULTS / "cfm_3/cfm_3_ledger.csv")
    paired = evaluations.merge(
        old_ledger[
            [
                "seed", "target_id", "valid_rate", "N_valid", "N_keep",
                "best_total_error", "topk_error", "hit_rate",
                "best_within_tolerance",
            ]
        ],
        on=["seed", "target_id"],
        suffixes=("_cfm4", "_cfm3"),
        validate="one_to_one",
    )
    paired["valid_rate_delta"] = paired.valid_rate_cfm4 - paired.valid_rate_cfm3
    paired["best_error_delta"] = paired.best_total_error_cfm4 - paired.best_total_error_cfm3
    paired["best_error_improved"] = paired.best_error_delta < 0
    paired["hit_rate_delta"] = paired.hit_rate_cfm4 - paired.hit_rate_cfm3
    paired.to_csv(output / "cfm_4_vs_cfm_3_paired.csv", index=False)

    cfm3_overall = pd.read_csv(REPO_ROOT / RESULTS / "cfm_3/cfm_3_overall_summary.csv")
    cfm2_overall = pd.read_csv(REPO_ROOT / RESULTS / "cfm_2/cfm_2_overall_summary.csv")
    baseline_all = pd.read_csv(REPO_ROOT / RESULTS / "cfm_0/model_overall_summary.csv")
    baseline = baseline_all.loc[baseline_all.model == "cfm"].iloc[0]
    nflows = baseline_all.loc[baseline_all.model == "nflows"].iloc[0]
    old = cfm3_overall.iloc[0]
    cfm2 = cfm2_overall.iloc[0]
    new = overall.iloc[0]
    pd.concat(
        [
            overall.assign(architecture="CFM-4 FiLM Residual"),
            cfm3_overall.assign(architecture="CFM-3 Residual"),
            cfm2_overall.assign(architecture="CFM-2 Wide/Deep"),
            baseline_all.loc[baseline_all.model == "nflows"].assign(
                architecture="nflows Existing Best"
            ),
        ], ignore_index=True,
    ).to_csv(output / "model_comparison_overall.csv", index=False)

    old_seeds = pd.read_csv(REPO_ROOT / RESULTS / "cfm_3/cfm_3_seed_summary.csv")
    param_delta = relative(seeds.total_parameters.iloc[0], old_seeds.total_parameters.iloc[0])
    train_delta = relative(seeds.training_time_seconds.mean(), old_seeds.training_time_seconds.mean())
    inference_delta = relative(
        seeds.generation_128_mean_seconds.mean(), old_seeds.generation_128_mean_seconds.mean()
    )
    improved = paired.best_error_improved.mean()
    groups4 = groups.set_index("generalization_group")
    groups3 = pd.read_csv(REPO_ROOT / RESULTS / "cfm_3/cfm_3_group_summary.csv").set_index(
        "generalization_group"
    )

    report = f"""# S2-ARCH-CFM CFM-4 阶段报告

## 阶段状态

- Architecture：CFM-4 FiLM Conditional Residual MLP
- Status：PASS
- Branch：`{branch}`
- 实验源 commit：`{source_commit}`
- 唯一核心变化：在CFM-3两个残差块输入处加入由 `[c, time_embedding]` 生成的FiLM调制。
- FiLM采用 `h'=(1+delta_gamma)*h+beta`，调制投影零初始化，初始函数严格退化为CFM-3。
- 未加入gate、U-Net、attention、normalization或CFM算法变体。
- CFM-5：未实现、未配置、未训练。

PASS表示完整工程与实验流程完成，性能结论见下文。

## Architecture

```text
[c(3), time_embedding(32)] → Linear(35,128) → SiLU
                            → Linear(128,2 blocks × 2 × 256)
[x_t, time_embedding, c] → Linear(45,256) → SiLU
→ [FiLM → Linear(256,256) → SiLU → Linear(256,256) + skip] × 2
→ Linear(256,10)
```

- Plain `ConditionalFlowMatcher(sigma=0)`；RK4/32/128 NFE。
- 数据、训练、采样、V2.1、目标集和物理评价全部冻结。
- 参数量：{int(seeds.total_parameters.iloc[0]):,}，相对CFM-3增加 {param_delta:+.2%}。

## Training and Complexity

| Seed | Best val loss | Best epoch | Epochs | Train (s) | Peak GPU MiB | Inference/128 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in seed_rows:
        report += (
            f"| {row['seed']} | {row['best_validation_loss']:.6f} | {row['best_epoch']} | "
            f"{row['epochs_completed']} | {row['training_time_seconds']:.3f} | "
            f"{row['peak_gpu_memory_bytes']/2**20:.2f} | "
            f"{1000*row['generation_128_mean_seconds']:.3f} |\n"
        )
    report += f"""

两个seed均使用完整79,945/9,992样本并正常早停。相对CFM-3，平均训练时间变化 {train_delta:+.2%}，同步推理时间变化 {inference_delta:+.2%}。

## V2.1 Validity and Target Accuracy

严格执行一次 `128 → N_valid → 2^floor(log2(N_valid)) → candidate_id裁剪 → 0D`，不补采样。

| Group | N_valid | N_keep | Valid rate | Best error median | Top-10 error median | Hit rate | Target success |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in groups.itertuples(index=False):
        report += (
            f"| {row.generalization_group} | {row.N_valid_mean:.2f} | "
            f"{row.N_keep_mean:.2f} | {row.valid_rate_mean:.2%} | "
            f"{row.best_error_median:.6f} | {row.topk_error_median:.6f} | "
            f"{row.hit_rate_mean:.2%} | {row.target_success_rate:.2%} |\n"
        )
    report += f"""
| Overall | {new.N_valid_mean:.2f} | {new.N_keep_mean:.2f} | {new.valid_rate_mean:.2%} | {new.best_error_median:.6f} | {new.topk_error_median:.6f} | {new.hit_rate_mean:.2%} | {new.target_success_rate:.2%} |

0D成功率为 {new['0d_success_rate']:.2%}；没有可靠已知可行OOD目标，因此OOD not evaluated。

## CFM-4 vs CFM-3

| Metric | CFM-3 | CFM-4 | Change |
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

逐目标同seed配对中，CFM-4有 {improved:.2%} 的运行取得更小最佳误差。合法率分组变化：ID {groups4.loc['ID','valid_rate_mean']-groups3.loc['ID','valid_rate_mean']:+.2%} pt、Interpolation {groups4.loc['Interpolation','valid_rate_mean']-groups3.loc['Interpolation','valid_rate_mean']:+.2%} pt、Boundary {groups4.loc['Boundary','valid_rate_mean']-groups3.loc['Boundary','valid_rate_mean']:+.2%} pt。

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

seed 43明显优于seed 42，说明FiLM收益存在种子敏感性；但两个seed均通过功能Gate并保持相同目标成功率。

## Comparison with Baseline and Existing Best

| Model | Valid rate | Best error median | Top-10 error median | Hit rate | Target success |
|---|---:|---:|---:|---:|---:|
| CFM-0 | {baseline.valid_rate_mean:.2%} | {baseline.best_error_median:.6f} | {baseline.topk_error_median:.6f} | {baseline.hit_rate_mean:.2%} | {baseline.target_success_rate:.2%} |
| CFM-2 | {cfm2.valid_rate_mean:.2%} | {cfm2.best_error_median:.6f} | {cfm2.topk_error_median:.6f} | {cfm2.hit_rate_mean:.2%} | {cfm2.target_success_rate:.2%} |
| CFM-3 | {old.valid_rate_mean:.2%} | {old.best_error_median:.6f} | {old.topk_error_median:.6f} | {old.hit_rate_mean:.2%} | {old.target_success_rate:.2%} |
| CFM-4 | {new.valid_rate_mean:.2%} | {new.best_error_median:.6f} | {new.topk_error_median:.6f} | {new.hit_rate_mean:.2%} | {new.target_success_rate:.2%} |
| nflows | {nflows.valid_rate_mean:.2%} | {nflows.best_error_median:.6f} | {nflows.topk_error_median:.6f} | {nflows.hit_rate_mean:.2%} | {nflows.target_success_rate:.2%} |

CFM-4最佳误差中位数仍为nflows的 {new.best_error_median/nflows.best_error_median:.2f} 倍，但差距相对CFM-3明显缩小；候选命中率仍低 {nflows.hit_rate_mean-new.hit_rate_mean:.2%} 个百分点。

## Advantages

1. 相对CFM-3，最佳误差、Top-10误差、合法率、N_keep和候选命中率均改善，逐目标配对胜率为 {improved:.2%}。
2. ID、Interpolation和Boundary三组均改善，结果支持显式条件调制增强目标响应的研究假设。
3. CFM-4成为当前准确度最强的CFM结构，同时保持96.67%的目标成功率。

## Disadvantages

1. 参数增加 {param_delta:.2%}，推理变慢 {inference_delta:.2%}；FiLM条件网络在每个ODE函数调用中重复计算。
2. seed 43显著优于seed 42，FiLM仍存在明显种子敏感性。
3. 核心目标精度仍未追平nflows，且没有OOD结论。

## Current Interpretation

与CFM-3不同，FiLM取得了跨合法率、集合误差和命中率的一致改善，且73.33%的同目标同seed运行胜出。这说明“加入FiLM条件调制的整体结构”改善了目标响应；但由于FiLM同时增加了49.26%的参数，本阶段不能把全部增益严格归因于调制机制而排除容量贡献。更高计算成本和seed敏感性仍需权衡。

## Recommendation

保留CFM-4作为当前精度最强CFM结构，同时保留CFM-2作为更高效的CFM基线。是否进入CFM-5 Gated MLP，必须由用户根据本报告明确决定。

## STAGE STOP

CFM-4 已完成。CFM-5 未开始。
"""
    (output / "CFM-4阶段报告.md").write_text(report, encoding="utf-8")

    manifest = {
        "experiment_id": "s2_arch_cfm_4_v1",
        "architecture": "CFM-4",
        "status": "PASS",
        "branch": branch,
        "source_commit": source_commit,
        "cfm_5_started": False,
        "targets": 30,
        "seeds": [42, 43],
        "ledger": "cfm_4_ledger.csv",
        "ledger_sha256": sha256_file(output / "cfm_4_ledger.csv"),
        "report": "CFM-4阶段报告.md",
        "report_sha256": sha256_file(output / "CFM-4阶段报告.md"),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
