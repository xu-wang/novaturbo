#!/usr/bin/env python3
"""Summarize S2-WP4B CFM-4 Plain vs GPU Sinkhorn OT-CFM."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_cfm_2 import aggregate, relative
from src.ai.baseline import sha256_file


TRAIN_DIR = Path(
    "data/generator_models/stage2/cfm_arch/s2_wp4b_cfm_4_gpu_sinkhorn_gpu6_v1"
)
EVAL_DIRS = {
    42: Path(
        "data/inverse_design/stage2/cfm_arch/"
        "s2_wp4b_cfm_4_gpu_sinkhorn_eval_seed42_v2_1"
    ),
    43: Path(
        "data/inverse_design/stage2/cfm_arch/"
        "s2_wp4b_cfm_4_gpu_sinkhorn_eval_seed43_v2_1"
    ),
}
INFERENCE = {
    42: Path(
        "data/inverse_design/stage2/cfm_arch/"
        "s2_wp4b_cfm_4_gpu_sinkhorn_inference_seed42.json"
    ),
    43: Path(
        "data/inverse_design/stage2/cfm_arch/"
        "s2_wp4b_cfm_4_gpu_sinkhorn_inference_seed43.json"
    ),
}
CONFIG = Path("config/experiments/cfm_arch/s2_wp4b_cfm_4_ot_train.yaml")
TARGETS = Path("config/experiments/cfm_arch/cfm_0_generalization_targets.csv")
PLAIN_DIR = Path("results/s2_arch_cfm/cfm_4")
OUTPUT = Path("results/stage2/s2_wp4b")


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def main() -> int:
    output = REPO_ROOT / OUTPUT
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
        raise RuntimeError("S2-WP4B requires both seeds on all 30 frozen targets")
    if evaluations.generalization_group.isna().any():
        raise RuntimeError("S2-WP4B contains an unclassified target")

    branch = git_value("branch", "--show-current")
    source_commit = git_value("rev-parse", "HEAD")
    ledger_parts = []
    seed_rows = []
    for seed in (42, 43):
        train = training.loc[training.seed == seed].iloc[0]
        timing = json.loads((REPO_ROOT / INFERENCE[seed]).read_text(encoding="utf-8"))
        checkpoint = TRAIN_DIR / f"cfm/seed_{seed}/best.pt"
        values = evaluations.loc[evaluations.seed == seed].copy()
        values.insert(0, "experiment_id", "s2_wp4b_cfm_4_gpu_sinkhorn_v1")
        values.insert(1, "architecture", "CFM-4-GPU-SINKHORN")
        values["parameter_count"] = int(train.total_parameters)
        values["checkpoint_size_bytes"] = int(train.best_checkpoint_bytes)
        values["training_time"] = float(train.training_elapsed_seconds)
        values["average_epoch_time"] = float(train.average_epoch_seconds)
        values["best_val_loss"] = float(train.best_validation_loss)
        values["generation_time"] = float(timing["mean_seconds"])
        values["peak_gpu_memory"] = int(train.peak_gpu_memory_bytes)
        values["git_commit"] = source_commit
        values["branch"] = branch
        values["config_path"] = str(CONFIG)
        values["checkpoint_path"] = str(checkpoint)
        values["checkpoint_sha256"] = sha256_file(REPO_ROOT / checkpoint)
        ledger_parts.append(values)
        seed_rows.append(
            {
                "method": "GPU Sinkhorn OT-CFM",
                "seed": seed,
                "total_parameters": int(train.total_parameters),
                "checkpoint_size_bytes": int(train.best_checkpoint_bytes),
                "best_validation_loss": train.best_validation_loss,
                "best_epoch": int(train.best_epoch),
                "epochs_completed": int(train.epochs_completed),
                "training_time_seconds": train.training_elapsed_seconds,
                "average_epoch_seconds": train.average_epoch_seconds,
                "peak_gpu_memory_bytes": int(train.peak_gpu_memory_bytes),
                "generation_128_mean_seconds": timing["mean_seconds"],
                "generation_128_median_seconds": timing["median_seconds"],
                "generation_128_std_seconds": timing["std_seconds"],
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
    ledger = pd.concat(ledger_parts, ignore_index=True)
    ledger.to_csv(output / "s2_wp4b_ledger.csv", index=False)
    with (output / "s2_wp4b_ledger.jsonl").open("w", encoding="utf-8") as stream:
        for record in ledger.replace({np.nan: None}).to_dict(orient="records"):
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(output / "s2_wp4b_seed_summary.csv", index=False)
    groups = aggregate(evaluations)
    groups.to_csv(output / "s2_wp4b_group_summary.csv", index=False)
    overall_input = evaluations.copy()
    overall_input["generalization_group"] = "Overall"
    overall = aggregate(overall_input)
    overall.to_csv(output / "s2_wp4b_overall_summary.csv", index=False)

    plain_ledger = pd.read_csv(REPO_ROOT / PLAIN_DIR / "cfm_4_ledger.csv")
    columns = [
        "seed", "target_id", "N_valid", "N_keep", "valid_rate",
        "best_total_error", "topk_error", "hit_rate", "best_within_tolerance",
    ]
    paired = evaluations.merge(
        plain_ledger[columns],
        on=["seed", "target_id"],
        suffixes=("_sinkhorn", "_plain"),
        validate="one_to_one",
    )
    paired["valid_rate_delta"] = paired.valid_rate_sinkhorn - paired.valid_rate_plain
    paired["best_error_delta"] = (
        paired.best_total_error_sinkhorn - paired.best_total_error_plain
    )
    paired["best_error_improved"] = paired.best_error_delta < 0
    paired["topk_error_delta"] = paired.topk_error_sinkhorn - paired.topk_error_plain
    paired["hit_rate_delta"] = paired.hit_rate_sinkhorn - paired.hit_rate_plain
    paired.to_csv(output / "plain_vs_gpu_sinkhorn_paired.csv", index=False)

    plain_overall = pd.read_csv(REPO_ROOT / PLAIN_DIR / "cfm_4_overall_summary.csv")
    plain_groups = pd.read_csv(REPO_ROOT / PLAIN_DIR / "cfm_4_group_summary.csv")
    plain_seeds = pd.read_csv(REPO_ROOT / PLAIN_DIR / "cfm_4_seed_summary.csv")
    comparison = pd.concat(
        [
            plain_overall.assign(method="Plain CFM"),
            overall.assign(method="GPU Sinkhorn OT-CFM"),
        ],
        ignore_index=True,
    )
    comparison.to_csv(output / "plain_vs_gpu_sinkhorn_overall.csv", index=False)

    current = overall.iloc[0]
    plain = plain_overall.iloc[0]
    group_index = groups.set_index("generalization_group")
    plain_group_index = plain_groups.set_index("generalization_group")
    paired_win = paired.best_error_improved.mean()
    seed_ratio = (
        max(row["best_error_median"] for row in seed_rows)
        / min(row["best_error_median"] for row in seed_rows)
    )
    plain_seed_ratio = (
        plain_seeds.best_error_median.max() / plain_seeds.best_error_median.min()
    )

    report = f"""# S2-WP4B：CFM-4 Plain CFM vs GPU Sinkhorn OT-CFM

## 实验状态与范围

- Status：**PASS**（工程与实验闭环完成）。
- Branch：`{branch}`；实验源 commit：`{source_commit}`。
- 固定速度场：CFM-4 FiLM Conditional Residual MLP，参数量{int(seeds.total_parameters.iloc[0]):,}。
- 唯一训练方法变量：Plain独立配对 vs GPU entropic Sinkhorn minibatch OT配对。
- Sinkhorn冻结参数：batch-mean cost normalization、正则化 `epsilon=0.1`、50次log-domain迭代。
- GPU：物理6；数据、batch=1024、seed、训练终止、采样器、RK4/32和V2.1全部冻结。
- 官方CPU Exact OT曾启动但因用户要求中止，未形成checkpoint，不进入本报告结果。

GPU Sinkhorn是近似正则化OT，不等同于TorchCFM/POT的Exact EMD；报告不得把二者混称。

## 为什么使用GPU近似OT

TorchCFM Exact OT将代价矩阵转为CPU NumPy并由POT求解。固定1024 batch时，CPU Exact OT运行约30分钟仍未完成第一个seed。GPU Sinkhorn用纯PyTorch在GPU 6计算传输计划，约{seeds.training_time_seconds.iloc[0]/60:.2f}和{seeds.training_time_seconds.iloc[1]/60:.2f}分钟完成两个seed的正式训练。

## Training and Complexity

| Seed | Best val loss | Best epoch | Epochs | Train (s) | Epoch (s) | Peak GPU MiB | Inference/128 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in seed_rows:
        report += (
            f"| {row['seed']} | {row['best_validation_loss']:.6f} | "
            f"{row['best_epoch']} | {row['epochs_completed']} | "
            f"{row['training_time_seconds']:.3f} | {row['average_epoch_seconds']:.3f} | "
            f"{row['peak_gpu_memory_bytes']/2**20:.2f} | "
            f"{1000*row['generation_128_mean_seconds']:.3f} |\n"
        )
    report += f"""

相对Plain CFM，GPU Sinkhorn平均训练时间变化 {relative(seeds.training_time_seconds.mean(), plain_seeds.training_time_seconds.mean()):+.2%}，平均epoch时间变化 {relative(seeds.average_epoch_seconds.mean(), plain_seeds.average_epoch_seconds.mean()):+.2%}，峰值显存变化 {relative(seeds.peak_gpu_memory_bytes.mean(), plain_seeds.peak_gpu_memory_bytes.mean()):+.2%}。推理阶段不计算OT，因此推理时间变化仅 {relative(seeds.generation_128_mean_seconds.mean(), plain_seeds.generation_128_mean_seconds.mean()):+.2%}。

Plain与OT训练损失定义的配对路径不同，绝对loss数值不能直接解释为物理精度；最终结论以V2.1回代为准。

## V2.1 分组结果

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
| Overall | {current.N_valid_mean:.2f} | {current.N_keep_mean:.2f} | {current.valid_rate_mean:.2%} | {current.best_error_median:.6f} | {current.topk_error_median:.6f} | {current.hit_rate_mean:.2%} | {current.target_success_rate:.2%} |

0D成功率为 {current['0d_success_rate']:.2%}；没有可靠已知可行OOD目标，因此OOD不评价。

## Plain vs GPU Sinkhorn

| Metric | Plain CFM-4 | GPU Sinkhorn OT-CFM | Change |
|---|---:|---:|---:|
| Parameters | {int(plain_seeds.total_parameters.iloc[0]):,} | {int(seeds.total_parameters.iloc[0]):,} | 0.00% |
| Valid rate | {plain.valid_rate_mean:.2%} | {current.valid_rate_mean:.2%} | {current.valid_rate_mean-plain.valid_rate_mean:+.2%} pt |
| N_keep mean | {plain.N_keep_mean:.2f} | {current.N_keep_mean:.2f} | {current.N_keep_mean-plain.N_keep_mean:+.2f} |
| Best error median | {plain.best_error_median:.6f} | {current.best_error_median:.6f} | {relative(current.best_error_median, plain.best_error_median):+.2%} |
| Top-10 error median | {plain.topk_error_median:.6f} | {current.topk_error_median:.6f} | {relative(current.topk_error_median, plain.topk_error_median):+.2%} |
| Candidate hit rate | {plain.hit_rate_mean:.2%} | {current.hit_rate_mean:.2%} | {current.hit_rate_mean-plain.hit_rate_mean:+.2%} pt |
| Target success | {plain.target_success_rate:.2%} | {current.target_success_rate:.2%} | {current.target_success_rate-plain.target_success_rate:+.2%} pt |

逐目标同seed配对中，GPU Sinkhorn有 {paired_win:.2%} 取得更小最佳误差。分组合法率变化：ID {group_index.loc['ID','valid_rate_mean']-plain_group_index.loc['ID','valid_rate_mean']:+.2%} pt、Interpolation {group_index.loc['Interpolation','valid_rate_mean']-plain_group_index.loc['Interpolation','valid_rate_mean']:+.2%} pt、Boundary {group_index.loc['Boundary','valid_rate_mean']-plain_group_index.loc['Boundary','valid_rate_mean']:+.2%} pt。

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

GPU Sinkhorn双seed最佳误差中位数之比为 {seed_ratio:.2f}，Plain为 {plain_seed_ratio:.2f}；数值越接近1表示seed稳定性越好。

## 判断

1. **目标精度温和改善**：最佳误差中位数降低 {abs(relative(current.best_error_median, plain.best_error_median)):.2%}，Top-10误差降低 {abs(relative(current.topk_error_median, plain.topk_error_median)):.2%}；但逐目标配对胜率只有 {paired_win:.2%}，不是压倒性优势。
2. **合法率没有实质改善**：总体只变化 {current.valid_rate_mean-plain.valid_rate_mean:+.2%} 个百分点，且ID下降、Interpolation和Boundary上升。
3. **seed稳定性改善**：双seed最佳误差比从 {plain_seed_ratio:.2f} 降到 {seed_ratio:.2f}，且两个seed目标成功率一致。
4. **成本权衡**：训练时间增加 {relative(seeds.training_time_seconds.mean(), plain_seeds.training_time_seconds.mean()):+.2%}，但OT只用于训练，部署推理成本基本不变。
5. **阶段判断**：GPU Sinkhorn可以作为CFM-4的精度/稳定性候选保留，但现有证据不足以淘汰Plain CFM-4；后者训练更便宜且候选命中率略高。
6. 本结论只适用于固定 `epsilon=0.1` 和50次迭代，不外推到Exact OT或其他正则化强度。下一实验必须由用户决定。

## STAGE STOP

S2-WP4B 已完成；未自动进入下一实验。
"""
    report_path = output / "S2-WP4B_Plain与GPU-Sinkhorn-OT-CFM对照报告.md"
    report_path.write_text(report, encoding="utf-8")
    manifest = {
        "experiment_id": "s2_wp4b_cfm_4_gpu_sinkhorn_v1",
        "status": "PASS",
        "branch": branch,
        "source_commit": source_commit,
        "plain_method": "ConditionalFlowMatcher",
        "ot_method": "GPU entropic Sinkhorn minibatch OT-CFM",
        "exact_ot": False,
        "sinkhorn": {
            "cost_normalization": "batch_mean",
            "regularization": 0.1,
            "iterations": 50,
        },
        "seeds": [42, 43],
        "targets": 30,
        "next_experiment_started": False,
        "ledger_sha256": sha256_file(output / "s2_wp4b_ledger.csv"),
        "report": report_path.name,
        "report_sha256": sha256_file(report_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
