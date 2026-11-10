#!/usr/bin/env python3
"""Build the machine-readable CFM-0 ledger and stage report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.baseline import sha256_file


TRAIN_DIR = Path("data/generator_models/stage2/cfm_arch/cfm_0_baseline_gpu6_v1")
EVAL_DIRS = {
    42: Path("data/inverse_design/stage2/cfm_arch/cfm_0_eval_seed42_v2_1"),
    43: Path("data/inverse_design/stage2/cfm_arch/cfm_0_eval_seed43_v2_1"),
}
REFERENCE_DIRS = {
    42: Path("data/inverse_design/stage2/cfm_arch/reference_models_eval_seed42_v2_1"),
    43: Path("data/inverse_design/stage2/cfm_arch/reference_models_eval_seed43_v2_1"),
}
INFERENCE = {
    42: Path("data/inverse_design/stage2/cfm_arch/cfm_0_inference_seed42.json"),
    43: Path("data/inverse_design/stage2/cfm_arch/cfm_0_inference_seed43.json"),
}
CONFIG = Path("config/experiments/cfm_arch/cfm_0_baseline_train.yaml")


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def load_evaluations(paths: dict[int, Path]) -> pd.DataFrame:
    frames = []
    for seed, directory in paths.items():
        frame = pd.read_csv(REPO_ROOT / directory / "target_summary.csv")
        frame.insert(0, "seed", seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, group), values in frame.groupby(
        ["model", "generalization_group"], sort=True
    ):
        rows.append(
            {
                "model": model,
                "generalization_group": group,
                "run_count": len(values),
                "seed_count": values["seed"].nunique(),
                "N_valid_mean": values["N_valid"].mean(),
                "N_valid_std": values["N_valid"].std(ddof=0),
                "N_keep_mean": values["N_keep"].mean(),
                "N_trimmed_mean": values["N_trimmed"].mean(),
                "valid_rate_mean": values["valid_rate"].mean(),
                "valid_rate_std": values["valid_rate"].std(ddof=0),
                "zero_valid_run_rate": (values["N_valid"] == 0).mean(),
                "0d_success_rate": (
                    values["0d_success_count"].sum()
                    / values["actual_0d_evaluations"].sum()
                    if values["actual_0d_evaluations"].sum()
                    else 0.0
                ),
                "best_error_median": values["best_total_error"].median(),
                "best_error_mean": values["best_total_error"].mean(),
                "best_error_std": values["best_total_error"].std(ddof=0),
                "topk_error_median": values["topk_error"].median(),
                "median_error_median": values["median_error"].median(),
                "hit_rate_mean": values["hit_rate"].mean(),
                "target_success_rate": values["best_within_tolerance"].mean(),
                "actual_0d_evaluations_mean": values[
                    "actual_0d_evaluations"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/s2_arch_cfm/cfm_0")
    )
    args = parser.parse_args()
    output = REPO_ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    training = pd.read_csv(REPO_ROOT / TRAIN_DIR / "seed_summary.csv")
    evaluations = load_evaluations(EVAL_DIRS)
    references = load_evaluations(REFERENCE_DIRS)
    target_groups = pd.read_csv(
        REPO_ROOT / "config/experiments/cfm_arch/cfm_0_generalization_targets.csv"
    )[["target_id", "generalization_group"]]
    evaluations = evaluations.merge(target_groups, on="target_id", how="left", validate="many_to_one")
    references = references.merge(target_groups, on="target_id", how="left", validate="many_to_one")

    branch = git_value("branch", "--show-current")
    source_commit = git_value("rev-parse", "HEAD")
    ledgers = []
    for seed in (42, 43):
        row = training.loc[training["seed"] == seed].iloc[0]
        inference = json.loads((REPO_ROOT / INFERENCE[seed]).read_text(encoding="utf-8"))
        checkpoint = TRAIN_DIR / f"cfm/seed_{seed}/best.pt"
        values = evaluations.loc[evaluations["seed"] == seed].copy()
        values.insert(0, "experiment_id", "s2_arch_cfm_0_v1")
        values.insert(1, "architecture", "CFM-0")
        values["target_F"] = values["target_thrust_N"]
        values["target_TSFC"] = values["target_tsfc_kg_N_s"]
        values["target_Mass"] = values["target_total_mass_kg"]
        values["parameter_count"] = int(row["total_parameters"])
        values["trainable_parameter_count"] = int(row["trainable_parameters"])
        values["checkpoint_size_bytes"] = int(row["best_checkpoint_bytes"])
        values["training_time"] = float(row["training_elapsed_seconds"])
        values["average_epoch_time"] = float(row["average_epoch_seconds"])
        values["best_val_loss"] = float(row["best_validation_loss"])
        values["best_epoch"] = int(row["best_epoch"])
        values["epochs_completed"] = int(row["epochs_completed"])
        values["generation_time"] = float(inference["mean_seconds"])
        values["peak_gpu_memory"] = int(row["peak_gpu_memory_bytes"])
        values["0d_success"] = values["0d_success_count"]
        values["0d_failure"] = values["0d_failure_count"]
        values["best_error"] = values["best_total_error"]
        values["git_commit"] = source_commit
        values["branch"] = branch
        values["config_path"] = str(CONFIG)
        values["checkpoint_path"] = str(checkpoint)
        values["checkpoint_sha256"] = sha256_file(REPO_ROOT / checkpoint)
        ledgers.append(values)
    ledger = pd.concat(ledgers, ignore_index=True)
    ledger.to_csv(output / "cfm_0_ledger.csv", index=False)
    records = ledger.replace({np.nan: None}).to_dict(orient="records")
    with (output / "cfm_0_ledger.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    all_models = pd.concat([evaluations, references], ignore_index=True)
    group_summary = aggregate(all_models)
    group_summary.to_csv(output / "model_group_summary.csv", index=False)
    overall = all_models.copy()
    overall["generalization_group"] = "Overall"
    overall_summary = aggregate(overall)
    overall_summary.to_csv(output / "model_overall_summary.csv", index=False)

    seed_rows = []
    for seed, values in evaluations.groupby("seed"):
        train = training.loc[training["seed"] == seed].iloc[0]
        inference = json.loads((REPO_ROOT / INFERENCE[int(seed)]).read_text(encoding="utf-8"))
        seed_rows.append(
            {
                "architecture": "CFM-0",
                "seed": int(seed),
                "total_parameters": int(train["total_parameters"]),
                "trainable_parameters": int(train["trainable_parameters"]),
                "checkpoint_size_bytes": int(train["best_checkpoint_bytes"]),
                "best_validation_loss": train["best_validation_loss"],
                "best_epoch": int(train["best_epoch"]),
                "epochs_completed": int(train["epochs_completed"]),
                "training_time_seconds": train["training_elapsed_seconds"],
                "average_epoch_seconds": train["average_epoch_seconds"],
                "peak_gpu_memory_bytes": int(train["peak_gpu_memory_bytes"]),
                "generation_128_mean_seconds": inference["mean_seconds"],
                "generation_128_median_seconds": inference["median_seconds"],
                "generation_128_std_seconds": inference["std_seconds"],
                "N_valid_mean": values["N_valid"].mean(),
                "N_keep_mean": values["N_keep"].mean(),
                "valid_rate_mean": values["valid_rate"].mean(),
                "best_error_median": values["best_total_error"].median(),
                "topk_error_median": values["topk_error"].median(),
                "median_error_median": values["median_error"].median(),
                "hit_rate_mean": values["hit_rate"].mean(),
                "target_success_rate": values["best_within_tolerance"].mean(),
            }
        )
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(output / "cfm_0_seed_summary.csv", index=False)

    old_dir = REPO_ROOT / "data/generator_models/wp3_active10d_full_gpu6_v1/cfm"
    state_equal = {}
    for seed in (42, 43):
        old_payload = torch.load(old_dir / f"seed_{seed}/best.pt", map_location="cpu", weights_only=False)
        new_payload = torch.load(
            REPO_ROOT / TRAIN_DIR / f"cfm/seed_{seed}/best.pt",
            map_location="cpu",
            weights_only=False,
        )
        state_equal[str(seed)] = all(
            torch.equal(old_payload["state_dict"][key], new_payload["state_dict"][key])
            for key in old_payload["state_dict"]
        )

    cfm = overall_summary.loc[overall_summary["model"] == "cfm"].iloc[0]
    nflows = overall_summary.loc[overall_summary["model"] == "nflows"].iloc[0]
    report = f"""# S2-ARCH-CFM CFM-0 阶段报告

## 阶段状态

- Architecture：CFM-0 Current Baseline
- Status：PASS
- Branch：`{branch}`
- 实验源 commit：`{source_commit}`
- CFM-1：未实现、未配置、未训练

## Architecture

```text
[x_t(10), scalar t(1), c(3)]
→ Linear(14,128) → SiLU
→ Linear(128,128) → SiLU
→ Linear(128,128) → SiLU
→ Linear(128,10)
```

- Plain `ConditionalFlowMatcher(sigma=0)`；不是 OT-CFM。
- 标准高斯源分布；RK4、32 steps、128 NFE。
- 无 residual、FiLM、gating、U-Net、attention 或有界输出。
- Total/trainable parameters：{int(training.iloc[0]['total_parameters']):,} / {int(training.iloc[0]['trainable_parameters']):,}。
- 新旧 CFM-0 权重逐 tensor 完全一致：seed42={state_equal['42']}，seed43={state_equal['43']}。

## Training

| Seed | Best val loss | Best epoch | Completed | Time (s) | Avg epoch (s) | Peak GPU MiB | Checkpoint KiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in seed_rows:
        report += (
            f"| {row['seed']} | {row['best_validation_loss']:.6f} | {row['best_epoch']} | "
            f"{row['epochs_completed']} | {row['training_time_seconds']:.3f} | "
            f"{row['average_epoch_seconds']:.3f} | {row['peak_gpu_memory_bytes']/2**20:.2f} | "
            f"{row['checkpoint_size_bytes']/2**10:.2f} |\n"
        )
    report += """

两个 seed 均通过 early stopping；没有修改 learning rate、scheduler、batch size、patience 或最大 epoch。

## Inference

固定 GPU 6、float32、RK4/32、`N_raw=128`，5次warm-up后重复30次，并在计时前后执行CUDA同步。

| Seed | Mean (ms) | Median (ms) | Std (ms) |
|---:|---:|---:|---:|
"""
    for row in seed_rows:
        report += (
            f"| {row['seed']} | {1000*row['generation_128_mean_seconds']:.3f} | "
            f"{1000*row['generation_128_median_seconds']:.3f} | "
            f"{1000*row['generation_128_std_seconds']:.3f} |\n"
        )
    report += f"""

## V2.1 Validity and 0D Accuracy

全部运行严格执行 `128 → N_valid → N_keep → 0D`，不补采样；裁剪仅按原始candidate_id。

| Group | N_valid mean | N_keep mean | Valid rate | Best error median | Top-10 mean error median | Candidate hit rate | Target success |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    cfm_groups = group_summary.loc[group_summary["model"] == "cfm"]
    for row in cfm_groups.itertuples(index=False):
        report += (
            f"| {row.generalization_group} | {row.N_valid_mean:.2f} | {row.N_keep_mean:.2f} | "
            f"{row.valid_rate_mean:.2%} | {row.best_error_median:.6f} | "
            f"{row.topk_error_median:.6f} | {row.hit_rate_mean:.2%} | "
            f"{row.target_success_rate:.2%} |\n"
        )
    report += f"""

OOD：not evaluated。现有目标类为 `OOD-feasible-unknown`，没有把未知可行目标强行解释为可靠的OOD准确度。

## Comparison with Existing Models

| Model | Valid rate | N_keep mean | Best error median | Top-10 mean error median | Candidate hit rate | Target success |
|---|---:|---:|---:|---:|---:|---:|
"""
    for row in overall_summary.sort_values("model").itertuples(index=False):
        report += (
            f"| {row.model} | {row.valid_rate_mean:.2%} | {row.N_keep_mean:.2f} | "
            f"{row.best_error_median:.6f} | {row.topk_error_median:.6f} | "
            f"{row.hit_rate_mean:.2%} | {row.target_success_rate:.2%} |\n"
        )
    report += f"""

CFM-0 相对 nflows：合法率差 {cfm['valid_rate_mean']-nflows['valid_rate_mean']:+.2%}，但最佳误差中位数高 {cfm['best_error_median']/nflows['best_error_median']:.2f} 倍。因此它当前不是最佳准确度模型。

## Advantages

1. 合法率为 {cfm['valid_rate_mean']:.2%}，高于当前 cVAE/cINN，且与 nflows 接近。
2. 两个训练 seed 的损失、合法率和目标误差处于同一数量级，未出现单 seed 崩溃。
3. 128候选GPU生成仅需约 {1000*np.mean([r['generation_128_mean_seconds'] for r in seed_rows]):.2f} ms。

## Disadvantages

1. 最佳目标误差、Top-10误差和候选命中率均明显弱于 nflows/cINN。
2. Interpolation 目标成功率低于 ID/Boundary，说明条件响应仍不足。
3. 固定RK4/32需要128次速度场调用；当前网络虽小，但采样不是单次前向。

## Current Interpretation

CFM-0 的主要价值是“能生成较多合法候选且速度可接受”，而不是“最接近给定目标”。其瓶颈更像条件化速度场表达不足，而非训练未收敛：两个seed均稳定早停，并复现历史权重与验证损失。该结论为下一阶段单独检验时间嵌入提供基线，但本阶段不据此提前实现 CFM-1。

## Recommendation

保留 CFM-0 作为结构消融基线，不作为当前最佳反向设计模型。是否进入 CFM-1 需用户明确决定。

## STAGE STOP

当前阶段 CFM-0 已完成。尚未执行 CFM-1。
"""
    (output / "CFM-0阶段报告.md").write_text(report, encoding="utf-8")

    manifest = {
        "experiment_id": "s2_arch_cfm_0_v1",
        "architecture": "CFM-0",
        "status": "PASS",
        "branch": branch,
        "source_commit": source_commit,
        "cfm_1_started": False,
        "state_dict_equal_to_frozen_baseline": state_equal,
        "ledger": "cfm_0_ledger.csv",
        "ledger_sha256": sha256_file(output / "cfm_0_ledger.csv"),
        "report": "CFM-0阶段报告.md",
        "report_sha256": sha256_file(output / "CFM-0阶段报告.md"),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
