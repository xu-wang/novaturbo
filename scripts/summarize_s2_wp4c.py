#!/usr/bin/env python3
"""Summarize S2-WP4C fixed-checkpoint RK4 ODE step ablation."""

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


STEPS = (8, 16, 32, 64)
SEEDS = (42, 43)
TARGETS = Path("config/experiments/cfm_arch/cfm_0_generalization_targets.csv")
OUTPUT = Path("results/stage2/s2_wp4c")
WP4B_EVAL_DIRS = {
    42: Path(
        "data/inverse_design/stage2/cfm_arch/"
        "s2_wp4b_cfm_4_gpu_sinkhorn_eval_seed42_v2_1"
    ),
    43: Path(
        "data/inverse_design/stage2/cfm_arch/"
        "s2_wp4b_cfm_4_gpu_sinkhorn_eval_seed43_v2_1"
    ),
}
VOLATILE_TIMING_COLUMNS = {
    "sampling_elapsed_seconds",
    "generator_time",
    "0d_time",
}


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def eval_dir(steps: int, seed: int) -> Path:
    return Path(
        f"data/inverse_design/stage2/cfm_arch/"
        f"s2_wp4c_steps{steps}_seed{seed}_v2_1"
    )


def timing_path(steps: int, seed: int) -> Path:
    return Path(
        f"data/inverse_design/stage2/cfm_arch/"
        f"s2_wp4c_inference_steps{steps}_seed{seed}.json"
    )


def main() -> int:
    output = REPO_ROOT / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    groups_map = pd.read_csv(REPO_ROOT / TARGETS)[
        ["target_id", "generalization_group"]
    ]
    frames = []
    timing_rows = []
    for steps in STEPS:
        for seed in SEEDS:
            frame = pd.read_csv(REPO_ROOT / eval_dir(steps, seed) / "target_summary.csv")
            frame.insert(0, "ode_steps", steps)
            frame.insert(1, "seed", seed)
            frames.append(frame)
            timing = json.loads(
                (REPO_ROOT / timing_path(steps, seed)).read_text(encoding="utf-8")
            )
            timing_rows.append(
                {
                    "ode_steps": steps,
                    "seed": seed,
                    "nfe": timing["mean_nfe"],
                    "mean_seconds": timing["mean_seconds"],
                    "median_seconds": timing["median_seconds"],
                    "std_seconds": timing["std_seconds"],
                    "checkpoint_ode_steps": timing["checkpoint_ode_steps"],
                }
            )
    evaluations = pd.concat(frames, ignore_index=True).merge(
        groups_map, on="target_id", how="left", validate="many_to_one"
    )
    if len(evaluations) != 240 or evaluations.target_id.nunique() != 30:
        raise RuntimeError("S2-WP4C requires 4 steps x 2 seeds x 30 targets")
    if evaluations.generalization_group.isna().any():
        raise RuntimeError("S2-WP4C contains an unclassified target")
    evaluations.to_csv(output / "s2_wp4c_ledger.csv", index=False)
    with (output / "s2_wp4c_ledger.jsonl").open("w", encoding="utf-8") as stream:
        for record in evaluations.replace({np.nan: None}).to_dict(orient="records"):
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    timings = pd.DataFrame(timing_rows)
    timings.to_csv(output / "s2_wp4c_inference.csv", index=False)
    overall_rows = []
    group_rows = []
    for steps in STEPS:
        values = evaluations.loc[evaluations.ode_steps == steps].copy()
        groups = aggregate(values)
        groups.insert(0, "ode_steps", steps)
        group_rows.append(groups)
        overall_values = values.copy()
        overall_values["generalization_group"] = "Overall"
        metric = aggregate(overall_values).iloc[0].to_dict()
        timing = timings.loc[timings.ode_steps == steps]
        metric.update(
            {
                "ode_steps": steps,
                "nfe": timing.nfe.mean(),
                "inference_mean_seconds": timing.mean_seconds.mean(),
                "inference_seed_std_seconds": timing.mean_seconds.std(ddof=0),
            }
        )
        overall_rows.append(metric)
    overall = pd.DataFrame(overall_rows).sort_values("ode_steps")
    overall.to_csv(output / "s2_wp4c_overall.csv", index=False)
    pd.concat(group_rows, ignore_index=True).to_csv(
        output / "s2_wp4c_group_summary.csv", index=False
    )

    baseline = evaluations.loc[evaluations.ode_steps == 32]
    paired_frames = []
    for steps in (8, 16, 64):
        candidate = evaluations.loc[evaluations.ode_steps == steps]
        columns = [
            "seed", "target_id", "N_valid", "N_keep", "valid_rate",
            "best_total_error", "topk_error", "hit_rate", "best_within_tolerance",
        ]
        paired = candidate[columns].merge(
            baseline[columns],
            on=["seed", "target_id"],
            suffixes=("_candidate", "_steps32"),
            validate="one_to_one",
        )
        paired.insert(0, "ode_steps", steps)
        paired["best_error_relative_change"] = (
            paired.best_total_error_candidate / paired.best_total_error_steps32 - 1.0
        )
        paired["best_error_improved"] = (
            paired.best_total_error_candidate < paired.best_total_error_steps32
        )
        paired["best_error_within_5pct"] = (
            paired.best_error_relative_change.abs() <= 0.05
        )
        paired["same_N_valid"] = paired.N_valid_candidate == paired.N_valid_steps32
        paired["same_N_keep"] = paired.N_keep_candidate == paired.N_keep_steps32
        paired_frames.append(paired)
    paired_all = pd.concat(paired_frames, ignore_index=True)
    paired_all.to_csv(output / "paired_vs_steps32.csv", index=False)

    reproduction_checks = []
    for seed in SEEDS:
        wp4b = pd.read_csv(REPO_ROOT / WP4B_EVAL_DIRS[seed] / "target_summary.csv")
        wp4c = pd.read_csv(REPO_ROOT / eval_dir(32, seed) / "target_summary.csv")
        stable_columns = [
            column
            for column in wp4b.columns
            if column not in VOLATILE_TIMING_COLUMNS
        ]
        reproduction_checks.append(
            {
                "seed": seed,
                "stable_columns_equal": wp4b[stable_columns].equals(
                    wp4c[stable_columns]
                ),
                "excluded_timing_columns": ",".join(
                    sorted(VOLATILE_TIMING_COLUMNS)
                ),
            }
        )
    reproduction = pd.DataFrame(reproduction_checks)
    reproduction.to_csv(output / "steps32_wp4b_reproduction.csv", index=False)
    if not reproduction.stable_columns_equal.all():
        raise RuntimeError("32-step evaluation did not reproduce the S2-WP4B baseline")

    metrics = overall.set_index("ode_steps")
    step8, step16, step32, step64 = (
        metrics.loc[8], metrics.loc[16], metrics.loc[32], metrics.loc[64]
    )
    paired16 = paired_all.loc[paired_all.ode_steps == 16]
    paired8 = paired_all.loc[paired_all.ode_steps == 8]
    paired64 = paired_all.loc[paired_all.ode_steps == 64]
    branch = git_value("branch", "--show-current")
    source_commit = git_value("rev-parse", "HEAD")

    report = f"""# S2-WP4C：GPU Sinkhorn CFM-4 ODE步数与推理成本消融

## 实验状态

- Status：**PASS**（4组ODE步数、双seed、30目标均完成）。
- Branch：`{branch}`；实验源 commit：`{source_commit}`。
- 固定checkpoint：S2-WP4B GPU Sinkhorn CFM-4 seed 42/43。
- 唯一变量：RK4固定步数 `8/16/32/64`；对应NFE `32/64/128/256`。
- 固定相同目标、采样seed、原始噪声、`N_raw=128`、V2.1过滤/裁剪、0D回代和目标误差。
- 不重新训练；checkpoint原生配置为32步，运行时覆盖只允许8/16/32/64。

## 基线复现检查

- 32步在seed 42/43上均逐列复现S2-WP4B结果。
- 候选数量、合法性、排序、目标误差、命中率和成功状态完全一致；仅排除不可复现的墙钟计时列 `generator_time`、`sampling_elapsed_seconds`、`0d_time`。
- 因此，本实验改变的是RK4积分步数，不是checkpoint、随机起点、目标、过滤协议或0D评价定义。

## Overall quality—NFE—time

| RK4 steps | NFE | Inference/128 | vs 32 time | Valid rate | N_keep | Best error | Top-10 error | Hit rate | Success |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in overall.itertuples(index=False):
        report += (
            f"| {int(row.ode_steps)} | {int(row.nfe)} | "
            f"{1000*row.inference_mean_seconds:.3f} ms | "
            f"{relative(row.inference_mean_seconds, step32.inference_mean_seconds):+.2%} | "
            f"{row.valid_rate_mean:.2%} | {row.N_keep_mean:.2f} | "
            f"{row.best_error_median:.6f} | {row.topk_error_median:.6f} | "
            f"{row.hit_rate_mean:.2%} | {row.target_success_rate:.2%} |\n"
        )
    report += f"""

## 16步相对32步

- 推理时间：{1000*step16.inference_mean_seconds:.3f} ms vs {1000*step32.inference_mean_seconds:.3f} ms，减少 {abs(relative(step16.inference_mean_seconds, step32.inference_mean_seconds)):.2%}。
- 最佳误差中位数：{step16.best_error_median:.6f} vs {step32.best_error_median:.6f}，变化 {relative(step16.best_error_median, step32.best_error_median):+.2%}。
- Top-10误差：{step16.topk_error_median:.6f} vs {step32.topk_error_median:.6f}，变化 {relative(step16.topk_error_median, step32.topk_error_median):+.2%}。
- 合法率变化：{step16.valid_rate_mean-step32.valid_rate_mean:+.2%} 个百分点；命中率变化：{step16.hit_rate_mean-step32.hit_rate_mean:+.2%} 个百分点。
- 60个同seed同目标配对中，{paired16.best_error_improved.mean():.2%} 的16步误差更低，{paired16.best_error_within_5pct.mean():.2%} 的相对差异在±5%内。
- `N_valid`完全相同率 {paired16.same_N_valid.mean():.2%}，`N_keep`完全相同率 {paired16.same_N_keep.mean():.2%}。
- 这表示16步在**总体统计质量**上接近32步，但并不逐目标复现相同候选；不能把总体中位数接近解释为每个目标均近似等价。

## 极端设置诊断

### 8步

- 推理时间减少 {abs(relative(step8.inference_mean_seconds, step32.inference_mean_seconds)):.2%}，但最佳误差变化 {relative(step8.best_error_median, step32.best_error_median):+.2%}，Top-10变化 {relative(step8.topk_error_median, step32.topk_error_median):+.2%}。
- 相对32步配对胜率 {paired8.best_error_improved.mean():.2%}，±5%等价率 {paired8.best_error_within_5pct.mean():.2%}。
- 结论：8步的积分误差已显著影响目标精度，不宜作为精度默认值。

### 64步

- 推理时间增加 {relative(step64.inference_mean_seconds, step32.inference_mean_seconds):+.2%}，最佳误差变化 {relative(step64.best_error_median, step32.best_error_median):+.2%}。
- 相对32步配对胜率 {paired64.best_error_improved.mean():.2%}，±5%等价率 {paired64.best_error_within_5pct.mean():.2%}。
- 结论：若64步没有稳定精度收益，则32步已基本消除主要离散误差。

## 决策

1. **推荐16步作为高吞吐筛选设置**：它将NFE从128减至64并使推理时间约减半，总体质量统计接近32步，但单目标候选可能变化。
2. **保留32步作为正式高精度复现设置**，用于最终报告或要求与既有实验严格一致的场景。
3. **不推荐8步**：速度收益更高，但目标误差明显退化。
4. **不推荐64步**：若无稳定配对收益，其额外NFE没有实际价值。
5. 本结论仅针对GPU Sinkhorn CFM-4与RK4，不外推至其他网络、Euler或Midpoint。

## STAGE STOP

S2-WP4C 已完成；未自动进入随机起点敏感性实验。
"""
    report_path = output / "S2-WP4C_ODE步数与推理成本消融报告.md"
    report_path.write_text(report, encoding="utf-8")
    manifest = {
        "experiment_id": "s2_wp4c_gpu_sinkhorn_cfm4_rk4_steps_v1",
        "status": "PASS",
        "branch": branch,
        "source_commit": source_commit,
        "ode_method": "rk4",
        "steps": list(STEPS),
        "nfe": [32, 64, 128, 256],
        "seeds": list(SEEDS),
        "targets": 30,
        "recommended_fast_steps": 16,
        "formal_reproduction_steps": 32,
        "next_experiment_started": False,
        "ledger_sha256": sha256_file(output / "s2_wp4c_ledger.csv"),
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
