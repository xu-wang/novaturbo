#!/usr/bin/env python3
"""Append the full gradient/NSGA strategy as a new comparison column."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def append(base_dir: Path, full_dir: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"complete comparison output exists: {output_dir}")
    base_columns = pd.read_csv(base_dir / "metric_columns.csv", index_col="metric")
    base_ledger = pd.read_csv(base_dir / "comparison_ledger.csv")
    full_summary = pd.read_csv(full_dir / "model_summary.csv")
    full_targets = pd.read_csv(full_dir / "target_summary.csv")
    pooled = full_summary.loc[
        full_summary["evaluation"] == "pooled_all_seeds"
    ].iloc[0]
    if "direct_0d_seconds" in full_targets:
        mean_direct_0d_seconds = float(full_targets["direct_0d_seconds"].mean())
    else:
        direct_0d_seconds = []
        for run_directory in full_targets["run_directory"]:
            decision = json.loads(
                (Path(run_directory) / "delivery" / "decision.json").read_text(
                    encoding="utf-8"
                )
            )
            direct_0d_seconds.append(float(decision["timing_seconds"]["direct_0d"]))
        mean_direct_0d_seconds = float(np.mean(direct_0d_seconds))
    mean_total_strategy_seconds = float(
        pooled["mean_generation_seconds"]
        + mean_direct_0d_seconds
        + pooled["mean_gradient_seconds"]
        + pooled["mean_nsga2_seconds"]
    )

    full_column = {
        "median_best_total_error": pooled["median_best_total_error"],
        "median_top10_error": pooled["median_top10_error"],
        "mean_physical_valid_rate": pooled["mean_physical_valid_rate"],
        "mean_hit_rate": pooled["mean_hit_rate"],
        "target_success_rate": pooled["target_success_rate"],
        "mean_diversity": pooled["mean_diversity"],
        "mean_generation_seconds": pooled["mean_generation_seconds"],
        "mean_diagnosis_seconds": np.nan,
        "mean_0d_seconds": mean_direct_0d_seconds,
        "total_0d_evaluations": pooled["total_0d_evaluations"],
        "mean_gradient_seconds": pooled["mean_gradient_seconds"],
        "mean_nsga2_seconds": pooled["mean_nsga2_seconds"],
        "mean_0d_evaluations": pooled["mean_0d_evaluations"],
        "nsga2_trigger_count": pooled["nsga2_trigger_count"],
        "mean_network_draw_count": full_targets["network_draw_count"].mean(),
        "mean_total_strategy_seconds": mean_total_strategy_seconds,
    }
    for metric in full_column:
        if metric not in base_columns.index:
            base_columns.loc[metric] = np.nan
    base_columns["cfm_diag_gradient_nsga"] = pd.Series(full_column)
    for model, frame in base_ledger.groupby("model", sort=False):
        base_columns.loc["mean_network_draw_count", model] = frame[
            "network_draw_count"
        ].mean()
        base_columns.loc["mean_total_strategy_seconds", model] = sum(
            float(base_columns.loc[metric, model])
            for metric in (
                "mean_generation_seconds",
                "mean_diagnosis_seconds",
                "mean_0d_seconds",
            )
            if pd.notna(base_columns.loc[metric, model])
        )

    cfm = base_ledger[base_ledger["model"] == "cfm"].copy()
    cfm["seed"] = cfm["evaluation"].str.extract(r"seed(\d+)").astype(int)
    paired = full_targets.merge(
        cfm[["seed", "target_id", "best_total_error"]].rename(
            columns={"best_total_error": "cfm_best_total_error"}
        ),
        on=["seed", "target_id"],
        how="inner",
        validate="one_to_one",
    )
    paired["full_strategy_best_error_win"] = (
        paired["best_total_error"] < paired["cfm_best_total_error"]
    )
    reranked = base_ledger[base_ledger["model"] == "cfm_diag_reranked"].copy()
    reranked["seed"] = reranked["evaluation"].str.extract(r"seed(\d+)").astype(int)
    paired = paired.merge(
        reranked[["seed", "target_id", "best_total_error"]].rename(
            columns={"best_total_error": "reranked_best_total_error"}
        ),
        on=["seed", "target_id"],
        how="left",
        validate="one_to_one",
    )
    tolerance = 1.0e-15
    paired["full_strategy_improves_reranked"] = (
        paired["best_total_error"] < paired["reranked_best_total_error"] - tolerance
    )
    paired["full_strategy_ties_reranked"] = (
        (paired["best_total_error"] - paired["reranked_best_total_error"]).abs()
        <= tolerance
    )

    output_dir.mkdir(parents=True)
    base_columns.to_csv(output_dir / "metric_columns.csv")
    paired.to_csv(output_dir / "full_strategy_paired_targets.csv", index=False)
    shutil.copy2(full_dir / "model_summary.csv", output_dir / "full_strategy_summary.csv")
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {"base_comparison": str(base_dir.resolve()), "full_strategy": str(full_dir.resolve())},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = [
        "# Complete Diag-CFM strategy comparison",
        "",
        "The full strategy is CFM-4 generation -> Diag-CFM prefilter -> "
        "forward-surrogate gradient repair -> true 0D validation -> conditional NSGA-II.",
        "",
        f"- Targets: {len(full_targets)}",
        f"- Median best error: {pooled['median_best_total_error']:.6g}",
        f"- Mean hit rate: {pooled['mean_hit_rate']:.2%}",
        f"- Target success rate: {pooled['target_success_rate']:.2%}",
        f"- Best-error win rate vs CFM-4: {paired['full_strategy_best_error_win'].mean():.2%}",
        f"- Mean 0D evaluations/target: {pooled['mean_0d_evaluations']:.1f}",
        f"- NSGA-II triggers: {int(pooled['nsga2_trigger_count'])}/{len(full_targets)}",
        f"- Mean generation/prefilter time: {pooled['mean_generation_seconds']:.4f}s/target",
        f"- Mean gradient time: {pooled['mean_gradient_seconds']:.4f}s/target",
        f"- Mean NSGA-II time amortized over all targets: {pooled['mean_nsga2_seconds']:.4f}s/target",
        f"- Mean direct 0D time: {mean_direct_0d_seconds:.4f}s/target",
        f"- Mean complete strategy time: {mean_total_strategy_seconds:.4f}s/target",
        f"- Strict improvements/ties/regressions vs prefilter-only: "
        f"{int(paired['full_strategy_improves_reranked'].sum())}/"
        f"{int(paired['full_strategy_ties_reranked'].sum())}/"
        f"{int((~paired['full_strategy_improves_reranked'] & ~paired['full_strategy_ties_reranked']).sum())}",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-comparison-dir", type=Path, required=True)
    parser.add_argument("--full-strategy-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    append(args.base_comparison_dir, args.full_strategy_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
