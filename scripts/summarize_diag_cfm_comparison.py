#!/usr/bin/env python3
"""Summarize paired Diag-CFM versus frozen CFM-4 WP4 evaluations."""

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


def _finite_median(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    return float(finite.median())


def _panels(frame: pd.DataFrame):
    yield "all", frame
    yield "ordinary", frame[frame["target_type"].isin({"ID", "Interpolation"})]
    yield "difficult_boundary", frame[frame["target_type"] == "Boundary"]


def _model_records(evaluation: str, frame: pd.DataFrame) -> list[dict]:
    records = []
    for panel_name, panel in _panels(frame):
        for model, model_frame in panel.groupby("model", sort=False):
            records.append(
                {
                    "evaluation": evaluation,
                    "panel": panel_name,
                    "model": model,
                    "target_count": len(model_frame),
                    "median_best_total_error": _finite_median(
                        model_frame["best_total_error"]
                    ),
                    "median_top10_error": _finite_median(model_frame["topk_error"]),
                    "mean_physical_valid_rate": float(model_frame["valid_rate"].mean()),
                    "mean_hit_rate": float(model_frame["hit_rate"].mean()),
                    "target_success_rate": float(
                        model_frame["best_within_tolerance"].astype(bool).mean()
                    ),
                    "mean_diversity": float(
                        model_frame["mean_nearest_neighbour_distance"].mean()
                    ),
                    "mean_generation_seconds": float(
                        model_frame["generator_time"].mean()
                    ),
                    "mean_diagnosis_seconds": float(
                        model_frame["intrinsic_diagnosis_time"].mean()
                    ),
                    "mean_0d_seconds": float(model_frame["0d_time"].mean()),
                    "total_0d_evaluations": int(
                        model_frame["actual_0d_evaluations"].sum()
                    ),
                    "mean_zero_deviation": float(
                        model_frame["intrinsic_zero_deviation_mean"].mean()
                    ),
                    "mean_self_consistency": float(
                        model_frame["intrinsic_self_consistency_mean"].mean()
                    ),
                }
            )
    return records


def summarize(evaluation_dirs: list[Path], output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"comparison output already exists: {output_dir}")
    rows = []
    sources = []
    for directory in evaluation_dirs:
        directory = directory.resolve()
        summary_path = directory / "target_summary.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing WP4 target summary: {summary_path}")
        frame = pd.read_csv(summary_path)
        models = set(frame["model"].astype(str))
        diag_models = {model for model in models if model.startswith("diag_cfm")}
        if "cfm" not in models or not diag_models:
            raise ValueError(
                f"{directory} must contain cfm and at least one diag_cfm model"
            )
        frame.insert(0, "evaluation", directory.name)
        rows.append(frame)
        sources.append(
            {
                "directory": str(directory),
                "target_summary": str(summary_path),
                "row_count": len(frame),
            }
        )
    ledger = pd.concat(rows, ignore_index=True)

    model_records = []
    for evaluation, evaluation_frame in ledger.groupby("evaluation", sort=False):
        model_records.extend(_model_records(evaluation, evaluation_frame))
    if len(evaluation_dirs) > 1:
        model_records.extend(_model_records("pooled_all_seeds", ledger))
    model_summary = pd.DataFrame(model_records)

    paired_records = []
    for (evaluation, target_id), group in ledger.groupby(
        ["evaluation", "target_id"], sort=False
    ):
        indexed = group.set_index("model")
        if "cfm" not in indexed.index:
            continue
        cfm = indexed.loc["cfm"]
        for challenger_model in indexed.index:
            if challenger_model == "cfm":
                continue
            challenger = indexed.loc[challenger_model]
            paired_records.append(
                {
                    "evaluation": evaluation,
                    "target_id": target_id,
                    "target_type": challenger["target_type"],
                    "challenger_model": challenger_model,
                    "sample_seed_equal": int(challenger["sample_seed"])
                    == int(cfm["sample_seed"]),
                    "challenger_best_total_error": challenger["best_total_error"],
                    "cfm_best_total_error": cfm["best_total_error"],
                    "challenger_best_error_win": challenger["best_total_error"]
                    < cfm["best_total_error"],
                    "challenger_valid_rate": challenger["valid_rate"],
                    "cfm_valid_rate": cfm["valid_rate"],
                    "challenger_hit_rate": challenger["hit_rate"],
                    "cfm_hit_rate": cfm["hit_rate"],
                    "challenger_diversity": challenger[
                        "mean_nearest_neighbour_distance"
                    ],
                    "cfm_diversity": cfm["mean_nearest_neighbour_distance"],
                    "challenger_generation_seconds": challenger["generator_time"],
                    "cfm_generation_seconds": cfm["generator_time"],
                }
            )
    paired = pd.DataFrame(paired_records)
    if len(paired) and not paired["sample_seed_equal"].all():
        raise RuntimeError("paired comparison used different per-target sample seeds")

    output_dir.mkdir(parents=True, exist_ok=False)
    ledger.to_csv(output_dir / "comparison_ledger.csv", index=False)
    model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_targets.csv", index=False)
    column_source = (
        model_summary[
            (model_summary["evaluation"] == "pooled_all_seeds")
            & (model_summary["panel"] == "all")
        ]
        if len(evaluation_dirs) > 1
        else model_summary[model_summary["panel"] == "all"]
    )
    column_metrics = [
        "median_best_total_error",
        "median_top10_error",
        "mean_physical_valid_rate",
        "mean_hit_rate",
        "target_success_rate",
        "mean_diversity",
        "mean_generation_seconds",
        "mean_diagnosis_seconds",
        "mean_0d_seconds",
        "total_0d_evaluations",
    ]
    metric_columns = column_source.set_index("model")[column_metrics].T
    metric_columns.index.name = "metric"
    metric_columns.to_csv(output_dir / "metric_columns.csv")
    (output_dir / "provenance.json").write_text(
        json.dumps({"sources": sources}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    per_seed = model_summary[model_summary["evaluation"] != "pooled_all_seeds"]
    pooled = model_summary[model_summary["evaluation"] == "pooled_all_seeds"]
    overall = per_seed[per_seed["panel"] == "all"]
    difficult = per_seed[per_seed["panel"] == "difficult_boundary"]
    report = [
        "# Diag-CFM vs GPU-Sinkhorn CFM-4 comparison",
        "",
        "All hit and target-error decisions below come from the frozen NovaTurbo "
        "Brayton 0D plus mass evaluator. Intrinsic Diag-CFM scores are diagnostics only.",
        "",
        "## Overall",
        "",
    ]
    for row in overall.to_dict(orient="records"):
        report.append(
            f"- {row['evaluation']} / {row['model']}: "
            f"median best error={row['median_best_total_error']:.6g}, "
            f"valid rate={row['mean_physical_valid_rate']:.2%}, "
            f"hit rate={row['mean_hit_rate']:.2%}, "
            f"target success={row['target_success_rate']:.2%}, "
            f"diversity={row['mean_diversity']:.6g}, "
            f"generation={row['mean_generation_seconds']:.6g}s/target, "
            f"diagnosis={row['mean_diagnosis_seconds']:.6g}s/target, "
            f"0D={row['mean_0d_seconds']:.6g}s/target "
            f"({row['total_0d_evaluations']} evaluations)."
        )
    if len(pooled):
        report.extend(["", "## Pooled across seeds", ""])
        for row in pooled[pooled["panel"] == "all"].to_dict(orient="records"):
            report.append(
                f"- {row['model']}: targets={row['target_count']}, "
                f"median best error={row['median_best_total_error']:.6g}, "
                f"median Top-10 error={row['median_top10_error']:.6g}, "
                f"valid rate={row['mean_physical_valid_rate']:.2%}, "
                f"hit rate={row['mean_hit_rate']:.2%}, "
                f"target success={row['target_success_rate']:.2%}, "
                f"diversity={row['mean_diversity']:.6g}, "
                f"generation={row['mean_generation_seconds']:.6g}s/target, "
                f"diagnosis={row['mean_diagnosis_seconds']:.6g}s/target, "
                f"0D={row['mean_0d_seconds']:.6g}s/target "
                f"({row['total_0d_evaluations']} evaluations)."
            )
    report.extend(["", "## Difficult Boundary targets", ""])
    for row in difficult.to_dict(orient="records"):
        report.append(
            f"- {row['evaluation']} / {row['model']}: "
            f"median best error={row['median_best_total_error']:.6g}, "
            f"valid rate={row['mean_physical_valid_rate']:.2%}, "
            f"target success={row['target_success_rate']:.2%}."
        )
    if len(paired):
        report.extend(
            [
                "",
                "## Paired result",
                "",
                f"- Comparisons: {len(paired)}",
                f"- Identical per-target sample seeds: {bool(paired['sample_seed_equal'].all())}",
            ]
        )
        for challenger, group in paired.groupby("challenger_model", sort=False):
            report.append(
                f"- {challenger} vs CFM-4 targets: {len(group)}; "
                f"best-error win rate: {group['challenger_best_error_win'].mean():.2%}"
            )
    (output_dir / "report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.evaluation_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
