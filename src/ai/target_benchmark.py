"""Target-directed initialization benchmark for conditional generators.

Unlike WP7/B0's global Pareto analysis, this benchmark evaluates the stated
inverse-design task: given a frozen performance target, which initializer puts
the population closest to that target under equal candidate and search budgets?
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from .baseline import sha256_file, write_json, write_yaml
from .inverse_design import run_inverse_design
from .wp7 import _selected_targets, _target_error_columns, _target_spec


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def load_target_benchmark_config(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "version",
        "target_suite",
        "optimizer_seeds",
        "generator_seed_offset",
        "model_sets",
        "base_run_config",
        "output",
    }
    missing = required.difference(value)
    if missing:
        raise ValueError(f"Target benchmark config is missing keys: {sorted(missing)}")
    if not value["optimizer_seeds"]:
        raise ValueError("optimizer_seeds cannot be empty")
    if not value["model_sets"]:
        raise ValueError("model_sets cannot be empty")
    for model_set in value["model_sets"]:
        model_ids = [str(model["model_id"]) for model in model_set["models"]]
        if not model_ids or len(model_ids) != len(set(model_ids)):
            raise ValueError("Every model set needs unique, non-empty model IDs")
    return value


def _run_id(training_seed: int, target_id: str, optimizer_seed: int, method: str) -> str:
    return (
        f"target_t{training_seed}_{target_id[:10]}_s{optimizer_seed}_{method}"
    )


def _materialize_run_config(
    base: Mapping[str, Any],
    *,
    run_id: str,
    method: str,
    checkpoint: Optional[Mapping[str, Any]],
    optimizer_seed: int,
    generator_seed_offset: int,
) -> Dict[str, Any]:
    value = copy.deepcopy(dict(base))
    value["run_id"] = run_id
    value["seeds"] = {
        "generator": int(generator_seed_offset) + int(optimizer_seed),
        "optimizer": int(optimizer_seed),
        "background": int(optimizer_seed),
    }
    value["optimizer"]["seed"] = int(optimizer_seed)
    if method == "random":
        value["mode"] = "random"
        value["model"] = {"model_id": None, "checkpoint_path": None}
        value["population"]["generator_ratio"] = 0.0
    elif method == "screened_random":
        value["mode"] = "screened_random"
        value["model"] = {"model_id": None, "checkpoint_path": None}
        value["population"]["generator_ratio"] = 1.0
    else:
        if checkpoint is None:
            raise ValueError(f"Generator method {method} is missing a checkpoint")
        value["mode"] = "generator"
        value["model"] = {
            "model_id": str(checkpoint["model_id"]),
            "checkpoint_path": str(checkpoint["checkpoint_path"]),
        }
        value["population"]["generator_ratio"] = 1.0
    value["output"]["refuse_overwrite"] = True
    return value


def _first_success(frame: pd.DataFrame) -> float:
    success = frame.loc[frame["within_tolerance"].astype(bool)]
    return float(success["evaluation_index"].min() + 1) if len(success) else math.nan


def _top_mean(frame: pd.DataFrame, k: int) -> float:
    values = frame.loc[frame["is_feasible"].astype(bool), "total_error"].sort_values()
    return float(values.head(k).mean()) if len(values) else math.nan


def _read_forward_results(path: Path) -> pd.DataFrame:
    """Read forward results while accepting a valid zero-candidate artifact."""

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["forward_valid", "total_error", "within_tolerance"])


def _collect_run_metrics(run: Mapping[str, Any], run_dir: Path) -> tuple[Dict[str, Any], pd.DataFrame]:
    ledger = pd.read_csv(run_dir / "optimization" / "evaluation_ledger.csv")
    target = json.loads((run_dir / "target.json").read_text(encoding="utf-8"))
    derived = json.loads((run_dir / "derived_metrics.json").read_text(encoding="utf-8"))
    ledger = _target_error_columns(ledger, target, derived)
    initial = ledger.loc[ledger["generation"].astype(int) == 0].copy()
    feasible_initial = initial.loc[initial["is_feasible"].astype(bool)]
    summary = json.loads(
        (run_dir / "summary" / "metrics.json").read_text(encoding="utf-8")
    )
    record = {
        **dict(run),
        "run_dir": str(run_dir),
        "initial_population_size": len(initial),
        "initial_feasible_rate": float(len(feasible_initial) / max(len(initial), 1)),
        "initial_best_error": (
            float(feasible_initial["total_error"].min()) if len(feasible_initial) else math.nan
        ),
        "initial_mean_error": (
            float(feasible_initial["total_error"].mean()) if len(feasible_initial) else math.nan
        ),
        "initial_median_error": (
            float(feasible_initial["total_error"].median()) if len(feasible_initial) else math.nan
        ),
        "initial_top5_mean_error": _top_mean(initial, 5),
        "initial_top10_mean_error": _top_mean(initial, 10),
        "initial_hit_rate": (
            float(initial["within_tolerance"].astype(bool).mean()) if len(initial) else 0.0
        ),
        "initial_has_hit": bool(initial["within_tolerance"].astype(bool).any()),
        "search_best_error": float(ledger["total_error"].min()),
        "search_has_hit": bool(ledger["within_tolerance"].astype(bool).any()),
        "search_evaluations_to_hit": _first_success(ledger),
        "prefilter_evaluations": int(summary["cost"]["prefilter_0d_evaluations"]),
        "nsga2_evaluations": int(summary["cost"]["nsga2_0d_evaluations"]),
        "total_evaluations": int(summary["cost"]["total_0d_evaluations"]),
    }
    if run["method"] != "random":
        forward = _read_forward_results(
            run_dir / "round_trip" / "forward_results.csv"
        )
        valid = forward.loc[forward["forward_valid"].astype(bool)]
        record.update(
            {
                "candidate_valid_rate": float(len(valid) / max(len(forward), 1)),
                "candidate_best_error": (
                    float(valid["total_error"].min()) if len(valid) else math.nan
                ),
                "candidate_hit_rate": (
                    float(valid["within_tolerance"].astype(bool).mean())
                    if len(valid)
                    else 0.0
                ),
            }
        )
    else:
        record.update(
            {
                "candidate_valid_rate": math.nan,
                "candidate_best_error": math.nan,
                "candidate_hit_rate": math.nan,
            }
        )

    curves = []
    for generation in sorted(ledger["generation"].astype(int).unique()):
        current = ledger.loc[ledger["generation"].astype(int) <= generation]
        feasible = current.loc[current["is_feasible"].astype(bool)]
        curves.append(
            {
                **dict(run),
                "generation": int(generation),
                "evaluation_count": len(current),
                "best_error": (
                    float(feasible["total_error"].min()) if len(feasible) else math.nan
                ),
                "has_hit": bool(current["within_tolerance"].astype(bool).any()),
            }
        )
    return record, pd.DataFrame(curves)


def _paired_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["training_seed", "target_id", "target_type", "optimizer_seed"]
    baselines = {
        method: group.set_index(keys)
        for method, group in metrics.loc[
            metrics["method"].isin(["random", "screened_random"])
        ].groupby("method")
    }
    rows = []
    for row in metrics.loc[~metrics["method"].isin(baselines)].to_dict(orient="records"):
        key = tuple(row[name] for name in keys)
        for baseline_method, baseline in baselines.items():
            if key not in baseline.index:
                continue
            base = baseline.loc[key]
            rows.append(
                {
                    **{name: row[name] for name in keys},
                    "model_id": row["method"],
                    "baseline": baseline_method,
                    "initial_best_error_improvement": float(
                        base["initial_best_error"] - row["initial_best_error"]
                    ),
                    "initial_top10_error_improvement": float(
                        base["initial_top10_mean_error"] - row["initial_top10_mean_error"]
                    ),
                    "initial_hit_rate_improvement": float(
                        row["initial_hit_rate"] - base["initial_hit_rate"]
                    ),
                    "search_best_error_improvement": float(
                        base["search_best_error"] - row["search_best_error"]
                    ),
                    "initial_best_win": bool(
                        row["initial_best_error"] < base["initial_best_error"]
                    ),
                    "initial_top10_win": bool(
                        row["initial_top10_mean_error"]
                        < base["initial_top10_mean_error"]
                    ),
                    "search_best_win": bool(
                        row["search_best_error"] < base["search_best_error"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _aggregate(metrics: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    ordinary = metrics[metrics["target_type"].isin(["ID", "Interpolation"])]
    records = []
    for method, group in ordinary.groupby("method", sort=False):
        record = {
            "method": method,
            "run_count": len(group),
            "initial_best_error_mean": float(group["initial_best_error"].mean()),
            "initial_top10_error_mean": float(group["initial_top10_mean_error"].mean()),
            "initial_hit_rate_mean": float(group["initial_hit_rate"].mean()),
            "initial_target_success_rate": float(group["initial_has_hit"].mean()),
            "search_best_error_mean": float(group["search_best_error"].mean()),
            "search_target_success_rate": float(group["search_has_hit"].mean()),
            "candidate_hit_rate_mean": float(group["candidate_hit_rate"].mean()),
        }
        for baseline in ("screened_random", "random"):
            comparisons = paired.loc[
                (paired["model_id"] == method)
                & (paired["baseline"] == baseline)
                & paired["target_type"].isin(["ID", "Interpolation"])
            ]
            record[f"vs_{baseline}_pair_count"] = len(comparisons)
            record[f"vs_{baseline}_initial_best_win_rate"] = (
                float(comparisons["initial_best_win"].mean()) if len(comparisons) else math.nan
            )
            record[f"vs_{baseline}_initial_top10_win_rate"] = (
                float(comparisons["initial_top10_win"].mean()) if len(comparisons) else math.nan
            )
            record[f"vs_{baseline}_search_best_win_rate"] = (
                float(comparisons["search_best_win"].mean()) if len(comparisons) else math.nan
            )
        records.append(record)
    return pd.DataFrame(records)


def _report(aggregate: pd.DataFrame) -> str:
    lines = [
        "# Target-directed initialization benchmark",
        "",
        "Primary question: given `[F*, TSFC*, Mass*]`, which method creates the closest generation-0 population?",
        "Raw-objective Hypervolume and Pareto coverage are not acceptance metrics in this benchmark.",
        "",
        "| Method | Initial best error | Initial Top-10 error | Initial hit rate | Final best error | vs Screened Random best-win |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        win = row.get("vs_screened_random_initial_best_win_rate", math.nan)
        lines.append(
            f"| {row['method']} | {row['initial_best_error_mean']:.6g} | "
            f"{row['initial_top10_error_mean']:.6g} | {row['initial_hit_rate_mean']:.2%} | "
            f"{row['search_best_error_mean']:.6g} | "
            f"{win:.2%} |" if math.isfinite(win) else
            f"| {row['method']} | {row['initial_best_error_mean']:.6g} | "
            f"{row['initial_top10_error_mean']:.6g} | {row['initial_hit_rate_mean']:.2%} | "
            f"{row['search_best_error_mean']:.6g} | — |"
        )
    lines.extend(
        [
            "",
            "Errors are equally weighted relative errors in thrust, TSFC, and mass. Lower is better.",
            "A hit requires the same candidate to satisfy all frozen WP4 tolerances.",
            "",
        ]
    )
    return "\n".join(lines)


def run_target_benchmark(
    config_path: Path,
    *,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
    resume: bool = False,
) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    config_path = Path(config_path).resolve()
    config = load_target_benchmark_config(config_path)
    output_dir = (
        output_dir_override.resolve()
        if output_dir_override is not None
        else _resolve(repo_root, config["output"]["directory"])
    )
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Target benchmark output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("targets", "run_configs", "runs"):
        (output_dir / name).mkdir(exist_ok=True)

    suite_path = _resolve(repo_root, config["target_suite"]["csv"])
    if sha256_file(suite_path) != str(config["target_suite"]["sha256"]):
        raise ValueError("Target suite SHA256 differs from benchmark config")
    targets = _selected_targets(
        pd.read_csv(suite_path), config["target_suite"]["selection"]
    )
    targets.to_csv(output_dir / "selected_targets.csv", index=False)
    base_path = _resolve(repo_root, config["base_run_config"])
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    tolerance_profile = str(
        yaml.safe_load(
            _resolve(repo_root, base["data"]["metrics_config"]).read_text(encoding="utf-8")
        )["metrics_version"]
    )
    started = dt.datetime.now(dt.timezone.utc)
    run_rows = []
    metric_rows = []
    curves = []
    for model_set in config["model_sets"]:
        training_seed = int(model_set["training_seed"])
        model_lookup = {
            str(model["model_id"]): model for model in model_set["models"]
        }
        methods = ["random", "screened_random", *model_lookup]
        for target_row in targets.to_dict(orient="records"):
            target = _target_spec(target_row, tolerance_profile)
            target_path = output_dir / "targets" / f"{target['target_id']}.yaml"
            if not target_path.exists():
                write_yaml(target_path, target)
            for optimizer_seed_value in config["optimizer_seeds"]:
                optimizer_seed = int(optimizer_seed_value)
                for method in methods:
                    run_id = _run_id(
                        training_seed, str(target["target_id"]), optimizer_seed, method
                    )
                    run = {
                        "run_id": run_id,
                        "training_seed": training_seed,
                        "target_id": str(target["target_id"]),
                        "target_type": str(target["target_type"]),
                        "optimizer_seed": optimizer_seed,
                        "method": method,
                    }
                    run_config = _materialize_run_config(
                        base,
                        run_id=run_id,
                        method=method,
                        checkpoint=model_lookup.get(method),
                        optimizer_seed=optimizer_seed,
                        generator_seed_offset=int(config["generator_seed_offset"]),
                    )
                    run_config_path = output_dir / "run_configs" / f"{run_id}.yaml"
                    if not run_config_path.exists():
                        write_yaml(run_config_path, run_config)
                    run_dir = run_inverse_design(
                        target_path,
                        run_config_path,
                        repo_root=repo_root,
                        output_dir_override=output_dir / "runs" / run_id,
                        resume=resume,
                    )
                    metric, curve = _collect_run_metrics(run, run_dir)
                    run_rows.append({**run, "status": "completed", "run_dir": str(run_dir)})
                    metric_rows.append(metric)
                    curves.append(curve)
                    pd.DataFrame(run_rows).to_csv(output_dir / "execution_status.csv", index=False)

    metrics = pd.DataFrame(metric_rows)
    convergence = pd.concat(curves, ignore_index=True)
    paired = _paired_comparison(metrics)
    aggregate = _aggregate(metrics, paired)
    metrics.to_csv(output_dir / "run_metrics.csv", index=False)
    convergence.to_csv(output_dir / "convergence.csv", index=False)
    paired.to_csv(output_dir / "paired_comparison.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate.csv", index=False)
    (output_dir / "benchmark_report.md").write_text(
        _report(aggregate), encoding="utf-8"
    )
    finished = dt.datetime.now(dt.timezone.utc)
    write_yaml(
        output_dir / "manifest.yaml",
        {
            "status": "complete",
            "version": config["version"],
            "primary_metric": "generation_0_target_weighted_relative_error",
            "hypervolume_is_primary": False,
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
            "elapsed_seconds": (finished - started).total_seconds(),
            "run_count": len(metrics),
            "all_runs_completed": len(metrics) == len(run_rows),
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
        },
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run target-directed benchmark")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = run_target_benchmark(
        args.config,
        output_dir_override=args.output_dir,
        resume=args.resume,
    )
    manifest = yaml.safe_load((output / "manifest.yaml").read_text(encoding="utf-8"))
    print(json.dumps({"output": str(output), "status": manifest["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
