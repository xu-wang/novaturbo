"""Stage-II WP6/G3 v2 orchestration and failure-aware statistics.

The implementation deliberately reuses the frozen WP4, target-benchmark, and
WP5 delivery engines.  WP6 owns only the experiment matrix, resume semantics,
and the confirmatory aggregation needed by G3.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from .baseline import sha256_file, write_json, write_yaml
from .stage2_baseline import verify_manifest
from .stage2_wp5 import run_stage2_wp5
from .stage2_wp5_matrix import _summary_row
from .inverse_design import run_inverse_design
from .target_benchmark import (
    _aggregate as _benchmark_aggregate,
    _collect_run_metrics,
    _materialize_run_config,
    _paired_comparison as _benchmark_paired,
    _report as _benchmark_report,
    _run_id as _benchmark_run_id,
    load_target_benchmark_config,
    run_target_benchmark,
)
from .wp7 import _selected_targets, _target_spec
from .wp4 import run_wp4


ORDINARY_TYPES = ("ID", "Interpolation")
REJECTION_TYPE = "Infeasible"


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _read_yaml(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def load_stage2_wp6_config(path: Path) -> Dict[str, Any]:
    config = _read_yaml(Path(path))
    required = {
        "version", "frozen_inputs", "targets", "seeds", "model_sets",
        "generator_only", "end_to_end", "statistics", "output",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"S2-WP6 config is missing keys: {sorted(missing)}")
    if int(config["generator_only"]["n_raw"]) != 128:
        raise ValueError("S2-WP6 requires N_raw=128")
    if config["generator_only"]["candidate_protocol"] != "v2_1_fixed_raw":
        raise ValueError("S2-WP6 requires the V2.1 fixed-raw protocol")
    training = [int(value) for value in config["seeds"]["training"]]
    model_seeds = [int(value["training_seed"]) for value in config["model_sets"]]
    if training != model_seeds or len(training) != len(set(training)):
        raise ValueError("S2-WP6 model_sets must exactly match unique training seeds")
    required_models = {
        str(config["statistics"]["primary_method"]),
        str(config["statistics"]["frozen_generator"]),
    }
    for model_set in config["model_sets"]:
        models = {str(model["model_id"]) for model in model_set["models"]}
        if not required_models.issubset(models):
            raise ValueError(
                f"Training seed {model_set['training_seed']} lacks {sorted(required_models)}"
            )
    if not config["seeds"]["optimizer"]:
        raise ValueError("S2-WP6 optimizer seeds cannot be empty")
    if int(config["statistics"]["bootstrap_replicates"]) < 100:
        raise ValueError("S2-WP6 needs at least 100 bootstrap replicates")
    return config


def _load_frozen_targets(
    repo_root: Path, config: Mapping[str, Any], maximum_per_type: Optional[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    suite_path = _resolve(repo_root, str(config["targets"]["suite_csv"]))
    if sha256_file(suite_path) != str(config["targets"]["suite_sha256"]):
        raise ValueError("S2-WP6 target suite hash differs from the frozen value")
    suite = pd.read_csv(suite_path)
    frozen_path = _resolve(
        repo_root, str(config["targets"]["frozen_end_to_end_ids"])
    )
    frozen = _read_yaml(frozen_path)
    if frozen["source_target_suite_sha256"] != str(config["targets"]["suite_sha256"]):
        raise ValueError("Frozen end-to-end IDs refer to another target suite")
    ids = [str(value) for value in frozen["target_ids"]]
    selected = suite.loc[suite["target_id"].astype(str).isin(ids)].copy()
    order = {target_id: index for index, target_id in enumerate(ids)}
    selected["frozen_order"] = selected["target_id"].astype(str).map(order)
    selected = selected.sort_values("frozen_order", kind="stable").reset_index(drop=True)
    if selected["target_id"].astype(str).tolist() != ids:
        raise ValueError("S2-WP6 could not resolve all frozen end-to-end targets")
    if maximum_per_type is not None:
        suite = (
            suite.groupby("target_type", sort=False, group_keys=False)
            .head(int(maximum_per_type)).reset_index(drop=True)
        )
        selected = (
            selected.groupby("target_type", sort=False, group_keys=False)
            .head(int(maximum_per_type)).reset_index(drop=True)
        )
    return suite, selected


def _generator_only_config(
    config: Mapping[str, Any], model_set: Mapping[str, Any], output: Path,
    target_suite_path: Path, maximum_per_type: Optional[int],
) -> Dict[str, Any]:
    go = config["generator_only"]
    seed = int(model_set["training_seed"])
    return {
        "version": f"{config['version']}__generator_only_seed{seed}",
        "metrics_config": go["metrics_config"],
        "data": {
            "dataset_csv": go["dataset_csv"], "schema": go["schema"],
            "normalizer": go["normalizer"], "target_suite": str(target_suite_path),
        },
        "models": {
            str(model["model_id"]): str(model["checkpoint_path"])
            for model in model_set["models"]
        },
        "runtime": {
            "device": go["device"], "sample_count": int(go["n_raw"]),
            "sample_seed": int(config["seeds"]["generator_only_base"]) + seed,
            "maximum_targets_per_type": maximum_per_type,
            "sampling": {"policy": "v2_1_fixed_raw"},
        },
        "output": {"directory": str(output), "refuse_overwrite": True},
    }


def _benchmark_config(
    config: Mapping[str, Any], target_ids: Sequence[str], output: Path
) -> Dict[str, Any]:
    allowed = set(str(value) for value in config["end_to_end"]["benchmark_models"])
    model_sets = []
    for model_set in config["model_sets"]:
        models = [
            {"model_id": str(model["model_id"]), "checkpoint_path": str(model["checkpoint_path"])}
            for model in model_set["models"] if str(model["model_id"]) in allowed
        ]
        if not models:
            raise ValueError("Every S2-WP6 benchmark model set must be non-empty")
        model_sets.append({"training_seed": int(model_set["training_seed"]), "models": models})
    return {
        "version": f"{config['version']}__end_to_end_benchmark",
        "target_suite": {
            "csv": config["targets"]["suite_csv"],
            "sha256": config["targets"]["suite_sha256"],
            "selection": {"target_ids": list(target_ids)},
        },
        "optimizer_seeds": [int(value) for value in config["seeds"]["optimizer"]],
        "generator_seed_offset": int(config["seeds"]["end_to_end_generator_offset"]),
        "model_sets": model_sets,
        "base_run_config": config["end_to_end"]["base_run_config"],
        "output": {"directory": str(output), "refuse_overwrite": True},
    }


def _delivery_matrix(
    config: Mapping[str, Any], targets: pd.DataFrame
) -> pd.DataFrame:
    allowed = set(str(value) for value in config["end_to_end"]["delivery_models"])
    rows = []
    offset = int(config["seeds"]["end_to_end_generator_offset"])
    for model_set in config["model_sets"]:
        training_seed = int(model_set["training_seed"])
        for model in model_set["models"]:
            model_id = str(model["model_id"])
            if model_id not in allowed:
                continue
            for target in targets.to_dict(orient="records"):
                target_id = str(target["target_id"])
                for optimizer_seed_value in config["seeds"]["optimizer"]:
                    optimizer_seed = int(optimizer_seed_value)
                    rows.append({
                        "run_id": (
                            f"auto__{model_id}__train{training_seed}__opt{optimizer_seed}__"
                            f"{target_id[:12]}"
                        ),
                        "arm": "auto_delivery", "model": model_id,
                        "training_seed": training_seed, "optimizer_seed": optimizer_seed,
                        "target_id": target_id, "target_type": str(target["target_type"]),
                        "checkpoint_path": str(model["checkpoint_path"]),
                        "sample_seed": offset + optimizer_seed,
                    })
    matrix = pd.DataFrame(rows)
    if matrix.empty or matrix["run_id"].duplicated().any():
        raise ValueError("S2-WP6 delivery matrix is empty or has duplicate run IDs")
    return matrix


def _delivery_run_config(
    template: Mapping[str, Any], config: Mapping[str, Any], row: Mapping[str, Any]
) -> Dict[str, Any]:
    value = copy.deepcopy(dict(template))
    value["version"] = f"{config['version']}__{row['run_id']}"
    value["target"] = {
        "suite_csv": config["targets"]["suite_csv"],
        "target_id": str(row["target_id"]),
        "tolerance_profile": config["targets"]["tolerance_profile"],
    }
    value["model"] = {
        "model_id": str(row["model"]),
        "checkpoint_path": str(row["checkpoint_path"]),
    }
    value["generation"]["n_raw"] = 128
    value["generation"]["candidate_protocol"] = "v2_1_fixed_raw"
    value["generation"]["seed"] = int(row["sample_seed"])
    value["delivery"]["mode"] = "generator_auto_refinement"
    value["delivery"]["minimum_hit_count"] = int(
        config["end_to_end"]["minimum_hit_count"]
    )
    value["delivery"]["maximum_delivered_candidates"] = int(
        config["end_to_end"]["maximum_delivered_candidates"]
    )
    value["delivery"]["extra_engineering_objectives"] = False
    value["nsga2"]["seed"] = int(row["optimizer_seed"])
    value["nsga2"]["warm_start_seed"] = int(row["optimizer_seed"])
    value["runtime"]["device"] = config["generator_only"]["device"]
    value["output"] = {"directory": "unused", "refuse_overwrite": True}
    return value


def prepare_stage2_wp6(
    config_path: Path, *, repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None, resume: bool = False,
    maximum_targets_per_type: Optional[int] = None,
) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    config_path = Path(config_path).resolve()
    config = load_stage2_wp6_config(config_path)
    baseline = verify_manifest(
        _resolve(repo_root, config["frozen_inputs"]["baseline_manifest"]),
        repo_root=repo_root,
    )
    if not baseline["ok"]:
        raise RuntimeError(f"Frozen Stage-II baseline verification failed: {baseline['issues']}")
    for key in ("wp5_manifest", "cfm_default_manifest"):
        if not _resolve(repo_root, config["frozen_inputs"][key]).is_file():
            raise FileNotFoundError(f"Missing frozen input: {key}")
    suite, end_targets = _load_frozen_targets(repo_root, config, maximum_targets_per_type)
    output = (
        output_dir_override.resolve() if output_dir_override is not None
        else _resolve(repo_root, config["output"]["directory"])
    )
    manifest_path = output / "manifest.yaml"
    if output.exists():
        if not resume or not manifest_path.is_file():
            raise FileExistsError(f"S2-WP6 output already exists: {output}")
        manifest = _read_yaml(manifest_path)
        expected = {
            "version": config["version"], "config_sha256": sha256_file(config_path),
            "maximum_targets_per_type": maximum_targets_per_type,
        }
        actual = {key: manifest.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"S2-WP6 resume mismatch: expected={expected}, actual={actual}")
        return output

    output.mkdir(parents=True)
    generated = output / "generated_configs"
    generated.mkdir()
    (output / "generator_only").mkdir()
    (output / "end_to_end").mkdir()
    shutil.copy2(config_path, output / "build_config.yaml")
    target_suite_path = _resolve(repo_root, config["targets"]["suite_csv"])
    go_paths = []
    for model_set in config["model_sets"]:
        seed = int(model_set["training_seed"])
        path = generated / f"generator_only_seed{seed}.yaml"
        write_yaml(path, _generator_only_config(
            config, model_set, output / "generator_only" / f"seed_{seed}",
            target_suite_path, maximum_targets_per_type,
        ))
        go_paths.append(str(path))
    target_ids = end_targets["target_id"].astype(str).tolist()
    benchmark_path = generated / "end_to_end_benchmark.yaml"
    write_yaml(
        benchmark_path,
        _benchmark_config(config, target_ids, output / "end_to_end" / "benchmark"),
    )
    matrix = _delivery_matrix(config, end_targets)
    matrix_path = output / "end_to_end" / "delivery_run_matrix.csv"
    matrix.to_csv(matrix_path, index=False)
    write_yaml(manifest_path, {
        "version": config["version"], "status": "prepared",
        "prepared_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "maximum_targets_per_type": maximum_targets_per_type,
        "smoke": maximum_targets_per_type is not None,
        "baseline_id": baseline["baseline_id"],
        "generator_only_target_count": int(len(suite)),
        "end_to_end_target_count": int(len(end_targets)),
        "planned_generator_only_runs": int(len(suite) * sum(len(x["models"]) for x in config["model_sets"])),
        "planned_benchmark_runs": int(
            len(end_targets) * len(config["seeds"]["optimizer"])
            * sum(2 + len(x["models"]) for x in _read_yaml(benchmark_path)["model_sets"])
        ),
        "planned_delivery_runs": int(len(matrix)),
        "generator_only_configs": go_paths,
        "benchmark_config": str(benchmark_path),
        "delivery_matrix": str(matrix_path),
    })
    return output


def _update_manifest(output: Path, **values: Any) -> None:
    path = output / "manifest.yaml"
    manifest = _read_yaml(path)
    manifest.update(values)
    write_yaml(path, manifest)


def _run_inverse_design_task(task: Mapping[str, str]) -> str:
    run_dir = Path(task["run_dir"])
    run_inverse_design(
        Path(task["target_path"]), Path(task["run_config_path"]),
        repo_root=Path(task["repo_root"]), output_dir_override=run_dir,
        resume=run_dir.exists(),
    )
    return str(run_dir)


def _run_delivery_task(task: Mapping[str, str]) -> str:
    run_stage2_wp5(
        Path(task["run_config_path"]), repo_root=Path(task["repo_root"]),
        output_dir_override=Path(task["run_dir"]),
    )
    return task["run_dir"]


def _run_target_benchmark_parallel(
    config_path: Path, *, repo_root: Path, output_dir: Path, workers: int,
) -> Path:
    """Run independent target-benchmark runs concurrently, then aggregate once."""

    config = load_target_benchmark_config(config_path)
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
    base = _read_yaml(base_path)
    metrics_config = _read_yaml(_resolve(repo_root, base["data"]["metrics_config"]))
    tolerance_profile = str(metrics_config["metrics_version"])
    run_rows = []
    tasks = []
    for model_set in config["model_sets"]:
        training_seed = int(model_set["training_seed"])
        model_lookup = {str(model["model_id"]): model for model in model_set["models"]}
        methods = ["random", "screened_random", *model_lookup]
        for target_row in targets.to_dict(orient="records"):
            target = _target_spec(target_row, tolerance_profile)
            target_path = output_dir / "targets" / f"{target['target_id']}.yaml"
            if not target_path.exists():
                write_yaml(target_path, target)
            for optimizer_seed_value in config["optimizer_seeds"]:
                optimizer_seed = int(optimizer_seed_value)
                for method in methods:
                    run_id = _benchmark_run_id(
                        training_seed, str(target["target_id"]), optimizer_seed, method
                    )
                    run = {
                        "run_id": run_id, "training_seed": training_seed,
                        "target_id": str(target["target_id"]),
                        "target_type": str(target["target_type"]),
                        "optimizer_seed": optimizer_seed, "method": method,
                    }
                    run_config_path = output_dir / "run_configs" / f"{run_id}.yaml"
                    if not run_config_path.exists():
                        write_yaml(run_config_path, _materialize_run_config(
                            base, run_id=run_id, method=method,
                            checkpoint=model_lookup.get(method),
                            optimizer_seed=optimizer_seed,
                            generator_seed_offset=int(config["generator_seed_offset"]),
                        ))
                    run_dir = output_dir / "runs" / run_id
                    run_rows.append({**run, "status": "completed", "run_dir": str(run_dir)})
                    completed = False
                    manifest_path = run_dir / "manifest.yaml"
                    if manifest_path.is_file():
                        completed = _read_yaml(manifest_path).get("status") == "completed"
                    if not completed:
                        tasks.append({
                            "target_path": str(target_path),
                            "run_config_path": str(run_config_path),
                            "repo_root": str(repo_root), "run_dir": str(run_dir),
                        })
    pd.DataFrame(run_rows).to_csv(output_dir / "execution_status.csv", index=False)
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_inverse_design_task, task) for task in tasks]
            for future in as_completed(futures):
                future.result()
    metric_rows = []
    curves = []
    for run in run_rows:
        metric, curve = _collect_run_metrics(run, Path(run["run_dir"]))
        metric_rows.append(metric)
        curves.append(curve)
    metrics = pd.DataFrame(metric_rows)
    convergence = pd.concat(curves, ignore_index=True)
    paired = _benchmark_paired(metrics)
    aggregate = _benchmark_aggregate(metrics, paired)
    metrics.to_csv(output_dir / "run_metrics.csv", index=False)
    convergence.to_csv(output_dir / "convergence.csv", index=False)
    paired.to_csv(output_dir / "paired_comparison.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate.csv", index=False)
    (output_dir / "benchmark_report.md").write_text(
        _benchmark_report(aggregate), encoding="utf-8"
    )
    write_yaml(output_dir / "manifest.yaml", {
        "status": "complete", "version": config["version"],
        "primary_metric": "generation_0_target_weighted_relative_error",
        "hypervolume_is_primary": False,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_count": int(len(metrics)), "all_runs_completed": True,
        "parallel_workers": int(workers), "config": str(config_path),
        "config_sha256": sha256_file(config_path),
    })
    return output_dir


def run_generator_only(output: Path, repo_root: Path) -> None:
    manifest = _read_yaml(output / "manifest.yaml")
    for path_value in manifest["generator_only_configs"]:
        path = Path(path_value)
        run_config = _read_yaml(path)
        run_output = Path(run_config["output"]["directory"])
        if (run_output / "wp4_manifest.yaml").is_file():
            continue
        if run_output.exists():
            raise RuntimeError(f"Incomplete generator-only run needs inspection: {run_output}")
        run_wp4(path, repo_root=repo_root, output_dir_override=run_output)
    _update_manifest(output, generator_only_completed=True)


def run_end_to_end(output: Path, repo_root: Path, config: Mapping[str, Any]) -> None:
    manifest = _read_yaml(output / "manifest.yaml")
    benchmark_path = Path(manifest["benchmark_config"])
    benchmark_output = output / "end_to_end" / "benchmark"
    workers = int(config["end_to_end"].get("parallel_workers", 6))
    if workers > 1:
        _run_target_benchmark_parallel(
            benchmark_path, repo_root=repo_root, output_dir=benchmark_output,
            workers=workers,
        )
    else:
        run_target_benchmark(
            benchmark_path, repo_root=repo_root, output_dir_override=benchmark_output,
            resume=benchmark_output.exists(),
        )
    matrix = pd.read_csv(manifest["delivery_matrix"])
    template_path = _resolve(repo_root, config["end_to_end"]["delivery_template"])
    template = _read_yaml(template_path)
    generated = output / "generated_configs" / "delivery"
    runs = output / "end_to_end" / "delivery_runs"
    generated.mkdir(exist_ok=True)
    runs.mkdir(exist_ok=True)
    completed = 0
    tasks = []
    for row in matrix.to_dict(orient="records"):
        run_id = str(row["run_id"])
        run_dir = runs / run_id
        if (run_dir / "manifest.yaml").is_file():
            completed += 1
            continue
        if run_dir.exists():
            raise RuntimeError(f"Incomplete delivery run needs inspection: {run_dir}")
        run_config_path = generated / f"{run_id}.yaml"
        write_yaml(run_config_path, _delivery_run_config(template, config, row))
        tasks.append({
            "run_config_path": str(run_config_path), "repo_root": str(repo_root),
            "run_dir": str(run_dir),
        })
    if tasks and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_delivery_task, task) for task in tasks]
            for future in as_completed(futures):
                future.result()
                completed += 1
                _update_manifest(output, delivery_completed_run_count=completed)
    else:
        for task in tasks:
            _run_delivery_task(task)
            completed += 1
            _update_manifest(output, delivery_completed_run_count=completed)
    rows = [
        _summary_row(row, runs / str(row["run_id"]))
        for row in matrix.to_dict(orient="records")
    ]
    pd.DataFrame(rows).to_csv(output / "end_to_end" / "delivery_target_summary.csv", index=False)
    _update_manifest(
        output, end_to_end_completed=True, delivery_completed_run_count=completed
    )


def _quantile_failure_aware(values: pd.Series, quantile: float) -> float:
    array = pd.to_numeric(values, errors="coerce").fillna(math.inf).to_numpy(float)
    # An observed-order quantile preserves failed runs as +inf without the
    # undefined inf-inf interpolation used by NumPy's default linear method.
    return (
        float(np.quantile(array, quantile, method="inverted_cdf"))
        if len(array) else math.nan
    )


def _bootstrap_ci(
    values: Sequence[float], *, statistic: str, replicates: int, seed: int,
    confidence: float,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan, math.nan, math.nan
    function = np.median if statistic == "median" else np.mean
    estimate = float(function(array))
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        draws[index] = function(array[rng.integers(0, len(array), size=len(array))])
    alpha = (1.0 - confidence) / 2.0
    return estimate, float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate extra."""

    columns = [str(value) for value in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.6g}" if math.isfinite(float(value)) else str(value))
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _generator_only_aggregate(
    output: Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = []
    for seed in config["seeds"]["training"]:
        path = output / "generator_only" / f"seed_{int(seed)}" / "target_summary.csv"
        frame = pd.read_csv(path)
        frame.insert(0, "training_seed", int(seed))
        frames.append(frame)
    targets = pd.concat(frames, ignore_index=True)
    targets.to_csv(output / "generator_only_target_summary.csv", index=False)
    fixed_k = int(config["generator_only"]["fixed_diversity_k"])
    rows = []
    for (model, target_type), frame in targets.groupby(["model", "target_type"], sort=True):
        fixed = frame.loc[pd.to_numeric(frame["selected_count"], errors="coerce") >= fixed_k]
        rows.append({
            "model": model, "target_type": target_type, "run_count": int(len(frame)),
            "best_error_median": _quantile_failure_aware(frame["best_total_error"], 0.5),
            "best_error_p95": _quantile_failure_aware(frame["best_total_error"], 0.95),
            "target_hit_rate": float(frame["best_within_tolerance"].astype(bool).mean()),
            "physical_valid_rate": float((frame["N_valid"] / frame["N_raw"]).mean()),
            "fixed_k_available_rate": float(len(fixed) / max(len(frame), 1)),
            "fixed_k_diversity_mean": float(pd.to_numeric(
                fixed["mean_nearest_neighbour_distance"], errors="coerce"
            ).mean()),
            "accepted_rate": float(frame["accepted"].astype(bool).mean()),
            "rejected_rate": float(frame["rejected"].astype(bool).mean()),
            "false_accept_rate": float(frame["false_accept"].astype(bool).mean()),
            "network_draws_total": int(frame["network_draw_count"].sum()),
            "zero_d_evaluations_total": int(frame["actual_0d_evaluations"].sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "generator_only_summary.csv", index=False)
    primary = str(config["statistics"]["primary_method"])
    baseline = str(config["statistics"]["frozen_generator"])
    left = targets.loc[targets["model"] == primary].copy()
    right = targets.loc[targets["model"] == baseline].copy()
    keys = ["training_seed", "target_id", "target_type"]
    paired = left.merge(right, on=keys, suffixes=("_primary", "_baseline"), validate="one_to_one")
    p_error = pd.to_numeric(paired["best_total_error_primary"], errors="coerce").fillna(math.inf)
    b_error = pd.to_numeric(paired["best_total_error_baseline"], errors="coerce").fillna(math.inf)
    paired["primary_error_win"] = p_error < b_error
    paired["log10_error_ratio"] = np.where(
        np.isfinite(p_error) & np.isfinite(b_error) & (p_error > 0) & (b_error > 0),
        np.log10(p_error) - np.log10(b_error), np.nan,
    )
    paired["valid_rate_delta"] = (
        paired["N_valid_primary"] / paired["N_raw_primary"]
        - paired["N_valid_baseline"] / paired["N_raw_baseline"]
    )
    paired["diversity_ratio"] = (
        paired["mean_nearest_neighbour_distance_primary"]
        / paired["mean_nearest_neighbour_distance_baseline"]
    )
    paired.to_csv(output / "generator_only_paired.csv", index=False)
    return targets, summary, paired


def _delivery_aggregate(output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = pd.read_csv(output / "end_to_end" / "delivery_target_summary.csv")
    targets["hit"] = targets["final_hit_count"] > 0
    targets["failure_aware_first_hit"] = pd.to_numeric(
        targets["direct_first_hit_evaluation"], errors="coerce"
    )
    refined_hit = targets["failure_aware_first_hit"].isna() | ~np.isfinite(
        targets["failure_aware_first_hit"]
    )
    targets.loc[refined_hit & targets["hit"], "failure_aware_first_hit"] = targets.loc[
        refined_hit & targets["hit"], "total_0d"
    ]
    targets.loc[~targets["hit"], "failure_aware_first_hit"] = (
        targets.loc[~targets["hit"], "total_0d"] + 1
    )
    rows = []
    for (model, target_type), frame in targets.groupby(["model", "target_type"], sort=True):
        rows.append({
            "model": model, "target_type": target_type, "run_count": int(len(frame)),
            "direct_error_median": _quantile_failure_aware(frame["direct_best_error"], 0.5),
            "final_error_median": _quantile_failure_aware(frame["final_best_error"], 0.5),
            "final_error_p95": _quantile_failure_aware(frame["final_best_error"], 0.95),
            "final_hit_rate": float(frame["hit"].mean()),
            "first_hit_evaluations_median": float(frame["failure_aware_first_hit"].median()),
            "total_0d_median": float(frame["total_0d"].median()),
            "total_0d_p95": float(frame["total_0d"].quantile(0.95)),
            "physical_valid_rate": float((frame["N_valid"] / frame["N_raw"]).mean()),
            "gradient_activation_rate": float(frame["used_design_gradient"].astype(bool).mean()),
            "nsga2_activation_rate": float(frame["used_nsga2"].astype(bool).mean()),
            "failed_delivery_count": int((frame["delivered_count"] <= 0).sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "end_to_end" / "delivery_summary.csv", index=False)
    targets.to_csv(output / "end_to_end" / "delivery_target_summary.csv", index=False)
    return targets, summary


def _paired_end_to_end(
    delivery: pd.DataFrame, benchmark: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    primary = str(config["statistics"]["primary_method"])
    keys = ["training_seed", "target_id", "target_type", "optimizer_seed"]
    p = delivery.loc[delivery["model"] == primary].copy()
    rows = []
    for baseline_name in (
        str(config["statistics"]["frozen_generator"]),
        str(config["statistics"]["required_baseline"]), "random",
    ):
        if baseline_name == str(config["statistics"]["frozen_generator"]):
            base = delivery.loc[delivery["model"] == baseline_name].copy()
            merged = p.merge(base, on=keys, suffixes=("_primary", "_baseline"), validate="one_to_one")
            primary_error = pd.to_numeric(
                merged["final_best_error_primary"], errors="coerce"
            ).fillna(math.inf)
            primary_hit_eval = pd.to_numeric(
                merged["failure_aware_first_hit_primary"], errors="coerce"
            )
            primary_budget = pd.to_numeric(
                merged["total_0d_primary"], errors="coerce"
            )
            baseline_error = pd.to_numeric(merged["final_best_error_baseline"], errors="coerce").fillna(math.inf)
            baseline_hit_eval = pd.to_numeric(merged["failure_aware_first_hit_baseline"], errors="coerce")
            baseline_budget = pd.to_numeric(merged["total_0d_baseline"], errors="coerce")
        else:
            base = benchmark.loc[benchmark["method"] == baseline_name].copy()
            merged = p.merge(base, on=keys, suffixes=("_primary", "_baseline"), validate="one_to_one")
            primary_error = pd.to_numeric(
                merged["final_best_error"], errors="coerce"
            ).fillna(math.inf)
            primary_hit_eval = pd.to_numeric(
                merged["failure_aware_first_hit"], errors="coerce"
            )
            primary_budget = pd.to_numeric(merged["total_0d"], errors="coerce")
            baseline_error = pd.to_numeric(merged["search_best_error"], errors="coerce").fillna(math.inf)
            baseline_hit_eval = pd.to_numeric(merged["search_evaluations_to_hit"], errors="coerce")
            baseline_hit_eval = baseline_hit_eval + pd.to_numeric(merged["prefilter_evaluations"], errors="coerce").fillna(0)
            baseline_budget = pd.to_numeric(merged["total_evaluations"], errors="coerce")
        baseline_hit_eval = baseline_hit_eval.where(np.isfinite(baseline_hit_eval), baseline_budget + 1)
        for index in range(len(merged)):
            p_error = float(primary_error.iloc[index])
            b_error = float(baseline_error.iloc[index])
            record = {key: merged.iloc[index][key] for key in keys}
            record.update({
                "baseline": baseline_name,
                "primary_final_error": p_error, "baseline_final_error": b_error,
                "primary_error_win": bool(p_error < b_error),
                "log10_error_ratio": (
                    math.log10(p_error) - math.log10(b_error)
                    if math.isfinite(p_error) and math.isfinite(b_error) and p_error > 0 and b_error > 0
                    else math.nan
                ),
                "first_hit_evaluation_delta": float(primary_hit_eval.iloc[index] - baseline_hit_eval.iloc[index]),
                "primary_total_0d": int(primary_budget.iloc[index]),
                "baseline_total_0d": int(baseline_budget.iloc[index]),
            })
            rows.append(record)
    return pd.DataFrame(rows)


def aggregate_stage2_wp6(output: Path, config: Mapping[str, Any]) -> Dict[str, Any]:
    _, go_summary, go_paired = _generator_only_aggregate(output, config)
    delivery, delivery_summary = _delivery_aggregate(output)
    benchmark = pd.read_csv(output / "end_to_end" / "benchmark" / "run_metrics.csv")
    paired = _paired_end_to_end(delivery, benchmark, config)
    paired.to_csv(output / "end_to_end" / "paired_comparison.csv", index=False)
    stats = config["statistics"]
    reps = int(stats["bootstrap_replicates"])
    seed = int(config["seeds"]["bootstrap"])
    confidence = float(stats["confidence_level"])
    ordinary_go = go_paired.loc[go_paired["target_type"].isin(ORDINARY_TYPES)]
    go_error = _bootstrap_ci(
        ordinary_go["log10_error_ratio"], statistic="median", replicates=reps,
        seed=seed, confidence=confidence,
    )
    go_valid = _bootstrap_ci(
        ordinary_go["valid_rate_delta"], statistic="mean", replicates=reps,
        seed=seed + 1, confidence=confidence,
    )
    go_diversity = _bootstrap_ci(
        ordinary_go["diversity_ratio"], statistic="median", replicates=reps,
        seed=seed + 2, confidence=confidence,
    )
    e2e_rows = []
    for baseline_name, frame in paired.loc[paired["target_type"].isin(ORDINARY_TYPES)].groupby("baseline", sort=True):
        error = _bootstrap_ci(
            frame["log10_error_ratio"], statistic="median", replicates=reps,
            seed=seed + 10 + len(e2e_rows), confidence=confidence,
        )
        first_hit = _bootstrap_ci(
            frame["first_hit_evaluation_delta"], statistic="median", replicates=reps,
            seed=seed + 20 + len(e2e_rows), confidence=confidence,
        )
        e2e_rows.append({
            "primary_method": stats["primary_method"], "baseline": baseline_name,
            "pair_count": int(len(frame)), "primary_error_win_rate": float(frame["primary_error_win"].mean()),
            "median_log10_error_ratio": error[0], "median_log10_error_ratio_ci_low": error[1],
            "median_log10_error_ratio_ci_high": error[2],
            "median_first_hit_evaluation_delta": first_hit[0],
            "median_first_hit_evaluation_delta_ci_low": first_hit[1],
            "median_first_hit_evaluation_delta_ci_high": first_hit[2],
        })
    e2e_summary = pd.DataFrame(e2e_rows)
    e2e_summary.to_csv(output / "end_to_end" / "paired_summary.csv", index=False)
    frozen_name = str(stats["frozen_generator"])
    screened_name = str(stats["required_baseline"])
    frozen_row = e2e_summary.loc[e2e_summary["baseline"] == frozen_name].iloc[0]
    screened_row = e2e_summary.loc[e2e_summary["baseline"] == screened_name].iloc[0]
    advantage_frozen = bool(
        float(frozen_row["median_log10_error_ratio_ci_high"]) < 0
        or float(frozen_row["median_first_hit_evaluation_delta_ci_high"]) < 0
    )
    advantage_screened = bool(
        float(screened_row["median_log10_error_ratio_ci_high"]) < 0
        or float(screened_row["median_first_hit_evaluation_delta_ci_high"]) < 0
    )
    infeasible = go_summary.loc[
        (go_summary["model"] == stats["primary_method"])
        & (go_summary["target_type"] == REJECTION_TYPE)
    ]
    false_accept = float(infeasible.iloc[0]["false_accept_rate"]) if len(infeasible) else math.nan
    seed_directions = []
    for _, frame in ordinary_go.groupby("training_seed", sort=True):
        finite = pd.to_numeric(frame["log10_error_ratio"], errors="coerce").dropna()
        seed_directions.append(float(finite.median()) < 0 if len(finite) else False)
    gate = {
        "gate": "G3 Method Validation Gate v2",
        "passed": bool(
            advantage_frozen and advantage_screened and go_valid[1] > 0
            and go_diversity[0] >= float(stats["diversity_noninferiority_ratio"])
            and false_accept <= float(stats["infeasible_false_accept_maximum"])
            and all(seed_directions)
        ),
        "primary_method": stats["primary_method"],
        "frozen_generator": frozen_name,
        "screened_random": screened_name,
        "ordinary_generator_only_median_log10_error_ratio": go_error[0],
        "ordinary_generator_only_error_ratio_ci": [go_error[1], go_error[2]],
        "ordinary_valid_rate_delta": go_valid[0],
        "ordinary_valid_rate_delta_ci": [go_valid[1], go_valid[2]],
        "ordinary_diversity_ratio": go_diversity[0],
        "ordinary_diversity_ratio_ci": [go_diversity[1], go_diversity[2]],
        "infeasible_false_accept_rate": false_accept,
        "stable_error_improvement_each_training_seed": all(seed_directions),
        "advantage_vs_frozen_generator": advantage_frozen,
        "advantage_vs_screened_random": advantage_screened,
        "all_failed_runs_retained": True,
        "hypervolume_is_gate_metric": False,
    }
    write_json(output / "g3_gate.json", gate)
    report = [
        "# S2-WP6 / G3 v2 Method Validation Report", "",
        f"- Gate: {'PASS' if gate['passed'] else 'FAIL'}",
        f"- Primary method: {stats['primary_method']}",
        f"- Frozen Generator: {frozen_name}",
        f"- Generator-only ordinary median log10 error ratio: {go_error[0]:.6g} "
        f"(95% CI {go_error[1]:.6g}, {go_error[2]:.6g}; negative favors primary)",
        f"- Ordinary physical-valid-rate delta: {go_valid[0]:.2%} "
        f"(95% CI {go_valid[1]:.2%}, {go_valid[2]:.2%})",
        f"- Fixed-K diversity ratio: {go_diversity[0]:.4f} "
        f"(95% CI {go_diversity[1]:.4f}, {go_diversity[2]:.4f})",
        f"- Infeasible false-accept rate: {false_accept:.2%}", "",
        "## End-to-end paired comparisons", "",
        _markdown_table(e2e_summary), "",
        "All failures remain in target-level statistics. OOD-feasible-unknown targets are reported as acceptance/rejection diagnostics, not false accepts.",
    ]
    (output / "g3_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _update_manifest(
        output, status="completed", completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        generator_only_completed=True, end_to_end_completed=True, aggregated=True,
        g3_passed=gate["passed"],
    )
    return gate


def run_stage2_wp6(
    config_path: Path, *, repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None, resume: bool = False,
    maximum_targets_per_type: Optional[int] = None,
    run_generator_only_stage: bool = False, run_end_to_end_stage: bool = False,
    aggregate: bool = False,
) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    config = load_stage2_wp6_config(Path(config_path))
    output = prepare_stage2_wp6(
        Path(config_path), repo_root=repo_root, output_dir_override=output_dir_override,
        resume=resume, maximum_targets_per_type=maximum_targets_per_type,
    )
    if run_generator_only_stage:
        run_generator_only(output, repo_root)
    if run_end_to_end_stage:
        run_end_to_end(output, repo_root, config)
    if aggregate:
        manifest = _read_yaml(output / "manifest.yaml")
        if not manifest.get("generator_only_completed") or not manifest.get("end_to_end_completed"):
            raise RuntimeError("Cannot aggregate S2-WP6 before both experiment layers complete")
        aggregate_stage2_wp6(output, config)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/stage2/s2_wp6_g3_v2.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--generator-only", action="store_true")
    parser.add_argument("--end-to-end", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--maximum-targets-per-type", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_all = bool(args.all)
    output = run_stage2_wp6(
        args.config, output_dir_override=args.output_dir, resume=args.resume,
        maximum_targets_per_type=args.maximum_targets_per_type,
        run_generator_only_stage=run_all or args.generator_only,
        run_end_to_end_stage=run_all or args.end_to_end,
        aggregate=(run_all or args.aggregate) and not args.prepare_only,
    )
    manifest = _read_yaml(output / "manifest.yaml")
    print(json.dumps({"output": str(output), "status": manifest["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
