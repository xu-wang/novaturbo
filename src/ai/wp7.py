"""WP7 B0 baseline benchmark and stage-wise diagnosis.

B0 is deliberately a benchmark, not a performance gate.  The runner creates
paired Random/Hybrid WP6 jobs, derives the no-NSGA A/C arms from their frozen
stage artifacts, and reports both NSGA-only and end-to-end evaluation budgets.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/novaturbo-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist

from .baseline import approximate_hypervolume, sha256_file, write_json, write_yaml
from .inverse_design import load_inverse_design_config, run_inverse_design


BUDGET_SCOPES = ("nsga_only", "end_to_end")
REQUIRED_TARGET_COLUMNS = {
    "target_id",
    "target_type",
    "expected_status",
    "thrust_N",
    "tsfc_kg_N_s",
}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_wp7_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "version",
        "target_suite",
        "seeds",
        "models",
        "base_run_config",
        "analysis",
        "output",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"WP7 config is missing keys: {sorted(missing)}")
    if not config["seeds"]:
        raise ValueError("WP7 requires at least one seed")
    model_ids = [str(value["model_id"]) for value in config["models"]]
    if len(model_ids) != len(set(model_ids)) or not model_ids:
        raise ValueError("WP7 models must be non-empty and have unique model_id values")
    reference = list(config["analysis"]["normalized_reference_point"])
    if not reference or any(float(value) <= 0.0 for value in reference):
        raise ValueError("analysis.normalized_reference_point must be positive")
    return config


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _selected_targets(suite: pd.DataFrame, selection: Mapping[str, Any]) -> pd.DataFrame:
    missing = REQUIRED_TARGET_COLUMNS.difference(suite.columns)
    if missing:
        raise ValueError(f"Target suite is missing columns: {sorted(missing)}")
    explicit = selection.get("target_ids")
    if explicit:
        target_ids = [str(value) for value in explicit]
        indexed = suite.set_index("target_id", drop=False)
        absent = sorted(set(target_ids).difference(indexed.index.astype(str)))
        if absent:
            raise ValueError(f"Configured target_ids are absent from suite: {absent}")
        selected = indexed.loc[target_ids].reset_index(drop=True)
    else:
        counts = {str(key): int(value) for key, value in selection["counts"].items()}
        parts = []
        for target_type, count in counts.items():
            available = suite.loc[suite["target_type"] == target_type]
            if len(available) < count:
                raise ValueError(
                    f"Target suite has {len(available)} {target_type} rows, needs {count}"
                )
            parts.append(available.sort_values("target_id", kind="stable").head(count))
        selected = pd.concat(parts, ignore_index=True)
    if selected["target_id"].duplicated().any():
        raise ValueError("WP7 target selection contains duplicate target_id values")
    return selected


def _target_spec(row: Mapping[str, Any], tolerance_profile: str) -> Dict[str, Any]:
    target = {
        "target_version": "wp7_b0_target_v1",
        "target_id": str(row["target_id"]),
        "target_type": str(row["target_type"]),
        "expected_status": str(row["expected_status"]),
        "thrust_N": float(row["thrust_N"]),
        "tsfc_kg_N_s": float(row["tsfc_kg_N_s"]),
        "tolerance_profile": tolerance_profile,
    }
    if row.get("total_mass_kg") is not None and not pd.isna(row["total_mass_kg"]):
        target["total_mass_kg"] = float(row["total_mass_kg"])
    return target


def _run_id(target_id: str, seed: int, model_id: Optional[str]) -> str:
    method = "random" if model_id is None else (
        "screened_random" if model_id == "screened_random" else f"{model_id}_hybrid"
    )
    return f"b0_{target_id[:10]}_s{seed}_{method}"


def build_run_matrix(
    config: Mapping[str, Any],
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Build physical Random, optional screened-Random, and Generator runs."""

    rows: List[Dict[str, Any]] = []
    for target in targets.to_dict(orient="records"):
        for seed_value in config["seeds"]:
            seed = int(seed_value)
            rows.append(
                {
                    "run_id": _run_id(str(target["target_id"]), seed, None),
                    "arm": "B",
                    "mode": "random",
                    "model_id": "random",
                    "target_id": str(target["target_id"]),
                    "target_type": str(target["target_type"]),
                    "seed": seed,
                }
            )
            if bool(config.get("screened_random_control", False)):
                rows.append(
                    {
                        "run_id": _run_id(
                            str(target["target_id"]), seed, "screened_random"
                        ),
                        "arm": "S",
                        "mode": "screened_random",
                        "model_id": "screened_random",
                        "target_id": str(target["target_id"]),
                        "target_type": str(target["target_type"]),
                        "seed": seed,
                    }
                )
            for model in config["models"]:
                model_id = str(model["model_id"])
                rows.append(
                    {
                        "run_id": _run_id(str(target["target_id"]), seed, model_id),
                        "arm": "D",
                        "mode": "hybrid",
                        "model_id": model_id,
                        "target_id": str(target["target_id"]),
                        "target_type": str(target["target_type"]),
                        "seed": seed,
                    }
                )
    return pd.DataFrame(rows)


def build_experiment_matrix(run_matrix: pd.DataFrame) -> pd.DataFrame:
    """Expand physical B/D runs into logical A/B/C/D experiment arms."""

    rows: List[Dict[str, Any]] = []
    for run in run_matrix.to_dict(orient="records"):
        if run["arm"] == "B":
            arms = (
                ("A", "random_only", "generation_0"),
                ("B", "random_nsga2", "complete_optimization"),
            )
        elif run["arm"] == "S":
            arms = (
                ("E", "screened_random_only", "generation_round_trip"),
                ("F", "screened_random_nsga2", "complete_optimization"),
            )
        else:
            arms = (
                ("C", f"{run['model_id']}_generator_only", "generation_round_trip"),
                ("D", f"{run['model_id']}_hybrid", "complete_optimization"),
            )
        for arm, method, source_stage in arms:
            rows.append(
                {
                    **run,
                    "logical_arm": arm,
                    "method": method,
                    "source_stage": source_stage,
                }
            )
    return pd.DataFrame(rows)


def _materialize_run_config(
    base: Mapping[str, Any],
    matrix_row: Mapping[str, Any],
    model_lookup: Mapping[str, Mapping[str, Any]],
    generator_seed_offset: int,
) -> Dict[str, Any]:
    value = copy.deepcopy(dict(base))
    seed = int(matrix_row["seed"])
    value["run_id"] = str(matrix_row["run_id"])
    value["mode"] = str(matrix_row["mode"])
    value["seeds"] = {
        "generator": int(generator_seed_offset) + seed,
        "optimizer": seed,
        "background": seed,
    }
    value["optimizer"]["seed"] = seed
    if value["mode"] == "random":
        value["model"] = {"model_id": None, "checkpoint_path": None}
        value["population"]["generator_ratio"] = 0.0
    elif value["mode"] == "screened_random":
        value["model"] = {"model_id": None, "checkpoint_path": None}
    else:
        model = model_lookup[str(matrix_row["model_id"])]
        value["model"] = {
            "model_id": str(model["model_id"]),
            "checkpoint_path": str(model["checkpoint_path"]),
        }
    value["output"]["refuse_overwrite"] = True
    return value


def _fairness_payload(run_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Hash everything that must be paired apart from the initializer itself."""

    return {
        "data": run_config["data"],
        "optimizer": run_config["optimizer"],
        "background_method": run_config["population"]["background_method"],
        "duplicate_tolerance": run_config["population"]["duplicate_tolerance"],
        "optimizer_seed": run_config["seeds"]["optimizer"],
        "background_seed": run_config["seeds"]["background"],
    }


def _screening_control_payload(run_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Freeze candidate, selector, mixture, and seed settings for D-vs-S."""

    return {
        "generation": run_config["generation"],
        "selection": run_config["selection"],
        "generator_ratio": run_config["population"]["generator_ratio"],
        "preference": run_config["population"].get("preference", {}),
        "generator_seed": run_config["seeds"]["generator"],
        "background_method": run_config["population"]["background_method"],
        "duplicate_tolerance": run_config["population"]["duplicate_tolerance"],
    }


def _nondominated(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) <= 1:
        return points.copy()
    points = np.unique(points, axis=0)
    order = np.argsort(np.sum(points, axis=1), kind="stable")
    front: List[np.ndarray] = []
    for index in order:
        point = points[index]
        if front and any(np.all(prior <= point) for prior in front):
            continue
        front = [prior for prior in front if not np.all(point <= prior)]
        front.append(point)
    return np.asarray(front, dtype=float)


def _objective_columns(frame: pd.DataFrame) -> List[str]:
    columns = [value for value in frame.columns if value.startswith("objective_internal_")]
    return sorted(columns, key=lambda value: int(value.rsplit("_", 1)[1]))


def _objective_scaling(
    reference_csv: Path,
    objective_names: Sequence[str],
    maximize: Sequence[bool],
) -> Tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(reference_csv)
    if "is_valid" in frame:
        frame = frame.loc[frame["is_valid"].astype(bool)]
    values = []
    for name, is_maximized in zip(objective_names, maximize):
        column = frame[name].to_numpy(dtype=float)
        values.append(-column if is_maximized else column)
    matrix = np.column_stack(values)
    finite = matrix[np.all(np.isfinite(matrix), axis=1)]
    if finite.size == 0:
        raise ValueError("Objective reference dataset has no finite valid rows")
    return np.min(finite, axis=0), np.max(finite, axis=0)


def _target_error_columns(
    ledger: pd.DataFrame,
    target: Mapping[str, Any],
    derived: Mapping[str, Any],
) -> pd.DataFrame:
    result = ledger.copy()
    thrust = result["performance__thrust_N"].to_numpy(dtype=float)
    tsfc = result["performance__tsfc_kg_N_s"].to_numpy(dtype=float)
    target_thrust = float(target["thrust_N"])
    target_tsfc = float(target["tsfc_kg_N_s"])
    eps = derived["epsilons"]
    weights = derived["weights"]
    error_f = np.abs(thrust - target_thrust) / max(abs(target_thrust), float(eps["thrust_N"]))
    error_t = np.abs(tsfc - target_tsfc) / max(abs(target_tsfc), float(eps["tsfc_kg_N_s"]))
    has_mass_target = (
        target.get("total_mass_kg") is not None
        and "performance__total_mass_kg" in result
    )
    if has_mass_target:
        mass = result["performance__total_mass_kg"].to_numpy(dtype=float)
        target_mass = float(target["total_mass_kg"])
        error_m = np.abs(mass - target_mass) / max(
            abs(target_mass), float(eps.get("total_mass_kg", 1.0e-15))
        )
    else:
        error_m = np.zeros(len(result), dtype=float)
    feasible = result["is_feasible"].astype(bool).to_numpy()
    violation = (~feasible).astype(float)
    total = (
        float(weights["thrust"]) * error_f
        + float(weights["tsfc"]) * error_t
        + float(weights.get("mass", 0.0)) * error_m
        + float(weights["constraint_violation"]) * violation
    )
    tolerances = derived["success_tolerances"]
    mass_within = (
        error_m <= float(tolerances.get("mass_relative", math.inf))
        if has_mass_target
        else np.ones(len(result), dtype=bool)
    )
    within = (
        feasible
        & (error_f <= float(tolerances["thrust_relative"]))
        & (error_t <= float(tolerances["tsfc_relative"]))
        & mass_within
        & (total <= float(tolerances["weighted_total"]))
    )
    result["thrust_relative_error"] = error_f
    result["tsfc_relative_error"] = error_t
    result["mass_relative_error"] = error_m
    result["constraint_violation"] = violation
    result["total_error"] = total
    result["within_tolerance"] = within
    return result


def _igd(points: np.ndarray, reference_front: np.ndarray) -> float:
    if len(points) == 0 or len(reference_front) == 0:
        return math.nan
    return float(np.mean(np.min(cdist(reference_front, points), axis=1)))


def _first_success_evaluation(frame: pd.DataFrame) -> float:
    successful = frame.loc[frame["within_tolerance"].astype(bool)]
    if successful.empty:
        return math.nan
    return float(successful["evaluation_index"].min() + 1)


def _curve_rows(
    run: Mapping[str, Any],
    ledger: pd.DataFrame,
    objective_min: np.ndarray,
    objective_max: np.ndarray,
    reference_point: np.ndarray,
    reference_front: np.ndarray,
    prefilter_count: int,
    prefilter_first_success: float,
    prefilter_best_error: float,
    sobol_power: int,
    prefilter_wall_seconds: float,
    optimizer_wall_seconds: float,
    total_nsga_evaluations: int,
) -> List[Dict[str, Any]]:
    objective_columns = _objective_columns(ledger)
    scale = np.maximum(objective_max - objective_min, 1.0e-12)
    rows: List[Dict[str, Any]] = []
    for generation in sorted(ledger["generation"].astype(int).unique()):
        current = ledger.loc[ledger["generation"].astype(int) <= generation]
        feasible = current.loc[current["is_feasible"].astype(bool)]
        points = feasible[objective_columns].to_numpy(dtype=float)
        front = _nondominated(points)
        normalized_front = (front - objective_min) / scale if len(front) else front
        hv = approximate_hypervolume(
            front,
            objective_min,
            objective_max,
            reference_point,
            sobol_power,
        )["value"]
        best_error = float(current["total_error"].min()) if len(current) else math.nan
        mean_violation = float(current["constraint_violation"].mean()) if len(current) else math.nan
        nsga_success = _first_success_evaluation(current)
        nsga_count = len(current)
        for budget_scope in BUDGET_SCOPES:
            is_e2e = budget_scope == "end_to_end"
            total_count = nsga_count + (prefilter_count if is_e2e else 0)
            nsga_wall = optimizer_wall_seconds * nsga_count / max(total_nsga_evaluations, 1)
            wall_seconds = nsga_wall + (prefilter_wall_seconds if is_e2e else 0.0)
            if is_e2e and math.isfinite(prefilter_first_success):
                evaluations_to_threshold = prefilter_first_success
            elif math.isfinite(nsga_success):
                evaluations_to_threshold = nsga_success + (prefilter_count if is_e2e else 0)
            else:
                evaluations_to_threshold = math.nan
            effective_best = (
                min(best_error, prefilter_best_error)
                if is_e2e and math.isfinite(prefilter_best_error)
                else best_error
            )
            if run["arm"] == "B":
                logical_arm = "B"
                method = "random_nsga2"
            elif run["arm"] == "S":
                logical_arm = "F"
                method = "screened_random_nsga2"
            else:
                logical_arm = "D"
                method = f"{run['model_id']}_hybrid"
            rows.append(
                {
                    **run,
                    "logical_arm": logical_arm,
                    "method": method,
                    "budget_scope": budget_scope,
                    "generation": int(generation),
                    "nsga2_evaluations": int(nsga_count),
                    "prefilter_evaluations": int(prefilter_count if is_e2e else 0),
                    "evaluation_count": int(total_count),
                    "hypervolume": float(hv),
                    "igd": _igd(normalized_front, reference_front),
                    "pareto_size": int(len(front)),
                    "feasible_rate": float(feasible.shape[0] / max(len(current), 1)),
                    "mean_constraint_violation": mean_violation,
                    "best_total_error": float(effective_best),
                    "within_tolerance": bool(math.isfinite(effective_best) and (
                        current["within_tolerance"].any()
                        or (is_e2e and math.isfinite(prefilter_first_success))
                    )),
                    "evaluations_to_threshold": evaluations_to_threshold,
                    "wall_seconds": float(wall_seconds),
                }
            )
    return rows


def _generator_stage_rows(run: Mapping[str, Any], run_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if run["arm"] not in {"D", "S"}:
        return [], []
    forward = pd.read_csv(run_dir / "round_trip" / "forward_results.csv")
    ranked = pd.read_csv(run_dir / "round_trip" / "ranked_candidates.csv")
    metrics = json.loads((run_dir / "round_trip" / "metrics.json").read_text(encoding="utf-8"))
    sampling = json.loads((run_dir / "generator" / "sampling_metrics.json").read_text(encoding="utf-8"))
    abstention = metrics.get("abstention", {})
    valid = forward.loc[forward.get("forward_valid", pd.Series(False, index=forward.index)).astype(bool)]
    selected = ranked.dropna(subset=["selected_rank"]) if "selected_rank" in ranked else ranked.iloc[0:0]

    is_screened_random = run["arm"] == "S"
    logical_arm = "E" if is_screened_random else "C"
    method = (
        "screened_random_only"
        if is_screened_random
        else f"{run['model_id']}_generator_only"
    )

    def values(frame: pd.DataFrame, stage: str) -> Dict[str, Any]:
        best = float(frame["total_error"].min()) if len(frame) and "total_error" in frame else math.nan
        success = float(frame["within_tolerance"].astype(bool).mean()) if len(frame) and "within_tolerance" in frame else 0.0
        return {
            **run,
            "logical_arm": logical_arm,
            "method": method,
            "stage": stage,
            "candidate_count": int(len(frame)),
            "valid_rate": float(len(valid) / max(len(forward), 1)),
            "success_rate": success,
            "best_total_error": best,
            "duplicate_rate": float(metrics.get("duplicate_rate", math.nan)),
            "mean_nearest_neighbour_distance": float(
                metrics.get("diversity", {}).get("mean_nearest_neighbour_distance", math.nan)
            ),
            "coverage_mean": float(np.mean(list(
                metrics.get("diversity", {}).get("coverage_fraction_by_dimension", {}).values()
            ))) if metrics.get("diversity", {}).get("coverage_fraction_by_dimension") else math.nan,
            "prefilter_evaluations": int(sampling.get("prefilter_0d_evaluations", 0)),
            "ode_nfe": int(sampling.get("ode_nfe", 0)),
            "abstention_accepted": bool(abstention.get("accepted", False)),
            "abstention_rejected": bool(abstention.get("rejected", False)),
            "false_accept": bool(abstention.get("false_accept", False)),
            "low_confidence": bool(abstention.get("low_confidence", False)),
        }

    prefix = "screened_random" if is_screened_random else "generator"
    stages = [values(valid, f"{prefix}_raw_valid"), values(selected, f"{prefix}_selected")]
    top_k = []
    for requested_k, item in metrics.get("top_k", {}).items():
        top_k.append(
            {
                **run,
                "logical_arm": logical_arm,
                "method": method,
                "requested_k": int(requested_k),
                "effective_k": int(item["effective_k"]),
                "success": bool(item["success"]),
                "best_total_error": float(item["best_total_error"]),
            }
        )
    return stages, top_k


def _run_metadata(run: Mapping[str, Any], run_dir: Path) -> Dict[str, Any]:
    metrics = json.loads((run_dir / "summary" / "metrics.json").read_text(encoding="utf-8"))
    return {
        **run,
        "run_dir": str(run_dir),
        "status": "completed",
        "prefilter_evaluations": int(metrics["cost"]["prefilter_0d_evaluations"]),
        "preference_evaluations": int(
            metrics["cost"].get("preference_19d_evaluations", 0)
        ),
        "initialization_evaluations": int(
            metrics["cost"].get(
                "initialization_0d_evaluations",
                metrics["cost"]["prefilter_0d_evaluations"],
            )
        ),
        "nsga2_evaluations": int(metrics["cost"]["nsga2_0d_evaluations"]),
        "total_evaluations": int(metrics["cost"]["total_0d_evaluations"]),
        "wall_seconds": float(metrics["cost"]["current_execution_wall_seconds"]),
        "generator_sampling_seconds": float(metrics["cost"]["generator_sampling_seconds"]),
        "round_trip_seconds": float(metrics["cost"]["round_trip_seconds"]),
        "adapter_seconds": float(metrics["cost"].get("adapter_seconds", 0.0)),
        "optimizer_seconds": float(metrics["cost"]["optimizer_seconds"]),
    }


def _aggregate(benchmark: pd.DataFrame, ci_z: float) -> pd.DataFrame:
    keys = ["logical_arm", "method", "model_id", "target_type", "budget_scope", "generation"]
    metrics = [
        "evaluation_count",
        "hypervolume",
        "igd",
        "pareto_size",
        "feasible_rate",
        "mean_constraint_violation",
        "best_total_error",
        "evaluations_to_threshold",
        "wall_seconds",
    ]
    rows: List[Dict[str, Any]] = []
    for key, group in benchmark.groupby(keys, dropna=False):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row["n"] = int(len(group))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else math.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else math.nan
            row[f"{metric}_ci95"] = (
                float(ci_z * values.std(ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else 0.0 if len(values) else math.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _matched_comparison(benchmark: pd.DataFrame) -> pd.DataFrame:
    rows = []
    random = benchmark.loc[benchmark["logical_arm"] == "B"]
    for run_id, hybrid_run in benchmark.loc[benchmark["logical_arm"] == "D"].groupby("run_id"):
        first = hybrid_run.iloc[0]
        paired_random = random.loc[
            (random["target_id"] == first["target_id"])
            & (random["seed"] == first["seed"])
        ]
        for budget_scope in BUDGET_SCOPES:
            hybrid_scope = hybrid_run.loc[hybrid_run["budget_scope"] == budget_scope]
            random_scope = paired_random.loc[paired_random["budget_scope"] == budget_scope]
            if hybrid_scope.empty or random_scope.empty:
                continue
            common_budget = int(
                min(hybrid_scope["evaluation_count"].max(), random_scope["evaluation_count"].max())
            )
            hybrid_eligible = hybrid_scope.loc[hybrid_scope["evaluation_count"] <= common_budget]
            random_eligible = random_scope.loc[random_scope["evaluation_count"] <= common_budget]
            if hybrid_eligible.empty or random_eligible.empty:
                continue
            hybrid = hybrid_eligible.sort_values("evaluation_count", kind="stable").iloc[-1]
            baseline = random_eligible.sort_values("evaluation_count", kind="stable").iloc[-1]
            rows.append(
                {
                    "target_id": hybrid["target_id"],
                    "target_type": hybrid["target_type"],
                    "seed": hybrid["seed"],
                    "model_id": hybrid["model_id"],
                    "budget_scope": budget_scope,
                    "common_budget": common_budget,
                    "hybrid_evaluation_count": int(hybrid["evaluation_count"]),
                    "random_evaluation_count": int(baseline["evaluation_count"]),
                    "hybrid_hypervolume": hybrid["hypervolume"],
                    "random_hypervolume": baseline["hypervolume"],
                    "hypervolume_delta": hybrid["hypervolume"] - baseline["hypervolume"],
                    "hybrid_best_total_error": hybrid["best_total_error"],
                    "random_best_total_error": baseline["best_total_error"],
                    "best_error_improvement": baseline["best_total_error"] - hybrid["best_total_error"],
                }
            )
    columns = [
        "target_id",
        "target_type",
        "seed",
        "model_id",
        "budget_scope",
        "common_budget",
        "hybrid_evaluation_count",
        "random_evaluation_count",
        "hybrid_hypervolume",
        "random_hypervolume",
        "hypervolume_delta",
        "hybrid_best_total_error",
        "random_best_total_error",
        "best_error_improvement",
    ]
    return pd.DataFrame(rows, columns=columns)


def _screening_control_comparison(benchmark: pd.DataFrame) -> pd.DataFrame:
    """Compare Generator+screening directly with Random+the same screening."""

    columns = [
        "target_id",
        "target_type",
        "seed",
        "model_id",
        "budget_scope",
        "common_budget",
        "generator_evaluation_count",
        "screened_random_evaluation_count",
        "generator_hypervolume",
        "screened_random_hypervolume",
        "hypervolume_delta",
        "generator_best_total_error",
        "screened_random_best_total_error",
        "best_error_improvement",
    ]
    screened = benchmark.loc[benchmark["logical_arm"] == "F"]
    if screened.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, Any]] = []
    for _, generator_run in benchmark.loc[
        benchmark["logical_arm"] == "D"
    ].groupby("run_id"):
        first = generator_run.iloc[0]
        paired = screened.loc[
            (screened["target_id"] == first["target_id"])
            & (screened["seed"] == first["seed"])
        ]
        for budget_scope in BUDGET_SCOPES:
            generator_scope = generator_run.loc[
                generator_run["budget_scope"] == budget_scope
            ]
            screened_scope = paired.loc[paired["budget_scope"] == budget_scope]
            if generator_scope.empty or screened_scope.empty:
                continue
            common_budget = int(
                min(
                    generator_scope["evaluation_count"].max(),
                    screened_scope["evaluation_count"].max(),
                )
            )
            generator_eligible = generator_scope.loc[
                generator_scope["evaluation_count"] <= common_budget
            ]
            screened_eligible = screened_scope.loc[
                screened_scope["evaluation_count"] <= common_budget
            ]
            if generator_eligible.empty or screened_eligible.empty:
                continue
            generator_row = generator_eligible.sort_values(
                "evaluation_count", kind="stable"
            ).iloc[-1]
            screened_row = screened_eligible.sort_values(
                "evaluation_count", kind="stable"
            ).iloc[-1]
            rows.append(
                {
                    "target_id": generator_row["target_id"],
                    "target_type": generator_row["target_type"],
                    "seed": generator_row["seed"],
                    "model_id": generator_row["model_id"],
                    "budget_scope": budget_scope,
                    "common_budget": common_budget,
                    "generator_evaluation_count": int(
                        generator_row["evaluation_count"]
                    ),
                    "screened_random_evaluation_count": int(
                        screened_row["evaluation_count"]
                    ),
                    "generator_hypervolume": generator_row["hypervolume"],
                    "screened_random_hypervolume": screened_row["hypervolume"],
                    "hypervolume_delta": (
                        generator_row["hypervolume"]
                        - screened_row["hypervolume"]
                    ),
                    "generator_best_total_error": generator_row["best_total_error"],
                    "screened_random_best_total_error": screened_row[
                        "best_total_error"
                    ],
                    "best_error_improvement": (
                        screened_row["best_total_error"]
                        - generator_row["best_total_error"]
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _failure_breakdown(run_matrix: pd.DataFrame, run_root: Path) -> pd.DataFrame:
    rows = []
    for run in run_matrix.to_dict(orient="records"):
        run_dir = run_root / str(run["run_id"])
        failure_path = run_dir / "round_trip" / "failures.csv"
        if not failure_path.is_file():
            continue
        failures = pd.read_csv(failure_path)
        if failures.empty or "failure_code" not in failures:
            continue
        for failure_code, count in failures["failure_code"].fillna("UNKNOWN").value_counts().items():
            rows.append({**run, "failure_code": str(failure_code), "count": int(count)})
    if not rows:
        grouped = pd.DataFrame(columns=["model_id", "target_type", "failure_code", "count"])
    else:
        raw = pd.DataFrame(rows)
        grouped = raw.groupby(
            ["model_id", "target_type", "failure_code"], as_index=False, dropna=False
        )["count"].sum()
    execution_path = run_root.parent / "execution_status.csv"
    if execution_path.is_file():
        execution = pd.read_csv(execution_path)
        failed = execution.loc[execution["status"] == "failed"]
        if len(failed):
            failed_rows = []
            for row in failed.to_dict(orient="records"):
                error_type = str(row.get("error", "UNKNOWN")).split(":", 1)[0]
                failed_rows.append(
                    {
                        "model_id": row["model_id"],
                        "target_type": row["target_type"],
                        "failure_code": f"RUN_FAILED__{error_type}",
                        "count": 1,
                    }
                )
            grouped = pd.concat([grouped, pd.DataFrame(failed_rows)], ignore_index=True)
            grouped = grouped.groupby(
                ["model_id", "target_type", "failure_code"], as_index=False, dropna=False
            )["count"].sum()
    return grouped


def _diagnosis_tables(
    benchmark: pd.DataFrame,
    stage: pd.DataFrame,
    matched: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    hypotheses = []
    initial = benchmark.loc[
        (benchmark["budget_scope"] == "nsga_only") & (benchmark["generation"] == 0)
    ]
    random_initial = initial.loc[initial["logical_arm"] == "B"].set_index(["target_id", "seed"])
    initial_differences = []
    for value in initial.loc[initial["logical_arm"] == "D"].to_dict(orient="records"):
        key = (value["target_id"], value["seed"])
        if key in random_initial.index:
            baseline = random_initial.loc[key]
            if isinstance(baseline, pd.DataFrame):
                baseline = baseline.iloc[0]
            initial_differences.append(
                {
                    "model_id": value["model_id"],
                    "target_type": value["target_type"],
                    "initial_hv_delta": value["hypervolume"] - baseline["hypervolume"],
                }
            )
    initial_differences = pd.DataFrame(initial_differences)
    groups = benchmark.loc[benchmark["logical_arm"] == "D", ["model_id", "target_type"]].drop_duplicates()
    for model_id, target_type in groups.sort_values(["model_id", "target_type"]).itertuples(index=False):
        model_stage = stage.loc[
            (stage["model_id"] == model_id) & (stage["target_type"] == target_type)
        ]
        raw = model_stage.loc[model_stage["stage"] == "generator_raw_valid"]
        selected = model_stage.loc[model_stage["stage"] == "generator_selected"]
        hybrid_initial = model_stage.loc[model_stage["stage"] == "hybrid_initial"]
        matched_model = matched.loc[
            (matched["model_id"] == model_id)
            & (matched["target_type"] == target_type)
            & (matched["budget_scope"] == "end_to_end")
        ]
        matched_nsga = matched.loc[
            (matched["model_id"] == model_id)
            & (matched["target_type"] == target_type)
            & (matched["budget_scope"] == "nsga_only")
        ]
        valid_rate = float(raw["valid_rate"].mean()) if len(raw) else math.nan
        success_rate = float(raw["success_rate"].mean()) if len(raw) else math.nan
        duplicate_rate = float(raw["duplicate_rate"].mean()) if len(raw) else math.nan
        coverage = float(raw["coverage_mean"].mean()) if len(raw) else math.nan
        hv_delta = float(matched_model["hypervolume_delta"].mean()) if len(matched_model) else math.nan
        final_nsga_delta = float(matched_nsga["hypervolume_delta"].mean()) if len(matched_nsga) else math.nan
        error_gain = float(matched_model["best_error_improvement"].mean()) if len(matched_model) else math.nan
        initial_group = initial_differences.loc[
            (initial_differences["model_id"] == model_id)
            & (initial_differences["target_type"] == target_type)
        ] if not initial_differences.empty else initial_differences
        initial_hv_delta = float(initial_group["initial_hv_delta"].mean()) if len(initial_group) else math.nan
        selected_error = float(selected["best_total_error"].mean()) if len(selected) else math.nan
        initial_error = float(hybrid_initial["best_total_error"].mean()) if len(hybrid_initial) else math.nan
        network_state = (
            "tolerance_hits_observed"
            if success_rate >= 0.01
            else "low_validity_and_sparse_hits"
            if valid_rate < 0.5
            else "few_or_no_tolerance_hits"
        )
        diversity_state = "possible_diversity_loss" if duplicate_rate > 0.1 or coverage < 0.2 else "no_obvious_diversity_failure"
        mapping_state = (
            "quality_diluted_after_mapping"
            if math.isfinite(selected_error)
            and math.isfinite(initial_error)
            and initial_error > 1.5 * max(selected_error, 1.0e-12)
            else "no_clear_mapping_dilution"
        )
        alignment_state = (
            "target_error_improves_but_hv_declines"
            if error_gain > 1.0e-9 and hv_delta < 0
            else "aligned_advantage"
            if error_gain > 1.0e-9 and hv_delta > 0
            else "no_joint_advantage"
        )
        optimization_state = (
            "initial_advantage_erased"
            if initial_hv_delta > 0 and final_nsga_delta <= 0
            else "advantage_emerges_during_search"
            if initial_hv_delta <= 0 and final_nsga_delta > 0
            else "no_initial_hv_advantage"
            if initial_hv_delta <= 0
            else "advantage_retained"
        )
        evidence = (
            f"valid_rate={valid_rate:.4g}, success_rate={success_rate:.4g}, "
            f"duplicate_rate={duplicate_rate:.4g}, coverage={coverage:.4g}, "
            f"selected_error={selected_error:.4g}, initial_error={initial_error:.4g}, "
            f"initial_hv_delta={initial_hv_delta:.4g}, final_nsga_hv_delta={final_nsga_delta:.4g}, "
            f"end_to_end_hv_delta={hv_delta:.4g}, best_error_improvement={error_gain:.4g}"
        )
        rows.extend(
            [
                {"model_id": model_id, "target_type": target_type, "layer": "generator", "assessment": network_state, "evidence": evidence},
                {"model_id": model_id, "target_type": target_type, "layer": "screening", "assessment": diversity_state, "evidence": evidence},
                {"model_id": model_id, "target_type": target_type, "layer": "5D_to_19D", "assessment": mapping_state, "evidence": evidence},
                {"model_id": model_id, "target_type": target_type, "layer": "target_alignment", "assessment": alignment_state, "evidence": evidence},
                {"model_id": model_id, "target_type": target_type, "layer": "NSGA-II", "assessment": optimization_state, "evidence": evidence},
            ]
        )
        target_key = str(target_type).lower().replace("-", "_")
        if network_state != "tolerance_hits_observed":
            hypotheses.append(
                {
                    "model_id": model_id,
                    "target_type": target_type,
                    "hypothesis_id": f"H_{model_id}_{target_key}_conditional_fidelity",
                    "evidence": evidence,
                    "falsifiable_hypothesis": "Improving conditional fidelity raises round-trip tolerance success at fixed candidate count.",
                    "controlled_test": "Keep data/targets/seeds fixed and ablate condition encoding or training loss.",
                }
            )
        if diversity_state == "possible_diversity_loss":
            hypotheses.append(
                {
                    "model_id": model_id,
                    "target_type": target_type,
                    "hypothesis_id": f"H_{model_id}_{target_key}_diversity",
                    "evidence": evidence,
                    "falsifiable_hypothesis": "Diversity-aware selection improves initial coverage without worsening Top-K error.",
                    "controlled_test": "Compare frozen error ranking with max-min/DPP selection at equal prefilter budget.",
                }
            )
        if mapping_state == "quality_diluted_after_mapping" or alignment_state == "target_error_improves_but_hv_declines":
            hypotheses.append(
                {
                    "model_id": model_id,
                    "target_type": target_type,
                    "hypothesis_id": f"H_{model_id}_{target_key}_interface",
                    "evidence": evidence,
                    "falsifiable_hypothesis": "The 5D-to-19D adapter or objective mismatch dilutes the generated prior.",
                    "controlled_test": "Ablate background variables and compare generation-0 versus final HV at equal budgets.",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(hypotheses)


def _write_diagnosis(path: Path, diagnosis: pd.DataFrame, matched: pd.DataFrame) -> None:
    lines = [
        "# WP7 B0 基线对照与问题诊断",
        "",
        "> B0 只记录现有冻结方法的表现，不设置 Generator 必须优于 Random 的通过条件。",
        "",
        "## 分层诊断",
        "",
        "| 模型 | 目标类型 | 环节 | 诊断 | 证据 |",
        "|---|---|---|---|---|",
    ]
    for row in diagnosis.to_dict(orient="records"):
        lines.append(
            f"| {row['model_id']} | {row['target_type']} | {row['layer']} | "
            f"{row['assessment']} | {row['evidence']} |"
        )
    lines.extend(["", "## 配对对照摘要", ""])
    if matched.empty:
        lines.append("没有足够的成对完成运行可供比较。")
    else:
        grouped = matched.groupby(["model_id", "budget_scope"])[
            ["hypervolume_delta", "best_error_improvement"]
        ].mean()
        lines.append("| 模型 | 预算 | 平均 HV 差值 | 平均目标误差改善 |")
        lines.append("|---|---|---:|---:|")
        for (model, budget), row in grouped.iterrows():
            lines.append(
                f"| {model} | {budget} | {row['hypervolume_delta']:.6g} | "
                f"{row['best_error_improvement']:.6g} |"
            )
    lines.extend(
        [
            "",
            "## 结论规则",
            "",
            "Generator/Hybrid 优于、等于或低于 Random 均是有效 B0 结果。"
            "本报告用于确定阶段 II 应优先修改网络、筛选、映射、目标对齐还是优化接口，"
            "不得据此把第一阶段判为 No-Go。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_hypotheses(path: Path, hypotheses: pd.DataFrame) -> None:
    lines = [
        "# 阶段 II 可证伪研究假设",
        "",
        "这些假设由 B0 证据生成，不对第一阶段冻结基线做任何回写或调参。",
        "",
    ]
    if hypotheses.empty:
        lines.append("当前 B0 没有触发预设瓶颈规则；应扩大 target/seed 后重新诊断。")
    else:
        for row in hypotheses.to_dict(orient="records"):
            lines.extend(
                [
                    f"## {row['hypothesis_id']}",
                    "",
                    f"- 模型：{row['model_id']}",
                    f"- 目标类型：{row['target_type']}",
                    f"- 证据：{row['evidence']}",
                    f"- 假设：{row['falsifiable_hypothesis']}",
                    f"- 对照实验：{row['controlled_test']}",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_curves(aggregate: pd.DataFrame, top_k: pd.DataFrame, stage: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for scope in BUDGET_SCOPES:
        current = aggregate.loc[aggregate["budget_scope"] == scope]
        for metric, ylabel, filename in (
            ("hypervolume_mean", "Hypervolume", f"hypervolume_{scope}.png"),
            ("best_total_error_mean", "Best target error", f"best_error_{scope}.png"),
        ):
            fig, axis = plt.subplots(figsize=(7.5, 5.0))
            for method, group in current.groupby("method"):
                grouped = group.groupby("evaluation_count_mean", as_index=False)[metric].mean().sort_values("evaluation_count_mean")
                axis.plot(grouped["evaluation_count_mean"], grouped[metric], marker="o", label=method)
            axis.set_xlabel("0D evaluation count")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.3)
            if len(current):
                axis.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(output / filename, dpi=160)
            plt.close(fig)
    if not top_k.empty:
        fig, axis = plt.subplots(figsize=(7.5, 5.0))
        values = top_k.groupby(["model_id", "requested_k"], as_index=False)["success"].mean()
        for model, group in values.groupby("model_id"):
            axis.plot(group["requested_k"], group["success"], marker="o", label=model)
        axis.set_xlabel("K")
        axis.set_ylabel("Top-K success rate")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output / "top_k_success.png", dpi=160)
        plt.close(fig)
    raw = stage.loc[stage["stage"] == "generator_raw_valid"] if not stage.empty else stage
    if len(raw):
        fig, axis = plt.subplots(figsize=(7.5, 5.0))
        labels, samples = [], []
        for model, group in raw.groupby("model_id"):
            values = group["mean_nearest_neighbour_distance"].dropna().to_numpy()
            if len(values):
                labels.append(model)
                samples.append(values)
        if samples:
            axis.boxplot(samples, tick_labels=labels)
            axis.set_ylabel("Mean nearest-neighbour distance")
            axis.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(output / "generator_diversity.png", dpi=160)
        plt.close(fig)


def _pareto_points(ledgers: Mapping[str, pd.DataFrame], run_matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by_id = run_matrix.set_index("run_id")
    for run_id, ledger in ledgers.items():
        feasible = ledger.loc[ledger["is_feasible"].astype(bool)]
        objective_columns = _objective_columns(feasible)
        if feasible.empty:
            continue
        front_values = _nondominated(feasible[objective_columns].to_numpy(dtype=float))
        for point in front_values:
            matches = np.all(np.isclose(feasible[objective_columns].to_numpy(dtype=float), point), axis=1)
            source = feasible.loc[matches].iloc[0]
            metadata = by_id.loc[run_id]
            rows.append(
                {
                    "run_id": run_id,
                    "model_id": metadata["model_id"],
                    "target_id": metadata["target_id"],
                    "target_type": metadata["target_type"],
                    "seed": metadata["seed"],
                    "thrust_N": source["performance__thrust_N"],
                    "tsfc_kg_N_s": source["performance__tsfc_kg_N_s"],
                }
            )
    return pd.DataFrame(rows)


def _plot_pareto(points: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 5.0))
    if not points.empty:
        first_target = points["target_id"].iloc[0]
        current = points.loc[points["target_id"] == first_target]
        for model, group in current.groupby("model_id"):
            axis.scatter(group["tsfc_kg_N_s"], group["thrust_N"], s=24, alpha=0.7, label=model)
        axis.legend(fontsize=7)
        axis.set_title(f"Target {str(first_target)[:10]}")
    axis.set_xlabel("TSFC [kg/(N s)]")
    axis.set_ylabel("Thrust [N]")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def analyze_wp7(
    output_dir: Path,
    config: Mapping[str, Any],
    run_matrix: pd.DataFrame,
    repo_root: Path,
) -> Dict[str, Any]:
    (output_dir / "comparison_curves").mkdir(parents=True, exist_ok=True)
    run_root = output_dir / "runs"
    completed = []
    ledgers: Dict[str, pd.DataFrame] = {}
    run_dirs: Dict[str, Path] = {}
    for run in run_matrix.to_dict(orient="records"):
        run_dir = run_root / str(run["run_id"])
        manifest_path = run_dir / "manifest.yaml"
        if not manifest_path.is_file():
            continue
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            continue
        target = yaml.safe_load((run_dir / "target.yaml").read_text(encoding="utf-8"))
        derived = json.loads((run_dir / "derived_metrics.json").read_text(encoding="utf-8"))
        ledger = pd.read_csv(run_dir / "optimization" / "evaluation_ledger.csv")
        ledgers[str(run["run_id"])] = _target_error_columns(ledger, target, derived)
        run_dirs[str(run["run_id"])] = run_dir
        completed.append(_run_metadata(run, run_dir))
    if not completed:
        raise RuntimeError("WP7 has no completed runs to analyze")

    base = load_inverse_design_config(_resolve(repo_root, str(config["base_run_config"])))
    analysis = config["analysis"]
    objective_names = list(base["optimizer"]["objectives"])
    maximize = [bool(value) for value in base["optimizer"]["maximize"]]
    objective_min, objective_max = _objective_scaling(
        _resolve(repo_root, str(analysis["objective_reference_dataset"])),
        objective_names,
        maximize,
    )
    scale = np.maximum(objective_max - objective_min, 1.0e-12)
    final_fronts = []
    for ledger in ledgers.values():
        feasible = ledger.loc[ledger["is_feasible"].astype(bool)]
        points = feasible[_objective_columns(ledger)].to_numpy(dtype=float)
        final_fronts.append(_nondominated(points))
    nonempty_fronts = [value for value in final_fronts if len(value)]
    if not nonempty_fronts:
        raise RuntimeError("WP7 completed runs contain no feasible objective vectors")
    pooled = np.vstack(nonempty_fronts)
    reference_front = (_nondominated(pooled) - objective_min) / scale
    pd.DataFrame(reference_front, columns=[f"normalized_objective_{i}" for i in range(reference_front.shape[1])]).to_csv(
        output_dir / "empirical_reference_front.csv", index=False
    )

    benchmark_rows: List[Dict[str, Any]] = []
    stage_rows: List[Dict[str, Any]] = []
    top_k_rows: List[Dict[str, Any]] = []
    reference_point = np.asarray(analysis["normalized_reference_point"], dtype=float)
    for run in run_matrix.to_dict(orient="records"):
        run_id = str(run["run_id"])
        if run_id not in ledgers:
            continue
        ledger = ledgers[run_id]
        run_dir = run_dirs[run_id]
        sampling = json.loads((run_dir / "generator" / "sampling_metrics.json").read_text(encoding="utf-8"))
        summary_metrics = json.loads((run_dir / "summary" / "metrics.json").read_text(encoding="utf-8"))
        prefilter_count = int(sampling.get("prefilter_0d_evaluations", 0))
        initialization_count = int(
            summary_metrics["cost"].get(
                "initialization_0d_evaluations", prefilter_count
            )
        )
        prefilter_first = math.nan
        prefilter_best = math.nan
        if run["arm"] in {"D", "S"}:
            forward = pd.read_csv(run_dir / "round_trip" / "forward_results.csv")
            valid_forward = forward.loc[forward["forward_valid"].astype(bool)] if len(forward) else forward
            if len(valid_forward):
                prefilter_best = float(valid_forward["total_error"].min())
                successful = forward.loc[
                    forward["forward_valid"].astype(bool)
                    & forward["within_tolerance"].astype(bool)
                ]
                if len(successful):
                    first_index = int(successful.index.min())
                    prefilter_first = float(
                        forward.loc[:first_index, "forward_valid"].astype(bool).sum()
                    )
        benchmark_rows.extend(
            _curve_rows(
                run,
                ledger,
                objective_min,
                objective_max,
                reference_point,
                reference_front,
                initialization_count,
                prefilter_first,
                prefilter_best,
                int(analysis["hypervolume_sobol_power"]),
                float(summary_metrics["cost"].get("generator_sampling_seconds", 0.0))
                + float(summary_metrics["cost"].get("round_trip_seconds", 0.0))
                + float(summary_metrics["cost"].get("adapter_seconds", 0.0)),
                float(summary_metrics["cost"].get("optimizer_seconds", 0.0)),
                int(summary_metrics["cost"]["nsga2_0d_evaluations"]),
            )
        )
        initial = ledger.loc[ledger["generation"].astype(int) == 0]
        if run["arm"] == "B":
            initial_logical_arm = "A"
            initial_method = "random_only"
            initial_stage = "random_initial"
        elif run["arm"] == "S":
            initial_logical_arm = "F"
            initial_method = "screened_random_nsga2"
            initial_stage = "screened_random_initial"
        else:
            initial_logical_arm = "D"
            initial_method = f"{run['model_id']}_hybrid"
            initial_stage = "hybrid_initial"
        stage_rows.append(
            {
                **run,
                "logical_arm": initial_logical_arm,
                "method": initial_method,
                "stage": initial_stage,
                "candidate_count": int(len(initial)),
                "valid_rate": float(initial["is_feasible"].astype(bool).mean()),
                "success_rate": float(initial["within_tolerance"].astype(bool).mean()),
                "best_total_error": float(initial["total_error"].min()),
                "duplicate_rate": math.nan,
                "mean_nearest_neighbour_distance": math.nan,
                "coverage_mean": math.nan,
                "prefilter_evaluations": 0,
                "ode_nfe": 0,
                "abstention_accepted": False,
                "abstention_rejected": False,
                "false_accept": False,
                "low_confidence": False,
            }
        )
        generated_stages, generated_top_k = _generator_stage_rows(run, run_dir)
        stage_rows.extend(generated_stages)
        top_k_rows.extend(generated_top_k)

    benchmark = pd.DataFrame(benchmark_rows)
    stage = pd.DataFrame(stage_rows)
    top_k = pd.DataFrame(top_k_rows)
    aggregate = _aggregate(benchmark, float(analysis.get("confidence_z", 1.96)))
    # Keep optimizer checkpoints in the stage attribution table as well as the
    # budget curves so a single file can localize where an advantage appears.
    for row in benchmark.loc[benchmark["budget_scope"] == "nsga_only"].to_dict(orient="records"):
        stage_rows.append(
            {
                **{key: row[key] for key in run_matrix.columns if key in row},
                "logical_arm": row["logical_arm"],
                "method": row["method"],
                "stage": f"optimization_generation_{int(row['generation'])}",
                "candidate_count": int(row["nsga2_evaluations"]),
                "valid_rate": float(row["feasible_rate"]),
                "success_rate": float(row["within_tolerance"]),
                "best_total_error": float(row["best_total_error"]),
                "hypervolume": float(row["hypervolume"]),
                "igd": float(row["igd"]),
                "pareto_size": int(row["pareto_size"]),
                "prefilter_evaluations": 0,
                "ode_nfe": 0,
            }
        )
    stage = pd.DataFrame(stage_rows)
    matched = _matched_comparison(benchmark)
    screening_control = _screening_control_comparison(benchmark)
    failures = _failure_breakdown(run_matrix, run_root)
    diagnosis, hypotheses = _diagnosis_tables(benchmark, stage, matched)
    robustness = (
        stage.loc[stage["logical_arm"] == "C"]
        .groupby(["model_id", "target_type"], as_index=False)
        .agg(
            run_stage_count=("run_id", "count"),
            valid_rate=("valid_rate", "mean"),
            success_rate=("success_rate", "mean"),
            rejection_rate=("abstention_rejected", "mean"),
            false_accept_rate=("false_accept", "mean"),
            low_confidence_rate=("low_confidence", "mean"),
        )
    )
    pareto_points = _pareto_points(ledgers, run_matrix)

    pd.DataFrame(completed).to_csv(output_dir / "run_summary.csv", index=False)
    benchmark.to_csv(output_dir / "baseline_benchmark.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_benchmark.csv", index=False)
    stage.to_csv(output_dir / "stage_attribution.csv", index=False)
    top_k.to_csv(output_dir / "top_k_metrics.csv", index=False)
    matched.to_csv(output_dir / "matched_comparison.csv", index=False)
    screening_control.to_csv(
        output_dir / "screening_control_comparison.csv", index=False
    )
    failures.to_csv(output_dir / "failure_breakdown.csv", index=False)
    robustness.to_csv(output_dir / "robustness_report.csv", index=False)
    pareto_points.to_csv(output_dir / "comparison_curves" / "pareto_front.csv", index=False)
    diagnosis.to_csv(output_dir / "diagnosis_table.csv", index=False)
    hypotheses.to_csv(output_dir / "stage2_hypotheses.csv", index=False)
    _write_diagnosis(output_dir / "baseline_diagnosis.md", diagnosis, matched)
    _write_hypotheses(output_dir / "stage2_hypotheses.md", hypotheses)
    _plot_curves(aggregate, top_k, stage, output_dir / "comparison_curves")
    _plot_pareto(pareto_points, output_dir / "comparison_curves" / "pareto_front.png")

    summary = {
        "benchmark": "B0 Baseline Benchmark",
        "benchmark_is_gate": False,
        "generator_superiority_required": False,
        "completed_run_count": len(completed),
        "planned_run_count": int(len(run_matrix)),
        "all_planned_runs_completed": len(completed) == len(run_matrix),
        "logical_arms": sorted(
            set(build_experiment_matrix(run_matrix)["logical_arm"].astype(str))
        ),
        "screened_random_control_enabled": bool(
            config.get("screened_random_control", False)
        ),
        "screening_control_comparison_count": int(len(screening_control)),
        "budget_scopes": list(BUDGET_SCOPES),
        "empirical_reference_front_size": int(len(reference_front)),
        "objective_normalization": {
            "names": objective_names,
            "minimum_internal": objective_min.tolist(),
            "maximum_internal": objective_max.tolist(),
            "reference_point_normalized": reference_point.tolist(),
            "reference_dataset": str(analysis["objective_reference_dataset"]),
            "reference_dataset_sha256": sha256_file(
                _resolve(repo_root, str(analysis["objective_reference_dataset"]))
            ),
        },
    }
    write_json(output_dir / "b0_summary.json", summary)
    return summary


def run_wp7_benchmark(
    config_path: Path,
    *,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
    resume: bool = False,
    analyze_only: bool = False,
) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    config_path = Path(config_path).resolve()
    config = load_wp7_config(config_path)
    target_suite_path = _resolve(repo_root, str(config["target_suite"]["csv"]))
    configured_hash = str(config["target_suite"].get("sha256", ""))
    if configured_hash and sha256_file(target_suite_path) != configured_hash:
        raise ValueError("WP7 target suite SHA256 differs from frozen config")
    base_path = _resolve(repo_root, str(config["base_run_config"]))
    base = load_inverse_design_config(base_path)
    model_lookup = {str(value["model_id"]): value for value in config["models"]}
    for model in model_lookup.values():
        checkpoint = _resolve(repo_root, str(model["checkpoint_path"]))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"WP7 model checkpoint is missing: {checkpoint}")
    targets = _selected_targets(pd.read_csv(target_suite_path), config["target_suite"]["selection"])
    run_matrix = build_run_matrix(config, targets)
    output_dir = (
        output_dir_override.resolve()
        if output_dir_override is not None
        else _resolve(repo_root, str(config["output"]["directory"]))
    )
    if output_dir.exists():
        if not resume and bool(config["output"].get("refuse_overwrite", True)):
            raise FileExistsError(f"WP7 output exists: {output_dir}")
        frozen_config = output_dir / "benchmark_config.yaml"
        if frozen_config.is_file() and sha256_file(frozen_config) != sha256_file(config_path):
            raise ValueError("Cannot resume WP7 with a changed benchmark config")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("inputs/targets", "inputs/runs", "runs", "comparison_curves"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    if not (output_dir / "benchmark_config.yaml").exists():
        shutil.copy2(config_path, output_dir / "benchmark_config.yaml")
    run_matrix.to_csv(output_dir / "run_matrix.csv", index=False)
    build_experiment_matrix(run_matrix).to_csv(output_dir / "experiment_matrix.csv", index=False)
    targets.to_csv(output_dir / "selected_targets.csv", index=False)

    target_paths = {}
    metrics_config_path = _resolve(repo_root, str(base["data"]["metrics_config"]))
    metrics_config = yaml.safe_load(metrics_config_path.read_text(encoding="utf-8"))
    tolerance_profile = str(metrics_config["metrics_version"])
    for target in targets.to_dict(orient="records"):
        path = output_dir / "inputs" / "targets" / f"{target['target_id']}.yaml"
        write_yaml(path, _target_spec(target, tolerance_profile))
        target_paths[str(target["target_id"])] = path

    generated_config_paths = {}
    fairness_rows = []
    for run in run_matrix.to_dict(orient="records"):
        run_config = _materialize_run_config(
            base,
            run,
            model_lookup,
            int(config.get("generator_seed_offset", 20250771)),
        )
        path = output_dir / "inputs" / "runs" / f"{run['run_id']}.yaml"
        write_yaml(path, run_config)
        load_inverse_design_config(path)
        generated_config_paths[str(run["run_id"])] = path
        fairness_rows.append(
            {
                **run,
                "fairness_hash": _canonical_hash(_fairness_payload(run_config)),
                "screening_control_hash": (
                    _canonical_hash(_screening_control_payload(run_config))
                    if run["arm"] in {"D", "S"}
                    else ""
                ),
            }
        )
    fairness = pd.DataFrame(fairness_rows)
    pair_counts = fairness.groupby(["target_id", "seed"])["fairness_hash"].nunique()
    fairness_passed = bool((pair_counts == 1).all())
    screened_groups = fairness.loc[fairness["arm"].isin(["D", "S"])]
    screened_pair_counts = screened_groups.groupby(
        ["target_id", "seed"]
    )["screening_control_hash"].nunique()
    screening_control_fairness_passed = bool(
        screened_pair_counts.empty or (screened_pair_counts == 1).all()
    )
    fairness_passed = fairness_passed and screening_control_fairness_passed
    fairness.to_csv(output_dir / "fairness_matrix.csv", index=False)
    write_json(
        output_dir / "fairness_checks.json",
        {
            "passed": fairness_passed,
            "initializer_excluded_from_hash": True,
            "paired_target_seed_groups": int(len(pair_counts)),
            "mismatched_groups": int((pair_counts != 1).sum()),
            "screening_control_fairness_passed": screening_control_fairness_passed,
            "screening_control_mismatched_groups": int(
                (screened_pair_counts != 1).sum()
            ),
        },
    )
    if not fairness_passed:
        raise RuntimeError("WP7 fairness hash differs within paired target/seed groups")

    started = dt.datetime.now(dt.timezone.utc)
    execution_rows = []
    if not analyze_only:
        for run in run_matrix.to_dict(orient="records"):
            run_dir = output_dir / "runs" / str(run["run_id"])
            manifest_path = run_dir / "manifest.yaml"
            try:
                run_resume = False
                if run_dir.exists():
                    if not resume:
                        raise FileExistsError(f"WP7 run exists: {run_dir}")
                    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
                    if manifest.get("status") == "completed":
                        execution_rows.append({**run, "status": "reused_completed", "error": ""})
                        continue
                    run_resume = True
                run_inverse_design(
                    target_paths[str(run["target_id"])],
                    generated_config_paths[str(run["run_id"])],
                    repo_root=repo_root,
                    output_dir_override=run_dir,
                    resume=run_resume,
                )
                execution_rows.append({**run, "status": "completed", "error": ""})
            except Exception as exc:  # preserve and report every failed experimental cell
                execution_rows.append(
                    {**run, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                )
        pd.DataFrame(execution_rows).to_csv(output_dir / "execution_status.csv", index=False)

    summary = analyze_wp7(output_dir, config, run_matrix, repo_root)
    finished = dt.datetime.now(dt.timezone.utc)
    manifest = {
        "wp": "WP7",
        "version": config["version"],
        "status": "completed" if summary["all_planned_runs_completed"] else "completed_with_run_failures",
        "benchmark": "B0",
        "benchmark_is_gate": False,
        "generator_superiority_required": False,
        "fairness_passed": fairness_passed,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "target_suite": str(target_suite_path),
        "target_suite_sha256": sha256_file(target_suite_path),
        "base_run_config": str(base_path),
        "base_run_config_sha256": sha256_file(base_path),
        "summary": summary,
    }
    write_yaml(output_dir / "wp7_manifest.yaml", manifest)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run WP7 B0 baseline benchmark")
    parser.add_argument("--config", type=Path, default=Path("config/inverse_design/wp7_b0.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = run_wp7_benchmark(
        args.config,
        output_dir_override=args.output_dir,
        resume=args.resume,
        analyze_only=args.analyze_only,
    )
    manifest = yaml.safe_load((output / "wp7_manifest.yaml").read_text(encoding="utf-8"))
    print(json.dumps({"output": str(output), "status": manifest["status"], "benchmark_is_gate": False}))
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
