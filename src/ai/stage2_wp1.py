"""Stage-II WP1 target-matrix freezing and baseline re-evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from .baseline import sha256_file, write_yaml
from .stage2_baseline import verify_manifest
from .target_benchmark import run_target_benchmark
from .wp4 import run_wp4


ORDINARY_TARGET_TYPES = ("ID", "Interpolation")


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def load_stage2_wp1_config(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "version",
        "baseline_manifest",
        "target_suite",
        "selection",
        "seeds",
        "model_sets",
        "generator_only",
        "end_to_end",
        "output",
    }
    missing = required.difference(value)
    if missing:
        raise ValueError(f"S2-WP1 config is missing keys: {sorted(missing)}")
    counts = value["selection"]["end_to_end_counts"]
    if set(counts) != set(ORDINARY_TARGET_TYPES):
        raise ValueError("end_to_end_counts must define ID and Interpolation")
    if any(int(count) <= 0 for count in counts.values()):
        raise ValueError("end_to_end target counts must be positive")
    if value["selection"]["method"] != "standardized_farthest_point":
        raise ValueError("Unsupported S2-WP1 target selection method")
    training_seeds = [int(item["training_seed"]) for item in value["model_sets"]]
    if training_seeds != [int(seed) for seed in value["seeds"]["training"]]:
        raise ValueError("model_sets and seeds.training differ")
    if len(training_seeds) != len(set(training_seeds)):
        raise ValueError("S2-WP1 training seeds must be unique")
    return value


def _transformed_conditions(
    frame: pd.DataFrame, condition_columns: Sequence[str]
) -> np.ndarray:
    values = frame[list(condition_columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("Target conditions must be positive and finite")
    transformed = values.copy()
    for index, name in enumerate(condition_columns):
        if name in {"tsfc_kg_N_s", "total_mass_kg"}:
            transformed[:, index] = np.log(transformed[:, index])
    scale = transformed.std(axis=0, ddof=0)
    if np.any(scale <= 0.0):
        raise ValueError("Target condition has zero variance")
    return (transformed - transformed.mean(axis=0)) / scale


def _farthest_point_order(
    frame: pd.DataFrame,
    *,
    condition_columns: Sequence[str],
) -> list[int]:
    """Return a deterministic coverage order independent of model results."""

    ordered = frame.sort_values("target_id", kind="stable").reset_index()
    coordinates = _transformed_conditions(ordered, condition_columns)
    centroid_distance = np.linalg.norm(coordinates, axis=1)
    first = int(np.argmin(centroid_distance))
    selected = [first]
    remaining = set(range(len(ordered))) - {first}
    while remaining:
        candidates = sorted(remaining)
        minimum_distances = np.asarray(
            [
                min(
                    float(np.linalg.norm(coordinates[index] - coordinates[chosen]))
                    for chosen in selected
                )
                for index in candidates
            ]
        )
        maximum = float(minimum_distances.max())
        tied = [
            index
            for index, distance in zip(candidates, minimum_distances)
            if math.isclose(float(distance), maximum, rel_tol=0.0, abs_tol=1.0e-14)
        ]
        chosen = min(tied, key=lambda index: str(ordered.loc[index, "target_id"]))
        selected.append(chosen)
        remaining.remove(chosen)
    return [int(ordered.loc[index, "index"]) for index in selected]


def build_target_matrix(
    suite: pd.DataFrame, selection: Mapping[str, Any]
) -> pd.DataFrame:
    required = {
        "target_id",
        "target_type",
        "thrust_N",
        "tsfc_kg_N_s",
        "total_mass_kg",
    }
    missing = required.difference(suite.columns)
    if missing:
        raise ValueError(f"Target suite is missing fields: {sorted(missing)}")
    if suite["target_id"].duplicated().any():
        raise ValueError("Target suite contains duplicate target IDs")

    condition_columns = tuple(selection["condition_columns"])
    output = suite.copy().reset_index(drop=True)
    output["generator_only"] = True
    output["end_to_end"] = False
    output["end_to_end_rank"] = pd.Series([pd.NA] * len(output), dtype="Int64")
    output["selection_method"] = "full_suite_only"

    ordinary = output.loc[output["target_type"].isin(ORDINARY_TARGET_TYPES)]
    ordinary_coordinates = _transformed_conditions(ordinary, condition_columns)
    minimum = ordinary_coordinates.min(axis=0)
    maximum = ordinary_coordinates.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0e-12)
    unit = (ordinary_coordinates - minimum) / span
    boundary = np.max(np.abs(2.0 * unit - 1.0), axis=1)
    output["condition_boundary_score"] = math.nan
    output.loc[ordinary.index, "condition_boundary_score"] = boundary

    counts = selection["end_to_end_counts"]
    for target_type in ORDINARY_TARGET_TYPES:
        group = output.loc[output["target_type"] == target_type]
        count = int(counts[target_type])
        if len(group) < count:
            raise ValueError(
                f"Target suite has {len(group)} {target_type} rows, needs {count}"
            )
        coverage_order = _farthest_point_order(
            group, condition_columns=condition_columns
        )
        chosen = coverage_order[:count]
        output.loc[chosen, "end_to_end"] = True
        output.loc[chosen, "end_to_end_rank"] = np.arange(1, count + 1)
        output.loc[chosen, "selection_method"] = "standardized_farthest_point"

    output["target_id"] = output["target_id"].astype(str)
    return output


def _end_to_end_target_ids(matrix: pd.DataFrame) -> list[str]:
    selected = matrix.loc[matrix["end_to_end"].astype(bool)].copy()
    selected["type_order"] = selected["target_type"].map(
        {name: index for index, name in enumerate(ORDINARY_TARGET_TYPES)}
    )
    selected = selected.sort_values(
        ["type_order", "end_to_end_rank", "target_id"], kind="stable"
    )
    return selected["target_id"].astype(str).tolist()


def _wp4_config(
    config: Mapping[str, Any],
    model_set: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    training_seed = int(model_set["training_seed"])
    return {
        "version": f"s2_wp1_generator_only_seed{training_seed}_v1",
        "metrics_config": config["generator_only"]["metrics_config"],
        "data": {
            "dataset_csv": config["generator_only"]["dataset_csv"],
            "schema": config["generator_only"]["schema"],
            "normalizer": config["generator_only"]["normalizer"],
            "target_suite": config["target_suite"]["csv"],
        },
        "models": {
            str(model["model_id"]): str(model["checkpoint_path"])
            for model in model_set["models"]
        },
        "runtime": {
            "device": config["generator_only"]["device"],
            "sample_count": int(config["generator_only"]["sample_count"]),
            "sample_seed": int(config["seeds"]["generator_only_base"]) + training_seed,
            "maximum_targets_per_type": None,
        },
        "output": {
            "directory": str(output_dir),
            "refuse_overwrite": True,
        },
    }


def _benchmark_config(
    config: Mapping[str, Any], target_ids: Sequence[str], output_dir: Path
) -> Dict[str, Any]:
    return {
        "version": "s2_wp1_end_to_end_baseline_v1",
        "target_suite": {
            "csv": config["target_suite"]["csv"],
            "sha256": config["target_suite"]["sha256"],
            "selection": {"target_ids": list(target_ids)},
        },
        "optimizer_seeds": [int(seed) for seed in config["seeds"]["optimizer"]],
        "generator_seed_offset": int(config["seeds"]["end_to_end_generator_offset"]),
        "model_sets": config["model_sets"],
        "base_run_config": config["end_to_end"]["base_run_config"],
        "output": {"directory": str(output_dir), "refuse_overwrite": True},
    }


def aggregate_generator_only(
    output_dir: Path, training_seeds: Sequence[int]
) -> Dict[str, Path]:
    """Combine frozen WP4 results without changing any underlying metric."""

    model_frames = []
    target_frames = []
    failure_frames = []
    for seed_value in training_seeds:
        training_seed = int(seed_value)
        seed_dir = output_dir / "generator_only" / f"seed_{training_seed}"
        required = ("model_summary.csv", "target_summary.csv", "functional_gate.json")
        missing = [name for name in required if not (seed_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Generator-only seed {training_seed} is incomplete: {missing}"
            )
        gate = json.loads((seed_dir / "functional_gate.json").read_text(encoding="utf-8"))
        if not bool(gate.get("passed")):
            raise RuntimeError(f"Generator-only seed {training_seed} failed its functional gate")

        model_frame = pd.read_csv(seed_dir / "model_summary.csv")
        model_frame.insert(0, "training_seed", training_seed)
        model_frames.append(model_frame)
        target_frame = pd.read_csv(seed_dir / "target_summary.csv")
        target_frame.insert(0, "training_seed", training_seed)
        target_frames.append(target_frame)
        failure_path = seed_dir / "failures.csv"
        if failure_path.is_file():
            failure_frame = pd.read_csv(failure_path)
            failure_frame.insert(0, "training_seed", training_seed)
            failure_frames.append(failure_frame)

    aggregate_dir = output_dir / "generator_only_aggregate"
    aggregate_dir.mkdir(exist_ok=True)
    models = pd.concat(model_frames, ignore_index=True)
    targets = pd.concat(target_frames, ignore_index=True)
    failures = (
        pd.concat(failure_frames, ignore_index=True)
        if failure_frames
        else pd.DataFrame(
            columns=[
                "training_seed",
                "model",
                "target_id",
                "target_type",
                "candidate_id",
                "failure_code",
                "solver_warnings",
            ]
        )
    )
    models.to_csv(aggregate_dir / "model_summary_by_seed.csv", index=False)
    targets.to_csv(aggregate_dir / "target_summary_by_seed.csv", index=False)

    targets = targets.copy()
    targets["forward_valid_target"] = targets["forward_valid_count"].astype(float) > 0
    targets["candidate_valid_rate"] = (
        targets["candidate_valid_count"].astype(float)
        / targets["candidate_count"].astype(float).clip(lower=1.0)
    )
    grouped_rows = []
    for (model, target_type), group in targets.groupby(
        ["model", "target_type"], sort=True
    ):
        finite_errors = pd.to_numeric(group["best_total_error"], errors="coerce")
        grouped_rows.append(
            {
                "model": model,
                "target_type": target_type,
                "training_seed_count": int(group["training_seed"].nunique()),
                "target_run_count": int(len(group)),
                "forward_valid_target_rate": float(group["forward_valid_target"].mean()),
                "within_tolerance_rate": float(group["best_within_tolerance"].astype(bool).mean()),
                "best_total_error_mean": float(finite_errors.mean()),
                "best_total_error_median": float(finite_errors.median()),
                "best_total_error_std": float(finite_errors.std(ddof=0)),
                "candidate_valid_rate_mean": float(group["candidate_valid_rate"].mean()),
                "selected_diversity_mean": float(
                    pd.to_numeric(
                        group["mean_nearest_neighbour_distance"], errors="coerce"
                    ).mean()
                ),
            }
        )
    target_type_summary = pd.DataFrame(grouped_rows)
    target_type_summary.to_csv(
        aggregate_dir / "target_type_summary.csv", index=False
    )

    if len(failures):
        failure_summary = (
            failures.groupby(
                ["model", "target_type", "failure_code"], dropna=False
            )
            .size()
            .rename("failure_count")
            .reset_index()
        )
        denominators = (
            failures.groupby(["model", "target_type"], dropna=False)
            .size()
            .rename("all_failure_count")
            .reset_index()
        )
        failure_summary = failure_summary.merge(
            denominators, on=["model", "target_type"], how="left"
        )
        failure_summary["fraction_of_model_type_failures"] = (
            failure_summary["failure_count"] / failure_summary["all_failure_count"]
        )
    else:
        failure_summary = pd.DataFrame(
            columns=[
                "model",
                "target_type",
                "failure_code",
                "failure_count",
                "all_failure_count",
                "fraction_of_model_type_failures",
            ]
        )
    failure_summary.to_csv(aggregate_dir / "failure_summary.csv", index=False)
    return {
        "model_summary_by_seed": aggregate_dir / "model_summary_by_seed.csv",
        "target_summary_by_seed": aggregate_dir / "target_summary_by_seed.csv",
        "target_type_summary": aggregate_dir / "target_type_summary.csv",
        "failure_summary": aggregate_dir / "failure_summary.csv",
    }


def aggregate_end_to_end(output_dir: Path, expected_run_count: int) -> Dict[str, Path]:
    """Add S2-WP1 target-type and difficulty views to the shared benchmark outputs."""

    benchmark_dir = output_dir / "end_to_end"
    metrics_path = benchmark_dir / "run_metrics.csv"
    paired_path = benchmark_dir / "paired_comparison.csv"
    if not metrics_path.is_file() or not paired_path.is_file():
        raise FileNotFoundError("End-to-end benchmark aggregation inputs are incomplete")
    metrics = pd.read_csv(metrics_path)
    paired = pd.read_csv(paired_path)
    if len(metrics) != int(expected_run_count):
        raise RuntimeError(
            f"End-to-end run count is {len(metrics)}, expected {expected_run_count}"
        )

    rows = []
    for (method, target_type), group in metrics.groupby(
        ["method", "target_type"], sort=True
    ):
        evaluations_to_hit = pd.to_numeric(
            group["search_evaluations_to_hit"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "method": method,
                "target_type": target_type,
                "run_count": int(len(group)),
                "initial_best_error_mean": float(group["initial_best_error"].mean()),
                "initial_best_error_std": float(group["initial_best_error"].std(ddof=0)),
                "initial_best_error_median": float(group["initial_best_error"].median()),
                "initial_top10_error_mean": float(
                    group["initial_top10_mean_error"].mean()
                ),
                "initial_top10_error_std": float(
                    group["initial_top10_mean_error"].std(ddof=0)
                ),
                "initial_target_success_rate": float(
                    group["initial_has_hit"].astype(bool).mean()
                ),
                "search_best_error_mean": float(group["search_best_error"].mean()),
                "search_best_error_std": float(
                    group["search_best_error"].std(ddof=0)
                ),
                "search_target_success_rate": float(
                    group["search_has_hit"].astype(bool).mean()
                ),
                "evaluations_to_hit_median": (
                    float(evaluations_to_hit.median())
                    if len(evaluations_to_hit)
                    else math.nan
                ),
                "total_evaluations_mean": float(group["total_evaluations"].mean()),
            }
        )
    type_summary = pd.DataFrame(rows)
    type_summary_path = benchmark_dir / "target_type_summary.csv"
    type_summary.to_csv(type_summary_path, index=False)

    paired_rows = []
    for (model, baseline, target_type), group in paired.groupby(
        ["model_id", "baseline", "target_type"], sort=True
    ):
        paired_rows.append(
            {
                "model_id": model,
                "baseline": baseline,
                "target_type": target_type,
                "pair_count": int(len(group)),
                "initial_best_win_rate": float(group["initial_best_win"].astype(bool).mean()),
                "initial_top10_win_rate": float(group["initial_top10_win"].astype(bool).mean()),
                "search_best_win_rate": float(group["search_best_win"].astype(bool).mean()),
                "initial_best_error_improvement_mean": float(
                    group["initial_best_error_improvement"].mean()
                ),
                "initial_top10_error_improvement_mean": float(
                    group["initial_top10_error_improvement"].mean()
                ),
            }
        )
    paired_summary = pd.DataFrame(paired_rows)
    paired_summary_path = benchmark_dir / "paired_summary_by_target_type.csv"
    paired_summary.to_csv(paired_summary_path, index=False)

    target_rows = []
    for (target_id, target_type), group in metrics.groupby(
        ["target_id", "target_type"], sort=True
    ):
        screened = group.loc[group["method"] == "screened_random"]
        target_rows.append(
            {
                "target_id": target_id,
                "target_type": target_type,
                "run_count": int(len(group)),
                "all_method_initial_best_error_median": float(
                    group["initial_best_error"].median()
                ),
                "all_method_initial_success_rate": float(
                    group["initial_has_hit"].astype(bool).mean()
                ),
                "screened_random_initial_best_error_mean": float(
                    screened["initial_best_error"].mean()
                ),
                "screened_random_initial_success_rate": float(
                    screened["initial_has_hit"].astype(bool).mean()
                ),
                "best_method_initial_error": float(group["initial_best_error"].min()),
                "best_method_search_error": float(group["search_best_error"].min()),
            }
        )
    difficulty = pd.DataFrame(target_rows).sort_values(
        ["screened_random_initial_best_error_mean", "target_id"],
        ascending=[False, True],
        kind="stable",
    )
    difficulty.insert(0, "difficulty_rank", np.arange(1, len(difficulty) + 1))
    difficulty_path = benchmark_dir / "target_difficulty.csv"
    difficulty.to_csv(difficulty_path, index=False)
    return {
        "target_type_summary": type_summary_path,
        "paired_summary_by_target_type": paired_summary_path,
        "target_difficulty": difficulty_path,
    }


def prepare_stage2_wp1(
    config_path: Path,
    *,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
    resume: bool = False,
) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    config_path = Path(config_path).resolve()
    config = load_stage2_wp1_config(config_path)
    baseline_path = _resolve(repo_root, config["baseline_manifest"])
    baseline_result = verify_manifest(baseline_path, repo_root=repo_root)
    if not baseline_result["ok"]:
        raise RuntimeError(f"Frozen baseline verification failed: {baseline_result['issues']}")

    suite_path = _resolve(repo_root, config["target_suite"]["csv"])
    actual_suite_sha256 = sha256_file(suite_path)
    if actual_suite_sha256 != str(config["target_suite"]["sha256"]):
        raise ValueError("S2-WP1 target suite SHA256 differs from frozen config")
    output_dir = (
        output_dir_override.resolve()
        if output_dir_override is not None
        else _resolve(repo_root, config["output"]["directory"])
    )
    if output_dir.exists():
        manifest_path = output_dir / "manifest.yaml"
        if resume and manifest_path.is_file():
            existing = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            if existing.get("version") != config["version"]:
                raise ValueError("Existing S2-WP1 output uses a different version")
            if existing.get("target_suite_sha256") != actual_suite_sha256:
                raise ValueError("Existing S2-WP1 output uses a different target suite")
            return output_dir
        raise FileExistsError(f"S2-WP1 output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    generated_dir = output_dir / "generated_configs"
    generated_dir.mkdir()

    suite = pd.read_csv(suite_path)
    matrix = build_target_matrix(suite, config["selection"])
    matrix_path = output_dir / "target_matrix.csv"
    matrix.to_csv(matrix_path, index=False)
    selected_ids = _end_to_end_target_ids(matrix)
    frozen_ids_path = _resolve(
        repo_root, str(config["selection"]["frozen_end_to_end_ids"])
    )
    frozen_selection = yaml.safe_load(frozen_ids_path.read_text(encoding="utf-8"))
    expected_ids = [str(value) for value in frozen_selection["target_ids"]]
    if selected_ids != expected_ids:
        raise ValueError(
            "Computed S2-WP1 end-to-end targets differ from the frozen target IDs"
        )
    if frozen_selection["source_target_suite_sha256"] != actual_suite_sha256:
        raise ValueError("Frozen target IDs refer to a different target suite")
    selected = matrix.loc[matrix["end_to_end"].astype(bool)].copy()
    selected.to_csv(output_dir / "end_to_end_targets.csv", index=False)

    wp4_configs = []
    for model_set in config["model_sets"]:
        training_seed = int(model_set["training_seed"])
        generated_path = generated_dir / f"generator_only_seed{training_seed}.yaml"
        value = _wp4_config(
            config,
            model_set,
            output_dir / "generator_only" / f"seed_{training_seed}",
        )
        write_yaml(generated_path, value)
        wp4_configs.append(generated_path)

    benchmark_config_path = generated_dir / "end_to_end_benchmark.yaml"
    write_yaml(
        benchmark_config_path,
        _benchmark_config(config, selected_ids, output_dir / "end_to_end"),
    )
    write_yaml(
        output_dir / "manifest.yaml",
        {
            "status": "prepared",
            "version": config["version"],
            "prepared_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "baseline_id": baseline_result["baseline_id"],
            "baseline_manifest": str(baseline_path),
            "target_suite": str(suite_path),
            "target_suite_sha256": actual_suite_sha256,
            "target_matrix": str(matrix_path),
            "target_matrix_sha256": sha256_file(matrix_path),
            "frozen_end_to_end_ids": str(frozen_ids_path),
            "frozen_end_to_end_ids_sha256": sha256_file(frozen_ids_path),
            "generator_only_target_count": int(matrix["generator_only"].sum()),
            "end_to_end_target_count": int(matrix["end_to_end"].sum()),
            "end_to_end_counts": {
                str(key): int(value)
                for key, value in selected["target_type"].value_counts().items()
            },
            "training_seeds": config["seeds"]["training"],
            "optimizer_seeds": config["seeds"]["optimizer"],
            "planned_generator_only_model_target_runs": int(
                len(matrix) * sum(len(item["models"]) for item in config["model_sets"])
            ),
            "planned_end_to_end_runs": int(
                len(selected_ids)
                * len(config["seeds"]["optimizer"])
                * sum(2 + len(item["models"]) for item in config["model_sets"])
            ),
            "generated_configs": [
                str(path) for path in [*wp4_configs, benchmark_config_path]
            ],
        },
    )
    return output_dir


def run_stage2_wp1(
    config_path: Path,
    *,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
    run_generator_only: bool = False,
    run_end_to_end: bool = False,
    resume: bool = False,
) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    output = prepare_stage2_wp1(
        config_path,
        repo_root=repo_root,
        output_dir_override=output_dir_override,
        resume=resume,
    )
    manifest_path = output / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if run_generator_only:
        if not manifest.get("generator_only_completed"):
            for path_value in manifest["generated_configs"][:-1]:
                generated_config = yaml.safe_load(
                    Path(path_value).read_text(encoding="utf-8")
                )
                generated_output = Path(generated_config["output"]["directory"])
                if (generated_output / "wp4_manifest.yaml").is_file():
                    continue
                run_wp4(Path(path_value), repo_root=repo_root)
        aggregate_paths = aggregate_generator_only(output, manifest["training_seeds"])
        manifest["generator_only_completed"] = True
        manifest["generator_only_aggregate"] = {
            name: str(path) for name, path in aggregate_paths.items()
        }
        manifest["status"] = "generator_only_completed"
        write_yaml(manifest_path, manifest)
    if run_end_to_end:
        if not manifest.get("end_to_end_completed"):
            benchmark_path = Path(manifest["generated_configs"][-1])
            benchmark_config = yaml.safe_load(
                benchmark_path.read_text(encoding="utf-8")
            )
            benchmark_output = Path(benchmark_config["output"]["directory"])
            benchmark_manifest = benchmark_output / "manifest.yaml"
            if not benchmark_manifest.is_file():
                run_target_benchmark(benchmark_path, repo_root=repo_root, resume=resume)
        aggregate_paths = aggregate_end_to_end(
            output, int(manifest["planned_end_to_end_runs"])
        )
        manifest["end_to_end_completed"] = True
        manifest["end_to_end_aggregate"] = {
            name: str(path) for name, path in aggregate_paths.items()
        }
        manifest["status"] = (
            "complete" if manifest.get("generator_only_completed") else "end_to_end_completed"
        )
        write_yaml(manifest_path, manifest)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/stage2/s2_wp1.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-generator-only", action="store_true")
    parser.add_argument("--run-end-to-end", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = run_stage2_wp1(
        args.config,
        output_dir_override=args.output_dir,
        run_generator_only=args.run_generator_only,
        run_end_to_end=args.run_end_to_end,
        resume=args.resume,
    )
    manifest = yaml.safe_load((output / "manifest.yaml").read_text(encoding="utf-8"))
    print(json.dumps({"output": str(output), "status": manifest["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
