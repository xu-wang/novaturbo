"""Stage-II WP2 orchestration for explicitly bounded 10D generators."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd
import yaml

from .baseline import sha256_file, write_json, write_yaml
from .stage2_baseline import verify_manifest
from .stage2_wp1 import aggregate_generator_only
from .wp3 import load_wp3_config, run_wp3
from .wp4 import run_wp4


MODEL_NAMES = ("cvae", "nflows", "cinn", "cfm")
ORDINARY_TARGET_TYPES = ("ID", "Interpolation")


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def load_stage2_wp2_config(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "version",
        "baseline_manifest",
        "reference_training_config",
        "smoke_training_config",
        "full_training_config",
        "wp1_output",
        "bounded_contract",
        "evaluation",
        "output",
    }
    missing = required.difference(value)
    if missing:
        raise ValueError(f"S2-WP2 config is missing keys: {sorted(missing)}")
    contract = value["bounded_contract"]
    if contract.get("kind") != "atanh_tanh":
        raise ValueError("S2-WP2 bounded contract must use atanh_tanh")
    if not bool(contract.get("enabled")):
        raise ValueError("S2-WP2 bounded contract must be enabled")
    return value


def _bounded_model_contract(contract: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "enabled": True,
        "epsilon": float(contract["epsilon"]),
        "input_tolerance": float(contract["input_tolerance"]),
        "saturation_threshold": float(contract["saturation_threshold"]),
    }


def _without_bounded_output(models: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(models))
    for value in result.values():
        value.pop("bounded_output", None)
    return result


def validate_training_configs(
    reference: Mapping[str, Any],
    smoke: Mapping[str, Any],
    full: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    if set(reference["models"]) != set(MODEL_NAMES):
        raise ValueError("Reference training config does not define the four frozen models")
    if _without_bounded_output(full["models"]) != reference["models"]:
        raise ValueError(
            "S2-WP2 full model architecture differs from the frozen reference beyond bounded_output"
        )
    for key in ("lockfile", "training_contract", "data", "runtime"):
        if full.get(key) != reference.get(key):
            raise ValueError(f"S2-WP2 full training config changed frozen field: {key}")
    expected = _bounded_model_contract(contract)
    for label, training in (("smoke", smoke), ("full", full)):
        if set(training["models"]) != set(MODEL_NAMES):
            raise ValueError(f"S2-WP2 {label} config must define all four models")
        for model_name, model_config in training["models"].items():
            if dict(model_config.get("bounded_output", {})) != expected:
                raise ValueError(
                    f"S2-WP2 {label} {model_name} bounded_output differs from contract"
                )
            if not bool(model_config.get("enabled")):
                raise ValueError(f"S2-WP2 {label} {model_name} must be enabled")
    for key in ("dataset_csv", "schema", "normalizer", "target_suite"):
        if smoke["data"].get(key) != reference["data"].get(key):
            raise ValueError(f"S2-WP2 smoke config changed frozen data field: {key}")
    if str(full["output"]["directory"]) == str(reference["output"]["directory"]):
        raise ValueError("S2-WP2 must not overwrite the frozen reference checkpoints")


def _training_paths(
    repo_root: Path, config: Mapping[str, Any]
) -> tuple[Path, Path, Path]:
    paths = tuple(
        _resolve(repo_root, str(config[name]))
        for name in (
            "reference_training_config",
            "smoke_training_config",
            "full_training_config",
        )
    )
    return paths  # type: ignore[return-value]


def prepare_stage2_wp2(
    config_path: Path,
    *,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
    resume: bool = False,
) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    config_path = Path(config_path).resolve()
    config = load_stage2_wp2_config(config_path)
    baseline_path = _resolve(repo_root, config["baseline_manifest"])
    baseline = verify_manifest(baseline_path, repo_root=repo_root)
    if not baseline["ok"]:
        raise RuntimeError(f"Frozen baseline verification failed: {baseline['issues']}")

    reference_path, smoke_path, full_path = _training_paths(repo_root, config)
    reference = load_wp3_config(reference_path)
    smoke = load_wp3_config(smoke_path)
    full = load_wp3_config(full_path)
    validate_training_configs(reference, smoke, full, config["bounded_contract"])

    output_dir = (
        output_dir_override.resolve()
        if output_dir_override is not None
        else _resolve(repo_root, config["output"]["directory"])
    )
    manifest_path = output_dir / "manifest.yaml"
    if output_dir.exists():
        if resume and manifest_path.is_file():
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != config["version"]:
                raise ValueError("Existing S2-WP2 output uses a different version")
            return output_dir
        raise FileExistsError(f"S2-WP2 output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "generated_configs").mkdir()
    write_yaml(
        manifest_path,
        {
            "status": "prepared",
            "version": config["version"],
            "prepared_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "baseline_id": baseline["baseline_id"],
            "baseline_manifest": str(baseline_path),
            "bounded_contract": dict(config["bounded_contract"]),
            "reference_training_config": str(reference_path),
            "reference_training_config_sha256": sha256_file(reference_path),
            "smoke_training_config": str(smoke_path),
            "smoke_training_config_sha256": sha256_file(smoke_path),
            "full_training_config": str(full_path),
            "full_training_config_sha256": sha256_file(full_path),
        },
    )
    return output_dir


def _validate_bounded_training_output(training_output: Path) -> Dict[str, Any]:
    summary_path = training_output / "seed_summary.csv"
    manifest_path = training_output / "wp3_manifest.yaml"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Bounded training output is incomplete: {training_output}")
    summary = pd.read_csv(summary_path)
    if not (summary["sample_out_of_bounds_rate"] == 0.0).all():
        raise RuntimeError("A bounded checkpoint produced out-of-bounds samples")
    diagnostic_rows = []
    for row in summary.itertuples(index=False):
        path = training_output / row.model / f"seed_{int(row.seed)}" / "sample_diagnostics.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if not bool(value.get("bounded_output_enabled")):
            raise RuntimeError(f"Bounded diagnostics missing for {row.model} seed {row.seed}")
        if float(value.get("output_out_of_bounds_rate", math.inf)) != 0.0:
            raise RuntimeError(f"Bounded diagnostics failed for {row.model} seed {row.seed}")
        diagnostic_rows.append(
            {
                "model": row.model,
                "seed": int(row.seed),
                "output_out_of_bounds_rate": value["output_out_of_bounds_rate"],
                "raw_pre_transform_outside_unit_rate": value[
                    "raw_pre_transform_outside_unit_rate"
                ],
                "output_near_boundary_rate": value["output_near_boundary_rate"],
                "output_min_boundary_distance": value[
                    "output_min_boundary_distance"
                ],
            }
        )
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(training_output / "bounded_diagnostics.csv", index=False)
    return {
        "training_output": str(training_output),
        "run_count": int(len(summary)),
        "all_samples_bounded": True,
        "bounded_diagnostics": str(training_output / "bounded_diagnostics.csv"),
    }


def _training_output(repo_root: Path, training_config_path: Path) -> Path:
    training = load_wp3_config(training_config_path)
    return _resolve(repo_root, training["output"]["directory"])


def _run_training(
    config_path: Path,
    *,
    repo_root: Path,
    resume: bool,
) -> Dict[str, Any]:
    output = _training_output(repo_root, config_path)
    if not (resume and (output / "wp3_manifest.yaml").is_file()):
        run_wp3(config_path, repo_root=repo_root)
    return _validate_bounded_training_output(output)


def _wp4_config(
    config: Mapping[str, Any],
    training: Mapping[str, Any],
    training_output: Path,
    seed: int,
    output_dir: Path,
) -> Dict[str, Any]:
    evaluation = config["evaluation"]
    return {
        "version": f"s2_wp2_bounded_round_trip_seed{seed}_v1",
        "metrics_config": evaluation["metrics_config"],
        "data": {
            "dataset_csv": evaluation["dataset_csv"],
            "schema": evaluation["schema"],
            "normalizer": evaluation["normalizer"],
            "target_suite": evaluation["target_suite"],
        },
        "models": {
            name: str(training_output / name / f"seed_{seed}" / "best.pt")
            for name in MODEL_NAMES
        },
        "runtime": {
            "device": evaluation["device"],
            "sample_count": int(evaluation["sample_count"]),
            "sample_seed": int(evaluation["sample_seed_base"]) + int(seed),
            "maximum_targets_per_type": None,
        },
        "output": {"directory": str(output_dir), "refuse_overwrite": True},
    }


def _validate_paired_seed_budget(paired: pd.DataFrame) -> None:
    if not (
        paired["sample_seed_baseline"].astype(int)
        == paired["sample_seed_bounded"].astype(int)
    ).all():
        raise RuntimeError("S2-WP2 and S2-WP1 sampling seeds are not exactly paired")
    if not (
        paired["candidate_count_baseline"].astype(int)
        == paired["candidate_count_bounded"].astype(int)
    ).all():
        raise RuntimeError("S2-WP2 and S2-WP1 candidate budgets are not exactly paired")


def _compare_with_wp1(
    bounded_output: Path,
    wp1_output: Path,
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    baseline_path = wp1_output / "generator_only_aggregate" / "target_summary_by_seed.csv"
    bounded_path = bounded_output / "generator_only_aggregate" / "target_summary_by_seed.csv"
    baseline = pd.read_csv(baseline_path)
    bounded = pd.read_csv(bounded_path)
    keys = ["training_seed", "model", "target_id", "target_type"]
    paired = baseline.merge(bounded, on=keys, suffixes=("_baseline", "_bounded"))
    if len(paired) != len(baseline) or len(paired) != len(bounded):
        raise RuntimeError("S2-WP2 and S2-WP1 target ledgers are not exactly paired")
    _validate_paired_seed_budget(paired)
    for suffix in ("baseline", "bounded"):
        paired[f"candidate_valid_rate_{suffix}"] = (
            paired[f"candidate_valid_count_{suffix}"]
            / paired[f"candidate_count_{suffix}"].clip(lower=1)
        )
    rows = []
    for (model, target_type), group in paired.groupby(
        ["model", "target_type"], sort=True
    ):
        baseline_error = group["best_total_error_baseline"].median()
        bounded_error = group["best_total_error_bounded"].median()
        baseline_diversity = group["mean_nearest_neighbour_distance_baseline"].mean()
        bounded_diversity = group["mean_nearest_neighbour_distance_bounded"].mean()
        rows.append(
            {
                "model": model,
                "target_type": target_type,
                "paired_target_runs": int(len(group)),
                "baseline_candidate_valid_rate": float(
                    group["candidate_valid_rate_baseline"].mean()
                ),
                "bounded_candidate_valid_rate": float(
                    group["candidate_valid_rate_bounded"].mean()
                ),
                "candidate_valid_rate_improvement": float(
                    (
                        group["candidate_valid_rate_bounded"]
                        - group["candidate_valid_rate_baseline"]
                    ).mean()
                ),
                "baseline_within_tolerance_rate": float(
                    group["best_within_tolerance_baseline"].astype(bool).mean()
                ),
                "bounded_within_tolerance_rate": float(
                    group["best_within_tolerance_bounded"].astype(bool).mean()
                ),
                "baseline_median_best_error": float(baseline_error),
                "bounded_median_best_error": float(bounded_error),
                "median_error_relative_change": float(
                    bounded_error / baseline_error - 1.0
                ) if baseline_error > 0 else math.nan,
                "diversity_retention": float(
                    bounded_diversity / baseline_diversity
                ) if baseline_diversity > 0 else math.nan,
            }
        )
    summary = pd.DataFrame(rows)
    summary_path = bounded_output / "bounded_ablation.csv"
    summary.to_csv(summary_path, index=False)

    ordinary = summary.loc[summary["target_type"].isin(ORDINARY_TARGET_TYPES)]
    failures = pd.read_csv(
        bounded_output / "generator_only_aggregate" / "failure_summary.csv"
    )
    out_of_bounds = int(
        failures.loc[failures["failure_code"] == "OUT_OF_BOUNDS", "failure_count"].sum()
    )
    maximum_regression = float(contract["maximum_ordinary_error_regression"])
    minimum_diversity = float(contract["minimum_diversity_retention"])
    gate = {
        "gate": "S2-WP2 bounded generation ablation",
        "passed": bool(
            out_of_bounds == 0
            and (ordinary["median_error_relative_change"] <= maximum_regression).all()
            and (ordinary["diversity_retention"] >= minimum_diversity).all()
        ),
        "output_out_of_bounds_failure_count": out_of_bounds,
        "maximum_allowed_error_regression": maximum_regression,
        "minimum_required_diversity_retention": minimum_diversity,
        "summary": str(summary_path),
    }
    write_json(bounded_output / "bounded_gate.json", gate)
    return gate


def run_stage2_wp2(
    config_path: Path,
    *,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
    run_smoke: bool = False,
    run_full_training: bool = False,
    run_round_trip: bool = False,
    resume: bool = False,
) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    config_path = Path(config_path).resolve()
    config = load_stage2_wp2_config(config_path)
    output = prepare_stage2_wp2(
        config_path,
        repo_root=repo_root,
        output_dir_override=output_dir_override,
        resume=resume,
    )
    manifest_path = output / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    _, smoke_path, full_path = _training_paths(repo_root, config)

    if run_smoke:
        manifest["smoke"] = _run_training(
            smoke_path, repo_root=repo_root, resume=resume
        )
        manifest["status"] = "smoke_completed"
        write_yaml(manifest_path, manifest)

    if run_full_training:
        manifest["full_training"] = _run_training(
            full_path, repo_root=repo_root, resume=resume
        )
        manifest["status"] = "training_completed"
        write_yaml(manifest_path, manifest)

    if run_round_trip:
        training = load_wp3_config(full_path)
        training_output = _training_output(repo_root, full_path)
        _validate_bounded_training_output(training_output)
        generated_configs = []
        for seed_value in training["runtime"]["seeds"]:
            seed = int(seed_value)
            path = output / "generated_configs" / f"round_trip_seed{seed}.yaml"
            if not path.is_file():
                write_yaml(
                    path,
                    _wp4_config(
                        config,
                        training,
                        training_output,
                        seed,
                        output / "generator_only" / f"seed_{seed}",
                    ),
                )
            generated_configs.append(path)
            run_dir = output / "generator_only" / f"seed_{seed}"
            if not (resume and (run_dir / "wp4_manifest.yaml").is_file()):
                run_wp4(path, repo_root=repo_root)
        aggregate_generator_only(
            output, [int(value) for value in training["runtime"]["seeds"]]
        )
        gate = _compare_with_wp1(
            output,
            _resolve(repo_root, config["wp1_output"]),
            config["bounded_contract"],
        )
        manifest["round_trip_configs"] = [str(path) for path in generated_configs]
        manifest["bounded_gate"] = gate
        manifest["status"] = "complete"
        write_yaml(manifest_path, manifest)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/stage2/s2_wp2.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--run-full-training", action="store_true")
    parser.add_argument("--run-round-trip", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = run_stage2_wp2(
        args.config,
        output_dir_override=args.output_dir,
        run_smoke=args.run_smoke,
        run_full_training=args.run_full_training,
        run_round_trip=args.run_round_trip,
        resume=args.resume,
    )
    manifest = yaml.safe_load((output / "manifest.yaml").read_text(encoding="utf-8"))
    print(json.dumps({"output": str(output), "status": manifest["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
