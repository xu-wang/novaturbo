"""Stage-II WP2B bounded rejection-refill experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd
import yaml

from .baseline import write_json, write_yaml
from .stage2_baseline import verify_manifest
from .stage2_wp1 import aggregate_generator_only
from .wp4 import run_wp4


ORDINARY_TARGET_TYPES = ("ID", "Interpolation")


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def load_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "version",
        "baseline_manifest",
        "wp1_config",
        "wp1_output",
        "smooth_bounded_output",
        "rejection_contract",
        "output",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"S2-WP2B config is missing keys: {sorted(missing)}")
    contract = config["rejection_contract"]
    if contract.get("policy") != "bounded_rejection":
        raise ValueError("S2-WP2B requires the bounded_rejection policy")
    if int(contract["accepted_candidate_count"]) <= 0:
        raise ValueError("accepted_candidate_count must be positive")
    if float(contract["max_draw_multiplier"]) < 1.0:
        raise ValueError("max_draw_multiplier must be at least one")
    return config


def _wp4_config(
    config: Mapping[str, Any],
    wp1: Mapping[str, Any],
    model_set: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    contract = config["rejection_contract"]
    training_seed = int(model_set["training_seed"])
    return {
        "version": f"s2_wp2b_rejection_seed{training_seed}_v1",
        "metrics_config": wp1["generator_only"]["metrics_config"],
        "data": {
            "dataset_csv": wp1["generator_only"]["dataset_csv"],
            "schema": wp1["generator_only"]["schema"],
            "normalizer": wp1["generator_only"]["normalizer"],
            "target_suite": wp1["target_suite"]["csv"],
        },
        "models": {
            str(model["model_id"]): str(model["checkpoint_path"])
            for model in model_set["models"]
        },
        "runtime": {
            "device": wp1["generator_only"]["device"],
            "sample_count": int(contract["accepted_candidate_count"]),
            "sample_seed": int(wp1["seeds"]["generator_only_base"])
            + training_seed,
            "maximum_targets_per_type": None,
            "sampling": {
                "policy": "bounded_rejection",
                "lower_bound": float(contract["lower_bound"]),
                "upper_bound": float(contract["upper_bound"]),
                "tolerance": float(contract["tolerance"]),
                "batch_size": int(contract["batch_size"]),
                "max_draw_multiplier": float(contract["max_draw_multiplier"]),
            },
        },
        "output": {"directory": str(output_dir), "refuse_overwrite": True},
    }


def prepare(
    config_path: Path,
    *,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    config = load_config(Path(config_path).resolve())
    baseline = verify_manifest(
        _resolve(repo_root, config["baseline_manifest"]), repo_root=repo_root
    )
    if not baseline["ok"]:
        raise RuntimeError(f"Frozen baseline verification failed: {baseline['issues']}")
    wp1_path = _resolve(repo_root, config["wp1_config"])
    wp1 = yaml.safe_load(wp1_path.read_text(encoding="utf-8"))
    if int(config["rejection_contract"]["accepted_candidate_count"]) != int(
        wp1["generator_only"]["sample_count"]
    ):
        raise ValueError("S2-WP2B must preserve the frozen delivered candidate budget")
    output = (
        Path(output_dir_override).resolve()
        if output_dir_override is not None
        else _resolve(repo_root, config["output"]["directory"])
    )
    if output.exists():
        raise FileExistsError(f"S2-WP2B output exists: {output}")
    output.mkdir(parents=True)
    (output / "generated_configs").mkdir()
    write_yaml(output / "build_config.yaml", config)
    write_yaml(
        output / "manifest.yaml",
        {
            "wp": "S2-WP2B",
            "version": config["version"],
            "status": "prepared",
            "prepared_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "baseline_id": baseline["baseline_id"],
            "rejection_contract": config["rejection_contract"],
        },
    )
    return output


def _paired_summary(
    baseline_output: Path,
    rejection_output: Path,
    bounded_output: Path,
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    relative = Path("generator_only_aggregate") / "target_summary_by_seed.csv"
    baseline = pd.read_csv(baseline_output / relative)
    rejection = pd.read_csv(rejection_output / relative)
    bounded = pd.read_csv(bounded_output / relative)
    keys = ["training_seed", "model", "target_id", "target_type"]
    paired = baseline.merge(rejection, on=keys, suffixes=("_baseline", "_rejection"))
    if len(paired) != len(baseline) or len(paired) != len(rejection):
        raise RuntimeError("S2-WP2B and S2-WP1 ledgers are not exactly paired")
    bounded_columns = keys + [
        "best_total_error",
        "best_within_tolerance",
        "candidate_valid_count",
        "candidate_count",
    ]
    paired = paired.merge(
        bounded[bounded_columns].rename(
            columns={
                name: f"{name}_smooth_bounded"
                for name in bounded_columns
                if name not in keys
            }
        ),
        on=keys,
    )
    if len(paired) != len(baseline):
        raise RuntimeError("Three-way S2-WP2 target ledgers are not exactly paired")
    if not (
        paired["sample_seed_baseline"].astype(int)
        == paired["sample_seed_rejection"].astype(int)
    ).all():
        raise RuntimeError("S2-WP2B did not preserve frozen sampling seeds")

    paired["individual_non_regression"] = (
        paired["best_total_error_rejection"]
        <= paired["best_total_error_baseline"] + 1.0e-12
    ) | paired["best_total_error_baseline"].isna()
    paired.to_csv(rejection_output / "paired_target_ledger.csv", index=False)

    rows = []
    for (model, target_type), group in paired.groupby(
        ["model", "target_type"], sort=True
    ):
        base_error = group["best_total_error_baseline"].median()
        rejection_error = group["best_total_error_rejection"].median()
        bounded_error = group["best_total_error_smooth_bounded"].median()
        base_diversity = group["mean_nearest_neighbour_distance_baseline"].mean()
        rejection_diversity = group[
            "mean_nearest_neighbour_distance_rejection"
        ].mean()
        base_valid = (
            group["candidate_valid_count_baseline"]
            / group["candidate_count_baseline"].clip(lower=1)
        ).mean()
        rejection_valid = (
            group["candidate_valid_count_rejection"]
            / group["candidate_count_rejection"].clip(lower=1)
        ).mean()
        rows.append(
            {
                "model": model,
                "target_type": target_type,
                "paired_target_runs": int(len(group)),
                "baseline_candidate_valid_rate": float(base_valid),
                "rejection_candidate_valid_rate": float(rejection_valid),
                "baseline_within_tolerance_rate": float(
                    group["best_within_tolerance_baseline"].astype(bool).mean()
                ),
                "rejection_within_tolerance_rate": float(
                    group["best_within_tolerance_rejection"].astype(bool).mean()
                ),
                "baseline_median_best_error": float(base_error),
                "smooth_bounded_median_best_error": float(bounded_error),
                "rejection_median_best_error": float(rejection_error),
                "rejection_error_relative_change": (
                    float(rejection_error / base_error - 1.0)
                    if base_error > 0
                    else math.nan
                ),
                "rejection_vs_smooth_bounded_error_change": (
                    float(rejection_error / bounded_error - 1.0)
                    if bounded_error > 0
                    else math.nan
                ),
                "individual_non_regression_rate": float(
                    group["individual_non_regression"].mean()
                ),
                "diversity_retention": (
                    float(rejection_diversity / base_diversity)
                    if base_diversity > 0
                    else math.nan
                ),
                "mean_network_draw_count": float(
                    group["network_draw_count"].mean()
                ),
                "mean_sampling_batch_count": float(
                    group["sampling_batch_count"].mean()
                ),
                "mean_sampling_elapsed_seconds": float(
                    group["sampling_elapsed_seconds"].mean()
                ),
                "mean_forward_valid_count": float(
                    group["forward_valid_count_rejection"].mean()
                ),
                "budget_exhaustion_rate": float(
                    group["sampling_budget_exhausted"].astype(bool).mean()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(rejection_output / "rejection_ablation.csv", index=False)

    ordinary = summary.loc[summary["target_type"].isin(ORDINARY_TARGET_TYPES)]
    failures = pd.read_csv(
        rejection_output / "generator_only_aggregate" / "failure_summary.csv"
    )
    out_of_bounds = int(
        failures.loc[
            failures["failure_code"] == "OUT_OF_BOUNDS", "failure_count"
        ].sum()
    )
    maximum_regression = float(contract["maximum_ordinary_error_regression"])
    minimum_non_regression = float(
        contract["minimum_ordinary_target_non_regression_rate"]
    )
    minimum_diversity = float(contract["minimum_diversity_retention"])
    gate = {
        "gate": "S2-WP2B bounded rejection-refill ablation",
        "passed": bool(
            out_of_bounds == 0
            and (ordinary["rejection_error_relative_change"] <= maximum_regression).all()
            and (
                ordinary["individual_non_regression_rate"]
                >= minimum_non_regression
            ).all()
            and (ordinary["diversity_retention"] >= minimum_diversity).all()
            and (ordinary["budget_exhaustion_rate"] == 0.0).all()
        ),
        "output_out_of_bounds_failure_count": out_of_bounds,
        "maximum_allowed_error_regression": maximum_regression,
        "minimum_required_target_non_regression_rate": minimum_non_regression,
        "minimum_required_diversity_retention": minimum_diversity,
        "summary": str(rejection_output / "rejection_ablation.csv"),
    }
    write_json(rejection_output / "rejection_gate.json", gate)
    return gate


def run(
    config_path: Path,
    *,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
    resume: bool = False,
) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    output = (
        Path(output_dir_override).resolve()
        if output_dir_override is not None
        else _resolve(repo_root, config["output"]["directory"])
    )
    if not output.exists():
        output = prepare(
            config_path,
            repo_root=repo_root,
            output_dir_override=output_dir_override,
        )
    elif not resume:
        raise FileExistsError(f"S2-WP2B output exists: {output}")

    wp1 = yaml.safe_load(
        _resolve(repo_root, config["wp1_config"]).read_text(encoding="utf-8")
    )
    generated = []
    seeds = []
    for model_set in wp1["model_sets"]:
        seed = int(model_set["training_seed"])
        seeds.append(seed)
        run_dir = output / "generator_only" / f"seed_{seed}"
        config_path_out = output / "generated_configs" / f"round_trip_seed{seed}.yaml"
        if not config_path_out.is_file():
            write_yaml(
                config_path_out,
                _wp4_config(config, wp1, model_set, run_dir),
            )
        generated.append(config_path_out)
        if not (resume and (run_dir / "wp4_manifest.yaml").is_file()):
            run_wp4(config_path_out, repo_root=repo_root)

    aggregate_generator_only(output, seeds)
    gate = _paired_summary(
        _resolve(repo_root, config["wp1_output"]),
        output,
        _resolve(repo_root, config["smooth_bounded_output"]),
        config["rejection_contract"],
    )
    manifest_path = output / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "complete",
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "generated_configs": [str(path) for path in generated],
            "rejection_gate": gate,
        }
    )
    write_yaml(manifest_path, manifest)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/stage2/s2_wp2_rejection.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = run(
        args.config,
        output_dir_override=args.output_dir,
        resume=args.resume,
    )
    manifest = yaml.safe_load((output / "manifest.yaml").read_text(encoding="utf-8"))
    print(json.dumps({"output": str(output), "status": manifest["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
