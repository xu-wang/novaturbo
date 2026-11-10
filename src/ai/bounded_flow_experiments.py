"""Evaluate nflows/cINN bounded-output variants without tuning on the extreme target."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from .baseline import sha256_file, write_json, write_yaml
from .candidate_protocol import V21_RAW_CANDIDATE_COUNT, apply_candidate_protocol_v21
from .condition_support import ConditionSupportModel
from .generator_dataset import GeneratorNormalizer, load_schema
from .generators import load_generator
from .round_trip import RoundTripEvaluator, derive_round_trip_metrics, load_metrics_config


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _stable_seed(base_seed: int, *parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int(base_seed) + int.from_bytes(digest[:4], "big") % 1_000_000


def load_bounded_flow_experiment_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"version", "data", "targets", "arms", "runtime", "output"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"bounded-flow config is missing keys: {sorted(missing)}")
    if int(config["runtime"].get("n_raw", 0)) != V21_RAW_CANDIDATE_COUNT:
        raise ValueError("bounded-flow experiments require runtime.n_raw=128")
    if not config["arms"]:
        raise ValueError("bounded-flow experiments require at least one arm")
    for arm in config["arms"]:
        if str(arm.get("model")) not in {"nflows", "cinn"}:
            raise ValueError("bounded-flow arms must use nflows or cinn")
        if not arm.get("checkpoint_path"):
            raise ValueError("every bounded-flow arm requires checkpoint_path")
    return config


def _load_targets(repo_root: Path, config: Mapping[str, Any], mode: str) -> pd.DataFrame:
    suite_path = _resolve(repo_root, str(config["suite_csv"]))
    frozen_path = _resolve(repo_root, str(config["frozen_ids"]))
    suite = pd.read_csv(suite_path)
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    if sha256_file(suite_path) != str(frozen["source_target_suite_sha256"]):
        raise ValueError("target suite differs from the frozen S2-WP1 selection")
    ids = [str(value) for value in frozen["target_ids"]]
    extreme_id = str(config["extreme_holdout_target_id"])
    if extreme_id not in ids:
        raise ValueError("extreme holdout is not a frozen target")
    selected_ids = [extreme_id] if mode == "extreme" else [value for value in ids if value != extreme_id]
    selected = suite.loc[suite["target_id"].astype(str).isin(selected_ids)].copy()
    order = {target_id: index for index, target_id in enumerate(selected_ids)}
    selected["target_order"] = selected["target_id"].astype(str).map(order)
    selected = selected.sort_values("target_order").reset_index(drop=True)
    maximum = config.get("maximum_targets_per_type") if mode == "development" else None
    if maximum is not None:
        selected = (
            selected.groupby("target_type", sort=False, group_keys=False)
            .head(int(maximum))
            .reset_index(drop=True)
        )
    return selected


def _support_model(
    dataset_path: Path,
    normalizer: GeneratorNormalizer,
    config: Mapping[str, Any],
) -> ConditionSupportModel:
    frame = pd.read_csv(dataset_path)
    names = normalizer.condition_names
    train = frame.loc[frame["split"] == "train", names]
    validation = frame.loc[frame["split"] == "val", names]
    return ConditionSupportModel.fit(
        normalizer.transform_conditions(train),
        normalizer.transform_conditions(validation),
        neighbor_count=int(config.get("neighbor_count", 5)),
        quantile=float(config.get("quantile", 0.99)),
    )


def _first_hit(frame: pd.DataFrame) -> float:
    if "within_tolerance" not in frame:
        return math.inf
    positions = np.flatnonzero(frame["within_tolerance"].astype(bool).to_numpy())
    return float(positions[0] + 1) if len(positions) else math.inf


def run_bounded_flow_experiments(
    config_path: Path,
    *,
    mode: str,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
) -> Path:
    if mode not in {"development", "extreme"}:
        raise ValueError("mode must be development or extreme")
    repo_root = (repo_root or Path.cwd()).resolve()
    config_path = Path(config_path).resolve()
    config = load_bounded_flow_experiment_config(config_path)
    output_dir = (
        output_dir_override.resolve()
        if output_dir_override is not None
        else _resolve(repo_root, str(config["output"][f"{mode}_directory"]))
    )
    if output_dir.exists():
        raise FileExistsError(f"bounded-flow output exists: {output_dir}")
    output_dir.mkdir(parents=True)

    data = config["data"]
    dataset_path = _resolve(repo_root, str(data["dataset_csv"]))
    schema_path = _resolve(repo_root, str(data["schema"]))
    normalizer_path = _resolve(repo_root, str(data["normalizer"]))
    metrics_path = _resolve(repo_root, str(data["metrics_config"]))
    normalizer = GeneratorNormalizer.load(normalizer_path)
    evaluator = RoundTripEvaluator(
        load_schema(schema_path),
        normalizer,
        load_metrics_config(metrics_path),
        derive_round_trip_metrics(dataset_path, normalizer, load_metrics_config(metrics_path)),
    )
    support = _support_model(dataset_path, normalizer, config.get("support", {}))
    targets = _load_targets(repo_root, config["targets"], mode)
    targets.to_csv(output_dir / "evaluated_targets.csv", index=False)
    rows = []
    checkpoint_rows = []
    device = str(config["runtime"].get("device", "auto"))
    n_raw = int(config["runtime"]["n_raw"])
    seed_base = int(config["runtime"]["sample_seed_base"])
    started = dt.datetime.now(dt.timezone.utc)

    for arm in config["arms"]:
        checkpoint = _resolve(repo_root, str(arm["checkpoint_path"]))
        if not checkpoint.is_file():
            raise FileNotFoundError(f"bounded-flow checkpoint is missing: {checkpoint}")
        generator = load_generator(checkpoint, device=device)
        if generator.generator_name != str(arm["model"]):
            raise ValueError(f"checkpoint model differs for arm={arm['arm_id']}")
        checkpoint_rows.append(
            {
                "arm_id": arm["arm_id"],
                "model": arm["model"],
                "training_seed": int(arm["training_seed"]),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
        for target in targets.to_dict(orient="records"):
            target_id = str(target["target_id"])
            normalized_target = evaluator.normalize_target(target)
            support_values = support.diagnose(normalized_target)
            sample_seed = _stable_seed(
                seed_base, arm["model"], arm["training_seed"], target_id
            )
            sampled = generator.sample(
                normalized_target, n_raw, seed=sample_seed, space="normalized"
            )
            normalized = sampled.values.numpy()
            physical = normalizer.inverse_design(normalized)
            candidate_ids = np.arange(n_raw, dtype=int)
            validation = evaluator.validate_candidates(
                normalized, physical, candidate_ids=candidate_ids
            )
            protocol = apply_candidate_protocol_v21(
                validation["candidate_id"].to_numpy(int),
                validation["candidate_valid"].to_numpy(bool),
            )
            kept = np.asarray(protocol.kept_positions, dtype=int)
            if len(kept):
                result = evaluator.evaluate(
                    target,
                    normalized[kept],
                    physical[kept],
                    candidate_ids=np.asarray(protocol.kept_candidate_ids, dtype=int),
                )
                forward = result.forward_results
                best_error = float(result.metrics.get("best", {}).get("total_error", math.inf))
                hit_count = int(forward["within_tolerance"].astype(bool).sum())
                forward_valid = int(forward["forward_valid"].astype(bool).sum())
            else:
                forward = pd.DataFrame()
                best_error = math.inf
                hit_count = 0
                forward_valid = 0
            outside = ((normalized < -1.0) | (normalized > 1.0)).any(axis=1)
            near_boundary = (np.abs(normalized) >= 0.99).any(axis=1)
            rows.append(
                {
                    "mode": mode,
                    "arm_id": str(arm["arm_id"]),
                    "model": str(arm["model"]),
                    "training_seed": int(arm["training_seed"]),
                    "target_id": target_id,
                    "target_type": str(target["target_type"]),
                    "sample_seed": sample_seed,
                    **support_values,
                    "N_raw": protocol.n_raw,
                    "N_valid": protocol.n_valid,
                    "N_keep": protocol.n_keep,
                    "N_trimmed": protocol.n_trimmed,
                    "out_of_bounds_rate": float(outside.mean()),
                    "near_boundary_rate": float(near_boundary.mean()),
                    "sample_max_abs": float(np.abs(normalized).max()),
                    "forward_valid_count": forward_valid,
                    "hit_count": hit_count,
                    "first_hit_evaluation": _first_hit(forward),
                    "best_total_error": best_error,
                    "sampling_diagnostics": json.dumps(sampled.diagnostics, sort_keys=True),
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "target_summary.csv", index=False)
    grouped = (
        summary.groupby(["arm_id", "model", "target_type"], sort=True)
        .agg(
            run_count=("target_id", "size"),
            zero_valid_rate=("N_valid", lambda values: float((values == 0).mean())),
            valid_rate=("N_valid", lambda values: float(values.sum() / (len(values) * n_raw))),
            hit_rate=("hit_count", lambda values: float((values > 0).mean())),
            best_error_median=("best_total_error", "median"),
            first_hit_median=("first_hit_evaluation", "median"),
            near_boundary_rate_mean=("near_boundary_rate", "mean"),
            sample_max_abs_max=("sample_max_abs", "max"),
        )
        .reset_index()
    )
    grouped.to_csv(output_dir / "aggregate_summary.csv", index=False)
    pd.DataFrame(checkpoint_rows).to_csv(output_dir / "checkpoint_manifest.csv", index=False)
    manifest = {
        "version": config["version"],
        "mode": mode,
        "status": "completed",
        "started_utc": started.isoformat(),
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target_count": len(targets),
        "arm_count": len(config["arms"]),
        "run_count": len(summary),
        "candidate_protocol": "v2_1_fixed_raw",
        "extreme_target_used_for_selection": mode == "extreme",
        "config_sha256": sha256_file(config_path),
    }
    write_yaml(output_dir / "manifest.yaml", manifest)
    write_json(
        output_dir / "support_calibration.json",
        {
            "neighbor_count": support.neighbor_count,
            "distance_threshold": support.distance_threshold,
            "max_abs_threshold": support.max_abs_threshold,
        },
    )
    return output_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("development", "extreme"), required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    output = run_bounded_flow_experiments(
        args.config, mode=args.mode, output_dir_override=args.output_dir
    )
    print(json.dumps({"output": str(output), "status": "completed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
