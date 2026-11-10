"""WP4 orchestration for physics-in-the-loop generator validation."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from .baseline import (
    collect_environment,
    collect_git_state,
    collect_source_manifest,
    sha256_file,
    write_json,
    write_yaml,
)
from .candidate_protocol import (
    V21_RAW_CANDIDATE_COUNT,
    apply_candidate_protocol_v21,
)
from .generator_dataset import GeneratorNormalizer, load_schema
from .generators import load_generator
from .round_trip import (
    FAILURE_CODES,
    OK,
    TARGET_OUT_OF_DOMAIN,
    DerivedRoundTripMetrics,
    RoundTripEvaluator,
    RoundTripResult,
    derive_round_trip_metrics,
    load_metrics_config,
    sampling_budget_exhausted_result,
    target_out_of_domain_result,
)


ORDINARY_TARGET_TYPES = {"ID", "Interpolation"}


def load_wp4_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "version",
        "metrics_config",
        "data",
        "models",
        "runtime",
        "output",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"WP4 config is missing keys: {sorted(missing)}")
    if not config["models"]:
        raise ValueError("WP4 config must define at least one model checkpoint")
    for model_label, model_value in config["models"].items():
        if isinstance(model_value, str):
            continue
        if not isinstance(model_value, Mapping):
            raise ValueError(
                f"WP4 model {model_label} must be a checkpoint path or mapping"
            )
        unsupported_model_fields = set(model_value).difference(
            {"checkpoint", "generator_name", "generator_overrides", "sampling"}
        )
        if unsupported_model_fields or not model_value.get("checkpoint"):
            raise ValueError(
                f"WP4 model {model_label} mapping requires checkpoint and only "
                "supports checkpoint/generator_name/generator_overrides/sampling"
            )
        model_overrides = dict(model_value.get("generator_overrides", {}))
        unsupported_model_overrides = set(model_overrides).difference(
            {"ode_steps", "ode_method"}
        )
        if unsupported_model_overrides:
            raise ValueError(
                f"WP4 model {model_label} has unsupported generator overrides: "
                f"{sorted(unsupported_model_overrides)}"
            )
        model_sampling = dict(model_value.get("sampling", {}))
        unsupported_sampling_fields = set(model_sampling).difference(
            {
                "policy",
                "draw_count",
                "reranker_checkpoint",
                "reranker_ode_method",
                "reranker_ode_steps",
            }
        )
        if unsupported_sampling_fields:
            raise ValueError(
                f"WP4 model {model_label} has unsupported sampling fields: "
                f"{sorted(unsupported_sampling_fields)}"
            )
        if model_sampling and str(model_sampling.get("policy")) != "diag_rerank_fixed_raw":
            raise ValueError(
                "per-model sampling currently supports only diag_rerank_fixed_raw"
            )
        if model_sampling and (
            int(model_sampling.get("draw_count", 0)) < 128
            or not model_sampling.get("reranker_checkpoint")
        ):
            raise ValueError(
                "diag_rerank_fixed_raw requires draw_count>=128 and reranker_checkpoint"
            )
    if int(config["runtime"]["sample_count"]) <= 0:
        raise ValueError("WP4 runtime.sample_count must be positive")
    maximum = config["runtime"].get("maximum_targets_per_type")
    if maximum is not None and int(maximum) <= 0:
        raise ValueError("maximum_targets_per_type must be null or positive")
    sampling = config["runtime"].get("sampling", {})
    policy = str(sampling.get("policy", "direct"))
    if policy not in {
        "direct",
        "bounded_rejection",
        "v2_1_fixed_raw",
        "diag_rerank_fixed_raw",
    }:
        raise ValueError(
            "runtime.sampling.policy must be direct, bounded_rejection, "
            "v2_1_fixed_raw, or diag_rerank_fixed_raw"
        )
    if policy in {"v2_1_fixed_raw", "diag_rerank_fixed_raw"} and int(
        config["runtime"]["sample_count"]
    ) != 128:
        raise ValueError("V2.1 sampling requires runtime.sample_count=128")
    if policy == "bounded_rejection":
        if int(sampling.get("batch_size", config["runtime"]["sample_count"])) <= 0:
            raise ValueError("runtime.sampling.batch_size must be positive")
        if float(sampling.get("max_draw_multiplier", 1.0)) < 1.0:
            raise ValueError(
                "runtime.sampling.max_draw_multiplier must be at least one"
            )
        if float(sampling.get("lower_bound", -1.0)) >= float(
            sampling.get("upper_bound", 1.0)
        ):
            raise ValueError("runtime.sampling lower_bound must be below upper_bound")
        if float(sampling.get("tolerance", 0.0)) < 0.0:
            raise ValueError("runtime.sampling.tolerance must be non-negative")
    generator_overrides = dict(config["runtime"].get("generator_overrides", {}))
    unsupported_overrides = set(generator_overrides).difference(
        {"ode_steps", "ode_method"}
    )
    if unsupported_overrides:
        raise ValueError(
            "runtime.generator_overrides only supports ode_steps and ode_method; got "
            f"{sorted(unsupported_overrides)}"
        )
    if "ode_steps" in generator_overrides and int(
        generator_overrides["ode_steps"]
    ) not in {8, 16, 30, 32, 64}:
        raise ValueError(
            "CFM ode_steps override must be 8, 16, 32, or 64; "
            "Diag-CFM additionally allows 30"
        )
    if "ode_method" in generator_overrides and str(
        generator_overrides["ode_method"]
    ) not in {"euler", "midpoint", "rk4", "dopri5"}:
        raise ValueError(
            "ode_method override must be euler, midpoint, rk4, or dopri5"
        )
    return config


def _resolve_model_spec(
    model_label: str, model_value: Any
) -> tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    if isinstance(model_value, str):
        return model_value, model_label, {}, {}
    return (
        str(model_value["checkpoint"]),
        str(model_value.get("generator_name", model_label)),
        dict(model_value.get("generator_overrides", {})),
        dict(model_value.get("sampling", {})),
    )


def _apply_generator_overrides(
    generator: Any, runtime: Mapping[str, Any]
) -> Dict[str, Any]:
    overrides = dict(runtime.get("generator_overrides", {}))
    if not overrides:
        return {}
    if getattr(generator, "generator_name", None) not in {"cfm", "diag_cfm"}:
        raise ValueError("generator_overrides are restricted to CFM-family ODE models")
    audit: Dict[str, Any] = {}
    if "ode_steps" in overrides:
        original_steps = int(generator.model_config.get("ode_steps", 32))
        effective_steps = int(overrides["ode_steps"])
        generator.model_config["ode_steps"] = effective_steps
        audit.update(
            {
                "ode_steps_original": original_steps,
                "ode_steps_effective": effective_steps,
            }
        )
    if "ode_method" in overrides:
        original_method = str(generator.model_config.get("ode_method", "rk4"))
        effective_method = str(overrides["ode_method"])
        generator.model_config["ode_method"] = effective_method
        audit.update(
            {
                "ode_method_original": original_method,
                "ode_method_effective": effective_method,
            }
        )
    return audit


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for WP4 but is unavailable")
    return requested


def _target_seed(base_seed: int, model: str, target_id: str) -> int:
    payload = f"{base_seed}:{model}:{target_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _rejection_batch_seed(sample_seed: int, batch_index: int) -> int:
    if batch_index == 0:
        return int(sample_seed)
    payload = f"{sample_seed}:bounded_rejection:{batch_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _sample_normalized_candidates(
    generator: Any,
    normalized_target: np.ndarray,
    sample_count: int,
    sample_seed: int,
    runtime: Mapping[str, Any],
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Sample directly or refill an in-bound pool through cheap rejection."""

    sampling = dict(runtime.get("sampling", {}))
    policy = str(sampling.get("policy", "direct"))
    started = time.perf_counter()
    if policy == "diag_rerank_fixed_raw":
        draw_count = int(sampling.get("draw_count", sample_count))
        if draw_count < sample_count:
            raise ValueError("diag_rerank draw_count must be at least sample_count")
        reranker = runtime.get("_diag_reranker")
        if reranker is None:
            raise ValueError("diag_rerank sampling requires a loaded Diag-CFM reranker")
        generated = generator.sample(
            normalized_target,
            draw_count,
            seed=sample_seed,
            space="normalized",
        )
        pool = generated.values.detach().cpu().numpy().astype(float)
        scoring_started = time.perf_counter()
        predicted = reranker.predict_conditions(pool).detach().cpu().numpy()
        scores = np.mean((predicted - normalized_target[None, :]) ** 2, axis=1)
        scoring_seconds = time.perf_counter() - scoring_started
        kept_positions = np.argsort(scores, kind="stable")[:sample_count]
        values = pool[kept_positions]
        selected_scores = scores[kept_positions]
        generator_wall = float(
            generated.diagnostics.get(
                "generation_wall_seconds",
                generated.diagnostics.get("wall_seconds", np.nan),
            )
        )
        diagnostics = {
            "sampling_policy": policy,
            "requested_count": int(sample_count),
            "returned_count": int(len(values)),
            "network_draw_count": int(draw_count),
            "batch_count": 1,
            "sampling_elapsed_seconds": time.perf_counter() - started,
            "budget_exhausted": False,
            "reranker_pool_count": int(draw_count),
            "reranker_score": "reverse_condition_mse",
            "generator_diagnostics": {
                **generated.diagnostics,
                "generation_wall_seconds": generator_wall,
                "diagnosis_wall_seconds": scoring_seconds,
                "candidate_diagnostics": {
                    "diag_reverse_condition_mse": selected_scores.tolist()
                },
            },
        }
        return values, diagnostics
    if policy in {"direct", "v2_1_fixed_raw"}:
        generated = generator.sample(
            normalized_target,
            sample_count,
            seed=sample_seed,
            space="normalized",
        )
        values = generated.values.detach().cpu().numpy().astype(float)
        diagnostics = {
            "sampling_policy": policy,
            "requested_count": int(sample_count),
            "returned_count": int(len(values)),
            "network_draw_count": int(len(values)),
            "batch_count": 1,
            "sampling_elapsed_seconds": time.perf_counter() - started,
            "budget_exhausted": False,
            "generator_diagnostics": generated.diagnostics,
        }
        return values, diagnostics

    if policy != "bounded_rejection":
        raise ValueError(f"Unsupported sampling policy: {policy}")

    lower = float(sampling.get("lower_bound", -1.0))
    upper = float(sampling.get("upper_bound", 1.0))
    tolerance = float(sampling.get("tolerance", 0.0))
    batch_size = int(sampling.get("batch_size", sample_count))
    max_draws = max(
        int(sample_count),
        int(math.ceil(sample_count * float(sampling.get("max_draw_multiplier", 1.0)))),
    )
    accepted_batches: List[np.ndarray] = []
    accepted_seen = 0
    network_draws = 0
    rejected_out_of_bounds = 0
    generator_batch_diagnostics: List[Dict[str, Any]] = []
    batch_index = 0
    while accepted_seen < sample_count and network_draws < max_draws:
        draw_count = (
            sample_count
            if batch_index == 0
            else min(batch_size, max_draws - network_draws)
        )
        draw_count = min(draw_count, max_draws - network_draws)
        generated = generator.sample(
            normalized_target,
            draw_count,
            seed=_rejection_batch_seed(sample_seed, batch_index),
            space="normalized",
        )
        values = generated.values.detach().cpu().numpy().astype(float)
        finite = np.isfinite(values).all(axis=1)
        in_bounds = finite & (values >= lower - tolerance).all(axis=1) & (
            values <= upper + tolerance
        ).all(axis=1)
        accepted = values[in_bounds]
        accepted_batches.append(accepted)
        accepted_seen += int(len(accepted))
        network_draws += int(len(values))
        rejected_out_of_bounds += int((~in_bounds).sum())
        generator_batch_diagnostics.append(
            {
                "batch_index": batch_index,
                "seed": _rejection_batch_seed(sample_seed, batch_index),
                "draw_count": int(len(values)),
                "accepted_count": int(len(accepted)),
                "generator_diagnostics": generated.diagnostics,
            }
        )
        batch_index += 1

    all_accepted = (
        np.concatenate(accepted_batches, axis=0)
        if accepted_batches
        else np.empty((0, int(generator.design_dim)), dtype=float)
    )
    returned = all_accepted[:sample_count]
    diagnostics = {
        "sampling_policy": "bounded_rejection",
        "requested_count": int(sample_count),
        "returned_count": int(len(returned)),
        "network_draw_count": int(network_draws),
        "range_accepted_count": int(accepted_seen),
        "range_rejected_count": int(rejected_out_of_bounds),
        "range_acceptance_rate": (
            float(accepted_seen / network_draws) if network_draws else 0.0
        ),
        "unused_in_bounds_count": int(max(accepted_seen - len(returned), 0)),
        "batch_count": int(batch_index),
        "max_draws": int(max_draws),
        "sampling_elapsed_seconds": time.perf_counter() - started,
        "budget_exhausted": bool(len(returned) < sample_count),
        "bounds": {
            "lower": lower,
            "upper": upper,
            "tolerance": tolerance,
        },
        "generator_batches": generator_batch_diagnostics,
    }
    return returned, diagnostics


def _safe_target_payload(target: Mapping[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in target.items():
        if pd.isna(value):
            payload[key] = None
        elif isinstance(value, np.generic):
            payload[key] = value.item()
        else:
            payload[key] = value
    return payload


def _select_targets(frame: pd.DataFrame, maximum_per_type: Optional[int]) -> pd.DataFrame:
    required = {
        "target_id",
        "target_type",
        "expected_status",
        "thrust_N",
        "tsfc_kg_N_s",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Target suite is missing fields: {sorted(missing)}")
    if frame["target_id"].duplicated().any():
        raise ValueError("Target suite contains duplicate target_id values")
    if maximum_per_type is None:
        return frame.reset_index(drop=True)
    return (
        frame.groupby("target_type", sort=False, group_keys=False)
        .head(int(maximum_per_type))
        .reset_index(drop=True)
    )


def _raw_candidate_frame(
    normalized: np.ndarray,
    physical: np.ndarray,
    evaluator: RoundTripEvaluator,
    model: str,
    target: Mapping[str, Any],
    sample_seed: int,
    generator_diagnostics: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    candidate_diagnostics = dict(
        (generator_diagnostics or {}).get("candidate_diagnostics", {})
    )
    for name, values in candidate_diagnostics.items():
        if len(values) != len(normalized):
            raise ValueError(
                f"candidate diagnostic {name} has {len(values)} values for "
                f"{len(normalized)} candidates"
            )
    records = []
    for index, (norm_row, physical_row) in enumerate(zip(normalized, physical)):
        record: Dict[str, Any] = {
            "model": model,
            "target_id": target["target_id"],
            "target_type": target["target_type"],
            "sample_seed": sample_seed,
            "candidate_id": index,
        }
        record.update(
            {
                f"normalized_{name}": norm_row[field_index]
                for field_index, name in enumerate(evaluator.design_names)
            }
        )
        record.update(
            {
                name: physical_row[field_index]
                for field_index, name in enumerate(evaluator.design_names)
            }
        )
        for name, values in candidate_diagnostics.items():
            record[f"intrinsic_{name}"] = float(values[index])
        records.append(record)
    return pd.DataFrame(records)


def _add_context(
    frame: pd.DataFrame,
    model: str,
    target: Mapping[str, Any],
    sample_seed: int,
) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "sample_seed", sample_seed)
    result.insert(0, "target_type", target["target_type"])
    result.insert(0, "target_id", target["target_id"])
    result.insert(0, "model", model)
    return result


def _target_summary_row(
    model: str,
    target: Mapping[str, Any],
    sample_seed: int,
    result: RoundTripResult,
    sample_diagnostics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    metrics = result.metrics
    best = metrics["best"]
    row: Dict[str, Any] = {
        "model": model,
        "target_id": target["target_id"],
        "target_type": target["target_type"],
        "expected_status": target["expected_status"],
        "sample_seed": sample_seed,
        "target_thrust_N": target["thrust_N"],
        "target_tsfc_kg_N_s": target["tsfc_kg_N_s"],
        "target_total_mass_kg": target.get("total_mass_kg", np.nan),
        "candidate_count": metrics["candidate_count"],
        "candidate_valid_count": metrics["candidate_valid_count"],
        "forward_valid_count": metrics["forward_valid_count"],
        "duplicate_count": metrics["duplicate_count"],
        "best_candidate_id": best["candidate_id"],
        "best_thrust_relative_error": best["thrust_relative_error"],
        "best_tsfc_relative_error": best["tsfc_relative_error"],
        "best_mass_relative_error": best.get("mass_relative_error", np.nan),
        "best_total_error": best["total_error"],
        "best_within_tolerance": best["within_tolerance"],
        "is_training_domain_target": metrics["abstention"][
            "is_training_domain_target"
        ],
        "is_ood_target": metrics["abstention"].get("is_ood_target", False),
        "is_rejection_target": metrics["abstention"]["is_rejection_target"],
        "accepted": metrics["abstention"].get("accepted", False),
        "rejected": metrics["abstention"]["rejected"],
        "false_accept": metrics["abstention"]["false_accept"],
        "selected_count": metrics["diversity"].get("selected_count", 0),
        "mean_nearest_neighbour_distance": metrics["diversity"].get(
            "mean_nearest_neighbour_distance", np.nan
        ),
    }
    for requested, top_k in metrics.get("top_k", {}).items():
        row[f"top_{requested}_effective_k"] = top_k["effective_k"]
        row[f"top_{requested}_success"] = top_k["success"]
        row[f"top_{requested}_best_total_error"] = top_k["best_total_error"]
    diagnostics = dict(sample_diagnostics or {})
    intrinsic = dict(diagnostics.get("generator_diagnostics", {}))
    valid_ranked = result.ranked_candidates.loc[
        result.ranked_candidates.get(
            "forward_valid", pd.Series(False, index=result.ranked_candidates.index)
        ).astype(bool)
    ]
    errors = pd.to_numeric(
        valid_ranked.get(
            "total_error", pd.Series(dtype=float, index=valid_ranked.index)
        ),
        errors="coerce",
    ).dropna()
    top_k = errors.head(min(10, len(errors)))
    failure_counts = metrics.get("failure_counts", {})
    actual_0d = int(diagnostics.get("actual_0d_evaluations", metrics["candidate_valid_count"]))
    success_0d = int(metrics["forward_valid_count"])
    row.update(
        {
            "sampling_policy": diagnostics.get("sampling_policy", "direct"),
            "network_draw_count": diagnostics.get(
                "network_draw_count", metrics["candidate_count"]
            ),
            "range_accepted_count": diagnostics.get(
                "range_accepted_count", metrics["candidate_valid_count"]
            ),
            "range_rejected_count": diagnostics.get("range_rejected_count", 0),
            "range_acceptance_rate": diagnostics.get("range_acceptance_rate", np.nan),
            "sampling_batch_count": diagnostics.get("batch_count", 1),
            "sampling_elapsed_seconds": diagnostics.get(
                "sampling_elapsed_seconds", np.nan
            ),
            "sampling_budget_exhausted": diagnostics.get(
                "budget_exhausted", False
            ),
            "N_raw": int(diagnostics.get("N_raw", metrics["candidate_count"])),
            "N_valid": int(
                diagnostics.get("N_valid", metrics["candidate_valid_count"])
            ),
            "N_keep": int(diagnostics.get("N_keep", metrics["candidate_count"])),
            "N_trimmed": int(diagnostics.get("N_trimmed", 0)),
            "valid_rate": float(diagnostics.get("valid_rate", np.nan)),
            "generator_time": float(
                diagnostics.get(
                    "generator_time",
                    intrinsic.get(
                        "generation_wall_seconds",
                        diagnostics.get("sampling_elapsed_seconds", np.nan),
                    ),
                )
            ),
            "intrinsic_diagnosis_time": float(
                intrinsic.get("diagnosis_wall_seconds", 0.0)
            ),
            "intrinsic_zero_deviation_mean": float(
                intrinsic.get("zero_deviation_mean", np.nan)
            ),
            "intrinsic_self_consistency_mean": float(
                intrinsic.get("self_consistency_mean", np.nan)
            ),
            "0d_time": float(diagnostics.get("0d_time", np.nan)),
            "actual_0d_evaluations": actual_0d,
            "0d_success_count": success_0d,
            "0d_failure_count": max(actual_0d - success_0d, 0),
            "0d_failure_reason": json.dumps(failure_counts, sort_keys=True),
            "topk_error": float(top_k.mean()) if len(top_k) else np.nan,
            "median_error": float(errors.median()) if len(errors) else np.nan,
            "q25_error": float(errors.quantile(0.25)) if len(errors) else np.nan,
            "q75_error": float(errors.quantile(0.75)) if len(errors) else np.nan,
            "mean_error": float(errors.mean()) if len(errors) else np.nan,
            "error_std": float(errors.std(ddof=0)) if len(errors) else np.nan,
            "hit_rate": float(valid_ranked["within_tolerance"].mean())
            if len(valid_ranked)
            else 0.0,
        }
    )
    return row


def _write_run_report(
    path: Path,
    model: str,
    target: Mapping[str, Any],
    result: RoundTripResult,
) -> None:
    metrics = result.metrics
    best = metrics["best"]
    lines = [
        "# WP4 Physics-in-the-loop Round-Trip Report",
        "",
        f"- Model: {model}",
        f"- Target: {target['target_id']}",
        f"- Target type: {target['target_type']}",
        f"- Expected status: {target['expected_status']}",
        f"- Candidate count: {metrics['candidate_count']}",
        f"- Candidate-valid count: {metrics['candidate_valid_count']}",
        f"- Brayton-valid count: {metrics['forward_valid_count']}",
        f"- Best total error: {best['total_error']}",
        f"- Best within tolerance: {best['within_tolerance']}",
        f"- Rejected: {metrics['abstention']['rejected']}",
        f"- False accept: {metrics['abstention']['false_accept']}",
        "",
        "## Top-K",
        "",
    ]
    for requested, entry in metrics.get("top_k", {}).items():
        lines.append(
            f"- Top-{requested}: effective_k={entry['effective_k']}, "
            f"success={entry['success']}, best_total_error={entry['best_total_error']}"
        )
    lines.extend(["", "## Failure counts", ""])
    if metrics["failure_counts"]:
        for code, count in metrics["failure_counts"].items():
            lines.append(f"- {code}: {count}")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_run_artifacts(
    run_dir: Path,
    target: Mapping[str, Any],
    raw: pd.DataFrame,
    result: RoundTripResult,
    model: str,
    sample_seed: int,
    sample_diagnostics: Mapping[str, Any],
    protocol_validation: Optional[pd.DataFrame] = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "target.json", _safe_target_payload(target))
    raw.to_csv(run_dir / "raw_candidates.csv", index=False)
    if protocol_validation is not None:
        protocol_validation.to_csv(
            run_dir / "protocol_validation.csv", index=False
        )
    _add_context(result.checked_candidates, model, target, sample_seed).to_csv(
        run_dir / "checked_candidates.csv", index=False
    )
    _add_context(result.forward_results, model, target, sample_seed).to_csv(
        run_dir / "forward_results.csv", index=False
    )
    _add_context(result.ranked_candidates, model, target, sample_seed).to_csv(
        run_dir / "ranked_candidates.csv", index=False
    )
    result.failures.to_csv(run_dir / "failures.csv", index=False)
    metrics = dict(result.metrics)
    metrics["sample_diagnostics"] = dict(sample_diagnostics)
    write_json(run_dir / "metrics.json", metrics)
    _write_run_report(run_dir / "report.md", model, target, result)


def _aggregate_model_summary(target_summary: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for model, frame in target_summary.groupby("model", sort=False):
        ordinary = frame.loc[frame["target_type"].isin(ORDINARY_TARGET_TYPES)]
        ood = frame.loc[frame["is_ood_target"]]
        rejection = frame.loc[frame["is_rejection_target"]]
        record: Dict[str, Any] = {
            "model": model,
            "target_count": len(frame),
            "ordinary_target_count": len(ordinary),
            "ordinary_forward_valid_target_rate": float(
                (ordinary["forward_valid_count"] > 0).mean()
            ) if len(ordinary) else np.nan,
            "ordinary_best_within_tolerance_rate": float(
                ordinary["best_within_tolerance"].mean()
            ) if len(ordinary) else np.nan,
            "ordinary_median_best_total_error": float(
                ordinary["best_total_error"].replace([np.inf, -np.inf], np.nan).median()
            ) if len(ordinary) else np.nan,
            "rejection_target_count": len(rejection),
            "rejection_rate": float(rejection["rejected"].mean()) if len(rejection) else np.nan,
            "false_accept_rate": float(rejection["false_accept"].mean()) if len(rejection) else np.nan,
            "ood_target_count": len(ood),
            "ood_success_rate": float(ood["accepted"].mean()) if len(ood) else np.nan,
            "ood_low_confidence_rate": float(ood["rejected"].mean()) if len(ood) else np.nan,
            "mean_valid_candidate_rate": float(
                (frame["forward_valid_count"] / frame["candidate_count"].replace(0, np.nan)).mean()
            ),
            "mean_selected_diversity": float(
                frame["mean_nearest_neighbour_distance"].mean()
            ),
        }
        for column in sorted(
            value for value in frame.columns if value.startswith("top_") and value.endswith("_success")
        ):
            record[f"ordinary_{column}_rate"] = float(ordinary[column].mean()) if len(ordinary) else np.nan
        records.append(record)
    return pd.DataFrame(records)


def _functional_gate(
    configured_models: Sequence[str],
    model_records: Sequence[Mapping[str, Any]],
    target_summary: pd.DataFrame,
    failures: pd.DataFrame,
) -> Dict[str, Any]:
    loaded = {
        str(record["model"])
        for record in model_records
        if record.get("load_status") == "passed"
    }
    unknown_failure_codes = sorted(
        set(failures.get("failure_code", pd.Series(dtype=str)).dropna().astype(str))
        .difference(FAILURE_CODES)
    )
    model_checks = {}
    for model in configured_models:
        model_targets = target_summary.loc[target_summary["model"] == model]
        ordinary = model_targets.loc[
            model_targets["target_type"].isin(ORDINARY_TARGET_TYPES)
        ]
        model_checks[model] = {
            "checkpoint_loaded": model in loaded,
            "target_runs_present": bool(len(model_targets)),
            "ordinary_forward_result_present": bool(
                len(ordinary) and (ordinary["forward_valid_count"] > 0).any()
            ),
        }
    passed = bool(
        all(all(check.values()) for check in model_checks.values())
        and not unknown_failure_codes
    )
    return {
        "gate": "G1 Generator Functional / Physics Round-Trip Gate",
        "passed": passed,
        "performance_superiority_required": False,
        "model_checks": model_checks,
        "all_failures_use_registered_codes": not unknown_failure_codes,
        "unknown_failure_codes": unknown_failure_codes,
    }


def _write_final_report(
    path: Path,
    gate: Mapping[str, Any],
    derived: DerivedRoundTripMetrics,
    model_summary: pd.DataFrame,
) -> None:
    lines = [
        "# WP4 Physics-in-the-loop Round-Trip 验收报告",
        "",
        f"- G1 functional gate: {'PASS' if gate['passed'] else 'FAIL'}",
        "- Generator > Random required: no",
        f"- Metrics version: {derived.metrics_version}",
        f"- Thrust relative tolerance: {derived.thrust_relative_tolerance:.8g}",
        f"- TSFC relative tolerance: {derived.tsfc_relative_tolerance:.8g}",
        f"- Weighted total tolerance: {derived.total_error_tolerance:.8g}",
        "",
        "## Model results",
        "",
    ]
    for row in model_summary.to_dict(orient="records"):
        lines.append(
            f"- {row['model']}: targets={row['target_count']}, "
            f"ordinary_forward_valid_rate={row['ordinary_forward_valid_target_rate']:.4f}, "
            f"ordinary_success_rate={row['ordinary_best_within_tolerance_rate']:.4f}, "
            f"ood_success_rate={row['ood_success_rate']:.4f}, "
            f"infeasible_false_accept_rate={row['false_accept_rate']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Gate interpretation",
            "",
            "G1 only verifies that every frozen WP3 model can be loaded, sampled, "
            "mapped into the Brayton solver, evaluated with one shared metric "
            "implementation, and traced with registered failure codes. Accuracy, "
            "diversity, rejection, and Top-K values are diagnostic baselines and do "
            "not create a Generator-vs-Random Go/No-Go decision in WP4.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_wp4(
    config_path: Path,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    config_path = Path(config_path).resolve()
    config = load_wp4_config(config_path)
    metrics_path = (repo_root / config["metrics_config"]).resolve()
    metrics_config = load_metrics_config(metrics_path)
    schema_path = (repo_root / config["data"]["schema"]).resolve()
    dataset_path = (repo_root / config["data"]["dataset_csv"]).resolve()
    normalizer_path = (repo_root / config["data"]["normalizer"]).resolve()
    targets_path = (repo_root / config["data"]["target_suite"]).resolve()
    for required_path in (metrics_path, schema_path, dataset_path, normalizer_path, targets_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"WP4 input does not exist: {required_path}")
    output_dir = (
        output_dir_override.resolve()
        if output_dir_override is not None
        else (repo_root / config["output"]["directory"]).resolve()
    )
    if output_dir.exists() and bool(config["output"].get("refuse_overwrite", True)):
        raise FileExistsError(f"WP4 output exists and will not be overwritten: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = dt.datetime.now(dt.timezone.utc)

    shutil.copy2(config_path, output_dir / "build_config.yaml")
    shutil.copy2(metrics_path, output_dir / "metrics_config.yaml")
    schema = load_schema(schema_path)
    normalizer = GeneratorNormalizer.load(normalizer_path)
    derived = derive_round_trip_metrics(dataset_path, normalizer, metrics_config)
    write_json(output_dir / "derived_metrics.json", derived.to_dict())
    target_frame = _select_targets(
        pd.read_csv(targets_path), config["runtime"].get("maximum_targets_per_type")
    )
    target_frame.to_csv(output_dir / "evaluated_targets.csv", index=False)
    device = _resolve_device(str(config["runtime"].get("device", "auto")))
    evaluator = RoundTripEvaluator(schema, normalizer, metrics_config, derived)
    sample_count = int(config["runtime"]["sample_count"])
    base_seed = int(config["runtime"]["sample_seed"])

    model_records: List[Dict[str, Any]] = []
    target_rows: List[Dict[str, Any]] = []
    checked_frames: List[pd.DataFrame] = []
    forward_frames: List[pd.DataFrame] = []
    ranked_frames: List[pd.DataFrame] = []
    failure_frames: List[pd.DataFrame] = []

    for model_name, model_value in config["models"].items():
        (
            checkpoint_value,
            expected_generator_name,
            model_overrides,
            model_sampling,
        ) = _resolve_model_spec(model_name, model_value)
        checkpoint = (repo_root / checkpoint_value).resolve()
        model_record: Dict[str, Any] = {
            "model": model_name,
            "expected_generator_name": expected_generator_name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
            "load_status": "failed",
            "error": "",
        }
        try:
            generator = load_generator(checkpoint, device=device)
            if generator.generator_name != expected_generator_name:
                raise ValueError(
                    f"Configured model {model_name} expects {expected_generator_name} "
                    f"but loaded {generator.generator_name}"
                )
            if generator.normalizer is None:
                raise ValueError("WP3 checkpoint does not contain a normalizer")
            if generator.normalizer.to_dict() != normalizer.to_dict():
                raise ValueError("Checkpoint normalizer differs from frozen WP4 normalizer")
            effective_runtime = dict(config["runtime"])
            effective_overrides = dict(
                effective_runtime.get("generator_overrides", {})
            )
            effective_overrides.update(model_overrides)
            effective_runtime["generator_overrides"] = effective_overrides
            if model_sampling:
                effective_runtime["sampling"] = model_sampling
            effective_sampling = dict(effective_runtime.get("sampling", {}))
            if str(effective_sampling.get("policy", "direct")) == "diag_rerank_fixed_raw":
                reranker_value = effective_sampling.get("reranker_checkpoint")
                if not reranker_value:
                    raise ValueError(
                        "diag_rerank_fixed_raw requires reranker_checkpoint"
                    )
                reranker_checkpoint = (repo_root / str(reranker_value)).resolve()
                reranker = load_generator(reranker_checkpoint, device=device)
                if reranker.generator_name != "diag_cfm":
                    raise ValueError("diag_rerank reranker must be a Diag-CFM checkpoint")
                reranker.model_config["ode_method"] = str(
                    effective_sampling.get("reranker_ode_method", "rk4")
                )
                reranker.model_config["ode_steps"] = int(
                    effective_sampling.get("reranker_ode_steps", 32)
                )
                effective_runtime["_diag_reranker"] = reranker
                model_record.update(
                    {
                        "reranker_checkpoint": str(reranker_checkpoint),
                        "reranker_checkpoint_sha256": sha256_file(
                            reranker_checkpoint
                        ),
                    }
                )
            model_record.update(_apply_generator_overrides(generator, effective_runtime))
            model_record["load_status"] = "passed"
        except Exception as exc:
            model_record["error"] = f"{type(exc).__name__}: {exc}"
            model_records.append(model_record)
            continue
        model_records.append(model_record)

        for target_value in target_frame.to_dict(orient="records"):
            target = _safe_target_payload(target_value)
            target_id = str(target["target_id"])
            seed_namespace = str(
                config["runtime"].get("sample_seed_namespace", model_name)
            )
            sample_seed = _target_seed(base_seed, seed_namespace, target_id)
            run_dir = output_dir / "artifacts" / "round_trip" / f"{model_name}__{target_id[:16]}"
            diagnostics: Dict[str, Any] = {
                "requested_count": sample_count,
                "returned_shape": None,
                "finite": None,
                "generator_diagnostics": {},
            }
            protocol_validation = None
            try:
                normalized_target = evaluator.normalize_target(target)
                normalized, generator_diagnostics = _sample_normalized_candidates(
                    generator,
                    normalized_target,
                    sample_count,
                    sample_seed,
                    effective_runtime,
                )
                physical = normalizer.inverse_design(normalized)
                diagnostics.update(
                    {
                        "returned_shape": list(normalized.shape),
                        "finite": bool(np.isfinite(normalized).all()),
                        "normalized_condition": normalized_target.tolist(),
                        **generator_diagnostics,
                    }
                )
                raw = _raw_candidate_frame(
                    normalized,
                    physical,
                    evaluator,
                    model_name,
                    target,
                    sample_seed,
                    generator_diagnostics.get("generator_diagnostics", {}),
                )
                if (
                    str(effective_runtime.get("sampling", {}).get("policy", "direct"))
                    in {"v2_1_fixed_raw", "diag_rerank_fixed_raw"}
                ):
                    candidate_ids = np.arange(len(normalized), dtype=int)
                    protocol_validation = evaluator.validate_candidates(
                        normalized, physical, candidate_ids=candidate_ids
                    )
                    protocol = apply_candidate_protocol_v21(
                        protocol_validation["candidate_id"].to_numpy(dtype=int),
                        protocol_validation["candidate_valid"].to_numpy(dtype=bool),
                        expected_raw_count=V21_RAW_CANDIDATE_COUNT,
                    )
                    protocol_validation["kept_by_v2_1"] = protocol_validation[
                        "candidate_id"
                    ].isin(protocol.kept_candidate_ids)
                    kept_positions = np.asarray(protocol.kept_positions, dtype=int)
                    normalized_for_0d = normalized[kept_positions]
                    physical_for_0d = physical[kept_positions]
                    kept_ids = np.asarray(protocol.kept_candidate_ids, dtype=int)
                    generator_diagnostics.update(
                        {
                            "N_raw": protocol.n_raw,
                            "N_valid": protocol.n_valid,
                            "N_keep": protocol.n_keep,
                            "N_trimmed": protocol.n_trimmed,
                            "valid_rate": protocol.valid_rate,
                            "kept_candidate_ids": list(protocol.kept_candidate_ids),
                            "generator_time": generator_diagnostics.get(
                                "generator_diagnostics", {}
                            ).get(
                                "generation_wall_seconds",
                                generator_diagnostics.get(
                                    "sampling_elapsed_seconds", np.nan
                                ),
                            ),
                        }
                    )
                else:
                    normalized_for_0d = normalized
                    physical_for_0d = physical
                    kept_ids = np.arange(len(normalized), dtype=int)
                if len(normalized_for_0d):
                    round_trip_started = time.perf_counter()
                    result = evaluator.evaluate(
                        target,
                        normalized_for_0d,
                        physical_for_0d,
                        candidate_ids=kept_ids,
                    )
                    generator_diagnostics["0d_time"] = (
                        time.perf_counter() - round_trip_started
                    )
                    generator_diagnostics["actual_0d_evaluations"] = int(
                        result.metrics["candidate_valid_count"]
                    )
                else:
                    result = sampling_budget_exhausted_result(
                        target,
                        model_name,
                        is_training_domain=evaluator.target_in_training_domain(target),
                    )
                    generator_diagnostics["0d_time"] = 0.0
                    generator_diagnostics["actual_0d_evaluations"] = 0
                diagnostics.update(generator_diagnostics)
            except ValueError as exc:
                if str(exc) != TARGET_OUT_OF_DOMAIN:
                    raise
                raw = pd.DataFrame()
                result = target_out_of_domain_result(target, model_name)
            _write_run_artifacts(
                run_dir,
                target,
                raw,
                result,
                model_name,
                sample_seed,
                diagnostics,
                protocol_validation=protocol_validation,
            )
            target_rows.append(
                _target_summary_row(
                    model_name, target, sample_seed, result, diagnostics
                )
            )
            if not result.checked_candidates.empty:
                checked_frames.append(
                    _add_context(result.checked_candidates, model_name, target, sample_seed)
                )
            if not result.forward_results.empty:
                forward_frames.append(
                    _add_context(result.forward_results, model_name, target, sample_seed)
                )
            if not result.ranked_candidates.empty:
                ranked_frames.append(
                    _add_context(result.ranked_candidates, model_name, target, sample_seed)
                )
            contextual_failures = result.failures.copy()
            contextual_failures.insert(0, "target_type", target["target_type"])
            contextual_failures.insert(0, "target_id", target_id)
            contextual_failures.insert(0, "model", model_name)
            failure_frames.append(contextual_failures)

    target_summary = pd.DataFrame(target_rows)
    failures = pd.concat(failure_frames, ignore_index=True) if failure_frames else pd.DataFrame(
        columns=["model", "target_id", "target_type", "candidate_id", "failure_code", "solver_warnings"]
    )
    checked = pd.concat(checked_frames, ignore_index=True) if checked_frames else pd.DataFrame()
    forward = pd.concat(forward_frames, ignore_index=True) if forward_frames else pd.DataFrame()
    ranked = pd.concat(ranked_frames, ignore_index=True) if ranked_frames else pd.DataFrame()
    model_summary = _aggregate_model_summary(target_summary) if not target_summary.empty else pd.DataFrame()
    target_summary.to_csv(output_dir / "target_summary.csv", index=False)
    model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    checked.to_csv(output_dir / "checked_candidates.csv", index=False)
    forward.to_csv(output_dir / "evaluation_ledger.csv", index=False)
    ranked.to_csv(output_dir / "ranked_candidates.csv", index=False)
    failures.to_csv(output_dir / "failures.csv", index=False)
    pd.DataFrame(model_records).to_csv(output_dir / "model_loads.csv", index=False)

    gate = _functional_gate(list(config["models"]), model_records, target_summary, failures)
    write_json(output_dir / "functional_gate.json", gate)
    _write_final_report(output_dir / "report.md", gate, derived, model_summary)
    environment = collect_environment()
    environment["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    write_json(output_dir / "environment.json", environment)
    write_json(output_dir / "git_state.json", collect_git_state(repo_root, output_dir))
    write_json(output_dir / "source_manifest.json", collect_source_manifest(repo_root))
    finished = dt.datetime.now(dt.timezone.utc)
    manifest = {
        "wp": "WP4",
        "version": config["version"],
        "status": "completed" if gate["passed"] else "functional_gate_failed",
        "functional_gate_passed": gate["passed"],
        "performance_superiority_required": False,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "target_suite": {
            "path": str(targets_path),
            "sha256": sha256_file(targets_path),
            "evaluated_count": len(target_frame),
            "is_complete_suite": config["runtime"].get("maximum_targets_per_type") is None,
        },
        "models": model_records,
        "artifacts": {
            "evaluation_ledger": "evaluation_ledger.csv",
            "target_summary": "target_summary.csv",
            "model_summary": "model_summary.csv",
            "functional_gate": "functional_gate.json",
            "report": "report.md",
        },
    }
    write_yaml(output_dir / "wp4_manifest.yaml", manifest)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run WP4 physics-in-the-loop round-trip validation"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config/generator_wp4.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = run_wp4(args.config, output_dir_override=args.output_dir)
    manifest = yaml.safe_load((output / "wp4_manifest.yaml").read_text(encoding="utf-8"))
    print(json.dumps({"output": str(output), "functional_gate_passed": manifest["functional_gate_passed"]}))
    return 0 if manifest["functional_gate_passed"] else 1
