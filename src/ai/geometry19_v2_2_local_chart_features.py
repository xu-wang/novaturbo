"""P2 train-fold-only local Jacobians and bootstrap stability for Geometry19."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge
import yaml

from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_local_chart_data_audit import (
    P1_PASS_STATUS,
    _effective_covariance_rank,
)
from src.ai.geometry19_v2_2_local_chart_protocol import (
    Geometry19LocalChartProtocolError,
    PROTOCOL_VERSION,
    validate_protocol_schema,
)


P2_PASS_STATUS = "LOCAL_CHART_MOE_P2_JACOBIAN_STABILITY_PASS"
P2_PAUSE_STATUS = "LOCAL_CHART_MOE_P2_PAUSE_INSUFFICIENT_RELIABLE_ROOTS"
PERFORMANCE_NAMES = ("thrust_N", "tsfc_kg_N_s", "mass_kg")
TSFC_INDEX = 1


class Geometry19LocalChartFeatureError(ValueError):
    """Raised when P2 inputs or train-fold isolation violate the protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Geometry19LocalChartFeatureError(message)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _derived_seed(base_seed: int, fold: int, root_id: str) -> int:
    value = f"{base_seed}|{fold}|{root_id}".encode("utf-8")
    return int.from_bytes(sha256(value).digest()[:8], "big")


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1.0e-14:
        return 0.0
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def _fit_jacobian(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    model = Ridge(alpha=float(alpha), fit_intercept=True, solver="svd")
    model.fit(x, y)
    return np.asarray(model.coef_, dtype=np.float64), np.asarray(model.intercept_, dtype=np.float64)


def _bootstrap_metrics(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    rng = np.random.default_rng(seed)
    jacobians: list[np.ndarray] = []
    for _ in range(replicates):
        indices = rng.integers(0, len(x), size=len(x))
        jacobian, _ = _fit_jacobian(x[indices], y[indices], alpha)
        jacobians.append(jacobian)
    stacked = np.stack(jacobians)
    mean_jacobian = np.mean(stacked, axis=0)
    deviations = np.linalg.norm(stacked - mean_jacobian[None, :, :], axis=(1, 2))
    mean_norm = float(np.linalg.norm(mean_jacobian))
    frobenius_cv = float(np.sqrt(np.mean(deviations**2)) / max(mean_norm, 1.0e-14))
    tsfc_reference = mean_jacobian[TSFC_INDEX]
    cosines = np.asarray(
        [_cosine(jacobian[TSFC_INDEX], tsfc_reference) for jacobian in stacked],
        dtype=np.float64,
    )
    metrics = {
        "alpha": float(alpha),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": int(seed),
        "bootstrap_mean_jacobian_frobenius_norm": mean_norm,
        "frobenius_coefficient_of_variation": frobenius_cv,
        "tsfc_gradient_cosine_q05": float(np.quantile(cosines, 0.05)),
        "tsfc_gradient_cosine_median": float(np.quantile(cosines, 0.50)),
        "tsfc_gradient_cosine_q95": float(np.quantile(cosines, 0.95)),
        "tsfc_zero_gradient_bootstrap_fraction": float(
            np.mean(np.linalg.norm(stacked[:, TSFC_INDEX, :], axis=1) <= 1.0e-14)
        ),
    }
    return metrics, stacked


def _estimate_root(
    root_id: str,
    records: Sequence[Mapping[str, Any]],
    *,
    fold: int,
    variable_names: Sequence[str],
    lower: np.ndarray,
    scale: np.ndarray,
    tolerances: np.ndarray,
    alpha_candidates: Sequence[float],
    replicates: int,
    base_seed: int,
    reliability_gate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray | float | bool | int]]:
    designs = np.asarray(
        [[float(row["geometry19"][name]) for name in variable_names] for row in records],
        dtype=np.float64,
    )
    performance = np.asarray(
        [[float(row["performance"][name]) for name in PERFORMANCE_NAMES] for row in records],
        dtype=np.float64,
    )
    normalized = (designs - lower[None, :]) / scale[None, :]
    scaled_log_performance = np.log(performance) / tolerances[None, :]
    design_center = np.mean(normalized, axis=0)
    performance_center = np.mean(scaled_log_performance, axis=0)
    delta_x = normalized - design_center[None, :]
    delta_y = scaled_log_performance - performance_center[None, :]
    rank, rank_tolerance, covariance_eigenvalues = _effective_covariance_rank(normalized)
    singular_values = np.linalg.svd(delta_x, compute_uv=False)
    effective_condition = (
        float(singular_values[0] / singular_values[rank - 1]) if rank > 0 else None
    )
    full_condition = effective_condition if rank == len(variable_names) else None

    alpha_metrics: list[dict[str, Any]] = []
    bootstrap_by_alpha: dict[float, np.ndarray] = {}
    for alpha_value in alpha_candidates:
        alpha = float(alpha_value)
        metrics, bootstraps = _bootstrap_metrics(
            delta_x,
            delta_y,
            alpha=alpha,
            replicates=replicates,
            seed=_derived_seed(base_seed, fold, root_id),
        )
        alpha_metrics.append(metrics)
        bootstrap_by_alpha[alpha] = bootstraps

    selected = max(
        alpha_metrics,
        key=lambda value: (
            float(value["tsfc_gradient_cosine_median"]),
            -float(value["frobenius_coefficient_of_variation"]),
            -float(value["alpha"]),
        ),
    )
    selected_alpha = float(selected["alpha"])
    jacobian, intercept = _fit_jacobian(delta_x, delta_y, selected_alpha)
    selected_bootstraps = bootstrap_by_alpha[selected_alpha]
    mean_bootstrap_jacobian = np.mean(selected_bootstraps, axis=0)

    failures: list[str] = []
    if len(records) < int(reliability_gate["minimum_rows"]):
        failures.append("rows_below_minimum")
    if rank < int(reliability_gate["minimum_effective_rank"]):
        failures.append("effective_rank_below_minimum")
    if float(selected["tsfc_gradient_cosine_median"]) < float(
        reliability_gate["minimum_tsfc_gradient_cosine_median"]
    ):
        failures.append("tsfc_gradient_cosine_below_minimum")
    if float(selected["frobenius_coefficient_of_variation"]) > float(
        reliability_gate["maximum_frobenius_coefficient_of_variation"]
    ):
        failures.append("frobenius_cv_above_maximum")
    reliable = not failures
    segments = sorted({str(row["assigned_segment"]) for row in records})
    audit = {
        "root_seed_id": root_id,
        "outer_fold": fold,
        "role": "outer_training_root",
        "rows": len(records),
        "segments": segments,
        "effective_rank": rank,
        "rank_tolerance": rank_tolerance,
        "effective_subspace_condition_number": effective_condition,
        "full_rank_condition_number": full_condition,
        "normalized_covariance_eigenvalues": covariance_eigenvalues,
        "alpha_selection_rule": "maximum_tsfc_cosine_median_then_minimum_frobenius_cv_then_smallest_alpha",
        "alpha_metrics": alpha_metrics,
        "selected_alpha": selected_alpha,
        "selected_tsfc_gradient_cosine_q05": selected["tsfc_gradient_cosine_q05"],
        "selected_tsfc_gradient_cosine_median": selected["tsfc_gradient_cosine_median"],
        "selected_tsfc_gradient_cosine_q95": selected["tsfc_gradient_cosine_q95"],
        "selected_frobenius_coefficient_of_variation": selected[
            "frobenius_coefficient_of_variation"
        ],
        "full_fit_jacobian_frobenius_norm": float(np.linalg.norm(jacobian)),
        "bootstrap_mean_vs_full_fit_frobenius_distance": float(
            np.linalg.norm(mean_bootstrap_jacobian - jacobian)
        ),
        "jacobian_reliable": reliable,
        "reliability_failures": failures,
        "unreliable_action": None
        if reliable
        else "retain_design_center_and_zero_jacobian_distance_weight",
    }
    arrays: dict[str, np.ndarray | float | bool | int] = {
        "design_center": design_center,
        "scaled_log_performance_center": performance_center,
        "jacobian": jacobian,
        "intercept": intercept,
        "selected_alpha": selected_alpha,
        "effective_rank": rank,
        "condition_number": math.inf if full_condition is None else full_condition,
        "frobenius_cv": float(selected["frobenius_coefficient_of_variation"]),
        "tsfc_cosine_median": float(selected["tsfc_gradient_cosine_median"]),
        "jacobian_reliable": reliable,
    }
    return audit, arrays


def _fold_summary(
    fold: int,
    audits: Sequence[Mapping[str, Any]],
    target_segments: Sequence[str],
    minimum_fraction: float,
) -> dict[str, Any]:
    reliable = [row for row in audits if bool(row["jacobian_reliable"])]
    reason_counts: Counter[str] = Counter(
        reason for row in audits for reason in row["reliability_failures"]
    )
    target: dict[str, Any] = {}
    for segment in target_segments:
        members = [row for row in audits if segment in row["segments"]]
        target[segment] = {
            "training_roots": len(members),
            "reliable_roots": sum(bool(row["jacobian_reliable"]) for row in members),
            "reliable_fraction": float(
                sum(bool(row["jacobian_reliable"]) for row in members) / len(members)
            ),
        }
    fraction = float(len(reliable) / len(audits))
    return {
        "fold": fold,
        "training_roots": len(audits),
        "reliable_roots": len(reliable),
        "unreliable_roots": len(audits) - len(reliable),
        "reliable_fraction": fraction,
        "minimum_reliable_fraction": minimum_fraction,
        "reliable_fraction_gate_pass": fraction >= minimum_fraction,
        "reliability_failure_counts": dict(sorted(reason_counts.items())),
        "selected_alpha_counts": dict(
            sorted(Counter(str(row["selected_alpha"]) for row in audits).items())
        ),
        "target_segments": target,
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Geometry19 V2.2 Local-Chart MoE P2 Jacobian Stability",
        "",
        f"- Status: `{summary['status']}`",
        f"- Root-fold estimates: {summary['root_fold_estimates']}",
        f"- Reliable root-fold estimates: {summary['reliable_root_fold_estimates']}",
        f"- Minimum fold reliable fraction: {summary['minimum_fold_reliable_fraction']:.3f}",
        f"- P3 authorized: `{str(summary['p3_authorized']).lower()}`",
        "- Confirmation content parsed: `false`",
        "- Formal/deployment use allowed: `false` / `false`",
        "",
        "## Outer-train-fold reliability",
        "",
        "| Fold | Train roots | Reliable | Fraction | low→mid | mid→high | Gate | α counts |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for fold in summary["folds"]:
        alpha_counts = ", ".join(
            f"{alpha}:{count}" for alpha, count in fold["selected_alpha_counts"].items()
        )
        lines.append(
            f"| {fold['fold']} | {fold['training_roots']} | {fold['reliable_roots']} | "
            f"{fold['reliable_fraction']:.3f} | "
            f"{fold['target_segments']['low_mass_low_to_mid']['reliable_fraction']:.3f} | "
            f"{fold['target_segments']['low_mass_mid_to_high']['reliable_fraction']:.3f} | "
            f"`{str(fold['reliable_fraction_gate_pass']).lower()}` | {alpha_counts} |"
        )
    lines.extend(
        [
            "",
            "Each estimate uses only the current outer training roots. Validation roots do not contribute to centers, Ridge fits, alpha selection, bootstrap stability, or reliability decisions. Unreliable roots remain represented by their design centers and receive zero Jacobian-distance weight in P3.",
            "",
        ]
    )
    return "\n".join(lines)


def run_local_jacobian_stability(
    repo_root: Path,
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Execute the frozen P2 local-Jacobian bootstrap experiment."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        validate_protocol_schema(config)
    except Geometry19LocalChartProtocolError as error:
        raise Geometry19LocalChartFeatureError(str(error)) from error

    resolved: dict[str, Path] = {}
    for role, item in config["inputs"].items():
        path = (repo_root / str(item["path"])).resolve()
        _require(path.is_file(), f"Frozen input is missing: {role}")
        _require(sha256_file(path) == str(item["sha256"]), f"Frozen input hash mismatch: {role}")
        resolved[role] = path

    runtime_root = (
        output_root.resolve()
        if output_root is not None
        else (repo_root / str(config["artifacts"]["runtime_root"])).resolve()
    )
    p1_path = runtime_root / "audits/input_audit.json"
    _require(p1_path.is_file(), "P1 input audit evidence is missing")
    p1 = json.loads(p1_path.read_text(encoding="utf-8"))
    _require(p1.get("status") == P1_PASS_STATUS, "P1 data/root/fold audit did not pass")
    _require(p1.get("protocol_sha256") == sha256_file(config_path), "P1 protocol hash no longer matches")
    _require(p1.get("confirmation_content_read") is False, "P1 confirmation isolation was violated")

    records = _read_jsonl(resolved["development_dataset"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["root_seed_id"])].append(row)
    folds = json.loads(resolved["development_folds"].read_text(encoding="utf-8"))
    assignments = {str(key): int(value) for key, value in folds["root_assignments"].items()}
    _require(set(grouped) == set(assignments), "P2 data roots differ from frozen fold roots")

    schema = yaml.safe_load(resolved["design_schema"].read_text(encoding="utf-8"))
    variables = schema["design_variables"]
    variable_names = [str(value["name"]) for value in variables]
    _require(len(variable_names) == 19, "P2 design schema is not 19D")
    lower = np.asarray([float(value["bounds"][0]) for value in variables], dtype=np.float64)
    scale = np.asarray(
        [float(value["bounds"][1]) - float(value["bounds"][0]) for value in variables],
        dtype=np.float64,
    )
    domain = yaml.safe_load(resolved["target_domain"].read_text(encoding="utf-8"))
    tolerances = np.asarray(
        [float(domain["axes"][name]["relative_tolerance"]) for name in PERFORMANCE_NAMES],
        dtype=np.float64,
    )
    jacobian_config = config["local_jacobian"]
    alpha_candidates = [float(value) for value in jacobian_config["ridge_alpha_candidates"]]
    replicates = int(jacobian_config["bootstrap_replicates"])
    base_seed = int(jacobian_config["bootstrap_seed"])
    reliability_gate = jacobian_config["reliability_gate"]
    minimum_fraction = float(jacobian_config["minimum_reliable_training_root_fraction"])

    fold_summaries: list[dict[str, Any]] = []
    artifact_hashes: dict[str, Any] = {}
    reliability_by_root: dict[str, list[bool]] = defaultdict(list)
    fold_count = int(config["outer_evaluation"]["fold_count"])
    for fold in range(fold_count):
        validation_roots = {root_id for root_id, value in assignments.items() if value == fold}
        train_roots = sorted(set(grouped) - validation_roots)
        audits: list[dict[str, Any]] = []
        arrays_by_root: list[dict[str, np.ndarray | float | bool | int]] = []
        for root_id in train_roots:
            audit, arrays = _estimate_root(
                root_id,
                grouped[root_id],
                fold=fold,
                variable_names=variable_names,
                lower=lower,
                scale=scale,
                tolerances=tolerances,
                alpha_candidates=alpha_candidates,
                replicates=replicates,
                base_seed=base_seed,
                reliability_gate=reliability_gate,
            )
            audits.append(audit)
            arrays_by_root.append(arrays)
            reliability_by_root[root_id].append(bool(audit["jacobian_reliable"]))

        fold_root = runtime_root / f"jacobians/fold-{fold}"
        audit_path = fold_root / "jacobian_audit.jsonl"
        npz_path = fold_root / "root_jacobians.npz"
        _write_jsonl(audit_path, audits)
        _write_npz(
            npz_path,
            root_ids=np.asarray(train_roots),
            validation_root_ids=np.asarray(sorted(validation_roots)),
            variable_names=np.asarray(variable_names),
            performance_names=np.asarray(PERFORMANCE_NAMES),
            design_centers=np.stack([value["design_center"] for value in arrays_by_root]),
            scaled_log_performance_centers=np.stack(
                [value["scaled_log_performance_center"] for value in arrays_by_root]
            ),
            jacobians=np.stack([value["jacobian"] for value in arrays_by_root]),
            intercepts=np.stack([value["intercept"] for value in arrays_by_root]),
            selected_alphas=np.asarray([value["selected_alpha"] for value in arrays_by_root]),
            effective_ranks=np.asarray([value["effective_rank"] for value in arrays_by_root]),
            condition_numbers=np.asarray([value["condition_number"] for value in arrays_by_root]),
            frobenius_cvs=np.asarray([value["frobenius_cv"] for value in arrays_by_root]),
            tsfc_cosine_medians=np.asarray(
                [value["tsfc_cosine_median"] for value in arrays_by_root]
            ),
            jacobian_reliable=np.asarray(
                [value["jacobian_reliable"] for value in arrays_by_root], dtype=np.bool_
            ),
        )
        fold_summary = _fold_summary(
            fold, audits, config["target_segments"], minimum_fraction
        )
        fold_summary["validation_roots_excluded"] = len(validation_roots)
        fold_summary["artifacts"] = {
            "root_jacobians": str(npz_path.relative_to(repo_root))
            if npz_path.is_relative_to(repo_root)
            else str(npz_path),
            "root_jacobians_sha256": sha256_file(npz_path),
            "jacobian_audit": str(audit_path.relative_to(repo_root))
            if audit_path.is_relative_to(repo_root)
            else str(audit_path),
            "jacobian_audit_sha256": sha256_file(audit_path),
        }
        fold_summaries.append(fold_summary)
        artifact_hashes[str(fold)] = fold_summary["artifacts"]

    root_reliability = {
        root_id: {
            "training_fold_estimates": len(values),
            "reliable_estimates": sum(values),
            "reliable_fraction": float(np.mean(values)),
        }
        for root_id, values in sorted(reliability_by_root.items())
    }
    p3_authorized = all(
        bool(fold["reliable_fraction_gate_pass"]) for fold in fold_summaries
    )
    summary_path = runtime_root / "audits/jacobian_stability_summary.json"
    report_path = runtime_root / "audits/P2_Jacobian_Stability_Report.md"
    implementation_path = Path(__file__).resolve()
    summary: dict[str, Any] = {
        "status": P2_PASS_STATUS if p3_authorized else P2_PAUSE_STATUS,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(config_path),
        "p1_input_audit_sha256": sha256_file(p1_path),
        "implementation_path": str(implementation_path.relative_to(repo_root)),
        "implementation_sha256": sha256_file(implementation_path),
        "coordinate_contract": {
            "design": "schema_range_normalized_then_root_centered",
            "performance": "natural_log_divided_by_target_relative_tolerance_then_root_centered",
            "jacobian_shape": [3, 19],
        },
        "alpha_selection_rule": "maximum_tsfc_cosine_median_then_minimum_frobenius_cv_then_smallest_alpha",
        "alpha_candidates": alpha_candidates,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": base_seed,
        "reliability_gate": dict(reliability_gate),
        "minimum_reliable_training_root_fraction": minimum_fraction,
        "root_fold_estimates": sum(fold["training_roots"] for fold in fold_summaries),
        "reliable_root_fold_estimates": sum(fold["reliable_roots"] for fold in fold_summaries),
        "minimum_fold_reliable_fraction": min(
            fold["reliable_fraction"] for fold in fold_summaries
        ),
        "minimum_target_segment_reliable_fraction": min(
            item["reliable_fraction"]
            for fold in fold_summaries
            for item in fold["target_segments"].values()
        ),
        "folds": fold_summaries,
        "root_reliability": root_reliability,
        "roots_reliable_in_all_training_folds": sum(
            value["reliable_fraction"] == 1.0 for value in root_reliability.values()
        ),
        "roots_unreliable_in_all_training_folds": sum(
            value["reliable_fraction"] == 0.0 for value in root_reliability.values()
        ),
        "p3_authorized": p3_authorized,
        "confirmation_hash_checked": True,
        "confirmation_content_read": False,
        "confirmation_reuse_allowed": False,
        "local_jacobian_fitting_executed": True,
        "surrogate_expert_training_executed": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "artifact_hashes": artifact_hashes,
        "next_authorized_stage": (
            "P3_training_fold_root_descriptors_and_chart_clustering"
            if p3_authorized
            else "controlled_signed_perturbations_for_unreliable_roots"
        ),
    }
    _write_text(report_path, _report(summary))
    summary["report_path"] = (
        str(report_path.relative_to(repo_root))
        if report_path.is_relative_to(repo_root)
        else str(report_path)
    )
    summary["report_sha256"] = sha256_file(report_path)
    _write_json(summary_path, summary)
    return summary
