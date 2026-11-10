"""Run the locked one-shot five-arm scoring on S3 v4.1 development rows."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_stable_root_s2_training import (
    OUTPUT_NAMES,
    _linear_predict,
    _local_predict,
    _schema_arrays,
)
from src.ai.geometry19_v2_2_stable_root_s3_v4_1_scoring_protocol import (
    LOCK_PASS,
    PROTOCOL_VERSION,
)


PASS_STATUS = "STABLE_ROOT_S3_V4_1_FORWARD_SURROGATE_PASS"
NO_ADVANCEMENT_STATUS = "STABLE_ROOT_S3_V4_1_FORWARD_NO_ADVANCEMENT"
TARGET_SEGMENTS = ("low_mass_low_to_mid", "low_mass_mid_to_high")


class Geometry19StableRootS3V41ScoringError(ValueError):
    """Raised when the frozen one-shot scoring contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Geometry19StableRootS3V41ScoringError(message)


def _canonical(row: Mapping[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_text(path, "".join(_canonical(row) + "\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_model(path: Path, arm_id: str) -> tuple[list[str], dict[str, np.ndarray]]:
    with np.load(path) as bundle:
        _require(str(bundle["arm_id"].item()) == arm_id, f"Model arm identity mismatch: {arm_id}")
        stable_ids = [str(value) for value in bundle["stable_root_ids"].tolist()]
        keys = {
            "all_78_global_root_balanced_ridge": ("coef", "intercept"),
            "stable_38_pooled_root_balanced_ridge": ("coef", "intercept"),
            "stable_38_nearest_root_local_ridge": ("coefs", "intercepts"),
            "stable_38_oracle_root_local_ridge": ("coefs", "intercepts"),
            "stable_38_global_plus_root_residual_ridge": (
                "base_coef",
                "base_intercept",
                "residual_coefs",
                "residual_intercepts",
            ),
        }[arm_id]
        arrays = {key: np.asarray(bundle[key], dtype=np.float64) for key in keys}
    _require(len(stable_ids) == 38 and len(set(stable_ids)) == 38, f"Stable roots changed: {arm_id}")
    _require(all(np.all(np.isfinite(value)) for value in arrays.values()), f"Non-finite model arrays: {arm_id}")
    return stable_ids, arrays


def _metrics(
    rows: Sequence[Mapping[str, Any]],
    truth: np.ndarray,
    prediction: np.ndarray,
    tolerances: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    rho = np.max(np.abs(prediction / truth - 1.0) / tolerances[None, :], axis=1)
    by_root: dict[str, list[float]] = defaultdict(list)
    by_segment_root: dict[tuple[str, str], list[float]] = defaultdict(list)
    for value, row in zip(rho.tolist(), rows):
        root = str(row["root_seed_id"])
        by_root[root].append(float(value))
        for segment in row["source_root_segments"]:
            if segment in TARGET_SEGMENTS:
                by_segment_root[(str(segment), root)].append(float(value))
    root_p95 = {root: float(np.quantile(values, 0.95)) for root, values in sorted(by_root.items())}
    by_segment: dict[str, Any] = {}
    for segment in TARGET_SEGMENTS:
        values = {
            root: float(np.quantile(items, 0.95))
            for (item_segment, root), items in by_segment_root.items()
            if item_segment == segment
        }
        _require(bool(values), f"No scoring rows for segment: {segment}")
        by_segment[segment] = {
            "roots": len(values),
            "worst_root_rho_p95": float(max(values.values())),
            "root_rho_p95_p95": float(np.quantile(list(values.values()), 0.95)),
            "root_rho_p95": dict(sorted(values.items())),
        }
    return (
        {
            "rows": len(rows),
            "roots": len(root_p95),
            "row_rho_p50": float(np.quantile(rho, 0.50)),
            "row_rho_p95": float(np.quantile(rho, 0.95)),
            "worst_root_rho_p95": float(max(root_p95.values())),
            "root_rho_p95_p95": float(np.quantile(list(root_p95.values()), 0.95)),
            "root_rho_p95": root_p95,
            "by_target_segment": by_segment,
            "all_predictions_finite_positive": bool(np.all(np.isfinite(prediction)) and np.all(prediction > 0.0)),
        },
        rho,
    )


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Stable-Root S3 v4.1 Five-Arm Development Model Scoring",
        "",
        f"- Status: `{summary['status']}`",
        f"- Common rows: {summary['scoring_rows']}",
        f"- Roots: {summary['scoring_roots']}",
        f"- Advancing arms: {', '.join(summary['advancing_arms']) or 'none'}",
        f"- Nearest-route accuracy: {summary['nearest_route_accuracy']:.6f}",
        f"- Nearest/oracle worst-root gap: {summary['nearest_vs_oracle_relative_worst_root_gap']:.6f}",
        "- Confirmation content parsed: `false`",
        "- Inverse design authorized: `false`",
        "",
        "| Arm | Worst root rho P95 | Low-to-mid worst | Mid-to-high worst | Gate pass |",
        "|---|---:|---:|---:|---|",
    ]
    for arm in summary["model_arms"]:
        metrics = arm["metrics"]
        lines.append(
            f"| `{arm['arm_id']}` | {metrics['worst_root_rho_p95']:.4f} | "
            f"{metrics['by_target_segment']['low_mass_low_to_mid']['worst_root_rho_p95']:.4f} | "
            f"{metrics['by_target_segment']['low_mass_mid_to_high']['worst_root_rho_p95']:.4f} | "
            f"`{str(arm['advancement_gate_pass']).lower()}` |"
        )
    lines.extend(
        [
            "",
            "This is a one-shot development comparison on a post-selected 33-root software subdomain. It is not fresh confirmation, unknown-manifold evidence, or inverse-design authorization.",
            "",
        ]
    )
    return "\n".join(lines)


def run_s3_v4_1_model_scoring(
    repo_root: Path,
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Load each frozen model once, predict common rows, seal, and score."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(config.get("protocol_version") == PROTOCOL_VERSION, "Scoring protocol mismatch")
    runtime_root = (repo_root / str(config["artifacts"]["runtime_root"])).resolve()
    lock_path = runtime_root / str(config["artifacts"]["protocol_lock"])
    _require(lock_path.is_file(), "Scoring protocol lock missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    _require(lock.get("status") == LOCK_PASS, "Scoring protocol lock did not pass")
    _require(lock.get("protocol_sha256") == sha256_file(config_path), "Scoring protocol changed after lock")
    _require(lock.get("model_scoring_authorized") is True, "Model scoring was not authorized")

    protected = {"sealed_confirmation"}
    resolved: dict[str, Path] = {}
    input_audit: dict[str, Any] = {}
    for role, item in config["inputs"].items():
        path = (repo_root / str(item["path"])).resolve()
        _require(path.is_file(), f"Missing scoring input: {role}")
        actual = sha256_file(path)
        _require(actual == str(item["sha256"]), f"Scoring input hash mismatch: {role}")
        resolved[role] = path
        input_audit[role] = {
            "path": str(item["path"]),
            "expected_sha256": str(item["sha256"]),
            "actual_sha256": actual,
            "match": True,
            "content_parsed_during_scoring": role not in protected,
        }

    ledger = _read_jsonl(resolved["sealed_physics_ledger"])
    rows = [row for row in ledger if row.get("evaluation_eligible") is True]
    registry = _read_jsonl(runtime_root / str(config["artifacts"]["eligible_row_registry"]))
    _require(len(rows) == len(registry) == int(config["evaluation_population"]["exact_rows"]), "Scoring population changed")
    _require([row["design_hash"] for row in rows] == [row["design_hash"] for row in registry], "Scoring row order changed")

    schema = yaml.safe_load(resolved["design_schema"].read_text(encoding="utf-8"))
    names, lower, upper = _schema_arrays(schema)
    x = np.asarray([[float(row["geometry19"][name]) for name in names] for row in rows], dtype=np.float64)
    x = (x - lower[None, :]) / (upper - lower)[None, :]
    truth = np.asarray([[float(row["performance"][name]) for name in OUTPUT_NAMES] for row in rows], dtype=np.float64)
    tolerances = np.asarray([float(config["scoring"]["target_relative_tolerances"][name]) for name in OUTPUT_NAMES])

    stable_registry = _read_jsonl(resolved["stable_root_registry"])
    center_by_id = {str(row["root_seed_id"]): np.asarray(row["normalized_design_center"], dtype=np.float64) for row in stable_registry}
    selected_ids = sorted({str(row["root_seed_id"]) for row in rows})
    _require(len(selected_ids) == 33 and all(root in center_by_id for root in selected_ids), "Selected router roots changed")

    manifest = json.loads(resolved["s2_model_manifest"].read_text(encoding="utf-8"))
    manifest_models = {str(item["arm_id"]): item for item in manifest["models"]}
    prediction_by_arm: dict[str, np.ndarray] = {}
    route_by_arm: dict[str, list[str | None]] = {}
    for frozen in config["models"]["exact_arms"]:
        arm = str(frozen["id"])
        item = manifest_models[arm]
        model_path = (repo_root / str(item["model_path"])).resolve()
        _require(sha256_file(model_path) == str(frozen["sha256"]), f"Model hash changed before scoring: {arm}")
        stable_ids, model = _load_model(model_path, arm)
        stable_index = {root: index for index, root in enumerate(stable_ids)}
        _require(all(root in stable_index for root in selected_ids), f"Selected root missing from model: {arm}")
        if arm in ("all_78_global_root_balanced_ridge", "stable_38_pooled_root_balanced_ridge"):
            prediction_log = _linear_predict(model, x)
            routes: list[str | None] = [None] * len(rows)
        elif arm == "stable_38_nearest_root_local_ridge":
            centers = np.asarray([center_by_id[root] for root in selected_ids])
            selected_route = np.argmin(np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2), axis=1)
            routes = [selected_ids[index] for index in selected_route.tolist()]
            route_indices = np.asarray([stable_index[root] for root in routes])
            prediction_log = _local_predict(model, x, route_indices)
        elif arm == "stable_38_oracle_root_local_ridge":
            routes = [str(row["root_seed_id"]) for row in rows]
            route_indices = np.asarray([stable_index[root] for root in routes])
            prediction_log = _local_predict(model, x, route_indices)
        else:
            routes = [str(row["root_seed_id"]) for row in rows]
            route_indices = np.asarray([stable_index[root] for root in routes])
            prediction_log = (
                x @ model["base_coef"].T
                + model["base_intercept"][None, :]
                + np.einsum("nij,nj->ni", model["residual_coefs"][route_indices], x)
                + model["residual_intercepts"][route_indices]
            )
        prediction = np.exp(prediction_log)
        _require(np.all(np.isfinite(prediction)) and np.all(prediction > 0.0), f"Invalid predictions: {arm}")
        prediction_by_arm[arm] = prediction
        route_by_arm[arm] = routes

    output = output_root.resolve() if output_root is not None else runtime_root
    ledger_path = output / str(config["artifacts"]["future_prediction_ledger"])
    summary_path = output / str(config["artifacts"]["future_scoring_summary"])
    report_path = output / str(config["artifacts"]["future_scoring_report"])
    _require(not any(path.exists() for path in (ledger_path, summary_path, report_path)), "Refusing to overwrite scoring artifacts")
    prediction_rows: list[dict[str, Any]] = []
    arm_ids = [str(item["id"]) for item in config["models"]["exact_arms"]]
    for index, row in enumerate(rows):
        prediction_rows.append(
            {
                "scoring_row_index": index,
                "design_hash": row["design_hash"],
                "root_seed_id": row["root_seed_id"],
                "source_root_segments": row["source_root_segments"],
                "radius_fraction": row["radius_fraction"],
                "combined_source_protocol": row["combined_source_protocol"],
                "truth": {name: float(truth[index, axis]) for axis, name in enumerate(OUTPUT_NAMES)},
                "arms": {
                    arm: {
                        "prediction": {name: float(prediction_by_arm[arm][index, axis]) for axis, name in enumerate(OUTPUT_NAMES)},
                        "routed_root_id": route_by_arm[arm][index],
                    }
                    for arm in arm_ids
                },
            }
        )
    _write_jsonl(ledger_path, prediction_rows)

    model_results: list[dict[str, Any]] = []
    metrics_by_arm: dict[str, Mapping[str, Any]] = {}
    rho_by_arm: dict[str, np.ndarray] = {}
    for arm in arm_ids:
        metrics, rho = _metrics(rows, truth, prediction_by_arm[arm], tolerances)
        metrics_by_arm[arm] = metrics
        rho_by_arm[arm] = rho
    comparator = str(config["accuracy_gates"]["comparator_arm"])
    base = metrics_by_arm[comparator]
    nearest = metrics_by_arm["stable_38_nearest_root_local_ridge"]
    oracle = metrics_by_arm["stable_38_oracle_root_local_ridge"]
    nearest_oracle_gap = nearest["worst_root_rho_p95"] / oracle["worst_root_rho_p95"] - 1.0
    advancing: list[str] = []
    eligible_arms = set(config["accuracy_gates"]["eligible_local_arms"])
    gates = config["accuracy_gates"]
    for arm in arm_ids:
        metrics = metrics_by_arm[arm]
        worst_reduction = 1.0 - metrics["worst_root_rho_p95"] / base["worst_root_rho_p95"]
        segment_reductions = {
            segment: 1.0
            - metrics["by_target_segment"][segment]["worst_root_rho_p95"]
            / base["by_target_segment"][segment]["worst_root_rho_p95"]
            for segment in TARGET_SEGMENTS
        }
        checks = {
            "eligible_local_arm": arm in eligible_arms,
            "all_predictions_finite_positive": bool(metrics["all_predictions_finite_positive"]),
            "worst_root_rho_p95_at_most_5": metrics["worst_root_rho_p95"] <= float(gates["maximum_worst_root_rho_p95"]),
            "each_segment_worst_root_rho_p95_at_most_5": all(metrics["by_target_segment"][segment]["worst_root_rho_p95"] <= float(gates["maximum_each_source_target_segment_worst_root_rho_p95"]) for segment in TARGET_SEGMENTS),
            "worst_root_reduction_at_least_15pct": worst_reduction >= float(gates["minimum_worst_root_rho_p95_reduction"]),
            "each_segment_reduction_at_least_10pct": all(value >= float(gates["minimum_each_source_target_segment_reduction"]) for value in segment_reductions.values()),
            "no_segment_regression_over_10pct": all(value >= -float(gates["maximum_any_source_target_segment_regression"]) for value in segment_reductions.values()),
            "nearest_oracle_gap_at_most_20pct_if_nearest": arm != "stable_38_nearest_root_local_ridge" or nearest_oracle_gap <= float(gates["maximum_nearest_vs_oracle_relative_rho_p95_gap"]),
        }
        passed = all(checks.values())
        if passed:
            advancing.append(arm)
        model_results.append(
            {
                "arm_id": arm,
                "metrics": metrics,
                "worst_root_rho_p95_reduction_vs_comparator": worst_reduction,
                "segment_worst_root_rho_p95_reduction_vs_comparator": segment_reductions,
                "gate_checks": checks,
                "advancement_gate_pass": passed,
            }
        )

    nearest_routes = route_by_arm["stable_38_nearest_root_local_ridge"]
    route_accuracy = float(np.mean([route == str(row["root_seed_id"]) for route, row in zip(nearest_routes, rows)]))
    status = PASS_STATUS if advancing else NO_ADVANCEMENT_STATUS
    summary: dict[str, Any] = {
        "status": status,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope_definition": config["scope_definition"],
        "adaptive_development_after_observed_failures": True,
        "not_fresh_confirmation": True,
        "protocol_path": _relative(config_path, repo_root),
        "protocol_sha256": sha256_file(config_path),
        "protocol_lock_path": _relative(lock_path, repo_root),
        "protocol_lock_sha256": sha256_file(lock_path),
        "implementation_path": _relative(Path(__file__), repo_root),
        "implementation_sha256": sha256_file(Path(__file__)),
        "input_hash_audit": input_audit,
        "all_input_hashes_match": True,
        "scoring_rows": len(rows),
        "scoring_roots": len(selected_ids),
        "common_row_identity_preserved": True,
        "model_arms": model_results,
        "advancing_arms": advancing,
        "nearest_route_accuracy": route_accuracy,
        "nearest_vs_oracle_relative_worst_root_gap": nearest_oracle_gap,
        "prediction_ledger_sealed_before_metrics": True,
        "model_manifest_content_read": True,
        "model_files_opened": True,
        "model_predictions_executed": True,
        "confirmation_content_read": False,
        "inverse_design_authorized": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "next_authorized_stage": "review_development_scoring_before_any_inverse_design_protocol" if advancing else "stop_and_review_forward_no_advancement",
        "artifacts": {
            "prediction_ledger": _relative(ledger_path, repo_root),
            "prediction_ledger_sha256": sha256_file(ledger_path),
        },
    }
    _write_text(report_path, _report(summary))
    summary["artifacts"]["report"] = _relative(report_path, repo_root)
    summary["artifacts"]["report_sha256"] = sha256_file(report_path)
    _write_json(summary_path, summary)
    return summary
