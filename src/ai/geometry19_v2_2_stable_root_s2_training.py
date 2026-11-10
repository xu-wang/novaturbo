"""S2 model selection and manifest lock for the stable-root forward surrogate."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np
from sklearn.linear_model import Ridge
import yaml

from src.ai.geometry19_surrogate import sha256_file


PROTOCOL_VERSION = "geometry19_v2_2_stable_root_s2_training_v1"
MODEL_PASS = "STABLE_ROOT_S2_MODEL_MANIFEST_LOCK_PASS"
REGRESSION_PASS = "STABLE_ROOT_S2_REGRESSION_PASS_S3_AUTHORIZED"
OUTPUT_NAMES = ("thrust_N", "tsfc_kg_N_s", "mass_kg")
TARGET_SEGMENTS = ("low_mass_low_to_mid", "low_mass_mid_to_high")


class Geometry19StableRootS2Error(ValueError):
    """Raised when S2 inputs, selection, serialization, or authorization fail."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Geometry19StableRootS2Error(message)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _artifact_path(path: Path, repo_root: Path) -> str:
    return str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path)


def validate_protocol(config: Mapping[str, Any]) -> None:
    _require(config.get("protocol_version") == PROTOCOL_VERSION, "S2 protocol version mismatch")
    _require(config.get("stage") == "S2_train_select_reload_and_hash_forward_models", "S2 stage mismatch")
    _require(config.get("formal_use_allowed") is False, "Formal use must remain forbidden")
    _require(config.get("deployment_allowed") is False, "Deployment must remain forbidden")
    inner = config.get("inner_selection", {})
    _require(inner.get("folds") == 4, "Inner fold count changed")
    _require(inner.get("assignment") == "sort_unique_design_hashes_within_each_stable_root_then_index_modulo_4", "Inner assignment changed")
    _require(inner.get("fresh_evaluation_visible") is False, "Fresh evaluation leakage")
    ridge = config.get("ridge", {})
    _require(ridge.get("solver") == "svd", "Ridge solver changed")
    _require(ridge.get("alpha_candidates") == [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0], "Alpha grid changed")
    _require(ridge.get("exact_array_reload_required") is True, "Exact reload gate missing")
    expected = [
        "all_78_global_root_balanced_ridge",
        "stable_38_pooled_root_balanced_ridge",
        "stable_38_nearest_root_local_ridge",
        "stable_38_oracle_root_local_ridge",
        "stable_38_global_plus_root_residual_ridge",
    ]
    _require([arm["id"] for arm in config.get("arms", [])] == expected, "S2 arms changed")
    confirmation = config.get("inputs", {}).get("sealed_confirmation", {})
    _require(confirmation.get("access") == "sha256_only_no_parse", "Confirmation access changed")


def _schema_arrays(schema: Mapping[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    variables = sorted(schema["design_variables"], key=lambda item: int(item["index"]))
    names = [str(item["name"]) for item in variables]
    lower = np.asarray([float(item["bounds"][0]) for item in variables])
    upper = np.asarray([float(item["bounds"][1]) for item in variables])
    _require(len(names) == 19 and np.all(upper > lower), "Invalid frozen design schema")
    return names, lower, upper


def _arrays(
    records: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    lower: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x = np.asarray(
        [[float(row["geometry19"][name]) for name in names] for row in records],
        dtype=np.float64,
    )
    y = np.asarray(
        [[float(row["performance"][name]) for name in OUTPUT_NAMES] for row in records],
        dtype=np.float64,
    )
    _require(np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(y > 0.0), "Non-finite or non-positive training data")
    return (x - lower[None, :]) / scale[None, :], np.log(y), [str(row["root_seed_id"]) for row in records]


def _root_balanced_weights(root_ids: Sequence[str]) -> np.ndarray:
    counts = Counter(root_ids)
    total = len(root_ids)
    roots = len(counts)
    weights = np.asarray([total / (roots * counts[root_id]) for root_id in root_ids])
    _require(np.isclose(np.mean(weights), 1.0), "Root weights do not have mean one")
    return weights


def _fit(x: np.ndarray, y_log: np.ndarray, root_ids: Sequence[str], alpha: float) -> dict[str, np.ndarray]:
    model = Ridge(alpha=alpha, fit_intercept=True, solver="svd")
    model.fit(x, y_log, sample_weight=_root_balanced_weights(root_ids))
    return {
        "coef": np.asarray(model.coef_, dtype=np.float64),
        "intercept": np.asarray(model.intercept_, dtype=np.float64),
    }


def _linear_predict(model: Mapping[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    return x @ model["coef"].T + model["intercept"][None, :]


def _fit_local(
    x: np.ndarray,
    y_log: np.ndarray,
    root_ids: Sequence[str],
    stable_ids: Sequence[str],
    alpha: float,
) -> dict[str, np.ndarray]:
    coefs: list[np.ndarray] = []
    intercepts: list[np.ndarray] = []
    root_array = np.asarray(root_ids)
    for root_id in stable_ids:
        indices = np.flatnonzero(root_array == root_id)
        _require(len(indices) > 0, f"No local training rows: {root_id}")
        model = _fit(x[indices], y_log[indices], [root_id] * len(indices), alpha)
        coefs.append(model["coef"])
        intercepts.append(model["intercept"])
    return {
        "coefs": np.asarray(coefs),
        "intercepts": np.asarray(intercepts),
    }


def _local_predict(
    model: Mapping[str, np.ndarray],
    x: np.ndarray,
    route_indices: np.ndarray,
) -> np.ndarray:
    return np.einsum("nij,nj->ni", model["coefs"][route_indices], x) + model[
        "intercepts"
    ][route_indices]


def _rho_metrics(
    truth_log: np.ndarray,
    prediction_log: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    tolerances: np.ndarray,
) -> dict[str, Any]:
    truth = np.exp(truth_log)
    prediction = np.exp(prediction_log)
    rho = np.max(np.abs(prediction / truth - 1.0) / tolerances[None, :], axis=1)
    by_root: dict[str, list[float]] = defaultdict(list)
    by_segment_root: dict[tuple[str, str], list[float]] = defaultdict(list)
    for value, row in zip(rho.tolist(), records):
        root_id = str(row["root_seed_id"])
        segment = str(row["assigned_segment"])
        by_root[root_id].append(float(value))
        by_segment_root[(segment, root_id)].append(float(value))
    root_p95 = {
        root_id: float(np.quantile(values, 0.95))
        for root_id, values in sorted(by_root.items())
    }
    by_segment: dict[str, Any] = {}
    for segment in TARGET_SEGMENTS:
        values = [
            float(np.quantile(items, 0.95))
            for (item_segment, _), items in by_segment_root.items()
            if item_segment == segment
        ]
        _require(bool(values), f"No validation rows for target segment {segment}")
        by_segment[segment] = {
            "roots": len(values),
            "worst_root_rho_p95": float(max(values)),
            "root_rho_p95_p95": float(np.quantile(values, 0.95)),
        }
    return {
        "rows": len(records),
        "roots": len(root_p95),
        "row_rho_p50": float(np.quantile(rho, 0.50)),
        "row_rho_p95": float(np.quantile(rho, 0.95)),
        "worst_stable_root_rho_p95": float(max(root_p95.values())),
        "root_rho_p95_p95": float(np.quantile(list(root_p95.values()), 0.95)),
        "by_target_segment": by_segment,
        "root_rho_p95": root_p95,
        "all_predictions_finite_positive": bool(
            np.all(np.isfinite(prediction)) and np.all(prediction > 0.0)
        ),
    }


def _selection_key(item: Mapping[str, Any]) -> tuple[float, float, float, float]:
    metrics = item["oof_metrics"]
    return (
        float(metrics["worst_stable_root_rho_p95"]),
        float(metrics["by_target_segment"][TARGET_SEGMENTS[0]]["worst_root_rho_p95"]),
        float(metrics["by_target_segment"][TARGET_SEGMENTS[1]]["worst_root_rho_p95"]),
        float(item["alpha"]),
    )


def _save_model(path: Path, arm_id: str, alpha: float, model: Mapping[str, np.ndarray], stable_ids: Sequence[str]) -> None:
    payload = {
        "arm_id": np.asarray(arm_id),
        "alpha": np.asarray(alpha, dtype=np.float64),
        "stable_root_ids": np.asarray(stable_ids),
        **{key: np.asarray(value) for key, value in model.items()},
    }
    _write_npz(path, **payload)


def _reload_exact(path: Path, expected: Mapping[str, np.ndarray], arm_id: str, alpha: float, stable_ids: Sequence[str]) -> dict[str, np.ndarray]:
    with np.load(path) as bundle:
        _require(str(bundle["arm_id"].item()) == arm_id, "Reloaded arm ID mismatch")
        _require(float(bundle["alpha"].item()) == alpha, "Reloaded alpha mismatch")
        _require([str(value) for value in bundle["stable_root_ids"].tolist()] == list(stable_ids), "Reloaded roots mismatch")
        reloaded = {key: np.asarray(bundle[key]) for key in expected}
    _require(all(np.array_equal(expected[key], reloaded[key]) for key in expected), "Model arrays changed after reload")
    return reloaded


def _report(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Stable-Root S2 Forward Model Training",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Models locked: {len(manifest['models'])}",
        "- Fresh coordinates read/generated: `false`",
        "- Confirmation content parsed: `false`",
        "- S3 authorized before full regression: `false`",
        "",
        "| Arm | Selected alpha | Inner worst-root rho P95 | Exact reload |",
        "|---|---:|---:|---|",
    ]
    for model in manifest["models"]:
        lines.append(
            f"| `{model['arm_id']}` | {model['selected_alpha']:.4g} | "
            f"{model['selected_oof_metrics']['worst_stable_root_rho_p95']:.4f} | "
            f"`{str(model['exact_reload_equal']).lower()}` |"
        )
    lines.extend(
        [
            "",
            "Inner metrics select regularization only. They are not fresh accuracy evidence. S3 remains locked until a zero-failure full-regression JUnit artifact is hashed together with this model manifest.",
            "",
        ]
    )
    return "\n".join(lines)


def run_s2_training(
    repo_root: Path,
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Select, fit, serialize, and reload all five S2 Ridge arms."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_protocol(config)
    resolved: dict[str, Path] = {}
    for role, item in config["inputs"].items():
        path = (repo_root / str(item["path"])).resolve()
        _require(path.is_file(), f"Missing S2 input: {role}")
        _require(sha256_file(path) == str(item["sha256"]), f"S2 input hash mismatch: {role}")
        resolved[role] = path
    parent_lock = json.loads(resolved["parent_protocol_lock"].read_text(encoding="utf-8"))
    _require(parent_lock.get("status") == config["inputs"]["parent_protocol_lock"]["required_status"], "Parent lock status mismatch")
    _require(parent_lock.get("s2_training_authorized") is True, "Parent did not authorize S2")
    _require(parent_lock.get("confirmation_content_read") is False, "Parent confirmation isolation failed")

    registry = _read_jsonl(resolved["stable_root_registry"])
    stable_ids = sorted(str(row["root_seed_id"]) for row in registry)
    _require(len(stable_ids) == 38 and len(set(stable_ids)) == 38, "Stable registry mismatch")
    centers = np.asarray([row["normalized_design_center"] for row in sorted(registry, key=lambda row: str(row["root_seed_id"]))])
    records = _read_jsonl(resolved["development_dataset"])
    stable_set = set(stable_ids)
    stable_records = [row for row in records if str(row["root_seed_id"]) in stable_set]
    _require(len(records) == 4529 and len(stable_records) == 2935, "S2 population mismatch")
    schema = yaml.safe_load(resolved["design_schema"].read_text(encoding="utf-8"))
    names, lower, upper = _schema_arrays(schema)
    scale = upper - lower
    target = yaml.safe_load(resolved["target_domain"].read_text(encoding="utf-8"))
    tolerances = np.asarray([float(target["axes"][name]["relative_tolerance"]) for name in OUTPUT_NAMES])
    x_all, y_all, roots_all = _arrays(records, names, lower, scale)
    x_stable, y_stable, roots_stable = _arrays(stable_records, names, lower, scale)
    stable_hash_to_fold: dict[str, int] = {}
    by_root_records: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in stable_records:
        by_root_records[str(row["root_seed_id"])].append(row)
    for root_id in stable_ids:
        for index, row in enumerate(sorted(by_root_records[root_id], key=lambda value: str(value["design_hash"]))):
            stable_hash_to_fold[str(row["design_hash"])] = index % 4
    stable_folds = np.asarray([stable_hash_to_fold[str(row["design_hash"])] for row in stable_records])
    all_stable_mask = np.asarray([root_id in stable_set for root_id in roots_all])
    stable_index_by_id = {root_id: index for index, root_id in enumerate(stable_ids)}

    runtime_root = output_root.resolve() if output_root is not None else (repo_root / str(config["artifacts"]["runtime_root"])).resolve()
    s3_paths = [
        runtime_root / "fresh_evaluation/S3_Prelabel_Coordinates.jsonl",
        runtime_root / "fresh_evaluation/S3_Sealed_V2_2_Evaluations.jsonl",
        runtime_root / "audits/S3_Fresh_Evaluation_Summary.json",
    ]
    _require(not any(path.exists() for path in s3_paths), "S3 artifact exists before S2 lock")
    model_dir = runtime_root / str(config["artifacts"]["model_directory"])
    selection_path = runtime_root / str(config["artifacts"]["inner_selection"])
    manifest_path = runtime_root / str(config["artifacts"]["model_manifest"])
    report_path = runtime_root / str(config["artifacts"]["training_report"])

    alpha_candidates = [float(value) for value in config["ridge"]["alpha_candidates"]]
    arm_ids = [str(item["id"]) for item in config["arms"]]
    candidate_results: dict[str, list[dict[str, Any]]] = {arm_id: [] for arm_id in arm_ids}
    for arm_id in arm_ids:
        for alpha in alpha_candidates:
            oof_prediction = np.empty_like(y_stable)
            route_correct: list[bool] = []
            for fold in range(4):
                validation = stable_folds == fold
                stable_train = ~validation
                validation_records = [row for row, flag in zip(stable_records, validation.tolist()) if flag]
                if arm_id == "all_78_global_root_balanced_ridge":
                    validation_hashes = {str(row["design_hash"]) for row in validation_records}
                    train_all = np.asarray([str(row["design_hash"]) not in validation_hashes for row in records])
                    model = _fit(x_all[train_all], y_all[train_all], np.asarray(roots_all)[train_all].tolist(), alpha)
                    prediction = _linear_predict(model, x_stable[validation])
                elif arm_id == "stable_38_pooled_root_balanced_ridge":
                    model = _fit(x_stable[stable_train], y_stable[stable_train], np.asarray(roots_stable)[stable_train].tolist(), alpha)
                    prediction = _linear_predict(model, x_stable[validation])
                elif arm_id in ("stable_38_nearest_root_local_ridge", "stable_38_oracle_root_local_ridge"):
                    model = _fit_local(x_stable[stable_train], y_stable[stable_train], np.asarray(roots_stable)[stable_train].tolist(), stable_ids, alpha)
                    if arm_id == "stable_38_nearest_root_local_ridge":
                        routes = np.argmin(np.linalg.norm(x_stable[validation, None, :] - centers[None, :, :], axis=2), axis=1)
                        actual = np.asarray([stable_index_by_id[str(row["root_seed_id"])] for row in validation_records])
                        route_correct.extend((routes == actual).tolist())
                    else:
                        routes = np.asarray([stable_index_by_id[str(row["root_seed_id"])] for row in validation_records])
                    prediction = _local_predict(model, x_stable[validation], routes)
                else:
                    base = _fit(x_stable[stable_train], y_stable[stable_train], np.asarray(roots_stable)[stable_train].tolist(), alpha)
                    base_train = _linear_predict(base, x_stable[stable_train])
                    residual = _fit_local(x_stable[stable_train], y_stable[stable_train] - base_train, np.asarray(roots_stable)[stable_train].tolist(), stable_ids, alpha)
                    routes = np.asarray([stable_index_by_id[str(row["root_seed_id"])] for row in validation_records])
                    prediction = _linear_predict(base, x_stable[validation]) + _local_predict(residual, x_stable[validation], routes)
                oof_prediction[validation] = prediction
            metrics = _rho_metrics(y_stable, oof_prediction, stable_records, tolerances)
            candidate_results[arm_id].append(
                {
                    "alpha": alpha,
                    "oof_metrics": metrics,
                    "nearest_route_accuracy": float(np.mean(route_correct)) if route_correct else None,
                }
            )

    selected = {
        arm_id: min(candidate_results[arm_id], key=_selection_key) for arm_id in arm_ids
    }
    models: list[dict[str, Any]] = []
    for arm_id in arm_ids:
        alpha = float(selected[arm_id]["alpha"])
        if arm_id == "all_78_global_root_balanced_ridge":
            model = _fit(x_all, y_all, roots_all, alpha)
            final_x = x_all
            final_routes = None
            prediction_before = _linear_predict(model, final_x)
        elif arm_id == "stable_38_pooled_root_balanced_ridge":
            model = _fit(x_stable, y_stable, roots_stable, alpha)
            final_x = x_stable
            final_routes = None
            prediction_before = _linear_predict(model, final_x)
        elif arm_id in ("stable_38_nearest_root_local_ridge", "stable_38_oracle_root_local_ridge"):
            model = _fit_local(x_stable, y_stable, roots_stable, stable_ids, alpha)
            final_x = x_stable
            if arm_id == "stable_38_nearest_root_local_ridge":
                final_routes = np.argmin(np.linalg.norm(x_stable[:, None, :] - centers[None, :, :], axis=2), axis=1)
            else:
                final_routes = np.asarray([stable_index_by_id[root_id] for root_id in roots_stable])
            prediction_before = _local_predict(model, final_x, final_routes)
        else:
            base = _fit(x_stable, y_stable, roots_stable, alpha)
            residual = _fit_local(x_stable, y_stable - _linear_predict(base, x_stable), roots_stable, stable_ids, alpha)
            model = {
                "base_coef": base["coef"],
                "base_intercept": base["intercept"],
                "residual_coefs": residual["coefs"],
                "residual_intercepts": residual["intercepts"],
            }
            final_x = x_stable
            final_routes = np.asarray([stable_index_by_id[root_id] for root_id in roots_stable])
            prediction_before = (
                final_x @ model["base_coef"].T
                + model["base_intercept"][None, :]
                + np.einsum("nij,nj->ni", model["residual_coefs"][final_routes], final_x)
                + model["residual_intercepts"][final_routes]
            )
        _require(all(np.all(np.isfinite(value)) for value in model.values()), f"Non-finite coefficients: {arm_id}")
        model_path = model_dir / f"{arm_id}.npz"
        _save_model(model_path, arm_id, alpha, model, stable_ids)
        reloaded = _reload_exact(model_path, model, arm_id, alpha, stable_ids)
        if arm_id in ("all_78_global_root_balanced_ridge", "stable_38_pooled_root_balanced_ridge"):
            prediction_after = _linear_predict(reloaded, final_x)
        elif arm_id in ("stable_38_nearest_root_local_ridge", "stable_38_oracle_root_local_ridge"):
            prediction_after = _local_predict(reloaded, final_x, final_routes)
        else:
            prediction_after = (
                final_x @ reloaded["base_coef"].T
                + reloaded["base_intercept"][None, :]
                + np.einsum("nij,nj->ni", reloaded["residual_coefs"][final_routes], final_x)
                + reloaded["residual_intercepts"][final_routes]
            )
        exact_prediction = bool(np.array_equal(prediction_before, prediction_after))
        _require(exact_prediction, f"Prediction changed after reload: {arm_id}")
        _require(np.all(np.isfinite(np.exp(prediction_after))) and np.all(np.exp(prediction_after) > 0.0), f"Invalid development prediction: {arm_id}")
        models.append(
            {
                "arm_id": arm_id,
                "selected_alpha": alpha,
                "selected_oof_metrics": selected[arm_id]["oof_metrics"],
                "selected_nearest_route_accuracy": selected[arm_id]["nearest_route_accuracy"],
                "model_path": _artifact_path(model_path, repo_root),
                "model_sha256": sha256_file(model_path),
                "coefficient_arrays": {key: list(value.shape) for key, value in model.items()},
                "all_coefficients_finite": True,
                "exact_reload_equal": True,
                "exact_prediction_reload_equal": exact_prediction,
                "eligible_development_predictions_finite_positive": True,
            }
        )

    selection_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "aggregation": config["inner_selection"]["aggregation"],
        "fresh_evaluation_visible": False,
        "support_rejection_used_for_selection": False,
        "candidate_results": candidate_results,
        "selected_alpha_by_arm": {arm_id: float(selected[arm_id]["alpha"]) for arm_id in arm_ids},
    }
    _write_json(selection_path, selection_payload)
    implementation_path = Path(__file__).resolve()
    manifest: dict[str, Any] = {
        "status": MODEL_PASS,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_path": _artifact_path(config_path, repo_root),
        "protocol_sha256": sha256_file(config_path),
        "parent_protocol_lock_sha256": sha256_file(resolved["parent_protocol_lock"]),
        "implementation_path": _artifact_path(implementation_path, repo_root),
        "implementation_sha256": sha256_file(implementation_path),
        "stable_roots": 38,
        "stable_development_rows": 2935,
        "all_root_comparator_rows": 4529,
        "models": models,
        "all_five_models_locked": len(models) == 5,
        "inner_selection_path": _artifact_path(selection_path, repo_root),
        "inner_selection_sha256": sha256_file(selection_path),
        "model_training_executed": True,
        "exact_reload_pass": all(model["exact_reload_equal"] and model["exact_prediction_reload_equal"] for model in models),
        "fresh_evaluation_visible_during_selection": False,
        "fresh_coordinates_read_or_generated": False,
        "fresh_physics_labels_read_or_generated": False,
        "confirmation_hash_checked": True,
        "confirmation_content_read": False,
        "inverse_design_authorized": False,
        "original_local_chart_p4_authorized": False,
        "full_regression_pass_recorded": False,
        "s3_coordinate_generation_authorized": False,
        "next_authorized_stage": "run_full_regression_and_lock_S2_authorization",
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }
    _write_text(report_path, _report(manifest))
    manifest["training_report_path"] = _artifact_path(report_path, repo_root)
    manifest["training_report_sha256"] = sha256_file(report_path)
    _write_json(manifest_path, manifest)
    return manifest


def authorize_s3_from_junit(
    repo_root: Path,
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Authorize S3 only from a locked model manifest and passing JUnit artifact."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_protocol(config)
    runtime_root = output_root.resolve() if output_root is not None else (repo_root / str(config["artifacts"]["runtime_root"])).resolve()
    manifest_path = runtime_root / str(config["artifacts"]["model_manifest"])
    junit_path = runtime_root / str(config["artifacts"]["regression_junit"])
    authorization_path = runtime_root / str(config["artifacts"]["regression_authorization"])
    _require(manifest_path.is_file(), "S2 model manifest missing")
    _require(junit_path.is_file(), "S2 regression JUnit missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("status") == MODEL_PASS, "S2 model manifest did not pass")
    _require(manifest.get("protocol_sha256") == sha256_file(config_path), "S2 protocol hash changed")
    _require(manifest.get("exact_reload_pass") is True, "S2 reload gate failed")
    root = ET.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    _require(tests >= int(config["regression_authorization"]["minimum_tests"]), "Regression test count below frozen minimum")
    _require(failures == 0 and errors == 0, "Regression JUnit contains failures or errors")
    s3_paths = [
        runtime_root / "fresh_evaluation/S3_Prelabel_Coordinates.jsonl",
        runtime_root / "fresh_evaluation/S3_Sealed_V2_2_Evaluations.jsonl",
    ]
    _require(not any(path.exists() for path in s3_paths), "S3 data already exists before authorization")
    authorization = {
        "status": REGRESSION_PASS,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_path": _artifact_path(config_path, repo_root),
        "protocol_sha256": sha256_file(config_path),
        "model_manifest_path": _artifact_path(manifest_path, repo_root),
        "model_manifest_sha256": sha256_file(manifest_path),
        "junit_path": _artifact_path(junit_path, repo_root),
        "junit_sha256": sha256_file(junit_path),
        "tests": tests,
        "failures": failures,
        "errors": errors,
        "s3_coordinate_generation_authorized": True,
        "fresh_coordinates_generated": False,
        "fresh_physics_labels_generated": False,
        "inverse_design_authorized": False,
        "original_local_chart_p4_authorized": False,
        "confirmation_content_read": False,
        "next_authorized_stage": "S3_generate_prelabel_coordinates_only",
    }
    _write_json(authorization_path, authorization)
    return authorization
