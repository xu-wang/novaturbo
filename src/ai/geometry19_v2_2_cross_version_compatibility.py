"""N0 compatibility audit for frozen targets/scorer and FAR-precision V2.2."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
import yaml

from src.ai.geometry19_far_precision_relabel import load as load_far_precision
from src.ai.geometry19_performance_gated_protocol import canonical, file_hash, read_rows, require, within
from src.ai.geometry19_v2_2_stable_root_s2_training import OUTPUT_NAMES, _local_predict, _schema_arrays
from src.ai.geometry19_v2_2_stable_root_s3_v4_1_scoring import _load_model
from src.physics.geometry19_v2_2_dataset import _evaluate_row


VERSION = "geometry19_v2_2_cross_version_compatibility_v1"
PASS = "V2_2_CROSS_VERSION_COMPATIBILITY_PASS"
FAIL = "V2_2_CROSS_VERSION_COMPATIBILITY_FAIL"


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write_once(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_once(path, b"".join(canonical(dict(row)) + b"\n" for row in rows))


def validate_protocol(config: Mapping[str, Any]) -> None:
    require(config["protocol_version"] == VERSION, "Compatibility protocol version changed")
    require(config["stage"] == "N0_target_and_surrogate_cross_version_compatibility", "Wrong audit stage")
    require(config["scope"] == "frozen_99_targets_and_frozen_s2_oracle_local_ridge", "Audit scope changed")
    runtime = config["runtime"]
    require(runtime["model_version"] == "geometry19_v2_2_far_precision_v1", "Wrong target runtime")
    require(runtime["worker_policy"] == "sequential_cpu", "Worker policy changed")
    evaluation = config["evaluation"]
    require(evaluation["performance_order"] == list(OUTPUT_NAMES), "Performance order changed")
    require(evaluation["target_source_calls"] == 99, "Target audit budget changed")
    require(evaluation["target_classes"] == {"center": 33, "typical": 33, "outer": 33}, "Target classes changed")
    require(evaluation["surrogate_arm"] == "stable_38_oracle_root_local_ridge", "Scorer arm changed")
    require(evaluation["exact_common_validation_rows"] == 3053 and evaluation["top_k"] == 8, "Scorer population changed")
    auth = config["authorization"]
    require(auth == {
        "compatibility_audit_authorized": True,
        "target_source_physics_calls_authorized": 99,
        "candidate_generation_authorized": False,
        "generator_physics_closure_authorized": False,
        "model_training_authorized": False,
        "confirmation_content_access": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }, "Audit authorization changed")
    require(config["gates"]["target_sources"] == {
        "require_all_solver_converged": True,
        "require_all_numerically_valid": True,
        "require_all_physics_domain_valid": True,
        "require_all_inside_target_domain": True,
        "maximum_performance_relative_change": 1e-6,
        "maximum_target_rho": 0.001,
    }, "Target gates changed")
    require(config["gates"]["surrogate_transport"] == {
        "require_all_predictions_finite_positive": True,
        "maximum_absolute_row_rho_change": 0.001,
        "maximum_row_rho_p95_degradation": 0.001,
        "maximum_worst_root_rho_p95_degradation": 0.001,
        "maximum_mean_target_spearman_degradation": 0.001,
        "maximum_single_target_spearman_degradation": 0.01,
        "maximum_top8_best_rho_degradation": 0.001,
    }, "Surrogate gates changed")


def _finite_positive_performance(row: Mapping[str, Any]) -> bool:
    performance = row.get("performance") or {}
    return all(type(performance.get(name)) in (int, float) and math.isfinite(performance[name]) and performance[name] > 0 for name in OUTPUT_NAMES)


def _eligible(row: Mapping[str, Any]) -> bool:
    assessment = row.get("constraint_assessment") or {}
    target = row.get("target_domain_assessment") or {}
    return bool(
        (row.get("solver") or {}).get("converged") is True
        and assessment.get("numerically_valid") is True
        and assessment.get("physics_domain_valid") is True
        and target.get("inside") is True
        and _finite_positive_performance(row)
    )


def _rho(values: np.ndarray, targets: np.ndarray, tolerances: np.ndarray) -> np.ndarray:
    return np.max(np.abs(values / targets - 1.0) / tolerances, axis=-1)


def _prediction_metrics(rows: Sequence[Mapping[str, Any]], truth: np.ndarray, prediction: np.ndarray, tolerances: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    values = _rho(prediction, truth, tolerances)
    by_root: dict[str, list[float]] = defaultdict(list)
    for value, row in zip(values.tolist(), rows):
        by_root[str(row["root_seed_id"])].append(float(value))
    root_p95 = {root: float(np.quantile(group, 0.95)) for root, group in sorted(by_root.items())}
    return ({
        "rows": len(rows),
        "roots": len(root_p95),
        "row_rho_p50": float(np.quantile(values, 0.5)),
        "row_rho_p95": float(np.quantile(values, 0.95)),
        "worst_root_rho_p95": float(max(root_p95.values())),
        "root_rho_p95": root_p95,
    }, values)


def _surrogate_audit(config: Mapping[str, Any], paths: Mapping[str, Path], targets: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    old_all = read_rows(paths["old_local_training_labels"])
    new_all = read_rows(paths["far_precision_labels"])
    require(len(old_all) == len(new_all) == 16896, "Label population changed")
    require([row["design_hash"] for row in old_all] == [row["design_hash"] for row in new_all], "Old/new label order changed")
    paired = [
        (old, new) for old, new in zip(old_all, new_all)
        if old.get("validation_eligible") is True and new.get("validation_eligible") is True
    ]
    require(len(paired) == config["evaluation"]["exact_common_validation_rows"], "Common validation population changed")
    old_rows = [item[0] for item in paired]
    new_rows = [item[1] for item in paired]
    schema = yaml.safe_load(paths["schema"].read_text(encoding="utf-8"))
    names, lower, upper = _schema_arrays(schema)
    x = np.asarray([[float(row["geometry19"][name]) for name in names] for row in new_rows], dtype=np.float64)
    x = (x - lower[None, :]) / (upper - lower)[None, :]
    arm = config["evaluation"]["surrogate_arm"]
    stable_ids, model = _load_model(paths["s2_oracle_local_ridge"], arm)
    route = {root: index for index, root in enumerate(stable_ids)}
    require(all(str(row["root_seed_id"]) in route for row in new_rows), "Validation root missing from scorer")
    route_indices = np.asarray([route[str(row["root_seed_id"])] for row in new_rows], dtype=int)
    prediction = np.exp(_local_predict(model, x, route_indices))
    old_truth = np.asarray([[float(row["performance"][name]) for name in OUTPUT_NAMES] for row in old_rows])
    new_truth = np.asarray([[float(row["performance"][name]) for name in OUTPUT_NAMES] for row in new_rows])
    tolerances = np.asarray([float(config["evaluation"]["relative_tolerances"][name]) for name in OUTPUT_NAMES])
    old_metrics, old_rho = _prediction_metrics(old_rows, old_truth, prediction, tolerances)
    new_metrics, new_rho = _prediction_metrics(new_rows, new_truth, prediction, tolerances)

    indices_by_root: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(new_rows):
        indices_by_root[str(row["root_seed_id"])].append(index)
    ranking_rows: list[dict[str, Any]] = []
    top_k = int(config["evaluation"]["top_k"])
    for target in targets:
        indices = np.asarray(indices_by_root[str(target["root_seed_id"])], dtype=int)
        require(len(indices) >= top_k, "Too few paired validation candidates for target ranking")
        target_values = np.asarray([float(target["performance"][name]) for name in OUTPUT_NAMES])
        predicted_target_rho = _rho(prediction[indices], target_values[None, :], tolerances)
        old_target_rho = _rho(old_truth[indices], target_values[None, :], tolerances)
        new_target_rho = _rho(new_truth[indices], target_values[None, :], tolerances)
        old_spearman = float(spearmanr(predicted_target_rho, old_target_rho).statistic)
        new_spearman = float(spearmanr(predicted_target_rho, new_target_rho).statistic)
        require(math.isfinite(old_spearman) and math.isfinite(new_spearman), "Non-finite target rank correlation")
        selected = np.argsort(predicted_target_rho, kind="stable")[:top_k]
        old_best = float(np.min(old_target_rho[selected]))
        new_best = float(np.min(new_target_rho[selected]))
        ranking_rows.append({
            "target_index": int(target["target_index"]),
            "target_id": target["target_id"],
            "root_seed_id": target["root_seed_id"],
            "target_class": target["target_class"],
            "candidate_rows": int(len(indices)),
            "old_spearman": old_spearman,
            "new_spearman": new_spearman,
            "spearman_degradation": old_spearman - new_spearman,
            "old_top8_best_rho": old_best,
            "new_top8_best_rho": new_best,
            "top8_best_rho_degradation": new_best - old_best,
        })
    gate = config["gates"]["surrogate_transport"]
    differences = np.abs(new_rho - old_rho)
    mean_spearman_degradation = float(np.mean([row["spearman_degradation"] for row in ranking_rows]))
    max_spearman_degradation = float(max(row["spearman_degradation"] for row in ranking_rows))
    max_top8_degradation = float(max(row["top8_best_rho_degradation"] for row in ranking_rows))
    checks = {
        "all_predictions_finite_positive": bool(np.all(np.isfinite(prediction)) and np.all(prediction > 0)),
        "maximum_absolute_row_rho_change": float(np.max(differences)) <= gate["maximum_absolute_row_rho_change"],
        "row_rho_p95_degradation": new_metrics["row_rho_p95"] - old_metrics["row_rho_p95"] <= gate["maximum_row_rho_p95_degradation"],
        "worst_root_rho_p95_degradation": new_metrics["worst_root_rho_p95"] - old_metrics["worst_root_rho_p95"] <= gate["maximum_worst_root_rho_p95_degradation"],
        "mean_target_spearman_degradation": mean_spearman_degradation <= gate["maximum_mean_target_spearman_degradation"],
        "single_target_spearman_degradation": max_spearman_degradation <= gate["maximum_single_target_spearman_degradation"],
        "top8_best_rho_degradation": max_top8_degradation <= gate["maximum_top8_best_rho_degradation"],
    }
    summary = {
        "pass": all(checks.values()),
        "checks": checks,
        "paired_validation_rows": len(new_rows),
        "paired_roots": len(indices_by_root),
        "prediction_metrics_old_physics": old_metrics,
        "prediction_metrics_far_precision": new_metrics,
        "maximum_absolute_row_rho_change": float(np.max(differences)),
        "mean_target_spearman_degradation": mean_spearman_degradation,
        "maximum_single_target_spearman_degradation": max_spearman_degradation,
        "maximum_top8_best_rho_degradation": max_top8_degradation,
        "newly_eligible_validation_rows_reported_separately": sum(
            old.get("validation_eligible") is not True and new.get("validation_eligible") is True
            for old, new in zip(old_all, new_all)
        ),
    }
    return summary, ranking_rows


def _report(summary: Mapping[str, Any]) -> str:
    target = summary["target_source_audit"]
    scorer = summary["surrogate_transport_audit"]
    return "\n".join([
        "# V2.2 FAR Precision Cross-Version Compatibility Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Target source calls: `{target['actual_calls']}`",
        f"- Target source compatible: `{str(target['pass']).lower()}`",
        f"- Maximum target performance relative change: `{target['maximum_performance_relative_change']:.12g}`",
        f"- Maximum target rho under frozen target values: `{target['maximum_target_rho']:.12g}`",
        f"- Paired scorer validation rows: `{scorer['paired_validation_rows']}`",
        f"- Scorer transport compatible: `{str(scorer['pass']).lower()}`",
        f"- Maximum absolute scorer row-rho change: `{scorer['maximum_absolute_row_rho_change']:.12g}`",
        f"- Mean target Spearman degradation: `{scorer['mean_target_spearman_degradation']:.12g}`",
        f"- Maximum selected-top8 best-rho degradation: `{scorer['maximum_top8_best_rho_degradation']:.12g}`",
        "",
        "A PASS only qualifies the frozen target values and frozen S2 Oracle local Ridge for transport to the FAR-precision V2.2 development closure. It does not establish absolute surrogate quality, generator success, unknown-root generalization, formal use, or deployment.",
        "",
    ])


def run_audit(repo_root: Path, config_path: Path, *, evaluator=None) -> dict[str, Any]:
    root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_protocol(config)
    paths: dict[str, Path] = {}
    input_audit: dict[str, Any] = {}
    for role, spec in config["inputs"].items():
        path = within(root, spec["path"])
        actual = file_hash(path)
        require(actual == spec["sha256"], f"Compatibility input changed: {role}")
        paths[role] = path
        input_audit[role] = {"path": spec["path"], "sha256": actual, "match": True}
    require(json.loads(paths["far_precision_summary"].read_text())["status"] == "RELABEL_PASS", "FAR precision relabel did not pass")
    targets = read_rows(paths["target_registry"])
    require(len(targets) == 99 and Counter(row["target_class"] for row in targets) == Counter(config["evaluation"]["target_classes"]), "Target registry changed")
    source_rows = {}
    for role in ("development_dataset", "old_target_physics_ledger"):
        for row in read_rows(paths[role]):
            source_rows.setdefault(row["design_hash"], row)
    require(all(row["source_design_hash"] in source_rows for row in targets), "Target source geometry missing")

    far = load_far_precision(root)
    context = far.candidate
    require(context.physics_runtime.model_version == config["runtime"]["model_version"], "FAR model identity changed")
    require(context.context_hash == config["runtime"]["context_sha256"], "FAR context identity changed")
    require(context.physics_runtime.physics_bundle_hash == config["runtime"]["bundle_sha256"], "FAR bundle identity changed")
    evaluator = evaluator or _evaluate_row
    output = within(root, config["artifacts"]["output_root"])
    require(not output.exists(), "Refusing to overwrite compatibility audit")
    output.mkdir(parents=True)
    journal_path = output / config["artifacts"]["call_journal"]
    target_results = []
    tolerances = np.asarray([float(config["evaluation"]["relative_tolerances"][name]) for name in OUTPUT_NAMES])
    with journal_path.open("xb") as journal:
        for target in targets:
            source = source_rows[target["source_design_hash"]]
            require(source["geometry19"] and source["design_hash"] == target["source_design_hash"], "Target source identity changed")
            call = {"target_index": target["target_index"], "target_id": target["target_id"], "design_hash": source["design_hash"], "event": "call_started"}
            journal.write(canonical(call) + b"\n")
            journal.flush()
            os.fsync(journal.fileno())
            evaluated = evaluator({
                "target_index": target["target_index"],
                "target_id": target["target_id"],
                "root_seed_id": target["root_seed_id"],
                "target_class": target["target_class"],
                "design_hash": source["design_hash"],
                "geometry19": deepcopy(source["geometry19"]),
            }, context)
            old_values = np.asarray([float(target["performance"][name]) for name in OUTPUT_NAMES])
            if _finite_positive_performance(evaluated):
                new_values = np.asarray([float(evaluated["performance"][name]) for name in OUTPUT_NAMES])
                relative = np.abs(new_values / old_values - 1.0)
                target_rho = float(np.max(relative / tolerances))
                relative_by_axis = {name: float(value) for name, value in zip(OUTPUT_NAMES, relative)}
            else:
                target_rho = None
                relative_by_axis = {name: None for name in OUTPUT_NAMES}
            target_results.append({
                **evaluated,
                "frozen_target_performance": target["performance"],
                "performance_relative_change": relative_by_axis,
                "target_rho": target_rho,
                "compatible_eligibility": _eligible(evaluated),
            })
    target_gate = config["gates"]["target_sources"]
    maximum_relative = max(value for row in target_results for value in row["performance_relative_change"].values() if value is not None)
    maximum_target_rho = max(row["target_rho"] for row in target_results if row["target_rho"] is not None)
    target_checks = {
        "all_solver_converged": all(row["solver"]["converged"] for row in target_results),
        "all_numerically_valid": all(row["constraint_assessment"]["numerically_valid"] for row in target_results),
        "all_physics_domain_valid": all(row["constraint_assessment"]["physics_domain_valid"] for row in target_results),
        "all_inside_target_domain": all((row.get("target_domain_assessment") or {}).get("inside") is True for row in target_results),
        "maximum_performance_relative_change": maximum_relative <= target_gate["maximum_performance_relative_change"],
        "maximum_target_rho": maximum_target_rho <= target_gate["maximum_target_rho"],
    }
    target_summary = {
        "pass": all(target_checks.values()),
        "checks": target_checks,
        "actual_calls": len(target_results),
        "eligible": sum(row["compatible_eligibility"] for row in target_results),
        "maximum_performance_relative_change": maximum_relative,
        "maximum_target_rho": maximum_target_rho,
        "by_class": {
            name: {"rows": sum(row["target_class"] == name for row in target_results), "eligible": sum(row["target_class"] == name and row["compatible_eligibility"] for row in target_results)}
            for name in ("center", "typical", "outer")
        },
    }
    surrogate_summary, surrogate_rows = _surrogate_audit(config, paths, targets)
    status = PASS if target_summary["pass"] and surrogate_summary["pass"] else FAIL
    summary = {
        "status": status,
        "protocol_version": VERSION,
        "protocol_sha256": file_hash(config_path),
        "runtime": config["runtime"],
        "input_audit": input_audit,
        "target_source_audit": target_summary,
        "surrogate_transport_audit": surrogate_summary,
        "candidate_generation_executed": False,
        "generator_physics_closure_executed": False,
        "model_training_executed": False,
        "confirmation_content_read": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "next_stage": config["interpretation"]["pass"] if status == PASS else config["interpretation"]["fail"],
    }
    _write_jsonl(output / config["artifacts"]["target_ledger"], target_results)
    _write_jsonl(output / config["artifacts"]["surrogate_ledger"], surrogate_rows)
    _write_json(output / config["artifacts"]["summary"], summary)
    _write_once(output / config["artifacts"]["report"], _report(summary).encode())
    return summary
