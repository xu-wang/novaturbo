"""Complete S3 v4 medoid coordinates in bilateral boundary tangent spaces."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_stable_root_s3_v4_medoid import (
    _canonical_bytes,
    _common_limit,
    _geometry,
    _hashes,
    _normalized,
    _read_jsonl,
    _relative,
    _schema_arrays,
    _write_json,
    _write_jsonl,
    _write_npz,
    _write_text,
    build_root_direction_model,
)


PROTOCOL_VERSION = "geometry19_v2_2_stable_root_s3_v4_1_boundary_completion_v1"
PASS_STATUS = "STABLE_ROOT_S3_V4_1_BOUNDARY_COMPLETION_LOCK_PASS"
FAIL_STATUS = "STABLE_ROOT_S3_V4_1_BOUNDARY_COMPLETION_LOCK_FAIL"


class Geometry19StableRootS3V41Error(ValueError):
    """Raised when the frozen v4.1 completion contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Geometry19StableRootS3V41Error(message)


def validate_protocol(config: Mapping[str, Any]) -> None:
    _require(config.get("protocol_version") == PROTOCOL_VERSION, "v4.1 protocol mismatch")
    _require(config.get("stage") == "S3_v4_1_complete_medoid_coordinates_in_bilateral_tangent_subspace", "v4.1 stage mismatch")
    _require(config.get("scope_definition") == "post_selected_33root_development_only", "v4.1 scope changed")
    _require(config.get("formal_use_allowed") is False, "Formal use must remain forbidden")
    _require(config.get("deployment_allowed") is False, "Deployment must remain forbidden")
    completion = config.get("completion_scope", {})
    _require(completion.get("carry_forward_complete_v4_roots") == 29, "Carry-forward root count changed")
    _require(completion.get("complete_missing_roots") == 4, "Missing root count changed")
    _require(len(set(completion.get("missing_root_ids", []))) == 4, "Missing roots changed")
    tangent = config.get("bilateral_tangent_subspace", {})
    _require(tangent.get("fixed_axis_rule") == "normalized_medoid_exactly_zero_or_one", "Fixed-axis rule changed")
    _require(tangent.get("fixed_axes_must_be_integer_variables") is True, "Non-integer axes cannot be silently fixed")
    _require(tangent.get("one_sided_variation_allowed") is False, "One-sided variation cannot be allowed")
    model = config.get("local_direction_model", {})
    _require(model.get("performance_values_read") is False, "Performance cannot shape completion directions")
    _require(model.get("primary_seed_base") == 20260921, "Primary seed changed")
    _require(model.get("reserve_seed_base") == 20260922, "Reserve seed changed")
    _require(model.get("primary_count_per_root") == 48, "Primary count changed")
    _require(model.get("reserve_count_per_root") == 256, "Reserve count changed")
    scaling = config.get("scaling", {})
    _require(scaling.get("shell_fractions") == [0.10, 0.20], "Shells changed")
    _require(scaling.get("minimum_actual_to_desired_step_ratio") == 0.25, "Step ratio changed")
    gate = config.get("combined_gate", {})
    _require(gate.get("exact_roots") == 33 and gate.get("exact_rows_total") == 3168, "Combined gate changed")
    authorization = config.get("authorization", {})
    _require(authorization.get("coordinate_completion_authorized_now") is True, "Completion not authorized")
    _require(authorization.get("physics_evaluation_executed_now") is False, "Physics cannot run now")
    _require(authorization.get("model_prediction_authorized") is False, "Prediction cannot run now")
    _require(config["inputs"]["sealed_confirmation"]["access"] == "sha256_only_no_parse", "Confirmation access changed")
    _require(config["inputs"]["s2_model_manifest"]["access"] == "sha256_only_no_parse", "Model access changed")


def _report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Stable-Root S3 v4.1 Boundary Completion",
            "",
            f"- Status: `{summary['status']}`",
            f"- v4 roots carried forward: {summary['carried_forward_roots']}",
            f"- Boundary roots completed: {summary['completed_boundary_roots']}",
            f"- Completion coordinate rows: {summary['completion_rows']}",
            f"- Combined coordinate rows: {summary['combined_rows']}",
            f"- Reserve pairs selected: {summary['reserve_pairs_selected']}",
            f"- Future physics protocol authorized: `{str(summary['future_physics_protocol_authorized']).lower()}`",
            "- Physics/model predictions executed: `false`",
            "- Confirmation/model manifest content read: `false`",
            "",
            "Discrete schema-bound axes are held fixed only for the four affected roots. The combined set is adaptive development evidence, not fresh confirmation.",
            "",
        ]
    )


def complete_s3_v4_boundary_coordinates(
    repo_root: Path,
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Carry forward 29 complete roots and generate four tangent-space completions."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_protocol(config)
    parsed_roles = {
        "v4_protocol",
        "v4_failure_summary",
        "v4_accepted_ledger",
        "v4_anchor_registry",
        "stable_root_registry",
        "design_schema",
        "development_dataset",
    }
    resolved: dict[str, Path] = {}
    input_audit: dict[str, Any] = {}
    for role, item in config["inputs"].items():
        path = (repo_root / str(item["path"])).resolve()
        _require(path.is_file(), f"Missing v4.1 input: {role}")
        actual = sha256_file(path)
        _require(actual == str(item["sha256"]), f"v4.1 input hash mismatch: {role}")
        resolved[role] = path
        input_audit[role] = {
            "path": str(item["path"]),
            "expected_sha256": str(item["sha256"]),
            "actual_sha256": actual,
            "match": True,
            "content_parsed": role in parsed_roles,
        }
    v4_summary = json.loads(resolved["v4_failure_summary"].read_text(encoding="utf-8"))
    _require(v4_summary.get("status") == config["inputs"]["v4_failure_summary"]["required_status"], "v4 failure status mismatch")
    carried = _read_jsonl(resolved["v4_accepted_ledger"])
    _require(len(carried) == int(config["inputs"]["v4_accepted_ledger"]["expected_rows"]), "v4 accepted row count mismatch")
    carried_counts = Counter(str(row["root_seed_id"]) for row in carried)
    _require(len(carried_counts) == 29 and set(carried_counts.values()) == {96}, "v4 carry-forward roots differ")
    missing_ids = sorted(str(value) for value in config["completion_scope"]["missing_root_ids"])
    _require(not (set(missing_ids) & set(carried_counts)), "Missing roots already appear in v4 accepted rows")
    _require(set(v4_summary["selected_pairs_per_root"]) == set(carried_counts), "v4 summary and ledger differ")

    v4_protocol = yaml.safe_load(resolved["v4_protocol"].read_text(encoding="utf-8"))
    prior_hashes: set[str] = set()
    for role in ("development_dataset", "v1_coordinate_ledger", "v2_coordinate_ledger", "v3_coordinate_ledger"):
        item = v4_protocol["inputs"][role]
        path = (repo_root / str(item["path"])).resolve()
        _require(sha256_file(path) == str(item["sha256"]), f"Inherited v4 exclusion hash mismatch: {role}")
        prior_hashes.update(_hashes(path))
    prior_hashes.update(str(row["design_hash"]) for row in carried)

    schema = yaml.safe_load(resolved["design_schema"].read_text(encoding="utf-8"))
    names, lower, upper, integer_names = _schema_arrays(schema)
    scale = upper - lower
    name_to_index = {name: index for index, name in enumerate(names)}
    development = _read_jsonl(resolved["development_dataset"])
    development_by_root: dict[str, list[dict[str, Any]]] = {}
    development_by_hash = {str(row["design_hash"]): row for row in development}
    for row in development:
        if str(row["root_seed_id"]) in missing_ids:
            development_by_root.setdefault(str(row["root_seed_id"]), []).append(row)
    registry = {str(row["root_seed_id"]): row for row in _read_jsonl(resolved["stable_root_registry"])}
    anchors = {str(row["root_seed_id"]): row for row in _read_jsonl(resolved["v4_anchor_registry"])}
    _require(set(missing_ids) <= set(anchors), "Missing v4 anchors")

    models: dict[str, dict[str, np.ndarray]] = {}
    fixed_indices: dict[str, list[int]] = {}
    centers: dict[str, np.ndarray] = {}
    for root_id in missing_ids:
        anchor_record = anchors[root_id]
        medoid_hash = str(anchor_record["medoid_design_hash"])
        medoid = development_by_hash[medoid_hash]
        center = np.asarray(anchor_record["normalized_medoid"], dtype=np.float64)
        indices = [index for index, value in enumerate(center) if value <= 1.0e-15 or value >= 1.0 - 1.0e-15]
        _require(bool(indices), f"Missing root has no exact boundary axis: {root_id}")
        _require(all(names[index] in integer_names for index in indices), f"Non-integer boundary axis would be fixed: {root_id}")
        model = build_root_direction_model(root_id, center, development_by_root[root_id], names, lower, scale, config)
        for key in ("primary_directions", "reserve_directions"):
            directions = model[key].copy()
            directions[:, indices] = 0.0
            norms = np.linalg.norm(directions, axis=1)
            _require(np.all(norms > 1.0e-15), f"Projected direction collapsed: {root_id}")
            model[key] = directions / norms[:, None]
        centers[root_id] = center
        fixed_indices[root_id] = indices
        models[root_id] = model
        _require(bool(medoid.get("training_candidate")), f"Completion medoid not V2.2 eligible: {root_id}")

    eta = float(config["scaling"]["common_two_sided_schema_safety_factor"])
    minimum_ratio = float(config["scaling"]["minimum_actual_to_desired_step_ratio"])
    selected_hashes: set[str] = set()
    screening: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for root_id in missing_ids:
        root = registry[root_id]
        center = centers[root_id]
        model = models[root_id]
        radius = float(root["support_radius_p95_normalized_l2"])
        fixed_names = [names[index] for index in fixed_indices[root_id]]
        for shell in (0.10, 0.20):
            candidates: list[tuple[str, int, np.ndarray]] = []
            for index in np.flatnonzero(model["primary_shell"] == shell).tolist():
                candidates.append(("primary", index, model["primary_directions"][index]))
            for index in np.flatnonzero(model["reserve_shell"] == shell).tolist():
                candidates.append(("reserve", index, model["reserve_directions"][index]))
            shell_selected = 0
            for pool, pool_index, direction in candidates:
                if shell_selected >= 24:
                    break
                desired = radius * shell
                actual = min(desired, eta * _common_limit(center, direction))
                ratio = actual / desired if desired > 0.0 else 0.0
                plus_raw = center + actual * direction
                minus_raw = center - actual * direction
                plus_geometry = _geometry(plus_raw, names, lower, scale, integer_names)
                minus_geometry = _geometry(minus_raw, names, lower, scale, integer_names)
                plus_hash = sha256(_canonical_bytes(plus_geometry)).hexdigest()
                minus_hash = sha256(_canonical_bytes(minus_geometry)).hexdigest()
                reasons: list[str] = []
                if ratio < minimum_ratio:
                    reasons.append("two_sided_safe_step_ratio_below_minimum")
                if not (np.all((plus_raw >= 0.0) & (plus_raw <= 1.0)) and np.all((minus_raw >= 0.0) & (minus_raw <= 1.0))):
                    reasons.append("schema_bounds_after_projection")
                if plus_hash == minus_hash:
                    reasons.append("pair_design_hashes_identical")
                if plus_hash in prior_hashes or minus_hash in prior_hashes:
                    reasons.append("prior_design_hash_duplicate")
                if plus_hash in selected_hashes or minus_hash in selected_hashes:
                    reasons.append("previously_selected_v4_1_duplicate")
                accepted = not reasons
                identity = {"root_seed_id": root_id, "shell": shell, "pool": pool, "pool_index": pool_index, "plus_hash": plus_hash, "minus_hash": minus_hash}
                row = {
                    "screening_id": sha256(_canonical_bytes(identity)).hexdigest(),
                    "root_seed_id": root_id,
                    "anchor_design_hash": str(root["medoid"]["design_hash"]),
                    "source_root_segments": root["segments"],
                    "fixed_boundary_axes": fixed_names,
                    "shell_fraction": shell,
                    "direction_pool": pool,
                    "pool_index": pool_index,
                    "direction_family": "root_local_covariance_projected_bilateral_tangent",
                    "desired_step_normalized_l2": desired,
                    "actual_step_normalized_l2": actual,
                    "actual_to_desired_step_ratio": ratio,
                    "screening_status": "PAIR_SELECTED" if accepted else "PAIR_REJECTED",
                    "rejection_reasons": reasons,
                    "plus_design_hash": plus_hash,
                    "minus_design_hash": minus_hash,
                    "label_status": "NOT_EVALUATED_DEVELOPMENT_COORDINATE_ONLY",
                }
                screening.append(row)
                if accepted:
                    selected_hashes.update((plus_hash, minus_hash))
                    pairs.append({**row, "plus_raw": plus_raw, "minus_raw": minus_raw, "plus_geometry": plus_geometry, "minus_geometry": minus_geometry})
                    shell_selected += 1
            if shell_selected < 24:
                screening.append(
                    {
                        "screening_id": sha256(_canonical_bytes({"root_seed_id": root_id, "shell": shell, "failure": "reserve_exhausted"})).hexdigest(),
                        "root_seed_id": root_id,
                        "anchor_design_hash": str(root["medoid"]["design_hash"]),
                        "source_root_segments": root["segments"],
                        "fixed_boundary_axes": fixed_names,
                        "shell_fraction": shell,
                        "direction_pool": "none",
                        "pool_index": None,
                        "direction_family": "none",
                        "desired_step_normalized_l2": radius * shell,
                        "actual_step_normalized_l2": None,
                        "actual_to_desired_step_ratio": None,
                        "screening_status": "SHELL_SELECTION_INCOMPLETE",
                        "rejection_reasons": ["reserve_directions_exhausted"],
                        "plus_design_hash": None,
                        "minus_design_hash": None,
                        "label_status": "NOT_EVALUATED_DEVELOPMENT_COORDINATE_ONLY",
                    }
                )

    by_root: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        by_root.setdefault(str(pair["root_seed_id"]), []).append(pair)
    completion_rows: list[dict[str, Any]] = []
    for root_id in missing_ids:
        for pair_index, pair in enumerate(by_root.get(root_id, [])):
            for sign, raw_key, geometry_key, hash_key in ((1, "plus_raw", "plus_geometry", "plus_design_hash"), (-1, "minus_raw", "minus_geometry", "minus_design_hash")):
                raw = pair[raw_key]
                proposal_hash = sha256(_canonical_bytes({"protocol_version": PROTOCOL_VERSION, "screening_id": pair["screening_id"], "root_seed_id": root_id, "pair_index": pair_index, "sign": sign, "raw": raw.tolist()})).hexdigest()
                completion_rows.append(
                    {
                        "attempt_id": f"s3v4c:{root_id[:12]}:{pair_index:02d}:{'p' if sign > 0 else 'm'}:{proposal_hash[:12]}",
                        "proposal_hash": proposal_hash,
                        "screening_id": pair["screening_id"],
                        "root_seed_id": root_id,
                        "anchor_design_hash": pair["anchor_design_hash"],
                        "source_root_segments": pair["source_root_segments"],
                        "fixed_boundary_axes": pair["fixed_boundary_axes"],
                        "selected_pair_index": pair_index,
                        "direction_pool": pair["direction_pool"],
                        "pool_index": pair["pool_index"],
                        "direction_family": pair["direction_family"],
                        "radius_fraction": pair["shell_fraction"],
                        "desired_step_normalized_l2": pair["desired_step_normalized_l2"],
                        "actual_step_normalized_l2": pair["actual_step_normalized_l2"],
                        "actual_to_desired_step_ratio": pair["actual_to_desired_step_ratio"],
                        "sign": sign,
                        "raw_normalized_geometry19": {name: float(raw[index]) for index, name in enumerate(names)},
                        "geometry19": pair[geometry_key],
                        "design_hash": pair[hash_key],
                        "coordinate_status": "COORDINATE_ELIGIBLE",
                        "label_status": "NOT_EVALUATED_DEVELOPMENT_COORDINATE_ONLY",
                        "combined_source_protocol": "v4_1_boundary_completion",
                    }
                )
    carried_for_combined = [{**row, "combined_source_protocol": "v4_medoid_covariance"} for row in carried]
    combined = sorted(carried_for_combined + completion_rows, key=lambda row: (str(row["root_seed_id"]), int(row["selected_pair_index"]), -int(row["sign"])))
    counts = Counter(str(row["root_seed_id"]) for row in combined)
    hashes = [str(row["design_hash"]) for row in combined]
    fixed_axes_preserved = all(
        all(row["geometry19"][name] == anchors[str(row["root_seed_id"])]["medoid_geometry19"][name] for name in row["fixed_boundary_axes"])
        for row in completion_rows
    )
    gate_pass = all(
        (
            len(completion_rows) == 384,
            len(combined) == 3168,
            len(counts) == 33,
            set(counts.values()) == {96},
            len(hashes) == len(set(hashes)),
            not (set(str(row["design_hash"]) for row in completion_rows) & prior_hashes),
            all(all(0.0 <= float(value) <= 1.0 for value in row["raw_normalized_geometry19"].values()) for row in combined),
            all(isinstance(row["geometry19"][name], int) for row in combined for name in integer_names),
            fixed_axes_preserved,
        )
    )
    status = PASS_STATUS if gate_pass else FAIL_STATUS

    runtime_root = output_root.resolve() if output_root is not None else (repo_root / str(config["artifacts"]["runtime_root"])).resolve()
    screening_path = runtime_root / str(config["artifacts"]["completion_screening_ledger"])
    completion_path = runtime_root / str(config["artifacts"]["completion_coordinates"])
    combined_path = runtime_root / str(config["artifacts"]["combined_coordinates"])
    direction_path = runtime_root / str(config["artifacts"]["direction_matrix"])
    summary_path = runtime_root / str(config["artifacts"]["summary"])
    report_path = runtime_root / str(config["artifacts"]["report"])
    _write_jsonl(screening_path, screening)
    _write_jsonl(completion_path, completion_rows)
    _write_jsonl(combined_path, combined)
    _write_npz(
        direction_path,
        root_ids=np.asarray(missing_ids),
        fixed_axis_masks=np.asarray([[index in fixed_indices[root_id] for index in range(19)] for root_id in missing_ids]),
        primary_directions=np.asarray([models[root_id]["primary_directions"] for root_id in missing_ids]),
        primary_shell=models[missing_ids[0]]["primary_shell"],
        reserve_directions=np.asarray([models[root_id]["reserve_directions"] for root_id in missing_ids]),
        reserve_shell=models[missing_ids[0]]["reserve_shell"],
    )
    rejection_reasons = Counter(reason for row in screening for reason in row["rejection_reasons"])
    summary: dict[str, Any] = {
        "status": status,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope_definition": config["scope_definition"],
        "adaptive_development_after_v4_failure": True,
        "not_fresh_confirmation": True,
        "protocol_path": _relative(config_path, repo_root),
        "protocol_sha256": sha256_file(config_path),
        "implementation_path": _relative(Path(__file__), repo_root),
        "implementation_sha256": sha256_file(Path(__file__)),
        "input_hash_audit": input_audit,
        "all_input_hashes_match": True,
        "carried_forward_roots": len(carried_counts),
        "carried_forward_rows": len(carried),
        "completed_boundary_roots": len(missing_ids),
        "missing_root_ids": missing_ids,
        "fixed_boundary_axes_per_root": {root_id: [names[index] for index in fixed_indices[root_id]] for root_id in missing_ids},
        "screening_pairs_considered": len(screening),
        "screening_pairs_rejected": sum(row["screening_status"] == "PAIR_REJECTED" for row in screening),
        "screening_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "reserve_pairs_selected": sum(row["screening_status"] == "PAIR_SELECTED" and row["direction_pool"] == "reserve" for row in screening),
        "completion_selected_pairs": len(pairs),
        "completion_rows": len(completion_rows),
        "combined_rows": len(combined),
        "combined_rows_per_root": dict(sorted(counts.items())),
        "combined_design_hashes_unique": len(hashes) == len(set(hashes)),
        "completion_prior_overlap": bool(set(str(row["design_hash"]) for row in completion_rows) & prior_hashes),
        "fixed_boundary_axes_preserved": fixed_axes_preserved,
        "physics_evaluations_executed": False,
        "performance_labels_written": False,
        "model_manifest_content_read": False,
        "model_files_opened": False,
        "model_predictions_executed": False,
        "confirmation_content_read": False,
        "future_physics_protocol_authorized": gate_pass,
        "inverse_design_authorized": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "next_authorized_stage": "freeze_v4_1_development_physics_acquisition_protocol" if gate_pass else "stop_and_review_v4_1_completion_failure",
        "artifacts": {
            "completion_screening_ledger": _relative(screening_path, repo_root),
            "completion_screening_ledger_sha256": sha256_file(screening_path),
            "completion_coordinates": _relative(completion_path, repo_root),
            "completion_coordinates_sha256": sha256_file(completion_path),
            "combined_coordinates": _relative(combined_path, repo_root),
            "combined_coordinates_sha256": sha256_file(combined_path),
            "direction_matrix": _relative(direction_path, repo_root),
            "direction_matrix_sha256": sha256_file(direction_path),
        },
    }
    _write_text(report_path, _report(summary))
    summary["artifacts"]["report"] = _relative(report_path, repo_root)
    summary["artifacts"]["report_sha256"] = sha256_file(report_path)
    _write_json(summary_path, summary)
    return summary
