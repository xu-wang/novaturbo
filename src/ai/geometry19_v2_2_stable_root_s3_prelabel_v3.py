"""Freeze fresh paired coordinates for the redefined 33-root stable subdomain."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import qmc
import yaml

from src.ai.geometry19_surrogate import sha256_file


PROTOCOL_VERSION = "geometry19_v2_2_stable_root_s3_prelabel_v3_33root"
PASS_STATUS = "STABLE_ROOT_S3_V3_33ROOT_PRELABEL_COORDINATE_LOCK_PASS"
FAIL_STATUS = "STABLE_ROOT_S3_V3_33ROOT_PRELABEL_COORDINATE_LOCK_FAIL"


class Geometry19StableRootS3V3Error(ValueError):
    """Raised when the frozen 33-root coordinate contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Geometry19StableRootS3V3Error(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def validate_protocol(config: Mapping[str, Any]) -> None:
    _require(config.get("protocol_version") == PROTOCOL_VERSION, "S3 v3 protocol mismatch")
    _require(
        config.get("stage") == "S3_v3_33root_independent_prelabel_coordinate_generation",
        "S3 v3 stage mismatch",
    )
    _require(config.get("scope_definition") == "bilateral_perturbable_stable_root_subdomain_only", "Scope definition changed")
    _require(config.get("formal_use_allowed") is False, "Formal use must remain forbidden")
    _require(config.get("deployment_allowed") is False, "Deployment must remain forbidden")
    scope = config.get("scope", {})
    selected = scope.get("selected_root_ids", [])
    excluded = scope.get("excluded_root_ids", [])
    _require(scope.get("selected_root_count") == len(selected) == 33, "Selected root count changed")
    _require(scope.get("excluded_root_count") == len(excluded) == 5, "Excluded root count changed")
    _require(len(set(selected)) == 33 and len(set(excluded)) == 5, "Root IDs are not unique")
    _require(not (set(selected) & set(excluded)), "Selected and excluded roots overlap")
    directions = config.get("directions", {})
    _require(directions.get("primary_count") == 48, "Primary direction count changed")
    _require(directions.get("primary_seed") == 20260917, "Primary direction seed changed")
    _require(directions.get("reserve_count") == 256, "Reserve direction count changed")
    _require(directions.get("reserve_seed") == 20260918, "Reserve direction seed changed")
    scaling = config.get("boundary_aware_scaling", {})
    _require(scaling.get("safety_factor_eta") == 0.95, "Safety factor changed")
    _require(scaling.get("minimum_actual_to_desired_ratio") == 0.25, "Minimum step ratio changed")
    _require(scaling.get("clipping_allowed") is False, "Clipping cannot be allowed")
    selection = config.get("selection", {})
    _require(selection.get("target_selected_pairs_per_root") == 48, "Pair target changed")
    _require(selection.get("target_selected_pairs_per_shell") == 24, "Shell target changed")
    _require(selection.get("final_prelabel_rows_total") == 3168, "Row target changed")
    authorization = config.get("authorization", {})
    _require(authorization.get("coordinate_generation_authorized_now") is True, "Coordinate generation not authorized")
    _require(authorization.get("physics_labeling_executed_now") is False, "Physics cannot run now")
    _require(authorization.get("model_prediction_authorized") is False, "Prediction cannot run now")
    _require(authorization.get("inverse_design_authorized") is False, "Inverse design cannot run now")
    _require(config["inputs"]["sealed_confirmation"]["access"] == "sha256_only_no_parse", "Confirmation access changed")
    _require(config["inputs"]["s2_model_manifest"]["access"] == "sha256_only_no_parse", "Model manifest access changed")


def build_v3_directions(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Build independent, deterministic primary and reserve Sobol pools."""

    definition = config["directions"]
    primary_engine = qmc.Sobol(d=19, scramble=True, seed=int(definition["primary_seed"]))
    primary = 2.0 * primary_engine.random_base2(m=6)[:48] - 1.0
    primary /= np.linalg.norm(primary, axis=1)[:, None]
    reserve_engine = qmc.Sobol(d=19, scramble=True, seed=int(definition["reserve_seed"]))
    reserve = 2.0 * reserve_engine.random_base2(m=8) - 1.0
    reserve /= np.linalg.norm(reserve, axis=1)[:, None]
    primary_shell = np.asarray([0.5 if index % 2 == 0 else 0.9 for index in range(48)])
    reserve_shell = np.asarray([0.5 if index % 2 == 0 else 0.9 for index in range(256)])
    _require(primary.shape == (48, 19), "Primary direction shape mismatch")
    _require(reserve.shape == (256, 19), "Reserve direction shape mismatch")
    _require(np.sum(primary_shell == 0.5) == np.sum(primary_shell == 0.9) == 24, "Primary shell mismatch")
    _require(np.sum(reserve_shell == 0.5) == np.sum(reserve_shell == 0.9) == 128, "Reserve shell mismatch")
    return {
        "primary_directions": primary,
        "primary_shell": primary_shell,
        "reserve_directions": reserve,
        "reserve_shell": reserve_shell,
    }


def _common_two_sided_limit(center: np.ndarray, direction: np.ndarray) -> float:
    active = np.abs(direction) > 1.0e-15
    return float(
        np.min(
            np.minimum(center[active], 1.0 - center[active])
            / np.abs(direction[active])
        )
    )


def _geometry_from_normalized(
    normalized: np.ndarray,
    names: Sequence[str],
    lower: np.ndarray,
    scale: np.ndarray,
    integer_names: set[str],
) -> dict[str, Any]:
    physical = lower + normalized * scale
    return {
        name: int(np.rint(physical[index])) if name in integer_names else float(physical[index])
        for index, name in enumerate(names)
    }


def _design_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    for row in _read_jsonl(path):
        value = row.get("design_hash")
        if value:
            hashes.add(str(value))
    return hashes


def _report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Stable-Root S3 v3 33-Root Independent Prelabel Coordinates",
            "",
            f"- Status: `{summary['status']}`",
            "- Applicable scope: 33 bilateral-perturbable stable roots only",
            f"- Excluded boundary roots: {summary['excluded_root_count']}",
            f"- Screening pairs considered: {summary['screening_pairs_considered']}",
            f"- Screening pairs rejected: {summary['screening_pairs_rejected']}",
            f"- Reserve pairs selected: {summary['reserve_pairs_selected']}",
            f"- Final selected pairs: {summary['selected_pairs']}",
            f"- Final prelabel rows: {summary['final_prelabel_rows']}",
            f"- Physics labeling authorized: `{str(summary['physics_labeling_authorized']).lower()}`",
            "- Physics evaluations executed: `false`",
            "- Model manifest/model files read: `false`",
            "- Confirmation content parsed: `false`",
            "",
            "The 33-root scope was selected after observing S3 v2 coordinate results. This is a restricted-domain experiment, not a repaired 38-root result. Coordinate success does not establish physics validity or surrogate accuracy.",
            "",
        ]
    )


def generate_s3_v3_prelabel_coordinates(
    repo_root: Path,
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Generate fresh coordinates without reading labels, models, or confirmation."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_protocol(config)
    resolved: dict[str, Path] = {}
    audit: dict[str, Any] = {}
    parsed_roles = {
        "v2_failure_summary",
        "v1_ledger",
        "v2_ledger",
        "stable_root_registry",
        "design_schema",
        "development_dataset",
    }
    for role, item in config["inputs"].items():
        path = (repo_root / str(item["path"])).resolve()
        _require(path.is_file(), f"Missing S3 v3 input: {role}")
        actual = sha256_file(path)
        _require(actual == str(item["sha256"]), f"S3 v3 input hash mismatch: {role}")
        resolved[role] = path
        audit[role] = {
            "path": str(item["path"]),
            "expected_sha256": str(item["sha256"]),
            "actual_sha256": actual,
            "match": True,
            "content_parsed": role in parsed_roles,
        }

    v2_summary = json.loads(resolved["v2_failure_summary"].read_text(encoding="utf-8"))
    _require(v2_summary.get("status") == config["inputs"]["v2_failure_summary"]["required_status"], "S3 v2 failure status mismatch")
    expected_selected = {
        root_id
        for root_id, count in v2_summary["selected_pairs_per_root"].items()
        if int(count) == 48
    }
    expected_excluded = set(v2_summary["selected_pairs_per_root"]) - expected_selected
    selected_ids = set(config["scope"]["selected_root_ids"])
    excluded_ids = set(config["scope"]["excluded_root_ids"])
    _require(selected_ids == expected_selected, "33-root selection does not reproduce frozen S3 v2 rule")
    _require(excluded_ids == expected_excluded, "Excluded root list does not reproduce frozen S3 v2 rule")

    schema = yaml.safe_load(resolved["design_schema"].read_text(encoding="utf-8"))
    variables = sorted(schema["design_variables"], key=lambda item: int(item["index"]))
    names = [str(item["name"]) for item in variables]
    lower = np.asarray([float(item["bounds"][0]) for item in variables])
    upper = np.asarray([float(item["bounds"][1]) for item in variables])
    scale = upper - lower
    integer_names = {
        str(item["name"])
        for item in variables
        if str(item.get("dtype", "")) == "int64"
    }
    registry_all = _read_jsonl(resolved["stable_root_registry"])
    registry_by_id = {str(row["root_seed_id"]): row for row in registry_all}
    _require(set(registry_by_id) == selected_ids | excluded_ids, "Stable registry does not match frozen 38-root scope")
    registry = [registry_by_id[root_id] for root_id in sorted(selected_ids)]
    prior_hashes = _design_hashes(resolved["development_dataset"])
    prior_hashes.update(_design_hashes(resolved["v1_ledger"]))
    prior_hashes.update(_design_hashes(resolved["v2_ledger"]))
    directions = build_v3_directions(config)
    eta = float(config["boundary_aware_scaling"]["safety_factor_eta"])
    minimum_ratio = float(config["boundary_aware_scaling"]["minimum_actual_to_desired_ratio"])
    selected_hashes: set[str] = set()
    screening: list[dict[str, Any]] = []
    selected_pairs: list[dict[str, Any]] = []

    for root_row in registry:
        root_id = str(root_row["root_seed_id"])
        center = np.asarray(root_row["normalized_design_center"], dtype=np.float64)
        radius = float(root_row["support_radius_p95_normalized_l2"])
        for shell in (0.5, 0.9):
            candidates: list[tuple[str, int, np.ndarray]] = []
            for index in np.flatnonzero(directions["primary_shell"] == shell).tolist():
                candidates.append(("primary", index, directions["primary_directions"][index]))
            for index in np.flatnonzero(directions["reserve_shell"] == shell).tolist():
                candidates.append(("reserve", index, directions["reserve_directions"][index]))
            shell_selected = 0
            for pool, pool_index, direction in candidates:
                if shell_selected >= 24:
                    break
                desired_step = radius * shell
                common_limit = _common_two_sided_limit(center, direction)
                actual_step = min(desired_step, eta * common_limit)
                step_ratio = actual_step / desired_step if desired_step > 0.0 else 0.0
                plus_raw = center + actual_step * direction
                minus_raw = center - actual_step * direction
                plus_geometry = _geometry_from_normalized(plus_raw, names, lower, scale, integer_names)
                minus_geometry = _geometry_from_normalized(minus_raw, names, lower, scale, integer_names)
                plus_hash = sha256(_canonical_bytes(plus_geometry)).hexdigest()
                minus_hash = sha256(_canonical_bytes(minus_geometry)).hexdigest()
                reasons: list[str] = []
                if step_ratio < minimum_ratio:
                    reasons.append("two_sided_safe_step_ratio_below_minimum")
                if not (
                    np.all((plus_raw >= 0.0) & (plus_raw <= 1.0))
                    and np.all((minus_raw >= 0.0) & (minus_raw <= 1.0))
                ):
                    reasons.append("schema_bounds_after_scaling")
                if plus_hash == minus_hash:
                    reasons.append("pair_design_hashes_identical")
                if plus_hash in prior_hashes or minus_hash in prior_hashes:
                    reasons.append("prior_dataset_or_S3_design_hash_duplicate")
                if plus_hash in selected_hashes or minus_hash in selected_hashes:
                    reasons.append("previously_selected_v3_design_hash_duplicate")
                accepted = not reasons
                identity = {
                    "root_seed_id": root_id,
                    "shell": shell,
                    "pool": pool,
                    "pool_index": pool_index,
                    "plus_hash": plus_hash,
                    "minus_hash": minus_hash,
                }
                row = {
                    "screening_id": sha256(_canonical_bytes(identity)).hexdigest(),
                    "root_seed_id": root_id,
                    "source_root_segments": root_row["segments"],
                    "scope_status": "IN_33ROOT_BILATERAL_SUBDOMAIN",
                    "shell_fraction": shell,
                    "direction_pool": pool,
                    "pool_index": pool_index,
                    "direction_family": "fresh_sobol_mixed",
                    "desired_step_normalized_l2": desired_step,
                    "actual_step_normalized_l2": actual_step,
                    "actual_to_desired_step_ratio": step_ratio,
                    "screening_status": "PAIR_SELECTED" if accepted else "PAIR_REJECTED",
                    "rejection_reasons": reasons,
                    "plus_design_hash": plus_hash,
                    "minus_design_hash": minus_hash,
                    "label_status": "NOT_EVALUATED_PRELABEL_ONLY",
                }
                screening.append(row)
                if accepted:
                    selected_hashes.update((plus_hash, minus_hash))
                    selected_pairs.append(
                        {
                            **row,
                            "plus_raw": plus_raw,
                            "minus_raw": minus_raw,
                            "plus_geometry": plus_geometry,
                            "minus_geometry": minus_geometry,
                        }
                    )
                    shell_selected += 1
            if shell_selected < 24:
                screening.append(
                    {
                        "screening_id": sha256(_canonical_bytes({"root_seed_id": root_id, "shell": shell, "failure": "reserve_exhausted"})).hexdigest(),
                        "root_seed_id": root_id,
                        "source_root_segments": root_row["segments"],
                        "scope_status": "IN_33ROOT_BILATERAL_SUBDOMAIN",
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
                        "label_status": "NOT_EVALUATED_PRELABEL_ONLY",
                    }
                )

    by_root: dict[str, list[dict[str, Any]]] = {}
    for pair in selected_pairs:
        by_root.setdefault(str(pair["root_seed_id"]), []).append(pair)
    final_rows: list[dict[str, Any]] = []
    for root_id in sorted(selected_ids):
        for pair_index, pair in enumerate(by_root.get(root_id, [])):
            for sign, raw_key, geometry_key, hash_key in (
                (1, "plus_raw", "plus_geometry", "plus_design_hash"),
                (-1, "minus_raw", "minus_geometry", "minus_design_hash"),
            ):
                raw = pair[raw_key]
                proposal_hash = sha256(
                    _canonical_bytes(
                        {
                            "protocol_version": PROTOCOL_VERSION,
                            "screening_id": pair["screening_id"],
                            "root_seed_id": root_id,
                            "selected_pair_index": pair_index,
                            "sign": sign,
                            "raw_normalized_geometry19": raw.tolist(),
                        }
                    )
                ).hexdigest()
                final_rows.append(
                    {
                        "attempt_id": f"s3v3:{root_id[:12]}:{pair_index:02d}:{'p' if sign > 0 else 'm'}:{proposal_hash[:12]}",
                        "proposal_hash": proposal_hash,
                        "screening_id": pair["screening_id"],
                        "root_seed_id": root_id,
                        "source_root_segments": pair["source_root_segments"],
                        "scope_status": "IN_33ROOT_BILATERAL_SUBDOMAIN",
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
                        "rejection_reasons": [],
                        "label_status": "NOT_EVALUATED_PRELABEL_ONLY",
                    }
                )

    pair_counts = Counter(str(row["root_seed_id"]) for row in selected_pairs)
    shell_counts = Counter((str(row["root_seed_id"]), str(row["shell_fraction"])) for row in selected_pairs)
    final_hashes = [str(row["design_hash"]) for row in final_rows]
    exact_pairs = set(pair_counts) == selected_ids and set(pair_counts.values()) == {48}
    exact_shells = all(
        shell_counts[(root_id, "0.5")] == 24 and shell_counts[(root_id, "0.9")] == 24
        for root_id in selected_ids
    )
    unique = len(final_hashes) == len(set(final_hashes))
    no_prior_overlap = not (set(final_hashes) & prior_hashes)
    inside = all(
        all(0.0 <= float(value) <= 1.0 for value in row["raw_normalized_geometry19"].values())
        for row in final_rows
    )
    gate_pass = all((exact_pairs, exact_shells, len(final_rows) == 3168, unique, no_prior_overlap, inside))
    status = PASS_STATUS if gate_pass else FAIL_STATUS

    runtime_root = output_root.resolve() if output_root is not None else (repo_root / str(config["artifacts"]["runtime_root"])).resolve()
    scope_path = runtime_root / str(config["artifacts"]["scope_registry"])
    screening_path = runtime_root / str(config["artifacts"]["screening_ledger"])
    final_path = runtime_root / str(config["artifacts"]["final_prelabel_ledger"])
    direction_path = runtime_root / str(config["artifacts"]["direction_matrix"])
    summary_path = runtime_root / str(config["artifacts"]["summary"])
    report_path = runtime_root / str(config["artifacts"]["report"])
    scope_registry = {
        "scope_definition": config["scope_definition"],
        "selection_rule": config["scope"]["selection_rule"],
        "selected_root_ids": sorted(selected_ids),
        "excluded_root_ids": sorted(excluded_ids),
        "excluded_status": config["scope"]["excluded_status"],
        "claims_forbidden": config["scope"]["claims_forbidden"],
        "post_selection_bias_acknowledged": True,
    }
    _write_json(scope_path, scope_registry)
    _write_jsonl(screening_path, screening)
    _write_jsonl(final_path, final_rows)
    _write_npz(direction_path, **directions)
    rejection_reasons = Counter(reason for row in screening for reason in row["rejection_reasons"])
    summary: dict[str, Any] = {
        "status": status,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope_definition": config["scope_definition"],
        "post_selection_bias_acknowledged": True,
        "selected_root_count": len(selected_ids),
        "excluded_root_count": len(excluded_ids),
        "excluded_root_ids": sorted(excluded_ids),
        "protocol_path": _relative(config_path, repo_root),
        "protocol_sha256": sha256_file(config_path),
        "implementation_path": _relative(Path(__file__), repo_root),
        "implementation_sha256": sha256_file(Path(__file__)),
        "input_hash_audit": audit,
        "all_input_hashes_match": True,
        "scope_selection_reproduces_frozen_v2": True,
        "screening_pairs_considered": len(screening),
        "screening_pairs_rejected": sum(row["screening_status"] == "PAIR_REJECTED" for row in screening),
        "screening_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "primary_pairs_selected": sum(row["screening_status"] == "PAIR_SELECTED" and row["direction_pool"] == "primary" for row in screening),
        "reserve_pairs_selected": sum(row["screening_status"] == "PAIR_SELECTED" and row["direction_pool"] == "reserve" for row in screening),
        "selected_pairs": len(selected_pairs),
        "selected_pairs_per_root": dict(sorted(pair_counts.items())),
        "selected_pairs_per_root_shell": {f"{root_id}:{shell}": count for (root_id, shell), count in sorted(shell_counts.items())},
        "minimum_selected_pairs_per_root": min(pair_counts.values()) if pair_counts else 0,
        "final_prelabel_rows": len(final_rows),
        "final_design_hashes_unique": unique,
        "prior_dataset_or_S3_overlap": not no_prior_overlap,
        "all_final_coordinates_inside_schema": inside,
        "model_manifest_content_read": False,
        "model_files_opened": False,
        "model_predictions_executed": False,
        "physics_evaluations_executed": False,
        "performance_labels_written": False,
        "confirmation_content_read": False,
        "physics_labeling_authorized": gate_pass,
        "inverse_design_authorized": False,
        "original_local_chart_p4_authorized": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "artifacts": {
            "scope_registry": _relative(scope_path, repo_root),
            "scope_registry_sha256": sha256_file(scope_path),
            "screening_ledger": _relative(screening_path, repo_root),
            "screening_ledger_sha256": sha256_file(screening_path),
            "final_prelabel_ledger": _relative(final_path, repo_root),
            "final_prelabel_ledger_sha256": sha256_file(final_path),
            "direction_matrix": _relative(direction_path, repo_root),
            "direction_matrix_sha256": sha256_file(direction_path),
        },
        "next_authorized_stage": "freeze_33root_V2_2_label_and_accuracy_protocol" if gate_pass else "stop_and_review_S3_v3_coordinate_failure",
    }
    _write_text(report_path, _report(summary))
    summary["artifacts"]["report"] = _relative(report_path, repo_root)
    summary["artifacts"]["report_sha256"] = sha256_file(report_path)
    _write_json(summary_path, summary)
    return summary
