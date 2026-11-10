"""Boundary-aware S3 v2 prelabel coordinate generation."""

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


PROTOCOL_VERSION = "geometry19_v2_2_stable_root_s3_prelabel_v2"
PASS_STATUS = "STABLE_ROOT_S3_V2_PRELABEL_COORDINATE_LOCK_PASS"
FAIL_STATUS = "STABLE_ROOT_S3_V2_PRELABEL_COORDINATE_LOCK_FAIL"


class Geometry19StableRootS3V2Error(ValueError):
    """Raised when the S3 v2 protocol or isolation contract fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Geometry19StableRootS3V2Error(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_text(
        path,
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
    )


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
    _require(config.get("protocol_version") == PROTOCOL_VERSION, "S3 v2 protocol mismatch")
    _require(config.get("stage") == "S3_v2_boundary_aware_prelabel_coordinate_generation", "S3 v2 stage mismatch")
    _require(config.get("formal_use_allowed") is False, "Formal use must remain forbidden")
    _require(config.get("deployment_allowed") is False, "Deployment must remain forbidden")
    scaling = config.get("boundary_aware_scaling", {})
    _require(scaling.get("safety_factor_eta") == 0.95, "Safety factor changed")
    _require(scaling.get("minimum_actual_to_desired_ratio") == 0.25, "Minimum step ratio changed")
    _require(scaling.get("clipping_allowed") is False, "Clipping cannot be allowed")
    reserve = config.get("reserve_directions", {})
    _require(reserve.get("count") == 64, "Reserve count changed")
    _require(reserve.get("sobol_seed") == 20260916, "Reserve seed changed")
    selection = config.get("selection", {})
    _require(selection.get("target_selected_pairs_per_root") == 48, "Selected pair count changed")
    _require(selection.get("target_selected_pairs_per_shell") == 24, "Shell pair count changed")
    _require(selection.get("final_prelabel_rows_total") == 3648, "Final row count changed")
    ledger = config.get("ledger", {})
    _require(ledger.get("performance_fields_allowed") is False, "Performance fields cannot be allowed")
    _require(ledger.get("physics_fields_allowed") is False, "Physics fields cannot be allowed")
    _require(ledger.get("model_prediction_fields_allowed") is False, "Prediction fields cannot be allowed")
    authorization = config.get("authorization", {})
    _require(authorization.get("physics_labeling_executed_now") is False, "Physics cannot run in prelabel v2")
    _require(authorization.get("model_prediction_authorized") is False, "Prediction cannot be authorized")
    _require(authorization.get("inverse_design_authorized") is False, "Inverse design cannot be authorized")
    _require(config.get("inputs", {}).get("sealed_confirmation", {}).get("access") == "sha256_only_no_parse", "Confirmation access changed")


def build_v2_directions(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Build frozen primary and reserve direction pools."""

    primary_axis = np.eye(19, dtype=np.float64)
    primary_engine = qmc.Sobol(d=19, scramble=True, seed=20260915)
    primary_sobol = 2.0 * primary_engine.random_base2(m=5)[:29] - 1.0
    primary_sobol /= np.linalg.norm(primary_sobol, axis=1)[:, None]
    primary = np.vstack([primary_axis, primary_sobol])
    primary_family = np.asarray(["schema_axis"] * 19 + ["sobol_mixed"] * 29)
    primary_family_index = np.asarray(list(range(19)) + list(range(29)), dtype=np.int64)
    primary_shell = np.asarray(
        [0.5 if index <= 9 else 0.9 for index in range(19)]
        + [0.5 if index <= 13 else 0.9 for index in range(29)]
    )
    reserve_engine = qmc.Sobol(
        d=19,
        scramble=True,
        seed=int(config["reserve_directions"]["sobol_seed"]),
    )
    reserve = 2.0 * reserve_engine.random_base2(m=6) - 1.0
    reserve /= np.linalg.norm(reserve, axis=1)[:, None]
    reserve_shell = np.asarray([0.5 if index % 2 == 0 else 0.9 for index in range(64)])
    _require(primary.shape == (48, 19) and reserve.shape == (64, 19), "Direction pool shape mismatch")
    _require(np.sum(primary_shell == 0.5) == np.sum(primary_shell == 0.9) == 24, "Primary shell mismatch")
    _require(np.sum(reserve_shell == 0.5) == np.sum(reserve_shell == 0.9) == 32, "Reserve shell mismatch")
    return {
        "primary_directions": primary,
        "primary_family": primary_family,
        "primary_family_index": primary_family_index,
        "primary_shell": primary_shell,
        "reserve_directions": reserve,
        "reserve_shell": reserve_shell,
    }


def _common_two_sided_limit(center: np.ndarray, direction: np.ndarray) -> float:
    active = np.abs(direction) > 1.0e-15
    limits = np.minimum(center[active], 1.0 - center[active]) / np.abs(direction[active])
    return float(np.min(limits))


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


def _report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Stable-Root S3 v2 Boundary-Aware Prelabel Coordinates",
            "",
            f"- Status: `{summary['status']}`",
            f"- Screening pairs considered: {summary['screening_pairs_considered']}",
            f"- Screening pairs rejected: {summary['screening_pairs_rejected']}",
            f"- Reserve pairs selected: {summary['reserve_pairs_selected']}",
            f"- Final selected pairs: {summary['selected_pairs']}",
            f"- Final prelabel rows: {summary['final_prelabel_rows']}",
            f"- Minimum pairs per root: {summary['minimum_selected_pairs_per_root']}",
            f"- Physics labeling authorized: `{str(summary['physics_labeling_authorized']).lower()}`",
            "- Physics evaluations executed: `false`",
            "- Model files opened/predictions executed: `false`",
            "- Performance fields written: `false`",
            "- Confirmation content parsed: `false`",
            "",
            "This v2 protocol is an adaptive coordinate-only response to the preserved v1 boundary failure. Coordinate success does not imply V2.2 physics validity or surrogate accuracy.",
            "",
        ]
    )


def generate_s3_v2_prelabel_coordinates(
    repo_root: Path,
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Create boundary-aware paired coordinates without physics or predictions."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_protocol(config)
    resolved: dict[str, Path] = {}
    input_hash_audit: dict[str, Any] = {}
    hash_only_roles = {"v1_ledger", "s2_model_manifest", "sealed_confirmation"}
    for role, item in config["inputs"].items():
        path = (repo_root / str(item["path"])).resolve()
        _require(path.is_file(), f"Missing S3 v2 input: {role}")
        actual = sha256_file(path)
        _require(actual == str(item["sha256"]), f"S3 v2 input hash mismatch: {role}")
        resolved[role] = path
        input_hash_audit[role] = {
            "path": str(item["path"]),
            "actual_sha256": actual,
            "expected_sha256": str(item["sha256"]),
            "match": True,
            "content_parsed": role not in hash_only_roles,
        }
    v1_summary = json.loads(resolved["v1_failure_summary"].read_text(encoding="utf-8"))
    _require(v1_summary.get("status") == config["inputs"]["v1_failure_summary"]["required_status"], "v1 failure status mismatch")
    _require(v1_summary.get("physics_labeling_authorized") is False, "v1 unexpectedly authorized physics")
    authorization = json.loads(resolved["s2_regression_authorization"].read_text(encoding="utf-8"))
    _require(authorization.get("status") == config["inputs"]["s2_regression_authorization"]["required_status"], "S2 authorization mismatch")
    _require(authorization.get("s3_coordinate_generation_authorized") is True, "S2 did not authorize coordinate generation")

    schema = yaml.safe_load(resolved["design_schema"].read_text(encoding="utf-8"))
    variables = sorted(schema["design_variables"], key=lambda item: int(item["index"]))
    names = [str(item["name"]) for item in variables]
    lower = np.asarray([float(item["bounds"][0]) for item in variables])
    upper = np.asarray([float(item["bounds"][1]) for item in variables])
    scale = upper - lower
    integer_names = set(config["integer_axis_policy"]["variable_names"])
    integer_indices = {index for index, name in enumerate(names) if name in integer_names}
    registry = sorted(_read_jsonl(resolved["stable_root_registry"]), key=lambda row: str(row["root_seed_id"]))
    _require(len(registry) == 38, "Stable registry count mismatch")
    development_hashes = {
        str(json.loads(line)["design_hash"])
        for line in resolved["development_dataset"].read_text(encoding="utf-8").splitlines()
    }
    directions = build_v2_directions(config)
    selected_hashes: set[str] = set()
    screening: list[dict[str, Any]] = []
    selected_pairs: list[dict[str, Any]] = []
    eta = float(config["boundary_aware_scaling"]["safety_factor_eta"])
    minimum_ratio = float(config["boundary_aware_scaling"]["minimum_actual_to_desired_ratio"])

    for root_row in registry:
        root_id = str(root_row["root_seed_id"])
        center = np.asarray(root_row["normalized_design_center"], dtype=np.float64)
        radius = float(root_row["support_radius_p95_normalized_l2"])
        root_selected: list[dict[str, Any]] = []
        for shell in (0.5, 0.9):
            candidates: list[tuple[str, int, str, int, np.ndarray]] = []
            for pool_index in np.flatnonzero(directions["primary_shell"] == shell).tolist():
                candidates.append(
                    (
                        "primary",
                        pool_index,
                        str(directions["primary_family"][pool_index]),
                        int(directions["primary_family_index"][pool_index]),
                        directions["primary_directions"][pool_index],
                    )
                )
            for reserve_index in np.flatnonzero(directions["reserve_shell"] == shell).tolist():
                candidates.append(
                    (
                        "reserve",
                        reserve_index,
                        "sobol_reserve",
                        reserve_index,
                        directions["reserve_directions"][reserve_index],
                    )
                )
            shell_selected = 0
            for pool, pool_index, family, family_index, direction in candidates:
                if shell_selected >= 24:
                    break
                desired_step = radius * shell
                reasons: list[str] = []
                integer_axis_special = (
                    family == "schema_axis" and family_index in integer_indices
                )
                if integer_axis_special:
                    integer_index = family_index
                    anchor = int(np.rint(lower[integer_index] + center[integer_index] * scale[integer_index]))
                    plus_value = anchor + 1
                    minus_value = anchor - 1
                    if plus_value > upper[integer_index] or minus_value < lower[integer_index]:
                        reasons.append("integer_axis_two_sided_step_illegal")
                    plus_raw = center.copy()
                    minus_raw = center.copy()
                    plus_raw[integer_index] = (plus_value - lower[integer_index]) / scale[integer_index]
                    minus_raw[integer_index] = (minus_value - lower[integer_index]) / scale[integer_index]
                    actual_step = None
                    step_ratio = None
                else:
                    common_limit = _common_two_sided_limit(center, direction)
                    actual_step = min(desired_step, eta * common_limit)
                    step_ratio = actual_step / desired_step if desired_step > 0.0 else 0.0
                    if step_ratio < minimum_ratio:
                        reasons.append("two_sided_safe_step_ratio_below_minimum")
                    plus_raw = center + actual_step * direction
                    minus_raw = center - actual_step * direction
                if not (
                    np.all((plus_raw >= 0.0) & (plus_raw <= 1.0))
                    and np.all((minus_raw >= 0.0) & (minus_raw <= 1.0))
                ):
                    reasons.append("schema_bounds_after_scaling")
                plus_geometry = _geometry_from_normalized(
                    plus_raw, names, lower, scale, integer_names
                )
                minus_geometry = _geometry_from_normalized(
                    minus_raw, names, lower, scale, integer_names
                )
                plus_hash = sha256(_canonical_bytes(plus_geometry)).hexdigest()
                minus_hash = sha256(_canonical_bytes(minus_geometry)).hexdigest()
                if plus_hash == minus_hash:
                    reasons.append("pair_design_hashes_identical")
                if plus_hash in development_hashes or minus_hash in development_hashes:
                    reasons.append("development_design_hash_duplicate")
                if plus_hash in selected_hashes or minus_hash in selected_hashes:
                    reasons.append("previously_selected_v2_design_hash_duplicate")
                accepted = not reasons
                screening_id = sha256(
                    _canonical_bytes(
                        {
                            "root_seed_id": root_id,
                            "shell": shell,
                            "pool": pool,
                            "pool_index": pool_index,
                            "plus_hash": plus_hash,
                            "minus_hash": minus_hash,
                        }
                    )
                ).hexdigest()
                screen_row = {
                    "screening_id": screening_id,
                    "root_seed_id": root_id,
                    "source_root_segments": root_row["segments"],
                    "shell_fraction": shell,
                    "direction_pool": pool,
                    "pool_index": pool_index,
                    "direction_family": family,
                    "direction_index": family_index,
                    "desired_step_normalized_l2": desired_step,
                    "actual_step_normalized_l2": actual_step,
                    "actual_to_desired_step_ratio": step_ratio,
                    "integer_axis_special": integer_axis_special,
                    "screening_status": "PAIR_SELECTED" if accepted else "PAIR_REJECTED",
                    "rejection_reasons": reasons,
                    "plus_design_hash": plus_hash,
                    "minus_design_hash": minus_hash,
                    "label_status": "NOT_EVALUATED_PRELABEL_ONLY",
                }
                screening.append(screen_row)
                if accepted:
                    selected_hashes.update((plus_hash, minus_hash))
                    selected_pair = {
                        **screen_row,
                        "plus_raw": plus_raw,
                        "minus_raw": minus_raw,
                        "plus_geometry": plus_geometry,
                        "minus_geometry": minus_geometry,
                    }
                    root_selected.append(selected_pair)
                    shell_selected += 1
            if shell_selected < 24:
                screening.append(
                    {
                        "screening_id": sha256(
                            _canonical_bytes(
                                {
                                    "root_seed_id": root_id,
                                    "shell": shell,
                                    "failure": "reserve_exhausted",
                                }
                            )
                        ).hexdigest(),
                        "root_seed_id": root_id,
                        "source_root_segments": root_row["segments"],
                        "shell_fraction": shell,
                        "direction_pool": "none",
                        "pool_index": None,
                        "direction_family": "none",
                        "direction_index": None,
                        "desired_step_normalized_l2": radius * shell,
                        "actual_step_normalized_l2": None,
                        "actual_to_desired_step_ratio": None,
                        "integer_axis_special": False,
                        "screening_status": "SHELL_SELECTION_INCOMPLETE",
                        "rejection_reasons": ["reserve_directions_exhausted"],
                        "plus_design_hash": None,
                        "minus_design_hash": None,
                        "label_status": "NOT_EVALUATED_PRELABEL_ONLY",
                    }
                )
        selected_pairs.extend(root_selected)

    final_ledger: list[dict[str, Any]] = []
    selected_by_root: dict[str, list[dict[str, Any]]] = {}
    for pair in selected_pairs:
        selected_by_root.setdefault(str(pair["root_seed_id"]), []).append(pair)
    for root_row in registry:
        root_id = str(root_row["root_seed_id"])
        root_pairs = selected_by_root.get(root_id, [])
        for pair_index, pair in enumerate(root_pairs):
            for sign, raw_key, geometry_key, hash_key in (
                (1, "plus_raw", "plus_geometry", "plus_design_hash"),
                (-1, "minus_raw", "minus_geometry", "minus_design_hash"),
            ):
                raw = pair[raw_key]
                proposal_identity = {
                    "protocol_version": PROTOCOL_VERSION,
                    "screening_id": pair["screening_id"],
                    "root_seed_id": root_id,
                    "selected_pair_index": pair_index,
                    "sign": sign,
                    "raw_normalized_geometry19": raw.tolist(),
                }
                proposal_hash = sha256(_canonical_bytes(proposal_identity)).hexdigest()
                final_ledger.append(
                    {
                        "attempt_id": f"s3v2:{root_id[:12]}:{pair_index:02d}:{'p' if sign > 0 else 'm'}:{proposal_hash[:12]}",
                        "proposal_hash": proposal_hash,
                        "screening_id": pair["screening_id"],
                        "root_seed_id": root_id,
                        "source_root_segments": pair["source_root_segments"],
                        "selected_pair_index": pair_index,
                        "direction_pool": pair["direction_pool"],
                        "pool_index": pair["pool_index"],
                        "direction_family": pair["direction_family"],
                        "direction_index": pair["direction_index"],
                        "radius_fraction": pair["shell_fraction"],
                        "desired_step_normalized_l2": pair[
                            "desired_step_normalized_l2"
                        ],
                        "actual_step_normalized_l2": pair[
                            "actual_step_normalized_l2"
                        ],
                        "actual_to_desired_step_ratio": pair[
                            "actual_to_desired_step_ratio"
                        ],
                        "integer_axis_special": pair["integer_axis_special"],
                        "sign": sign,
                        "raw_normalized_geometry19": {
                            name: float(raw[index]) for index, name in enumerate(names)
                        },
                        "geometry19": pair[geometry_key],
                        "design_hash": pair[hash_key],
                        "coordinate_status": "COORDINATE_ELIGIBLE",
                        "rejection_reasons": [],
                        "label_status": "NOT_EVALUATED_PRELABEL_ONLY",
                    }
                )

    pair_counts = Counter(str(row["root_seed_id"]) for row in selected_pairs)
    shell_counts = Counter(
        (str(row["root_seed_id"]), str(row["shell_fraction"])) for row in selected_pairs
    )
    final_hashes = [str(row["design_hash"]) for row in final_ledger]
    forbidden_fields = {
        "performance",
        "physics",
        "prediction",
        "rho",
        "solver",
        "map_states",
    }
    exact_roots = set(pair_counts) == {str(row["root_seed_id"]) for row in registry}
    exact_pairs = exact_roots and set(pair_counts.values()) == {48}
    exact_shells = exact_roots and all(
        shell_counts[(root_id, "0.5")] == 24
        and shell_counts[(root_id, "0.9")] == 24
        for root_id in pair_counts
    )
    exact_rows = len(final_ledger) == 3648
    unique_hashes = len(final_hashes) == len(set(final_hashes))
    no_development_overlap = not (set(final_hashes) & development_hashes)
    coordinates_inside = all(
        all(0.0 <= float(value) <= 1.0 for value in row["raw_normalized_geometry19"].values())
        for row in final_ledger
    )
    zero_forbidden = all(not (forbidden_fields & set(row)) for row in final_ledger)
    gate_pass = all(
        (
            exact_roots,
            exact_pairs,
            exact_shells,
            exact_rows,
            unique_hashes,
            no_development_overlap,
            coordinates_inside,
            zero_forbidden,
        )
    )
    status = PASS_STATUS if gate_pass else FAIL_STATUS

    runtime_root = (
        output_root.resolve()
        if output_root is not None
        else (repo_root / str(config["artifacts"]["runtime_root"])).resolve()
    )
    screening_path = runtime_root / str(config["artifacts"]["screening_ledger"])
    final_path = runtime_root / str(config["artifacts"]["final_prelabel_ledger"])
    direction_path = runtime_root / str(config["artifacts"]["direction_matrix"])
    summary_path = runtime_root / str(config["artifacts"]["summary"])
    report_path = runtime_root / str(config["artifacts"]["report"])
    _write_jsonl(screening_path, screening)
    _write_jsonl(final_path, final_ledger)
    _write_npz(direction_path, **directions)
    rejection_reasons = Counter(
        reason for row in screening for reason in row["rejection_reasons"]
    )
    reserve_selected = sum(
        row["screening_status"] == "PAIR_SELECTED"
        and row["direction_pool"] == "reserve"
        for row in screening
    )
    implementation_path = Path(__file__).resolve()
    summary: dict[str, Any] = {
        "status": status,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_path": _artifact_path(config_path, repo_root),
        "protocol_sha256": sha256_file(config_path),
        "implementation_path": _artifact_path(implementation_path, repo_root),
        "implementation_sha256": sha256_file(implementation_path),
        "input_hash_audit": input_hash_audit,
        "all_input_hashes_match": True,
        "v1_failure_preserved": True,
        "adaptive_after_v1_failure": True,
        "screening_pairs_considered": len(screening),
        "screening_pairs_rejected": sum(
            row["screening_status"] == "PAIR_REJECTED" for row in screening
        ),
        "screening_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "primary_pairs_selected": sum(
            row["screening_status"] == "PAIR_SELECTED"
            and row["direction_pool"] == "primary"
            for row in screening
        ),
        "reserve_pairs_selected": reserve_selected,
        "selected_pairs": len(selected_pairs),
        "selected_pairs_per_root": dict(sorted(pair_counts.items())),
        "selected_pairs_per_root_shell": {
            f"{root_id}:{shell}": count
            for (root_id, shell), count in sorted(shell_counts.items())
        },
        "minimum_selected_pairs_per_root": min(pair_counts.values())
        if pair_counts
        else 0,
        "final_prelabel_rows": len(final_ledger),
        "final_design_hashes_unique": unique_hashes,
        "final_development_overlap": not no_development_overlap,
        "all_final_coordinates_inside_schema": coordinates_inside,
        "zero_performance_physics_or_prediction_fields": zero_forbidden,
        "model_manifest_metadata_read": False,
        "model_files_opened": False,
        "model_predictions_executed": False,
        "physics_evaluations_executed": False,
        "performance_labels_written": False,
        "confirmation_hash_checked": True,
        "confirmation_content_read": False,
        "physics_labeling_authorized": gate_pass,
        "inverse_design_authorized": False,
        "original_local_chart_p4_authorized": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "artifacts": {
            "screening_ledger": _artifact_path(screening_path, repo_root),
            "screening_ledger_sha256": sha256_file(screening_path),
            "final_prelabel_ledger": _artifact_path(final_path, repo_root),
            "final_prelabel_ledger_sha256": sha256_file(final_path),
            "direction_matrix": _artifact_path(direction_path, repo_root),
            "direction_matrix_sha256": sha256_file(direction_path),
        },
        "next_authorized_stage": "S3_v2_evaluate_frozen_coordinates_with_unified_V2_2"
        if gate_pass
        else "stop_and_review_S3_v2_coordinate_failure",
    }
    _write_text(report_path, _report(summary))
    summary["artifacts"]["report"] = _artifact_path(report_path, repo_root)
    summary["artifacts"]["report_sha256"] = sha256_file(report_path)
    _write_json(summary_path, summary)
    return summary
