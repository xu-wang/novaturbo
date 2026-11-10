"""Generate and seal S3 coordinates before any physics label or prediction."""

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


PROTOCOL_VERSION = "geometry19_v2_2_stable_root_s3_prelabel_v1"
PASS_STATUS = "STABLE_ROOT_S3_PRELABEL_COORDINATE_LOCK_PASS"
FAIL_STATUS = "STABLE_ROOT_S3_PRELABEL_COORDINATE_LOCK_FAIL"


class Geometry19StableRootS3PrelabelError(ValueError):
    """Raised when S3 prelabel isolation or coordinate contracts fail."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Geometry19StableRootS3PrelabelError(message)


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
    _require(config.get("protocol_version") == PROTOCOL_VERSION, "S3 protocol version mismatch")
    _require(config.get("stage") == "S3_generate_and_seal_prelabel_coordinates", "S3 stage mismatch")
    _require(config.get("formal_use_allowed") is False, "Formal use must remain forbidden")
    _require(config.get("deployment_allowed") is False, "Deployment must remain forbidden")
    directions = config.get("directions", {})
    _require(directions.get("dimension") == 19, "Direction dimension changed")
    _require(directions.get("axis_directions") == 19, "Axis direction count changed")
    _require(directions.get("sobol_directions") == 29, "Sobol direction count changed")
    _require(directions.get("sobol_seed") == 20260915, "Sobol seed changed")
    ledger = config.get("ledger", {})
    _require(ledger.get("roots") == 38, "Root count changed")
    _require(ledger.get("attempts_per_root") == 96, "Attempts per root changed")
    _require(ledger.get("total_attempts") == 3648, "Total attempt count changed")
    _require(ledger.get("performance_fields_allowed") is False, "Performance fields cannot be allowed")
    _require(ledger.get("physics_evaluation_fields_allowed") is False, "Physics fields cannot be allowed")
    gate = config.get("prelabel_gate", {})
    _require(gate.get("minimum_coordinate_eligible_per_root") == 80, "Minimum eligible coordinates changed")
    _require(gate.get("require_model_files_unopened") is True, "Model isolation gate missing")
    authorization = config.get("authorization", {})
    _require(authorization.get("model_prediction_authorized") is False, "Model prediction cannot be authorized")
    _require(authorization.get("adaptive_coordinate_revision_authorized") is False, "Adaptive revision cannot be authorized")
    _require(authorization.get("inverse_design_authorized") is False, "Inverse design cannot be authorized")
    confirmation = config.get("inputs", {}).get("sealed_confirmation", {})
    _require(confirmation.get("access") == "sha256_only_no_parse", "Confirmation access changed")


def build_direction_matrix(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Create the frozen 19 axes plus 29 normalized Sobol directions."""

    axis = np.eye(19, dtype=np.float64)
    engine = qmc.Sobol(d=19, scramble=True, seed=int(config["directions"]["sobol_seed"]))
    sobol = 2.0 * engine.random_base2(m=5)[:29] - 1.0
    norms = np.linalg.norm(sobol, axis=1)
    _require(np.all(norms > 0.0), "Sobol direction contains a zero vector")
    sobol = sobol / norms[:, None]
    directions = np.vstack([axis, sobol])
    family = np.asarray(["schema_axis"] * 19 + ["sobol_mixed"] * 29)
    family_index = np.asarray(list(range(19)) + list(range(29)), dtype=np.int64)
    radius_fraction = np.asarray(
        [0.50 if index <= 9 else 0.90 for index in range(19)]
        + [0.50 if index <= 13 else 0.90 for index in range(29)],
        dtype=np.float64,
    )
    _require(int(np.sum(radius_fraction == 0.50)) == 24, "0.50 shell allocation mismatch")
    _require(int(np.sum(radius_fraction == 0.90)) == 24, "0.90 shell allocation mismatch")
    return {
        "directions": directions,
        "direction_family": family,
        "direction_family_index": family_index,
        "radius_fraction": radius_fraction,
    }


def _report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Stable-Root S3 Prelabel Coordinate Lock",
            "",
            f"- Status: `{summary['status']}`",
            f"- Attempt rows: {summary['attempt_rows']}",
            f"- Coordinate-eligible rows: {summary['coordinate_eligible_rows']}",
            f"- Rejected rows: {summary['coordinate_rejected_rows']}",
            f"- Minimum eligible rows per root: {summary['minimum_coordinate_eligible_per_root_observed']}",
            f"- Bounds rejects: {summary['rejection_reasons'].get('schema_bounds', 0)}",
            f"- Development duplicates: {summary['rejection_reasons'].get('development_design_hash_duplicate', 0)}",
            f"- Within-S3 duplicates: {summary['rejection_reasons'].get('within_S3_design_hash_duplicate', 0)}",
            f"- Physics labeling authorized: `{str(summary['physics_labeling_authorized']).lower()}`",
            "- Physics evaluations executed: `false`",
            "- Model files opened/predictions executed: `false`",
            "- Performance fields written: `false`",
            "- Confirmation content parsed: `false`",
            "",
            "Coordinate eligibility only establishes a frozen, non-duplicated, schema-bounded proposal set. It does not imply V2.2 convergence, physics-domain validity, target-segment membership, or surrogate accuracy.",
            "",
        ]
    )


def generate_s3_prelabel_coordinates(
    repo_root: Path,
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Generate all frozen coordinates without opening model files or evaluating physics."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_protocol(config)
    resolved: dict[str, Path] = {}
    input_hash_audit: dict[str, Any] = {}
    for role, item in config["inputs"].items():
        path = (repo_root / str(item["path"])).resolve()
        _require(path.is_file(), f"Missing S3 input: {role}")
        actual = sha256_file(path)
        _require(actual == str(item["sha256"]), f"S3 input hash mismatch: {role}")
        resolved[role] = path
        input_hash_audit[role] = {
            "path": str(item["path"]),
            "expected_sha256": str(item["sha256"]),
            "actual_sha256": actual,
            "match": True,
            "content_parsed": role != "sealed_confirmation",
        }
    authorization = json.loads(
        resolved["s2_regression_authorization"].read_text(encoding="utf-8")
    )
    _require(
        authorization.get("status")
        == config["inputs"]["s2_regression_authorization"]["required_status"],
        "S2 regression authorization status mismatch",
    )
    _require(
        authorization.get("s3_coordinate_generation_authorized") is True,
        "S2 did not authorize S3 coordinate generation",
    )
    _require(authorization.get("confirmation_content_read") is False, "S2 confirmation isolation failed")
    manifest = json.loads(resolved["s2_model_manifest"].read_text(encoding="utf-8"))
    _require(
        manifest.get("status") == config["inputs"]["s2_model_manifest"]["required_status"],
        "S2 model manifest status mismatch",
    )
    _require(manifest.get("exact_reload_pass") is True, "S2 model reload gate failed")

    schema = yaml.safe_load(resolved["design_schema"].read_text(encoding="utf-8"))
    variables = sorted(schema["design_variables"], key=lambda item: int(item["index"]))
    names = [str(item["name"]) for item in variables]
    lower = np.asarray([float(item["bounds"][0]) for item in variables])
    upper = np.asarray([float(item["bounds"][1]) for item in variables])
    scale = upper - lower
    integer_names = set(config["coordinate_conversion"]["integer_variable_names"])
    _require(
        integer_names == {str(item["name"]) for item in variables if str(item["dtype"]) == "int64"},
        "Frozen integer variable set differs from schema",
    )
    registry = sorted(
        _read_jsonl(resolved["stable_root_registry"]),
        key=lambda row: str(row["root_seed_id"]),
    )
    _require(len(registry) == 38, "Stable registry count mismatch")
    development_hashes = {
        str(json.loads(line)["design_hash"])
        for line in resolved["development_dataset"].read_text(encoding="utf-8").splitlines()
    }
    _require(len(development_hashes) == 4529, "Development design hash count mismatch")
    direction_bundle = build_direction_matrix(config)

    ledger: list[dict[str, Any]] = []
    accepted_hashes: set[str] = set()
    for root_row in registry:
        root_id = str(root_row["root_seed_id"])
        center = np.asarray(root_row["normalized_design_center"], dtype=np.float64)
        radius = float(root_row["support_radius_p95_normalized_l2"])
        _require(center.shape == (19,) and radius > 0.0, f"Invalid registry geometry: {root_id}")
        attempt_index = 0
        for pair_index in range(48):
            direction = direction_bundle["directions"][pair_index]
            family = str(direction_bundle["direction_family"][pair_index])
            family_index = int(direction_bundle["direction_family_index"][pair_index])
            radius_fraction = float(direction_bundle["radius_fraction"][pair_index])
            for sign in (1, -1):
                raw = center + sign * radius_fraction * radius * direction
                identity = {
                    "protocol_version": PROTOCOL_VERSION,
                    "root_seed_id": root_id,
                    "pair_index": pair_index,
                    "direction_family": family,
                    "direction_index": family_index,
                    "radius_fraction": radius_fraction,
                    "sign": sign,
                    "raw_normalized_geometry19": raw.tolist(),
                }
                proposal_hash = sha256(_canonical_bytes(identity)).hexdigest()
                reasons: list[str] = []
                geometry: dict[str, Any] | None = None
                design_hash: str | None = None
                if not np.all((raw >= 0.0) & (raw <= 1.0)):
                    reasons.append("schema_bounds")
                else:
                    physical = lower + raw * scale
                    geometry = {}
                    for index, name in enumerate(names):
                        geometry[name] = (
                            int(np.rint(physical[index]))
                            if name in integer_names
                            else float(physical[index])
                        )
                    design_hash = sha256(_canonical_bytes(geometry)).hexdigest()
                    if design_hash in development_hashes:
                        reasons.append("development_design_hash_duplicate")
                    if design_hash in accepted_hashes:
                        reasons.append("within_S3_design_hash_duplicate")
                    if not reasons:
                        accepted_hashes.add(design_hash)
                status = "COORDINATE_ELIGIBLE" if not reasons else "COORDINATE_REJECTED"
                ledger.append(
                    {
                        "attempt_index": attempt_index,
                        "attempt_id": f"s3pre:{root_id[:12]}:{pair_index:02d}:{'p' if sign > 0 else 'm'}:{proposal_hash[:12]}",
                        "proposal_hash": proposal_hash,
                        "root_seed_id": root_id,
                        "source_root_segments": root_row["segments"],
                        "pair_index": pair_index,
                        "direction_family": family,
                        "direction_index": family_index,
                        "radius_fraction": radius_fraction,
                        "support_radius_normalized_l2": radius,
                        "sign": sign,
                        "raw_normalized_geometry19": {
                            name: float(raw[index]) for index, name in enumerate(names)
                        },
                        "coordinate_status": status,
                        "rejection_reasons": reasons,
                        "geometry19": geometry,
                        "design_hash": design_hash,
                        "label_status": "NOT_EVALUATED_PRELABEL_ONLY",
                    }
                )
                attempt_index += 1
        _require(attempt_index == 96, f"Attempt count mismatch: {root_id}")

    by_root_attempts = Counter(str(row["root_seed_id"]) for row in ledger)
    eligible = [row for row in ledger if row["coordinate_status"] == "COORDINATE_ELIGIBLE"]
    by_root_eligible = Counter(str(row["root_seed_id"]) for row in eligible)
    rejection_reasons = Counter(
        reason for row in ledger for reason in row["rejection_reasons"]
    )
    eligible_hashes = [str(row["design_hash"]) for row in eligible]
    required_fields = set(config["ledger"]["required_fields"])
    forbidden_fields = {"performance", "physics", "prediction", "rho", "solver", "map_states"}
    exact_rows = len(ledger) == int(config["prelabel_gate"]["exact_attempt_rows"])
    exact_per_root = set(by_root_attempts.values()) == {
        int(config["prelabel_gate"]["exact_attempts_per_root"])
    }
    minimum_eligible = min(by_root_eligible.get(str(row["root_seed_id"]), 0) for row in registry)
    fields_complete = all(required_fields <= set(row) for row in ledger)
    zero_forbidden = all(not (forbidden_fields & set(row)) for row in ledger)
    unique_eligible = len(eligible_hashes) == len(set(eligible_hashes))
    no_development_overlap = not (set(eligible_hashes) & development_hashes)
    gate_pass = all(
        (
            exact_rows,
            exact_per_root,
            minimum_eligible
            >= int(config["prelabel_gate"]["minimum_coordinate_eligible_per_root"]),
            fields_complete,
            zero_forbidden,
            unique_eligible,
            no_development_overlap,
        )
    )
    status = PASS_STATUS if gate_pass else FAIL_STATUS

    runtime_root = (
        output_root.resolve()
        if output_root is not None
        else (repo_root / str(config["artifacts"]["runtime_root"])).resolve()
    )
    ledger_path = runtime_root / str(config["artifacts"]["prelabel_ledger"])
    direction_path = runtime_root / str(config["artifacts"]["direction_matrix"])
    summary_path = runtime_root / str(config["artifacts"]["prelabel_summary"])
    report_path = runtime_root / str(config["artifacts"]["prelabel_report"])
    _write_jsonl(ledger_path, ledger)
    _write_npz(direction_path, **direction_bundle)
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
        "attempt_rows": len(ledger),
        "roots": len(registry),
        "attempts_per_root": dict(sorted(by_root_attempts.items())),
        "coordinate_eligible_rows": len(eligible),
        "coordinate_rejected_rows": len(ledger) - len(eligible),
        "coordinate_eligible_per_root": dict(sorted(by_root_eligible.items())),
        "minimum_coordinate_eligible_per_root_observed": minimum_eligible,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "eligible_by_direction_family": dict(
            sorted(Counter(str(row["direction_family"]) for row in eligible).items())
        ),
        "eligible_by_radius_fraction": dict(
            sorted(Counter(str(row["radius_fraction"]) for row in eligible).items())
        ),
        "exact_attempt_rows_pass": exact_rows,
        "exact_attempts_per_root_pass": exact_per_root,
        "required_fields_pass": fields_complete,
        "unique_eligible_design_hashes_pass": unique_eligible,
        "no_eligible_development_overlap_pass": no_development_overlap,
        "zero_performance_or_physics_fields_pass": zero_forbidden,
        "model_manifest_metadata_read": True,
        "model_files_opened": False,
        "model_predictions_executed": False,
        "physics_evaluations_executed": False,
        "performance_labels_written": False,
        "confirmation_hash_checked": True,
        "confirmation_content_read": False,
        "physics_labeling_authorized": gate_pass,
        "adaptive_coordinate_revision_authorized": False,
        "inverse_design_authorized": False,
        "original_local_chart_p4_authorized": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "artifacts": {
            "prelabel_ledger": _artifact_path(ledger_path, repo_root),
            "prelabel_ledger_sha256": sha256_file(ledger_path),
            "direction_matrix": _artifact_path(direction_path, repo_root),
            "direction_matrix_sha256": sha256_file(direction_path),
        },
        "next_authorized_stage": "S3_evaluate_frozen_coordinates_with_unified_V2_2"
        if gate_pass
        else "review_prelabel_coordinate_failures_without_adaptive_revision",
    }
    _write_text(report_path, _report(summary))
    summary["artifacts"]["prelabel_report"] = _artifact_path(report_path, repo_root)
    summary["artifacts"]["prelabel_report_sha256"] = sha256_file(report_path)
    _write_json(summary_path, summary)
    return summary
