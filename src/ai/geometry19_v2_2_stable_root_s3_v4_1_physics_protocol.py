"""Lock the V2.2 development physics-acquisition protocol for S3 v4.1."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.ai.geometry19_surrogate import sha256_file


PROTOCOL_VERSION = "geometry19_v2_2_stable_root_s3_v4_1_physics_acquisition_v1"
LOCK_PASS = "STABLE_ROOT_S3_V4_1_PHYSICS_PROTOCOL_LOCK_PASS"


class Geometry19StableRootS3V41PhysicsProtocolError(ValueError):
    """Raised when the frozen v4.1 acquisition contract cannot be locked."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Geometry19StableRootS3V41PhysicsProtocolError(message)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def validate_protocol(config: Mapping[str, Any]) -> None:
    _require(config.get("protocol_version") == PROTOCOL_VERSION, "Protocol version mismatch")
    _require(config.get("stage") == "S3_v4_1_development_physics_acquisition_protocol_lock", "Stage mismatch")
    _require(config.get("scope_definition") == "post_selected_33root_medoid_covariance_development_only", "Scope changed")
    _require(config.get("formal_use_allowed") is False, "Formal use must remain forbidden")
    _require(config.get("deployment_allowed") is False, "Deployment must remain forbidden")
    labeling = config.get("physics_labeling", {})
    _require(labeling.get("pipeline_profile") == "v2_2_development", "Physics profile changed")
    _require(labeling.get("expected_coordinates") == 3168, "Coordinate count changed")
    _require(labeling.get("expected_roots") == 33, "Root count changed")
    _require(labeling.get("expected_rows_per_root") == 96, "Rows per root changed")
    _require(labeling.get("shard_size") == 96, "Shard size changed")
    _require(labeling.get("performance_adaptive_top_up") is False, "Adaptive top-up cannot be allowed")
    _require(labeling.get("coordinate_deletion_after_labeling") is False, "Coordinate deletion cannot be allowed")
    gates = config.get("acquisition_gates", {})
    _require(gates.get("exact_roots") == 33, "Root gate changed")
    _require(gates.get("exact_evaluations") == 3168, "Evaluation gate changed")
    _require(gates.get("exact_rows_per_root") == 96, "Per-root row gate changed")
    _require(gates.get("minimum_evaluation_eligible_per_root") == 32, "Per-root eligible gate changed")
    _require(gates.get("minimum_evaluation_eligible_fraction_overall") == 0.80, "Overall gate changed")
    _require(gates.get("minimum_evaluation_eligible_fraction_each_source_target_segment") == 0.80, "Segment gate changed")
    _require(config["inputs"]["sealed_confirmation"]["access"] == "sha256_only_no_parse", "Confirmation access changed")
    _require(config["inputs"]["s2_model_manifest"]["access_before_acquisition_pass"] == "sha256_only_no_parse", "Model access changed")
    policy = config.get("post_acquisition_policy", {})
    _require(policy.get("model_scoring_executed_in_this_stage") is False, "Model scoring cannot run in acquisition stage")
    _require(policy.get("coordinate_or_gate_adaptation_after_results") == "forbidden", "Post-result adaptation cannot be allowed")
    _require(policy.get("inverse_design_authorized") is False, "Inverse design cannot be authorized")


def _report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Stable-Root S3 v4.1 V2.2 Physics Protocol Lock",
            "",
            f"- Status: `{summary['status']}`",
            f"- Coordinates locked: {summary['coordinate_rows']}",
            f"- Roots locked: {summary['root_count']}",
            f"- Rows per root: {summary['rows_per_root']}",
            f"- Future shards: {summary['future_shard_count']}",
            "- Physics evaluations executed: `false`",
            "- Model manifest/model files opened: `false`",
            "- Confirmation content parsed: `false`",
            "- Model scoring authorized now: `false`",
            "- Inverse design authorized: `false`",
            "",
            "This is an adaptive development acquisition protocol after the observed v3 physics and v4 coordinate failures. The coordinates are not a fresh confirmation set. A failed acquisition gate must be sealed and stopped without coordinate, threshold, or performance-adaptive changes.",
            "",
        ]
    )


def lock_s3_v4_1_physics_protocol(
    repo_root: Path,
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Verify and freeze acquisition inputs without running V2.2 or models."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_protocol(config)
    protected_roles = {"s2_model_manifest", "sealed_confirmation"}
    resolved: dict[str, Path] = {}
    input_audit: dict[str, Any] = {}
    for role, item in config["inputs"].items():
        path = (repo_root / str(item["path"])).resolve()
        _require(path.is_file(), f"Missing protocol input: {role}")
        actual = sha256_file(path)
        _require(actual == str(item["sha256"]), f"Protocol input hash mismatch: {role}")
        resolved[role] = path
        input_audit[role] = {
            "path": str(item["path"]),
            "expected_sha256": str(item["sha256"]),
            "actual_sha256": actual,
            "match": True,
            "content_parsed": role not in protected_roles,
        }

    v4_summary = json.loads(resolved["v4_failure_summary"].read_text(encoding="utf-8"))
    _require(v4_summary.get("status") == config["inputs"]["v4_failure_summary"]["required_status"], "v4 failure evidence changed")
    v41_summary = json.loads(resolved["v4_1_coordinate_summary"].read_text(encoding="utf-8"))
    _require(v41_summary.get("status") == config["inputs"]["v4_1_coordinate_summary"]["required_status"], "v4.1 coordinate gate did not pass")
    _require(v41_summary.get("future_physics_protocol_authorized") is True, "v4.1 did not authorize a physics protocol")
    _require(v41_summary.get("physics_evaluations_executed") is False, "v4.1 already executed physics")
    _require(v41_summary.get("model_predictions_executed") is False, "v4.1 already executed predictions")

    coordinates = _read_jsonl(resolved["coordinate_ledger"])
    counts = Counter(str(row["root_seed_id"]) for row in coordinates)
    hashes = [str(row["design_hash"]) for row in coordinates]
    sources = Counter(str(row["combined_source_protocol"]) for row in coordinates)
    _require(len(coordinates) == 3168, "Coordinate ledger row count mismatch")
    _require(len(counts) == 33 and set(counts.values()) == {96}, "Coordinate root coverage mismatch")
    _require(len(set(hashes)) == len(hashes), "Coordinate design hashes are not unique")
    _require(sources == {"v4_medoid_covariance": 2784, "v4_1_boundary_completion": 384}, "Coordinate source composition changed")
    forbidden = {"performance", "physics", "prediction", "rho", "solver", "map_states"}
    _require(all(forbidden.isdisjoint(row) for row in coordinates), "Prelabel ledger contains forbidden result fields")
    _require(all(row.get("label_status") == "NOT_EVALUATED_DEVELOPMENT_COORDINATE_ONLY" for row in coordinates), "Coordinate label status changed")

    runtime_root = output_root.resolve() if output_root is not None else (repo_root / str(config["artifacts"]["runtime_root"])).resolve()
    future_keys = (
        "future_physics_shards",
        "future_sealed_physics_ledger",
        "future_physics_summary",
        "future_physics_report",
        "future_prediction_ledger",
    )
    future_paths = {key: runtime_root / str(config["artifacts"][key]) for key in future_keys}
    _require(not any(path.exists() for path in future_paths.values()), "A future acquisition or prediction artifact already exists")
    lock_path = runtime_root / str(config["artifacts"]["protocol_lock"])
    report_path = runtime_root / str(config["artifacts"]["protocol_lock_report"])
    summary: dict[str, Any] = {
        "status": LOCK_PASS,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope_definition": config["scope_definition"],
        "adaptive_development_after_observed_failures": True,
        "not_fresh_confirmation": True,
        "protocol_path": _relative(config_path, repo_root),
        "protocol_sha256": sha256_file(config_path),
        "implementation_path": _relative(Path(__file__), repo_root),
        "implementation_sha256": sha256_file(Path(__file__)),
        "input_hash_audit": input_audit,
        "all_input_hashes_match": True,
        "coordinate_rows": len(coordinates),
        "root_count": len(counts),
        "rows_per_root": min(counts.values()),
        "coordinate_source_rows": dict(sorted(sources.items())),
        "coordinate_design_hashes_unique": True,
        "coordinate_result_fields_absent": True,
        "acquisition_gates": config["acquisition_gates"],
        "future_shard_count": len(coordinates) // int(config["physics_labeling"]["shard_size"]),
        "future_artifacts_absent_at_lock": {key: not path.exists() for key, path in future_paths.items()},
        "physics_evaluations_executed": False,
        "performance_labels_written": False,
        "model_manifest_content_read": False,
        "model_files_opened": False,
        "model_predictions_executed": False,
        "confirmation_content_read": False,
        "physics_acquisition_authorized": True,
        "model_scoring_authorized_now": False,
        "inverse_design_authorized": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "next_authorized_stage": "execute_v4_1_unified_V2_2_development_physics_acquisition",
    }
    _write_text(report_path, _report(summary))
    summary["report_path"] = _relative(report_path, repo_root)
    summary["report_sha256"] = sha256_file(report_path)
    _write_json(lock_path, summary)
    return summary
