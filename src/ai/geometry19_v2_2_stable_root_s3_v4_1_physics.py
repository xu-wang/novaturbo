"""Evaluate and seal unified V2.2 labels for S3 v4.1 coordinates."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_stable_root_s3_v3_physics import (
    _canonical_bytes,
    _eligible,
    _read_jsonl,
    _relative,
    _write_json,
    _write_jsonl,
    _write_text,
)
from src.ai.geometry19_v2_2_stable_root_s3_v4_1_physics_protocol import (
    LOCK_PASS,
    PROTOCOL_VERSION,
    validate_protocol,
)
from src.physics.geometry19_v2_2_dataset import _evaluate_row


PASS_STATUS = "STABLE_ROOT_S3_V4_1_PHYSICS_ACQUISITION_PASS"
FAIL_STATUS = "STABLE_ROOT_S3_V4_1_PHYSICS_ACQUISITION_FAIL"


class Geometry19StableRootS3V41PhysicsError(ValueError):
    """Raised when the frozen v4.1 physics acquisition contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Geometry19StableRootS3V41PhysicsError(message)


def _report(summary: Mapping[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# Stable-Root S3 v4.1 V2.2 Development Physics Acquisition",
        "",
        f"- Status: `{summary['status']}`",
        f"- Evaluated coordinates: {counts['evaluated']}",
        f"- Solver converged: {counts['solver_converged']}",
        f"- Numerically valid: {counts['numerically_valid']}",
        f"- Physics-domain valid: {counts['physics_domain_valid']}",
        f"- Inside target domain: {counts['inside_target_domain']}",
        f"- Evaluation eligible: {counts['evaluation_eligible']}",
        f"- Overall eligible fraction: {summary['evaluation_eligible_fraction_overall']:.6f}",
        f"- Minimum eligible rows per root: {summary['minimum_evaluation_eligible_per_root_observed']}",
        f"- Separate scoring protocol authorized: `{str(summary['future_scoring_protocol_authorized']).lower()}`",
        "- Model scoring executed/authorized now: `false`",
        "- Model manifest/model files opened: `false`",
        "- Confirmation content parsed: `false`",
        "",
        "These results are adaptive development evidence after the observed v3 physics and v4 coordinate failures. They are not a fresh confirmation result. All 3,168 rows were sealed before applying the acquisition gate.",
        "",
    ]
    return "\n".join(lines)


def run_s3_v4_1_physics_acquisition(
    repo_root: Path,
    config_path: Path,
    *,
    output_root: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Evaluate all locked coordinates without parsing or running S2 models."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_protocol(config)
    protected_roles = {"s2_model_manifest", "sealed_confirmation"}
    resolved: dict[str, Path] = {}
    input_audit: dict[str, Any] = {}
    for role, item in config["inputs"].items():
        path = (repo_root / str(item["path"])).resolve()
        _require(path.is_file(), f"Missing acquisition input: {role}")
        actual = sha256_file(path)
        _require(actual == str(item["sha256"]), f"Acquisition input hash mismatch: {role}")
        resolved[role] = path
        input_audit[role] = {
            "path": str(item["path"]),
            "expected_sha256": str(item["sha256"]),
            "actual_sha256": actual,
            "match": True,
            "content_parsed_during_acquisition": role not in protected_roles,
        }

    runtime_root = (repo_root / str(config["artifacts"]["runtime_root"])).resolve()
    lock_path = runtime_root / str(config["artifacts"]["protocol_lock"])
    _require(lock_path.is_file(), "Physics protocol lock artifact is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    _require(lock.get("status") == LOCK_PASS, "Physics protocol lock did not pass")
    _require(lock.get("protocol_sha256") == sha256_file(config_path), "Physics protocol changed after lock")
    _require(lock.get("physics_acquisition_authorized") is True, "Physics acquisition was not authorized")
    _require(lock.get("model_scoring_authorized_now") is False, "Model scoring was unexpectedly authorized")

    coordinate_summary = json.loads(resolved["v4_1_coordinate_summary"].read_text(encoding="utf-8"))
    _require(coordinate_summary.get("status") == config["inputs"]["v4_1_coordinate_summary"]["required_status"], "Coordinate gate did not pass")
    _require(coordinate_summary.get("future_physics_protocol_authorized") is True, "Coordinate stage did not authorize acquisition protocol")
    coordinates = _read_jsonl(resolved["coordinate_ledger"])
    hashes = [str(row["design_hash"]) for row in coordinates]
    per_root_coordinates = Counter(str(row["root_seed_id"]) for row in coordinates)
    labeling = config["physics_labeling"]
    _require(len(coordinates) == int(labeling["expected_coordinates"]), "Coordinate ledger row count mismatch")
    _require(len(per_root_coordinates) == int(labeling["expected_roots"]), "Coordinate root count mismatch")
    _require(set(per_root_coordinates.values()) == {int(labeling["expected_rows_per_root"])}, "Coordinate rows per root mismatch")
    _require(len(set(hashes)) == len(hashes), "Coordinate design hashes are not unique")

    output = output_root.resolve() if output_root is not None else runtime_root
    shards_root = output / str(config["artifacts"]["future_physics_shards"])
    sealed_path = output / str(config["artifacts"]["future_sealed_physics_ledger"])
    summary_path = output / str(config["artifacts"]["future_physics_summary"])
    report_path = output / str(config["artifacts"]["future_physics_report"])
    prediction_path = output / str(config["artifacts"]["future_prediction_ledger"])
    _require(not prediction_path.exists(), "Prediction artifact exists before acquisition")
    if shards_root.exists() and not resume:
        raise Geometry19StableRootS3V41PhysicsError(f"Refusing to overwrite physics shards: {shards_root}")
    shards_root.mkdir(parents=True, exist_ok=True)

    context = build_pipeline_context(repo_root, pipeline_profile="v2_2_development")
    _require(context.target_domain is not None, "V2.2 target domain unavailable")
    _require(context.target_domain.source_hash == sha256_file(resolved["target_domain"]), "Runtime target-domain hash mismatch")
    protocol_hash = sha256_file(config_path)
    coordinate_hash = sha256_file(resolved["coordinate_ledger"])
    shard_size = int(labeling["shard_size"])
    evaluated: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    for start in range(0, len(coordinates), shard_size):
        index = start // shard_size
        inputs = coordinates[start : start + shard_size]
        shard_path = shards_root / f"part-{index:05d}.jsonl"
        metadata_path = shards_root / f"part-{index:05d}.meta.json"
        identity = {
            "protocol_hash": protocol_hash,
            "coordinate_ledger_hash": coordinate_hash,
            "pipeline_context_hash": context.context_hash,
            "input_hash": sha256(_canonical_bytes(inputs)).hexdigest(),
        }
        if resume and shard_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            _require(all(metadata.get(key) == value for key, value in identity.items()), f"Shard identity mismatch: {index}")
            _require(metadata.get("output_hash") == sha256_file(shard_path), f"Shard output hash mismatch: {index}")
            rows = _read_jsonl(shard_path)
        else:
            rows = []
            for coordinate in inputs:
                result = _evaluate_row(dict(coordinate), context)
                result["evaluation_eligible"] = _eligible(result)
                result["evaluation_status"] = "EVALUATION_ELIGIBLE" if result["evaluation_eligible"] else "EVALUATION_REJECTED"
                rows.append(result)
            _write_jsonl(shard_path, rows)
            metadata = {
                **identity,
                "shard_index": index,
                "row_count": len(rows),
                "output_hash": sha256_file(shard_path),
            }
            _write_json(metadata_path, metadata)
        _require(len(rows) == len(inputs), f"Shard row count mismatch: {index}")
        _require([row["design_hash"] for row in rows] == [row["design_hash"] for row in inputs], f"Shard coordinate identity changed: {index}")
        evaluated.extend(rows)
        shard_records.append(metadata)

    _require(len(shard_records) == 33, "Physics shard count mismatch")
    _require([row["design_hash"] for row in evaluated] == hashes, "Sealed ledger coordinate order or identity changed")
    _write_jsonl(sealed_path, evaluated)

    counts = {
        "evaluated": len(evaluated),
        "solver_converged": sum(bool(row["solver"]["converged"]) for row in evaluated),
        "numerically_valid": sum(bool(row["constraint_assessment"]["numerically_valid"]) for row in evaluated),
        "physics_domain_valid": sum(bool(row["constraint_assessment"]["physics_domain_valid"]) for row in evaluated),
        "inside_target_domain": sum(bool(row.get("target_domain_assessment") and row["target_domain_assessment"]["inside"]) for row in evaluated),
        "evaluation_eligible": sum(bool(row["evaluation_eligible"]) for row in evaluated),
    }
    per_root_total = Counter(str(row["root_seed_id"]) for row in evaluated)
    per_root_eligible = Counter(str(row["root_seed_id"]) for row in evaluated if row["evaluation_eligible"])
    segment_total: defaultdict[str, int] = defaultdict(int)
    segment_eligible: defaultdict[str, int] = defaultdict(int)
    for row in evaluated:
        for segment in row["source_root_segments"]:
            if segment not in ("low_mass_low_to_mid", "low_mass_mid_to_high"):
                continue
            segment_total[str(segment)] += 1
            if row["evaluation_eligible"]:
                segment_eligible[str(segment)] += 1
    segment_fractions = {segment: segment_eligible[segment] / segment_total[segment] for segment in sorted(segment_total)}
    overall_fraction = counts["evaluation_eligible"] / counts["evaluated"]
    gates = config["acquisition_gates"]
    acquisition_pass = all(
        (
            len(per_root_total) == int(gates["exact_roots"]),
            counts["evaluated"] == int(gates["exact_evaluations"]),
            set(per_root_total.values()) == {int(gates["exact_rows_per_root"])},
            min(per_root_eligible.get(root, 0) for root in per_root_total) >= int(gates["minimum_evaluation_eligible_per_root"]),
            overall_fraction >= float(gates["minimum_evaluation_eligible_fraction_overall"]),
            bool(segment_fractions),
            all(value >= float(gates["minimum_evaluation_eligible_fraction_each_source_target_segment"]) for value in segment_fractions.values()),
            len({row["design_hash"] for row in evaluated}) == len(evaluated),
        )
    )
    status = PASS_STATUS if acquisition_pass else FAIL_STATUS
    summary: dict[str, Any] = {
        "status": status,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope_definition": config["scope_definition"],
        "adaptive_development_after_observed_failures": True,
        "not_fresh_confirmation": True,
        "protocol_path": _relative(config_path, repo_root),
        "protocol_sha256": protocol_hash,
        "protocol_lock_path": _relative(lock_path, repo_root),
        "protocol_lock_sha256": sha256_file(lock_path),
        "implementation_path": _relative(Path(__file__), repo_root),
        "implementation_sha256": sha256_file(Path(__file__)),
        "input_hash_audit": input_audit,
        "all_input_hashes_match": True,
        "counts": counts,
        "evaluation_eligible_fraction_overall": overall_fraction,
        "evaluation_eligible_per_root": dict(sorted(per_root_eligible.items())),
        "minimum_evaluation_eligible_per_root_observed": min(per_root_eligible.get(root, 0) for root in per_root_total),
        "source_target_segment_total": dict(sorted(segment_total.items())),
        "source_target_segment_evaluation_eligible": dict(sorted(segment_eligible.items())),
        "source_target_segment_evaluation_eligible_fraction": segment_fractions,
        "coordinate_design_hashes_preserved_in_order": True,
        "sealed_design_hashes_unique": len({row["design_hash"] for row in evaluated}) == len(evaluated),
        "all_shards_complete_before_gate": len(shard_records) == 33 and all(item["row_count"] == 96 for item in shard_records),
        "pipeline_context_hash": context.context_hash,
        "physics_bundle_hash": context.physics_runtime.physics_bundle_hash,
        "target_domain_hash": context.target_domain.source_hash,
        "model_manifest_content_read": False,
        "model_files_opened": False,
        "model_predictions_executed": False,
        "confirmation_content_read": False,
        "model_scoring_executed": False,
        "model_scoring_authorized_now": False,
        "future_scoring_protocol_authorized": acquisition_pass,
        "inverse_design_authorized": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "next_authorized_stage": "freeze_separate_v4_1_frozen_model_scoring_protocol" if acquisition_pass else "stop_and_review_v4_1_physics_acquisition_failure",
        "artifacts": {
            "sealed_physics_ledger": _relative(sealed_path, repo_root),
            "sealed_physics_ledger_sha256": sha256_file(sealed_path),
            "physics_shards": [_relative(shards_root / f"part-{index:05d}.jsonl", repo_root) for index in range(len(shard_records))],
            "physics_shard_metadata": shard_records,
        },
    }
    _write_text(report_path, _report(summary))
    summary["artifacts"]["report"] = _relative(report_path, repo_root)
    summary["artifacts"]["report_sha256"] = sha256_file(report_path)
    _write_json(summary_path, summary)
    return summary
