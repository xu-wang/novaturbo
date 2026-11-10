"""Seal unified V2.2 labels for the frozen S3 v3 33-root coordinates."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.ai.geometry19_surrogate import sha256_file
from src.physics.geometry19_v2_2_dataset import _evaluate_row


PROTOCOL_VERSION = "geometry19_v2_2_stable_root_s3_v3_33root_evaluation_v1"
PASS_STATUS = "STABLE_ROOT_S3_V3_33ROOT_PHYSICS_ACQUISITION_PASS"
FAIL_STATUS = "STABLE_ROOT_S3_V3_33ROOT_PHYSICS_ACQUISITION_FAIL"


class Geometry19StableRootS3V3PhysicsError(ValueError):
    """Raised when the frozen physics acquisition contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Geometry19StableRootS3V3PhysicsError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_text(path, "".join(_canonical_bytes(row).decode("utf-8") + "\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def validate_protocol(config: Mapping[str, Any]) -> None:
    _require(config.get("protocol_version") == PROTOCOL_VERSION, "Evaluation protocol mismatch")
    _require(config.get("stage") == "S3_v3_33root_physics_label_seal_then_model_score", "Evaluation stage mismatch")
    _require(config.get("scope_definition") == "bilateral_perturbable_stable_root_subdomain_only", "Scope changed")
    _require(config.get("formal_use_allowed") is False, "Formal use must remain forbidden")
    _require(config.get("deployment_allowed") is False, "Deployment must remain forbidden")
    labeling = config.get("physics_labeling", {})
    _require(labeling.get("pipeline_profile") == "v2_2_development", "Physics profile changed")
    _require(labeling.get("expected_coordinates") == 3168, "Coordinate count changed")
    _require(labeling.get("performance_adaptive_top_up") is False, "Adaptive top-up cannot be allowed")
    gates = config.get("acquisition_gates", {})
    _require(gates.get("exact_roots") == 33, "Root gate changed")
    _require(gates.get("minimum_evaluation_eligible_per_root") == 32, "Per-root gate changed")
    _require(gates.get("minimum_evaluation_eligible_fraction_overall") == 0.80, "Overall gate changed")
    _require(gates.get("minimum_evaluation_eligible_fraction_each_source_target_segment") == 0.80, "Segment gate changed")
    _require(config["inputs"]["sealed_confirmation"]["access"] == "sha256_only_no_parse", "Confirmation access changed")
    _require(config["inputs"]["s2_model_manifest"]["label_stage_access"] == "sha256_only_no_parse", "Model-manifest label-stage access changed")


def _eligible(row: Mapping[str, Any]) -> bool:
    performance = row.get("performance")
    finite_positive = bool(
        isinstance(performance, Mapping)
        and all(
            isinstance(performance.get(name), (int, float))
            and math.isfinite(float(performance[name]))
            and float(performance[name]) > 0.0
            for name in ("thrust_N", "tsfc_kg_N_s", "mass_kg")
        )
    )
    target = row.get("target_domain_assessment")
    return bool(
        row.get("solver", {}).get("converged")
        and row.get("constraint_assessment", {}).get("numerically_valid")
        and row.get("constraint_assessment", {}).get("physics_domain_valid")
        and finite_positive
        and isinstance(target, Mapping)
        and target.get("inside")
    )


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Stable-Root S3 v3 33-Root V2.2 Physics Acquisition",
        "",
        f"- Status: `{summary['status']}`",
        f"- Evaluated coordinates: {summary['counts']['evaluated']}",
        f"- Solver converged: {summary['counts']['solver_converged']}",
        f"- Physics-domain valid: {summary['counts']['physics_domain_valid']}",
        f"- Inside target domain: {summary['counts']['inside_target_domain']}",
        f"- Evaluation eligible: {summary['counts']['evaluation_eligible']}",
        f"- Overall eligible fraction: {summary['evaluation_eligible_fraction_overall']:.6f}",
        f"- Minimum eligible rows per root: {summary['minimum_evaluation_eligible_per_root_observed']}",
        f"- Model scoring authorized: `{str(summary['model_scoring_authorized']).lower()}`",
        "- Model manifest/model files opened: `false`",
        "- Confirmation content parsed: `false`",
        "",
        "All labels were produced by the unified V2.2 development runtime and sealed before model scoring. Results remain limited to the post-selected 33-root software subdomain.",
        "",
    ]
    return "\n".join(lines)


def run_s3_v3_physics_acquisition(
    repo_root: Path,
    config_path: Path,
    *,
    output_root: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Evaluate and seal all coordinates without opening the S2 models."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_protocol(config)
    parsed_roles = {
        "coordinate_summary",
        "coordinate_ledger",
        "stable_root_registry",
        "design_schema",
        "target_domain",
        "coordinate_protocol",
    }
    resolved: dict[str, Path] = {}
    input_audit: dict[str, Any] = {}
    for role, item in config["inputs"].items():
        path = (repo_root / str(item["path"])).resolve()
        _require(path.is_file(), f"Missing evaluation input: {role}")
        actual = sha256_file(path)
        _require(actual == str(item["sha256"]), f"Evaluation input hash mismatch: {role}")
        resolved[role] = path
        input_audit[role] = {
            "path": str(item["path"]),
            "expected_sha256": str(item["sha256"]),
            "actual_sha256": actual,
            "match": True,
            "content_parsed_during_label_stage": role in parsed_roles,
        }
    coordinate_summary = json.loads(resolved["coordinate_summary"].read_text(encoding="utf-8"))
    _require(coordinate_summary.get("status") == config["inputs"]["coordinate_summary"]["required_status"], "Coordinate gate did not pass")
    _require(coordinate_summary.get("physics_labeling_authorized") is True, "Coordinate stage did not authorize labeling")
    coordinates = _read_jsonl(resolved["coordinate_ledger"])
    _require(len(coordinates) == 3168, "Coordinate ledger row count mismatch")
    coordinate_hashes = [str(row["design_hash"]) for row in coordinates]
    _require(len(set(coordinate_hashes)) == len(coordinate_hashes), "Coordinate design hashes are not unique")
    _require(len({str(row["root_seed_id"]) for row in coordinates}) == 33, "Coordinate root count mismatch")

    runtime_root = output_root.resolve() if output_root is not None else (repo_root / str(config["artifacts"]["runtime_root"])).resolve()
    shards_root = runtime_root / str(config["artifacts"]["physics_shards"])
    sealed_path = runtime_root / str(config["artifacts"]["sealed_physics_ledger"])
    summary_path = runtime_root / str(config["artifacts"]["physics_summary"])
    report_path = runtime_root / str(config["artifacts"]["physics_report"])
    if shards_root.exists() and not resume:
        raise Geometry19StableRootS3V3PhysicsError(f"Refusing to overwrite physics shards: {shards_root}")
    shards_root.mkdir(parents=True, exist_ok=True)
    context = build_pipeline_context(repo_root, pipeline_profile="v2_2_development")
    _require(context.target_domain is not None, "V2.2 target domain unavailable")
    _require(context.target_domain.source_hash == sha256_file(resolved["target_domain"]), "Runtime target-domain hash mismatch")
    protocol_hash = sha256_file(config_path)
    coordinate_hash = sha256_file(resolved["coordinate_ledger"])
    shard_size = int(config["physics_labeling"]["shard_size"])
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
        _require([row["design_hash"] for row in rows] == [row["design_hash"] for row in inputs], f"Shard coordinate identity changed: {index}")
        evaluated.extend(rows)
        shard_records.append(metadata)
    _require([row["design_hash"] for row in evaluated] == coordinate_hashes, "Sealed ledger coordinate order or identity changed")
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
    segment_fractions = {
        segment: segment_eligible[segment] / segment_total[segment]
        for segment in sorted(segment_total)
    }
    overall_fraction = counts["evaluation_eligible"] / counts["evaluated"]
    gates = config["acquisition_gates"]
    acquisition_pass = all(
        (
            len(per_root_total) == int(gates["exact_roots"]),
            counts["evaluated"] == int(gates["exact_evaluations"]),
            set(per_root_total.values()) == {96},
            min(per_root_eligible.get(root_id, 0) for root_id in per_root_total) >= int(gates["minimum_evaluation_eligible_per_root"]),
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
        "selected_root_count": len(per_root_total),
        "excluded_boundary_roots_remain_out_of_scope": True,
        "post_selection_bias_acknowledged": True,
        "protocol_path": _relative(config_path, repo_root),
        "protocol_sha256": protocol_hash,
        "implementation_path": _relative(Path(__file__), repo_root),
        "implementation_sha256": sha256_file(Path(__file__)),
        "input_hash_audit": input_audit,
        "all_input_hashes_match": True,
        "counts": counts,
        "evaluation_eligible_fraction_overall": overall_fraction,
        "evaluation_eligible_per_root": dict(sorted(per_root_eligible.items())),
        "minimum_evaluation_eligible_per_root_observed": min(per_root_eligible.get(root_id, 0) for root_id in per_root_total),
        "source_target_segment_total": dict(sorted(segment_total.items())),
        "source_target_segment_evaluation_eligible": dict(sorted(segment_eligible.items())),
        "source_target_segment_evaluation_eligible_fraction": segment_fractions,
        "coordinate_design_hashes_preserved": True,
        "sealed_design_hashes_unique": len({row["design_hash"] for row in evaluated}) == len(evaluated),
        "pipeline_context_hash": context.context_hash,
        "physics_bundle_hash": context.physics_runtime.physics_bundle_hash,
        "target_domain_hash": context.target_domain.source_hash,
        "model_manifest_content_read": False,
        "model_files_opened": False,
        "model_predictions_executed": False,
        "confirmation_content_read": False,
        "model_scoring_authorized": acquisition_pass,
        "inverse_design_authorized": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "next_authorized_stage": "score_five_frozen_S2_models_once_on_sealed_common_rows" if acquisition_pass else "stop_and_review_33root_physics_acquisition_failure",
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
