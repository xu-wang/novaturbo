"""Lock one-shot five-arm scoring on the sealed S3 v4.1 development rows."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.ai.geometry19_surrogate import sha256_file


PROTOCOL_VERSION = "geometry19_v2_2_stable_root_s3_v4_1_model_scoring_v1"
LOCK_PASS = "STABLE_ROOT_S3_V4_1_MODEL_SCORING_PROTOCOL_LOCK_PASS"


class Geometry19StableRootS3V41ScoringProtocolError(ValueError):
    """Raised when the frozen one-shot scoring contract cannot be locked."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Geometry19StableRootS3V41ScoringProtocolError(message)


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
    _write_text(path, "".join(_canonical_bytes(row).decode("utf-8") + "\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _hash_string_sequence(values: Sequence[str]) -> str:
    return sha256(json.dumps(list(values), separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_protocol(config: Mapping[str, Any]) -> None:
    _require(config.get("protocol_version") == PROTOCOL_VERSION, "Protocol version mismatch")
    _require(config.get("stage") == "S3_v4_1_five_arm_model_scoring_protocol_lock", "Stage mismatch")
    _require(config.get("scope_definition") == "post_selected_33root_medoid_covariance_development_only", "Scope changed")
    _require(config.get("formal_use_allowed") is False, "Formal use must remain forbidden")
    _require(config.get("deployment_allowed") is False, "Deployment must remain forbidden")
    population = config.get("evaluation_population", {})
    _require(population.get("exact_rows") == 3039, "Scoring row count changed")
    _require(population.get("exact_roots") == 33, "Scoring root count changed")
    _require(population.get("minimum_rows_per_root") == 67, "Minimum root rows changed")
    _require(population.get("identical_rows_and_order_for_all_arms") is True, "Common-row rule changed")
    _require(population.get("rows_may_enter_training") is False, "Evaluation rows cannot enter training")
    expected = [
        "all_78_global_root_balanced_ridge",
        "stable_38_pooled_root_balanced_ridge",
        "stable_38_nearest_root_local_ridge",
        "stable_38_oracle_root_local_ridge",
        "stable_38_global_plus_root_residual_ridge",
    ]
    _require([item["id"] for item in config.get("models", {}).get("exact_arms", [])] == expected, "Scoring arms changed")
    scoring = config.get("scoring", {})
    _require(scoring.get("outputs") == ["thrust_N", "tsfc_kg_N_s", "mass_kg"], "Output order changed")
    _require(scoring.get("model_output_transform") == "natural_log", "Output transform changed")
    _require(scoring.get("prediction_adaptive_changes") == "forbidden", "Adaptive scoring cannot be allowed")
    gates = config.get("accuracy_gates", {})
    _require(gates.get("maximum_worst_root_rho_p95") == 5.0, "Worst-root gate changed")
    _require(gates.get("minimum_worst_root_rho_p95_reduction") == 0.15, "Improvement gate changed")
    _require(gates.get("maximum_nearest_vs_oracle_relative_rho_p95_gap") == 0.20, "Routing gap gate changed")
    _require(config["inputs"]["sealed_confirmation"]["access"] == "sha256_only_no_parse", "Confirmation access changed")
    authorization = config.get("authorization", {})
    _require(authorization.get("model_loading_or_prediction_executed_now") is False, "Lock stage cannot run models")
    _require(authorization.get("inverse_design_authorized") is False, "Inverse design cannot be authorized")


def _report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Stable-Root S3 v4.1 Five-Arm Model-Scoring Protocol Lock",
            "",
            f"- Status: `{summary['status']}`",
            f"- Common eligible rows: {summary['eligible_rows']}",
            f"- Roots: {summary['eligible_roots']}",
            f"- Minimum rows per root: {summary['minimum_rows_per_root']}",
            f"- Frozen model arms: {len(summary['model_arms'])}",
            "- Model manifest parsed: `true`",
            "- Model files hash checked: `true`",
            "- Model arrays loaded: `false`",
            "- Predictions executed: `false`",
            "- Confirmation content parsed: `false`",
            "- Inverse design authorized: `false`",
            "",
            "The future score is a one-shot development comparison on identical post-selected v4.1 rows. It is not fresh confirmation or unknown-manifold evidence.",
            "",
        ]
    )


def lock_s3_v4_1_model_scoring_protocol(
    repo_root: Path,
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Freeze rows, models, metrics, and gates without loading model arrays."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_protocol(config)
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
            "content_parsed": role != "sealed_confirmation",
        }

    acquisition = json.loads(resolved["acquisition_summary"].read_text(encoding="utf-8"))
    _require(acquisition.get("status") == config["inputs"]["acquisition_summary"]["required_status"], "Acquisition gate did not pass")
    _require(acquisition.get("future_scoring_protocol_authorized") is True, "Acquisition did not authorize a scoring protocol")
    _require(acquisition.get("model_predictions_executed") is False, "Predictions already executed during acquisition")
    lock = json.loads(resolved["acquisition_protocol_lock"].read_text(encoding="utf-8"))
    _require(lock.get("status") == config["inputs"]["acquisition_protocol_lock"]["required_status"], "Acquisition protocol lock changed")

    ledger = _read_jsonl(resolved["sealed_physics_ledger"])
    eligible = [row for row in ledger if row.get("evaluation_eligible") is True]
    design_hashes = [str(row["design_hash"]) for row in eligible]
    population = config["evaluation_population"]
    root_counts = Counter(str(row["root_seed_id"]) for row in eligible)
    _require(len(ledger) == 3168, "Sealed ledger row count changed")
    _require(len(eligible) == int(population["exact_rows"]), "Eligible scoring row count changed")
    _require(len(root_counts) == int(population["exact_roots"]), "Eligible scoring root count changed")
    _require(min(root_counts.values()) == int(population["minimum_rows_per_root"]), "Minimum eligible root rows changed")
    _require(len(set(design_hashes)) == len(design_hashes), "Eligible design hashes are not unique")
    _require(_hash_string_sequence(design_hashes) == population["design_hash_sequence_sha256"], "Eligible design-hash sequence changed")
    _require(all(all(isinstance(row["performance"].get(name), (int, float)) and math.isfinite(float(row["performance"][name])) and float(row["performance"][name]) > 0 for name in config["scoring"]["outputs"]) for row in eligible), "Eligible performance values are not finite positive")

    manifest = json.loads(resolved["s2_model_manifest"].read_text(encoding="utf-8"))
    _require(manifest.get("status") == config["inputs"]["s2_model_manifest"]["required_status"], "Model manifest gate did not pass")
    _require(manifest.get("all_five_models_locked") is True, "Five models are not locked")
    manifest_models = {str(item["arm_id"]): item for item in manifest["models"]}
    model_audit: list[dict[str, Any]] = []
    for frozen in config["models"]["exact_arms"]:
        arm = str(frozen["id"])
        _require(arm in manifest_models, f"Missing model arm: {arm}")
        item = manifest_models[arm]
        _require(item["model_sha256"] == frozen["sha256"], f"Manifest model hash changed: {arm}")
        path = (repo_root / str(item["model_path"])).resolve()
        _require(path.is_file(), f"Missing model file: {arm}")
        actual = sha256_file(path)
        _require(actual == frozen["sha256"], f"Model file hash mismatch: {arm}")
        model_audit.append({
            "arm_id": arm,
            "model_path": str(item["model_path"]),
            "expected_sha256": str(frozen["sha256"]),
            "actual_sha256": actual,
            "match": True,
            "model_arrays_loaded": False,
        })

    runtime_root = output_root.resolve() if output_root is not None else (repo_root / str(config["artifacts"]["runtime_root"])).resolve()
    future_keys = ("future_prediction_ledger", "future_scoring_summary", "future_scoring_report")
    future_paths = {key: runtime_root / str(config["artifacts"][key]) for key in future_keys}
    _require(not any(path.exists() for path in future_paths.values()), "A future scoring artifact already exists")
    registry_path = runtime_root / str(config["artifacts"]["eligible_row_registry"])
    lock_path = runtime_root / str(config["artifacts"]["protocol_lock"])
    report_path = runtime_root / str(config["artifacts"]["protocol_lock_report"])
    registry = [
        {
            "scoring_row_index": index,
            "design_hash": row["design_hash"],
            "root_seed_id": row["root_seed_id"],
            "source_root_segments": row["source_root_segments"],
            "radius_fraction": row["radius_fraction"],
            "combined_source_protocol": row["combined_source_protocol"],
        }
        for index, row in enumerate(eligible)
    ]
    _write_jsonl(registry_path, registry)
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
        "eligible_rows": len(eligible),
        "eligible_roots": len(root_counts),
        "minimum_rows_per_root": min(root_counts.values()),
        "eligible_design_hash_sequence_sha256": _hash_string_sequence(design_hashes),
        "model_arms": [item["id"] for item in config["models"]["exact_arms"]],
        "model_file_audit": model_audit,
        "model_manifest_content_read": True,
        "model_files_hash_checked": True,
        "model_arrays_loaded": False,
        "model_predictions_executed": False,
        "confirmation_content_read": False,
        "metrics_and_gates_frozen": True,
        "future_artifacts_absent_at_lock": {key: not path.exists() for key, path in future_paths.items()},
        "model_scoring_authorized": True,
        "inverse_design_authorized": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "next_authorized_stage": "execute_one_shot_v4_1_five_arm_model_scoring",
        "artifacts": {
            "eligible_row_registry": _relative(registry_path, repo_root),
            "eligible_row_registry_sha256": sha256_file(registry_path),
        },
    }
    _write_text(report_path, _report(summary))
    summary["artifacts"]["report"] = _relative(report_path, repo_root)
    summary["artifacts"]["report_sha256"] = sha256_file(report_path)
    _write_json(lock_path, summary)
    return summary
