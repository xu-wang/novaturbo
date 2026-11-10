"""P0 protocol validation and immutable input locking for local-chart MoE."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.ai.geometry19_surrogate import sha256_file


PROTOCOL_VERSION = "geometry19_v2_2_local_chart_moe_v1"
P0_PASS_STATUS = "LOCAL_CHART_MOE_P0_PROTOCOL_LOCK_PASS"


class Geometry19LocalChartProtocolError(ValueError):
    """Raised when the preregistered P0 contract is incomplete or altered."""


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Geometry19LocalChartProtocolError(message)


def validate_protocol_schema(config: Mapping[str, Any]) -> None:
    """Validate the decisions that P0 must freeze before any model experiment."""

    _require(config.get("protocol_version") == PROTOCOL_VERSION, "Protocol version mismatch")
    _require(config.get("stage") == "P0_protocol_preregistration_and_input_lock", "P0 stage is not frozen")
    _require(config.get("formal_use_allowed") is False, "Formal use must remain forbidden")
    _require(config.get("deployment_allowed") is False, "Deployment must remain forbidden")
    _require(config.get("default_pipeline_modified") is False, "Default pipeline must remain unchanged")

    inputs = config.get("inputs")
    _require(isinstance(inputs, Mapping), "inputs must be a mapping")
    required_inputs = {
        "design_schema",
        "target_domain",
        "development_dataset",
        "development_folds",
        "observed_ablation_summary",
        "observed_oof_predictions",
        "observed_support_model",
        "sealed_confirmation",
    }
    _require(set(inputs) == required_inputs, "Frozen input set is incomplete or expanded")
    for role, value in inputs.items():
        _require(isinstance(value, Mapping), f"Input {role} must be a mapping")
        _require(bool(value.get("path")), f"Input {role} has no path")
        digest = str(value.get("sha256", ""))
        _require(len(digest) == 64, f"Input {role} has no complete SHA-256")

    confirmation = inputs["sealed_confirmation"]
    _require(confirmation.get("access") == "sha256_only_no_parse", "Confirmation access must be hash-only")
    _require(confirmation.get("confirmation_content_read") is False, "Confirmation content read must be false")
    _require(confirmation.get("confirmation_reuse_allowed") is False, "Confirmation reuse must be false")

    _require(config.get("target_segments") == ["low_mass_low_to_mid", "low_mass_mid_to_high"], "Target segments changed")
    outer = config.get("outer_evaluation", {})
    _require(outer.get("fold_count") == 5, "Outer fold count must remain five")
    _require(outer.get("fold_assignment_frozen") is True, "Outer folds must be frozen")
    _require(outer.get("regrouping_allowed") is False, "Outer regrouping must be forbidden")
    _require(outer.get("support_rejection_used_for_primary_metric") is False, "Support rejection cannot filter the primary metric")

    inner = config.get("inner_selection", {})
    _require(inner.get("method") == "GroupKFold", "Inner selection must use GroupKFold")
    _require(inner.get("grouping_key") == "root_seed_id", "Inner grouping key changed")
    _require(inner.get("fold_count") == 4, "Inner fold count must remain four")
    _require(inner.get("outer_validation_visible_during_selection") is False, "Outer validation leakage is forbidden")

    _require(config.get("charts", {}).get("candidate_k") == [3, 4, 6, 8], "Chart K candidates changed")
    expected_arms = [
        "legacy_no_change",
        "global_ridge",
        "nearest_root_local_ridge",
        "hard_chart_ridge",
        "soft_chart_ridge",
        "soft_chart_rbf",
        "residual_soft_chart_ridge",
        "oracle_chart_ridge",
    ]
    actual_arms = [str(value.get("id")) for value in config.get("experiment_arms", [])]
    _require(actual_arms == expected_arms, "Experiment arms changed")
    _require(config.get("routing", {}).get("top_k") == 2, "Routing top-k must remain two")
    _require(config.get("routing", {}).get("learned_gate_allowed_in_first_round") is False, "Learned gating is forbidden in round one")

    training = config.get("training", {})
    _require(training.get("neural_seeds") == [42], "Neural seeds changed")
    _require(training.get("physical_gpu") == 6, "Physical GPU must be six")
    _require(training.get("process_device") == "cuda:0", "Process device must be cuda:0")
    _require(training.get("allow_cpu_fallback") is False, "Silent CPU fallback is forbidden")
    _require(training.get("deterministic_algorithms") is True, "Deterministic algorithms must be enabled")

    gates = config.get("gates", {})
    advancement = gates.get("advancement", {})
    readiness = gates.get("readiness", {})
    _require(advancement.get("minimum_reduction_vs_legacy_no_change") == 0.20, "Legacy advancement gate changed")
    _require(advancement.get("minimum_reduction_vs_global_ridge") == 0.15, "Ridge advancement gate changed")
    _require(advancement.get("minimum_reduction_each_target_segment_vs_global_ridge") == 0.10, "Segment advancement gate changed")
    _require(advancement.get("minimum_unseen_target_root_support_acceptance_fraction") == 0.80, "Support gate changed")
    _require(readiness.get("maximum_cross_fold_worst_root_rho_p95") == 5.0, "Absolute readiness gate changed")
    _require(readiness.get("maximum_tsfc_dominant_roots_in_worst_10") == 7, "TSFC readiness gate changed")


def _load_development_roots(dataset_path: Path) -> tuple[int, set[str], dict[str, set[str]]]:
    rows = 0
    roots: set[str] = set()
    roots_by_segment: dict[str, set[str]] = defaultdict(set)
    with dataset_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            root_id = str(record["root_seed_id"])
            segment = str(record["assigned_segment"])
            rows += 1
            roots.add(root_id)
            roots_by_segment[segment].add(root_id)
    return rows, roots, roots_by_segment


def _fold_audit(
    config: Mapping[str, Any], dataset_path: Path, folds_path: Path
) -> dict[str, Any]:
    dataset_input = config["inputs"]["development_dataset"]
    folds_input = config["inputs"]["development_folds"]
    rows, dataset_roots, roots_by_segment = _load_development_roots(dataset_path)
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    assignments = {str(key): int(value) for key, value in folds["root_assignments"].items()}
    fold_count = int(folds_input["expected_fold_count"])

    _require(rows == int(dataset_input["expected_rows"]), "Development row count changed")
    _require(len(dataset_roots) == int(dataset_input["expected_global_roots"]), "Development root count changed")
    _require(set(assignments) == dataset_roots, "Fold assignments do not cover exactly the development roots")
    _require(set(assignments.values()) == set(range(fold_count)), "Fold IDs are not the frozen contiguous five folds")
    _require(folds.get("grouping_key") == folds_input["grouping_key"], "Fold grouping key changed")
    _require(int(folds.get("fold_count")) == fold_count, "Fold count file disagrees with protocol")

    expected_per_fold = int(
        config["outer_evaluation"]["expected_validation_roots_per_target_segment_per_fold"]
    )
    segment_counts: dict[str, Any] = {}
    for segment in config["target_segments"]:
        root_ids = roots_by_segment[segment]
        per_fold = {
            str(fold): sum(assignments[root_id] == fold for root_id in root_ids)
            for fold in range(fold_count)
        }
        _require(
            all(value == expected_per_fold for value in per_fold.values()),
            f"Target segment {segment} no longer has {expected_per_fold} validation roots per fold",
        )
        segment_counts[segment] = {"global_roots": len(root_ids), "validation_roots_by_fold": per_fold}

    roots_by_fold = {
        str(fold): sum(value == fold for value in assignments.values())
        for fold in range(fold_count)
    }
    return {
        "rows": rows,
        "global_roots": len(dataset_roots),
        "assigned_roots": len(assignments),
        "root_fold_exclusive": True,
        "root_set_exact_match": True,
        "fold_count": fold_count,
        "roots_by_fold": roots_by_fold,
        "target_segments": segment_counts,
    }


def _report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Geometry19 V2.2 Local-Chart MoE P0 Protocol Input Lock",
        "",
        f"- Status: `{payload['status']}`",
        f"- Protocol: `{payload['protocol_version']}`",
        f"- Protocol SHA-256: `{payload['protocol_sha256']}`",
        f"- Development rows / roots: {payload['fold_audit']['rows']} / {payload['fold_audit']['global_roots']}",
        "- Confirmation content parsed: `false`",
        "- Confirmation reuse allowed: `false`",
        "- Formal/deployment use allowed: `false` / `false`",
        "",
        "## Frozen input hash audit",
        "",
        "| Role | Match | Access | SHA-256 |",
        "|---|---:|---|---|",
    ]
    for item in payload["input_hash_audit"]:
        lines.append(
            f"| `{item['role']}` | `{str(item['match']).lower()}` | `{item['access']}` | `{item['actual_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Root-exclusive outer folds",
            "",
            "| Segment | Global roots | Fold validation roots |",
            "|---|---:|---|",
        ]
    )
    for segment, item in payload["fold_audit"]["target_segments"].items():
        fold_values = "/".join(str(value) for value in item["validation_roots_by_fold"].values())
        lines.append(f"| `{segment}` | {item['global_roots']} | {fold_values} |")
    lines.extend(
        [
            "",
            "P0 only freezes the development experiment contract. No Jacobian, chart, router, expert, OOF prediction, or confirmation evaluation is produced in this stage.",
            "",
        ]
    )
    return "\n".join(lines)


def run_protocol_input_lock(
    repo_root: Path,
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Validate P0 and emit evidence without parsing sealed confirmation content."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_protocol_schema(config)

    input_hash_audit: list[dict[str, Any]] = []
    resolved: dict[str, Path] = {}
    for role, item in config["inputs"].items():
        path = (repo_root / str(item["path"])).resolve()
        _require(path.is_file(), f"Frozen input is missing: {role}: {path}")
        actual = sha256_file(path)
        expected = str(item["sha256"])
        _require(actual == expected, f"Frozen input hash mismatch: {role}")
        resolved[role] = path
        input_hash_audit.append(
            {
                "role": role,
                "path": str(path.relative_to(repo_root)),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": True,
                "access": str(item["access"]),
            }
        )

    fold_audit = _fold_audit(
        config,
        resolved["development_dataset"],
        resolved["development_folds"],
    )
    runtime_root = (
        output_root.resolve()
        if output_root is not None
        else (repo_root / str(config["artifacts"]["runtime_root"])).resolve()
    )
    audit_path = runtime_root / str(config["artifacts"]["protocol_input_lock"])
    report_path = runtime_root / str(config["artifacts"]["protocol_input_lock_report"])
    implementation_path = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "status": P0_PASS_STATUS,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(config_path.relative_to(repo_root)),
        "protocol_sha256": sha256_file(config_path),
        "implementation_path": str(implementation_path.relative_to(repo_root)),
        "implementation_sha256": sha256_file(implementation_path),
        "protocol_schema_validated": True,
        "all_input_hashes_match": True,
        "confirmation_hash_checked": True,
        "confirmation_content_read": False,
        "confirmation_reuse_allowed": False,
        "outer_folds_frozen": True,
        "model_work_executed": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "input_hash_audit": input_hash_audit,
        "fold_audit": fold_audit,
        "next_authorized_stage": "P1_data_root_lineage_and_fold_integrity_audit",
    }
    _write_json(audit_path, payload)
    _write_text(report_path, _report(payload))
    return payload
