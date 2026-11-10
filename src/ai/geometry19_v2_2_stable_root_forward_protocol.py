"""Protocol lock for stable-root S2 models and S3 fresh evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.ai.geometry19_surrogate import sha256_file


PROTOCOL_VERSION = "geometry19_v2_2_stable_root_forward_fresh_eval_v1"
LOCK_PASS = "STABLE_ROOT_S2_S3_PROTOCOL_LOCK_PASS"


class Geometry19StableRootForwardProtocolError(ValueError):
    """Raised when the S2/S3 protocol lock cannot be established."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Geometry19StableRootForwardProtocolError(message)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _artifact_path(path: Path, repo_root: Path) -> str:
    return str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path)


def validate_protocol(config: Mapping[str, Any]) -> None:
    _require(config.get("protocol_version") == PROTOCOL_VERSION, "Protocol version mismatch")
    _require(config.get("stage") == "S2_S3_forward_model_and_fresh_evaluation_protocol_lock", "Stage mismatch")
    _require(config.get("formal_use_allowed") is False, "Formal use must remain forbidden")
    _require(config.get("deployment_allowed") is False, "Deployment must remain forbidden")
    population = config.get("population", {})
    _require(population.get("stable_roots") == 38, "Stable-root population changed")
    _require(population.get("stable_development_rows") == 2935, "Stable-row population changed")
    model = config.get("S2_model_development", {})
    _require(model.get("output_transform") == "natural_log", "Output transform changed")
    _require(model.get("sample_weighting") == "equal_total_weight_per_root", "Root weighting changed")
    _require(model.get("fresh_evaluation_visible_during_selection") is False, "Fresh evaluation leakage")
    split = model.get("inner_split", {})
    _require(split.get("grouping_key") == "design_hash", "Inner grouping key changed")
    _require(split.get("keep_each_stable_root_represented_in_each_training_fold") is True, "Root stratification missing")
    _require(split.get("parent_seed_id_overlap_report_required") is True, "Lineage-overlap audit missing")
    expected_arms = [
        "all_78_global_root_balanced_ridge",
        "stable_38_pooled_root_balanced_ridge",
        "stable_38_nearest_root_local_ridge",
        "stable_38_oracle_root_local_ridge",
        "stable_38_global_plus_root_residual_ridge",
    ]
    _require([item["id"] for item in model.get("arms", [])] == expected_arms, "S2 arms changed")
    _require(model.get("ridge_alpha_candidates") == [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0], "Ridge grid changed")
    fresh = config.get("S3_fresh_evaluation", {})
    _require(fresh.get("attempts_per_root") == 96, "Attempts per root changed")
    _require(fresh.get("antithetic_pairs_per_root") == 48, "Pair count changed")
    _require(fresh.get("prelabel_coordinate_ledger_required") is True, "Prelabel ledger missing")
    _require(fresh.get("evaluation_rows_may_enter_training") is False, "Evaluation training leakage")
    _require(fresh.get("adaptive_performance_error_top_up") == "forbidden", "Adaptive top-up must remain forbidden")
    authorization = config.get("authorization", {})
    _require(authorization.get("inverse_design_authorized") is False, "Inverse design cannot be authorized")
    _require(authorization.get("original_local_chart_P4_authorized") is False, "Original P4 cannot be authorized")
    confirmation = config.get("inputs", {}).get("sealed_confirmation", {})
    _require(confirmation.get("access") == "sha256_only_no_parse", "Confirmation access changed")


def _inner_split_audit(
    stable_ids: set[str], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_root: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        root_id = str(row["root_seed_id"])
        if root_id in stable_ids:
            by_root[root_id].append(row)
    fold_rows: Counter[int] = Counter()
    fold_roots: dict[int, set[str]] = defaultdict(set)
    root_fold_counts: dict[str, dict[str, int]] = {}
    parent_groups_per_root: dict[str, int] = {}
    roots_with_parent_overlap: list[str] = []
    for root_id in sorted(stable_ids):
        rows = sorted(by_root[root_id], key=lambda row: str(row["design_hash"]))
        _require(len({str(row["design_hash"]) for row in rows}) == len(rows), f"Duplicate design hash in root {root_id}")
        assignments = {str(row["design_hash"]): index % 4 for index, row in enumerate(rows)}
        counts = Counter(assignments.values())
        _require(set(counts) == {0, 1, 2, 3}, f"Root missing inner validation fold: {root_id}")
        root_fold_counts[root_id] = {str(fold): counts[fold] for fold in range(4)}
        for row in rows:
            fold = assignments[str(row["design_hash"])]
            fold_rows[fold] += 1
            fold_roots[fold].add(root_id)
        parents: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            parents[str(row.get("parent_seed_id"))].add(assignments[str(row["design_hash"])])
        parent_groups_per_root[root_id] = len(parents)
        if any(len(folds) > 1 for folds in parents.values()):
            roots_with_parent_overlap.append(root_id)
    return {
        "method": "sort_unique_design_hashes_within_each_root_then_round_robin",
        "fold_rows": {str(fold): fold_rows[fold] for fold in range(4)},
        "fold_roots": {str(fold): len(fold_roots[fold]) for fold in range(4)},
        "all_38_roots_represented_in_every_validation_fold": all(
            len(fold_roots[fold]) == 38 for fold in range(4)
        ),
        "root_fold_row_counts": root_fold_counts,
        "parent_groups_per_root_minimum": min(parent_groups_per_root.values()),
        "parent_groups_per_root_maximum": max(parent_groups_per_root.values()),
        "roots_with_fewer_than_4_parent_groups": sum(
            value < 4 for value in parent_groups_per_root.values()
        ),
        "roots_with_parent_seed_id_spanning_inner_folds": roots_with_parent_overlap,
        "parent_seed_id_overlap_is_known_inner_selection_limitation": True,
        "final_claim_depends_on_fresh_S3": True,
    }


def _report(summary: Mapping[str, Any]) -> str:
    split = summary["inner_split_audit"]
    return "\n".join(
        [
            "# Stable-Root S2/S3 Protocol Lock",
            "",
            f"- Status: `{summary['status']}`",
            f"- Stable roots locked: {summary['stable_roots']}",
            f"- Stable development rows locked: {summary['stable_development_rows']}",
            f"- S2 arms locked: {len(summary['model_arm_ids'])}",
            f"- S3 maximum attempts locked: {summary['fresh_evaluation_maximum_attempts']}",
            f"- Roots with fewer than four parent groups: {split['roots_with_fewer_than_4_parent_groups']}",
            f"- All roots represented in every design-hash inner fold: `{str(split['all_38_roots_represented_in_every_validation_fold']).lower()}`",
            "- S2 training authorized: `true`",
            "- S3 generation authorized before model-manifest lock: `false`",
            "- Model training executed: `false`",
            "- Fresh coordinates generated: `false`",
            "- Fresh physics labels generated: `false`",
            "- Inverse design authorized: `false`",
            "- Original P4 authorized: `false`",
            "- Confirmation content parsed: `false`",
            "",
            "Parent-seed grouping is infeasible for root-stratified four-fold selection because 28 of 38 stable roots have fewer than four parent groups. Inner selection therefore uses unique design hashes within each root and records parent-lineage overlap. Only sealed S3 data may support the final accuracy claim.",
            "",
        ]
    )


def lock_stable_root_forward_protocol(
    repo_root: Path,
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Lock S2/S3 inputs and rules without training or data generation."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_protocol(config)
    resolved: dict[str, Path] = {}
    hash_audit: dict[str, Any] = {}
    for role, item in config["inputs"].items():
        path = (repo_root / str(item["path"])).resolve()
        _require(path.is_file(), f"Missing input: {role}")
        actual = sha256_file(path)
        _require(actual == str(item["sha256"]), f"Input hash mismatch: {role}")
        resolved[role] = path
        hash_audit[role] = {
            "path": str(item["path"]),
            "expected_sha256": str(item["sha256"]),
            "actual_sha256": actual,
            "match": True,
            "content_parsed": role != "sealed_confirmation",
        }

    support_summary = json.loads(
        resolved["stable_root_support_summary"].read_text(encoding="utf-8")
    )
    required_status = config["inputs"]["stable_root_support_summary"]["required_status"]
    _require(support_summary.get("status") == required_status, "S1 status mismatch")
    _require(support_summary.get("s2_anchored_mode_authorized") is True, "S1 did not authorize anchored S2")
    _require(support_summary.get("fresh_data_generated") is False, "S1 unexpectedly generated fresh data")
    _require(support_summary.get("forward_surrogate_training_executed") is False, "S1 unexpectedly trained a model")
    _require(support_summary.get("confirmation_content_read") is False, "S1 confirmation isolation failed")

    registry = _read_jsonl(resolved["stable_root_registry"])
    stable_ids = {str(row["root_seed_id"]) for row in registry}
    _require(len(registry) == len(stable_ids) == 38, "Stable registry is not 38 unique roots")
    _require(sum(int(row["development_rows"]) for row in registry) == 2935, "Stable registry row count mismatch")
    records = _read_jsonl(resolved["development_dataset"])
    _require(len(records) == 4529, "Development row count mismatch")
    _require(len({str(row["root_seed_id"]) for row in records}) == 78, "Development root count mismatch")
    _require(len({str(row["design_hash"]) for row in records}) == len(records), "Development design hashes are not unique")
    split_audit = _inner_split_audit(stable_ids, records)
    _require(split_audit["all_38_roots_represented_in_every_validation_fold"], "Inner split root coverage failed")
    _require(split_audit["roots_with_fewer_than_4_parent_groups"] == 28, "Parent-group feasibility changed")

    fresh = config["S3_fresh_evaluation"]
    direction_pairs = sum(int(value) for value in fresh["direction_families"].values())
    shell_pairs = sum(int(item["pairs_per_root"]) for item in fresh["normalized_radius_shells"])
    _require(direction_pairs == shell_pairs == int(fresh["antithetic_pairs_per_root"]), "S3 pair allocation mismatch")
    _require(int(fresh["attempts_per_root"]) == 2 * direction_pairs, "S3 attempt count mismatch")
    _require(int(fresh["maximum_attempts_total"]) == 38 * int(fresh["attempts_per_root"]), "S3 maximum total mismatch")

    runtime_root = (
        output_root.resolve()
        if output_root is not None
        else (repo_root / str(config["artifacts"]["runtime_root"])).resolve()
    )
    future_keys = (
        "future_model_manifest",
        "future_prelabel_ledger",
        "future_evaluation_ledger",
        "future_evaluation_summary",
    )
    future_paths = {
        key: runtime_root / str(config["artifacts"][key]) for key in future_keys
    }
    _require(not any(path.exists() for path in future_paths.values()), "A future S2/S3 artifact already exists at lock time")

    lock_path = runtime_root / str(config["artifacts"]["protocol_lock"])
    report_path = runtime_root / str(config["artifacts"]["protocol_lock_report"])
    implementation_path = Path(__file__).resolve()
    summary: dict[str, Any] = {
        "status": LOCK_PASS,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_path": _artifact_path(config_path, repo_root),
        "protocol_sha256": sha256_file(config_path),
        "implementation_path": _artifact_path(implementation_path, repo_root),
        "implementation_sha256": sha256_file(implementation_path),
        "input_hash_audit": hash_audit,
        "all_input_hashes_match": True,
        "stable_roots": 38,
        "stable_development_rows": 2935,
        "all_root_comparator_roots": 78,
        "all_root_comparator_rows": 4529,
        "model_arm_ids": [
            item["id"] for item in config["S2_model_development"]["arms"]
        ],
        "ridge_alpha_candidates": config["S2_model_development"]["ridge_alpha_candidates"],
        "inner_split_audit": split_audit,
        "fresh_evaluation_attempts_per_root": int(fresh["attempts_per_root"]),
        "fresh_evaluation_maximum_attempts": int(fresh["maximum_attempts_total"]),
        "fresh_evaluation_minimum_valid_points_per_root": int(fresh["valid_points_minimum_per_root"]),
        "fresh_evaluation_rules_frozen": True,
        "metrics_and_gates_frozen": True,
        "future_artifacts_absent_at_lock": {
            key: not path.exists() for key, path in future_paths.items()
        },
        "s2_training_authorized": True,
        "s3_generation_authorized_now": False,
        "s3_generation_authorized_after_s2_model_manifest_lock": True,
        "model_training_executed": False,
        "fresh_coordinates_generated": False,
        "fresh_physics_labels_generated": False,
        "inverse_design_authorized": False,
        "original_local_chart_p4_authorized": False,
        "confirmation_hash_checked": True,
        "confirmation_content_read": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "next_authorized_stage": "S2_train_select_reload_and_hash_forward_models",
    }
    _write_text(report_path, _report(summary))
    summary["report_path"] = _artifact_path(report_path, repo_root)
    summary["report_sha256"] = sha256_file(report_path)
    _write_json(lock_path, summary)
    return summary
