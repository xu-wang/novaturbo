"""S0/S1 registry and design-support audit for stable Geometry19 roots."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from src.ai.geometry19_surrogate import sha256_file


PROTOCOL_VERSION = "geometry19_v2_2_stable_root_conditional_surrogate_v1"
PASS_BOTH = "STABLE_ROOT_S1_ANCHORED_AND_BLIND_SUPPORT_PASS"
PASS_ANCHORED = "STABLE_ROOT_S1_ANCHORED_PASS_BLIND_SUPPORT_NOT_IDENTIFIABLE"
AUDIT_FAIL = "STABLE_ROOT_S1_SUPPORT_AUDIT_FAIL"


class Geometry19StableRootSupportError(ValueError):
    """Raised when a stable-root protocol or evidence contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Geometry19StableRootSupportError(message)


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
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
        ),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _artifact_path(path: Path, repo_root: Path) -> str:
    return str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path)


def validate_protocol(config: Mapping[str, Any]) -> None:
    _require(config.get("protocol_version") == PROTOCOL_VERSION, "Protocol version mismatch")
    _require(config.get("stage") == "S0_S1_stable_root_registry_and_support_audit", "Stage mismatch")
    _require(config.get("formal_use_allowed") is False, "Formal use must remain forbidden")
    _require(config.get("deployment_allowed") is False, "Deployment must remain forbidden")
    selection = config.get("stable_root_selection", {})
    _require(selection.get("expected_stable_roots") == 38, "Stable-root count changed")
    _require(selection.get("expected_always_unreliable_roots") == 32, "Unreliable-root count changed")
    _require(selection.get("expected_mixed_roots") == 8, "Mixed-root count changed")
    _require(selection.get("post_selection_from_full_development") is True, "Post-selection limitation missing")
    _require(selection.get("fresh_evaluation_required_before_accuracy_claim") is True, "Fresh evidence requirement missing")
    next_stage = config.get("next_stage_contract", {})
    _require(next_stage.get("generate_fresh_data_now") is False, "S0/S1 cannot generate fresh data")
    _require(next_stage.get("train_forward_surrogate_now") is False, "S0/S1 cannot train a surrogate")
    _require(next_stage.get("train_inverse_model_now") is False, "S0/S1 cannot train inverse models")
    _require(next_stage.get("unlock_original_P4") is False, "Original P4 must remain locked")
    confirmation = config.get("inputs", {}).get("sealed_confirmation", {})
    _require(confirmation.get("access") == "sha256_only_no_parse", "Confirmation access changed")


def _schema_arrays(schema: Mapping[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    variables = sorted(schema["design_variables"], key=lambda item: int(item["index"]))
    names = [str(item["name"]) for item in variables]
    lower = np.asarray([float(item["bounds"][0]) for item in variables], dtype=np.float64)
    upper = np.asarray([float(item["bounds"][1]) for item in variables], dtype=np.float64)
    _require(len(names) == 19 and len(set(names)) == 19, "Design schema is not unique 19D")
    _require(np.all(upper > lower), "Design schema contains invalid bounds")
    return names, lower, upper


def _normalize_geometry(
    geometry: Mapping[str, Any], names: Sequence[str], lower: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    _require(set(geometry) == set(names), "Geometry keys do not match frozen 19D schema")
    values = np.asarray([float(geometry[name]) for name in names], dtype=np.float64)
    _require(np.all(np.isfinite(values)), "Geometry contains non-finite values")
    return (values - lower) / scale


def _rates_by_segment(
    records: Sequence[Mapping[str, Any]], accepted: np.ndarray
) -> dict[str, dict[str, Any]]:
    totals: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    for record, is_accepted in zip(records, accepted.tolist()):
        segment = str(record["assigned_segment"])
        totals[segment] += 1
        hits[segment] += int(is_accepted)
    return {
        segment: {
            "rows": totals[segment],
            "accepted_rows": hits[segment],
            "acceptance_fraction": hits[segment] / totals[segment],
        }
        for segment in sorted(totals)
    }


def _report(summary: Mapping[str, Any]) -> str:
    audit = summary["support_audit"]
    return "\n".join(
        [
            "# Geometry19 V2.2 Stable-Root S0/S1 Support Audit",
            "",
            f"- Status: `{summary['status']}`",
            f"- Stable roots: {summary['stable_roots']}",
            f"- Stable development rows: {summary['stable_rows']}",
            f"- Always-unreliable roots: {summary['always_unreliable_roots']}",
            f"- Mixed roots: {summary['mixed_roots']}",
            f"- Anchored support gate: `{str(summary['anchored_support_gate_pass']).lower()}`",
            f"- Blind support gate: `{str(summary['blind_support_gate_pass']).lower()}`",
            f"- Anchored stable-row acceptance: {audit['anchored_stable_row_acceptance_fraction']:.3f}",
            f"- Blind stable-row acceptance: {audit['blind_stable_row_acceptance_fraction']:.3f}",
            f"- Always-unreliable false acceptance: {audit['always_unreliable_false_acceptance_fraction']:.3f}",
            f"- Nearest-center root identification accuracy: {audit['stable_nearest_center_root_accuracy']:.3f}",
            f"- Overlapping stable support spheres: {audit['overlapping_stable_sphere_pairs']}",
            f"- S2 authorized in known-anchor mode: `{str(summary['s2_anchored_mode_authorized']).lower()}`",
            f"- S2 blind-input mode authorized: `{str(summary['s2_blind_mode_authorized']).lower()}`",
            "- Fresh evaluation generated: `false`",
            "- Forward surrogate trained: `false`",
            "- Inverse model trained: `false`",
            "- Confirmation content parsed: `false`",
            "",
            "The 38-root set was selected after observing full-development Jacobian diagnostics. These rates are support-feasibility diagnostics, not fresh accuracy evidence. A new sealed V2.2 neighborhood evaluation is required before any accuracy claim.",
            "",
        ]
    )


def run_stable_root_support_audit(
    repo_root: Path,
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Run S0/S1 without generating data, fitting a surrogate, or parsing confirmation."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_protocol(config)

    resolved_inputs: dict[str, Path] = {}
    input_hash_audit: dict[str, Any] = {}
    for role, item in config["inputs"].items():
        path = (repo_root / str(item["path"])).resolve()
        _require(path.is_file(), f"Missing input: {role}")
        actual = sha256_file(path)
        expected = str(item["sha256"])
        _require(actual == expected, f"Input hash mismatch: {role}")
        resolved_inputs[role] = path
        input_hash_audit[role] = {
            "path": str(item["path"]),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": True,
            "content_parsed": role != "sealed_confirmation",
        }

    p2 = json.loads(resolved_inputs["p2_jacobian_summary"].read_text(encoding="utf-8"))
    p3 = json.loads(resolved_inputs["p3_chart_summary"].read_text(encoding="utf-8"))
    _require(p2.get("confirmation_content_read") is False, "P2 confirmation isolation failed")
    _require(p3.get("confirmation_content_read") is False, "P3 confirmation isolation failed")
    _require(p3.get("status") == config["inputs"]["p3_chart_summary"]["required_status"], "P3 status mismatch")
    _require(p3.get("p4_authorized") is False, "P3 unexpectedly authorizes P4")

    root_reliability = p2["root_reliability"]
    stable_ids = sorted(
        root_id
        for root_id, value in root_reliability.items()
        if int(value["training_fold_estimates"]) == 4
        and int(value["reliable_estimates"]) == int(value["training_fold_estimates"])
    )
    unreliable_ids = sorted(
        root_id
        for root_id, value in root_reliability.items()
        if int(value["training_fold_estimates"]) == 4
        and int(value["reliable_estimates"]) == 0
    )
    mixed_ids = sorted(set(root_reliability) - set(stable_ids) - set(unreliable_ids))
    selection = config["stable_root_selection"]
    _require(len(stable_ids) == int(selection["expected_stable_roots"]), "Stable-root count mismatch")
    _require(len(unreliable_ids) == int(selection["expected_always_unreliable_roots"]), "Unreliable-root count mismatch")
    _require(len(mixed_ids) == int(selection["expected_mixed_roots"]), "Mixed-root count mismatch")

    inventory_rows = _read_jsonl(resolved_inputs["root_inventory"])
    inventory = {str(row["root_seed_id"]): row for row in inventory_rows}
    _require(set(inventory) == set(root_reliability), "P1/P2 root sets differ")
    records = _read_jsonl(resolved_inputs["development_dataset"])
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_root[str(record["root_seed_id"])].append(record)
    _require(set(by_root) == set(inventory), "Dataset/inventory root sets differ")
    _require(sum(len(by_root[root_id]) for root_id in stable_ids) == int(selection["expected_stable_rows"]), "Stable-row count mismatch")

    schema = yaml.safe_load(resolved_inputs["design_schema"].read_text(encoding="utf-8"))
    names, lower, upper = _schema_arrays(schema)
    scale = upper - lower
    stable_centers = np.asarray(
        [
            _normalize_geometry(inventory[root_id]["design_center"], names, lower, scale)
            for root_id in stable_ids
        ]
    )

    stable_arrays: dict[str, np.ndarray] = {}
    radii: list[float] = []
    registry: list[dict[str, Any]] = []
    for stable_index, root_id in enumerate(stable_ids):
        root_records = by_root[root_id]
        values = np.asarray(
            [
                _normalize_geometry(record["geometry19"], names, lower, scale)
                for record in root_records
            ]
        )
        stable_arrays[root_id] = values
        distances = np.linalg.norm(values - stable_centers[stable_index], axis=1)
        radius = float(np.quantile(distances, 0.95))
        _require(np.isfinite(radius) and radius > 0.0, f"Invalid support radius: {root_id}")
        radii.append(radius)
        medoid_index = int(np.argmin(distances))
        performance = {
            name: [float(record["performance"][name]) for record in root_records]
            for name in ("thrust_N", "tsfc_kg_N_s", "mass_kg")
        }
        root_audit = inventory[root_id]
        registry.append(
            {
                "root_seed_id": root_id,
                "selection_rule": "P2 reliable_estimates=4/4",
                "reliable_estimates": 4,
                "training_fold_estimates": 4,
                "development_rows": len(root_records),
                "outer_fold_for_historical_audit_only": int(root_audit["outer_fold"]),
                "segments": root_audit["segments"],
                "segment_row_counts": root_audit["segment_row_counts"],
                "source": root_audit["source"],
                "root_origin_values": root_audit["root_origin_values"],
                "normalized_covariance_effective_rank": int(root_audit["normalized_covariance_effective_rank"]),
                "design_center": root_audit["design_center"],
                "normalized_design_center": stable_centers[stable_index].tolist(),
                "support_radius_p95_normalized_l2": radius,
                "anchored_development_acceptance_fraction": float(np.mean(distances <= radius)),
                "medoid": {
                    "sample_id": root_records[medoid_index]["sample_id"],
                    "design_hash": root_records[medoid_index]["design_hash"],
                    "distance_to_center_normalized_l2": float(distances[medoid_index]),
                },
                "performance_envelope_for_interpretation_only": {
                    name: {"min": min(values_), "max": max(values_)}
                    for name, values_ in performance.items()
                },
            }
        )
    radius_array = np.asarray(radii, dtype=np.float64)

    def classify(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(
            [_normalize_geometry(row["geometry19"], names, lower, scale) for row in rows]
        )
        distances = np.linalg.norm(values[:, None, :] - stable_centers[None, :, :], axis=2)
        normalized_margin = distances / radius_array[None, :]
        nearest = np.argmin(distances, axis=1)
        accepted = np.min(normalized_margin, axis=1) <= 1.0
        return accepted, nearest, np.min(normalized_margin, axis=1)

    stable_records = [record for root_id in stable_ids for record in by_root[root_id]]
    unreliable_records = [record for root_id in unreliable_ids for record in by_root[root_id]]
    mixed_records = [record for root_id in mixed_ids for record in by_root[root_id]]
    stable_accept, stable_nearest, stable_margin = classify(stable_records)
    unreliable_accept, _, unreliable_margin = classify(unreliable_records)
    mixed_accept, _, mixed_margin = classify(mixed_records)
    actual_stable_index = np.asarray(
        [stable_ids.index(str(record["root_seed_id"])) for record in stable_records]
    )
    anchored_accept = np.asarray(
        [
            np.linalg.norm(
                _normalize_geometry(record["geometry19"], names, lower, scale)
                - stable_centers[stable_ids.index(str(record["root_seed_id"]))]
            )
            <= radius_array[stable_ids.index(str(record["root_seed_id"]))]
            for record in stable_records
        ],
        dtype=np.bool_,
    )

    center_distances = np.linalg.norm(
        stable_centers[:, None, :] - stable_centers[None, :, :], axis=2
    )
    overlaps: list[dict[str, Any]] = []
    for left in range(len(stable_ids)):
        for right in range(left + 1, len(stable_ids)):
            if center_distances[left, right] <= radius_array[left] + radius_array[right]:
                overlaps.append(
                    {
                        "left_root_seed_id": stable_ids[left],
                        "right_root_seed_id": stable_ids[right],
                        "center_distance": float(center_distances[left, right]),
                        "sum_support_radii": float(radius_array[left] + radius_array[right]),
                    }
                )

    support_config = config["support_audit"]
    minimum_rows = int(support_config["minimum_rows_per_stable_root"])
    minimum_rank = int(support_config["minimum_effective_rank_per_stable_root"])
    anchored_gate = all(
        item["development_rows"] >= minimum_rows
        and item["normalized_covariance_effective_rank"] >= minimum_rank
        for item in registry
    ) and float(np.mean(anchored_accept)) >= float(support_config["stable_acceptance_minimum"])
    blind_gate = (
        anchored_gate
        and float(np.mean(stable_accept)) >= float(support_config["stable_acceptance_minimum"])
        and float(np.mean(unreliable_accept))
        <= float(support_config["always_unreliable_false_acceptance_maximum"])
    )
    status = PASS_BOTH if blind_gate else PASS_ANCHORED if anchored_gate else AUDIT_FAIL

    runtime_root = (
        output_root.resolve()
        if output_root is not None
        else (repo_root / str(config["artifacts"]["runtime_root"])).resolve()
    )
    input_lock_path = runtime_root / str(config["artifacts"]["input_lock"])
    registry_path = runtime_root / str(config["artifacts"]["stable_root_registry"])
    audit_path = runtime_root / str(config["artifacts"]["support_audit"])
    summary_path = runtime_root / str(config["artifacts"]["summary"])
    report_path = runtime_root / str(config["artifacts"]["report"])
    implementation_path = Path(__file__).resolve()

    input_lock = {
        "status": "STABLE_ROOT_S0_INPUT_LOCK_PASS",
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_path": _artifact_path(config_path, repo_root),
        "protocol_sha256": sha256_file(config_path),
        "input_hash_audit": input_hash_audit,
        "all_input_hashes_match": True,
        "confirmation_hash_checked": True,
        "confirmation_content_read": False,
        "fresh_data_generated": False,
        "model_training_executed": False,
    }
    support_audit = {
        "protocol_version": PROTOCOL_VERSION,
        "method": config["support_audit"],
        "development_diagnostic_only": True,
        "anchored_stable_row_acceptance_fraction": float(np.mean(anchored_accept)),
        "blind_stable_row_acceptance_fraction": float(np.mean(stable_accept)),
        "always_unreliable_false_acceptance_fraction": float(np.mean(unreliable_accept)),
        "mixed_root_acceptance_fraction_diagnostic_only": float(np.mean(mixed_accept)),
        "stable_nearest_center_root_accuracy": float(np.mean(stable_nearest == actual_stable_index)),
        "stable_normalized_margin_p95": float(np.quantile(stable_margin, 0.95)),
        "always_unreliable_normalized_margin_p05": float(np.quantile(unreliable_margin, 0.05)),
        "mixed_normalized_margin_p05": float(np.quantile(mixed_margin, 0.05)),
        "anchored_acceptance_by_segment": _rates_by_segment(stable_records, anchored_accept),
        "blind_stable_acceptance_by_segment": _rates_by_segment(stable_records, stable_accept),
        "always_unreliable_false_acceptance_by_segment": _rates_by_segment(unreliable_records, unreliable_accept),
        "overlapping_stable_sphere_pairs": len(overlaps),
        "stable_sphere_overlaps": overlaps,
        "anchored_support_gate_pass": anchored_gate,
        "blind_support_gate_pass": blind_gate,
    }
    _write_json(input_lock_path, input_lock)
    _write_jsonl(registry_path, registry)
    _write_json(audit_path, support_audit)

    root_segment_counts = Counter(
        segment for root_id in stable_ids for segment in inventory[root_id]["segments"]
    )
    source_counts = Counter(inventory[root_id]["source"] for root_id in stable_ids)
    historical_fold_counts = Counter(int(inventory[root_id]["outer_fold"]) for root_id in stable_ids)
    summary: dict[str, Any] = {
        "status": status,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_path": _artifact_path(config_path, repo_root),
        "protocol_sha256": sha256_file(config_path),
        "implementation_path": _artifact_path(implementation_path, repo_root),
        "implementation_sha256": sha256_file(implementation_path),
        "stable_roots": len(stable_ids),
        "stable_rows": len(stable_records),
        "always_unreliable_roots": len(unreliable_ids),
        "mixed_roots": len(mixed_ids),
        "stable_root_ids": stable_ids,
        "always_unreliable_root_ids": unreliable_ids,
        "mixed_root_ids": mixed_ids,
        "stable_root_segment_memberships": dict(sorted(root_segment_counts.items())),
        "stable_root_sources": dict(sorted(source_counts.items())),
        "stable_root_historical_outer_folds": {
            str(key): value for key, value in sorted(historical_fold_counts.items())
        },
        "selection_bias_acknowledged": True,
        "fresh_evaluation_required_before_accuracy_claim": True,
        "support_audit": support_audit,
        "anchored_support_gate_pass": anchored_gate,
        "blind_support_gate_pass": blind_gate,
        "s2_anchored_mode_authorized": anchored_gate,
        "s2_blind_mode_authorized": blind_gate,
        "original_p4_authorized": False,
        "fresh_data_generated": False,
        "forward_surrogate_training_executed": False,
        "inverse_model_training_executed": False,
        "confirmation_hash_checked": True,
        "confirmation_content_read": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "next_authorized_stage": "S2_known_anchor_forward_model_protocol_and_S3_fresh_evaluation_protocol"
        if anchored_gate
        else "stable_root_data_and_rank_remediation",
        "artifacts": {
            "input_lock": _artifact_path(input_lock_path, repo_root),
            "input_lock_sha256": sha256_file(input_lock_path),
            "stable_root_registry": _artifact_path(registry_path, repo_root),
            "stable_root_registry_sha256": sha256_file(registry_path),
            "support_audit": _artifact_path(audit_path, repo_root),
            "support_audit_sha256": sha256_file(audit_path),
        },
    }
    _write_text(report_path, _report(summary))
    summary["artifacts"]["report"] = _artifact_path(report_path, repo_root)
    summary["artifacts"]["report_sha256"] = sha256_file(report_path)
    _write_json(summary_path, summary)
    return summary
