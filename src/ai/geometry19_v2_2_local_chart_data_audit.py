"""P1 data, root-lineage, and fold-integrity audit for local-chart MoE."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_local_chart_protocol import (
    Geometry19LocalChartProtocolError,
    P0_PASS_STATUS,
    PROTOCOL_VERSION,
    validate_protocol_schema,
)


P1_PASS_STATUS = "LOCAL_CHART_MOE_P1_DATA_ROOT_FOLD_AUDIT_PASS"
NEW_ROOT_ORIGIN = "independent_global_root_center_pca_lhs"
PERFORMANCE_NAMES = ("thrust_N", "tsfc_kg_N_s", "mass_kg")


class Geometry19LocalChartDataAuditError(ValueError):
    """Raised when the frozen development data violates the P1 contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Geometry19LocalChartDataAuditError(message)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _effective_covariance_rank(values: np.ndarray) -> tuple[int, float, list[float]]:
    """Return standard numerical rank and covariance spectrum for centered values."""

    centered = values - np.mean(values, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    largest = float(singular_values[0]) if len(singular_values) else 0.0
    tolerance = float(max(centered.shape) * np.finfo(np.float64).eps * largest)
    rank = int(np.sum(singular_values > tolerance))
    if len(values) > 1:
        covariance = centered.T @ centered / float(len(values) - 1)
        eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    else:
        eigenvalues = np.zeros(values.shape[1], dtype=np.float64)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return rank, tolerance, [float(value) for value in eigenvalues]


def _source_for_records(records: Sequence[Mapping[str, Any]]) -> tuple[str, list[str]]:
    origins = sorted({str(row.get("root_origin", "<missing>")) for row in records})
    _require(len(origins) == 1, "A root contains mixed lineage origins")
    if origins == [NEW_ROOT_ORIGIN]:
        return "known_manifold_root_discovery_v1_new", origins
    _require(origins == ["<missing>"], f"Unexpected root origin: {origins}")
    return "pre_root_discovery_development", origins


def _validate_rows(
    records: Sequence[Mapping[str, Any]], variables: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    variable_names = [str(value["name"]) for value in variables]
    design_hashes: set[str] = set()
    root_ids: set[str] = set()
    physics_versions: Counter[str] = Counter()
    target_domain_hashes: Counter[str] = Counter()
    for index, row in enumerate(records):
        _require(row.get("training_candidate") is True, f"Row {index} is not a training candidate")
        constraint = row.get("constraint_assessment", {})
        _require(constraint.get("physics_domain_valid") is True, f"Row {index} is not physics-domain valid")
        design_hash = str(row.get("design_hash", ""))
        _require(bool(design_hash), f"Row {index} has no design hash")
        _require(design_hash not in design_hashes, f"Duplicate design hash: {design_hash}")
        design_hashes.add(design_hash)
        root_id = str(row.get("root_seed_id", ""))
        _require(bool(root_id), f"Row {index} has no root seed ID")
        root_ids.add(root_id)

        geometry = row.get("geometry19", {})
        _require(set(geometry) == set(variable_names), f"Row {index} does not contain the frozen 19D schema")
        for variable in variables:
            name = str(variable["name"])
            value = float(geometry[name])
            lower, upper = (float(bound) for bound in variable["bounds"])
            _require(math.isfinite(value), f"Row {index} has non-finite design value: {name}")
            _require(lower <= value <= upper, f"Row {index} is outside schema bounds: {name}")

        performance = row.get("performance", {})
        for name in PERFORMANCE_NAMES:
            value = float(performance[name])
            _require(math.isfinite(value) and value > 0.0, f"Row {index} has invalid performance: {name}")
        physics_versions[str(row.get("physics_model_version", "<missing>"))] += 1
        target_domain_hashes[str(row.get("target_domain_hash", "<missing>"))] += 1

    return {
        "rows": len(records),
        "unique_design_hashes": len(design_hashes),
        "global_roots": len(root_ids),
        "all_training_candidate": True,
        "all_physics_domain_valid": True,
        "all_performance_finite_positive": True,
        "all_design_hashes_unique": True,
        "all_geometry_complete_finite_in_schema_bounds": True,
        "physics_model_versions": dict(sorted(physics_versions.items())),
        "target_domain_hashes": dict(sorted(target_domain_hashes.items())),
    }


def _root_inventory(
    records: Sequence[Mapping[str, Any]],
    variables: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, int],
    minimum_rank: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    variable_names = [str(value["name"]) for value in variables]
    lower = np.asarray([float(value["bounds"][0]) for value in variables])
    scale = np.asarray([float(value["bounds"][1]) - float(value["bounds"][0]) for value in variables])
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["root_seed_id"])].append(row)

    inventory: list[dict[str, Any]] = []
    source_roots: dict[str, set[str]] = defaultdict(set)
    insufficient: list[str] = []
    cross_segment: list[str] = []
    rank_counts: Counter[int] = Counter()
    for root_id in sorted(grouped):
        root_records = grouped[root_id]
        source, origins = _source_for_records(root_records)
        source_roots[source].add(root_id)
        designs = np.asarray(
            [[float(row["geometry19"][name]) for name in variable_names] for row in root_records],
            dtype=np.float64,
        )
        normalized = (designs - lower[None, :]) / scale[None, :]
        rank, tolerance, eigenvalues = _effective_covariance_rank(normalized)
        rank_counts[rank] += 1
        sufficient = rank >= minimum_rank
        if not sufficient:
            insufficient.append(root_id)
        segments = sorted({str(row["assigned_segment"]) for row in root_records})
        if len(segments) > 1:
            cross_segment.append(root_id)
        performances = np.asarray(
            [[float(row["performance"][name]) for name in PERFORMANCE_NAMES] for row in root_records],
            dtype=np.float64,
        )
        inventory.append(
            {
                "root_seed_id": root_id,
                "outer_fold": int(assignments[root_id]),
                "source": source,
                "root_origin_values": origins,
                "rows": len(root_records),
                "segments": segments,
                "segment_row_counts": dict(sorted(Counter(str(row["assigned_segment"]) for row in root_records).items())),
                "design_center": {name: float(value) for name, value in zip(variable_names, np.mean(designs, axis=0))},
                "performance_center": {name: float(value) for name, value in zip(PERFORMANCE_NAMES, np.mean(performances, axis=0))},
                "normalized_covariance_effective_rank": rank,
                "normalized_covariance_rank_tolerance": tolerance,
                "normalized_covariance_trace": float(sum(eigenvalues)),
                "normalized_covariance_eigenvalues": eigenvalues,
                "local_variation_dimension_sufficient_for_p2": sufficient,
            }
        )

    new_roots = source_roots["known_manifold_root_discovery_v1_new"]
    existing_roots = source_roots["pre_root_discovery_development"]
    overlap = sorted(new_roots & existing_roots)
    _require(not overlap, "New and existing root IDs overlap")
    _require((new_roots | existing_roots) == set(grouped), "Root lineage does not cover all roots")
    return inventory, {
        "global_roots": len(grouped),
        "new_roots": len(new_roots),
        "existing_roots": len(existing_roots),
        "new_existing_root_overlap": overlap,
        "lineage_partition_complete": True,
        "cross_segment_roots": cross_segment,
        "cross_segment_root_count": len(cross_segment),
        "minimum_effective_rank_for_p2": minimum_rank,
        "rank_distribution": {str(key): value for key, value in sorted(rank_counts.items())},
        "insufficient_local_variation_roots": insufficient,
        "insufficient_local_variation_root_count": len(insufficient),
        "sufficient_local_variation_root_count": len(grouped) - len(insufficient),
    }


def _split_summary(
    records: Sequence[Mapping[str, Any]],
    selected_roots: set[str],
) -> dict[str, Any]:
    selected = [row for row in records if str(row["root_seed_id"]) in selected_roots]
    pairs = {(str(row["assigned_segment"]), str(row["root_seed_id"])) for row in selected}
    by_segment: dict[str, Any] = {}
    for segment in sorted({str(row["assigned_segment"]) for row in selected}):
        segment_rows = [row for row in selected if str(row["assigned_segment"]) == segment]
        by_segment[segment] = {
            "rows": len(segment_rows),
            "roots": len({str(row["root_seed_id"]) for row in segment_rows}),
        }
    return {
        "rows": len(selected),
        "global_roots": len(selected_roots),
        "segment_root_pairs": len(pairs),
        "by_segment": by_segment,
    }


def _fold_inventory(
    records: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, int],
    target_segments: Sequence[str],
    fold_count: int,
    expected_target_roots: int,
) -> dict[str, Any]:
    all_roots = set(assignments)
    folds: list[dict[str, Any]] = []
    for fold in range(fold_count):
        validation_roots = {root_id for root_id, value in assignments.items() if value == fold}
        train_roots = all_roots - validation_roots
        _require(not (train_roots & validation_roots), f"Root leakage in outer fold {fold}")
        validation = _split_summary(records, validation_roots)
        train = _split_summary(records, train_roots)
        validation_target_roots = {
            segment: validation["by_segment"].get(segment, {}).get("roots", 0)
            for segment in target_segments
        }
        _require(
            all(value == expected_target_roots for value in validation_target_roots.values()),
            f"Outer fold {fold} target-root count changed",
        )
        folds.append(
            {
                "fold": fold,
                "train": train,
                "validation": validation,
                "validation_roots_by_target_segment": validation_target_roots,
                "root_leakage_count": 0,
            }
        )
    return {
        "fold_count": fold_count,
        "grouping_key": "root_seed_id",
        "all_roots_assigned_once": True,
        "cross_fold_root_leakage": False,
        "expected_validation_roots_per_target_segment_per_fold": expected_target_roots,
        "folds": folds,
    }


def _report(summary: Mapping[str, Any]) -> str:
    rows = summary["row_contract"]
    roots = summary["root_contract"]
    lines = [
        "# Geometry19 V2.2 Local-Chart MoE P1 Data/Root/Fold Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Development rows / roots: {rows['rows']} / {rows['global_roots']}",
        f"- Existing / new roots: {roots['existing_roots']} / {roots['new_roots']}",
        f"- Cross-segment roots: {roots['cross_segment_root_count']}",
        f"- P2 rank-sufficient / insufficient roots: {roots['sufficient_local_variation_root_count']} / {roots['insufficient_local_variation_root_count']}",
        "- Confirmation content parsed: `false`",
        "- Formal/deployment use allowed: `false` / `false`",
        "",
        "## Outer fold inventory",
        "",
        "| Fold | Train rows/roots | Validation rows/roots | Validation segment-root pairs | Target roots |",
        "|---:|---:|---:|---:|---|",
    ]
    for fold in summary["fold_contract"]["folds"]:
        target = "/".join(str(fold["validation_roots_by_target_segment"][segment]) for segment in summary["target_segments"])
        lines.append(
            f"| {fold['fold']} | {fold['train']['rows']}/{fold['train']['global_roots']} | "
            f"{fold['validation']['rows']}/{fold['validation']['global_roots']} | "
            f"{fold['validation']['segment_root_pairs']} | {target} |"
        )
    lines.extend(
        [
            "",
            "P1 establishes data and grouping integrity only. Rank-insufficient roots remain in the dataset and are not discarded. No Jacobian, chart, router, expert, or OOF model evaluation is performed.",
            "",
        ]
    )
    return "\n".join(lines)


def run_data_root_fold_audit(
    repo_root: Path,
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Execute P1 after verifying the immutable P0 protocol and inputs."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        validate_protocol_schema(config)
    except Geometry19LocalChartProtocolError as error:
        raise Geometry19LocalChartDataAuditError(str(error)) from error

    resolved: dict[str, Path] = {}
    for role, item in config["inputs"].items():
        path = (repo_root / str(item["path"])).resolve()
        _require(path.is_file(), f"Frozen input is missing: {role}")
        _require(sha256_file(path) == str(item["sha256"]), f"Frozen input hash mismatch: {role}")
        resolved[role] = path

    runtime_root = (
        output_root.resolve()
        if output_root is not None
        else (repo_root / str(config["artifacts"]["runtime_root"])).resolve()
    )
    p0_path = runtime_root / "audits/protocol_input_lock.json"
    _require(p0_path.is_file(), "P0 protocol input lock evidence is missing")
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    _require(p0.get("status") == P0_PASS_STATUS, "P0 protocol input lock did not pass")
    _require(p0.get("protocol_sha256") == sha256_file(config_path), "P0 protocol hash no longer matches")
    _require(p0.get("confirmation_content_read") is False, "P0 confirmation isolation was violated")

    records = _read_jsonl(resolved["development_dataset"])
    schema = yaml.safe_load(resolved["design_schema"].read_text(encoding="utf-8"))
    variables = schema["design_variables"]
    _require(len(variables) == 19, "Frozen design schema is not 19D")
    folds = json.loads(resolved["development_folds"].read_text(encoding="utf-8"))
    assignments = {str(key): int(value) for key, value in folds["root_assignments"].items()}

    row_contract = _validate_rows(records, variables)
    expected_rows = int(config["inputs"]["development_dataset"]["expected_rows"])
    expected_roots = int(config["inputs"]["development_dataset"]["expected_global_roots"])
    _require(row_contract["rows"] == expected_rows, "P1 development row count changed")
    _require(row_contract["global_roots"] == expected_roots, "P1 development root count changed")
    _require(set(assignments) == {str(row["root_seed_id"]) for row in records}, "P1 fold roots differ from data roots")

    minimum_rank = int(config["local_jacobian"]["reliability_gate"]["minimum_effective_rank"])
    root_inventory, root_contract = _root_inventory(records, variables, assignments, minimum_rank)
    fold_contract = _fold_inventory(
        records,
        assignments,
        config["target_segments"],
        int(config["outer_evaluation"]["fold_count"]),
        int(config["outer_evaluation"]["expected_validation_roots_per_target_segment_per_fold"]),
    )

    root_path = runtime_root / "audits/root_inventory.jsonl"
    fold_path = runtime_root / "audits/fold_inventory.json"
    report_path = runtime_root / "audits/P1_Data_Root_Fold_Audit_Report.md"
    _write_jsonl(root_path, root_inventory)
    _write_json(fold_path, fold_contract)
    implementation_path = Path(__file__).resolve()
    summary: dict[str, Any] = {
        "status": P1_PASS_STATUS,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(config_path),
        "p0_input_lock_sha256": sha256_file(p0_path),
        "implementation_path": str(implementation_path.relative_to(repo_root)),
        "implementation_sha256": sha256_file(implementation_path),
        "target_segments": list(config["target_segments"]),
        "row_contract": row_contract,
        "root_contract": root_contract,
        "fold_contract": fold_contract,
        "confirmation_hash_checked": True,
        "confirmation_content_read": False,
        "confirmation_reuse_allowed": False,
        "model_work_executed": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "artifacts": {
            "root_inventory": str(root_path.relative_to(repo_root)) if root_path.is_relative_to(repo_root) else str(root_path),
            "root_inventory_sha256": sha256_file(root_path),
            "fold_inventory": str(fold_path.relative_to(repo_root)) if fold_path.is_relative_to(repo_root) else str(fold_path),
            "fold_inventory_sha256": sha256_file(fold_path),
            "report": str(report_path.relative_to(repo_root)) if report_path.is_relative_to(repo_root) else str(report_path),
        },
        "next_authorized_stage": "P2_training_fold_local_jacobian_and_stability_estimation",
    }
    _write_text(report_path, _report(summary))
    summary["artifacts"]["report_sha256"] = sha256_file(report_path)
    input_audit_path = runtime_root / "audits/input_audit.json"
    _write_json(input_audit_path, summary)
    return summary
