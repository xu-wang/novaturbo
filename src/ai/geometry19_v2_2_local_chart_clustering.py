"""Reliability-aware P3 chart identification for Geometry19 local-chart MoE."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, rand_score
import yaml

from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_local_chart_features import P2_PASS_STATUS


P3_PROTOCOL_VERSION = "geometry19_v2_2_local_chart_clustering_v1"
RESPONSE_PASS = "LOCAL_CHART_MOE_P3_RESPONSE_CHART_PASS"
DESIGN_ONLY = "LOCAL_CHART_MOE_P3_DESIGN_ONLY_STRUCTURE"
UNSTABLE = "LOCAL_CHART_MOE_P3_CHART_IDENTIFICATION_UNSTABLE"
INFEASIBLE = "K_INFEASIBLE_FOR_RELIABLE_CORE"


class Geometry19LocalChartClusteringError(ValueError):
    """Raised when P3 inputs, isolation, or clustering contracts are violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Geometry19LocalChartClusteringError(message)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def validate_p3_protocol(config: Mapping[str, Any]) -> None:
    _require(config.get("protocol_version") == P3_PROTOCOL_VERSION, "P3 protocol version mismatch")
    _require(config.get("stage") == "P3_reliability_aware_chart_identification", "P3 stage mismatch")
    _require(config.get("formal_use_allowed") is False, "Formal use must remain forbidden")
    _require(config.get("deployment_allowed") is False, "Deployment must remain forbidden")
    _require(config.get("candidate_k") == [3, 4, 6, 8], "P3 candidate K changed")
    _require(config.get("views") == ["design_only", "response_aware"], "P3 views changed")
    core = config.get("core_clustering", {})
    _require(core.get("reliable_roots_only") is True, "Core charts must use reliable roots only")
    _require(core.get("minimum_final_charts") == 2, "Minimum final charts changed")
    _require(core.get("minimum_reliable_core_roots_per_chart") == 4, "Minimum core roots changed")
    attachment = config.get("unreliable_root_attachment", {})
    _require(attachment.get("may_change_core_linkage") is False, "Attached roots cannot shape core linkage")
    _require(attachment.get("jacobian_distance_weight") == 0.0, "Unreliable Jacobian weight must be zero")
    _require(attachment.get("minimum_total_roots_per_chart") == 5, "Minimum total roots changed")
    selection = config.get("selection", {})
    _require(selection.get("p3_selects_final_k") is False, "P3 cannot select final K")
    _require(selection.get("outer_validation_labels_used") is False, "Outer validation labels are forbidden")
    _require(selection.get("performance_centers_used_for_clustering") is False, "Performance centers cannot drive clustering")
    _require(selection.get("cross_fold_categories_used_as_features") is False, "Cross-fold reliability categories cannot be features")


def _pairwise_euclidean(values: np.ndarray) -> np.ndarray:
    differences = values[:, None, :] - values[None, :, :]
    return np.linalg.norm(differences, axis=-1)


def _pairwise_frobenius(values: np.ndarray) -> np.ndarray:
    differences = values[:, None, :, :] - values[None, :, :, :]
    return np.linalg.norm(differences, axis=(-2, -1))


def _nonzero_pairwise_median(distance: np.ndarray) -> float:
    upper = distance[np.triu_indices(len(distance), k=1)]
    nonzero = upper[upper > 0.0]
    _require(len(nonzero) > 0, "Pairwise distance has no nonzero scale")
    return float(np.median(nonzero))


def _canonical_labels(labels: np.ndarray, root_ids: Sequence[str]) -> np.ndarray:
    groups: dict[int, list[str]] = defaultdict(list)
    for label, root_id in zip(labels.tolist(), root_ids):
        groups[int(label)].append(str(root_id))
    order = sorted(groups, key=lambda label: min(groups[label]))
    mapping = {old: new for new, old in enumerate(order)}
    return np.asarray([mapping[int(value)] for value in labels], dtype=np.int64)


def _mean_between(distance: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(distance[np.ix_(left, right)]))


def _merge_undersized(
    labels: np.ndarray,
    distance: np.ndarray,
    root_ids: Sequence[str],
    *,
    minimum_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels = _canonical_labels(labels, root_ids)
    merges: list[dict[str, Any]] = []
    while True:
        unique = sorted(set(labels.tolist()))
        groups = {label: np.flatnonzero(labels == label) for label in unique}
        small = [label for label in unique if len(groups[label]) < minimum_size]
        if not small or len(unique) == 1:
            break
        source = min(
            small,
            key=lambda label: (len(groups[label]), min(root_ids[index] for index in groups[label])),
        )
        candidates = [label for label in unique if label != source]
        target = min(
            candidates,
            key=lambda label: (
                _mean_between(distance, groups[source], groups[label]),
                min(root_ids[index] for index in groups[label]),
            ),
        )
        merges.append(
            {
                "source_min_root_id": min(root_ids[index] for index in groups[source]),
                "source_size": len(groups[source]),
                "target_min_root_id": min(root_ids[index] for index in groups[target]),
                "target_size_before": len(groups[target]),
                "distance": _mean_between(distance, groups[source], groups[target]),
            }
        )
        labels[labels == source] = target
        labels = _canonical_labels(labels, root_ids)
    return labels, merges


def _cluster_candidate(
    root_ids: Sequence[str],
    reliable: np.ndarray,
    design_distance: np.ndarray,
    core_distance: np.ndarray,
    *,
    requested_k: int,
    minimum_final_charts: int,
    minimum_core: int,
    minimum_total: int,
) -> dict[str, Any]:
    core_indices = np.flatnonzero(reliable)
    maximum_feasible_k = len(core_indices) // minimum_core
    if requested_k > maximum_feasible_k:
        return {
            "status": INFEASIBLE,
            "requested_k": requested_k,
            "maximum_feasible_k": maximum_feasible_k,
            "reliable_core_roots": len(core_indices),
            "structural_gate_pass": False,
        }
    core_ids = [root_ids[index] for index in core_indices]
    active_distance = core_distance[np.ix_(core_indices, core_indices)]
    initial = AgglomerativeClustering(
        n_clusters=requested_k,
        metric="precomputed",
        linkage="average",
    ).fit_predict(active_distance)
    core_labels, core_merges = _merge_undersized(
        initial.astype(np.int64),
        active_distance,
        core_ids,
        minimum_size=minimum_core,
    )

    labels = np.full(len(root_ids), -1, dtype=np.int64)
    labels[core_indices] = core_labels
    for index in np.flatnonzero(~reliable):
        choices = sorted(set(core_labels.tolist()))
        selected = min(
            choices,
            key=lambda label: (
                float(
                    np.mean(
                        design_distance[
                            index,
                            core_indices[np.flatnonzero(core_labels == label)],
                        ]
                    )
                ),
                min(
                    core_ids[position]
                    for position in np.flatnonzero(core_labels == label)
                ),
            ),
        )
        labels[index] = selected
    labels = _canonical_labels(labels, root_ids)
    labels, total_merges = _merge_undersized(
        labels,
        design_distance,
        root_ids,
        minimum_size=minimum_total,
    )

    chart_counts: dict[str, Any] = {}
    for chart in sorted(set(labels.tolist())):
        members = np.flatnonzero(labels == chart)
        chart_counts[str(chart)] = {
            "total_roots": len(members),
            "reliable_core_roots": int(np.sum(reliable[members])),
            "attached_unreliable_roots": int(np.sum(~reliable[members])),
        }
    structural_pass = len(chart_counts) >= minimum_final_charts and all(
        item["total_roots"] >= minimum_total
        and item["reliable_core_roots"] >= minimum_core
        for item in chart_counts.values()
    )
    return {
        "status": "CANDIDATE_CLUSTERED" if structural_pass else "CANDIDATE_STRUCTURE_FAIL",
        "requested_k": requested_k,
        "maximum_feasible_k": maximum_feasible_k,
        "initial_k": requested_k,
        "final_k": len(chart_counts),
        "reliable_core_roots": len(core_indices),
        "attached_unreliable_roots": int(np.sum(~reliable)),
        "core_merges": core_merges,
        "total_size_merges": total_merges,
        "chart_counts": chart_counts,
        "structural_gate_pass": structural_pass,
        "labels": labels,
    }


def _coherence(labels: np.ndarray, jacobians: np.ndarray, reliable: np.ndarray) -> dict[str, Any]:
    indices = np.flatnonzero(reliable)
    within: list[float] = []
    between: list[float] = []
    for left, right in combinations(indices.tolist(), 2):
        distance = float(np.linalg.norm(jacobians[left, 1] - jacobians[right, 1]))
        (within if labels[left] == labels[right] else between).append(distance)
    within_median = float(np.median(within)) if within else None
    between_median = float(np.median(between)) if between else None
    ratio = (
        float(within_median / between_median)
        if within_median is not None and between_median is not None and between_median > 0.0
        else None
    )
    return {
        "reliable_pair_count": len(within) + len(between),
        "within_pair_count": len(within),
        "between_pair_count": len(between),
        "within_tsfc_jacobian_distance_median": within_median,
        "between_tsfc_jacobian_distance_median": between_median,
        "within_between_ratio": ratio,
    }


def _chart_details(
    labels: np.ndarray,
    root_ids: Sequence[str],
    reliable: np.ndarray,
    design_centers: np.ndarray,
    performance_centers: np.ndarray,
    root_inventory: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for chart in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == chart)
        members = [root_ids[index] for index in indices]
        segments = Counter(
            segment for root_id in members for segment in root_inventory[root_id]["segments"]
        )
        details.append(
            {
                "chart_id": int(chart),
                "root_ids": members,
                "total_roots": len(members),
                "reliable_core_root_ids": [
                    root_ids[index] for index in indices if reliable[index]
                ],
                "attached_unreliable_root_ids": [
                    root_ids[index] for index in indices if not reliable[index]
                ],
                "segment_root_memberships": dict(sorted(segments.items())),
                "normalized_design_center": np.mean(design_centers[indices], axis=0).tolist(),
                "normalized_design_min": np.min(design_centers[indices], axis=0).tolist(),
                "normalized_design_max": np.max(design_centers[indices], axis=0).tolist(),
                "performance_center_for_interpretation_only": {
                    name: float(value)
                    for name, value in zip(
                        ("thrust_N", "tsfc_kg_N_s", "mass_kg"),
                        np.mean(performance_centers[indices], axis=0),
                    )
                },
                "performance_min_for_interpretation_only": {
                    name: float(value)
                    for name, value in zip(
                        ("thrust_N", "tsfc_kg_N_s", "mass_kg"),
                        np.min(performance_centers[indices], axis=0),
                    )
                },
                "performance_max_for_interpretation_only": {
                    name: float(value)
                    for name, value in zip(
                        ("thrust_N", "tsfc_kg_N_s", "mass_kg"),
                        np.max(performance_centers[indices], axis=0),
                    )
                },
            }
        )
    return details


def _stability(
    fold_candidates: Mapping[int, Mapping[str, Any]],
    root_inventory: Mapping[str, Mapping[str, Any]],
    target_segments: Sequence[str],
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    root_neighborhood_switches: dict[str, list[float]] = defaultdict(list)
    for left_fold, right_fold in combinations(sorted(fold_candidates), 2):
        left = fold_candidates[left_fold]
        right = fold_candidates[right_fold]
        if not left.get("structural_gate_pass") or not right.get("structural_gate_pass"):
            continue
        left_assignments = left["assignments"]
        right_assignments = right["assignments"]
        shared = sorted(set(left_assignments) & set(right_assignments))
        left_labels = [left_assignments[root_id] for root_id in shared]
        right_labels = [right_assignments[root_id] for root_id in shared]
        by_segment: dict[str, Any] = {}
        for segment in target_segments:
            members = [
                root_id for root_id in shared if segment in root_inventory[root_id]["segments"]
            ]
            by_segment[segment] = {
                "shared_roots": len(members),
                "adjusted_rand_index": float(
                    adjusted_rand_score(
                        [left_assignments[root_id] for root_id in members],
                        [right_assignments[root_id] for root_id in members],
                    )
                )
                if len(members) >= 2
                else None,
                "co_clustering_agreement": float(
                    rand_score(
                        [left_assignments[root_id] for root_id in members],
                        [right_assignments[root_id] for root_id in members],
                    )
                )
                if len(members) >= 2
                else None,
            }
        comparisons.append(
            {
                "left_fold": left_fold,
                "right_fold": right_fold,
                "shared_roots": len(shared),
                "adjusted_rand_index": float(adjusted_rand_score(left_labels, right_labels)),
                "co_clustering_agreement": float(rand_score(left_labels, right_labels)),
                "target_segments": by_segment,
            }
        )
        for root_id in shared:
            peers = [value for value in shared if value != root_id]
            left_neighbors = {
                value
                for value in peers
                if left_assignments[value] == left_assignments[root_id]
            }
            right_neighbors = {
                value
                for value in peers
                if right_assignments[value] == right_assignments[root_id]
            }
            union = left_neighbors | right_neighbors
            distance = 0.0 if not union else 1.0 - len(left_neighbors & right_neighbors) / len(union)
            root_neighborhood_switches[root_id].append(float(distance))

    overall_ari = [value["adjusted_rand_index"] for value in comparisons]
    overall_rand = [value["co_clustering_agreement"] for value in comparisons]
    target_summary: dict[str, Any] = {}
    for segment in target_segments:
        aris = [
            value["target_segments"][segment]["adjusted_rand_index"]
            for value in comparisons
            if value["target_segments"][segment]["adjusted_rand_index"] is not None
        ]
        agreements = [
            value["target_segments"][segment]["co_clustering_agreement"]
            for value in comparisons
            if value["target_segments"][segment]["co_clustering_agreement"] is not None
        ]
        target_summary[segment] = {
            "median_adjusted_rand_index": float(np.median(aris)) if aris else None,
            "minimum_adjusted_rand_index": float(min(aris)) if aris else None,
            "median_co_clustering_agreement": float(np.median(agreements))
            if agreements
            else None,
        }
    root_switch = {
        root_id: float(np.mean(values))
        for root_id, values in sorted(root_neighborhood_switches.items())
    }
    switch_values = list(root_switch.values())
    return {
        "fold_pair_comparisons": comparisons,
        "comparison_count": len(comparisons),
        "median_adjusted_rand_index": float(np.median(overall_ari)) if overall_ari else None,
        "minimum_adjusted_rand_index": float(min(overall_ari)) if overall_ari else None,
        "median_co_clustering_agreement": float(np.median(overall_rand))
        if overall_rand
        else None,
        "target_segments": target_summary,
        "root_neighborhood_switch_fraction": root_switch,
        "root_neighborhood_switch_p50": float(np.quantile(switch_values, 0.50))
        if switch_values
        else None,
        "root_neighborhood_switch_p95": float(np.quantile(switch_values, 0.95))
        if switch_values
        else None,
    }


def _stability_gate(
    stability: Mapping[str, Any],
    target_segments: Sequence[str],
    protocol: Mapping[str, Any],
) -> bool:
    overall = stability.get("median_adjusted_rand_index")
    if overall is None or overall < float(protocol["minimum_overall_median_ari"]):
        return False
    return all(
        stability["target_segments"][segment]["median_adjusted_rand_index"] is not None
        and stability["target_segments"][segment]["median_adjusted_rand_index"]
        >= float(protocol["minimum_each_target_segment_median_ari"])
        for segment in target_segments
    )


def _report(summary: Mapping[str, Any]) -> str:
    def metric(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.3f}"

    lines = [
        "# Geometry19 V2.2 Local-Chart MoE P3 Chart Identification",
        "",
        f"- Status: `{summary['status']}`",
        f"- Response-qualified K: {summary['response_qualified_k']}",
        f"- Design-only-qualified K: {summary['design_only_qualified_k']}",
        f"- P4 authorized: `{str(summary['p4_authorized']).lower()}`",
        "- P3 final K selected: `false`",
        "- Confirmation content parsed: `false`",
        "- Expert/router/MoE trained: `false`",
        "",
        "## Candidate diagnostics",
        "",
        "| K | Design ARI | Response ARI | low→mid ARI | mid→high ARI | TSFC ratio reduction | Response qualified |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["response_comparison"]["candidates"]:
        response = item["response_stability"]
        lines.append(
            f"| {item['requested_k']} | {metric(item['design_stability']['median_adjusted_rand_index'])} | "
            f"{metric(response['median_adjusted_rand_index'])} | "
            f"{metric(response['target_segments']['low_mass_low_to_mid']['median_adjusted_rand_index'])} | "
            f"{metric(response['target_segments']['low_mass_mid_to_high']['median_adjusted_rand_index'])} | "
            f"{metric(item['median_tsfc_ratio_reduction_vs_design_only'])} | "
            f"`{str(item['response_qualified']).lower()}` |"
        )
    lines.extend(
        [
            "",
            "Reliable roots alone form chart cores. Unreliable roots attach afterward by design distance and cannot alter core linkage. Cross-fold 38/8/32 categories are diagnostic only and are never used as clustering features. Candidate K values are qualified here; final K selection remains deferred to P7 inner GroupKFold.",
            "",
        ]
    )
    return "\n".join(lines)


def run_chart_identification(
    repo_root: Path,
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Execute revised P3 without training experts or reading confirmation content."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_p3_protocol(config)
    for role in ("parent_protocol", "p1_input_audit", "p1_root_inventory", "p2_summary"):
        item = config[role]
        path = (repo_root / str(item["path"])).resolve()
        _require(path.is_file(), f"P3 input missing: {role}")
        _require(sha256_file(path) == str(item["sha256"]), f"P3 input hash mismatch: {role}")

    parent_path = (repo_root / config["parent_protocol"]["path"]).resolve()
    parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    confirmation_path = (repo_root / parent["inputs"]["sealed_confirmation"]["path"]).resolve()
    _require(
        sha256_file(confirmation_path) == parent["inputs"]["sealed_confirmation"]["sha256"],
        "Sealed confirmation hash mismatch",
    )
    p1_inventory_path = (repo_root / config["p1_root_inventory"]["path"]).resolve()
    root_inventory = {
        str(row["root_seed_id"]): row for row in _read_jsonl(p1_inventory_path)
    }
    p2_path = (repo_root / config["p2_summary"]["path"]).resolve()
    p2 = json.loads(p2_path.read_text(encoding="utf-8"))
    _require(p2.get("status") == P2_PASS_STATUS, "P2 did not authorize P3")
    _require(p2.get("p3_authorized") is True, "P2 P3 authorization is false")
    _require(p2.get("confirmation_content_read") is False, "P2 confirmation isolation failed")

    runtime_root = (
        output_root.resolve()
        if output_root is not None
        else (repo_root / str(config["artifacts"]["runtime_root"])).resolve()
    )
    target_segments = [str(value) for value in config["target_segments"]]
    minimum_core = int(config["core_clustering"]["minimum_reliable_core_roots_per_chart"])
    minimum_final_charts = int(config["core_clustering"]["minimum_final_charts"])
    minimum_total = int(
        config["unreliable_root_attachment"]["minimum_total_roots_per_chart"]
    )
    candidates: dict[str, dict[int, dict[int, dict[str, Any]]]] = {
        view: {int(k): {} for k in config["candidate_k"]} for view in config["views"]
    }
    fold_artifacts: dict[str, Any] = {}

    for fold in range(int(parent["outer_evaluation"]["fold_count"])):
        p2_fold = next(value for value in p2["folds"] if int(value["fold"]) == fold)
        npz_path = repo_root / p2_fold["artifacts"]["root_jacobians"]
        audit_path = repo_root / p2_fold["artifacts"]["jacobian_audit"]
        _require(sha256_file(npz_path) == p2_fold["artifacts"]["root_jacobians_sha256"], f"Fold {fold} NPZ hash mismatch")
        _require(sha256_file(audit_path) == p2_fold["artifacts"]["jacobian_audit_sha256"], f"Fold {fold} audit hash mismatch")
        with np.load(npz_path) as bundle:
            root_ids = [str(value) for value in bundle["root_ids"].tolist()]
            validation_root_ids = [
                str(value) for value in bundle["validation_root_ids"].tolist()
            ]
            design_centers = np.asarray(bundle["design_centers"], dtype=np.float64)
            jacobians = np.asarray(bundle["jacobians"], dtype=np.float64)
            reliable = np.asarray(bundle["jacobian_reliable"], dtype=np.bool_)
            variable_names = np.asarray(bundle["variable_names"])
            scaled_performance = np.asarray(
                bundle["scaled_log_performance_centers"], dtype=np.float64
            )
        _require(not (set(root_ids) & set(validation_root_ids)), f"Fold {fold} validation leakage")
        _require(len(root_ids) == len(set(root_ids)), f"Fold {fold} duplicate train roots")
        tolerances = np.asarray(
            [
                float(
                    yaml.safe_load(
                        (repo_root / parent["inputs"]["target_domain"]["path"]).read_text(
                            encoding="utf-8"
                        )
                    )["axes"][name]["relative_tolerance"]
                )
                for name in ("thrust_N", "tsfc_kg_N_s", "mass_kg")
            ],
            dtype=np.float64,
        )
        physical_performance = np.exp(scaled_performance * tolerances[None, :])
        design_raw = _pairwise_euclidean(design_centers)
        design_scale = _nonzero_pairwise_median(design_raw)
        design_distance = design_raw / design_scale
        jacobian_raw = _pairwise_frobenius(jacobians)
        core_indices = np.flatnonzero(reliable)
        jacobian_scale = _nonzero_pairwise_median(
            jacobian_raw[np.ix_(core_indices, core_indices)]
        )
        jacobian_distance = jacobian_raw / jacobian_scale
        response_distance = design_distance + jacobian_distance

        fold_root = runtime_root / f"charts/fold-{fold}"
        descriptor_path = fold_root / "root_descriptors.npz"
        distance_path = fold_root / "root_distance_components.npz"
        masked_jacobian = jacobian_distance.copy()
        masked_jacobian[~(reliable[:, None] & reliable[None, :])] = np.nan
        masked_response = response_distance.copy()
        masked_response[~(reliable[:, None] & reliable[None, :])] = np.nan
        _write_npz(
            descriptor_path,
            root_ids=np.asarray(root_ids),
            validation_root_ids=np.asarray(validation_root_ids),
            variable_names=variable_names,
            normalized_design_centers=design_centers,
            jacobians=jacobians,
            jacobian_reliable=reliable,
            performance_centers_for_interpretation_only=physical_performance,
        )
        _write_npz(
            distance_path,
            root_ids=np.asarray(root_ids),
            design_distance=design_distance,
            jacobian_distance_reliable_pairs_only=masked_jacobian,
            response_distance_reliable_pairs_only=masked_response,
            reliable_mask=reliable,
            design_distance_scale=np.asarray(design_scale),
            jacobian_distance_scale=np.asarray(jacobian_scale),
        )
        fold_artifacts[str(fold)] = {
            "root_descriptors": str(descriptor_path.relative_to(repo_root))
            if descriptor_path.is_relative_to(repo_root)
            else str(descriptor_path),
            "root_descriptors_sha256": sha256_file(descriptor_path),
            "root_distance_components": str(distance_path.relative_to(repo_root))
            if distance_path.is_relative_to(repo_root)
            else str(distance_path),
            "root_distance_components_sha256": sha256_file(distance_path),
            "training_roots": len(root_ids),
            "validation_roots_excluded": len(validation_root_ids),
            "reliable_core_roots": int(np.sum(reliable)),
            "design_distance_scale": design_scale,
            "jacobian_distance_scale": jacobian_scale,
        }

        for view in config["views"]:
            active_distance = design_distance if view == "design_only" else response_distance
            for requested_k_value in config["candidate_k"]:
                requested_k = int(requested_k_value)
                result = _cluster_candidate(
                    root_ids,
                    reliable,
                    design_distance,
                    active_distance,
                    requested_k=requested_k,
                    minimum_final_charts=minimum_final_charts,
                    minimum_core=minimum_core,
                    minimum_total=minimum_total,
                )
                candidate_root = fold_root / view / f"k-{requested_k}"
                if result["status"] == INFEASIBLE:
                    assignments_payload = {
                        key: value for key, value in result.items() if key != "labels"
                    }
                    assignment_path = candidate_root / "chart_assignments.json"
                    _write_json(assignment_path, assignments_payload)
                    candidates[view][requested_k][fold] = {
                        **assignments_payload,
                        "assignment_path": str(assignment_path.relative_to(repo_root))
                        if assignment_path.is_relative_to(repo_root)
                        else str(assignment_path),
                        "assignment_sha256": sha256_file(assignment_path),
                    }
                    continue
                labels = np.asarray(result.pop("labels"), dtype=np.int64)
                assignments = {
                    root_id: int(labels[index]) for index, root_id in enumerate(root_ids)
                }
                coherence = _coherence(labels, jacobians, reliable)
                details = _chart_details(
                    labels,
                    root_ids,
                    reliable,
                    design_centers,
                    physical_performance,
                    root_inventory,
                )
                assignments_payload = {
                    **result,
                    "view": view,
                    "outer_fold": fold,
                    "training_roots": len(root_ids),
                    "validation_roots_excluded": validation_root_ids,
                    "assignments": {
                        root_id: {
                            "chart_id": assignments[root_id],
                            "role": "reliable_core"
                            if reliable[index]
                            else "design_only_attachment",
                            "jacobian_reliable": bool(reliable[index]),
                        }
                        for index, root_id in enumerate(root_ids)
                    },
                }
                chart_payload = {
                    "view": view,
                    "outer_fold": fold,
                    "requested_k": requested_k,
                    "final_k": result["final_k"],
                    "structural_gate_pass": result["structural_gate_pass"],
                    "tsfc_response_coherence": coherence,
                    "charts": details,
                    "performance_centers_used_for_clustering": False,
                }
                assignment_path = candidate_root / "chart_assignments.json"
                chart_path = candidate_root / "chart_summary.json"
                _write_json(assignment_path, assignments_payload)
                _write_json(chart_path, chart_payload)
                stored = {
                    **result,
                    "assignments": assignments,
                    "coherence": coherence,
                    "assignment_path": str(assignment_path.relative_to(repo_root))
                    if assignment_path.is_relative_to(repo_root)
                    else str(assignment_path),
                    "assignment_sha256": sha256_file(assignment_path),
                    "chart_summary_path": str(chart_path.relative_to(repo_root))
                    if chart_path.is_relative_to(repo_root)
                    else str(chart_path),
                    "chart_summary_sha256": sha256_file(chart_path),
                }
                candidates[view][requested_k][fold] = stored

    stability_payload: dict[str, Any] = {
        "protocol_version": P3_PROTOCOL_VERSION,
        "label_invariant": True,
        "cross_fold_categories_used_as_features": False,
        "candidates": {},
    }
    for view in config["views"]:
        stability_payload["candidates"][view] = {}
        for requested_k_value in config["candidate_k"]:
            requested_k = int(requested_k_value)
            stability_payload["candidates"][view][str(requested_k)] = _stability(
                candidates[view][requested_k], root_inventory, target_segments
            )

    comparison_candidates: list[dict[str, Any]] = []
    design_qualified: list[int] = []
    response_qualified: list[int] = []
    stability_gate_config = config["cross_fold_stability"]
    response_gate = config["response_value_gate"]
    for requested_k_value in config["candidate_k"]:
        requested_k = int(requested_k_value)
        design_stability = stability_payload["candidates"]["design_only"][str(requested_k)]
        response_stability = stability_payload["candidates"]["response_aware"][str(requested_k)]
        design_structural = all(
            candidates["design_only"][requested_k][fold].get("structural_gate_pass", False)
            for fold in candidates["design_only"][requested_k]
        ) and len(candidates["design_only"][requested_k]) == 5
        response_structural = all(
            candidates["response_aware"][requested_k][fold].get("structural_gate_pass", False)
            for fold in candidates["response_aware"][requested_k]
        ) and len(candidates["response_aware"][requested_k]) == 5
        design_stable = design_structural and _stability_gate(
            design_stability, target_segments, stability_gate_config
        )
        response_stable = response_structural and _stability_gate(
            response_stability, target_segments, stability_gate_config
        )
        if design_stable:
            design_qualified.append(requested_k)

        fold_comparisons: list[dict[str, Any]] = []
        for fold in range(5):
            design_result = candidates["design_only"][requested_k][fold]
            response_result = candidates["response_aware"][requested_k][fold]
            design_ratio = design_result.get("coherence", {}).get("within_between_ratio")
            response_ratio = response_result.get("coherence", {}).get("within_between_ratio")
            reduction = (
                float((design_ratio - response_ratio) / design_ratio)
                if design_ratio is not None
                and response_ratio is not None
                and design_ratio > 0.0
                else None
            )
            fold_comparisons.append(
                {
                    "fold": fold,
                    "design_only_ratio": design_ratio,
                    "response_aware_ratio": response_ratio,
                    "ratio_reduction": reduction,
                }
            )
        reductions = [
            value["ratio_reduction"]
            for value in fold_comparisons
            if value["ratio_reduction"] is not None
        ]
        response_ratios = [
            value["response_aware_ratio"]
            for value in fold_comparisons
            if value["response_aware_ratio"] is not None
        ]
        median_reduction = float(np.median(reductions)) if reductions else None
        median_response_ratio = float(np.median(response_ratios)) if response_ratios else None
        design_ari = design_stability["median_adjusted_rand_index"]
        response_ari = response_stability["median_adjusted_rand_index"]
        ari_degradation = (
            float(design_ari - response_ari)
            if design_ari is not None and response_ari is not None
            else None
        )
        response_value_pass = (
            median_response_ratio is not None
            and median_response_ratio
            <= float(response_gate["maximum_response_aware_median_ratio"])
            and median_reduction is not None
            and median_reduction
            >= float(response_gate["minimum_median_ratio_reduction_vs_design_only"])
            and ari_degradation is not None
            and ari_degradation
            <= float(
                response_gate["maximum_overall_median_ari_degradation_vs_design_only"]
            )
        )
        qualified = response_stable and response_value_pass
        if qualified:
            response_qualified.append(requested_k)
        comparison_candidates.append(
            {
                "requested_k": requested_k,
                "design_structural_gate_pass": design_structural,
                "response_structural_gate_pass": response_structural,
                "design_stability_gate_pass": design_stable,
                "response_stability_gate_pass": response_stable,
                "design_stability": design_stability,
                "response_stability": response_stability,
                "fold_tsfc_coherence_comparison": fold_comparisons,
                "median_response_aware_tsfc_within_between_ratio": median_response_ratio,
                "median_tsfc_ratio_reduction_vs_design_only": median_reduction,
                "overall_median_ari_degradation_vs_design_only": ari_degradation,
                "response_value_gate_pass": response_value_pass,
                "response_qualified": qualified,
            }
        )

    response_comparison = {
        "protocol_version": P3_PROTOCOL_VERSION,
        "gates": config["response_value_gate"],
        "candidates": comparison_candidates,
        "response_qualified_k": response_qualified,
        "design_only_qualified_k": design_qualified,
        "p3_selects_final_k": False,
    }
    weak_segment = "low_mass_mid_to_high"
    weak_audit_candidates: list[dict[str, Any]] = []
    for view in config["views"]:
        for requested_k_value in config["candidate_k"]:
            requested_k = int(requested_k_value)
            for fold, result in candidates[view][requested_k].items():
                if not result.get("structural_gate_pass"):
                    continue
                members = [
                    root_id
                    for root_id in result["assignments"]
                    if weak_segment in root_inventory[root_id]["segments"]
                ]
                audit_rows = _read_jsonl(
                    repo_root / p2["folds"][fold]["artifacts"]["jacobian_audit"]
                )
                reliable_by_root = {
                    str(row["root_seed_id"]): bool(row["jacobian_reliable"])
                    for row in audit_rows
                }
                chart_counts = Counter(result["assignments"][root_id] for root_id in members)
                weak_audit_candidates.append(
                    {
                        "view": view,
                        "requested_k": requested_k,
                        "fold": fold,
                        "training_segment_roots": len(members),
                        "reliable_core_segment_roots": sum(
                            reliable_by_root[root_id] for root_id in members
                        ),
                        "design_only_attached_segment_roots": sum(
                            not reliable_by_root[root_id] for root_id in members
                        ),
                        "represented_charts": len(chart_counts),
                        "chart_root_counts": {
                            str(key): value for key, value in sorted(chart_counts.items())
                        },
                        "collapsed_to_single_chart": len(chart_counts) == 1,
                    }
                )
    weak_segment_audit = {
        "segment": weak_segment,
        "candidates": weak_audit_candidates,
        "cross_fold_stability": {
            view: {
                str(k): stability_payload["candidates"][view][str(k)]["target_segments"][
                    weak_segment
                ]
                for k in config["candidate_k"]
            }
            for view in config["views"]
        },
    }

    if response_qualified:
        status = RESPONSE_PASS
        p4_authorized = True
        next_stage = "P4_design_support_and_interpretable_soft_routing"
    elif design_qualified:
        status = DESIGN_ONLY
        p4_authorized = False
        next_stage = "review_design_only_locality_before_P4"
    else:
        status = UNSTABLE
        p4_authorized = False
        next_stage = "controlled_signed_perturbations_for_unreliable_roots"

    stability_path = runtime_root / config["artifacts"]["cross_fold_stability"]
    comparison_path = runtime_root / config["artifacts"]["response_comparison"]
    weak_path = runtime_root / config["artifacts"]["weak_segment_audit"]
    report_path = runtime_root / config["artifacts"]["report"]
    summary_path = runtime_root / config["artifacts"]["chart_summary"]
    _write_json(stability_path, stability_payload)
    _write_json(comparison_path, response_comparison)
    _write_json(weak_path, weak_segment_audit)
    implementation_path = Path(__file__).resolve()
    candidate_artifacts = {
        view: {
            str(requested_k): {
                str(fold): {
                    key: result[key]
                    for key in (
                        "status",
                        "final_k",
                        "structural_gate_pass",
                        "assignment_path",
                        "assignment_sha256",
                        "chart_summary_path",
                        "chart_summary_sha256",
                    )
                    if key in result
                }
                for fold, result in sorted(candidates[view][requested_k].items())
            }
            for requested_k in (int(value) for value in config["candidate_k"])
        }
        for view in config["views"]
    }
    summary: dict[str, Any] = {
        "status": status,
        "protocol_version": P3_PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(config_path.relative_to(repo_root)),
        "protocol_sha256": sha256_file(config_path),
        "parent_protocol_sha256": sha256_file(parent_path),
        "p2_summary_sha256": sha256_file(p2_path),
        "implementation_path": str(implementation_path.relative_to(repo_root)),
        "implementation_sha256": sha256_file(implementation_path),
        "fold_artifacts": fold_artifacts,
        "candidate_artifacts": candidate_artifacts,
        "response_comparison": response_comparison,
        "response_qualified_k": response_qualified,
        "design_only_qualified_k": design_qualified,
        "p3_selects_final_k": False,
        "final_k_selection_stage": "P7_inner_GroupKFold",
        "p4_authorized": p4_authorized,
        "confirmation_hash_checked": True,
        "confirmation_content_read": False,
        "expert_training_executed": False,
        "router_training_executed": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "artifacts": {
            "cross_fold_stability": str(stability_path.relative_to(repo_root))
            if stability_path.is_relative_to(repo_root)
            else str(stability_path),
            "cross_fold_stability_sha256": sha256_file(stability_path),
            "response_comparison": str(comparison_path.relative_to(repo_root))
            if comparison_path.is_relative_to(repo_root)
            else str(comparison_path),
            "response_comparison_sha256": sha256_file(comparison_path),
            "weak_segment_audit": str(weak_path.relative_to(repo_root))
            if weak_path.is_relative_to(repo_root)
            else str(weak_path),
            "weak_segment_audit_sha256": sha256_file(weak_path),
        },
        "next_authorized_stage": next_stage,
    }
    _write_text(report_path, _report(summary))
    summary["artifacts"]["report"] = (
        str(report_path.relative_to(repo_root))
        if report_path.is_relative_to(repo_root)
        else str(report_path)
    )
    summary["artifacts"]["report_sha256"] = sha256_file(report_path)
    _write_json(summary_path, summary)
    return summary
