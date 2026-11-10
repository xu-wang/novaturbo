"""Development-only bridge-root expansion for weak Geometry19 V2.2 segments."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.physics.geometry19_v2_2_balanced_dataset import (
    _deduplicate_training_rows,
    _evaluate_assigned,
)
from src.physics.geometry19_v2_2_dataset import (
    Geometry19V22DatasetError,
    _artifact_path,
    _canonical_bytes,
    _hash_file,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from src.physics.geometry19_v2_2_directed_recovery import (
    _directed_plan,
    _fit_root_model,
    _normalized_design,
    _scaled_performance,
)


BRIDGE_ROOT_VERSION = "geometry19_v2_2_bridge_root_expansion_v1"
SUPPORTED_BRIDGE_ROOT_VERSIONS = {
    BRIDGE_ROOT_VERSION,
    "geometry19_v2_2_bridge_root_expansion_v2",
    "geometry19_v2_2_bridge_root_expansion_v3",
    "geometry19_v2_2_bridge_root_expansion_v4",
    "geometry19_v2_2_bridge_root_expansion_v5",
    "geometry19_v2_2_bridge_root_expansion_v6",
}


def _load_protocol(root: Path, source: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(source)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(protocol, dict)
        or protocol.get("dataset_version") not in SUPPORTED_BRIDGE_ROOT_VERSIONS
    ):
        raise Geometry19V22DatasetError("Bridge-root protocol version mismatch")
    if protocol.get("profile") != "v2_2_development":
        raise Geometry19V22DatasetError("Bridge-root expansion requires v2_2_development")
    if protocol.get("formal_use_allowed") is not False:
        raise Geometry19V22DatasetError("Bridge-root expansion cannot be formal")
    sampling = protocol.get("sampling", {})
    if not 1 <= len(sampling.get("segments", [])) <= 2 or not sampling.get("rounds"):
        raise Geometry19V22DatasetError("One or two segments and at least one round are required")
    if int(protocol.get("development_folds", {}).get("fold_count", 0)) < 2:
        raise Geometry19V22DatasetError("At least two development folds are required")
    return path, protocol


def _load_locked_split(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    spec = protocol["frozen_split"]
    manifest_path = (root / spec["manifest_path"]).resolve()
    if _hash_file(manifest_path) != spec["manifest_sha256"]:
        raise Geometry19V22DatasetError("Frozen split manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    role_rows: dict[str, list[dict[str, Any]]] = {}
    for role in ("train", "validation", "confirmation"):
        role_spec = spec[role]
        path = (root / role_spec["path"]).resolve()
        if _hash_file(path) != role_spec["sha256"]:
            raise Geometry19V22DatasetError(f"Frozen {role} split hash mismatch")
        if manifest["hashes"][f"{role}_split_hash"] != role_spec["sha256"]:
            raise Geometry19V22DatasetError(f"Frozen {role} manifest binding mismatch")
        role_rows[role] = _read_jsonl(path)
    roots = {
        role: {str(row["root_seed_id"]) for row in rows}
        for role, rows in role_rows.items()
    }
    if roots["confirmation"] & (roots["train"] | roots["validation"]):
        raise Geometry19V22DatasetError("Confirmation roots leak into development")
    return [*role_rows["train"], *role_rows["validation"]], role_rows["confirmation"], manifest


def _load_prior_qualified_development(
    root: Path,
    protocol: Mapping[str, Any],
    confirmation_roots: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for spec in protocol.get("prior_development_expansions", []):
        manifest_path = (root / spec["manifest_path"]).resolve()
        accepted_path = (root / spec["accepted_path"]).resolve()
        if _hash_file(manifest_path) != spec["manifest_sha256"]:
            raise Geometry19V22DatasetError("Prior expansion manifest hash mismatch")
        if _hash_file(accepted_path) != spec["accepted_sha256"]:
            raise Geometry19V22DatasetError("Prior expansion accepted-data hash mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["hashes"]["accepted_bridge_roots_hash"] != spec["accepted_sha256"]:
            raise Geometry19V22DatasetError("Prior expansion manifest/data binding mismatch")
        qualified_by_segment = {
            segment: set(values["qualified_rows_by_bridge_root"])
            for segment, values in manifest["counts"]["by_segment"].items()
        }
        accepted = _read_jsonl(accepted_path)
        retained = [
            row
            for row in accepted
            if str(row["root_seed_id"])
            in qualified_by_segment.get(str(row["assigned_segment"]), set())
        ]
        retained_roots = {str(row["root_seed_id"]) for row in retained}
        if retained_roots & confirmation_roots:
            raise Geometry19V22DatasetError("Confirmation root in prior development expansion")
        rows.extend(retained)
        audit.append(
            {
                "dataset_version": manifest["dataset_version"],
                "manifest_path": spec["manifest_path"],
                "accepted_path": spec["accepted_path"],
                "retained_rows": len(retained),
                "retained_roots": len(retained_roots),
                "retained_roots_by_segment": {
                    segment: len(
                        {
                            str(row["root_seed_id"])
                            for row in retained
                            if row["assigned_segment"] == segment
                        }
                    )
                    for segment in sorted(qualified_by_segment)
                },
            }
        )
    return rows, audit


def _root_centers(
    rows: Sequence[dict[str, Any]],
    variables: Sequence[Mapping[str, Any]],
    *,
    segment_id: str | None = None,
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        if segment_id is not None and row["assigned_segment"] != segment_id:
            continue
        grouped[str(row["root_seed_id"])].append(_normalized_design(row, variables))
    return {
        root_id: np.mean(np.stack(values), axis=0)
        for root_id, values in sorted(grouped.items())
    }


def _bridge_candidates(
    centers: Mapping[str, np.ndarray],
    interpolation_fractions: Sequence[float],
    minimum_endpoint_l2: float,
    minimum_l2_to_existing: float,
    *,
    rows_by_root: Mapping[str, Sequence[dict[str, Any]]] | None = None,
    variables: Sequence[Mapping[str, Any]] | None = None,
    domain: Any | None = None,
) -> list[dict[str, Any]]:
    root_ids = sorted(centers)
    existing = np.stack([centers[root_id] for root_id in root_ids])
    candidates: list[dict[str, Any]] = []
    for left_index, left_id in enumerate(root_ids):
        for right_id in root_ids[left_index + 1 :]:
            left = centers[left_id]
            right = centers[right_id]
            endpoint_performance_l2 = 0.0
            left_design_hash = None
            right_design_hash = None
            if rows_by_root is not None:
                if variables is None or domain is None:
                    raise Geometry19V22DatasetError(
                        "Matched bridge endpoints require variables and target domain"
                    )
                left_rows = list(rows_by_root[left_id])
                right_rows = list(rows_by_root[right_id])
                left_performance = np.stack(
                    [_scaled_performance(domain, row["performance"]) for row in left_rows]
                )
                right_performance = np.stack(
                    [_scaled_performance(domain, row["performance"]) for row in right_rows]
                )
                performance_distances = np.linalg.norm(
                    left_performance[:, None, :] - right_performance[None, :, :], axis=2
                )
                left_row_index, right_row_index = np.unravel_index(
                    int(np.argmin(performance_distances)), performance_distances.shape
                )
                left_row = left_rows[left_row_index]
                right_row = right_rows[right_row_index]
                left = _normalized_design(left_row, variables)
                right = _normalized_design(right_row, variables)
                endpoint_performance_l2 = float(
                    performance_distances[left_row_index, right_row_index]
                )
                left_design_hash = str(left_row["design_hash"])
                right_design_hash = str(right_row["design_hash"])
            endpoint_l2 = float(np.linalg.norm(left - right))
            if endpoint_l2 < minimum_endpoint_l2:
                continue
            for fraction in interpolation_fractions:
                alpha = float(fraction)
                center = (1.0 - alpha) * left + alpha * right
                nearest = float(np.min(np.linalg.norm(existing - center[None, :], axis=1)))
                if nearest < minimum_l2_to_existing:
                    continue
                candidates.append(
                    {
                        "left_root_id": left_id,
                        "right_root_id": right_id,
                        "interpolation_fraction": alpha,
                        "endpoint_l2": endpoint_l2,
                        "endpoint_performance_l2": endpoint_performance_l2,
                        "left_design_hash": left_design_hash,
                        "right_design_hash": right_design_hash,
                        "minimum_l2_to_existing": nearest,
                        "center_x": center,
                    }
                )
    if not candidates:
        raise Geometry19V22DatasetError("No bridge candidates satisfy separation gates")
    return candidates


def _select_bridge_roots(
    centers: Mapping[str, np.ndarray],
    *,
    segment_id: str,
    interpolation_fractions: Sequence[float],
    bridge_root_count: int,
    minimum_endpoint_l2: float,
    minimum_l2_to_existing: float,
    minimum_l2_between_bridges: float,
    dataset_version: str = BRIDGE_ROOT_VERSION,
    performance_distance_penalty: float = 0.0,
    rows_by_root: Mapping[str, Sequence[dict[str, Any]]] | None = None,
    variables: Sequence[Mapping[str, Any]] | None = None,
    domain: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = _bridge_candidates(
        centers,
        interpolation_fractions,
        minimum_endpoint_l2,
        minimum_l2_to_existing,
        rows_by_root=rows_by_root,
        variables=variables,
        domain=domain,
    )
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < bridge_root_count:
        def score(
            candidate: Mapping[str, Any],
        ) -> tuple[float, float, float, float, str, str, float]:
            separation = float(candidate["minimum_l2_to_existing"])
            if selected:
                separation = min(
                    separation,
                    min(
                        float(np.linalg.norm(candidate["center_x"] - item["center_x"]))
                        for item in selected
                    ),
                )
            return (
                separation
                - performance_distance_penalty
                * float(candidate["endpoint_performance_l2"]),
                separation,
                float(candidate["endpoint_l2"]),
                -float(candidate["endpoint_performance_l2"]),
                str(candidate["left_root_id"]),
                str(candidate["right_root_id"]),
                float(candidate["interpolation_fraction"]),
            )

        chosen = max(remaining, key=score)
        chosen_separation = score(chosen)[1]
        if chosen_separation < minimum_l2_between_bridges:
            break
        identity = {
            "dataset_version": dataset_version,
            "segment_id": segment_id,
            "left_root_id": chosen["left_root_id"],
            "right_root_id": chosen["right_root_id"],
            "interpolation_fraction": chosen["interpolation_fraction"],
        }
        selected.append(
            {
                **chosen,
                "minimum_l2_at_selection": chosen_separation,
                "root_seed_id": sha256(_canonical_bytes(identity)).hexdigest(),
                "anchor_id": (
                    f"bridge:{segment_id}:{chosen['left_root_id'][:8]}:"
                    f"{chosen['right_root_id'][:8]}:{chosen['interpolation_fraction']:.3f}"
                ),
            }
        )
        remaining.remove(chosen)
    if len(selected) < bridge_root_count:
        raise Geometry19V22DatasetError(
            f"Insufficient separated bridge roots for {segment_id}: {len(selected)}"
        )
    return selected, candidates


def _select_nullspace_roots(
    centers: Mapping[str, np.ndarray],
    rows_by_root: Mapping[str, Sequence[dict[str, Any]]],
    variables: Sequence[Mapping[str, Any]],
    domain: Any,
    *,
    dataset_version: str,
    segment_id: str,
    bridge_root_count: int,
    directions_per_parent: int,
    nullspace_step_l2: Sequence[float],
    maximum_observations_per_root: int,
    ridge: float,
    inverse_ridge: float,
    minimum_l2_to_existing: float,
    minimum_l2_between_bridges: float,
    predicted_performance_drift_penalty: float,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Propose roots along fitted local performance-nullspace directions."""
    existing = np.stack(list(centers.values()))
    candidates: list[dict[str, Any]] = []
    for parent_index, parent_root_id in enumerate(sorted(rows_by_root)):
        ranked = sorted(
            rows_by_root[parent_root_id],
            key=lambda row: (
                float(row["assigned_segment_assessment"]["normalized_distance"]),
                str(row["design_hash"]),
            ),
        )[:maximum_observations_per_root]
        if len(ranked) < 4:
            continue
        anchor = ranked[0]
        anchor_x = _normalized_design(anchor, variables)
        anchor_y = _scaled_performance(domain, anchor["performance"])
        x = np.stack([_normalized_design(row, variables) for row in ranked])
        y = np.stack([_scaled_performance(domain, row["performance"]) for row in ranked])
        delta_x = x - anchor_x
        delta_y = y - anchor_y
        coefficients = np.linalg.solve(
            delta_x.T @ delta_x + ridge * np.eye(delta_x.shape[1]),
            delta_x.T @ delta_y,
        )
        jacobian = coefficients.T
        nullspace_projector = np.eye(len(variables)) - jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + inverse_ridge * np.eye(jacobian.shape[0]),
            jacobian,
        )
        for direction_index in range(directions_per_parent):
            rng = np.random.default_rng(
                random_seed + parent_index * 1009 + direction_index * 97
            )
            direction = nullspace_projector @ rng.normal(size=len(variables))
            norm = float(np.linalg.norm(direction))
            if norm <= 1.0e-12:
                continue
            direction /= norm
            step_l2 = float(nullspace_step_l2[direction_index % len(nullspace_step_l2)])
            candidate_x = np.clip(anchor_x + step_l2 * direction, 0.0, 1.0)
            actual_step_l2 = float(np.linalg.norm(candidate_x - anchor_x))
            nearest = float(
                np.min(np.linalg.norm(existing - candidate_x[None, :], axis=1))
            )
            if nearest < minimum_l2_to_existing:
                continue
            predicted_drift = float(
                np.linalg.norm(jacobian @ (candidate_x - anchor_x))
            )
            identity = {
                "dataset_version": dataset_version,
                "segment_id": segment_id,
                "parent_root_id": parent_root_id,
                "anchor_design_hash": str(anchor["design_hash"]),
                "direction_index": direction_index,
                "step_l2": step_l2,
            }
            candidates.append(
                {
                    "root_seed_id": sha256(_canonical_bytes(identity)).hexdigest(),
                    "anchor_id": (
                        f"nullbridge:{segment_id}:{parent_root_id[:8]}:"
                        f"{direction_index:02d}:{step_l2:.3f}"
                    ),
                    "parent_root_id": parent_root_id,
                    "anchor_design_hash": str(anchor["design_hash"]),
                    "direction_index": direction_index,
                    "requested_step_l2": step_l2,
                    "actual_step_l2": actual_step_l2,
                    "predicted_performance_drift_l2": predicted_drift,
                    "jacobian_rank": int(np.linalg.matrix_rank(jacobian)),
                    "observation_count": len(ranked),
                    "minimum_l2_to_existing": nearest,
                    "center_x": candidate_x,
                }
            )
    if not candidates:
        raise Geometry19V22DatasetError(
            f"No local-nullspace candidates for {segment_id}"
        )
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < bridge_root_count:
        def score(candidate: Mapping[str, Any]) -> tuple[float, float, float, str]:
            separation = float(candidate["minimum_l2_to_existing"])
            if selected:
                separation = min(
                    separation,
                    min(
                        float(np.linalg.norm(candidate["center_x"] - item["center_x"]))
                        for item in selected
                    ),
                )
            return (
                separation
                - predicted_performance_drift_penalty
                * float(candidate["predicted_performance_drift_l2"]),
                separation,
                -float(candidate["predicted_performance_drift_l2"]),
                str(candidate["root_seed_id"]),
            )

        chosen = max(remaining, key=score)
        chosen_separation = score(chosen)[1]
        if chosen_separation < minimum_l2_between_bridges:
            break
        selected.append({**chosen, "minimum_l2_at_selection": chosen_separation})
        remaining.remove(chosen)
    if len(selected) < bridge_root_count:
        raise Geometry19V22DatasetError(
            f"Insufficient separated local-nullspace roots for {segment_id}: {len(selected)}"
        )
    return selected, candidates


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def _coverage_summary(
    reference_candidates: Sequence[Mapping[str, Any]],
    before_centers: Mapping[str, np.ndarray],
    after_centers: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    reference = np.stack([item["center_x"] for item in reference_candidates])
    before = np.stack(list(before_centers.values()))
    after = np.stack(list(after_centers.values()))
    before_distances = np.min(
        np.linalg.norm(reference[:, None, :] - before[None, :, :], axis=2), axis=1
    )
    after_distances = np.min(
        np.linalg.norm(reference[:, None, :] - after[None, :, :], axis=2), axis=1
    )
    before_summary = _distribution(before_distances)
    after_summary = _distribution(after_distances)
    reduction = (
        (before_summary["p95"] - after_summary["p95"]) / before_summary["p95"]
        if before_summary["p95"] > 0.0
        else 0.0
    )
    return {
        "reference_bridge_points": len(reference_candidates),
        "before": before_summary,
        "after": after_summary,
        "p95_reduction_fraction": float(reduction),
    }


def _assign_development_folds(
    rows: Sequence[dict[str, Any]],
    variables: Sequence[Mapping[str, Any]],
    target_segments: Sequence[str],
    *,
    fold_count: int,
    distance_strata: int,
) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["root_seed_id"])].append(row)
    centers = _root_centers(rows, variables)
    root_ids = sorted(grouped)
    center_array = np.stack([centers[root_id] for root_id in root_ids])
    distances = np.linalg.norm(
        center_array[:, None, :] - center_array[None, :, :], axis=2
    )
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    boundaries = np.quantile(nearest, np.linspace(0.0, 1.0, distance_strata + 1))
    summaries: list[dict[str, Any]] = []
    for index, root_id in enumerate(root_ids):
        root_rows = grouped[root_id]
        stratum = min(
            int(np.searchsorted(boundaries[1:-1], nearest[index], side="right")),
            distance_strata - 1,
        )
        summaries.append(
            {
                "root_seed_id": root_id,
                "rows": len(root_rows),
                "segments": sorted(
                    {row["assigned_segment"] for row in root_rows if row["assigned_segment"] in target_segments}
                ),
                "nearest_root_center_l2": float(nearest[index]),
                "distance_stratum": stratum,
            }
        )
    fold_state = [
        {
            "rows": 0,
            "roots": 0,
            "strata": defaultdict(int),
            "segments": defaultdict(int),
        }
        for _ in range(fold_count)
    ]
    assignments: dict[str, int] = {}
    for summary in sorted(
        summaries,
        key=lambda item: (
            -len(item["segments"]),
            item["distance_stratum"],
            -item["rows"],
            item["root_seed_id"],
        ),
    ):
        stratum = summary["distance_stratum"]
        segments = summary["segments"]
        selected_fold = min(
            range(fold_count),
            key=lambda fold: (
                max(
                    (fold_state[fold]["segments"][segment] for segment in segments),
                    default=0,
                ),
                sum(fold_state[fold]["segments"][segment] for segment in segments),
                fold_state[fold]["strata"][stratum],
                fold_state[fold]["roots"],
                fold_state[fold]["rows"],
                fold,
            ),
        )
        assignments[summary["root_seed_id"]] = selected_fold
        state = fold_state[selected_fold]
        state["rows"] += summary["rows"]
        state["roots"] += 1
        state["strata"][stratum] += 1
        for segment in segments:
            state["segments"][segment] += 1
    fold_summaries: list[dict[str, Any]] = []
    for fold in range(fold_count):
        validation = [row for row in rows if assignments[str(row["root_seed_id"])] == fold]
        train = [row for row in rows if assignments[str(row["root_seed_id"])] != fold]
        fold_summaries.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "train_roots": len({row["root_seed_id"] for row in train}),
                "validation_rows": len(validation),
                "validation_roots": len({row["root_seed_id"] for row in validation}),
                "validation_roots_by_target_segment": {
                    segment: len(
                        {
                            row["root_seed_id"]
                            for row in validation
                            if row["assigned_segment"] == segment
                        }
                    )
                    for segment in target_segments
                },
            }
        )
    metadata = {
        "fold_count": fold_count,
        "distance_strata": distance_strata,
        "nearest_root_center_l2_boundaries": [float(value) for value in boundaries],
    }
    return assignments, fold_summaries, metadata


def _render_report(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Geometry19 V2.2 development 桥接根扩充报告",
        "",
        f"- 状态：`{manifest['status']}`",
        f"- 物理评价：{manifest['counts']['evaluated']}",
        f"- physics-domain valid：{manifest['counts']['physics_domain_valid']}",
        f"- 新有效候选：{manifest['counts']['accepted_unique']}",
        f"- 通过单根门的新增候选：{manifest['counts']['qualified_accepted_unique']}",
        f"- development 合并候选：{manifest['counts']['development_combined_unique']}",
        "- confirmation：保持封存，未参与桥接选择、拟合、合并或折分配",
        "- 正式用途：禁止",
        "",
        "## 根覆盖结果",
        "",
        "| 线段 | 原 development 根 | 选定桥接根 | 命中根 | 合格根 | 合并后根 | 覆盖 P95 降幅 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for segment in manifest["sampling"]["segments"]:
        values = manifest["counts"]["by_segment"][segment]
        coverage = manifest["coverage"][segment]
        lines.append(
            f"| `{segment}` | {values['development_roots_before']} | "
            f"{values['selected_bridge_roots']} | {values['bridge_roots_with_candidates']} | "
            f"{values['qualified_bridge_roots']} | {values['development_roots_after']} | "
            f"{coverage['p95_reduction_fraction']:.1%} |"
        )
    lines.extend(
        [
            "",
            "桥接中心只由已公开 development 根的归一化 19D 中点候选产生，并以 maximin 规则填补覆盖空隙；后续局部线性模型只用于提议，最终标签全部由统一 V2.2 物理运行时重新计算。",
            "",
            "## Development 根级折",
            "",
            "新 development 数据已按全局 `root_seed_id` 绑定到单一验证折，并按最近根中心距离分层。每折训练时应使用其余折，以 worst-root rho P95 选模；不得读取现有 confirmation。",
            "",
            "## 使用限制",
            "",
            "该数据仍为 19D 简化 V2.2 软件标签，不涉及真实几何或工程可制造性。只有 development 方案冻结后，才能另行生成至少三个全新 confirmation 根/线段并做一次性验收。",
            "",
        ]
    )
    return "\n".join(lines)


def run_bridge_root_expansion(
    repository_root: str | Path,
    protocol_path: str | Path,
    output_root: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Fill development root-center gaps without reusing confirmation."""
    root = Path(repository_root).resolve()
    protocol_source, protocol = _load_protocol(root, protocol_path)
    output = Path(output_root).resolve()
    if output.exists() and not resume:
        raise Geometry19V22DatasetError(f"Refusing to overwrite bridge expansion: {output}")
    output.mkdir(parents=True, exist_ok=True)
    development_rows, confirmation_rows, split_manifest = _load_locked_split(root, protocol)
    confirmation_roots = {str(row["root_seed_id"]) for row in confirmation_rows}
    prior_rows, prior_expansion_audit = _load_prior_qualified_development(
        root, protocol, confirmation_roots
    )
    development_rows = _deduplicate_training_rows([*development_rows, *prior_rows])
    development_roots = {str(row["root_seed_id"]) for row in development_rows}
    if confirmation_roots & development_roots:
        raise Geometry19V22DatasetError("Confirmation root present in development input")

    context = build_pipeline_context(root, pipeline_profile="v2_2_development")
    domain = context.target_domain
    assert domain is not None
    variables = list(context.physics_runtime.design_schema["design_variables"])
    sampling = protocol["sampling"]
    dataset_version = str(protocol["dataset_version"])
    segments = tuple(str(value) for value in sampling["segments"])
    known_segments = {segment[0] for segment in domain.segments}
    if not set(segments) < known_segments:
        raise Geometry19V22DatasetError("Bridge segments must be a strict domain subset")

    selected: dict[str, list[dict[str, Any]]] = {}
    reference_candidates: dict[str, list[dict[str, Any]]] = {}
    before_centers: dict[str, dict[str, np.ndarray]] = {}
    for segment in segments:
        before_centers[segment] = _root_centers(
            development_rows, variables, segment_id=segment
        )
        if set(before_centers[segment]) & confirmation_roots:
            raise Geometry19V22DatasetError("Confirmation root used as bridge endpoint")
        endpoint_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in development_rows:
            if row["assigned_segment"] == segment:
                endpoint_rows[str(row["root_seed_id"])].append(row)
        if sampling.get("candidate_strategy") == "local_jacobian_nullspace":
            selected[segment], reference_candidates[segment] = _select_nullspace_roots(
                before_centers[segment],
                endpoint_rows,
                variables,
                domain,
                dataset_version=dataset_version,
                segment_id=segment,
                bridge_root_count=int(sampling["bridge_root_count_per_segment"]),
                directions_per_parent=int(sampling["directions_per_parent"]),
                nullspace_step_l2=sampling["nullspace_step_l2"],
                maximum_observations_per_root=int(
                    sampling["maximum_observations_per_root"]
                ),
                ridge=float(sampling["ridge"]),
                inverse_ridge=float(sampling["inverse_ridge"]),
                minimum_l2_to_existing=float(sampling["minimum_l2_to_existing"]),
                minimum_l2_between_bridges=float(
                    sampling["minimum_l2_between_bridges"]
                ),
                predicted_performance_drift_penalty=float(
                    sampling["predicted_performance_drift_penalty"]
                ),
                random_seed=int(sampling["random_seed"]),
            )
        else:
            selected[segment], reference_candidates[segment] = _select_bridge_roots(
                before_centers[segment],
                segment_id=segment,
                interpolation_fractions=sampling["interpolation_fractions"],
                bridge_root_count=int(sampling["bridge_root_count_per_segment"]),
                minimum_endpoint_l2=float(sampling["minimum_endpoint_l2"]),
                minimum_l2_to_existing=float(sampling["minimum_l2_to_existing"]),
                minimum_l2_between_bridges=float(sampling["minimum_l2_between_bridges"]),
                dataset_version=dataset_version,
                performance_distance_penalty=float(
                    sampling.get("performance_distance_penalty", 0.0)
                ),
                rows_by_root=(
                    endpoint_rows
                    if sampling.get("match_endpoint_performance", False)
                    else None
                ),
                variables=variables,
                domain=domain,
            )

    protocol_hash = _hash_file(protocol_source)
    generated_rows: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    model_audit: list[dict[str, Any]] = []
    selected_by_id = {
        item["root_seed_id"]: item for values in selected.values() for item in values
    }
    for round_index, round_spec in enumerate(sampling["rounds"]):
        accepted_prior = _deduplicate_training_rows(generated_rows)
        valid_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in generated_rows:
            if row.get("performance") and row["constraint_assessment"]["physics_domain_valid"]:
                valid_by_root[str(row["root_seed_id"])].append(row)
        for segment_index, segment in enumerate(segments):
            root_ids = [item["root_seed_id"] for item in selected[segment]]
            accepted_counts = {
                root_id: sum(
                    row["assigned_segment"] == segment
                    and str(row["root_seed_id"]) == root_id
                    for row in accepted_prior
                )
                for root_id in root_ids
            }
            active_ids = [
                root_id
                for root_id in root_ids
                if accepted_counts[root_id] < int(sampling["target_accepted_rows_per_root"])
            ]
            models: list[dict[str, Any]] = []
            for root_id in active_ids:
                observations = valid_by_root[root_id]
                if round_index > 0 and len(observations) >= int(sampling["minimum_observations_for_projection"]):
                    model = _fit_root_model(
                        domain,
                        variables,
                        segment,
                        observations,
                        maximum_observations=int(sampling["maximum_observations_per_root"]),
                        ridge=float(sampling["ridge"]),
                        inverse_ridge=float(sampling["inverse_ridge"]),
                        trust_radius=float(round_spec["trust_radius_fraction"]),
                        step_gain=float(round_spec["step_gain"]),
                    )
                    model["proposal_method"] = "local_jacobian_projection"
                else:
                    seed = selected_by_id[root_id]
                    model = {
                        "root_seed_id": root_id,
                        "anchor_id": seed["anchor_id"],
                        "center_x": seed["center_x"],
                        "proposal_method": "root_center_bridge",
                        "observation_count": len(observations),
                    }
                models.append(model)
            model_audit.append(
                {
                    "adaptive_round": round_index,
                    "assigned_segment": segment,
                    "active_root_ids": active_ids,
                    "accepted_before_round": accepted_counts,
                    "models": [
                        {key: value for key, value in model.items() if key != "center_x"}
                        for model in models
                    ],
                }
            )
            plan = _directed_plan(
                variables,
                models,
                segment_id=segment,
                round_index=round_index,
                sample_count=int(round_spec["samples_per_active_root"]) * len(models),
                perturbation_radius=float(round_spec["perturbation_radius_fraction"]),
                discrete_step_probability=float(sampling["discrete_step_probability"]),
                random_seed=int(sampling["random_seed"]) + round_index * 1019 + segment_index * 97,
            ) if models else []
            for row in plan:
                row["sample_id"] = row["sample_id"].replace("directed:", "bridge:", 1)
            shard_path = output / "shards" / f"round-{round_index:02d}-{segment}.jsonl"
            metadata_path = shard_path.with_suffix(".meta.json")
            identity = {
                "dataset_version": dataset_version,
                "protocol_hash": protocol_hash,
                "pipeline_context_hash": context.context_hash,
                "input_hash": sha256(_canonical_bytes(plan)).hexdigest(),
            }
            if resume and shard_path.exists() and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if any(metadata.get(key) != value for key, value in identity.items()):
                    raise Geometry19V22DatasetError(
                        f"Bridge shard identity mismatch: {round_index}/{segment}"
                    )
                if metadata.get("output_hash") != _hash_file(shard_path):
                    raise Geometry19V22DatasetError(
                        f"Bridge shard output hash mismatch: {round_index}/{segment}"
                    )
                rows = _read_jsonl(shard_path)
            else:
                rows = [_evaluate_assigned(row, context) for row in plan]
                _write_jsonl(shard_path, rows)
                metadata = {
                    **identity,
                    "adaptive_round": round_index,
                    "assigned_segment": segment,
                    "active_root_count": len(active_ids),
                    "row_count": len(rows),
                    "physics_domain_valid_count": sum(
                        row["constraint_assessment"]["physics_domain_valid"] for row in rows
                    ),
                    "training_candidate_count": sum(row["training_candidate"] for row in rows),
                    "output_hash": _hash_file(shard_path),
                }
                _write_json(metadata_path, metadata)
            generated_rows.extend(rows)
            shard_records.append(metadata)

    raw_path = output / "raw_dataset.jsonl"
    _write_jsonl(raw_path, generated_rows)
    accepted = _deduplicate_training_rows(generated_rows)
    accepted_path = output / "accepted_bridge_roots.jsonl"
    _write_jsonl(accepted_path, accepted)
    gate = protocol["coverage_gate"]
    qualified_current_roots: set[str] = set()
    for segment in segments:
        counts = {
            str(row["root_seed_id"]): sum(
                candidate["assigned_segment"] == segment
                and str(candidate["root_seed_id"]) == str(row["root_seed_id"])
                for candidate in accepted
            )
            for row in accepted
            if row["assigned_segment"] == segment
        }
        qualified_current_roots.update(
            root_id
            for root_id, value in counts.items()
            if value >= int(gate["minimum_accepted_rows_per_bridge_root"])
        )
    qualified_accepted = [
        row for row in accepted if str(row["root_seed_id"]) in qualified_current_roots
    ]
    qualified_path = output / "qualified_bridge_roots.jsonl"
    _write_jsonl(qualified_path, qualified_accepted)
    combined = _deduplicate_training_rows([*development_rows, *qualified_accepted])
    combined_path = output / "development_dataset.jsonl"
    _write_jsonl(combined_path, combined)
    if {str(row["root_seed_id"]) for row in combined} & confirmation_roots:
        raise Geometry19V22DatasetError("Confirmation root leaked into expanded development data")

    by_segment: dict[str, Any] = {}
    coverage: dict[str, Any] = {}
    coverage_pass = True
    for segment in segments:
        selected_ids = [item["root_seed_id"] for item in selected[segment]]
        counts = {
            root_id: sum(
                row["assigned_segment"] == segment and str(row["root_seed_id"]) == root_id
                for row in accepted
            )
            for root_id in selected_ids
        }
        qualified = {
            root_id: value
            for root_id, value in counts.items()
            if value >= int(gate["minimum_accepted_rows_per_bridge_root"])
        }
        after = _root_centers(combined, variables, segment_id=segment)
        coverage[segment] = _coverage_summary(
            reference_candidates[segment], before_centers[segment], after
        )
        segment_pass = (
            len(qualified) >= int(gate["minimum_qualified_bridge_roots_per_segment"])
            and coverage[segment]["p95_reduction_fraction"]
            >= float(gate["minimum_bridge_reference_p95_reduction_fraction"])
        )
        coverage_pass = coverage_pass and segment_pass
        by_segment[segment] = {
            "development_roots_before": len(before_centers[segment]),
            "selected_bridge_roots": len(selected_ids),
            "bridge_roots_with_candidates": sum(value > 0 for value in counts.values()),
            "qualified_bridge_roots": len(qualified),
            "accepted_rows": sum(counts.values()),
            "accepted_rows_by_bridge_root": counts,
            "qualified_rows_by_bridge_root": qualified,
            "development_roots_after": len(after),
            "gate_pass": segment_pass,
        }

    fold_spec = protocol["development_folds"]
    fold_target_segments = tuple(fold_spec.get("target_segments", segments))
    total_development_roots_by_target_segment = {
        segment: len(
            {
                str(row["root_seed_id"])
                for row in combined
                if row["assigned_segment"] == segment
            }
        )
        for segment in fold_target_segments
    }
    minimum_total_roots = gate.get("minimum_total_development_roots_per_target_segment")
    total_root_gate_pass = (
        True
        if minimum_total_roots is None
        else all(
            value >= int(minimum_total_roots)
            for value in total_development_roots_by_target_segment.values()
        )
    )
    coverage_pass = coverage_pass and total_root_gate_pass
    assignments, fold_summaries, fold_metadata = _assign_development_folds(
        combined,
        variables,
        fold_target_segments,
        fold_count=int(fold_spec["fold_count"]),
        distance_strata=int(fold_spec["distance_strata"]),
    )
    minimum_fold_roots = int(fold_spec["minimum_validation_roots_per_target_segment"])
    fold_gate_pass = all(
        all(
            summary["validation_roots_by_target_segment"][segment] >= minimum_fold_roots
            for segment in fold_target_segments
        )
        for summary in fold_summaries
    )
    fold_payload = {
        "grouping_key": "root_seed_id",
        "selection_metric": "development_multi_fold_worst_root_rho_p95",
        "confirmation_reuse_allowed": False,
        "root_assignments": dict(sorted(assignments.items())),
        "fold_summaries": fold_summaries,
        **fold_metadata,
    }
    fold_path = output / "development_root_folds.json"
    _write_json(fold_path, fold_payload)

    gate_pass = coverage_pass and fold_gate_pass
    manifest = {
        "dataset_version": dataset_version,
        "status": "BRIDGE_DEVELOPMENT_GATE_PASS" if gate_pass else "BRIDGE_DEVELOPMENT_GATE_FAIL",
        "formal_use_allowed": False,
        "confirmation_reuse_allowed": False,
        "deployment_allowed": False,
        "default_pipeline_changed": False,
        "development_retraining_ready": gate_pass,
        "counts": {
            "evaluated": len(generated_rows),
            "physics_domain_valid": sum(
                row["constraint_assessment"]["physics_domain_valid"] for row in generated_rows
            ),
            "accepted_unique": len(accepted),
            "qualified_accepted_unique": len(qualified_accepted),
            "development_base_rows": len(development_rows),
            "development_combined_unique": len(combined),
            "confirmation_rows_sealed": len(confirmation_rows),
            "confirmation_roots_sealed": len(confirmation_roots),
            "prior_qualified_development_rows": len(prior_rows),
            "development_roots_by_target_segment": total_development_roots_by_target_segment,
            "by_segment": by_segment,
        },
        "coverage_gate_pass": coverage_pass,
        "fold_gate_pass": fold_gate_pass,
        "coverage_gate": gate,
        "coverage": coverage,
        "development_folds": {**fold_spec, **fold_payload},
        "sampling": sampling,
        "prior_expansion_audit": prior_expansion_audit,
        "bridge_selection_audit": {
            segment: [
                {key: value for key, value in item.items() if key != "center_x"}
                for item in values
            ]
            for segment, values in selected.items()
        },
        "model_audit": model_audit,
        "shards": shard_records,
        "frozen_split_status": split_manifest["status"],
        "hashes": {
            "protocol_hash": protocol_hash,
            "frozen_split_manifest_hash": _hash_file(
                (root / protocol["frozen_split"]["manifest_path"]).resolve()
            ),
            "raw_dataset_hash": _hash_file(raw_path),
            "accepted_bridge_roots_hash": _hash_file(accepted_path),
            "qualified_bridge_roots_hash": _hash_file(qualified_path),
            "development_dataset_hash": _hash_file(combined_path),
            "development_root_folds_hash": _hash_file(fold_path),
            "sealed_confirmation_hash": protocol["frozen_split"]["confirmation"]["sha256"],
            "pipeline_context_hash": context.context_hash,
            "physics_bundle_hash": context.physics_runtime.physics_bundle_hash,
            "target_domain_hash": domain.source_hash,
        },
        "paths": {
            "raw_dataset": _artifact_path(raw_path, root),
            "accepted_bridge_roots": _artifact_path(accepted_path, root),
            "qualified_bridge_roots": _artifact_path(qualified_path, root),
            "development_dataset": _artifact_path(combined_path, root),
            "development_root_folds": _artifact_path(fold_path, root),
        },
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    report_path = output / "Bridge_Root_Expansion_Report.md"
    report_path.write_text(_render_report(manifest), encoding="utf-8")
    return manifest
