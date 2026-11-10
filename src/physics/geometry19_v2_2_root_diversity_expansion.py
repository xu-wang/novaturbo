"""Independent-root expansion for weak V2.2 low-mass target segments."""

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
)


ROOT_DIVERSITY_VERSION = "geometry19_v2_2_independent_root_expansion_v1"


def _load_protocol(root: Path, source: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(source)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("dataset_version") != ROOT_DIVERSITY_VERSION:
        raise Geometry19V22DatasetError("Independent-root protocol version mismatch")
    if payload.get("profile") != "v2_2_development":
        raise Geometry19V22DatasetError("Independent-root expansion requires v2_2_development")
    if payload.get("formal_use_allowed") is not False:
        raise Geometry19V22DatasetError("Independent-root expansion cannot be formal")
    segments = payload.get("sampling", {}).get("segments")
    rounds = payload.get("sampling", {}).get("rounds")
    if not isinstance(segments, list) or len(segments) != 2 or len(set(segments)) != 2:
        raise Geometry19V22DatasetError("Exactly two weak low-mass segments are required")
    if not isinstance(rounds, list) or not rounds:
        raise Geometry19V22DatasetError("Independent-root rounds are missing")
    return path, payload


def _valid_by_root(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("performance"):
            continue
        if not row.get("constraint_assessment", {}).get("physics_domain_valid", False):
            continue
        grouped[str(row["root_seed_id"])].append(row)
    return grouped


def _select_novel_root_models(
    domain: Any,
    variables: Sequence[Mapping[str, Any]],
    observations: Sequence[dict[str, Any]],
    existing_rows: Sequence[dict[str, Any]],
    segment_id: str,
    sampling: Mapping[str, Any],
    first_round: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Preselect roots before new evaluations using only frozen source observations."""
    grouped = _valid_by_root(observations)
    existing_segment = [
        row for row in existing_rows if row["assigned_segment"] == segment_id
    ]
    existing_roots = {str(row["root_seed_id"]) for row in existing_segment}
    existing_x = np.stack(
        [_normalized_design(row, variables) for row in existing_segment]
    )
    eligible = [
        rows
        for root_id, rows in grouped.items()
        if root_id not in existing_roots
        and len(rows) >= int(sampling["minimum_observations_per_root"])
    ]
    eligible.sort(key=lambda rows: str(rows[0]["root_seed_id"]))
    candidates = [
        _fit_root_model(
            domain,
            variables,
            segment_id,
            rows,
            maximum_observations=int(sampling["maximum_observations_per_root"]),
            ridge=float(sampling["ridge"]),
            inverse_ridge=float(sampling["inverse_ridge"]),
            trust_radius=float(first_round["trust_radius_fraction"]),
            step_gain=float(first_round["step_gain"]),
        )
        for rows in eligible
    ]
    candidates.sort(
        key=lambda model: (
            float(model["anchor_distance"]),
            str(model["root_seed_id"]),
        )
    )
    threshold = float(sampling["minimum_normalized_l2_separation"])
    selected: list[dict[str, Any]] = []
    selected_centers: list[np.ndarray] = []
    for model in candidates:
        center = np.asarray(model["center_x"], dtype=float)
        distance_existing = float(
            np.min(np.linalg.norm(existing_x - center[None, :], axis=1))
        )
        distance_selected = (
            float(
                np.min(
                    np.linalg.norm(
                        np.stack(selected_centers) - center[None, :], axis=1
                    )
                )
            )
            if selected_centers
            else None
        )
        if distance_existing < threshold:
            continue
        if distance_selected is not None and distance_selected < threshold:
            continue
        model["minimum_l2_to_existing_segment_rows"] = distance_existing
        model["minimum_l2_to_selected_novel_centers"] = distance_selected
        selected.append(model)
        selected_centers.append(center)
        if len(selected) == int(sampling["root_pool_size_per_segment"]):
            break
    if len(selected) < int(sampling["root_pool_size_per_segment"]):
        raise Geometry19V22DatasetError(
            f"Insufficient separated novel roots for {segment_id}: {len(selected)}"
        )
    return selected


def _render_report(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Geometry19 V2.2 低质量线段独立根扩充报告",
        "",
        f"- 状态：`{manifest['status']}`",
        f"- 新评价：{manifest['counts']['evaluated']}",
        f"- physics-domain valid：{manifest['counts']['physics_domain_valid']}",
        f"- 新候选：{manifest['counts']['accepted_unique']}",
        f"- 合并候选：{manifest['counts']['combined_accepted_unique']}",
        "- 正式用途：禁止",
        "- 代理重训：本切片不执行",
        "",
        "## 新根结果",
        "",
        "| 线段 | 既有根 | 选定新根 | 命中新根 | 合格新根 | 合并后根 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for segment in manifest["sampling"]["segments"]:
        values = manifest["counts"]["by_segment"][segment]
        lines.append(
            f"| `{segment}` | {values['existing_roots']} | {values['selected_novel_roots']} | "
            f"{values['novel_roots_with_candidates']} | {values['qualified_novel_roots']} | "
            f"{values['combined_roots']} |"
        )
    lines.extend(
        [
            "",
            "每个合格新根必须至少贡献协议规定数量的、由统一 V2.2 物理运行时重新计算的走廊内候选。根选择排除了该线段已有 root_seed_id，并在归一化 19D 中执行最小 L2 分离门。",
            "",
            "已有 baseline 的 test 已被查看，因此本切片不擅自重分 train/validation/test，也不把合并数据称为新的无泄漏训练集。下一步需在重训前单独冻结新的根级开发/确认角色。",
            "",
        ]
    )
    return "\n".join(lines)


def run_root_diversity_expansion(
    repository_root: str | Path,
    protocol_path: str | Path,
    output_root: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Generate V2.2-labelled candidates from roots absent in weak segments."""
    root = Path(repository_root).resolve()
    protocol_source, protocol = _load_protocol(root, protocol_path)
    output = Path(output_root).resolve()
    if output.exists() and not resume:
        raise Geometry19V22DatasetError(f"Refusing to overwrite expansion: {output}")
    output.mkdir(parents=True, exist_ok=True)
    context = build_pipeline_context(root, pipeline_profile="v2_2_development")
    domain = context.target_domain
    assert domain is not None
    sampling = protocol["sampling"]
    configured_segments = tuple(str(value) for value in sampling["segments"])
    known_segments = {segment[0] for segment in domain.segments}
    if not set(configured_segments) < known_segments:
        raise Geometry19V22DatasetError("Expansion segments must be a strict domain subset")

    source_spec = protocol["source_observations"]
    source_path = (root / source_spec["path"]).resolve()
    if _hash_file(source_path) != source_spec["sha256"]:
        raise Geometry19V22DatasetError("Source-observation hash mismatch")
    base_spec = protocol["base_dataset"]
    base_path = (root / base_spec["path"]).resolve()
    if _hash_file(base_path) != base_spec["sha256"]:
        raise Geometry19V22DatasetError("Base accepted-dataset hash mismatch")
    source_rows = _read_jsonl(source_path)
    base_rows = _read_jsonl(base_path)
    variables = list(context.physics_runtime.design_schema["design_variables"])
    first_round = sampling["rounds"][0]
    selected = {
        segment: _select_novel_root_models(
            domain,
            variables,
            source_rows,
            base_rows,
            segment,
            sampling,
            first_round,
        )
        for segment in configured_segments
    }
    selected_ids = {
        segment: tuple(str(model["root_seed_id"]) for model in models)
        for segment, models in selected.items()
    }
    protocol_hash = _hash_file(protocol_source)
    recovery_rows: list[dict[str, Any]] = []
    model_audit: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    for round_index, round_spec in enumerate(sampling["rounds"]):
        prior_rows = tuple(recovery_rows)
        observations = [*source_rows, *prior_rows]
        grouped = _valid_by_root(observations)
        accepted_prior = _deduplicate_training_rows(prior_rows)
        accepted_counts = {
            (segment, root_id): sum(
                row["assigned_segment"] == segment
                and str(row["root_seed_id"]) == root_id
                for row in accepted_prior
            )
            for segment in configured_segments
            for root_id in selected_ids[segment]
        }
        for segment_index, segment in enumerate(configured_segments):
            active_ids = [
                root_id
                for root_id in selected_ids[segment]
                if accepted_counts[(segment, root_id)]
                < int(sampling["target_accepted_rows_per_root"])
            ]
            models = [
                _fit_root_model(
                    domain,
                    variables,
                    segment,
                    grouped[root_id],
                    maximum_observations=int(sampling["maximum_observations_per_root"]),
                    ridge=float(sampling["ridge"]),
                    inverse_ridge=float(sampling["inverse_ridge"]),
                    trust_radius=float(round_spec["trust_radius_fraction"]),
                    step_gain=float(round_spec["step_gain"]),
                )
                for root_id in active_ids
            ]
            model_audit.append(
                {
                    "adaptive_round": round_index,
                    "assigned_segment": segment,
                    "active_root_ids": active_ids,
                    "accepted_before_round": {
                        root_id: accepted_counts[(segment, root_id)]
                        for root_id in selected_ids[segment]
                    },
                    "models": [
                        {key: value for key, value in model.items() if key != "center_x"}
                        for model in models
                    ],
                }
            )
            sample_count = int(round_spec["samples_per_active_root"]) * len(models)
            plan = (
                _directed_plan(
                    variables,
                    models,
                    segment_id=segment,
                    round_index=round_index,
                    sample_count=sample_count,
                    perturbation_radius=float(round_spec["perturbation_radius_fraction"]),
                    discrete_step_probability=float(sampling["discrete_step_probability"]),
                    random_seed=(
                        int(sampling["random_seed"])
                        + round_index * 1013
                        + segment_index * 89
                    ),
                )
                if models
                else []
            )
            for row in plan:
                row["sample_id"] = row["sample_id"].replace("directed:", "rootdiv:", 1)
            shard_path = output / "shards" / f"round-{round_index:02d}-{segment}.jsonl"
            metadata_path = shard_path.with_suffix(".meta.json")
            identity = {
                "dataset_version": ROOT_DIVERSITY_VERSION,
                "protocol_hash": protocol_hash,
                "pipeline_context_hash": context.context_hash,
                "input_hash": sha256(_canonical_bytes(plan)).hexdigest(),
            }
            if resume and shard_path.exists() and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if any(metadata.get(key) != value for key, value in identity.items()):
                    raise Geometry19V22DatasetError(
                        f"Independent-root shard identity mismatch: {round_index}/{segment}"
                    )
                if metadata.get("output_hash") != _hash_file(shard_path):
                    raise Geometry19V22DatasetError(
                        f"Independent-root shard output hash mismatch: {round_index}/{segment}"
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
            recovery_rows.extend(rows)
            shard_records.append(metadata)

    raw_path = output / "raw_dataset.jsonl"
    _write_jsonl(raw_path, recovery_rows)
    accepted = _deduplicate_training_rows(recovery_rows)
    accepted_path = output / "accepted_new_roots.jsonl"
    _write_jsonl(accepted_path, accepted)
    combined = _deduplicate_training_rows([*base_rows, *accepted])
    combined_path = output / "combined_accepted_dataset.jsonl"
    _write_jsonl(combined_path, combined)

    gate = protocol["diversity_gate"]
    by_segment: dict[str, Any] = {}
    gate_pass = True
    for segment in configured_segments:
        existing_roots = {
            str(row["root_seed_id"])
            for row in base_rows
            if row["assigned_segment"] == segment
        }
        counts = {
            root_id: sum(
                row["assigned_segment"] == segment
                and str(row["root_seed_id"]) == root_id
                for row in accepted
            )
            for root_id in selected_ids[segment]
        }
        qualified = {
            root_id: count
            for root_id, count in counts.items()
            if count >= int(gate["minimum_accepted_rows_per_new_root"])
        }
        combined_roots = {
            str(row["root_seed_id"])
            for row in combined
            if row["assigned_segment"] == segment
        }
        segment_pass = len(qualified) >= int(gate["minimum_qualified_new_roots_per_segment"])
        gate_pass = gate_pass and segment_pass
        by_segment[segment] = {
            "existing_roots": len(existing_roots),
            "selected_novel_roots": len(selected_ids[segment]),
            "novel_roots_with_candidates": sum(count > 0 for count in counts.values()),
            "qualified_novel_roots": len(qualified),
            "accepted_rows": sum(counts.values()),
            "accepted_rows_by_selected_root": counts,
            "qualified_rows_by_root": qualified,
            "combined_roots": len(combined_roots),
            "gate_pass": segment_pass,
        }

    manifest = {
        "dataset_version": ROOT_DIVERSITY_VERSION,
        "status": "ROOT_DIVERSITY_GATE_PASS" if gate_pass else "ROOT_DIVERSITY_GATE_FAIL",
        "formal_use_allowed": False,
        "data_expansion_gate_pass": gate_pass,
        "surrogate_retraining_ready": False,
        "surrogate_retraining_blocker": "fresh_root_level_split_not_frozen",
        "counts": {
            "evaluated": len(recovery_rows),
            "physics_domain_valid": sum(
                row["constraint_assessment"]["physics_domain_valid"] for row in recovery_rows
            ),
            "accepted_unique": len(accepted),
            "combined_accepted_unique": len(combined),
            "by_segment": by_segment,
        },
        "diversity_gate": gate,
        "sampling": sampling,
        "root_selection_audit": {
            segment: [
                {key: value for key, value in model.items() if key != "center_x"}
                for model in models
            ]
            for segment, models in selected.items()
        },
        "model_audit": model_audit,
        "shards": shard_records,
        "hashes": {
            "protocol_hash": protocol_hash,
            "source_observations_hash": _hash_file(source_path),
            "base_dataset_hash": _hash_file(base_path),
            "raw_dataset_hash": _hash_file(raw_path),
            "accepted_new_roots_hash": _hash_file(accepted_path),
            "combined_accepted_dataset_hash": _hash_file(combined_path),
            "pipeline_context_hash": context.context_hash,
            "physics_bundle_hash": context.physics_runtime.physics_bundle_hash,
            "target_domain_hash": domain.source_hash,
        },
        "paths": {
            "raw_dataset": _artifact_path(raw_path, root),
            "accepted_new_roots": _artifact_path(accepted_path, root),
            "combined_accepted_dataset": _artifact_path(combined_path, root),
        },
    }
    _write_json(output / "manifest.json", manifest)
    (output / "Root_Diversity_Report.md").write_text(
        _render_report(manifest), encoding="utf-8"
    )
    return manifest
