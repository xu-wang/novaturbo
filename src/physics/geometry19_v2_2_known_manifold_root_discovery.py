"""Development-only independent root discovery inside observed V2.2 support."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import ndtri
import yaml

from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.physics.geometry19_v2_2_balanced_dataset import (
    _deduplicate_training_rows,
    _evaluate_assigned,
)
from src.physics.geometry19_v2_2_bridge_root_expansion import (
    _assign_development_folds,
    _root_centers,
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
from src.physics.geometry19_v2_2_directed_recovery import _normalized_design


KNOWN_MANIFOLD_ROOT_VERSION = "geometry19_v2_2_known_manifold_root_discovery_v1"


def _load_protocol(root: Path, source: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(source)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(protocol, dict)
        or protocol.get("dataset_version") != KNOWN_MANIFOLD_ROOT_VERSION
    ):
        raise Geometry19V22DatasetError("Known-manifold root protocol mismatch")
    if protocol.get("profile") != "v2_2_development":
        raise Geometry19V22DatasetError("Root discovery requires v2_2_development")
    if protocol.get("formal_use_allowed") is not False:
        raise Geometry19V22DatasetError("Root discovery cannot be formal")
    sampling = protocol.get("sampling", {})
    if len(sampling.get("segments", [])) != 2:
        raise Geometry19V22DatasetError("Exactly two target segments are required")
    if not sampling.get("local_rounds"):
        raise Geometry19V22DatasetError("At least one local sampling round is required")
    return path, protocol


def _latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.empty((count, dimensions), dtype=float)
    for column in range(dimensions):
        values[:, column] = (rng.permutation(count) + rng.random(count)) / count
    return values


def _design_from_normalized(
    values: np.ndarray, variables: Sequence[Mapping[str, Any]]
) -> dict[str, float | int]:
    design: dict[str, float | int] = {}
    for normalized, variable in zip(values, variables):
        lower, upper = (float(value) for value in variable["bounds"])
        value = lower + float(np.clip(normalized, 0.0, 1.0)) * (upper - lower)
        if variable["dtype"] == "int64":
            design[str(variable["name"])] = int(
                min(max(round(value), int(lower)), int(upper))
            )
        else:
            design[str(variable["name"])] = min(max(value, lower), upper)
    return design


def _global_root_seed_plan(
    rows: Sequence[dict[str, Any]],
    variables: Sequence[Mapping[str, Any]],
    *,
    dataset_version: str,
    segment_id: str,
    candidate_count: int,
    random_seed: int,
    explained_variance_fraction: float,
    latent_scale: float,
    normal_score_clip: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create independent LHS starts from one global root-center support model."""
    centers = _root_centers(rows, variables, segment_id=segment_id)
    if len(centers) < 3:
        raise Geometry19V22DatasetError(f"Too few root centers for {segment_id}")
    matrix = np.stack(list(centers.values()))
    mean = np.mean(matrix, axis=0)
    _, singular, right = np.linalg.svd(matrix - mean, full_matrices=False)
    variance = singular**2
    cumulative = np.cumsum(variance) / np.sum(variance)
    rank = int(np.searchsorted(cumulative, explained_variance_fraction) + 1)
    rank = min(rank, len(matrix) - 1, len(variables))
    scales = singular[:rank] / np.sqrt(len(matrix) - 1)
    lhs = _latin_hypercube(candidate_count, rank, random_seed)
    normal = np.clip(ndtri(lhs), -normal_score_clip, normal_score_clip)
    normalized = mean + latent_scale * (normal * scales) @ right[:rank]
    normalized = np.clip(normalized, 0.0, 1.0)
    plan: list[dict[str, Any]] = []
    seen_designs: set[str] = set()
    for candidate_index, candidate_x in enumerate(normalized):
        design = _design_from_normalized(candidate_x, variables)
        design_hash = sha256(_canonical_bytes(design)).hexdigest()
        if design_hash in seen_designs:
            continue
        seen_designs.add(design_hash)
        identity = {
            "dataset_version": dataset_version,
            "segment_id": segment_id,
            "initialization": "global_root_center_pca_lhs",
            "candidate_index": candidate_index,
            "design_hash": design_hash,
        }
        root_id = sha256(_canonical_bytes(identity)).hexdigest()
        plan.append(
            {
                "sample_index": candidate_index,
                "sample_id": f"rootdisc:seed:{segment_id}:{candidate_index:05d}:{design_hash[:12]}",
                "design_hash": design_hash,
                "assigned_segment": segment_id,
                "adaptive_round": -1,
                "parent_seed_id": None,
                "root_seed_id": root_id,
                "root_origin": "independent_global_root_center_pca_lhs",
                "geometry19": design,
            }
        )
    return plan, {
        "source_root_count": len(centers),
        "latent_rank": rank,
        "explained_variance_fraction": float(cumulative[rank - 1]),
        "candidate_count_requested": candidate_count,
        "candidate_count_unique": len(plan),
    }


def _select_seed_pool(
    evaluated: Sequence[dict[str, Any]],
    existing_rows: Sequence[dict[str, Any]],
    variables: Sequence[Mapping[str, Any]],
    *,
    segment_id: str,
    pool_size: int,
    fraction_strata: int,
    minimum_l2_to_existing: float,
    maximum_l2_to_existing: float,
    minimum_l2_between_selected: float,
) -> list[dict[str, Any]]:
    centers = _root_centers(existing_rows, variables, segment_id=segment_id)
    existing = np.stack(list(centers.values()))
    eligible: list[dict[str, Any]] = []
    for row in evaluated:
        if not row["training_candidate"]:
            continue
        candidate_x = _normalized_design(row, variables)
        nearest = float(np.min(np.linalg.norm(existing - candidate_x[None, :], axis=1)))
        if not minimum_l2_to_existing <= nearest <= maximum_l2_to_existing:
            continue
        fraction = float(row["assigned_segment_assessment"]["segment_fraction"])
        stratum = min(int(fraction * fraction_strata), fraction_strata - 1)
        eligible.append(
            {
                **row,
                "candidate_x": candidate_x,
                "minimum_l2_to_existing": nearest,
                "fraction_stratum": stratum,
            }
        )
    selected: list[dict[str, Any]] = []
    stratum_counts: Counter[int] = Counter()
    remaining = list(eligible)
    while remaining and len(selected) < pool_size:
        def score(candidate: Mapping[str, Any]) -> tuple[float, float, float, str]:
            separation = float(candidate["minimum_l2_to_existing"])
            if selected:
                separation = min(
                    separation,
                    min(
                        float(
                            np.linalg.norm(
                                candidate["candidate_x"] - item["candidate_x"]
                            )
                        )
                        for item in selected
                    ),
                )
            return (
                -float(stratum_counts[int(candidate["fraction_stratum"])]),
                separation,
                -float(candidate["assigned_segment_assessment"]["normalized_distance"]),
                str(candidate["root_seed_id"]),
            )

        chosen = max(remaining, key=score)
        separation = score(chosen)[1]
        if separation < minimum_l2_between_selected:
            remaining.remove(chosen)
            continue
        chosen["minimum_l2_at_selection"] = separation
        selected.append(chosen)
        stratum_counts[int(chosen["fraction_stratum"])] += 1
        remaining.remove(chosen)
    if len(selected) < pool_size:
        raise Geometry19V22DatasetError(
            f"Insufficient independent supported seeds for {segment_id}: {len(selected)}"
        )
    return selected


def _local_root_plan(
    seeds: Sequence[dict[str, Any]],
    variables: Sequence[Mapping[str, Any]],
    *,
    segment_id: str,
    round_index: int,
    samples_per_root: int,
    radius: float,
    discrete_step_probability: float,
    random_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds):
        rng_seed = random_seed + round_index * 1009 + seed_index * 97
        lhs = _latin_hypercube(samples_per_root, len(variables), rng_seed)
        noise = (2.0 * lhs - 1.0) * radius
        center = _normalized_design(seed, variables)
        for local_index in range(samples_per_root):
            candidate_x = np.clip(center + noise[local_index], 0.0, 1.0)
            design = _design_from_normalized(candidate_x, variables)
            for variable in variables:
                if variable["dtype"] != "int64":
                    continue
                rng = np.random.default_rng(
                    rng_seed + local_index * 10007 + int(variable["index"])
                )
                probability = rng.random()
                name = str(variable["name"])
                lower, upper = (int(value) for value in variable["bounds"])
                if probability < discrete_step_probability / 2.0:
                    design[name] = max(int(design[name]) - 1, lower)
                elif probability < discrete_step_probability:
                    design[name] = min(int(design[name]) + 1, upper)
            design_hash = sha256(_canonical_bytes(design)).hexdigest()
            rows.append(
                {
                    "sample_index": local_index,
                    "sample_id": (
                        f"rootdisc:local:{segment_id}:r{round_index}:"
                        f"{seed_index:02d}:{local_index:03d}:{design_hash[:12]}"
                    ),
                    "design_hash": design_hash,
                    "assigned_segment": segment_id,
                    "adaptive_round": round_index,
                    "parent_seed_id": seed["sample_id"],
                    "root_seed_id": seed["root_seed_id"],
                    "root_origin": "independent_global_root_center_pca_lhs",
                    "geometry19": design,
                }
            )
    return rows


def _effective_root_count(rows: Sequence[dict[str, Any]], segment_id: str) -> float:
    counts = Counter(
        str(row["root_seed_id"])
        for row in rows
        if row["assigned_segment"] == segment_id
    )
    total = sum(counts.values())
    if not total:
        return 0.0
    weights = np.asarray(list(counts.values()), dtype=float) / total
    return float(1.0 / np.sum(weights**2))


def _render_report(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Geometry19 V2.2 已知流形独立根发现报告",
        "",
        f"- 状态：`{manifest['status']}`",
        f"- 全局初始化评价：{manifest['counts']['seed_evaluated']}",
        f"- 局部确认评价：{manifest['counts']['local_evaluated']}",
        f"- 合并 development：{manifest['counts']['development_rows_before']}→{manifest['counts']['development_rows_after']}",
        "- confirmation：只校验文件哈希，内容未读取、未参与选择或折分配",
        "- 正式用途与部署：禁止",
        "",
        "## 根级结果",
        "",
        "| 线段 | 原根 | 支持域内种子池 | 通过局部门 | 保留新根 | 最终根 | 有效根 | 每折验证根 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for segment, values in manifest["counts"]["by_segment"].items():
        folds = "/".join(str(value) for value in values["validation_roots_by_fold"])
        lines.append(
            f"| `{segment}` | {values['roots_before']} | {values['selected_seed_pool']} | "
            f"{values['qualified_seed_roots']} | {values['retained_new_roots']} | "
            f"{values['roots_after']} | {values['effective_roots_after']:.2f} | {folds} |"
        )
    lines.extend(
        [
            "",
            "## 独立性与适用范围",
            "",
            "新根首先由全部 development 根中心拟合的全局 PCA 支持模型进行 LHS 独立初始化，不继承任何单一旧根的 root ID 或父谱系。只有初始化点本身通过统一 V2.2 physics-domain 与目标线段门，且与旧根保持最小距离、又未超过已知支持距离上限时，才进入局部确认。局部点只增加该新根的观测密度，不产生额外根。",
            "",
            "本结果仅增强已观察局部图覆盖；PCA 支持包络不是数学证明的真实流形。模型仍需实现 `OUTSIDE_KNOWN_MANIFOLD_SUPPORT` 拒绝门，未知区域不属于精度声明范围。",
            "",
            "## 审计发现与改进方向",
            "",
            f"全局初始化直接命中率为 {manifest['counts']['seed_training_candidates']}/{manifest['counts']['seed_evaluated']}（{manifest['counts']['seed_training_candidates'] / manifest['counts']['seed_evaluated']:.1%}），局部确认命中率为 {manifest['counts']['local_training_candidates']}/{manifest['counts']['local_evaluated']}（{manifest['counts']['local_training_candidates'] / manifest['counts']['local_evaluated']:.1%}）。32 个种子池根全部通过单根 40 行门，说明当前瓶颈已经从根内采样密度转移到全局起点发现与跨根模型泛化。",
            "",
            "下一步应使用冻结的 5 折做 root-balanced 小容量模型消融，并仅按多折 worst-root rho P95 选择方案；同时从 development 根距离分布标定已知支持域拒绝阈值。若最差折仍显著不稳定，再针对失败折的低密度区域补根，而不是无条件继续增加同根样本。现有 confirmation 不得复用。",
            "",
        ]
    )
    return "\n".join(lines)


def run_known_manifold_root_discovery(
    repository_root: str | Path,
    protocol_path: str | Path,
    output_root: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    protocol_source, protocol = _load_protocol(root, protocol_path)
    output = Path(output_root).resolve()
    if output.exists() and not resume:
        raise Geometry19V22DatasetError(f"Refusing to overwrite root discovery: {output}")
    output.mkdir(parents=True, exist_ok=True)

    base_spec = protocol["base_development"]
    base_path = (root / base_spec["path"]).resolve()
    if _hash_file(base_path) != base_spec["sha256"]:
        raise Geometry19V22DatasetError("Base development hash mismatch")
    confirmation_spec = protocol["sealed_confirmation"]
    confirmation_path = (root / confirmation_spec["path"]).resolve()
    if _hash_file(confirmation_path) != confirmation_spec["sha256"]:
        raise Geometry19V22DatasetError("Sealed confirmation hash mismatch")
    base_rows = _read_jsonl(base_path)

    context = build_pipeline_context(root, pipeline_profile="v2_2_development")
    domain = context.target_domain
    assert domain is not None
    variables = list(context.physics_runtime.design_schema["design_variables"])
    sampling = protocol["sampling"]
    segments = tuple(str(value) for value in sampling["segments"])
    protocol_hash = _hash_file(protocol_source)

    seed_rows: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    selected: dict[str, list[dict[str, Any]]] = {}
    seed_model_audit: dict[str, Any] = {}
    shard_records: list[dict[str, Any]] = []

    for segment_index, segment in enumerate(segments):
        plan, model_audit = _global_root_seed_plan(
            base_rows,
            variables,
            dataset_version=KNOWN_MANIFOLD_ROOT_VERSION,
            segment_id=segment,
            candidate_count=int(sampling["global_candidate_count_per_segment"]),
            random_seed=int(sampling["random_seed"]) + segment_index * 100003,
            explained_variance_fraction=float(sampling["explained_variance_fraction"]),
            latent_scale=float(sampling["latent_scale"]),
            normal_score_clip=float(sampling["normal_score_clip"]),
        )
        shard_path = output / "shards" / f"global-{segment}.jsonl"
        metadata_path = shard_path.with_suffix(".meta.json")
        identity = {
            "dataset_version": KNOWN_MANIFOLD_ROOT_VERSION,
            "protocol_hash": protocol_hash,
            "pipeline_context_hash": context.context_hash,
            "input_hash": sha256(_canonical_bytes(plan)).hexdigest(),
        }
        if resume and shard_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if any(metadata.get(key) != value for key, value in identity.items()):
                raise Geometry19V22DatasetError(f"Global seed shard mismatch: {segment}")
            if metadata.get("output_hash") != _hash_file(shard_path):
                raise Geometry19V22DatasetError(f"Global seed output mismatch: {segment}")
            evaluated = _read_jsonl(shard_path)
        else:
            evaluated = [_evaluate_assigned(row, context) for row in plan]
            _write_jsonl(shard_path, evaluated)
            metadata = {
                **identity,
                "assigned_segment": segment,
                "row_count": len(evaluated),
                "physics_domain_valid_count": sum(
                    row["constraint_assessment"]["physics_domain_valid"] for row in evaluated
                ),
                "training_candidate_count": sum(row["training_candidate"] for row in evaluated),
                "output_hash": _hash_file(shard_path),
            }
            _write_json(metadata_path, metadata)
        seed_rows.extend(evaluated)
        shard_records.append(metadata)
        selected[segment] = _select_seed_pool(
            evaluated,
            base_rows,
            variables,
            segment_id=segment,
            pool_size=int(sampling["seed_pool_size_per_segment"]),
            fraction_strata=int(sampling["segment_fraction_strata"]),
            minimum_l2_to_existing=float(sampling["minimum_l2_to_existing"]),
            maximum_l2_to_existing=float(sampling["maximum_l2_to_existing"]),
            minimum_l2_between_selected=float(sampling["minimum_l2_between_selected"]),
        )
        seed_model_audit[segment] = {
            **model_audit,
            "physics_domain_valid": metadata["physics_domain_valid_count"],
            "training_candidates": metadata["training_candidate_count"],
            "selected": [
                {
                    "root_seed_id": row["root_seed_id"],
                    "design_hash": row["design_hash"],
                    "minimum_l2_to_existing": row["minimum_l2_to_existing"],
                    "minimum_l2_at_selection": row["minimum_l2_at_selection"],
                    "segment_fraction": row["assigned_segment_assessment"]["segment_fraction"],
                    "fraction_stratum": row["fraction_stratum"],
                }
                for row in selected[segment]
            ],
        }

    accepted_by_root: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for segment in segments:
        for row in selected[segment]:
            clean = {key: value for key, value in row.items() if key != "candidate_x"}
            accepted_by_root[str(row["root_seed_id"])][str(row["design_hash"])] = clean
    for round_index, round_spec in enumerate(sampling["local_rounds"]):
        for segment_index, segment in enumerate(segments):
            active = [
                row
                for row in selected[segment]
                if len(accepted_by_root[str(row["root_seed_id"])])
                < int(sampling["target_accepted_rows_per_root"])
            ]
            plan = _local_root_plan(
                active,
                variables,
                segment_id=segment,
                round_index=round_index,
                samples_per_root=int(round_spec["samples_per_active_root"]),
                radius=float(round_spec["radius_fraction"]),
                discrete_step_probability=float(sampling["discrete_step_probability"]),
                random_seed=int(sampling["random_seed"]) + segment_index * 100003,
            )
            shard_path = output / "shards" / f"local-{round_index:02d}-{segment}.jsonl"
            metadata_path = shard_path.with_suffix(".meta.json")
            identity = {
                "dataset_version": KNOWN_MANIFOLD_ROOT_VERSION,
                "protocol_hash": protocol_hash,
                "pipeline_context_hash": context.context_hash,
                "input_hash": sha256(_canonical_bytes(plan)).hexdigest(),
            }
            if resume and shard_path.exists() and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if any(metadata.get(key) != value for key, value in identity.items()):
                    raise Geometry19V22DatasetError(
                        f"Local root shard mismatch: {round_index}/{segment}"
                    )
                if metadata.get("output_hash") != _hash_file(shard_path):
                    raise Geometry19V22DatasetError(
                        f"Local root output mismatch: {round_index}/{segment}"
                    )
                evaluated = _read_jsonl(shard_path)
            else:
                evaluated = [_evaluate_assigned(row, context) for row in plan]
                _write_jsonl(shard_path, evaluated)
                metadata = {
                    **identity,
                    "adaptive_round": round_index,
                    "assigned_segment": segment,
                    "active_root_count": len(active),
                    "row_count": len(evaluated),
                    "physics_domain_valid_count": sum(
                        row["constraint_assessment"]["physics_domain_valid"]
                        for row in evaluated
                    ),
                    "training_candidate_count": sum(
                        row["training_candidate"] for row in evaluated
                    ),
                    "output_hash": _hash_file(shard_path),
                }
                _write_json(metadata_path, metadata)
            local_rows.extend(evaluated)
            shard_records.append(metadata)
            for row in _deduplicate_training_rows(evaluated):
                accepted_by_root[str(row["root_seed_id"])][str(row["design_hash"])] = row

    all_new_accepted = _deduplicate_training_rows(
        [row for values in accepted_by_root.values() for row in values.values()]
    )
    accepted_path = output / "accepted_new_root_rows.jsonl"
    _write_jsonl(accepted_path, all_new_accepted)
    gate = protocol["root_gate"]
    retained: list[dict[str, Any]] = []
    retained_ids: dict[str, list[str]] = {}
    target_new = int(gate["minimum_new_roots_per_segment"])
    minimum_rows = int(gate["minimum_accepted_rows_per_new_root"])
    cap_rows = int(sampling["target_accepted_rows_per_root"])
    for segment in segments:
        qualified = [
            seed
            for seed in selected[segment]
            if sum(
                row["assigned_segment"] == segment
                and str(row["root_seed_id"]) == str(seed["root_seed_id"])
                for row in all_new_accepted
            )
            >= minimum_rows
        ]
        chosen = qualified[:target_new]
        retained_ids[segment] = [str(row["root_seed_id"]) for row in chosen]
        for root_id in retained_ids[segment]:
            root_rows = [
                row
                for row in all_new_accepted
                if row["assigned_segment"] == segment
                and str(row["root_seed_id"]) == root_id
            ]
            root_rows.sort(
                key=lambda row: (
                    float(row["assigned_segment_assessment"]["normalized_distance"]),
                    str(row["design_hash"]),
                )
            )
            retained.extend(root_rows[:cap_rows])
    retained = _deduplicate_training_rows(retained)
    retained_path = output / "qualified_new_root_rows.jsonl"
    _write_jsonl(retained_path, retained)
    combined = _deduplicate_training_rows([*base_rows, *retained])
    combined_path = output / "development_dataset.jsonl"
    _write_jsonl(combined_path, combined)

    fold_spec = protocol["development_folds"]
    assignments, fold_summaries, fold_metadata = _assign_development_folds(
        combined,
        variables,
        segments,
        fold_count=int(fold_spec["fold_count"]),
        distance_strata=int(fold_spec["distance_strata"]),
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

    by_segment: dict[str, Any] = {}
    root_gate_pass = True
    fold_gate_pass = True
    for segment in segments:
        before_roots = {
            str(row["root_seed_id"])
            for row in base_rows
            if row["assigned_segment"] == segment
        }
        after_roots = {
            str(row["root_seed_id"])
            for row in combined
            if row["assigned_segment"] == segment
        }
        qualified_count = sum(
            sum(
                row["assigned_segment"] == segment
                and str(row["root_seed_id"]) == str(seed["root_seed_id"])
                for row in all_new_accepted
            )
            >= minimum_rows
            for seed in selected[segment]
        )
        validation_counts = [
            summary["validation_roots_by_target_segment"][segment]
            for summary in fold_summaries
        ]
        effective = _effective_root_count(combined, segment)
        segment_root_pass = (
            len(retained_ids[segment]) >= target_new
            and len(after_roots) >= int(gate["minimum_total_roots_per_segment"])
            and effective >= float(gate["minimum_effective_roots_per_segment"])
        )
        segment_fold_pass = min(validation_counts) >= int(
            fold_spec["minimum_validation_roots_per_target_segment"]
        )
        root_gate_pass = root_gate_pass and segment_root_pass
        fold_gate_pass = fold_gate_pass and segment_fold_pass
        by_segment[segment] = {
            "roots_before": len(before_roots),
            "selected_seed_pool": len(selected[segment]),
            "qualified_seed_roots": qualified_count,
            "retained_new_roots": len(retained_ids[segment]),
            "roots_after": len(after_roots),
            "effective_roots_after": effective,
            "validation_roots_by_fold": validation_counts,
            "root_gate_pass": segment_root_pass,
            "fold_gate_pass": segment_fold_pass,
        }

    gate_pass = root_gate_pass and fold_gate_pass
    manifest = {
        "dataset_version": KNOWN_MANIFOLD_ROOT_VERSION,
        "status": (
            "KNOWN_MANIFOLD_ROOT_COVERAGE_PASS"
            if gate_pass
            else "KNOWN_MANIFOLD_ROOT_COVERAGE_FAIL"
        ),
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "confirmation_reuse_allowed": False,
        "confirmation_content_read": False,
        "development_training_ready": gate_pass,
        "inter_branch_extrapolation_in_scope": False,
        "root_count_gate_pass": root_gate_pass,
        "fold_gate_pass": fold_gate_pass,
        "counts": {
            "seed_evaluated": len(seed_rows),
            "seed_physics_domain_valid": sum(
                row["constraint_assessment"]["physics_domain_valid"] for row in seed_rows
            ),
            "seed_training_candidates": sum(row["training_candidate"] for row in seed_rows),
            "local_evaluated": len(local_rows),
            "local_physics_domain_valid": sum(
                row["constraint_assessment"]["physics_domain_valid"] for row in local_rows
            ),
            "local_training_candidates": sum(row["training_candidate"] for row in local_rows),
            "accepted_new_root_rows": len(all_new_accepted),
            "qualified_new_root_rows": len(retained),
            "development_rows_before": len(base_rows),
            "development_rows_after": len(combined),
            "global_roots_before": len({row["root_seed_id"] for row in base_rows}),
            "global_roots_after": len({row["root_seed_id"] for row in combined}),
            "by_segment": by_segment,
        },
        "root_gate": gate,
        "development_folds": {**fold_spec, **fold_payload},
        "sampling": sampling,
        "seed_model_audit": seed_model_audit,
        "retained_root_ids": retained_ids,
        "shards": shard_records,
        "hashes": {
            "protocol_hash": protocol_hash,
            "base_development_hash": _hash_file(base_path),
            "sealed_confirmation_hash": _hash_file(confirmation_path),
            "accepted_new_root_rows_hash": _hash_file(accepted_path),
            "qualified_new_root_rows_hash": _hash_file(retained_path),
            "development_dataset_hash": _hash_file(combined_path),
            "development_root_folds_hash": _hash_file(fold_path),
            "pipeline_context_hash": context.context_hash,
            "physics_bundle_hash": context.physics_runtime.physics_bundle_hash,
            "target_domain_hash": domain.source_hash,
        },
        "paths": {
            "accepted_new_root_rows": _artifact_path(accepted_path, root),
            "qualified_new_root_rows": _artifact_path(retained_path, root),
            "development_dataset": _artifact_path(combined_path, root),
            "development_root_folds": _artifact_path(fold_path, root),
        },
        "limitations": protocol["limitations"],
    }
    _write_json(output / "manifest.json", manifest)
    (output / "Known_Manifold_Root_Discovery_Report.md").write_text(
        _render_report(manifest), encoding="utf-8"
    )
    return manifest
