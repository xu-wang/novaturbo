"""Segment-balanced adaptive data expansion for simplified Geometry19 V2.2."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.physics.geometry19_v2_2_dataset import (
    Geometry19V22DatasetError,
    _artifact_path,
    _canonical_bytes,
    _evaluate_row,
    _hash_file,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)


BALANCED_DATASET_VERSION = "geometry19_v2_2_segment_balanced_expansion_v1"


def _load_protocol(root: Path, source: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(source)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("dataset_version") != BALANCED_DATASET_VERSION:
        raise Geometry19V22DatasetError("Balanced dataset protocol version mismatch")
    if payload.get("profile") != "v2_2_development":
        raise Geometry19V22DatasetError("Balanced expansion must use v2_2_development")
    if payload.get("formal_use_allowed") is not False:
        raise Geometry19V22DatasetError("Balanced expansion cannot be formal")
    sampling = payload.get("sampling", {})
    segments = sampling.get("segments")
    rounds = sampling.get("rounds")
    if not isinstance(segments, list) or len(segments) != 5 or len(set(segments)) != 5:
        raise Geometry19V22DatasetError("Exactly five unique target-domain segments are required")
    if not isinstance(rounds, list) or not rounds:
        raise Geometry19V22DatasetError("At least one adaptive round is required")
    return path, payload


def _segment_assessment(domain: Any, performance: Mapping[str, Any], segment_id: str) -> dict[str, Any]:
    point = domain._scaled_log_point(performance)
    match = [segment for segment in domain.segments if segment[0] == segment_id]
    if len(match) != 1:
        raise Geometry19V22DatasetError(f"Unknown target-domain segment: {segment_id}")
    _, start_name, end_name = match[0]
    start = tuple(
        math.log(value) / scale
        for value, scale in zip(domain.nodes[start_name], domain.log_scales)
    )
    end = tuple(
        math.log(value) / scale
        for value, scale in zip(domain.nodes[end_name], domain.log_scales)
    )
    distance, fraction = domain._segment_distance(point, start, end)
    inside = distance <= domain.maximum_normalized_distance + 1.0e-12
    return {
        "inside": inside,
        "status": (
            "INSIDE_ASSIGNED_V2_2_TARGET_SEGMENT"
            if inside
            else "OUTSIDE_ASSIGNED_V2_2_TARGET_SEGMENT"
        ),
        "normalized_distance": distance,
        "assigned_segment": segment_id,
        "segment_fraction": fraction,
    }


def _rank_parent_pool(
    domain: Any,
    segment_id: str,
    seed_rows: Sequence[dict[str, Any]],
    evaluated_rows: Sequence[dict[str, Any]],
    pool_size: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in [*seed_rows, *evaluated_rows]:
        performance = row.get("performance")
        if not performance:
            continue
        if "constraint_assessment" in row and not row["constraint_assessment"][
            "physics_domain_valid"
        ]:
            continue
        root_seed_id = str(row.get("root_seed_id") or row.get("seed_id"))
        assessment = _segment_assessment(domain, performance, segment_id)
        candidates.append(
            {
                **row,
                "root_seed_id": root_seed_id,
                "parent_id": str(row.get("sample_id") or row.get("seed_id")),
                "assigned_segment_assessment": assessment,
            }
        )
    candidates.sort(
        key=lambda row: (
            row["assigned_segment_assessment"]["normalized_distance"],
            row["root_seed_id"],
            row["parent_id"],
        )
    )
    diverse: list[dict[str, Any]] = []
    used_roots: set[str] = set()
    for row in candidates:
        if row["root_seed_id"] in used_roots:
            continue
        diverse.append(row)
        used_roots.add(row["root_seed_id"])
        if len(diverse) == pool_size:
            break
    if not diverse:
        raise Geometry19V22DatasetError(f"No valid parent pool for {segment_id}")
    return diverse


def _round_plan(
    schema: Mapping[str, Any],
    parents: Sequence[dict[str, Any]],
    *,
    segment_id: str,
    round_index: int,
    sample_count: int,
    radius: float,
    discrete_step_probability: float,
    random_seed: int,
) -> list[dict[str, Any]]:
    variables = list(schema["design_variables"])
    rng = np.random.default_rng(random_seed)
    lhs = np.empty((sample_count, len(variables)), dtype=float)
    for column in range(len(variables)):
        lhs[:, column] = (
            rng.permutation(sample_count) + rng.random(sample_count)
        ) / sample_count
    offsets = (2.0 * lhs - 1.0) * radius
    parent_indices = np.arange(sample_count) % len(parents)
    rng.shuffle(parent_indices)
    rows: list[dict[str, Any]] = []
    for local_index, parent_index in enumerate(parent_indices):
        parent = parents[int(parent_index)]
        design: dict[str, float | int] = {}
        for column, variable in enumerate(variables):
            name = str(variable["name"])
            lower, upper = (float(value) for value in variable["bounds"])
            if variable["dtype"] == "int64":
                draw = rng.random()
                if draw < discrete_step_probability / 2.0:
                    step = -1
                elif draw < discrete_step_probability:
                    step = 1
                else:
                    step = 0
                value = int(parent["geometry19"][name]) + step
                design[name] = int(min(max(value, int(lower)), int(upper)))
            else:
                value = float(parent["geometry19"][name]) + offsets[
                    local_index, column
                ] * (upper - lower)
                design[name] = min(max(value, lower), upper)
        design_hash = sha256(_canonical_bytes(design)).hexdigest()
        rows.append(
            {
                "sample_index": local_index,
                "sample_id": (
                    f"balanced:{segment_id}:r{round_index}:"
                    f"{local_index:04d}:{design_hash[:12]}"
                ),
                "design_hash": design_hash,
                "assigned_segment": segment_id,
                "adaptive_round": round_index,
                "parent_seed_id": parent["parent_id"],
                "root_seed_id": parent["root_seed_id"],
                "geometry19": design,
            }
        )
    return rows


def _evaluate_assigned(row: dict[str, Any], context: Any) -> dict[str, Any]:
    evaluated = _evaluate_row(row, context)
    assigned = None
    if evaluated["performance"] is not None:
        assigned = _segment_assessment(
            context.target_domain, evaluated["performance"], row["assigned_segment"]
        )
    evaluated["assigned_segment_assessment"] = assigned
    evaluated["training_candidate"] = bool(
        evaluated["constraint_assessment"]["physics_domain_valid"]
        and assigned is not None
        and assigned["inside"]
    )
    return evaluated


def _deduplicate_training_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row["training_candidate"]:
            continue
        existing = best.get(row["design_hash"])
        if existing is None or row["assigned_segment_assessment"][
            "normalized_distance"
        ] < existing["assigned_segment_assessment"]["normalized_distance"]:
            best[row["design_hash"]] = row
    return sorted(
        best.values(), key=lambda row: (row["assigned_segment"], row["design_hash"])
    )


def _grouped_split(
    rows: Sequence[dict[str, Any]], fractions: Mapping[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    names = ("train", "validation", "test")
    weights = {name: float(fractions[name]) for name in names}
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise Geometry19V22DatasetError("Split fractions must sum to one")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["assigned_segment"], row["root_seed_id"])].append(row)
    output = {name: [] for name in names}
    assignments: dict[str, str] = {}
    by_segment: dict[str, list[tuple[tuple[str, str], list[dict[str, Any]]]]] = defaultdict(list)
    for key, group_rows in groups.items():
        by_segment[key[0]].append((key, group_rows))
    for segment_id, segment_groups in sorted(by_segment.items()):
        segment_groups.sort(key=lambda item: (-len(item[1]), item[0][1]))
        counts = {name: 0 for name in names}
        total = sum(len(item[1]) for item in segment_groups)
        for index, (key, group_rows) in enumerate(segment_groups):
            remaining_names = names[index:] if index < len(names) else names
            if index < len(names):
                selected = remaining_names[0]
            else:
                selected = min(
                    names,
                    key=lambda name: (
                        counts[name] / max(weights[name] * total, 1.0e-12), name
                    ),
                )
            output[selected].extend(group_rows)
            counts[selected] += len(group_rows)
            assignments[f"{segment_id}:{key[1]}"] = selected
    for name in names:
        output[name].sort(
            key=lambda row: (row["assigned_segment"], row["design_hash"])
        )
    return output, assignments


def run_segment_balanced_expansion(
    repository_root: str | Path,
    protocol_path: str | Path,
    output_root: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run deterministic adaptive rounds, then freeze grouped dataset splits."""

    root = Path(repository_root).resolve()
    protocol_source, protocol = _load_protocol(root, protocol_path)
    output = Path(output_root).resolve()
    if output.exists() and not resume:
        raise Geometry19V22DatasetError(f"Refusing to overwrite expansion: {output}")
    output.mkdir(parents=True, exist_ok=True)
    context = build_pipeline_context(root, pipeline_profile="v2_2_development")
    domain = context.target_domain
    assert domain is not None
    configured_segments = tuple(protocol["sampling"]["segments"])
    if set(configured_segments) != {segment[0] for segment in domain.segments}:
        raise Geometry19V22DatasetError("Protocol segments differ from target domain")
    seed_path = (root / protocol["seed_pack"]["path"]).resolve()
    if _hash_file(seed_path) != protocol["seed_pack"]["sha256"]:
        raise Geometry19V22DatasetError("Seed-pack hash mismatch")
    seeds = _read_jsonl(seed_path)
    protocol_hash = _hash_file(protocol_source)
    all_rows: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    parent_audit: list[dict[str, Any]] = []
    sampling = protocol["sampling"]
    for round_index, round_spec in enumerate(sampling["rounds"]):
        prior_round_rows = tuple(all_rows)
        for segment_index, segment_id in enumerate(configured_segments):
            parents = _rank_parent_pool(
                domain,
                segment_id,
                seeds,
                prior_round_rows,
                int(sampling["parent_pool_size"]),
            )
            parent_audit.append(
                {
                    "adaptive_round": round_index,
                    "assigned_segment": segment_id,
                    "parents": [
                        {
                            "parent_id": row["parent_id"],
                            "root_seed_id": row["root_seed_id"],
                            "normalized_distance": row[
                                "assigned_segment_assessment"
                            ]["normalized_distance"],
                        }
                        for row in parents
                    ],
                }
            )
            plan = _round_plan(
                context.physics_runtime.design_schema,
                parents,
                segment_id=segment_id,
                round_index=round_index,
                sample_count=int(round_spec["samples_per_segment"]),
                radius=float(round_spec["local_radius_fraction"]),
                discrete_step_probability=float(
                    sampling["discrete_step_probability"]
                ),
                random_seed=(
                    int(sampling["random_seed"])
                    + round_index * 1009
                    + segment_index * 97
                ),
            )
            shard_path = (
                output
                / "shards"
                / f"round-{round_index:02d}-{segment_id}.jsonl"
            )
            metadata_path = shard_path.with_suffix(".meta.json")
            identity = {
                "dataset_version": BALANCED_DATASET_VERSION,
                "protocol_hash": protocol_hash,
                "pipeline_context_hash": context.context_hash,
                "input_hash": sha256(_canonical_bytes(plan)).hexdigest(),
            }
            if resume and shard_path.exists() and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if any(metadata.get(key) != value for key, value in identity.items()):
                    raise Geometry19V22DatasetError(
                        f"Adaptive shard identity mismatch: {round_index}/{segment_id}"
                    )
                if metadata.get("output_hash") != _hash_file(shard_path):
                    raise Geometry19V22DatasetError(
                        f"Adaptive shard output hash mismatch: {round_index}/{segment_id}"
                    )
                rows = _read_jsonl(shard_path)
            else:
                rows = [_evaluate_assigned(row, context) for row in plan]
                _write_jsonl(shard_path, rows)
                metadata = {
                    **identity,
                    "adaptive_round": round_index,
                    "assigned_segment": segment_id,
                    "row_count": len(rows),
                    "training_candidate_count": sum(
                        row["training_candidate"] for row in rows
                    ),
                    "output_hash": _hash_file(shard_path),
                }
                _write_json(metadata_path, metadata)
            all_rows.extend(rows)
            shard_records.append(metadata)
    raw_path = output / "raw_dataset.jsonl"
    _write_jsonl(raw_path, all_rows)
    accepted = _deduplicate_training_rows(all_rows)
    accepted_path = output / "accepted_dataset.jsonl"
    _write_jsonl(accepted_path, accepted)
    splits, group_assignments = _grouped_split(accepted, protocol["split"])
    split_paths: dict[str, Path] = {}
    for split_name, rows in splits.items():
        path = output / "splits" / f"{split_name}.jsonl"
        _write_jsonl(path, rows)
        split_paths[split_name] = path
    segment_counts = {
        segment_id: sum(row["assigned_segment"] == segment_id for row in accepted)
        for segment_id in configured_segments
    }
    segment_root_counts = {
        segment_id: len(
            {
                row["root_seed_id"]
                for row in accepted
                if row["assigned_segment"] == segment_id
            }
        )
        for segment_id in configured_segments
    }
    gate = protocol["coverage_gate"]
    coverage_pass = all(
        segment_counts[segment_id] >= int(gate["minimum_rows_per_segment"])
        and segment_root_counts[segment_id]
        >= int(gate["minimum_root_seeds_per_segment"])
        for segment_id in configured_segments
    )
    manifest = {
        "dataset_version": BALANCED_DATASET_VERSION,
        "status": "COVERAGE_GATE_PASS" if coverage_pass else "COVERAGE_GATE_FAIL",
        "formal_use_allowed": False,
        "training_ready": coverage_pass,
        "coverage_gate_pass": coverage_pass,
        "counts": {
            "evaluated": len(all_rows),
            "physics_domain_valid": sum(
                row["constraint_assessment"]["physics_domain_valid"]
                for row in all_rows
            ),
            "accepted_unique": len(accepted),
            "by_segment": segment_counts,
            "root_seeds_by_segment": segment_root_counts,
            "splits": {name: len(rows) for name, rows in splits.items()},
        },
        "coverage_gate": gate,
        "sampling": sampling,
        "split": protocol["split"],
        "group_assignments": group_assignments,
        "parent_audit": parent_audit,
        "shards": shard_records,
        "hashes": {
            "protocol_hash": protocol_hash,
            "seed_pack_hash": _hash_file(seed_path),
            "raw_dataset_hash": _hash_file(raw_path),
            "accepted_dataset_hash": _hash_file(accepted_path),
            "pipeline_context_hash": context.context_hash,
            "physics_bundle_hash": context.physics_runtime.physics_bundle_hash,
            "target_domain_hash": domain.source_hash,
            **{
                f"{name}_split_hash": _hash_file(path)
                for name, path in split_paths.items()
            },
        },
        "paths": {
            "raw_dataset": _artifact_path(raw_path, root),
            "accepted_dataset": _artifact_path(accepted_path, root),
            "splits": {
                name: _artifact_path(path, root) for name, path in split_paths.items()
            },
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest
