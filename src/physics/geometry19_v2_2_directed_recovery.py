"""Local-sensitivity directed recovery for under-covered V2.2 target segments."""

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
from src.physics.geometry19_v2_2_balanced_dataset import (
    _deduplicate_training_rows,
    _evaluate_assigned,
    _grouped_split,
    _segment_assessment,
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


DIRECTED_RECOVERY_VERSION = "geometry19_v2_2_directed_segment_recovery_v1"
PERFORMANCE_AXES = ("thrust_N", "tsfc_kg_N_s", "mass_kg")


def _load_protocol(root: Path, source: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(source)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("dataset_version") != DIRECTED_RECOVERY_VERSION:
        raise Geometry19V22DatasetError("Directed recovery protocol version mismatch")
    if payload.get("profile") != "v2_2_development":
        raise Geometry19V22DatasetError("Directed recovery must use v2_2_development")
    if payload.get("formal_use_allowed") is not False:
        raise Geometry19V22DatasetError("Directed recovery cannot be formal")
    rounds = payload.get("sampling", {}).get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise Geometry19V22DatasetError("Directed recovery rounds are missing")
    return path, payload


def _normalized_design(
    row: Mapping[str, Any], variables: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    return np.asarray(
        [
            (float(row["geometry19"][variable["name"]]) - float(variable["bounds"][0]))
            / (float(variable["bounds"][1]) - float(variable["bounds"][0]))
            for variable in variables
        ],
        dtype=float,
    )


def _scaled_performance(domain: Any, performance: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(domain._scaled_log_point(performance), dtype=float)


def _segment_projection(domain: Any, performance: Mapping[str, Any], segment_id: str) -> np.ndarray:
    assessment = _segment_assessment(domain, performance, segment_id)
    _, start_name, end_name = next(
        segment for segment in domain.segments if segment[0] == segment_id
    )
    start = np.asarray(
        [
            math.log(value) / scale
            for value, scale in zip(domain.nodes[start_name], domain.log_scales)
        ],
        dtype=float,
    )
    end = np.asarray(
        [
            math.log(value) / scale
            for value, scale in zip(domain.nodes[end_name], domain.log_scales)
        ],
        dtype=float,
    )
    return start + float(assessment["segment_fraction"]) * (end - start)


def _fit_root_model(
    domain: Any,
    variables: Sequence[Mapping[str, Any]],
    segment_id: str,
    rows: Sequence[dict[str, Any]],
    *,
    maximum_observations: int,
    ridge: float,
    inverse_ridge: float,
    trust_radius: float,
    step_gain: float,
) -> dict[str, Any]:
    ranked = sorted(
        rows,
        key=lambda row: (
            _segment_assessment(domain, row["performance"], segment_id)[
                "normalized_distance"
            ],
            row.get("sample_id", row.get("design_hash", "")),
        ),
    )
    anchor = ranked[0]
    anchor_x = _normalized_design(anchor, variables)
    local = sorted(
        ranked,
        key=lambda row: (
            float(np.max(np.abs(_normalized_design(row, variables) - anchor_x))),
            row.get("sample_id", row.get("design_hash", "")),
        ),
    )[:maximum_observations]
    x = np.stack([_normalized_design(row, variables) for row in local])
    y = np.stack([_scaled_performance(domain, row["performance"]) for row in local])
    anchor_y = _scaled_performance(domain, anchor["performance"])
    delta_x = x - anchor_x
    delta_y = y - anchor_y
    coefficients = np.linalg.solve(
        delta_x.T @ delta_x + ridge * np.eye(delta_x.shape[1]),
        delta_x.T @ delta_y,
    )
    jacobian = coefficients.T
    target_y = _segment_projection(domain, anchor["performance"], segment_id)
    desired = target_y - anchor_y
    step = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + inverse_ridge * np.eye(len(PERFORMANCE_AXES)),
        desired,
    )
    step = np.clip(step_gain * step, -trust_radius, trust_radius)
    center_x = np.clip(anchor_x + step, 0.0, 1.0)
    predicted_y = anchor_y + jacobian @ (center_x - anchor_x)
    predicted_performance = {
        axis: math.exp(value * scale)
        for axis, value, scale in zip(
            PERFORMANCE_AXES, predicted_y, domain.log_scales
        )
    }
    fitted = anchor_y + delta_x @ coefficients
    rmse = np.sqrt(np.mean((fitted - y) ** 2, axis=0))
    return {
        "root_seed_id": str(anchor["root_seed_id"]),
        "anchor_id": str(anchor.get("sample_id") or anchor["design_hash"]),
        "anchor_design": anchor["geometry19"],
        "anchor_distance": _segment_assessment(
            domain, anchor["performance"], segment_id
        )["normalized_distance"],
        "center_x": center_x,
        "observation_count": len(local),
        "jacobian_rank": int(np.linalg.matrix_rank(jacobian)),
        "fit_rmse_scaled": {
            axis: float(value) for axis, value in zip(PERFORMANCE_AXES, rmse)
        },
        "step_linf": float(np.max(np.abs(center_x - anchor_x))),
        "predicted_distance": _segment_assessment(
            domain, predicted_performance, segment_id
        )["normalized_distance"],
    }


def _select_models(
    domain: Any,
    variables: Sequence[Mapping[str, Any]],
    observations: Sequence[dict[str, Any]],
    segment_id: str,
    sampling: Mapping[str, Any],
    round_spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        if not row.get("performance"):
            continue
        if not row.get("constraint_assessment", {}).get("physics_domain_valid", False):
            continue
        grouped[str(row["root_seed_id"])].append(row)
    eligible = [
        rows
        for rows in grouped.values()
        if len(rows) >= int(sampling["minimum_observations_per_root"])
    ]
    eligible.sort(
        key=lambda rows: (
            min(
                _segment_assessment(domain, row["performance"], segment_id)[
                    "normalized_distance"
                ]
                for row in rows
            ),
            rows[0]["root_seed_id"],
        )
    )
    selected = eligible[: int(sampling["root_model_count"])]
    if len(selected) < int(sampling["minimum_root_models"]):
        raise Geometry19V22DatasetError(
            f"Insufficient root models for directed segment {segment_id}"
        )
    return [
        _fit_root_model(
            domain,
            variables,
            segment_id,
            rows,
            maximum_observations=int(sampling["maximum_observations_per_root"]),
            ridge=float(sampling["ridge"]),
            inverse_ridge=float(sampling["inverse_ridge"]),
            trust_radius=float(round_spec["trust_radius_fraction"]),
            step_gain=float(round_spec["step_gain"]),
        )
        for rows in selected
    ]


def _directed_plan(
    variables: Sequence[Mapping[str, Any]],
    models: Sequence[dict[str, Any]],
    *,
    segment_id: str,
    round_index: int,
    sample_count: int,
    perturbation_radius: float,
    discrete_step_probability: float,
    random_seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(random_seed)
    lhs = np.empty((sample_count, len(variables)), dtype=float)
    for column in range(len(variables)):
        lhs[:, column] = (
            rng.permutation(sample_count) + rng.random(sample_count)
        ) / sample_count
    noise = (2.0 * lhs - 1.0) * perturbation_radius
    model_indices = np.arange(sample_count) % len(models)
    rng.shuffle(model_indices)
    rows: list[dict[str, Any]] = []
    for local_index, model_index in enumerate(model_indices):
        model = models[int(model_index)]
        candidate_x = np.clip(model["center_x"] + noise[local_index], 0.0, 1.0)
        design: dict[str, float | int] = {}
        for column, variable in enumerate(variables):
            name = str(variable["name"])
            lower, upper = (float(value) for value in variable["bounds"])
            value = lower + float(candidate_x[column]) * (upper - lower)
            if variable["dtype"] == "int64":
                value = int(round(value))
                draw = rng.random()
                if draw < discrete_step_probability / 2.0:
                    value -= 1
                elif draw < discrete_step_probability:
                    value += 1
                design[name] = int(min(max(value, int(lower)), int(upper)))
            else:
                design[name] = min(max(value, lower), upper)
        design_hash = sha256(_canonical_bytes(design)).hexdigest()
        rows.append(
            {
                "sample_index": local_index,
                "sample_id": (
                    f"directed:{segment_id}:r{round_index}:"
                    f"{local_index:04d}:{design_hash[:12]}"
                ),
                "design_hash": design_hash,
                "assigned_segment": segment_id,
                "adaptive_round": round_index,
                "parent_seed_id": model["anchor_id"],
                "root_seed_id": model["root_seed_id"],
                "geometry19": design,
            }
        )
    return rows


def run_directed_recovery(
    repository_root: str | Path,
    protocol_path: str | Path,
    output_root: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Run directed recovery and combine it with the frozen balanced dataset."""

    root = Path(repository_root).resolve()
    protocol_source, protocol = _load_protocol(root, protocol_path)
    output = Path(output_root).resolve()
    if output.exists() and not resume:
        raise Geometry19V22DatasetError(f"Refusing to overwrite recovery: {output}")
    output.mkdir(parents=True, exist_ok=True)
    context = build_pipeline_context(root, pipeline_profile="v2_2_development")
    domain = context.target_domain
    assert domain is not None
    base = protocol["base_expansion"]
    base_raw = (root / base["raw_dataset"]).resolve()
    base_accepted = (root / base["accepted_dataset"]).resolve()
    if _hash_file(base_raw) != base["raw_dataset_sha256"]:
        raise Geometry19V22DatasetError("Base raw-dataset hash mismatch")
    if _hash_file(base_accepted) != base["accepted_dataset_sha256"]:
        raise Geometry19V22DatasetError("Base accepted-dataset hash mismatch")
    base_rows = _read_jsonl(base_raw)
    base_accepted_rows = _read_jsonl(base_accepted)
    variables = list(context.physics_runtime.design_schema["design_variables"])
    sampling = protocol["sampling"]
    configured_segments = tuple(sampling["segments"])
    known_segments = {segment[0] for segment in domain.segments}
    if not set(configured_segments) < known_segments:
        raise Geometry19V22DatasetError("Recovery segments must be a strict domain subset")
    protocol_hash = _hash_file(protocol_source)
    recovery_rows: list[dict[str, Any]] = []
    model_audit: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    for round_index, round_spec in enumerate(sampling["rounds"]):
        prior_rows = tuple(recovery_rows)
        observations = [*base_rows, *prior_rows]
        for segment_index, segment_id in enumerate(configured_segments):
            models = _select_models(
                domain, variables, observations, segment_id, sampling, round_spec
            )
            model_audit.append(
                {
                    "adaptive_round": round_index,
                    "assigned_segment": segment_id,
                    "models": [
                        {key: value for key, value in model.items() if key != "center_x"}
                        for model in models
                    ],
                }
            )
            plan = _directed_plan(
                variables,
                models,
                segment_id=segment_id,
                round_index=round_index,
                sample_count=int(round_spec["samples_per_segment"]),
                perturbation_radius=float(round_spec["perturbation_radius_fraction"]),
                discrete_step_probability=float(
                    sampling["discrete_step_probability"]
                ),
                random_seed=(
                    int(sampling["random_seed"])
                    + round_index * 1013
                    + segment_index * 89
                ),
            )
            shard_path = output / "shards" / f"round-{round_index:02d}-{segment_id}.jsonl"
            metadata_path = shard_path.with_suffix(".meta.json")
            identity = {
                "dataset_version": DIRECTED_RECOVERY_VERSION,
                "protocol_hash": protocol_hash,
                "pipeline_context_hash": context.context_hash,
                "input_hash": sha256(_canonical_bytes(plan)).hexdigest(),
            }
            if resume and shard_path.exists() and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if any(metadata.get(key) != value for key, value in identity.items()):
                    raise Geometry19V22DatasetError(
                        f"Directed shard identity mismatch: {round_index}/{segment_id}"
                    )
                if metadata.get("output_hash") != _hash_file(shard_path):
                    raise Geometry19V22DatasetError(
                        f"Directed shard output hash mismatch: {round_index}/{segment_id}"
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
            recovery_rows.extend(rows)
            shard_records.append(metadata)
    raw_path = output / "recovery_raw_dataset.jsonl"
    _write_jsonl(raw_path, recovery_rows)
    combined = _deduplicate_training_rows([*base_accepted_rows, *recovery_rows])
    combined_path = output / "combined_accepted_dataset.jsonl"
    _write_jsonl(combined_path, combined)
    splits, assignments = _grouped_split(combined, protocol["split"])
    split_paths: dict[str, Path] = {}
    for split_name, rows in splits.items():
        path = output / "splits" / f"{split_name}.jsonl"
        _write_jsonl(path, rows)
        split_paths[split_name] = path
    all_segments = tuple(segment[0] for segment in domain.segments)
    segment_counts = {
        segment_id: sum(row["assigned_segment"] == segment_id for row in combined)
        for segment_id in all_segments
    }
    root_counts = {
        segment_id: len(
            {
                row["root_seed_id"]
                for row in combined
                if row["assigned_segment"] == segment_id
            }
        )
        for segment_id in all_segments
    }
    gate = protocol["coverage_gate"]
    coverage_pass = all(
        segment_counts[segment_id] >= int(gate["minimum_rows_per_segment"])
        and root_counts[segment_id] >= int(gate["minimum_root_seeds_per_segment"])
        for segment_id in all_segments
    )
    manifest = {
        "dataset_version": DIRECTED_RECOVERY_VERSION,
        "status": "COVERAGE_GATE_PASS" if coverage_pass else "COVERAGE_GATE_FAIL",
        "formal_use_allowed": False,
        "training_ready": coverage_pass,
        "coverage_gate_pass": coverage_pass,
        "counts": {
            "recovery_evaluated": len(recovery_rows),
            "recovery_physics_domain_valid": sum(
                row["constraint_assessment"]["physics_domain_valid"]
                for row in recovery_rows
            ),
            "recovery_training_candidates": sum(
                row["training_candidate"] for row in recovery_rows
            ),
            "combined_accepted_unique": len(combined),
            "by_segment": segment_counts,
            "root_seeds_by_segment": root_counts,
            "splits": {name: len(rows) for name, rows in splits.items()},
        },
        "coverage_gate": gate,
        "sampling": sampling,
        "split": protocol["split"],
        "group_assignments": assignments,
        "model_audit": model_audit,
        "shards": shard_records,
        "hashes": {
            "protocol_hash": protocol_hash,
            "base_raw_dataset_hash": _hash_file(base_raw),
            "base_accepted_dataset_hash": _hash_file(base_accepted),
            "recovery_raw_dataset_hash": _hash_file(raw_path),
            "combined_accepted_dataset_hash": _hash_file(combined_path),
            "pipeline_context_hash": context.context_hash,
            "physics_bundle_hash": context.physics_runtime.physics_bundle_hash,
            "target_domain_hash": domain.source_hash,
            **{
                f"{name}_split_hash": _hash_file(path)
                for name, path in split_paths.items()
            },
        },
        "paths": {
            "recovery_raw_dataset": _artifact_path(raw_path, root),
            "combined_accepted_dataset": _artifact_path(combined_path, root),
            "splits": {
                name: _artifact_path(path, root) for name, path in split_paths.items()
            },
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest
