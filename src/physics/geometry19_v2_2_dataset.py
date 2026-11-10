"""Reproducible restricted-domain dataset pilot for simplified Geometry19 V2.2."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.physics import Environment, Geometry19
from src.physics.constraint_profiles import assess_constraint_profile


DATASET_VERSION = "geometry19_v2_2_restricted_pilot_v1"


class Geometry19V22DatasetError(ValueError):
    """Raised when the frozen pilot protocol or an artifact is inconsistent."""


def _hash_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(_canonical_bytes(row).decode("utf-8") + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _artifact_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def freeze_seed_pack(
    repository_root: str | Path,
    source_campaigns: Iterable[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze unique V2.2 physics-valid designs from audited C-stage ledgers."""

    root = Path(repository_root).resolve()
    output = Path(output_path)
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    if output.exists():
        raise Geometry19V22DatasetError(f"Refusing to overwrite seed pack: {output}")
    domain = build_pipeline_context(root, pipeline_profile="v2_2_development").target_domain
    assert domain is not None
    sources: list[dict[str, str]] = []
    unique: dict[str, dict[str, Any]] = {}
    for campaign_value in source_campaigns:
        campaign = Path(campaign_value)
        campaign = campaign.resolve() if campaign.is_absolute() else (root / campaign).resolve()
        ledgers = sorted(campaign.glob("*/wp19_23c/physics_forward_ledger.jsonl"))
        if not ledgers:
            raise Geometry19V22DatasetError(f"No C-stage ledgers found: {campaign}")
        for ledger in ledgers:
            sources.append(
                {"path": str(ledger.relative_to(root)), "sha256": _hash_file(ledger)}
            )
            for row in _read_jsonl(ledger):
                if not row.get("physics_acceptable") or not row.get("performance"):
                    continue
                design_hash = str(row["design_hash"])
                assessment = domain.assess(row["performance"]).as_mapping()
                unique.setdefault(
                    design_hash,
                    {
                        "seed_id": design_hash,
                        "geometry19": row["geometry19"],
                        "performance": row["performance"],
                        "target_domain_assessment": assessment,
                        "source_evaluation_id": row.get("evaluation_id"),
                    },
                )
    rows = sorted(
        unique.values(),
        key=lambda row: (
            row["target_domain_assessment"]["normalized_distance"], row["seed_id"]
        ),
    )
    _write_jsonl(output, rows)
    manifest = {
        "seed_pack_version": "geometry19_v2_2_audited_seed_pack_v1",
        "profile": "v2_2_development",
        "seed_count": len(rows),
        "inside_target_domain_count": sum(
            bool(row["target_domain_assessment"]["inside"]) for row in rows
        ),
        "seed_pack_path": str(output.relative_to(root)),
        "seed_pack_sha256": _hash_file(output),
        "source_ledgers": sources,
    }
    _write_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


def _load_protocol(root: Path, protocol_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(protocol_path)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("dataset_version") != DATASET_VERSION:
        raise Geometry19V22DatasetError("Dataset protocol version mismatch")
    if payload.get("profile") != "v2_2_development":
        raise Geometry19V22DatasetError("Pilot must use v2_2_development")
    if payload.get("formal_use_allowed") is not False:
        raise Geometry19V22DatasetError("Pilot cannot be marked for formal use")
    return path, payload


def _sample_plan(
    schema: Mapping[str, Any], seeds: list[dict[str, Any]], sampling: Mapping[str, Any]
) -> list[dict[str, Any]]:
    count = int(sampling["sample_count"])
    radius = float(sampling["local_radius_fraction"])
    maximum_seed_distance = float(sampling["maximum_seed_domain_distance"])
    eligible = [
        row
        for row in seeds
        if float(row["target_domain_assessment"]["normalized_distance"])
        <= maximum_seed_distance
    ]
    if not eligible:
        raise Geometry19V22DatasetError("No seed lies within the configured support distance")
    rng = np.random.default_rng(int(sampling["random_seed"]))
    variables = list(schema["design_variables"])
    dimensions = len(variables)
    lhs = np.empty((count, dimensions), dtype=float)
    for column in range(dimensions):
        lhs[:, column] = (rng.permutation(count) + rng.random(count)) / count
    offsets = (2.0 * lhs - 1.0) * radius
    seed_indices = rng.integers(0, len(eligible), size=count)
    anchor_count = min(len(eligible), count)
    seed_indices[:anchor_count] = np.arange(anchor_count)
    offsets[:anchor_count, :] = 0.0
    rows: list[dict[str, Any]] = []
    for sample_index in range(count):
        seed = eligible[int(seed_indices[sample_index])]
        design: dict[str, float | int] = {}
        for column, variable in enumerate(variables):
            name = str(variable["name"])
            lower, upper = (float(value) for value in variable["bounds"])
            value = float(seed["geometry19"][name]) + offsets[sample_index, column] * (
                upper - lower
            )
            value = min(max(value, lower), upper)
            design[name] = int(round(value)) if variable["dtype"] == "int64" else value
        design_hash = _hash_bytes(_canonical_bytes(design))
        rows.append(
            {
                "sample_index": sample_index,
                "sample_id": f"pilot:{sample_index:06d}:{design_hash[:12]}",
                "design_hash": design_hash,
                "parent_seed_id": seed["seed_id"],
                "geometry19": design,
            }
        )
    return rows


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return {str(key): _finite_or_none(item) for key, item in value.items()}
    return str(getattr(value, "value", value))


def _evaluate_row(row: dict[str, Any], context: Any) -> dict[str, Any]:
    runtime = context.physics_runtime
    result = runtime.evaluate(Geometry19.from_mapping(row["geometry19"]), Environment())
    assessment = assess_constraint_profile(
        result,
        runtime.constraint_profile,
        conservation_tolerances=runtime.conservation_tolerances,
    )
    performance = _finite_or_none(result.performance) if result.performance else None
    target_assessment = None
    if assessment.numerically_valid and performance is not None:
        target_assessment = context.target_domain.assess(performance).as_mapping()
    map_states = {
        name: {
            key: _finite_or_none(values.get(key))
            for key in (
                "map_id",
                "map_version",
                "map_in_domain",
                "map_extrapolation_distance",
                "map_uncertainty",
            )
            if key in values
        }
        for name, values in result.components.items()
    }
    failure = result.solver.failure_code
    training_candidate = bool(
        assessment.physics_domain_valid
        and target_assessment is not None
        and target_assessment["inside"]
    )
    return {
        **row,
        "solver": {
            "converged": result.solver.converged,
            "failure_code": None if failure is None else failure.value,
            "iterations": result.solver.iterations,
            "evaluations": result.solver.evaluations,
            "flow_residual": _finite_or_none(result.solver.flow_residual),
        },
        "performance": performance,
        "constraint_assessment": assessment.as_mapping(),
        "target_domain_assessment": target_assessment,
        "training_candidate": training_candidate,
        "map_states": map_states,
        "conservation": _finite_or_none(asdict(result.conservation)),
        "physics_model_version": result.physics_model_version,
        "physics_bundle_hash": runtime.physics_bundle_hash,
        "pipeline_context_hash": context.context_hash,
        "target_domain_hash": context.target_domain.source_hash,
    }


def run_restricted_pilot(
    repository_root: str | Path,
    protocol_path: str | Path,
    output_root: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Evaluate the deterministic pilot plan in hash-checked resumable shards."""

    root = Path(repository_root).resolve()
    protocol_source, protocol = _load_protocol(root, protocol_path)
    output = Path(output_root).resolve()
    if output.exists() and not resume:
        raise Geometry19V22DatasetError(f"Refusing to overwrite pilot: {output}")
    output.mkdir(parents=True, exist_ok=True)
    context = build_pipeline_context(root, pipeline_profile="v2_2_development")
    if context.target_domain is None:
        raise Geometry19V22DatasetError("V2.2 target domain is unavailable")
    seed_path = (root / protocol["seed_pack"]["path"]).resolve()
    if _hash_file(seed_path) != protocol["seed_pack"]["sha256"]:
        raise Geometry19V22DatasetError("Seed-pack hash mismatch")
    seeds = _read_jsonl(seed_path)
    plan = _sample_plan(context.physics_runtime.design_schema, seeds, protocol["sampling"])
    plan_path = output / "sample_plan.jsonl"
    if plan_path.exists() and _read_jsonl(plan_path) != plan:
        raise Geometry19V22DatasetError("Resume sample plan differs")
    if not plan_path.exists():
        _write_jsonl(plan_path, plan)
    protocol_hash = _hash_file(protocol_source)
    shard_size = int(protocol["sampling"]["shard_size"])
    all_rows: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    for start in range(0, len(plan), shard_size):
        shard_index = start // shard_size
        inputs = plan[start : start + shard_size]
        shard_path = output / "shards" / f"part-{shard_index:05d}.jsonl"
        metadata_path = shard_path.with_suffix(".meta.json")
        identity = {
            "dataset_version": DATASET_VERSION,
            "protocol_hash": protocol_hash,
            "pipeline_context_hash": context.context_hash,
            "input_hash": _hash_bytes(_canonical_bytes(inputs)),
        }
        if resume and shard_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if any(metadata.get(key) != value for key, value in identity.items()):
                raise Geometry19V22DatasetError(f"Shard identity mismatch: {shard_index}")
            if metadata.get("output_hash") != _hash_file(shard_path):
                raise Geometry19V22DatasetError(f"Shard output hash mismatch: {shard_index}")
            rows = _read_jsonl(shard_path)
        else:
            rows = [_evaluate_row(row, context) for row in inputs]
            _write_jsonl(shard_path, rows)
            metadata = {
                **identity,
                "shard_index": shard_index,
                "row_count": len(rows),
                "output_hash": _hash_file(shard_path),
            }
            _write_json(metadata_path, metadata)
        all_rows.extend(rows)
        shard_records.append(metadata)
    raw_path = output / "raw_dataset.jsonl"
    _write_jsonl(raw_path, all_rows)
    training_rows = [row for row in all_rows if row["training_candidate"]]
    training_path = output / "training_candidates.jsonl"
    _write_jsonl(training_path, training_rows)
    counts = {
        "evaluated": len(all_rows),
        "solver_converged": sum(row["solver"]["converged"] for row in all_rows),
        "numerically_valid": sum(
            row["constraint_assessment"]["numerically_valid"] for row in all_rows
        ),
        "physics_domain_valid": sum(
            row["constraint_assessment"]["physics_domain_valid"] for row in all_rows
        ),
        "engineering_feasible": sum(
            row["constraint_assessment"]["engineering_feasible"] for row in all_rows
        ),
        "inside_target_domain": sum(
            bool(row["target_domain_assessment"] and row["target_domain_assessment"]["inside"])
            for row in all_rows
        ),
        "training_candidates": len(training_rows),
    }
    gate = protocol["pilot_gate"]
    pilot_gate_pass = (
        counts["physics_domain_valid"] >= int(gate["minimum_physics_domain_valid"])
        and counts["training_candidates"] >= int(gate["minimum_training_candidates"])
    )
    manifest = {
        "dataset_version": DATASET_VERSION,
        "status": "PILOT_GATE_PASS" if pilot_gate_pass else "PILOT_GATE_FAIL",
        "formal_use_allowed": False,
        "training_ready": False,
        "pilot_gate_pass": pilot_gate_pass,
        "counts": counts,
        "sampling": protocol["sampling"],
        "pilot_gate": gate,
        "hashes": {
            "protocol_hash": protocol_hash,
            "seed_pack_hash": _hash_file(seed_path),
            "sample_plan_hash": _hash_file(plan_path),
            "raw_dataset_hash": _hash_file(raw_path),
            "training_candidates_hash": _hash_file(training_path),
            "pipeline_context_hash": context.context_hash,
            "physics_bundle_hash": context.physics_runtime.physics_bundle_hash,
            "target_domain_hash": context.target_domain.source_hash,
        },
        "shards": shard_records,
        "paths": {
            "raw_dataset": _artifact_path(raw_path, root),
            "training_candidates": _artifact_path(training_path, root),
            "sample_plan": _artifact_path(plan_path, root),
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest
