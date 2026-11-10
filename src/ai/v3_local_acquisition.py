"""Acquire fresh V3 local labels under the frozen L2 center split."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import yaml

from src.ai.v3_local_data_contract import verify_v3_local_data_contract
from src.ai.v3_local_pilot import _paired_designs
from src.physics.constraint_profiles import ConstraintProfile, assess_constraint_profile
from src.physics.geometry22_data import Geometry22Normalizer
from src.physics.geometry22_design import Geometry22Design
from src.physics.v3_development_runtime import build_v3_development_physics_runtime


CONFIG = "config/experiments/geometry22_v3_local_acquisition_v1.yaml"
CODE = "src/ai/v3_local_acquisition.py"
PAIR_CODE = "src/ai/v3_local_pilot.py"
PERFORMANCE = ("thrust_N", "tsfc_kg_N_s", "mass_kg")
SPLITS = ("development", "validation", "holdout")


class V3LocalAcquisitionError(ValueError):
    """The source, coordinate lock, or label identity is incompatible."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _config(root: Path) -> dict[str, Any]:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if (
        config.get("version") != "geometry22_v3_local_acquisition_v1"
        or config.get("scope") != "center_disjoint_local_development_labels_only"
        or config.get("integer_policy") != "fixed_at_source_center"
        or config.get("split_before_labels") is not True
        or config.get("adaptive_top_up") is not False
        or config.get("training_allowed") is not False
        or config.get("formal_use_allowed") is not False
        or config.get("deployment_allowed") is not False
        or [pair.get("pair_index") for pair in config.get("pair_plan", [])] != [0, 1, 2, 3]
        or [float(pair["radius"]) for pair in config["pair_plan"]] != [0.02, 0.02, 0.05, 0.05]
        or config.get("diagnostic_partition", {}).get("status")
        != "borrowed_v2_2_partition_for_v3_software_diagnostic_only"
    ):
        raise V3LocalAcquisitionError("V3 local acquisition config changed scope")
    return config


def _sources(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    contract = root / config["source_contract"]["path"]
    centers = root / config["source_centers"]["path"] / "centers.jsonl"
    pilot = root / config["source_pilot"]["path"] / "coordinates.jsonl"
    if (
        _sha(contract / "manifest.json") != config["source_contract"]["manifest_sha256"]
        or _sha(contract / "center_split_registry.jsonl") != config["source_contract"]["registry_sha256"]
        or _sha(centers) != config["source_centers"]["centers_sha256"]
        or _sha(pilot) != config["source_pilot"]["coordinate_sha256"]
    ):
        raise V3LocalAcquisitionError("L2/L0/L1 frozen source hash changed")
    verify_v3_local_data_contract(root, contract)
    registry = _rows(contract / "center_split_registry.jsonl")
    source_rows = _rows(centers)
    if len(registry) != len(source_rows) or len(registry) != 33:
        raise V3LocalAcquisitionError("L2 center count changed")
    by_id = {row["design_id"]: row for row in source_rows}
    if len(by_id) != 33 or set(by_id) != {row["center_design_id"] for row in registry}:
        raise V3LocalAcquisitionError("L2 center identity changed")
    return registry, by_id


def _coordinate_rows(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry, centers = _sources(root, config)
    runtime = build_v3_development_physics_runtime(root)
    normalizer = Geometry22Normalizer()
    integer = tuple(i for i, field in enumerate(normalizer.fields) if field["dtype"] == "int64")
    continuous = tuple(i for i in range(22) if i not in integer)
    if len(integer) != 3 or len(continuous) != 19:
        raise V3LocalAcquisitionError("Unexpected V3 coordinate partition")
    rows = []
    for item in registry:
        center = centers[item["center_design_id"]]
        if (
            item["design_schema_hash"] != runtime.design_schema_hash
            or item["physics_bundle_hash"] != runtime.physics_bundle_hash
        ):
            raise V3LocalAcquisitionError("L2 runtime identity changed")
        design = Geometry22Design.from_mapping(center["design"])
        for pair in config["pair_plan"]:
            low, high, attempt = _paired_designs(
                design, item["center_design_id"], pair["pair_index"], float(pair["radius"]),
                normalizer, continuous, integer, config["direction_seed"],
                config["maximum_direction_attempts_per_pair"],
            )
            for sign, sample in ((-1, low), (1, high)):
                rows.append({
                    "coordinate_index": len(rows),
                    "center_design_id": item["center_design_id"],
                    "center_split": item["center_split"],
                    "pair_index": pair["pair_index"],
                    "pair_sign": sign,
                    "direction_attempt": attempt,
                    "requested_normalized_l2_radius": float(pair["radius"]),
                    "design_id": sample.design_id(),
                    "design": sample.as_mapping(),
                    "normalized_design": list(normalizer.transform(sample)),
                    "design_schema_hash": runtime.design_schema_hash,
                    "physics_bundle_hash": runtime.physics_bundle_hash,
                    "training_allowed": False,
                })
    design_ids = [row["design_id"] for row in rows]
    old_ids = {row["design_id"] for row in _rows(root / config["source_pilot"]["path"] / "coordinates.jsonl")}
    if len(rows) != 264 or len(set(design_ids)) != 264 or set(design_ids) & old_ids:
        raise V3LocalAcquisitionError("New local coordinates overlap or count changed")
    identity = {
        "config_sha256": _sha(root / CONFIG),
        "code_sha256": _sha(root / CODE),
        "pair_generator_code_sha256": _sha(root / PAIR_CODE),
        "l2_manifest_sha256": config["source_contract"]["manifest_sha256"],
        "l2_registry_sha256": config["source_contract"]["registry_sha256"],
        "runtime_contract": runtime.contract_mapping(),
        "integer_coordinate_indices_1_based": [i + 1 for i in integer],
    }
    return rows, identity


def _coordinate_manifest(rows: list[dict[str, Any]], identity: dict[str, Any], content: bytes) -> dict[str, Any]:
    return {
        "status": "l3_center_disjoint_coordinates_frozen_before_labels",
        "identity": identity,
        "coordinate_count": len(rows),
        "center_count": len({row["center_design_id"] for row in rows}),
        "split_counts": dict(Counter(row["center_split"] for row in rows)),
        "radius_counts": {str(radius): sum(row["requested_normalized_l2_radius"] == radius for row in rows)
                          for radius in (0.02, 0.05)},
        "coordinate_sha256": hashlib.sha256(content).hexdigest(),
        "training_allowed": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }


def prepare_v3_local_acquisition(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / config["output_root"]
    if output.exists():
        raise V3LocalAcquisitionError(f"Refusing to overwrite local acquisition: {output}")
    rows, identity = _coordinate_rows(root, config)
    content = b"".join(_bytes(row) for row in rows)
    manifest = _coordinate_manifest(rows, identity, content)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".v3_local_acquisition_", dir=output.parent))
    try:
        (temporary / "coordinates.jsonl").write_bytes(content)
        (temporary / "coordinate_manifest.json").write_bytes(_bytes(manifest))
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return verify_v3_local_acquisition_coordinates(root, output)


def verify_v3_local_acquisition_coordinates(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    try:
        rows, identity = _coordinate_rows(root, _config(root))
        content = b"".join(_bytes(row) for row in rows)
        manifest = json.loads((output / "coordinate_manifest.json").read_text(encoding="utf-8"))
        if (
            _rows(output / "coordinates.jsonl") != rows
            or _sha(output / "coordinates.jsonl") != hashlib.sha256(content).hexdigest()
            or manifest != _coordinate_manifest(rows, identity, content)
        ):
            raise V3LocalAcquisitionError("L3 coordinate lock changed")
    except V3LocalAcquisitionError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise V3LocalAcquisitionError("Cannot verify L3 coordinates") from exc
    return manifest


def _label_rows(root: Path, coordinates: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    profile_path = root / config["diagnostic_partition"]["path"]
    if _sha(profile_path) != config["diagnostic_partition"]["sha256"]:
        raise V3LocalAcquisitionError("Borrowed diagnostic partition changed")
    profile = ConstraintProfile.from_yaml(profile_path)
    runtime = build_v3_development_physics_runtime(root)
    labels = []
    for coordinate in coordinates:
        result = runtime.evaluate(Geometry22Design.from_mapping(coordinate["design"]))
        engine = result.engine_result
        diagnostic = assess_constraint_profile(
            engine, profile, conservation_tolerances=runtime.base_runtime.conservation_tolerances,
        ).as_mapping()
        performance = (
            {name: float(engine.performance[name]) for name in PERFORMANCE}
            if engine.solver.converged else None
        )
        finite = performance is not None and all(math.isfinite(v) for v in performance.values())
        if not finite:
            performance = None
        candidate = bool(finite and diagnostic["numerically_valid"] and diagnostic["physics_domain_valid"])
        labels.append({
            "coordinate_index": coordinate["coordinate_index"],
            "design_id": result.design_id,
            "center_design_id": coordinate["center_design_id"],
            "center_split": coordinate["center_split"],
            "pair_index": coordinate["pair_index"],
            "pair_sign": coordinate["pair_sign"],
            "requested_normalized_l2_radius": coordinate["requested_normalized_l2_radius"],
            "design_schema_hash": result.design_schema_hash,
            "physics_bundle_hash": result.physics_bundle_hash,
            "physics_model_version": result.physics_model_version,
            "solver_converged": engine.solver.converged,
            "solver_failure_code": engine.solver.failure_code.value if engine.solver.failure_code else None,
            "flow_root_count": engine.operating_point.get("flow_root_count"),
            "performance": performance,
            "performance_finite": finite,
            "legacy_partition_development_diagnostic": diagnostic,
            "software_regression_candidate": candidate,
            "engine_feasible": bool(engine.feasible),
            "training_allowed": False,
            "formal_use_allowed": False,
            "deployment_allowed": False,
        })
    return labels


def _coverage(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for label in labels:
        key = (label["center_design_id"], label["center_split"], label["requested_normalized_l2_radius"])
        groups.setdefault(key, []).append(label)
    return [
        {
            "center_design_id": center,
            "center_split": split,
            "radius": radius,
            "sample_count": len(group),
            "solver_converged_count": sum(row["solver_converged"] for row in group),
            "borrowed_domain_count": sum(bool(row["legacy_partition_development_diagnostic"]["physics_domain_valid"])
                                         for row in group),
            "regression_candidate_count": sum(row["software_regression_candidate"] for row in group),
            "engine_feasible_count": sum(row["engine_feasible"] for row in group),
        }
        for (center, split, radius), group in sorted(groups.items())
    ]


def _complete_manifest(labels: list[dict[str, Any]], coordinate_manifest_sha: str,
                       labels_sha: str, coverage_sha: str, diagnostic_sha: str) -> dict[str, Any]:
    return {
        "status": "l3_v3_center_disjoint_local_labels_development_only",
        "coordinate_manifest_sha256": coordinate_manifest_sha,
        "labels_sha256": labels_sha,
        "coverage_sha256": coverage_sha,
        "diagnostic_partition_sha256": diagnostic_sha,
        "row_count": len(labels),
        "split_counts": dict(Counter(row["center_split"] for row in labels)),
        "solver_converged_count": sum(row["solver_converged"] for row in labels),
        "software_regression_candidate_count": sum(row["software_regression_candidate"] for row in labels),
        "engine_feasible_count": sum(row["engine_feasible"] for row in labels),
        "source_v2_2_labels_reused": False,
        "training_allowed": False,
        "physics_validation_status": "not_completed",
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }


def evaluate_v3_local_acquisition(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / config["output_root"]
    verify_v3_local_acquisition_coordinates(root, output)
    if any((output / name).exists() for name in ("labels.jsonl", "coverage.jsonl", "complete_manifest.json")):
        raise V3LocalAcquisitionError("L3 labels already exist; refusing to overwrite")
    coordinates = _rows(output / "coordinates.jsonl")
    labels = _label_rows(root, coordinates, config)
    label_content = b"".join(_bytes(row) for row in labels)
    coverage_content = b"".join(_bytes(row) for row in _coverage(labels))
    manifest = _complete_manifest(
        labels, _sha(output / "coordinate_manifest.json"),
        hashlib.sha256(label_content).hexdigest(), hashlib.sha256(coverage_content).hexdigest(),
        config["diagnostic_partition"]["sha256"],
    )
    for name, content in (("labels.jsonl", label_content), ("coverage.jsonl", coverage_content),
                          ("complete_manifest.json", _bytes(manifest))):
        temporary = output / f".{name}.tmp"
        try:
            temporary.write_bytes(content)
            os.link(temporary, output / name)
        finally:
            temporary.unlink(missing_ok=True)
    return verify_v3_local_acquisition_labels(root, output)


def verify_v3_local_acquisition_labels(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / config["output_root"]
    prelabel = verify_v3_local_acquisition_coordinates(root, output)
    try:
        labels = _rows(output / "labels.jsonl")
        coordinates = _rows(output / "coordinates.jsonl")
        coverage = _rows(output / "coverage.jsonl")
        manifest = json.loads((output / "complete_manifest.json").read_text(encoding="utf-8"))
        expected_manifest = _complete_manifest(
            labels, _sha(output / "coordinate_manifest.json"), _sha(output / "labels.jsonl"),
            _sha(output / "coverage.jsonl"), config["diagnostic_partition"]["sha256"],
        )
        if (
            len(labels) != len(coordinates)
            or manifest != expected_manifest
            or coverage != _coverage(labels)
            or prelabel["coordinate_count"] != len(labels)
        ):
            raise V3LocalAcquisitionError("L3 label or coverage manifest changed")
        schema_hash = prelabel["identity"]["runtime_contract"]["design_schema_hash"]
        physics_hash = prelabel["identity"]["runtime_contract"]["physics_bundle_hash"]
        for coordinate, label in zip(coordinates, labels, strict=True):
            diagnostic = label["legacy_partition_development_diagnostic"]
            performance = label["performance"]
            finite = performance is not None and all(math.isfinite(float(performance[name])) for name in PERFORMANCE)
            if (
                any(label[name] != coordinate[name] for name in (
                    "coordinate_index", "design_id", "center_design_id", "center_split", "pair_index",
                    "pair_sign", "requested_normalized_l2_radius"
                ))
                or label["design_schema_hash"] != schema_hash
                or label["physics_bundle_hash"] != physics_hash
                or diagnostic["constraint_profile_hash"] != config["diagnostic_partition"]["sha256"]
                or label["performance_finite"] != finite
                or (not label["solver_converged"] and performance is not None)
                or (label["solver_converged"] and label["performance_finite"] and performance is None)
                or (label["solver_failure_code"] is None) != label["solver_converged"]
                or label["software_regression_candidate"] != bool(
                    finite and diagnostic["numerically_valid"] and diagnostic["physics_domain_valid"]
                )
                or label["training_allowed"] is not False
                or label["formal_use_allowed"] is not False
                or label["deployment_allowed"] is not False
            ):
                raise V3LocalAcquisitionError("L3 label identity or qualification changed")
    except V3LocalAcquisitionError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise V3LocalAcquisitionError("Cannot verify L3 labels") from exc
    return manifest
