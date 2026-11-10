"""Frozen paired 22D local coordinates, followed by fresh V3 labels."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any

import yaml

from src.ai.v3_local_center_audit import verify_v3_local_center_audit
from src.physics.constraint_profiles import ConstraintProfile, assess_constraint_profile
from src.physics.geometry22_data import Geometry22Normalizer
from src.physics.geometry22_design import Geometry22Design
from src.physics.v3_development_runtime import build_v3_development_physics_runtime


CONFIG = "config/experiments/geometry22_v3_local_pilot_v1.yaml"
CODE = "src/ai/v3_local_pilot.py"
PERFORMANCE = ("thrust_N", "tsfc_kg_N_s", "mass_kg")


class V3LocalPilotError(ValueError):
    """Pilot coordinates, source, or physics labels are incompatible."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _config(root: Path) -> dict[str, Any]:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if (
        config.get("version") != "geometry22_v3_local_pilot_v1"
        or config.get("split_before_labels") is not True
        or config.get("adaptive_top_up") is not False
        or config.get("training_allowed") is not False
        or config.get("formal_use_allowed") is not False
        or config.get("deployment_allowed") is not False
        or config.get("integer_policy") != "fixed_at_source_center"
        or [item["pair_index"] for item in config.get("pair_plan", [])] != [0, 1, 2, 3]
    ):
        raise V3LocalPilotError("V3 local pilot config changed scope")
    return config


def _centers(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    source = root / config["source_centers"]["path"]
    if (
        _sha(source / "complete_manifest.json") != config["source_centers"]["manifest_sha256"]
        or _sha(source / "centers.jsonl") != config["source_centers"]["centers_sha256"]
    ):
        raise V3LocalPilotError("L0 local center source changed")
    verify_v3_local_center_audit(root, source)
    centers = _rows(source / "centers.jsonl")
    if len(centers) != config["source_centers"]["expected_centers"] or not all(
        row["candidate_for_local_pilot"] for row in centers
    ):
        raise V3LocalPilotError("L0 local center candidate set changed")
    return centers


def _paired_designs(
    center: Geometry22Design, source_id: str, pair_index: int, radius: float,
    normalizer: Geometry22Normalizer, continuous: tuple[int, ...], integer: tuple[int, ...],
    seed: int, maximum_attempts: int,
) -> tuple[Geometry22Design, Geometry22Design, int]:
    center_norm = normalizer.transform(center)
    seed_text = f"{seed}:{source_id}:{pair_index}"
    generator = random.Random(int(hashlib.sha256(seed_text.encode()).hexdigest(), 16))
    for attempt in range(1, maximum_attempts + 1):
        draws = [generator.gauss(0.0, 1.0) for _ in continuous]
        magnitude = math.sqrt(sum(value * value for value in draws))
        if magnitude == 0.0:
            continue
        delta = {axis: radius * value / magnitude for axis, value in zip(continuous, draws, strict=True)}
        if any(
            center_norm[axis] - abs(value) < -1.0 or center_norm[axis] + abs(value) > 1.0
            for axis, value in delta.items()
        ):
            continue
        designs = []
        for sign in (-1, 1):
            coordinates = list(center_norm)
            for axis, value in delta.items():
                coordinates[axis] += sign * value
            design = normalizer.inverse(coordinates)
            if any(design.coordinates[axis] != center.coordinates[axis] for axis in integer):
                raise V3LocalPilotError("Local pilot changed an integer coordinate")
            actual = math.dist(
                [normalizer.transform(design)[axis] for axis in continuous],
                [center_norm[axis] for axis in continuous],
            )
            if not math.isclose(actual, radius, rel_tol=1e-9, abs_tol=1e-10):
                raise V3LocalPilotError("Local pilot radius changed during inversion")
            designs.append(design)
        return designs[0], designs[1], attempt
    raise V3LocalPilotError(f"Cannot place paired local coordinates for {source_id}:{pair_index}")


def _coordinate_rows(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    centers = _centers(root, config)
    normalizer = Geometry22Normalizer()
    integer = tuple(i for i, field in enumerate(normalizer.fields) if field["dtype"] == "int64")
    continuous = tuple(i for i in range(22) if i not in integer)
    if len(integer) != 3 or len(continuous) != 19:
        raise V3LocalPilotError("Unexpected V3 integer/continuous coordinate partition")
    runtime = build_v3_development_physics_runtime(root)
    rows = []
    for center_index, source in enumerate(centers):
        center = Geometry22Design.from_mapping(source["design"])
        for pair in config["pair_plan"]:
            pair_index = pair["pair_index"]
            radius = float(pair["radius"])
            low, high, attempt = _paired_designs(
                center, source["design_id"], pair_index, radius, normalizer,
                continuous, integer, config["direction_seed"],
                config["maximum_direction_attempts_per_pair"],
            )
            for sign, design in ((-1, low), (1, high)):
                rows.append({
                    "coordinate_index": len(rows),
                    "center_index": center_index,
                    "center_design_id": source["design_id"],
                    "source_v2_2_root_seed_id": source["source_v2_2_root_seed_id"],
                    "pair_index": pair_index,
                    "pair_sign": sign,
                    "direction_attempt": attempt,
                    "requested_normalized_l2_radius": radius,
                    "split": pair["split"],
                    "design_id": design.design_id(),
                    "design": design.as_mapping(),
                    "normalized_design": list(normalizer.transform(design)),
                    "design_schema_hash": runtime.design_schema_hash,
                    "training_allowed": False,
                })
    if len(rows) != len(centers) * len(config["pair_plan"]) * 2:
        raise V3LocalPilotError("Local pilot coordinate count changed")
    if len({row["design_id"] for row in rows}) != len(rows):
        raise V3LocalPilotError("Local pilot contains duplicate 22D designs")
    identity = {
        "config_sha256": _sha(root / CONFIG),
        "code_sha256": _sha(root / CODE),
        "l0_manifest_sha256": config["source_centers"]["manifest_sha256"],
        "l0_centers_sha256": config["source_centers"]["centers_sha256"],
        "runtime_contract": runtime.contract_mapping(),
        "integer_coordinate_indices_1_based": [axis + 1 for axis in integer],
        "continuous_dimension": len(continuous),
    }
    return rows, identity


def prepare_v3_local_pilot(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / config["output_root"]
    if output.exists():
        raise V3LocalPilotError(f"Refusing to overwrite local pilot: {output}")
    rows, identity = _coordinate_rows(root, config)
    content = b"".join(_json_bytes(row) for row in rows)
    manifest = {
        "status": "coordinates_frozen_before_v3_labels",
        "identity": identity,
        "coordinate_count": len(rows),
        "center_count": config["source_centers"]["expected_centers"],
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "radius_counts": {str(radius): sum(row["requested_normalized_l2_radius"] == radius for row in rows)
                          for radius in sorted({row["requested_normalized_l2_radius"] for row in rows})},
        "coordinate_sha256": hashlib.sha256(content).hexdigest(),
        "training_allowed": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".v3_local_pilot_", dir=output.parent))
    try:
        (temporary / "coordinates.jsonl").write_bytes(content)
        (temporary / "coordinate_manifest.json").write_bytes(_json_bytes(manifest))
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return verify_v3_local_pilot_coordinates(root, output)


def verify_v3_local_pilot_coordinates(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    config = _config(root)
    try:
        manifest = json.loads((output / "coordinate_manifest.json").read_text(encoding="utf-8"))
        rows = _rows(output / "coordinates.jsonl")
        expected, identity = _coordinate_rows(root, config)
        if (
            manifest["status"] != "coordinates_frozen_before_v3_labels"
            or manifest["identity"] != identity
            or manifest["coordinate_count"] != len(rows)
            or manifest["coordinate_count"] != len(expected)
            or manifest["center_count"] != config["source_centers"]["expected_centers"]
            or manifest["coordinate_sha256"] != _sha(output / "coordinates.jsonl")
            or manifest["split_counts"] != dict(Counter(row["split"] for row in rows))
            or manifest["radius_counts"] != {
                str(radius): sum(row["requested_normalized_l2_radius"] == radius for row in rows)
                for radius in sorted({row["requested_normalized_l2_radius"] for row in rows})
            }
            or manifest["training_allowed"] is not False
            or manifest["formal_use_allowed"] is not False
            or manifest["deployment_allowed"] is not False
            or rows != expected
        ):
            raise V3LocalPilotError("Local pilot coordinate lock changed")
    except V3LocalPilotError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise V3LocalPilotError("Cannot verify local pilot coordinates") from exc
    return manifest


def _label_rows(root: Path, coordinates: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    profile_path = root / config["diagnostic_partition"]["path"]
    if _sha(profile_path) != config["diagnostic_partition"]["sha256"]:
        raise V3LocalPilotError("Local pilot diagnostic partition changed")
    profile = ConstraintProfile.from_yaml(profile_path)
    runtime = build_v3_development_physics_runtime(root)
    labels = []
    for coordinate in coordinates:
        result = runtime.evaluate(Geometry22Design.from_mapping(coordinate["design"]))
        engine = result.engine_result
        diagnostic = assess_constraint_profile(
            engine, profile,
            conservation_tolerances=runtime.base_runtime.conservation_tolerances,
        ).as_mapping()
        performance = (
            {name: float(engine.performance[name]) for name in PERFORMANCE}
            if engine.solver.converged else None
        )
        if performance is not None and not all(math.isfinite(value) for value in performance.values()):
            raise V3LocalPilotError("Local pilot performance is non-finite")
        labels.append({
            "coordinate_index": coordinate["coordinate_index"],
            "design_id": result.design_id,
            "split": coordinate["split"],
            "center_design_id": coordinate["center_design_id"],
            "pair_index": coordinate["pair_index"],
            "pair_sign": coordinate["pair_sign"],
            "design_schema_hash": result.design_schema_hash,
            "physics_bundle_hash": result.physics_bundle_hash,
            "physics_model_version": result.physics_model_version,
            "solver_converged": engine.solver.converged,
            "solver_failure_code": engine.solver.failure_code.value if engine.solver.failure_code else None,
            "flow_root_count": engine.operating_point.get("flow_root_count"),
            "performance": performance,
            "legacy_partition_development_diagnostic": diagnostic,
            "development_domain_retained": bool(diagnostic["numerically_valid"] and diagnostic["physics_domain_valid"]),
            "engine_feasible": bool(engine.feasible),
            "training_allowed": False,
            "formal_use_allowed": False,
            "deployment_allowed": False,
        })
    return labels


def evaluate_v3_local_pilot(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / config["output_root"]
    prelabel = verify_v3_local_pilot_coordinates(root, output)
    if (output / "labels.jsonl").exists() or (output / "complete_manifest.json").exists():
        raise V3LocalPilotError("Local pilot labels already exist; refusing to overwrite")
    coordinates = _rows(output / "coordinates.jsonl")
    labels = _label_rows(root, coordinates, config)
    content = b"".join(_json_bytes(row) for row in labels)
    manifest = {
        "status": "v3_local_pilot_labeled_development_only",
        "coordinate_manifest_sha256": _sha(output / "coordinate_manifest.json"),
        "coordinate_sha256": prelabel["coordinate_sha256"],
        "labels_sha256": hashlib.sha256(content).hexdigest(),
        "diagnostic_partition_sha256": config["diagnostic_partition"]["sha256"],
        "row_count": len(labels),
        "split_counts": dict(Counter(row["split"] for row in labels)),
        "solver_converged_count": sum(row["solver_converged"] for row in labels),
        "development_domain_retained_count": sum(row["development_domain_retained"] for row in labels),
        "engine_feasible_count": sum(row["engine_feasible"] for row in labels),
        "source_v2_2_labels_reused": False,
        "training_allowed": False,
        "physics_validation_status": "not_completed",
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }
    temporary = output / ".labels.jsonl.tmp"
    try:
        temporary.write_bytes(content)
        os.link(temporary, output / "labels.jsonl")
    finally:
        temporary.unlink(missing_ok=True)
    (output / "complete_manifest.json").write_bytes(_json_bytes(manifest))
    return verify_v3_local_pilot_labels(root, output)


def verify_v3_local_pilot_labels(
    repository_root: str | Path, output_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / config["output_root"]
    prelabel = verify_v3_local_pilot_coordinates(root, output)
    try:
        manifest = json.loads((output / "complete_manifest.json").read_text(encoding="utf-8"))
        labels_path = output / "labels.jsonl"
        if _sha(labels_path) != manifest["labels_sha256"]:
            raise V3LocalPilotError("Local pilot labels changed")
        labels = _rows(labels_path)
        coordinates = _rows(output / "coordinates.jsonl")
        physics_hash = prelabel["identity"]["runtime_contract"]["physics_bundle_hash"]
        schema_hash = prelabel["identity"]["runtime_contract"]["design_schema_hash"]
        if (
            manifest["status"] != "v3_local_pilot_labeled_development_only"
            or manifest["coordinate_manifest_sha256"] != _sha(output / "coordinate_manifest.json")
            or manifest["coordinate_sha256"] != prelabel["coordinate_sha256"]
            or manifest["diagnostic_partition_sha256"] != config["diagnostic_partition"]["sha256"]
            or manifest["row_count"] != len(labels)
            or manifest["row_count"] != len(coordinates)
            or manifest["split_counts"] != dict(Counter(row["split"] for row in labels))
            or manifest["solver_converged_count"] != sum(row["solver_converged"] for row in labels)
            or manifest["development_domain_retained_count"] != sum(row["development_domain_retained"] for row in labels)
            or manifest["engine_feasible_count"] != sum(row["engine_feasible"] for row in labels)
            or manifest["source_v2_2_labels_reused"] is not False
            or manifest["training_allowed"] is not False
            or manifest["formal_use_allowed"] is not False
            or manifest["deployment_allowed"] is not False
        ):
            raise V3LocalPilotError("Local pilot label manifest changed")
        for coordinate, label in zip(coordinates, labels, strict=True):
            if (
                label["coordinate_index"] != coordinate["coordinate_index"]
                or label["design_id"] != coordinate["design_id"]
                or label["split"] != coordinate["split"]
                or label["center_design_id"] != coordinate["center_design_id"]
                or label["pair_index"] != coordinate["pair_index"]
                or label["pair_sign"] != coordinate["pair_sign"]
                or label["design_schema_hash"] != schema_hash
                or label["physics_bundle_hash"] != physics_hash
                or label["legacy_partition_development_diagnostic"]["constraint_profile_hash"]
                != config["diagnostic_partition"]["sha256"]
                or label["development_domain_retained"] != bool(
                    label["legacy_partition_development_diagnostic"]["numerically_valid"]
                    and label["legacy_partition_development_diagnostic"]["physics_domain_valid"]
                )
                or (label["performance"] is None) != (not label["solver_converged"])
                or (label["solver_failure_code"] is None) != label["solver_converged"]
                or label["training_allowed"] is not False
                or label["formal_use_allowed"] is not False
                or label["deployment_allowed"] is not False
            ):
                raise V3LocalPilotError("Local pilot label identity changed")
            if label["performance"] is not None and not all(
                math.isfinite(float(label["performance"][name])) for name in PERFORMANCE
            ):
                raise V3LocalPilotError("Local pilot label is non-finite")
    except V3LocalPilotError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise V3LocalPilotError("Cannot verify local pilot labels") from exc
    return manifest
