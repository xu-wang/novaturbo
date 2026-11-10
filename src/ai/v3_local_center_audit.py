"""Relabel frozen V2.2 medoid coordinates with the V3 development backend."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import yaml

from src.physics.constraint_profiles import ConstraintProfile, assess_constraint_profile
from src.physics.geometry22_data import Geometry22Normalizer
from src.physics.geometry22_design import Geometry22Design, SCHEMA_PATH
from src.physics.v3_development_runtime import build_v3_development_physics_runtime


CONFIG = "config/experiments/geometry22_v3_local_center_audit_v1.yaml"
CODE = "src/ai/v3_local_center_audit.py"
OUTPUT_NAMES = ("thrust_N", "tsfc_kg_N_s", "mass_kg")


class V3LocalCenterAuditError(ValueError):
    """An audit input, row, or recorded identity is incompatible."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _read_config(root: Path) -> dict[str, Any]:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if (
        config.get("version") != "geometry22_v3_local_center_audit_v1"
        or config.get("new_coordinates") != "explicit_v3_schema_reference_values"
        or config.get("training_allowed") is not False
        or config.get("formal_use_allowed") is not False
        or config.get("deployment_allowed") is not False
    ):
        raise V3LocalCenterAuditError("V3 local center audit config changed scope")
    return config


def _source_rows(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    source = root / config["source_anchor_registry"]["path"]
    if _sha(source) != config["source_anchor_registry"]["sha256"]:
        raise V3LocalCenterAuditError("V2.2 anchor registry hash changed")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if (
        len(rows) != config["source_anchor_registry"]["expected_rows"]
        or len({row["root_seed_id"] for row in rows}) != len(rows)
        or len({row["medoid_design_hash"] for row in rows}) != len(rows)
    ):
        raise V3LocalCenterAuditError("V2.2 medoid source count or identity changed")
    return rows


def _evaluate(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = _source_rows(root, config)
    profile_path = root / config["diagnostic_partition"]["path"]
    if _sha(profile_path) != config["diagnostic_partition"]["sha256"]:
        raise V3LocalCenterAuditError("Diagnostic partition hash changed")
    profile = ConstraintProfile.from_yaml(profile_path)
    runtime = build_v3_development_physics_runtime(root)
    normalizer = Geometry22Normalizer()
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    appended = {field["name"]: field["reference_value"] for field in schema["design_variables"][19:]}
    rows = []
    for index, source in enumerate(source_rows):
        design = Geometry22Design.from_mapping({**source["medoid_geometry19"], **appended})
        result = runtime.evaluate(design)
        engine = result.engine_result
        diagnostic = assess_constraint_profile(
            engine, profile,
            conservation_tolerances=runtime.base_runtime.conservation_tolerances,
        ).as_mapping()
        performance = (
            {name: float(engine.performance[name]) for name in OUTPUT_NAMES}
            if engine.solver.converged else None
        )
        if performance is not None and not all(math.isfinite(value) for value in performance.values()):
            raise V3LocalCenterAuditError("V3 local center produced non-finite performance")
        candidate = bool(diagnostic["numerically_valid"] and diagnostic["physics_domain_valid"])
        rows.append({
            "source_index": index,
            "source_v2_2_root_seed_id": source["root_seed_id"],
            "source_v2_2_medoid_sample_id": source["medoid_sample_id"],
            "source_v2_2_medoid_design_hash": source["medoid_design_hash"],
            "source_role": "coordinate_seed_only_no_v2_2_labels_reused",
            "design_id": design.design_id(),
            "design": design.as_mapping(),
            "normalized_design": list(normalizer.transform(design)),
            "geometry19_projection_hash": design.geometry19_projection_hash(),
            "design_schema_hash": result.design_schema_hash,
            "physics_bundle_hash": result.physics_bundle_hash,
            "physics_model_version": result.physics_model_version,
            "solver_converged": engine.solver.converged,
            "solver_failure_code": engine.solver.failure_code.value if engine.solver.failure_code else None,
            "flow_root_count": engine.operating_point.get("flow_root_count"),
            "performance": performance,
            "legacy_partition_development_diagnostic": diagnostic,
            "engine_feasible": bool(engine.feasible),
            "candidate_for_local_pilot": candidate,
            "training_allowed": False,
            "formal_use_allowed": False,
            "deployment_allowed": False,
        })
    if len({row["design_id"] for row in rows}) != len(rows):
        raise V3LocalCenterAuditError("V3 local center designs are not unique")
    identity = {
        "config_sha256": _sha(root / CONFIG),
        "source_anchor_sha256": _sha(root / config["source_anchor_registry"]["path"]),
        "diagnostic_partition_sha256": profile.source_hash,
        "audit_code_sha256": _sha(root / CODE),
        "runtime_contract": runtime.contract_mapping(),
        "new_coordinate_values": appended,
    }
    return rows, identity


def verify_v3_local_center_audit(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    config = _read_config(root)
    try:
        manifest = json.loads((output / "complete_manifest.json").read_text(encoding="utf-8"))
        cases = output / "centers.jsonl"
        if _sha(cases) != manifest["files"]["centers.jsonl"]:
            raise V3LocalCenterAuditError("V3 local center file changed")
        rows = [json.loads(line) for line in cases.read_text(encoding="utf-8").splitlines() if line.strip()]
        runtime = build_v3_development_physics_runtime(root)
        normalizer = Geometry22Normalizer()
        if (
            manifest["status"] != "software_local_center_audit_only"
            or manifest["row_count"] != len(rows)
            or manifest["row_count"] != config["source_anchor_registry"]["expected_rows"]
            or manifest["identity"]["config_sha256"] != _sha(root / CONFIG)
            or manifest["identity"]["source_anchor_sha256"] != _sha(root / config["source_anchor_registry"]["path"])
            or manifest["identity"]["diagnostic_partition_sha256"] != config["diagnostic_partition"]["sha256"]
            or manifest["identity"]["audit_code_sha256"] != _sha(root / CODE)
            or manifest["identity"]["runtime_contract"] != runtime.contract_mapping()
            or manifest["solver_converged_count"] != sum(row["solver_converged"] for row in rows)
            or manifest["candidate_count"] != sum(row["candidate_for_local_pilot"] for row in rows)
            or manifest["engine_feasible_count"] != sum(row["engine_feasible"] for row in rows)
            or manifest["source_v2_2_labels_reused"] is not False
            or manifest["training_allowed"] is not False
            or manifest["formal_use_allowed"] is not False
            or manifest["deployment_allowed"] is not False
        ):
            raise V3LocalCenterAuditError("V3 local center audit identity changed")
        source_rows = _source_rows(root, config)
        for index, (row, source) in enumerate(zip(rows, source_rows, strict=True)):
            design = Geometry22Design.from_mapping(row["design"])
            if (
                row["source_index"] != index
                or row["source_v2_2_root_seed_id"] != source["root_seed_id"]
                or row["source_v2_2_medoid_sample_id"] != source["medoid_sample_id"]
                or row["source_v2_2_medoid_design_hash"] != source["medoid_design_hash"]
                or design.geometry19_projection().as_mapping() != source["medoid_geometry19"]
                or design.component_parameters() != manifest["identity"]["new_coordinate_values"]
                or row["design_id"] != design.design_id()
                or row["normalized_design"] != list(normalizer.transform(design))
                or row["geometry19_projection_hash"] != design.geometry19_projection_hash()
                or row["design_schema_hash"] != runtime.design_schema_hash
                or row["physics_bundle_hash"] != runtime.physics_bundle_hash
                or row["candidate_for_local_pilot"] != bool(
                    row["legacy_partition_development_diagnostic"]["numerically_valid"]
                    and row["legacy_partition_development_diagnostic"]["physics_domain_valid"]
                )
                or row["legacy_partition_development_diagnostic"]["constraint_profile_hash"]
                != config["diagnostic_partition"]["sha256"]
                or (row["performance"] is None) != (not row["solver_converged"])
                or row["training_allowed"] is not False
                or row["formal_use_allowed"] is not False
                or row["deployment_allowed"] is not False
            ):
                raise V3LocalCenterAuditError("V3 local center row identity changed")
    except V3LocalCenterAuditError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise V3LocalCenterAuditError("Cannot verify V3 local center audit") from exc
    return manifest


def build_v3_local_center_audit(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _read_config(root)
    output = Path(output_root).resolve() if output_root is not None else root / config["output_root"]
    if output.exists():
        raise V3LocalCenterAuditError(f"Refusing to overwrite V3 local center audit: {output}")
    rows, identity = _evaluate(root, config)
    content = b"".join(_bytes(row) for row in rows)
    manifest = {
        "status": "software_local_center_audit_only",
        "identity": identity,
        "row_count": len(rows),
        "solver_converged_count": sum(row["solver_converged"] for row in rows),
        "candidate_count": sum(row["candidate_for_local_pilot"] for row in rows),
        "engine_feasible_count": sum(row["engine_feasible"] for row in rows),
        "source_v2_2_labels_reused": False,
        "training_allowed": False,
        "physics_validation_status": "not_completed",
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "files": {"centers.jsonl": hashlib.sha256(content).hexdigest()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".v3_local_centers_", dir=output.parent))
    try:
        (temporary / "centers.jsonl").write_bytes(content)
        (temporary / "complete_manifest.json").write_bytes(_bytes(manifest))
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return verify_v3_local_center_audit(root, output)
