"""Preregister root-disjoint V3 centers, then label development centers only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from typing import Any

import numpy as np
import yaml

from src.physics.constraint_profiles import ConstraintProfile, assess_constraint_profile
from src.physics.geometry22_data import Geometry22Normalizer
from src.physics.geometry22_design import Geometry22Design, SCHEMA_PATH
from src.physics.v3_development_runtime import build_v3_development_physics_runtime


CONFIG = "config/experiments/geometry22_v3_new_centers_l6d_v1.yaml"
PERFORMANCE = ("thrust_N", "tsfc_kg_N_s", "mass_kg")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def config_at(root: Path) -> dict[str, Any]:
    cfg = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    assert cfg["version"] == "geometry22_v3_new_centers_l6d_v1"
    assert cfg["training_allowed"] is False and cfg["formal_use_allowed"] is False
    for name in ("source_pool", "old_v3_centers", "diagnostic_partition"):
        assert sha(root / cfg[name]["path"]) == cfg[name]["sha256"], name
    return cfg


def registry_rows(root: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    old = read_jsonl(root / cfg["old_v3_centers"]["path"])
    excluded_roots = {row["source_v2_2_root_seed_id"] for row in old}
    excluded_designs = {row["design_id"] for row in old}
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    added = {field["name"]: field["reference_value"] for field in schema["design_variables"][19:]}
    normalizer = Geometry22Normalizer()
    groups: dict[str, list[tuple[str, dict[str, Any], np.ndarray]]] = defaultdict(list)
    # Source performance and V2.2 physics records are deliberately not accessed.
    for source in read_jsonl(root / cfg["source_pool"]["path"]):
        root_id = source["root_seed_id"]
        if root_id in excluded_roots:
            continue
        design = Geometry22Design.from_mapping({**source["geometry19"], **added})
        if design.design_id() in excluded_designs:
            continue
        groups[root_id].append((source["sample_id"], design.as_mapping(), np.asarray(normalizer.transform(design))))
    if len(groups) != 45:
        raise ValueError(f"Expected 45 fresh V2.2 coordinate roots, got {len(groups)}")
    roots = sorted(groups, key=lambda root_id: (hashlib.sha256(root_id.encode()).hexdigest(), root_id))
    split_by_root = {root_id: ("development" if i < 32 else "validation" if i < 38 else "holdout") for i, root_id in enumerate(roots)}
    rows = []
    for root_id in roots:
        candidates = groups[root_id]
        mean = np.mean([item[2] for item in candidates], axis=0)
        sample_id, mapping, vector = min(candidates, key=lambda item: (float(np.linalg.norm(item[2] - mean)), Geometry22Design.from_mapping(item[1]).design_id()))
        design = Geometry22Design.from_mapping(mapping)
        rows.append({
            "root_id": root_id,
            "split": split_by_root[root_id],
            "source_sample_id": sample_id,
            "source_role": "coordinate_only_no_v2_2_performance_reuse",
            "root_candidate_count": len(candidates),
            "design_id": design.design_id(),
            "design": mapping,
            "normalized_design": vector.tolist(),
            "v3_label_status": "unread_at_registration",
            "training_allowed": False,
            "formal_use_allowed": False,
            "deployment_allowed": False,
        })
    if len({row["design_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate registered designs")
    return rows


def register(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    output = root / cfg["registry_output"]
    if output.exists():
        raise FileExistsError(output)
    rows = registry_rows(root, cfg)
    output.mkdir(parents=True)
    write_jsonl(output / "center_registry.jsonl", rows)
    manifest = {
        "status": "preregistered_before_v3_labels",
        "config_sha256": sha(root / CONFIG),
        "source_pool_sha256": cfg["source_pool"]["sha256"],
        "old_v3_centers_sha256": cfg["old_v3_centers"]["sha256"],
        "center_registry_sha256": sha(output / "center_registry.jsonl"),
        "split_counts": {split: sum(row["split"] == split for row in rows) for split in ("development", "validation", "holdout")},
        "v2_2_labels_reused": False,
        "v3_labels_read": False,
        "training_allowed": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def label_development(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    registration = root / cfg["registry_output"]
    manifest = json.loads((registration / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "preregistered_before_v3_labels" or manifest["v3_labels_read"] is not False:
        raise ValueError("Invalid preregistration")
    if manifest["config_sha256"] != sha(root / CONFIG) or manifest["center_registry_sha256"] != sha(registration / "center_registry.jsonl"):
        raise ValueError("Preregistration changed")
    all_rows = read_jsonl(registration / "center_registry.jsonl")
    if all_rows != registry_rows(root, cfg):
        raise ValueError("Registered center selection changed")
    output = root / cfg["development_label_output"]
    if output.exists():
        raise FileExistsError(output)
    profile = ConstraintProfile.from_yaml(root / cfg["diagnostic_partition"]["path"])
    runtime = build_v3_development_physics_runtime(root)
    labeled = []
    for row in all_rows:
        if row["split"] != "development":
            continue
        design = Geometry22Design.from_mapping(row["design"])
        result = runtime.evaluate(design)
        engine = result.engine_result
        diagnostic = assess_constraint_profile(engine, profile, conservation_tolerances=runtime.base_runtime.conservation_tolerances).as_mapping()
        labeled.append({
            "design_id": row["design_id"],
            "root_id": row["root_id"],
            "split": "development",
            "solver_converged": bool(engine.solver.converged),
            "solver_failure_code": engine.solver.failure_code.value if engine.solver.failure_code else None,
            "performance": {name: float(engine.performance[name]) for name in PERFORMANCE} if engine.solver.converged else None,
            "legacy_partition_development_diagnostic": diagnostic,
            "candidate_for_local_pilot": bool(diagnostic["numerically_valid"] and diagnostic["physics_domain_valid"]),
            "engine_feasible": bool(engine.feasible),
            "design_schema_hash": result.design_schema_hash,
            "physics_bundle_hash": result.physics_bundle_hash,
            "physics_model_version": result.physics_model_version,
            "training_allowed": False,
            "formal_use_allowed": False,
            "deployment_allowed": False,
        })
    output.mkdir(parents=True)
    write_jsonl(output / "development_labels.jsonl", labeled)
    result_manifest = {
        "status": "development_only_v3_software_labels",
        "preregistration_sha256": sha(registration / "manifest.json"),
        "center_registry_sha256": manifest["center_registry_sha256"],
        "development_labels_sha256": sha(output / "development_labels.jsonl"),
        "development_count": len(labeled),
        "converged_count": sum(row["solver_converged"] for row in labeled),
        "candidate_count": sum(row["candidate_for_local_pilot"] for row in labeled),
        "engine_feasible_count": sum(row["engine_feasible"] for row in labeled),
        "validation_or_holdout_labels_read": False,
        "v2_2_labels_reused": False,
        "runtime_contract": runtime.contract_mapping(),
        "training_allowed": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }
    write_json(output / "manifest.json", result_manifest)
    return result_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("register", "label-development"))
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.repository_root.resolve()
    cfg = config_at(root)
    result = register(root, cfg) if args.command == "register" else label_development(root, cfg)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
