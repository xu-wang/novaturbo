"""Separate local V3 development supplement for L6e coverage failures."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc
import yaml

from src.ai.geometry19_performance_gated_coordinates import hard_failure
from src.ai.v3_local_training_data import _evaluate_shard, _validate_shard
from src.ai.v3_new_center_local_data_l6e import dump, rows, sha, write_new
from src.physics.geometry22_data import Geometry22Normalizer
from src.physics.geometry22_design import Geometry22Design
from src.physics.v3_development_runtime import build_v3_development_physics_runtime


CONFIG = "config/experiments/geometry22_v3_new_center_local_supplement_l6e_v1.yaml"


def cfg_at(root: Path) -> dict[str, Any]:
    cfg = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if cfg["version"] != "geometry22_v3_new_center_local_supplement_l6e_v1" or cfg["training_allowed"] is not False or cfg["formal_use_allowed"] is not False or cfg["deployment_allowed"] is not False:
        raise ValueError("L6e supplement scope changed")
    for spec, filenames in (("source_base", (("complete_manifest.json", "complete_manifest_sha256"), ("coverage.jsonl", "coverage_sha256"))),
                            ("source_registry", (("", "sha256"),)),
                            ("diagnostic_partition", (("", "sha256"),))):
        for name, key in filenames:
            if sha(root / cfg[spec]["path"] / name) != cfg[spec][key]:
                raise ValueError(f"L6e supplement source changed: {spec}")
    base = root / cfg["source_base"]["path"]
    manifest = json.loads((base / "complete_manifest.json").read_text(encoding="utf-8"))
    failures = {(row["center_design_id"], row["band_index"]) for row in manifest["band_gate_failures"]}
    failing_centers = {row["center_design_id"] for row in manifest["center_gate_failures"]}
    plan = cfg["plan"]
    if len(plan) != 11 or {(row["center_design_id"], row["band_index"]) for row in plan} != failures or {row["center_design_id"] for row in plan} != failing_centers:
        raise ValueError("L6e supplement must match frozen development failures")
    return cfg


def coordinate_rows(root: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    base = root / cfg["source_base"]["path"]
    registry = rows(root / cfg["source_registry"]["path"])
    centers = {row["design_id"]: row for row in registry if row["split"] == "development"}
    excluded = {row["design_id"] for row in registry}
    excluded.update(row["design_id"] for row in rows(base / "coordinates.jsonl"))
    normalizer = Geometry22Normalizer()
    fields = list(normalizer.fields)
    integer = [i for i, field in enumerate(fields) if field["dtype"] == "int64"]
    continuous = [i for i in range(22) if i not in integer]
    runtime = build_v3_development_physics_runtime(root)
    output = []
    for item in cfg["plan"]:
        center_id, band, needed = item["center_design_id"], item["band_index"], item["pairs"]
        center = Geometry22Design.from_mapping(centers[center_id]["design"])
        baseline = np.asarray(normalizer.transform(center), dtype=float)
        pools = [qmc.Sobol(d=20, scramble=True, seed=int(seed) ^ int(center_id[:8], 16)).random_base2(m=cfg["pool_base2_exponent"])
                 for seed in (cfg["primary_seed"], cfg["reserve_seed"])]
        lower, upper = cfg["bands"][band]
        accepted = 0
        for attempt in range(cfg["maximum_attempts_per_band"]):
            pool_id, offset = divmod(attempt, cfg["pool_rows_per_band"])
            draw = pools[pool_id][band * cfg["pool_rows_per_band"] + offset]
            direction = 2 * draw[:19] - 1
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            radius = lower + (upper - lower) * (1e-6 + (1 - 2e-6) * float(draw[19]))
            delta = radius * direction / norm
            pair = []
            for sign in (-1, 1):
                vector = baseline.copy()
                vector[continuous] += sign * delta
                if np.any(vector < -1) or np.any(vector > 1):
                    break
                design = normalizer.inverse(vector.tolist())
                normalized = list(normalizer.transform(design))
                actual = math.dist([normalized[i] for i in continuous], [baseline[i] for i in continuous])
                if (any(design.coordinates[i] != center.coordinates[i] for i in integer)
                        or hard_failure(design.as_mapping(), fields, 0.2) is not None
                        or design.design_id() in excluded
                        or not lower < actual <= upper
                        or not math.isclose(actual, radius, rel_tol=1e-9, abs_tol=1e-10)):
                    break
                pair.append((sign, design, normalized, actual))
            if len(pair) != 2:
                continue
            for sign, design, normalized, actual in pair:
                output.append({
                    "coordinate_index": len(output),
                    "center_design_id": center_id,
                    "center_split": "development",
                    "band_index": band,
                    "pair_index": accepted,
                    "pair_sign": sign,
                    "source_attempt_index": attempt,
                    "requested_normalized_l2_radius": radius,
                    "actual_normalized_l2_radius": actual,
                    "design_id": design.design_id(),
                    "design": design.as_mapping(),
                    "normalized_design": normalized,
                    "design_schema_hash": runtime.design_schema_hash,
                    "physics_bundle_hash": runtime.physics_bundle_hash,
                    "training_allowed": False,
                })
                excluded.add(design.design_id())
            accepted += 1
            if accepted == needed:
                break
        if accepted != needed:
            raise ValueError(f"Cannot fill L6e supplement {center_id}:{band}: {accepted}")
    if len(output) != 1216 or len({row["design_id"] for row in output}) != len(output):
        raise ValueError("L6e supplement coordinate count changed")
    return output


def prepare(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    output = root / cfg["output_root"]
    if output.exists():
        raise FileExistsError(output)
    generated = coordinate_rows(root, cfg)
    output.mkdir(parents=True)
    write_new(output / "coordinates.jsonl", b"".join(dump(row) for row in generated))
    manifest = {
        "status": "l6e_development_supplement_coordinates_frozen_before_labels",
        "config_sha256": sha(root / CONFIG),
        "base_manifest_sha256": cfg["source_base"]["complete_manifest_sha256"],
        "coordinate_sha256": sha(output / "coordinates.jsonl"),
        "coordinate_count": len(generated),
        "center_count": 5,
        "validation_or_holdout_sampled_or_labeled": False,
        "training_allowed": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }
    write_new(output / "coordinate_manifest.json", dump(manifest))
    return manifest


def evaluate(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    base = root / cfg["source_base"]["path"]
    output = root / cfg["output_root"]
    pre = json.loads((output / "coordinate_manifest.json").read_text(encoding="utf-8"))
    if pre["status"] != "l6e_development_supplement_coordinates_frozen_before_labels" or pre["config_sha256"] != sha(root / CONFIG) or pre["coordinate_sha256"] != sha(output / "coordinates.jsonl"):
        raise ValueError("L6e supplement coordinate lock changed")
    coordinates = rows(output / "coordinates.jsonl")
    if coordinates != coordinate_rows(root, cfg):
        raise ValueError("L6e supplement deterministic coordinates changed")
    if (output / "complete_manifest.json").exists():
        raise FileExistsError(output / "complete_manifest.json")
    runtime = build_v3_development_physics_runtime(root)
    diagnostic = cfg["diagnostic_partition"]
    supplemental = []
    for item in cfg["plan"]:
        chunk = [row for row in coordinates if row["center_design_id"] == item["center_design_id"] and row["band_index"] == item["band_index"]]
        labeled = _evaluate_shard(str(root), chunk, diagnostic)
        _validate_shard(chunk, labeled, runtime.design_schema_hash, runtime.physics_bundle_hash, diagnostic["sha256"])
        supplemental.extend(labeled)
    write_new(output / "labels.jsonl", b"".join(dump(row) for row in supplemental))
    counts = Counter((row["center_design_id"], row["band_index"]) for row in supplemental if row["software_regression_candidate"])
    coverage = rows(base / "coverage.jsonl")
    combined = []
    for row in coverage:
        copy = dict(row)
        copy["supplement_candidate_count"] = counts[(row["center_design_id"], row["band_index"])]
        copy["combined_candidate_count"] = row["software_regression_candidate_count"] + copy["supplement_candidate_count"]
        combined.append(copy)
    write_new(output / "combined_coverage.jsonl", b"".join(dump(row) for row in combined))
    by_center = Counter()
    for row in combined:
        by_center[row["center_design_id"]] += row["combined_candidate_count"]
    failures = [row for row in combined if row["combined_candidate_count"] < 80]
    center_failures = [{"center_design_id": center, "candidate_count": count} for center, count in sorted(by_center.items()) if count < 384]
    manifest = {
        "status": "l6e_development_supplement_v3_software_labels_complete",
        "coordinate_manifest_sha256": sha(output / "coordinate_manifest.json"),
        "labels_sha256": sha(output / "labels.jsonl"),
        "combined_coverage_sha256": sha(output / "combined_coverage.jsonl"),
        "row_count": len(supplemental),
        "solver_converged_count": sum(row["solver_converged"] for row in supplemental),
        "software_regression_candidate_count": sum(row["software_regression_candidate"] for row in supplemental),
        "engine_feasible_count": sum(row["engine_feasible"] for row in supplemental),
        "combined_candidate_count": sum(row["combined_candidate_count"] for row in combined),
        "combined_minimum_center_candidate_count": min(by_center.values()),
        "combined_minimum_center_band_candidate_count": min(row["combined_candidate_count"] for row in combined),
        "band_gate_failures": failures,
        "center_gate_failures": center_failures,
        "data_coverage_gate_pass": not failures and not center_failures,
        "validation_or_holdout_sampled_or_labeled": False,
        "training_allowed": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }
    write_new(output / "complete_manifest.json", dump(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cfg = cfg_at(root)
    result = prepare(root, cfg) if args.command == "prepare" else evaluate(root, cfg)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
