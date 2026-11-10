"""Freeze and label local 22D samples around preregistered development centers."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from src.physics.geometry22_data import Geometry22Normalizer
from src.physics.geometry22_design import Geometry22Design
from src.physics.v3_development_runtime import build_v3_development_physics_runtime


CONFIG = "config/experiments/geometry22_v3_new_center_local_data_l6e_v1.yaml"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write_new(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)


def config_at(root: Path) -> dict[str, Any]:
    cfg = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    plan = cfg["coordinate_plan"]
    if (
        cfg["version"] != "geometry22_v3_new_center_local_data_l6e_v1"
        or cfg["scope"] != "development_only_local_sampling_before_extended_gate"
        or plan["center_count"] != 32
        or plan["bands"] != [[0.0, 0.02], [0.02, 0.04], [0.04, 0.06], [0.06, 0.08]]
        or plan["pairs_per_center_band"] != 64
        or plan["integer_policy"] != "fixed_at_source_center"
        or plan["pair_policy"] != "positive_negative_symmetric_no_clipping_or_shrinking"
        or plan["adaptive_top_up"] is not False
        or cfg["physics"]["adaptive_top_up"] is not False
        or cfg["training_allowed"] is not False
        or cfg["formal_use_allowed"] is not False
        or cfg["deployment_allowed"] is not False
    ):
        raise ValueError("L6e scope changed")
    for spec, files in (("source_registry", ("manifest.json", "center_registry.jsonl")),
                        ("source_development_labels", ("manifest.json", "development_labels.jsonl"))):
        item = cfg[spec]
        for name, key in zip(files, ("manifest_sha256", "registry_sha256" if spec == "source_registry" else "labels_sha256"), strict=True):
            if sha(root / item["path"] / name) != item[key]:
                raise ValueError(f"L6e source changed: {spec}/{name}")
    diagnostic = cfg["physics"]["diagnostic_partition"]
    if sha(root / diagnostic["path"]) != diagnostic["sha256"]:
        raise ValueError("L6e diagnostic profile changed")
    return cfg


def development_centers(root: Path, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], set[str]]:
    registered = rows(root / cfg["source_registry"]["path"] / "center_registry.jsonl")
    labeled = rows(root / cfg["source_development_labels"]["path"] / "development_labels.jsonl")
    centers = [row for row in registered if row["split"] == "development"]
    eligible = {row["design_id"] for row in labeled if row["candidate_for_local_pilot"]}
    if len(registered) != 45 or len(centers) != 32 or len(labeled) != 32 or eligible != {row["design_id"] for row in centers}:
        raise ValueError("L6e development center population changed")
    return centers, {row["design_id"] for row in registered}


def coordinate_rows(root: Path, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    centers, excluded = development_centers(root, cfg)
    normalizer = Geometry22Normalizer()
    fields = list(normalizer.fields)
    integer = [i for i, field in enumerate(fields) if field["dtype"] == "int64"]
    continuous = [i for i in range(22) if i not in integer]
    if len(integer) != 3 or len(continuous) != 19:
        raise ValueError("L6e V3 coordinate partition changed")
    runtime = build_v3_development_physics_runtime(root)
    plan = cfg["coordinate_plan"]
    output = []
    rejected = Counter()
    for source in centers:
        center = Geometry22Design.from_mapping(source["design"])
        baseline = np.asarray(normalizer.transform(center), dtype=float)
        pools = [qmc.Sobol(d=20, scramble=True, seed=int(seed) ^ int(source["design_id"][:8], 16)).random_base2(m=plan["sobol_pool_base2_exponent"])
                 for seed in (plan["primary_seed"], plan["reserve_seed"])]
        for band, (lower, upper) in enumerate(plan["bands"]):
            accepted = 0
            for attempt in range(plan["maximum_attempts_per_band"]):
                pool_id, offset = divmod(attempt, plan["pool_rows_per_band"])
                draw = pools[pool_id][band * plan["pool_rows_per_band"] + offset]
                direction = 2 * draw[:19] - 1
                magnitude = np.linalg.norm(direction)
                if magnitude == 0:
                    rejected["zero_direction"] += 1
                    continue
                radius = lower + (upper - lower) * (1e-6 + (1 - 2e-6) * float(draw[19]))
                delta = radius * direction / magnitude
                pair = []
                failure = None
                for sign in (-1, 1):
                    vector = baseline.copy()
                    vector[continuous] += sign * delta
                    if np.any(vector < -1) or np.any(vector > 1):
                        failure = "schema_bounds"
                        break
                    design = normalizer.inverse(vector.tolist())
                    normalized = list(normalizer.transform(design))
                    actual = math.dist([normalized[i] for i in continuous], [baseline[i] for i in continuous])
                    if (any(design.coordinates[i] != center.coordinates[i] for i in integer)
                            or hard_failure(design.as_mapping(), fields, 0.2) is not None
                            or design.design_id() in excluded
                            or not lower < actual <= upper
                            or not math.isclose(actual, radius, rel_tol=1e-9, abs_tol=1e-10)):
                        failure = "geometry_or_identity"
                        break
                    pair.append((sign, design, normalized, actual))
                if failure:
                    rejected[failure] += 1
                    continue
                for sign, design, normalized, actual in pair:
                    output.append({
                        "coordinate_index": len(output),
                        "center_design_id": source["design_id"],
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
                if accepted == plan["pairs_per_center_band"]:
                    break
            if accepted != plan["pairs_per_center_band"]:
                raise ValueError(f"Cannot fill L6e center {source['design_id']} band {band}: {accepted}")
    if len(output) != 16384 or len({row["design_id"] for row in output}) != len(output):
        raise ValueError("L6e coordinate count or uniqueness changed")
    return output, dict(rejected)


def prepare(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    output = root / cfg["output_root"]
    if output.exists():
        raise FileExistsError(output)
    generated, rejected = coordinate_rows(root, cfg)
    output.mkdir(parents=True)
    write_new(output / "coordinates.jsonl", b"".join(dump(row) for row in generated))
    manifest = {
        "status": "l6e_development_coordinates_frozen_before_labels",
        "config_sha256": sha(root / CONFIG),
        "coordinate_sha256": sha(output / "coordinates.jsonl"),
        "coordinate_count": len(generated),
        "center_count": 32,
        "band_rows_per_center": 128,
        "rejected_attempts": rejected,
        "validation_or_holdout_sampled_or_labeled": False,
        "training_allowed": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }
    write_new(output / "coordinate_manifest.json", dump(manifest))
    return manifest


def evaluate(root: Path, cfg: dict[str, Any], workers: int) -> dict[str, Any]:
    output = root / cfg["output_root"]
    lock = json.loads((output / "coordinate_manifest.json").read_text(encoding="utf-8"))
    coordinates = rows(output / "coordinates.jsonl")
    if lock["status"] != "l6e_development_coordinates_frozen_before_labels" or lock["config_sha256"] != sha(root / CONFIG) or lock["coordinate_sha256"] != sha(output / "coordinates.jsonl"):
        raise ValueError("L6e coordinate lock changed")
    if len(coordinates) != 16384 or any(row["center_split"] != "development" for row in coordinates):
        raise ValueError("L6e coordinates include nondevelopment designs")
    if not 1 <= workers <= 16:
        raise ValueError("Invalid worker count")
    if (output / "complete_manifest.json").exists():
        raise FileExistsError(output / "complete_manifest.json")
    chunks = [coordinates[i:i + 128] for i in range(0, len(coordinates), 128)]
    if len(chunks) != 128 or any(len({(r["center_design_id"], r["band_index"]) for r in chunk}) != 1 for chunk in chunks):
        raise ValueError("L6e shard identity changed")
    runtime = build_v3_development_physics_runtime(root)
    diagnostic = cfg["physics"]["diagnostic_partition"]
    shard_dir = output / "shards"
    shard_dir.mkdir(exist_ok=True)
    pending = [i for i in range(128) if not (shard_dir / f"part-{i:05d}.jsonl").exists()]
    for i in range(128):
        path = shard_dir / f"part-{i:05d}.jsonl"
        if path.exists():
            _validate_shard(chunks[i], rows(path), runtime.design_schema_hash, runtime.physics_bundle_hash, diagnostic["sha256"])
    print(f"L6e shards: {128-len(pending)} sealed, {len(pending)} pending", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_evaluate_shard, str(root), chunks[i], diagnostic): i for i in pending}
            for count, future in enumerate(as_completed(futures), start=1):
                i = futures[future]
                label_rows = future.result()
                _validate_shard(chunks[i], label_rows, runtime.design_schema_hash, runtime.physics_bundle_hash, diagnostic["sha256"])
                write_new(shard_dir / f"part-{i:05d}.jsonl", b"".join(dump(row) for row in label_rows))
                if count % 8 == 0 or count == len(pending):
                    print(f"L6e shards sealed: {128-len(pending)+count}/128", flush=True)
    labels = []
    shard_hashes = []
    for i in range(128):
        path = shard_dir / f"part-{i:05d}.jsonl"
        part = rows(path)
        _validate_shard(chunks[i], part, runtime.design_schema_hash, runtime.physics_bundle_hash, diagnostic["sha256"])
        labels.extend(part)
        shard_hashes.append(sha(path))
    coverage = []
    for i, chunk in enumerate(chunks):
        part = labels[128*i:128*(i+1)]
        coverage.append({
            "center_design_id": chunk[0]["center_design_id"],
            "band_index": chunk[0]["band_index"],
            "sample_count": len(part),
            "solver_converged_count": sum(row["solver_converged"] for row in part),
            "software_regression_candidate_count": sum(row["software_regression_candidate"] for row in part),
            "engine_feasible_count": sum(row["engine_feasible"] for row in part),
        })
    write_new(output / "labels.jsonl", b"".join(dump(row) for row in labels))
    write_new(output / "coverage.jsonl", b"".join(dump(row) for row in coverage))
    by_center = Counter()
    for row in coverage:
        by_center[row["center_design_id"]] += row["software_regression_candidate_count"]
    minimum_band = cfg["physics"]["minimum_candidate_per_center_band"]
    minimum_center = cfg["physics"]["minimum_candidate_per_center"]
    manifest = {
        "status": "l6e_development_local_v3_software_labels_complete",
        "coordinate_manifest_sha256": sha(output / "coordinate_manifest.json"),
        "coordinate_sha256": lock["coordinate_sha256"],
        "labels_sha256": sha(output / "labels.jsonl"),
        "coverage_sha256": sha(output / "coverage.jsonl"),
        "shard_sha256": shard_hashes,
        "row_count": len(labels),
        "center_count": len(by_center),
        "solver_converged_count": sum(row["solver_converged"] for row in labels),
        "software_regression_candidate_count": sum(row["software_regression_candidate"] for row in labels),
        "engine_feasible_count": sum(row["engine_feasible"] for row in labels),
        "band_gate_failures": [row for row in coverage if row["software_regression_candidate_count"] < minimum_band],
        "center_gate_failures": [{"center_design_id": center, "candidate_count": count} for center, count in sorted(by_center.items()) if count < minimum_center],
        "validation_or_holdout_sampled_or_labeled": False,
        "v2_2_labels_reused": False,
        "training_allowed": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }
    manifest["data_coverage_gate_pass"] = not manifest["band_gate_failures"] and not manifest["center_gate_failures"]
    write_new(output / "complete_manifest.json", dump(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cfg = config_at(root)
    result = prepare(root, cfg) if args.command == "prepare" else evaluate(root, cfg, args.workers)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
