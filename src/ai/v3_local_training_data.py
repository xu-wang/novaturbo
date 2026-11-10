"""Frozen V3 local training-data coordinates and resumable physics acquisition."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
from scipy.stats import qmc
import yaml

from src.ai.geometry19_performance_gated_coordinates import hard_failure
from src.ai.v3_local_acquisition import _label_rows as label_rows
from src.ai.v3_local_data_contract import verify_v3_local_data_contract
from src.physics.geometry22_data import Geometry22Normalizer
from src.physics.geometry22_design import Geometry22Design
from src.physics.v3_development_runtime import build_v3_development_physics_runtime


CONFIG = "config/experiments/geometry22_v3_local_training_data_v1.yaml"
CODE = "src/ai/v3_local_training_data.py"
GEOMETRY_HELPER = "src/ai/geometry19_performance_gated_coordinates.py"
LABEL_HELPER = "src/ai/v3_local_acquisition.py"
SPLITS = ("development", "validation", "holdout")


class V3LocalTrainingDataError(ValueError):
    """A frozen data identity, coordinate, or label gate failed."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _config(root: Path) -> dict[str, Any]:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    plan = config.get("coordinate_plan", {})
    physics = config.get("physics", {})
    if (
        config.get("version") != "geometry22_v3_local_training_data_v1"
        or config.get("scope") != "software_development_data_acquisition_no_training"
        or plan.get("centers") != 33
        or plan.get("bands") != [[0.0, 0.02], [0.02, 0.04], [0.04, 0.06], [0.06, 0.08]]
        or plan.get("pairs_per_center_band") != 64
        or plan.get("rows_per_center") != 512
        or plan.get("total_rows") != 16896
        or plan.get("normalized_distance") != "euclidean_l2_over_19_continuous_22d_schema_coordinates"
        or plan.get("direction_method") != "scrambled_sobol_19_signed_axes_plus_radial_axis"
        or plan.get("sobol_pool_base2_exponent") != 13
        or plan.get("pool_rows_per_band") != 2048
        or plan.get("maximum_attempts_per_band") != 4096
        or plan.get("pair_policy") != "positive_negative_symmetric_no_clipping_or_shrinking"
        or plan.get("integer_policy") != "fixed_at_source_center"
        or plan.get("geometric_screen") != "v2_2_analytic_geometry_checks_applied_to_22d_projection"
        or plan.get("split") != "inherit_frozen_l2_center_split_before_labels"
        or plan.get("adaptive_top_up") is not False
        or physics.get("software_candidate") != "converged_finite_three_performance_and_borrowed_diagnostic_domain"
        or physics.get("shard_rows") != 128
        or physics.get("minimum_candidate_per_center_band") != 80
        or physics.get("minimum_candidate_per_center") != 384
        or physics.get("adaptive_top_up") is not False
        or config.get("training_allowed") is not False
        or config.get("formal_use_allowed") is not False
        or config.get("deployment_allowed") is not False
    ):
        raise V3LocalTrainingDataError("V3 training-data config changed scope")
    return config


def _source(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], set[str]]:
    contract = root / config["source_contract"]["path"]
    center_path = root / config["source_centers"]["path"]
    if (
        _sha(contract / "manifest.json") != config["source_contract"]["manifest_sha256"]
        or _sha(contract / "center_split_registry.jsonl") != config["source_contract"]["registry_sha256"]
        or _sha(center_path) != config["source_centers"]["sha256"]
    ):
        raise V3LocalTrainingDataError("Frozen L0/L2 source changed")
    verify_v3_local_data_contract(root, contract)
    registry = _rows(contract / "center_split_registry.jsonl")
    centers = {row["design_id"]: row for row in _rows(center_path)}
    if len(registry) != 33 or len(centers) != 33 or {row["center_design_id"] for row in registry} != set(centers):
        raise V3LocalTrainingDataError("Frozen L2 center population changed")
    excluded = set(centers)
    for source in config["excluded_coordinates"]:
        path = root / source["path"]
        if _sha(path) != source["sha256"]:
            raise V3LocalTrainingDataError("Earlier local coordinates changed")
        excluded.update(row["design_id"] for row in _rows(path))
    return registry, centers, excluded


def _coordinate_rows(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry, centers, excluded = _source(root, config)
    normalizer = Geometry22Normalizer()
    runtime = build_v3_development_physics_runtime(root)
    fields = normalizer.fields
    integer = tuple(i for i, field in enumerate(fields) if field["dtype"] == "int64")
    continuous = tuple(i for i in range(22) if i not in integer)
    if len(integer) != 3 or len(continuous) != 19:
        raise V3LocalTrainingDataError("Unexpected V3 integer partition")
    plan = config["coordinate_plan"]
    rows: list[dict[str, Any]] = []
    screening = Counter()
    for item in registry:
        center = centers[item["center_design_id"]]
        if (item["design_schema_hash"] != runtime.design_schema_hash
                or item["physics_bundle_hash"] != runtime.physics_bundle_hash):
            raise V3LocalTrainingDataError("L2 runtime identity changed")
        design = Geometry22Design.from_mapping(center["design"])
        center_norm = np.asarray(normalizer.transform(design), dtype=float)
        seen_for_center: set[str] = set()
        pools = []
        for base_seed in (plan["primary_seed"], plan["reserve_seed"]):
            seed = int(base_seed) ^ int(item["center_design_id"][:8], 16)
            pools.append(qmc.Sobol(d=20, scramble=True, seed=seed).random_base2(m=plan["sobol_pool_base2_exponent"]))
        for band_index, (lower, upper) in enumerate(plan["bands"]):
            accepted = 0
            for attempt in range(plan["maximum_attempts_per_band"]):
                pool_id, index = divmod(attempt, plan["pool_rows_per_band"])
                if pool_id >= len(pools):
                    break
                draw = pools[pool_id][band_index * plan["pool_rows_per_band"] + index]
                direction = 2 * draw[:19] - 1
                magnitude = float(np.linalg.norm(direction))
                if magnitude == 0:
                    screening["zero_direction"] += 1
                    continue
                direction /= magnitude
                radius = lower + (upper - lower) * (1e-6 + (1 - 2e-6) * float(draw[19]))
                delta = radius * direction
                plus = center_norm[list(continuous)] + delta
                minus = center_norm[list(continuous)] - delta
                if np.any(plus > 1) or np.any(plus < -1) or np.any(minus > 1) or np.any(minus < -1):
                    screening["schema_bounds"] += 1
                    continue
                pair = []
                reason = None
                for sign in (-1, 1):
                    normalized = center_norm.copy()
                    normalized[list(continuous)] += sign * delta
                    sample = normalizer.inverse(normalized.tolist())
                    if any(sample.coordinates[i] != design.coordinates[i] for i in integer):
                        reason = "integer_state"
                        break
                    if hard_failure(sample.as_mapping(), list(fields), 0.2) is not None:
                        reason = "hard_geometry"
                        break
                    design_id = sample.design_id()
                    if design_id in excluded or design_id in seen_for_center:
                        reason = "existing_design"
                        break
                    serialized_normalized = list(normalizer.transform(sample))
                    actual = math.dist(
                        [serialized_normalized[i] for i in continuous],
                        [center_norm[i] for i in continuous],
                    )
                    if not lower < actual <= upper or not math.isclose(actual, radius, rel_tol=1e-9, abs_tol=1e-10):
                        reason = "serialized_radius"
                        break
                    pair.append((sign, sample, serialized_normalized, actual))
                if reason is not None:
                    screening[reason] += 1
                    continue
                for sign, sample, normalized, actual in pair:
                    rows.append({
                        "coordinate_index": len(rows),
                        "center_design_id": item["center_design_id"],
                        "center_split": item["center_split"],
                        "band_index": band_index,
                        "pair_index": accepted,
                        "pair_sign": sign,
                        "source_attempt_index": attempt,
                        "requested_normalized_l2_radius": radius,
                        "actual_normalized_l2_radius": actual,
                        "design_id": sample.design_id(),
                        "design": sample.as_mapping(),
                        "normalized_design": normalized,
                        "design_schema_hash": runtime.design_schema_hash,
                        "physics_bundle_hash": runtime.physics_bundle_hash,
                        "training_allowed": False,
                    })
                    seen_for_center.add(sample.design_id())
                    excluded.add(sample.design_id())
                accepted += 1
                if accepted == plan["pairs_per_center_band"]:
                    break
            if accepted != plan["pairs_per_center_band"]:
                raise V3LocalTrainingDataError(
                    f"Cannot fill center {item['center_design_id']} band {band_index}: {accepted}/64"
                )
    if len(rows) != plan["total_rows"] or len({row["design_id"] for row in rows}) != len(rows):
        raise V3LocalTrainingDataError("V3 local coordinate count or uniqueness changed")
    identity = {
        "config_sha256": _sha(root / CONFIG),
        "code_sha256": _sha(root / CODE),
        "geometry_helper_sha256": _sha(root / GEOMETRY_HELPER),
        "label_helper_sha256": _sha(root / LABEL_HELPER),
        "l2_manifest_sha256": config["source_contract"]["manifest_sha256"],
        "l2_registry_sha256": config["source_contract"]["registry_sha256"],
        "runtime_contract": runtime.contract_mapping(),
        "numpy_version": np.__version__,
        "integer_coordinate_indices_1_based": [i + 1 for i in integer],
        "screening_rejections": dict(screening),
    }
    return rows, identity


def _coordinate_manifest(rows: list[dict[str, Any]], identity: dict[str, Any], content: bytes) -> dict[str, Any]:
    return {
        "status": "v3_local_training_coordinates_frozen_before_labels",
        "identity": identity,
        "coordinate_count": len(rows),
        "center_count": len({row["center_design_id"] for row in rows}),
        "split_counts": dict(Counter(row["center_split"] for row in rows)),
        "band_counts": {str(b): sum(row["band_index"] == b for row in rows) for b in range(4)},
        "coordinate_sha256": hashlib.sha256(content).hexdigest(),
        "training_allowed": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }


def prepare_v3_local_training_data(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / config["output_root"]
    if output.exists():
        raise V3LocalTrainingDataError(f"Refusing to overwrite V3 training data: {output}")
    rows, identity = _coordinate_rows(root, config)
    content = b"".join(_bytes(row) for row in rows)
    manifest = _coordinate_manifest(rows, identity, content)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".v3_training_data_", dir=output.parent))
    try:
        (temporary / "coordinates.jsonl").write_bytes(content)
        (temporary / "coordinate_manifest.json").write_bytes(_bytes(manifest))
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return verify_v3_local_training_coordinates(root, output, rebuild=False)


def verify_v3_local_training_coordinates(repository_root: str | Path, output_root: str | Path,
                                         *, rebuild: bool = False) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    try:
        config = _config(root)
        manifest = json.loads((output / "coordinate_manifest.json").read_text(encoding="utf-8"))
        rows = _rows(output / "coordinates.jsonl")
        if (
            manifest["status"] != "v3_local_training_coordinates_frozen_before_labels"
            or manifest["identity"]["config_sha256"] != _sha(root / CONFIG)
            or manifest["identity"]["code_sha256"] != _sha(root / CODE)
            or manifest["identity"]["geometry_helper_sha256"] != _sha(root / GEOMETRY_HELPER)
            or manifest["identity"]["label_helper_sha256"] != _sha(root / LABEL_HELPER)
            or manifest["identity"]["runtime_contract"] != build_v3_development_physics_runtime(root).contract_mapping()
            or manifest["coordinate_count"] != len(rows)
            or manifest["coordinate_count"] != config["coordinate_plan"]["total_rows"]
            or manifest["coordinate_sha256"] != _sha(output / "coordinates.jsonl")
            or manifest["split_counts"] != dict(Counter(row["center_split"] for row in rows))
            or manifest["band_counts"] != {str(b): sum(row["band_index"] == b for row in rows) for b in range(4)}
            or manifest["training_allowed"] is not False
            or manifest["formal_use_allowed"] is not False
            or manifest["deployment_allowed"] is not False
        ):
            raise V3LocalTrainingDataError("V3 training coordinate lock changed")
        registry, centers, excluded = _source(root, config)
        assignment = {row["center_design_id"]: row["center_split"] for row in registry}
        if manifest["center_count"] != len(assignment) or len({r["design_id"] for r in rows}) != len(rows):
            raise V3LocalTrainingDataError("V3 training coordinate population changed")
        normalizer = Geometry22Normalizer()
        integer = [i for i, field in enumerate(normalizer.fields) if field["dtype"] == "int64"]
        counts = Counter()
        for row in rows:
            center_id = row["center_design_id"]
            design = Geometry22Design.from_mapping(row["design"])
            center = Geometry22Design.from_mapping(centers[center_id]["design"])
            band = row["band_index"]
            lower, upper = config["coordinate_plan"]["bands"][band]
            radius = math.dist(
                [normalizer.transform(design)[i] for i in range(22) if i not in integer],
                [normalizer.transform(center)[i] for i in range(22) if i not in integer],
            )
            if (
                row["center_split"] != assignment[center_id]
                or row["design_id"] != design.design_id()
                or row["design_id"] in excluded
                or row["design_schema_hash"] != manifest["identity"]["runtime_contract"]["design_schema_hash"]
                or row["physics_bundle_hash"] != manifest["identity"]["runtime_contract"]["physics_bundle_hash"]
                or row["normalized_design"] != list(normalizer.transform(design))
                or any(design.coordinates[i] != center.coordinates[i] for i in integer)
                or hard_failure(design.as_mapping(), list(normalizer.fields), 0.2) is not None
                or not lower < radius <= upper
                or not math.isclose(radius, row["actual_normalized_l2_radius"], rel_tol=1e-9, abs_tol=1e-10)
                or not math.isclose(radius, row["requested_normalized_l2_radius"], rel_tol=1e-9, abs_tol=1e-10)
                or row["training_allowed"] is not False
            ):
                raise V3LocalTrainingDataError("V3 training coordinate identity changed")
            counts[(center_id, band)] += 1
        if any(counts[(center_id, band)] != 128 for center_id in assignment for band in range(4)):
            raise V3LocalTrainingDataError("V3 center-band quotas changed")
        for low, high in zip(rows[::2], rows[1::2], strict=True):
            if (
                low["center_design_id"] != high["center_design_id"]
                or low["band_index"] != high["band_index"]
                or low["pair_index"] != high["pair_index"]
                or (low["pair_sign"], high["pair_sign"]) != (-1, 1)
                or not math.isclose(math.dist(low["normalized_design"], high["normalized_design"]),
                                    2 * low["actual_normalized_l2_radius"], rel_tol=1e-9, abs_tol=1e-10)
            ):
                raise V3LocalTrainingDataError("V3 local pair identity changed")
        if rebuild:
            expected, identity = _coordinate_rows(root, config)
            content = b"".join(_bytes(row) for row in expected)
            if rows != expected or manifest != _coordinate_manifest(expected, identity, content):
                raise V3LocalTrainingDataError("V3 training coordinate deterministic rebuild changed")
    except V3LocalTrainingDataError:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise V3LocalTrainingDataError("Cannot verify V3 training coordinates") from exc
    return manifest


def _shards(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if len(rows) != 16896:
        raise V3LocalTrainingDataError("Training coordinate count changed before sharding")
    chunks = [rows[i:i + 128] for i in range(0, len(rows), 128)]
    if len(chunks) != 132 or any(
        len(chunk) != 128
        or len({row["center_design_id"] for row in chunk}) != 1
        or len({row["band_index"] for row in chunk}) != 1
        for chunk in chunks
    ):
        raise V3LocalTrainingDataError("V3 shard crosses center/band")
    return chunks


def _evaluate_shard(root_string: str, chunk: list[dict[str, Any]], diagnostic: dict[str, str]) -> list[dict[str, Any]]:
    labels = label_rows(Path(root_string), chunk, {"diagnostic_partition": diagnostic})
    for coordinate, label in zip(chunk, labels, strict=True):
        label["band_index"] = coordinate["band_index"]
        label["actual_normalized_l2_radius"] = coordinate["actual_normalized_l2_radius"]
    return labels


def _validate_shard(coordinates: list[dict[str, Any]], labels: list[dict[str, Any]],
                    schema_hash: str, physics_hash: str, diagnostic_hash: str) -> None:
    if len(coordinates) != len(labels):
        raise V3LocalTrainingDataError("V3 shard row count changed")
    for coordinate, label in zip(coordinates, labels, strict=True):
        diagnostic = label["legacy_partition_development_diagnostic"]
        performance = label["performance"]
        finite = performance is not None and all(
            type(performance.get(name)) in (int, float) and math.isfinite(performance[name])
            for name in ("thrust_N", "tsfc_kg_N_s", "mass_kg")
        )
        if (
            any(label[name] != coordinate[name] for name in (
                "coordinate_index", "design_id", "center_design_id", "center_split",
                "band_index", "pair_index", "pair_sign", "actual_normalized_l2_radius",
            ))
            or label["design_schema_hash"] != schema_hash
            or label["physics_bundle_hash"] != physics_hash
            or diagnostic["constraint_profile_hash"] != diagnostic_hash
            or label["performance_finite"] != finite
            or (not label["solver_converged"] and performance is not None)
            or (label["solver_converged"] and finite and performance is None)
            or (label["solver_failure_code"] is None) != label["solver_converged"]
            or label["software_regression_candidate"] != bool(
                finite and diagnostic["numerically_valid"] and diagnostic["physics_domain_valid"]
            )
            or label["training_allowed"] is not False
            or label["formal_use_allowed"] is not False
            or label["deployment_allowed"] is not False
        ):
            raise V3LocalTrainingDataError("V3 shard label identity or qualification changed")


def _coverage(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for label in labels:
        groups[(label["center_design_id"], label["center_split"], label["band_index"])].append(label)
    return [
        {
            "center_design_id": center,
            "center_split": split,
            "band_index": band,
            "sample_count": len(group),
            "solver_converged_count": sum(row["solver_converged"] for row in group),
            "software_regression_candidate_count": sum(row["software_regression_candidate"] for row in group),
            "engine_feasible_count": sum(row["engine_feasible"] for row in group),
        }
        for (center, split, band), group in sorted(groups.items())
    ]


def _complete_manifest(labels: list[dict[str, Any]], coverage: list[dict[str, Any]],
                       coordinate_manifest_sha: str, coordinate_sha: str,
                       labels_sha: str, coverage_sha: str, shard_hashes: list[str],
                       config: dict[str, Any]) -> dict[str, Any]:
    minimum_band = config["physics"]["minimum_candidate_per_center_band"]
    minimum_center = config["physics"]["minimum_candidate_per_center"]
    by_center = Counter()
    for row in coverage:
        by_center[row["center_design_id"]] += row["software_regression_candidate_count"]
    failures = [
        {"center_design_id": row["center_design_id"], "band_index": row["band_index"],
         "candidate_count": row["software_regression_candidate_count"]}
        for row in coverage if row["software_regression_candidate_count"] < minimum_band
    ]
    center_failures = [
        {"center_design_id": center, "candidate_count": count}
        for center, count in sorted(by_center.items()) if count < minimum_center
    ]
    return {
        "status": "v3_local_training_data_acquisition_complete",
        "coordinate_manifest_sha256": coordinate_manifest_sha,
        "coordinate_sha256": coordinate_sha,
        "labels_sha256": labels_sha,
        "coverage_sha256": coverage_sha,
        "shard_sha256": shard_hashes,
        "diagnostic_partition_sha256": config["physics"]["diagnostic_partition_sha256"],
        "row_count": len(labels),
        "center_count": len(by_center),
        "split_counts": dict(Counter(row["center_split"] for row in labels)),
        "solver_converged_count": sum(row["solver_converged"] for row in labels),
        "software_regression_candidate_count": sum(row["software_regression_candidate"] for row in labels),
        "engine_feasible_count": sum(row["engine_feasible"] for row in labels),
        "minimum_candidate_per_center_band": minimum_band,
        "minimum_candidate_per_center": minimum_center,
        "band_gate_failures": failures,
        "center_gate_failures": center_failures,
        "data_coverage_gate_pass": not failures and not center_failures,
        "source_v2_2_labels_reused": False,
        "training_allowed": False,
        "physics_validation_status": "not_completed",
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }


def _link_content(target: Path, content: bytes) -> None:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(content)
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate_v3_local_training_data(repository_root: str | Path, output_root: str | Path | None = None,
                                    *, workers: int = 4) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / config["output_root"]
    prelabel = verify_v3_local_training_coordinates(root, output)
    if workers < 1 or workers > 16:
        raise V3LocalTrainingDataError("Workers must be between 1 and 16")
    if (output / "complete_manifest.json").exists():
        raise V3LocalTrainingDataError("Complete V3 training data already exists")
    if (output / "labels.jsonl").exists() or (output / "coverage.jsonl").exists():
        raise V3LocalTrainingDataError("Orphan aggregate output exists; inspect before resuming")
    lock = output / ".runner.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise V3LocalTrainingDataError("V3 training data runner lock exists") from exc
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(f"pid={os.getpid()}\n")
        rows = _rows(output / "coordinates.jsonl")
        chunks = _shards(rows)
        shard_dir = output / "shards"
        shard_dir.mkdir(exist_ok=True)
        schema_hash = prelabel["identity"]["runtime_contract"]["design_schema_hash"]
        physics_hash = prelabel["identity"]["runtime_contract"]["physics_bundle_hash"]
        diagnostic = {
            "path": config["physics"]["diagnostic_partition_path"],
            "sha256": config["physics"]["diagnostic_partition_sha256"],
        }
        if _sha(root / diagnostic["path"]) != diagnostic["sha256"]:
            raise V3LocalTrainingDataError("Borrowed diagnostic profile changed")
        pending = []
        for index, chunk in enumerate(chunks):
            path = shard_dir / f"part-{index:05d}.jsonl"
            if path.exists():
                _validate_shard(chunk, _rows(path), schema_hash, physics_hash, diagnostic["sha256"])
            else:
                pending.append(index)
        print(f"V3 local data shards: {len(chunks)-len(pending)} sealed, {len(pending)} pending", flush=True)
        if pending:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_evaluate_shard, str(root), chunks[index], diagnostic): index
                    for index in pending
                }
                completed = 0
                for future in as_completed(futures):
                    index = futures[future]
                    labels = future.result()
                    _validate_shard(chunks[index], labels, schema_hash, physics_hash, diagnostic["sha256"])
                    _link_content(shard_dir / f"part-{index:05d}.jsonl", b"".join(_bytes(row) for row in labels))
                    completed += 1
                    if completed % 8 == 0 or completed == len(pending):
                        print(f"V3 local data shards sealed: {len(chunks)-len(pending)+completed}/{len(chunks)}", flush=True)
        labels = []
        shard_hashes = []
        for index, chunk in enumerate(chunks):
            path = shard_dir / f"part-{index:05d}.jsonl"
            shard_labels = _rows(path)
            _validate_shard(chunk, shard_labels, schema_hash, physics_hash, diagnostic["sha256"])
            labels.extend(shard_labels)
            shard_hashes.append(_sha(path))
        label_content = b"".join(_bytes(row) for row in labels)
        coverage = _coverage(labels)
        coverage_content = b"".join(_bytes(row) for row in coverage)
        manifest = _complete_manifest(
            labels, coverage, _sha(output / "coordinate_manifest.json"),
            prelabel["coordinate_sha256"], hashlib.sha256(label_content).hexdigest(),
            hashlib.sha256(coverage_content).hexdigest(), shard_hashes, config,
        )
        _link_content(output / "labels.jsonl", label_content)
        _link_content(output / "coverage.jsonl", coverage_content)
        _link_content(output / "complete_manifest.json", _bytes(manifest))
    finally:
        lock.unlink(missing_ok=True)
    return verify_v3_local_training_labels(root, output)


def verify_v3_local_training_labels(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / config["output_root"]
    prelabel = verify_v3_local_training_coordinates(root, output)
    try:
        rows = _rows(output / "coordinates.jsonl")
        chunks = _shards(rows)
        labels = _rows(output / "labels.jsonl")
        coverage = _rows(output / "coverage.jsonl")
        manifest = json.loads((output / "complete_manifest.json").read_text(encoding="utf-8"))
        schema_hash = prelabel["identity"]["runtime_contract"]["design_schema_hash"]
        physics_hash = prelabel["identity"]["runtime_contract"]["physics_bundle_hash"]
        diagnostic_hash = config["physics"]["diagnostic_partition_sha256"]
        shard_hashes = []
        for index, chunk in enumerate(chunks):
            path = output / "shards" / f"part-{index:05d}.jsonl"
            shard_labels = _rows(path)
            _validate_shard(chunk, shard_labels, schema_hash, physics_hash, diagnostic_hash)
            if labels[index*128:(index+1)*128] != shard_labels:
                raise V3LocalTrainingDataError("V3 aggregate label/shard mismatch")
            shard_hashes.append(_sha(path))
        if (
            len(labels) != 16896
            or len(coverage) != 132
            or coverage != _coverage(labels)
            or manifest != _complete_manifest(
                labels, coverage, _sha(output / "coordinate_manifest.json"),
                prelabel["coordinate_sha256"], _sha(output / "labels.jsonl"),
                _sha(output / "coverage.jsonl"), shard_hashes, config,
            )
        ):
            raise V3LocalTrainingDataError("V3 complete data manifest changed")
    except V3LocalTrainingDataError:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise V3LocalTrainingDataError("Cannot verify V3 training labels") from exc
    return manifest
