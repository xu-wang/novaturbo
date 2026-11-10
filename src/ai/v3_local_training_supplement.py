"""Separate development-only supplement for failed V3 center coverage gates."""

from __future__ import annotations

from collections import Counter
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
from src.ai.v3_local_data_contract import verify_v3_local_data_contract
from src.ai.v3_local_training_data import (
    _bytes, _evaluate_shard, _rows, _sha, _validate_shard,
    verify_v3_local_training_labels,
)
from src.physics.geometry22_data import Geometry22Normalizer
from src.physics.geometry22_design import Geometry22Design
from src.physics.v3_development_runtime import build_v3_development_physics_runtime


CONFIG = "config/experiments/geometry22_v3_local_training_data_dev_supplement_v1.yaml"
CODE = "src/ai/v3_local_training_supplement.py"
GEOMETRY_HELPER = "src/ai/geometry19_performance_gated_coordinates.py"
LABEL_HELPER = "src/ai/v3_local_acquisition.py"
EXPECTED_PLAN = [
    ("41dbc5085a359060bd832c54b3abfb7d1ab727ad7d4c6fa22c0d0ca4f9f41615", 0, 16),
    ("dbbf6262f2b282a51b83383634587e396f8d658cf012531129170eaab391af0f", 1, 32),
    ("dbbf6262f2b282a51b83383634587e396f8d658cf012531129170eaab391af0f", 2, 80),
    ("dbbf6262f2b282a51b83383634587e396f8d658cf012531129170eaab391af0f", 3, 160),
]


class V3LocalTrainingSupplementError(ValueError):
    """The supplemental coordinates or combined gate cannot be trusted."""


def _config(root: Path) -> dict[str, Any]:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    plan = config.get("coordinate_plan", {})
    if (
        config.get("version") != "geometry22_v3_local_training_data_dev_supplement_v1"
        or config.get("scope") != "versioned_development_center_coverage_supplement_only"
        or [(r.get("center_design_id"), r.get("band_index"), r.get("pairs"))
            for r in config.get("development_plan", [])] != EXPECTED_PLAN
        or plan.get("bands") != [[0.0, 0.02], [0.02, 0.04], [0.04, 0.06], [0.06, 0.08]]
        or plan.get("pool_base2_exponent") != 13
        or plan.get("pool_rows_per_band") != 2048
        or plan.get("maximum_attempts_per_band") != 4096
        or plan.get("pair_policy") != "positive_negative_symmetric_no_clipping_or_shrinking"
        or plan.get("integer_policy") != "fixed_at_source_center"
        or plan.get("split") != "development_only_from_frozen_l2_registry"
        or plan.get("adaptive_top_up") is not False
        or config.get("training_allowed") is not False
        or config.get("formal_use_allowed") is not False
        or config.get("deployment_allowed") is not False
    ):
        raise V3LocalTrainingSupplementError("V3 supplement config changed scope")
    return config


def _source(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]], set[str]]:
    base = root / config["source_base"]["path"]
    contract = root / config["source_contract"]["path"]
    center_path = root / config["source_centers"]["path"]
    if (
        _sha(base / "complete_manifest.json") != config["source_base"]["complete_manifest_sha256"]
        or _sha(base / "coordinates.jsonl") != config["source_base"]["coordinate_sha256"]
        or _sha(base / "labels.jsonl") != config["source_base"]["labels_sha256"]
        or _sha(contract / "manifest.json") != config["source_contract"]["manifest_sha256"]
        or _sha(contract / "center_split_registry.jsonl") != config["source_contract"]["registry_sha256"]
        or _sha(center_path) != config["source_centers"]["sha256"]
    ):
        raise V3LocalTrainingSupplementError("Frozen base/L2/L0 source changed")
    verify_v3_local_data_contract(root, contract)
    registry = {row["center_design_id"]: row for row in _rows(contract / "center_split_registry.jsonl")}
    centers = {row["design_id"]: row for row in _rows(center_path)}
    base_coverage = _rows(base / "coverage.jsonl")
    development = [row for row in base_coverage if row["center_split"] == "development"]
    by_center = Counter()
    for row in development:
        by_center[row["center_design_id"]] += row["software_regression_candidate_count"]
    failed_centers = {center for center, count in by_center.items() if count < 384}
    failed_bands = {(row["center_design_id"], row["band_index"])
                    for row in development if row["software_regression_candidate_count"] < 80}
    if (
        failed_centers != {EXPECTED_PLAN[0][0], EXPECTED_PLAN[1][0]}
        or failed_bands != {(EXPECTED_PLAN[1][0], b) for b in (1, 2, 3)}
        or any(registry[center]["center_split"] != "development" for center, _, _ in EXPECTED_PLAN)
    ):
        raise V3LocalTrainingSupplementError("Development-only coverage basis changed")
    excluded = {row["design_id"] for row in _rows(base / "coordinates.jsonl")}
    for path in (root / "data/validation/v3_22d_local_pilot_v1/coordinates.jsonl",
                 root / "data/validation/v3_22d_local_acquisition_v1/coordinates.jsonl"):
        excluded.update(row["design_id"] for row in _rows(path))
    excluded.update(centers)
    return registry, centers, excluded


def _coordinate_rows(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry, centers, excluded = _source(root, config)
    normalizer = Geometry22Normalizer()
    fields = normalizer.fields
    integer = tuple(i for i, field in enumerate(fields) if field["dtype"] == "int64")
    continuous = tuple(i for i in range(22) if i not in integer)
    runtime = build_v3_development_physics_runtime(root)
    rows = []
    screening = Counter()
    plan = config["coordinate_plan"]
    pools: dict[str, list[np.ndarray]] = {}
    for center_id, band, pair_count in EXPECTED_PLAN:
        center = Geometry22Design.from_mapping(centers[center_id]["design"])
        center_norm = np.asarray(normalizer.transform(center), dtype=float)
        if center_id not in pools:
            pools[center_id] = [
                qmc.Sobol(d=20, scramble=True, seed=int(seed) ^ int(center_id[:8], 16))
                .random_base2(m=plan["pool_base2_exponent"])
                for seed in (plan["primary_seed"], plan["reserve_seed"])
            ]
        lower, upper = plan["bands"][band]
        accepted = 0
        for attempt in range(plan["maximum_attempts_per_band"]):
            pool_id, index = divmod(attempt, plan["pool_rows_per_band"])
            draw = pools[center_id][pool_id][band * plan["pool_rows_per_band"] + index]
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
                serialized = list(normalizer.transform(sample))
                actual = math.dist([serialized[i] for i in continuous], [center_norm[i] for i in continuous])
                if (
                    any(sample.coordinates[i] != center.coordinates[i] for i in integer)
                    or hard_failure(sample.as_mapping(), list(fields), 0.2) is not None
                    or sample.design_id() in excluded
                    or not lower < actual <= upper
                    or not math.isclose(actual, radius, rel_tol=1e-9, abs_tol=1e-10)
                ):
                    reason = "geometry_or_identity"
                    break
                pair.append((sign, sample, serialized, actual))
            if reason is not None:
                screening[reason] += 1
                continue
            for sign, sample, serialized, actual in pair:
                rows.append({
                    "coordinate_index": len(rows),
                    "center_design_id": center_id,
                    "center_split": "development",
                    "band_index": band,
                    "pair_index": accepted,
                    "pair_sign": sign,
                    "source_attempt_index": attempt,
                    "requested_normalized_l2_radius": radius,
                    "actual_normalized_l2_radius": actual,
                    "design_id": sample.design_id(),
                    "design": sample.as_mapping(),
                    "normalized_design": serialized,
                    "design_schema_hash": runtime.design_schema_hash,
                    "physics_bundle_hash": runtime.physics_bundle_hash,
                    "training_allowed": False,
                })
                excluded.add(sample.design_id())
            accepted += 1
            if accepted == pair_count:
                break
        if accepted != pair_count:
            raise V3LocalTrainingSupplementError(f"Cannot fill supplement {center_id}:{band}: {accepted}")
    if len(rows) != 576 or len({row["design_id"] for row in rows}) != len(rows):
        raise V3LocalTrainingSupplementError("Supplement count or uniqueness changed")
    identity = {
        "config_sha256": _sha(root / CONFIG),
        "code_sha256": _sha(root / CODE),
        "geometry_helper_sha256": _sha(root / GEOMETRY_HELPER),
        "label_helper_sha256": _sha(root / LABEL_HELPER),
        "base_manifest_sha256": config["source_base"]["complete_manifest_sha256"],
        "base_coordinate_sha256": config["source_base"]["coordinate_sha256"],
        "base_labels_sha256": config["source_base"]["labels_sha256"],
        "l2_manifest_sha256": config["source_contract"]["manifest_sha256"],
        "runtime_contract": runtime.contract_mapping(),
        "screening_rejections": dict(screening),
    }
    return rows, identity


def _coordinate_manifest(rows: list[dict[str, Any]], identity: dict[str, Any], content: bytes) -> dict[str, Any]:
    return {
        "status": "v3_development_supplement_coordinates_frozen_before_labels",
        "identity": identity,
        "coordinate_count": len(rows),
        "center_count": len({row["center_design_id"] for row in rows}),
        "band_counts": dict(Counter(str(row["band_index"]) for row in rows)),
        "coordinate_sha256": hashlib.sha256(content).hexdigest(),
        "training_allowed": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }


def prepare_v3_training_supplement(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / config["output_root"]
    if output.exists():
        raise V3LocalTrainingSupplementError(f"Refusing to overwrite V3 supplement: {output}")
    rows, identity = _coordinate_rows(root, config)
    content = b"".join(_bytes(row) for row in rows)
    manifest = _coordinate_manifest(rows, identity, content)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".v3_dev_supplement_", dir=output.parent))
    try:
        (temporary / "coordinates.jsonl").write_bytes(content)
        (temporary / "coordinate_manifest.json").write_bytes(_bytes(manifest))
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return verify_v3_training_supplement_coordinates(root, output)


def verify_v3_training_supplement_coordinates(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    try:
        expected, identity = _coordinate_rows(root, _config(root))
        content = b"".join(_bytes(row) for row in expected)
        manifest = json.loads((output / "coordinate_manifest.json").read_text(encoding="utf-8"))
        if (
            _rows(output / "coordinates.jsonl") != expected
            or _sha(output / "coordinates.jsonl") != hashlib.sha256(content).hexdigest()
            or manifest != _coordinate_manifest(expected, identity, content)
        ):
            raise V3LocalTrainingSupplementError("V3 supplement coordinate lock changed")
    except V3LocalTrainingSupplementError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise V3LocalTrainingSupplementError("Cannot verify V3 supplement coordinates") from exc
    return manifest


def _groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups = []
    offset = 0
    for center, band, pairs in EXPECTED_PLAN:
        group = rows[offset:offset + 2*pairs]
        if len(group) != 2*pairs or any(
            row["center_design_id"] != center or row["band_index"] != band
            or row["center_split"] != "development" for row in group
        ):
            raise V3LocalTrainingSupplementError("Supplement group plan changed")
        groups.append(group)
        offset += 2*pairs
    if offset != len(rows):
        raise V3LocalTrainingSupplementError("Supplement row count changed")
    return groups


def _supplement_coverage(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = _groups(labels)
    return [
        {
            "center_design_id": center,
            "center_split": "development",
            "band_index": band,
            "sample_count": len(group),
            "solver_converged_count": sum(row["solver_converged"] for row in group),
            "software_regression_candidate_count": sum(row["software_regression_candidate"] for row in group),
            "engine_feasible_count": sum(row["engine_feasible"] for row in group),
        }
        for (center, band, _), group in zip(EXPECTED_PLAN, groups, strict=True)
    ]


def _combined_coverage(base_coverage: list[dict[str, Any]], supplement: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined = {
        (row["center_design_id"], row["band_index"]): dict(row)
        for row in base_coverage
    }
    if len(combined) != 132:
        raise V3LocalTrainingSupplementError("Base coverage population changed")
    for row in supplement:
        key = (row["center_design_id"], row["band_index"])
        prior = combined[key]
        if prior["center_split"] != "development":
            raise V3LocalTrainingSupplementError("Supplement reached non-development center")
        for name in ("sample_count", "solver_converged_count", "software_regression_candidate_count",
                     "engine_feasible_count"):
            prior[name] += row[name]
    return [combined[key] for key in sorted(combined)]


def _complete_manifest(root: Path, config: dict[str, Any], output: Path,
                       labels: list[dict[str, Any]], coverage: list[dict[str, Any]],
                       combined: list[dict[str, Any]], group_hashes: list[str]) -> dict[str, Any]:
    base = root / config["source_base"]["path"]
    base_manifest = json.loads((base / "complete_manifest.json").read_text(encoding="utf-8"))
    minimum_band = config["physics"]["minimum_candidate_per_center_band"]
    minimum_center = config["physics"]["minimum_candidate_per_center"]
    by_center = Counter()
    for row in combined:
        by_center[row["center_design_id"]] += row["software_regression_candidate_count"]
    band_failures = [
        {"center_design_id": row["center_design_id"], "band_index": row["band_index"],
         "candidate_count": row["software_regression_candidate_count"]}
        for row in combined if row["software_regression_candidate_count"] < minimum_band
    ]
    center_failures = [
        {"center_design_id": center, "candidate_count": count}
        for center, count in sorted(by_center.items()) if count < minimum_center
    ]
    return {
        "status": "v3_local_training_development_supplement_complete",
        "coordinate_manifest_sha256": _sha(output / "coordinate_manifest.json"),
        "coordinate_sha256": _sha(output / "coordinates.jsonl"),
        "labels_sha256": _sha(output / "labels.jsonl"),
        "coverage_sha256": _sha(output / "coverage.jsonl"),
        "combined_coverage_sha256": _sha(output / "combined_coverage.jsonl"),
        "group_sha256": group_hashes,
        "base_complete_manifest_sha256": config["source_base"]["complete_manifest_sha256"],
        "row_count": len(labels),
        "center_count": len({row["center_design_id"] for row in labels}),
        "solver_converged_count": sum(row["solver_converged"] for row in labels),
        "software_regression_candidate_count": sum(row["software_regression_candidate"] for row in labels),
        "engine_feasible_count": sum(row["engine_feasible"] for row in labels),
        "combined_row_count": base_manifest["row_count"] + len(labels),
        "combined_software_regression_candidate_count": (
            base_manifest["software_regression_candidate_count"]
            + sum(row["software_regression_candidate"] for row in labels)
        ),
        "combined_band_gate_failures": band_failures,
        "combined_center_gate_failures": center_failures,
        "combined_data_coverage_gate_pass": not band_failures and not center_failures,
        "holdout_rows_added": 0,
        "validation_rows_added": 0,
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


def evaluate_v3_training_supplement(repository_root: str | Path, output_root: str | Path | None = None,
                                    *, workers: int = 4) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / config["output_root"]
    prelabel = verify_v3_training_supplement_coordinates(root, output)
    if workers < 1 or workers > 4:
        raise V3LocalTrainingSupplementError("Workers must be between 1 and 4")
    if any((output / name).exists() for name in (
        "labels.jsonl", "coverage.jsonl", "combined_coverage.jsonl", "complete_manifest.json"
    )):
        raise V3LocalTrainingSupplementError("Supplement labels already exist; refusing to overwrite")
    if _sha(root / config["physics"]["diagnostic_partition_path"]) != config["physics"]["diagnostic_partition_sha256"]:
        raise V3LocalTrainingSupplementError("Borrowed diagnostic profile changed")
    rows = _rows(output / "coordinates.jsonl")
    groups = _groups(rows)
    diagnostic = {
        "path": config["physics"]["diagnostic_partition_path"],
        "sha256": config["physics"]["diagnostic_partition_sha256"],
    }
    schema_hash = prelabel["identity"]["runtime_contract"]["design_schema_hash"]
    physics_hash = prelabel["identity"]["runtime_contract"]["physics_bundle_hash"]
    results: list[list[dict[str, Any]] | None] = [None] * len(groups)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_evaluate_shard, str(root), group, diagnostic): i
                   for i, group in enumerate(groups)}
        for future in as_completed(futures):
            index = futures[future]
            labels = future.result()
            _validate_shard(groups[index], labels, schema_hash, physics_hash, diagnostic["sha256"])
            results[index] = labels
    if any(group is None for group in results):
        raise V3LocalTrainingSupplementError("Missing supplement label group")
    labels = [row for group in results for row in group]
    coverage = _supplement_coverage(labels)
    base_coverage = _rows(root / config["source_base"]["path"] / "coverage.jsonl")
    combined = _combined_coverage(base_coverage, coverage)
    group_dir = output / "groups"
    group_dir.mkdir(exist_ok=True)
    group_hashes = []
    for index, group in enumerate(results):
        path = group_dir / f"part-{index:02d}.jsonl"
        _link_content(path, b"".join(_bytes(row) for row in group))
        group_hashes.append(_sha(path))
    for name, content in (
        ("labels.jsonl", b"".join(_bytes(row) for row in labels)),
        ("coverage.jsonl", b"".join(_bytes(row) for row in coverage)),
        ("combined_coverage.jsonl", b"".join(_bytes(row) for row in combined)),
    ):
        _link_content(output / name, content)
    manifest = _complete_manifest(root, config, output, labels, coverage, combined, group_hashes)
    _link_content(output / "complete_manifest.json", _bytes(manifest))
    return verify_v3_training_supplement_labels(root, output)


def verify_v3_training_supplement_labels(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / config["output_root"]
    prelabel = verify_v3_training_supplement_coordinates(root, output)
    try:
        base = root / config["source_base"]["path"]
        verify_v3_local_training_labels(root, base)
        rows = _rows(output / "coordinates.jsonl")
        groups = _groups(rows)
        labels = _rows(output / "labels.jsonl")
        label_groups = _groups(labels)
        coverage = _rows(output / "coverage.jsonl")
        combined = _rows(output / "combined_coverage.jsonl")
        manifest = json.loads((output / "complete_manifest.json").read_text(encoding="utf-8"))
        hashes = []
        for index, (coordinates, group) in enumerate(zip(groups, label_groups, strict=True)):
            _validate_shard(
                coordinates, group,
                prelabel["identity"]["runtime_contract"]["design_schema_hash"],
                prelabel["identity"]["runtime_contract"]["physics_bundle_hash"],
                config["physics"]["diagnostic_partition_sha256"],
            )
            path = output / "groups" / f"part-{index:02d}.jsonl"
            if _rows(path) != group:
                raise V3LocalTrainingSupplementError("Supplement group/aggregate mismatch")
            hashes.append(_sha(path))
        if (
            len(labels) != 576
            or coverage != _supplement_coverage(labels)
            or combined != _combined_coverage(_rows(base / "coverage.jsonl"), coverage)
            or manifest != _complete_manifest(root, config, output, labels, coverage, combined, hashes)
        ):
            raise V3LocalTrainingSupplementError("Supplement label or combined gate changed")
    except V3LocalTrainingSupplementError:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise V3LocalTrainingSupplementError("Cannot verify V3 supplement labels") from exc
    return manifest
