"""Freeze V3 target-to-local-center routing data without training a model."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


CONFIG = "config/experiments/geometry22_v3_target_route_l6a_v1.yaml"
CODE = "src/ai/v3_target_route_protocol.py"
NAMES = ("thrust_N", "tsfc_kg_N_s", "mass_kg")
SPLITS = ("development", "validation", "holdout")


class V3TargetRouteProtocolError(ValueError):
    """A routing source, split, or sealed artifact changed."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_new(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(value if isinstance(value, bytes) else _bytes(value))


def _config(root: Path) -> dict[str, Any]:
    cfg = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if (
        cfg.get("version") != "geometry22_v3_target_route_l6a_v1"
        or cfg.get("scope") != "software_target_routing_protocol_and_coverage_audit_no_training"
        or cfg.get("runtime_condition") != list(NAMES)
        or cfg.get("runtime_input_only") != "three_positive_performance_targets"
        or cfg.get("route_classes") != "development_center_ids_sorted_lexicographically"
        or cfg.get("route_class_count") != 23
        or cfg.get("target_transform") != "natural_log_then_route_fit_only_equal_center_mean_std"
        or cfg.get("internal_split", {}).get("unit") != "source_center_and_pair_index"
        or cfg["internal_split"].get("allocation") != "each_center_and_band_sorted_pair_modulo_5_eq_0_for_tuning"
        or cfg["internal_split"].get("fitting_source") != "development_only"
        or cfg.get("target_support", {}).get("metric") != "euclidean_l2_of_normalized_log_performance"
        or cfg["target_support"].get("radius") != "fitting_rows_distance_p95_per_center"
        or cfg["target_support"].get("top_k") != [1, 2]
        or cfg["target_support"].get("no_eligible_center") != "abstain"
        or cfg.get("design_support", {}).get("radius") != 0.08
        or cfg["design_support"].get("integer_policy") != "fixed_at_source_center"
        or cfg.get("cross_center_diagnostic", {}).get("selected_bands") != [0, 2, 3]
        or cfg["cross_center_diagnostic"].get("selection_within_center_band") != "smallest_design_id"
        or cfg.get("physical_gpu_for_future_training") != 6
        or cfg.get("source_records_training_allowed") is not False
        or cfg.get("training_executed") is not False
        or cfg.get("formal_use_allowed") is not False
        or cfg.get("deployment_allowed") is not False
    ):
        raise V3TargetRouteProtocolError("L6a routing protocol changed scope")
    return cfg


def _sources(root: Path, cfg: dict[str, Any]) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    spec = cfg["source_dataset"]
    source = root / spec["path"]
    if _sha(source / "manifest.json") != spec["manifest_sha256"]:
        raise V3TargetRouteProtocolError("L4 dataset manifest changed")
    for split in SPLITS:
        if _sha(source / f"{split}_candidates.jsonl") != spec[f"{split}_sha256"]:
            raise V3TargetRouteProtocolError(f"L4 {split} dataset changed")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["split_counts"] != spec["expected_counts"]
        or manifest["center_counts"] != spec["expected_centers"]
        or manifest["identity"]["design_schema_hash"] != cfg["schema"]["design_schema_hash"]
        or manifest["identity"]["physics_bundle_hash"] != cfg["schema"]["physics_bundle_hash"]
        or manifest["training_allowed"] is not False
        or manifest["formal_use_allowed"] is not False
        or manifest["deployment_allowed"] is not False
    ):
        raise V3TargetRouteProtocolError("L4 source population or scope changed")
    for name in ("center_registry", "center_designs"):
        if _sha(root / cfg[name]["path"]) != cfg[name]["sha256"]:
            raise V3TargetRouteProtocolError(f"L6a {name} changed")
    registry_rows = _rows(root / cfg["center_registry"]["path"])
    design_rows = _rows(root / cfg["center_designs"]["path"])
    registry = {r["center_design_id"]: r for r in registry_rows}
    designs = {r["design_id"]: r for r in design_rows}
    if len(registry) != 33 or len(designs) != 33 or set(registry) != set(designs):
        raise V3TargetRouteProtocolError("L6a center registration changed")
    schema_path = root / cfg["schema"]["path"]
    if _sha(schema_path) != cfg["schema"]["design_schema_hash"]:
        raise V3TargetRouteProtocolError("L6a schema file changed")
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    variables = sorted(schema["design_variables"], key=lambda v: v["index"])
    if len(variables) != 22 or [v["index"] for v in variables] != list(range(1, 23)):
        raise V3TargetRouteProtocolError("L6a schema dimension changed")
    integer = [v["name"] for v in variables if v["dtype"] == "int64"]
    if set(integer) != {"compressor_blade_count", "combustor_injector_count", "turbine_blade_count"}:
        raise V3TargetRouteProtocolError("L6a integer axes changed")
    for center, record in registry.items():
        if (
            record["design_schema_hash"] != cfg["schema"]["design_schema_hash"]
            or record["physics_bundle_hash"] != cfg["schema"]["physics_bundle_hash"]
            or {name: designs[center]["design"][name] for name in integer} != record["integer_state"]
        ):
            raise V3TargetRouteProtocolError("L6a center/schema identity changed")
    return source, manifest, registry, designs


def _split_rows(source: Path, split: str, cfg: dict[str, Any], registry: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _rows(source / f"{split}_candidates.jsonl")
    spec = cfg["source_dataset"]
    if len(rows) != spec["expected_counts"][split] or len({r["center_design_id"] for r in rows}) != spec["expected_centers"][split]:
        raise V3TargetRouteProtocolError(f"L6a {split} population changed")
    if len({r["design_id"] for r in rows}) != len(rows):
        raise V3TargetRouteProtocolError(f"L6a duplicate {split} design")
    for row in rows:
        values = [row["performance"][name] for name in NAMES]
        if (
            row["center_split"] != split
            or registry[row["center_design_id"]]["center_split"] != split
            or row["design_schema_hash"] != cfg["schema"]["design_schema_hash"]
            or row["physics_bundle_hash"] != cfg["schema"]["physics_bundle_hash"]
            or row["software_regression_candidate"] is not True
            or row["training_allowed"] is not False
            or row["formal_use_allowed"] is not False
            or row["deployment_allowed"] is not False
            or row["band_index"] not in range(4)
            or len(row["normalized_design"]) != 22
            or any(not math.isfinite(value) or value <= 0 for value in values)
        ):
            raise V3TargetRouteProtocolError(f"L6a {split} identity or labels changed")
    return rows


def _internal_registry(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
    for row in rows:
        groups[(row["center_design_id"], row["band_index"])].add((row["source"], row["source_coordinate_index"] // 2))
    assignments = {
        (center, band, source, pair): "tuning" if index % 5 == 0 else "fit"
        for (center, band), pairs in groups.items()
        for index, (source, pair) in enumerate(sorted(pairs))
    }
    return [{
        "target_id": row["design_id"],
        "source_center_id": row["center_design_id"],
        "band_index": row["band_index"],
        "pair_key": f"{row['source']}:{row['source_coordinate_index'] // 2}",
        "internal_split": assignments[(row["center_design_id"], row["band_index"], row["source"], row["source_coordinate_index"] // 2)],
        "target_performance": {name: row["performance"][name] for name in NAMES},
    } for row in rows]


def _target_array(records: list[dict[str, Any]]) -> np.ndarray:
    return np.log(np.asarray([[r["target_performance"][name] for name in NAMES] for r in records], dtype=np.float64))


def _development_assets(rows: list[dict[str, Any]], registry: dict[str, dict[str, Any]],
                        designs: dict[str, dict[str, Any]], cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    records = _internal_registry(rows)
    fit_records = [record for record in records if record["internal_split"] == "fit"]
    centers = sorted({r["source_center_id"] for r in fit_records})
    if len(centers) != 23:
        raise V3TargetRouteProtocolError("L6a fitting centers changed")
    logs = _target_array(fit_records)
    center_arrays = {center: logs[[i for i, r in enumerate(fit_records) if r["source_center_id"] == center]]
                     for center in centers}
    means = np.asarray([center_arrays[center].mean(axis=0) for center in centers])
    mean = means.mean(axis=0)
    variance = np.mean([np.mean((center_arrays[center] - mean)**2, axis=0) for center in centers], axis=0)
    std = np.sqrt(variance)
    if np.any(std <= 1e-12):
        raise V3TargetRouteProtocolError("L6a target scale collapsed")
    normalization = {"fit_split": "development_internal_fit", "condition_order": list(NAMES),
                     "log_mean": mean.tolist(), "log_std": std.tolist(), "equal_center_weight": True}
    support = []
    for index, center in enumerate(centers):
        z = (center_arrays[center] - mean) / std
        centroid = z.mean(axis=0)
        distance = np.linalg.norm(z - centroid, axis=1)
        radius = float(np.quantile(distance, 0.95))
        if not math.isfinite(radius) or radius <= 0:
            raise V3TargetRouteProtocolError("L6a empty target support")
        design = designs[center]
        support.append({
            "center_id": center,
            "route_class_index": index,
            "integer_state": registry[center]["integer_state"],
            "center_normalized_design": design["normalized_design"],
            "design_support_radius": cfg["design_support"]["radius"],
            "fit_rows": len(center_arrays[center]),
            "target_log_min": center_arrays[center].min(axis=0).tolist(),
            "target_log_max": center_arrays[center].max(axis=0).tolist(),
            "target_centroid_normalized_log": centroid.tolist(),
            "target_radius_p95_normalized_log": radius,
        })
    center_matrix = np.asarray([s["target_centroid_normalized_log"] for s in support])
    radii = np.asarray([s["target_radius_p95_normalized_log"] for s in support])
    overlap_pairs = sum(bool(float(np.linalg.norm(center_matrix[i] - center_matrix[j])) <= radii[i] + radii[j])
                        for i in range(23) for j in range(i + 1, 23))
    counts = Counter(r["internal_split"] for r in records)
    per_center = {center: dict(Counter(r["internal_split"] for r in records if r["source_center_id"] == center))
                  for center in centers}
    audit = {"development_targets": len(records), "fit_targets": counts["fit"], "tuning_targets": counts["tuning"],
             "centers": len(centers), "center_target_support_overlap_pairs": overlap_pairs,
             "possible_center_pairs": 253, "per_center_internal_counts": per_center,
             "training_executed": False, "holdout_content_parsed": False}
    return {"records": records}, normalization, {"centers": support}, audit


def _expected_freeze(root: Path) -> dict[str, Any]:
    cfg = _config(root)
    source, manifest, registry, designs = _sources(root, cfg)
    rows = _split_rows(source, "development", cfg, registry)
    targets, normalization, support, audit = _development_assets(rows, registry, designs, cfg)
    lock = {
        "status": "v3_l6a_target_route_protocol_frozen_before_cross_center_audit",
        "config_sha256": _sha(root / CONFIG), "code_sha256": _sha(root / CODE),
        "source_manifest_sha256": cfg["source_dataset"]["manifest_sha256"],
        "split_sha256": {split: cfg["source_dataset"][f"{split}_sha256"] for split in SPLITS},
        "design_schema_hash": manifest["identity"]["design_schema_hash"],
        "physics_bundle_hash": manifest["identity"]["physics_bundle_hash"],
        "route_center_ids": [s["center_id"] for s in support["centers"]],
        "runtime_condition": list(NAMES), "holdout_content_parsed": False,
        "validation_content_parsed": False, "training_executed": False,
        "formal_use_allowed": False, "deployment_allowed": False,
    }
    return {"protocol_lock.json": lock, "target_registry.jsonl": b"".join(_bytes(r) for r in targets["records"]),
            "normalization.json": normalization, "center_support.json": support,
            "development_audit.json": audit}


def freeze_v3_target_route(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    cfg = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / cfg["output_root"]
    if output.exists():
        raise V3TargetRouteProtocolError("L6a output exists; refusing to overwrite")
    assets = _expected_freeze(root)
    output.mkdir(parents=True)
    for name, value in assets.items():
        _write_new(output / name, value)
    return verify_v3_target_route_freeze(root, output)


def verify_v3_target_route_freeze(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root, output = Path(repository_root).resolve(), Path(output_root).resolve()
    expected = _expected_freeze(root)
    for name, value in expected.items():
        content = value if isinstance(value, bytes) else _bytes(value)
        if (output / name).read_bytes() != content:
            raise V3TargetRouteProtocolError(f"L6a frozen asset changed: {name}")
    return expected["protocol_lock.json"]


def _coverage(rows: list[dict[str, Any]], support: list[dict[str, Any]], normalization: dict[str, Any]) -> dict[str, Any]:
    targets = np.log(np.asarray([[r["performance"][name] for name in NAMES] for r in rows], dtype=np.float64))
    z = (targets - np.asarray(normalization["log_mean"])) / np.asarray(normalization["log_std"])
    center = np.asarray([s["target_centroid_normalized_log"] for s in support])
    radius = np.asarray([s["target_radius_p95_normalized_log"] for s in support])
    score = np.linalg.norm(z[:, None, :] - center[None, :, :], axis=2) / radius[None, :]
    eligible = score <= 1.0
    nearest = score.argmin(axis=1)
    per_center = {}
    for center_id in sorted({r["center_design_id"] for r in rows}):
        indices = [index for index, row in enumerate(rows) if row["center_design_id"] == center_id]
        covered = eligible[indices].any(axis=1)
        per_center[center_id] = {"targets": len(indices), "eligible_any": int(covered.sum()),
                                 "coverage_fraction": float(covered.mean()),
                                 "nearest_normalized_distance_p50": float(np.quantile(score[indices].min(axis=1), 0.5)),
                                 "nearest_normalized_distance_p95": float(np.quantile(score[indices].min(axis=1), 0.95))}
    return {
        "targets": len(rows), "eligible_any": int(eligible.any(axis=1).sum()),
        "coverage_fraction": float(eligible.any(axis=1).mean()),
        "eligible_two_or_more": int((eligible.sum(axis=1) >= 2).sum()),
        "abstentions": int((~eligible.any(axis=1)).sum()),
        "nearest_center_even_if_ineligible_counts": dict(sorted(Counter(support[i]["center_id"] for i in nearest).items())),
        "per_source_center": per_center,
    }


def _expected_audit(root: Path, output: Path) -> dict[str, Any]:
    cfg = _config(root)
    source, _, registry, _ = _sources(root, cfg)
    rows = _split_rows(source, "validation", cfg, registry)
    support = json.loads((output / "center_support.json").read_text(encoding="utf-8"))["centers"]
    normalization = json.loads((output / "normalization.json").read_text(encoding="utf-8"))
    selected = []
    for center in sorted({r["center_design_id"] for r in rows}):
        for band in cfg["cross_center_diagnostic"]["selected_bands"]:
            candidates = [r for r in rows if r["center_design_id"] == center and r["band_index"] == band]
            if not candidates:
                raise V3TargetRouteProtocolError("L6a validation target band absent")
            row = min(candidates, key=lambda r: r["design_id"])
            selected.append({"target_id": row["design_id"], "source_center_id_for_offline_audit_only": center,
                             "band_index": band, "target_performance": {name: row["performance"][name] for name in NAMES}})
    selected_ids = {r["target_id"] for r in selected}
    report = {
        "status": "v3_l6a_old_validation_target_support_diagnostic_only",
        "validation_dataset_sha256": cfg["source_dataset"]["validation_sha256"],
        "development_only_domain": True,
        "validation_used_to_set_threshold": False,
        "old_validation_used_in_l5_selection": True,
        "old_holdout_content_parsed_in_l6a": False,
        "all_validation_targets": _coverage(rows, support, normalization),
        "selected_target_ids": sorted(selected_ids),
        "selected_targets": _coverage([r for r in rows if r["design_id"] in selected_ids], support, normalization),
        "model_training_executed": False,
        "generation_executed": False,
        "physics_evaluations_executed": 0,
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }
    return {"cross_center_targets.jsonl": b"".join(_bytes(r) for r in selected),
            "cross_center_audit.json": report}


def audit_v3_target_route(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    cfg = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / cfg["output_root"]
    verify_v3_target_route_freeze(root, output)
    if any((output / name).exists() for name in ("cross_center_targets.jsonl", "cross_center_audit.json", "complete_manifest.json")):
        raise V3TargetRouteProtocolError("L6a cross-center audit exists; refusing to repeat")
    assets = _expected_audit(root, output)
    for name, value in assets.items():
        _write_new(output / name, value)
    complete = {"status": "v3_l6a_target_route_protocol_and_coverage_audit_complete",
                "artifacts_sha256": {name: _sha(output / name) for name in (
                    "protocol_lock.json", "target_registry.jsonl", "normalization.json", "center_support.json",
                    "development_audit.json", "cross_center_targets.jsonl", "cross_center_audit.json")},
                "training_executed": False, "holdout_content_parsed": False,
                "formal_use_allowed": False, "deployment_allowed": False}
    _write_new(output / "complete_manifest.json", complete)
    return verify_v3_target_route_complete(root, output)


def verify_v3_target_route_complete(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root, output = Path(repository_root).resolve(), Path(output_root).resolve()
    verify_v3_target_route_freeze(root, output)
    expected = _expected_audit(root, output)
    for name, value in expected.items():
        content = value if isinstance(value, bytes) else _bytes(value)
        if (output / name).read_bytes() != content:
            raise V3TargetRouteProtocolError(f"L6a audit asset changed: {name}")
    manifest = json.loads((output / "complete_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["status"] != "v3_l6a_target_route_protocol_and_coverage_audit_complete"
        or manifest["artifacts_sha256"] != {name: _sha(output / name) for name in (
            "protocol_lock.json", "target_registry.jsonl", "normalization.json", "center_support.json",
            "development_audit.json", "cross_center_targets.jsonl", "cross_center_audit.json")}
        or manifest["training_executed"] is not False
        or manifest["holdout_content_parsed"] is not False
        or manifest["formal_use_allowed"] is not False
        or manifest["deployment_allowed"] is not False
    ):
        raise V3TargetRouteProtocolError("L6a manifest changed")
    return manifest
