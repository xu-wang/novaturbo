"""Materialize sealed, center-disjoint V3 software regression candidates."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import yaml

from src.ai.v3_local_training_data import verify_v3_local_training_labels
from src.ai.v3_local_training_supplement import verify_v3_training_supplement_labels


CONFIG = "config/experiments/geometry22_v3_local_candidate_dataset_v1.yaml"
CODE = "src/ai/v3_local_candidate_dataset.py"
SPLITS = ("development", "validation", "holdout")


class V3LocalCandidateDatasetError(ValueError):
    """A source, split, qualification, or frozen output changed."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _config(root: Path) -> dict[str, Any]:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if (
        config.get("version") != "geometry22_v3_local_candidate_dataset_v1"
        or config.get("scope") != "software_regression_candidates_no_training"
        or set(config.get("sources", {})) != {"base", "development_supplement"}
        or config.get("candidate_rule") != "converged_finite_three_performance_and_borrowed_diagnostic_domain"
        or config.get("split_rule") != "frozen_l2_center_split_no_reassignment"
        or config.get("required_combined_coverage_gate") is not True
        or config.get("training_allowed") is not False
        or config.get("formal_use_allowed") is not False
        or config.get("deployment_allowed") is not False
    ):
        raise V3LocalCandidateDatasetError("V3 candidate dataset config changed scope")
    return config


def _expected(root: Path, config: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    source_rows = {}
    manifests = {}
    for name, spec in config["sources"].items():
        path = root / spec["path"]
        if (
            _sha(path / "complete_manifest.json") != spec["manifest_sha256"]
            or _sha(path / "coordinates.jsonl") != spec["coordinates_sha256"]
            or _sha(path / "labels.jsonl") != spec["labels_sha256"]
        ):
            raise V3LocalCandidateDatasetError(f"Frozen V3 data source changed: {name}")
        manifests[name] = (
            verify_v3_local_training_labels(root, path) if name == "base"
            else verify_v3_training_supplement_labels(root, path)
        )
        source_rows[name] = (_rows(path / "coordinates.jsonl"), _rows(path / "labels.jsonl"))
    if (
        manifests["development_supplement"]["combined_data_coverage_gate_pass"] is not True
        or manifests["development_supplement"]["combined_row_count"] != 17472
    ):
        raise V3LocalCandidateDatasetError("Combined center/band data coverage gate failed")
    output: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    seen = set()
    centers: dict[str, str] = {}
    schema_hashes = set()
    physics_hashes = set()
    for source_name in ("base", "development_supplement"):
        coordinates, labels = source_rows[source_name]
        if len(coordinates) != len(labels):
            raise V3LocalCandidateDatasetError("Coordinate/label count changed")
        for coordinate, label in zip(coordinates, labels, strict=True):
            if coordinate["design_id"] != label["design_id"] or coordinate["center_split"] != label["center_split"]:
                raise V3LocalCandidateDatasetError("Coordinate/label identity changed")
            if not label["software_regression_candidate"]:
                continue
            design_id = coordinate["design_id"]
            center = coordinate["center_design_id"]
            split = coordinate["center_split"]
            if design_id in seen or (center in centers and centers[center] != split):
                raise V3LocalCandidateDatasetError("Duplicate design or cross-split center")
            if source_name == "development_supplement" and split != "development":
                raise V3LocalCandidateDatasetError("Supplement leaked into validation/holdout")
            seen.add(design_id)
            centers[center] = split
            schema_hashes.add(coordinate["design_schema_hash"])
            physics_hashes.add(coordinate["physics_bundle_hash"])
            output[split].append({
                "source": source_name,
                "source_coordinate_index": coordinate["coordinate_index"],
                "design_id": design_id,
                "center_design_id": center,
                "center_split": split,
                "band_index": coordinate["band_index"],
                "normalized_design": coordinate["normalized_design"],
                "performance": label["performance"],
                "design_schema_hash": coordinate["design_schema_hash"],
                "physics_bundle_hash": coordinate["physics_bundle_hash"],
                "software_regression_candidate": True,
                "training_allowed": False,
                "formal_use_allowed": False,
                "deployment_allowed": False,
            })
    if (
        len(schema_hashes) != 1 or len(physics_hashes) != 1
        or sum(len(rows) for rows in output.values()) != manifests["development_supplement"]["combined_software_regression_candidate_count"]
        or len(centers) != 33
        or Counter(centers.values()) != {"development": 23, "validation": 5, "holdout": 5}
    ):
        raise V3LocalCandidateDatasetError("V3 candidate population or runtime changed")
    identity = {
        "config_sha256": _sha(root / CONFIG),
        "code_sha256": _sha(root / CODE),
        "source_hashes": {name: {key: spec[key] for key in ("manifest_sha256", "coordinates_sha256", "labels_sha256")}
                          for name, spec in config["sources"].items()},
        "design_schema_hash": next(iter(schema_hashes)),
        "physics_bundle_hash": next(iter(physics_hashes)),
    }
    return output, identity


def _manifest(rows: dict[str, list[dict[str, Any]]], identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "v3_software_regression_candidates_center_disjoint_coverage_pass",
        "identity": identity,
        "candidate_count": sum(len(group) for group in rows.values()),
        "split_counts": {split: len(rows[split]) for split in SPLITS},
        "center_counts": {split: len({row["center_design_id"] for row in rows[split]}) for split in SPLITS},
        "files": {f"{split}_candidates.jsonl": hashlib.sha256(b"".join(_bytes(row) for row in rows[split])).hexdigest()
                  for split in SPLITS},
        "combined_data_coverage_gate_pass": True,
        "training_allowed": False,
        "physics_validation_status": "not_completed",
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }


def build_v3_local_candidate_dataset(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / config["output_root"]
    if output.exists():
        raise V3LocalCandidateDatasetError(f"Refusing to overwrite V3 candidate dataset: {output}")
    rows, identity = _expected(root, config)
    manifest = _manifest(rows, identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".v3_candidate_dataset_", dir=output.parent))
    try:
        for split in SPLITS:
            (temporary / f"{split}_candidates.jsonl").write_bytes(b"".join(_bytes(row) for row in rows[split]))
        (temporary / "manifest.json").write_bytes(_bytes(manifest))
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return verify_v3_local_candidate_dataset(root, output)


def verify_v3_local_candidate_dataset(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    try:
        rows, identity = _expected(root, _config(root))
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest != _manifest(rows, identity) or any(
            _rows(output / f"{split}_candidates.jsonl") != rows[split]
            or _sha(output / f"{split}_candidates.jsonl") != manifest["files"][f"{split}_candidates.jsonl"]
            for split in SPLITS
        ):
            raise V3LocalCandidateDatasetError("Frozen V3 candidate dataset changed")
    except V3LocalCandidateDatasetError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise V3LocalCandidateDatasetError("Cannot verify V3 candidate dataset") from exc
    return manifest
