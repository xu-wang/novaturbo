"""Training-data acquisition invariants without retraining or physics reruns."""

import hashlib
import json
from pathlib import Path

import pytest

from src.ai.v3_local_training_data import (
    V3LocalTrainingDataError,
    verify_v3_local_training_coordinates,
    verify_v3_local_training_labels,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/validation/v3_22d_local_training_data_v1"


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_frozen_coordinates_are_unique_center_disjoint_and_balanced():
    manifest = verify_v3_local_training_coordinates(ROOT, OUTPUT, rebuild=True)
    rows = _rows(OUTPUT / "coordinates.jsonl")
    registry = _rows(ROOT / "data/validation/v3_22d_local_data_contract_v1/center_split_registry.jsonl")
    assignment = {row["center_design_id"]: row["center_split"] for row in registry}
    assert manifest["coordinate_count"] == len(rows) == 16896
    assert manifest["split_counts"] == {"development": 11776, "validation": 2560, "holdout": 2560}
    assert manifest["band_counts"] == {"0": 4224, "1": 4224, "2": 4224, "3": 4224}
    assert len({row["design_id"] for row in rows}) == len(rows)
    assert all(row["center_split"] == assignment[row["center_design_id"]] for row in rows)
    assert all(row["training_allowed"] is False for row in rows)
    for name in ("v3_22d_local_pilot_v1", "v3_22d_local_acquisition_v1"):
        earlier = _rows(ROOT / "data/validation" / name / "coordinates.jsonl")
        assert not {row["design_id"] for row in rows} & {row["design_id"] for row in earlier}


def test_labels_and_center_band_coverage_are_sealed():
    manifest = verify_v3_local_training_labels(ROOT, OUTPUT)
    coverage = _rows(OUTPUT / "coverage.jsonl")
    assert manifest["row_count"] == 16896
    assert manifest["center_count"] == 33
    assert len(manifest["shard_sha256"]) == len(coverage) == 132
    assert all(row["sample_count"] == 128 for row in coverage)
    assert sum(row["software_regression_candidate_count"] for row in coverage) == manifest["software_regression_candidate_count"]
    assert manifest["training_allowed"] is False
    assert manifest["formal_use_allowed"] is False
    assert manifest["deployment_allowed"] is False


def test_rehashed_coordinate_split_tamper_is_rejected(tmp_path):
    output = tmp_path / "altered"
    output.mkdir()
    coordinate_path = output / "coordinates.jsonl"
    coordinate_path.write_bytes((OUTPUT / "coordinates.jsonl").read_bytes())
    manifest_path = output / "coordinate_manifest.json"
    manifest = json.loads((OUTPUT / "coordinate_manifest.json").read_text())
    rows = _rows(coordinate_path)
    first = next(i for i, row in enumerate(rows) if row["center_split"] == "development")
    second = next(i for i, row in enumerate(rows) if row["center_split"] == "holdout")
    rows[first]["center_split"], rows[second]["center_split"] = (
        rows[second]["center_split"], rows[first]["center_split"]
    )
    coordinate_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
    manifest["coordinate_sha256"] = hashlib.sha256(coordinate_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(V3LocalTrainingDataError, match="coordinate identity changed"):
        verify_v3_local_training_coordinates(ROOT, output)
