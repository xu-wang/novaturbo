"""L3 gates for prelabel center splits and fresh V3 labels."""

import hashlib
import json
import math
from pathlib import Path
import shutil

import pytest

from src.ai.v3_local_acquisition import (
    V3LocalAcquisitionError,
    prepare_v3_local_acquisition,
    verify_v3_local_acquisition_coordinates,
    verify_v3_local_acquisition_labels,
)
from src.physics.geometry22_data import Geometry22Normalizer


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/validation/v3_22d_local_acquisition_v1"


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_coordinates_are_new_center_disjoint_and_paired():
    manifest = verify_v3_local_acquisition_coordinates(ROOT, OUTPUT)
    rows = _rows(OUTPUT / "coordinates.jsonl")
    old = _rows(ROOT / "data/validation/v3_22d_local_pilot_v1/coordinates.jsonl")
    registry = _rows(ROOT / "data/validation/v3_22d_local_data_contract_v1/center_split_registry.jsonl")
    assignment = {row["center_design_id"]: row["center_split"] for row in registry}
    assert manifest["coordinate_count"] == len(rows) == 264
    assert manifest["split_counts"] == {"development": 184, "validation": 40, "holdout": 40}
    assert manifest["radius_counts"] == {"0.02": 132, "0.05": 132}
    assert len({row["design_id"] for row in rows}) == 264
    assert not {row["design_id"] for row in rows} & {row["design_id"] for row in old}
    normalizer = Geometry22Normalizer()
    integers = [i for i, field in enumerate(normalizer.fields) if field["dtype"] == "int64"]
    assert len(integers) == 3
    for low, high in zip(rows[::2], rows[1::2], strict=True):
        assert low["center_split"] == high["center_split"] == assignment[low["center_design_id"]]
        assert (low["pair_sign"], high["pair_sign"]) == (-1, 1)
        assert all(low["normalized_design"][i] == high["normalized_design"][i] for i in integers)
        assert math.isclose(
            math.dist(low["normalized_design"], high["normalized_design"]),
            2 * low["requested_normalized_l2_radius"], rel_tol=1e-9, abs_tol=1e-10,
        )
    assert all(row["training_allowed"] is False for row in rows)


def test_labels_and_coverage_keep_qualification_distinct():
    manifest = verify_v3_local_acquisition_labels(ROOT, OUTPUT)
    labels = _rows(OUTPUT / "labels.jsonl")
    coverage = _rows(OUTPUT / "coverage.jsonl")
    assert manifest["row_count"] == len(labels) == 264
    assert len(coverage) == 66
    assert all(row["sample_count"] == 4 for row in coverage)
    assert sum(row["regression_candidate_count"] for row in coverage) == manifest["software_regression_candidate_count"]
    assert manifest["training_allowed"] is False
    assert manifest["formal_use_allowed"] is False
    assert all(row["software_regression_candidate"] <= row["solver_converged"] for row in labels)
    assert all(row["training_allowed"] is False for row in labels)


def test_rebuild_and_rehashed_split_tamper_are_rejected(tmp_path):
    rebuilt = tmp_path / "rebuilt"
    result = prepare_v3_local_acquisition(ROOT, rebuilt)
    committed = verify_v3_local_acquisition_coordinates(ROOT, OUTPUT)
    assert result["coordinate_sha256"] == committed["coordinate_sha256"]
    with pytest.raises(V3LocalAcquisitionError, match="Refusing to overwrite"):
        prepare_v3_local_acquisition(ROOT, rebuilt)

    altered = tmp_path / "altered"
    shutil.copytree(OUTPUT, altered)
    path = altered / "coordinates.jsonl"
    rows = _rows(path)
    rows[0]["center_split"] = "holdout" if rows[0]["center_split"] != "holdout" else "development"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    manifest_path = altered / "coordinate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["coordinate_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(V3LocalAcquisitionError, match="coordinate lock changed"):
        verify_v3_local_acquisition_coordinates(ROOT, altered)
