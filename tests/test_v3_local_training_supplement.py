"""Development-only supplement and sealed candidate-split gates."""

import json
from pathlib import Path

import pytest

from src.ai.v3_local_candidate_dataset import verify_v3_local_candidate_dataset
from src.ai.v3_local_training_supplement import (
    V3LocalTrainingSupplementError,
    prepare_v3_training_supplement,
    verify_v3_training_supplement_labels,
)


ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT = ROOT / "data/validation/v3_22d_local_training_data_dev_supplement_v1"
DATASET = ROOT / "data/validation/v3_22d_local_candidate_dataset_v1"


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_supplement_is_separate_and_combined_coverage_passes():
    manifest = verify_v3_training_supplement_labels(ROOT, SUPPLEMENT)
    rows = _rows(SUPPLEMENT / "coordinates.jsonl")
    base = _rows(ROOT / "data/validation/v3_22d_local_training_data_v1/coordinates.jsonl")
    assert manifest["row_count"] == len(rows) == 576
    assert manifest["combined_row_count"] == 17472
    assert manifest["combined_data_coverage_gate_pass"] is True
    assert manifest["combined_band_gate_failures"] == []
    assert manifest["combined_center_gate_failures"] == []
    assert manifest["holdout_rows_added"] == manifest["validation_rows_added"] == 0
    assert manifest["training_allowed"] is False
    assert all(row["center_split"] == "development" for row in rows)
    assert not {row["design_id"] for row in rows} & {row["design_id"] for row in base}
    combined = _rows(SUPPLEMENT / "combined_coverage.jsonl")
    assert min(row["software_regression_candidate_count"] for row in combined) >= 80
    assert min(sum(row["software_regression_candidate_count"] for row in combined
                   if row["center_design_id"] == center)
               for center in {row["center_design_id"] for row in combined}) >= 384


def test_supplement_prelabel_rebuild_is_deterministic(tmp_path):
    rebuilt = tmp_path / "rebuilt"
    result = prepare_v3_training_supplement(ROOT, rebuilt)
    committed = json.loads((SUPPLEMENT / "coordinate_manifest.json").read_text())
    assert result["coordinate_sha256"] == committed["coordinate_sha256"]
    with pytest.raises(V3LocalTrainingSupplementError, match="Refusing to overwrite"):
        prepare_v3_training_supplement(ROOT, rebuilt)


def test_candidate_files_are_center_disjoint_and_remain_development_only():
    manifest = verify_v3_local_candidate_dataset(ROOT, DATASET)
    assert manifest["candidate_count"] == 16168
    assert manifest["split_counts"] == {"development": 11289, "validation": 2394, "holdout": 2485}
    assert manifest["center_counts"] == {"development": 23, "validation": 5, "holdout": 5}
    assert manifest["combined_data_coverage_gate_pass"] is True
    assert manifest["training_allowed"] is False
    groups = {split: _rows(DATASET / f"{split}_candidates.jsonl")
              for split in ("development", "validation", "holdout")}
    assert len({row["design_id"] for rows in groups.values() for row in rows}) == 16168
    centers = [{row["center_design_id"] for row in rows} for rows in groups.values()]
    assert all(not centers[i] & centers[j] for i in range(3) for j in range(i + 1, 3))
    assert all(row["source"] == "base" for split in ("validation", "holdout") for row in groups[split])
