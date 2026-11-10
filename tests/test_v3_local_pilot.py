"""L1 gates for frozen paired coordinates and independently generated V3 labels."""

import hashlib
import json
import math
from pathlib import Path
import shutil

import pytest

from src.ai.v3_local_pilot import (
    V3LocalPilotError,
    prepare_v3_local_pilot,
    verify_v3_local_pilot_coordinates,
    verify_v3_local_pilot_labels,
)
from src.physics.geometry22_data import Geometry22Normalizer
from src.physics.geometry22_design import Geometry22Design


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/validation/v3_22d_local_pilot_v1"


def _rows(name):
    return [json.loads(line) for line in (OUTPUT / name).read_text().splitlines()]


def test_local_coordinates_were_frozen_before_labels_and_keep_integer_state():
    manifest = verify_v3_local_pilot_coordinates(ROOT, OUTPUT)
    coordinates = _rows("coordinates.jsonl")
    assert manifest["coordinate_count"] == len(coordinates) == 264
    assert manifest["split_counts"] == {"development": 132, "validation": 66, "holdout": 66}
    assert manifest["coordinate_sha256"] == hashlib.sha256((OUTPUT / "coordinates.jsonl").read_bytes()).hexdigest()
    assert manifest["training_allowed"] is False
    assert len({row["design_id"] for row in coordinates}) == 264
    normalizer = Geometry22Normalizer()
    integers = [index for index, field in enumerate(normalizer.fields) if field["dtype"] == "int64"]
    assert len(integers) == 3
    for low, high in zip(coordinates[::2], coordinates[1::2], strict=True):
        a = Geometry22Design.from_mapping(low["design"])
        b = Geometry22Design.from_mapping(high["design"])
        assert low["center_design_id"] == high["center_design_id"]
        assert low["pair_index"] == high["pair_index"]
        assert (low["pair_sign"], high["pair_sign"]) == (-1, 1)
        assert low["split"] == high["split"]
        assert all(a.coordinates[index] == b.coordinates[index] for index in integers)
        assert math.isclose(
            math.dist(low["normalized_design"], high["normalized_design"]),
            2 * low["requested_normalized_l2_radius"], rel_tol=1e-9, abs_tol=1e-10,
        )


def test_local_labels_match_prelabel_identity_and_report_scope():
    manifest = verify_v3_local_pilot_labels(ROOT, OUTPUT)
    coordinates = _rows("coordinates.jsonl")
    labels = _rows("labels.jsonl")
    assert manifest["row_count"] == len(labels) == 264
    assert manifest["solver_converged_count"] == 261
    assert manifest["development_domain_retained_count"] == 251
    assert manifest["engine_feasible_count"] == 0
    assert manifest["source_v2_2_labels_reused"] is False
    assert manifest["training_allowed"] is False
    assert len({row["center_design_id"] for row in labels if row["development_domain_retained"]}) == 33
    assert all(a["design_id"] == b["design_id"] and a["split"] == b["split"]
               for a, b in zip(coordinates, labels, strict=True))


def test_prelabel_rebuild_and_postlabel_tamper_are_rejected(tmp_path):
    rebuilt_root = tmp_path / "rebuilt"
    rebuilt = prepare_v3_local_pilot(ROOT, rebuilt_root)
    committed = verify_v3_local_pilot_coordinates(ROOT, OUTPUT)
    assert rebuilt["coordinate_sha256"] == committed["coordinate_sha256"]
    with pytest.raises(V3LocalPilotError, match="Refusing to overwrite"):
        prepare_v3_local_pilot(ROOT, rebuilt_root)

    altered = tmp_path / "altered"
    shutil.copytree(OUTPUT, altered)
    labels_path = altered / "labels.jsonl"
    labels = [json.loads(line) for line in labels_path.read_text().splitlines()]
    labels[0]["design_id"] = "0" * 64
    labels_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in labels))
    manifest_path = altered / "complete_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["labels_sha256"] = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(V3LocalPilotError, match="label identity changed"):
        verify_v3_local_pilot_labels(ROOT, altered)
