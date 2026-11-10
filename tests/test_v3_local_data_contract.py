"""L2 center grouping and source-identity gates."""

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from src.ai.v3_local_data_contract import (
    V3LocalDataContractError,
    freeze_v3_local_data_contract,
    verify_v3_local_data_contract,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/validation/v3_22d_local_data_contract_v1"


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_future_assignment_is_center_disjoint_and_source_complete():
    manifest = verify_v3_local_data_contract(ROOT, OUTPUT)
    registry = _rows(OUTPUT / "center_split_registry.jsonl")
    centers = _rows(ROOT / "data/validation/v3_22d_local_center_audit_v1/centers.jsonl")
    pilot = _rows(ROOT / "data/validation/v3_22d_local_pilot_v1/coordinates.jsonl")
    assert manifest["center_count"] == len(registry) == 33
    assert manifest["split_counts"] == {"development": 23, "validation": 5, "holdout": 5}
    assert manifest["stratum_split_counts"] == {
        "1": {"development": 17, "validation": 3, "holdout": 4},
        "2": {"development": 6, "validation": 2, "holdout": 1},
    }
    assert {row["center_design_id"] for row in registry} == {row["design_id"] for row in centers}
    assert len({row["center_design_id"] for row in registry}) == 33
    assert len({row["center_design_id"] for row in pilot if row["split"] == "development"}) == 33
    assert len({row["center_design_id"] for row in pilot if row["split"] == "validation"}) == 33
    assert len({row["center_design_id"] for row in pilot if row["split"] == "holdout"}) == 33
    assert manifest["future_labels_present"] is False
    assert manifest["source_l0_l1_training_allowed"] is False
    assert manifest["formal_use_allowed"] is False


def test_rebuild_is_deterministic_and_rejects_rehashed_assignment_tamper(tmp_path):
    rebuilt_root = tmp_path / "rebuilt"
    rebuilt = freeze_v3_local_data_contract(ROOT, rebuilt_root)
    committed = verify_v3_local_data_contract(ROOT, OUTPUT)
    assert rebuilt["registry_sha256"] == committed["registry_sha256"]
    with pytest.raises(V3LocalDataContractError, match="Refusing to overwrite"):
        freeze_v3_local_data_contract(ROOT, rebuilt_root)

    altered = tmp_path / "altered"
    shutil.copytree(OUTPUT, altered)
    registry_path = altered / "center_split_registry.jsonl"
    registry = _rows(registry_path)
    registry[0]["center_split"] = "holdout" if registry[0]["center_split"] != "holdout" else "development"
    registry_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in registry))
    manifest_path = altered / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["registry_sha256"] = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(V3LocalDataContractError, match="registry changed"):
        verify_v3_local_data_contract(ROOT, altered)
