"""Development-only checks for V3 relabeling of frozen V2.2 medoids."""

import hashlib
import json
from pathlib import Path

import pytest

from src.ai.v3_local_center_audit import (
    V3LocalCenterAuditError,
    build_v3_local_center_audit,
    verify_v3_local_center_audit,
)
from src.physics.geometry22_design import Geometry22Design


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/validation/v3_22d_local_center_audit_v1"


def test_v3_local_center_audit_keeps_22d_identity_and_source_role():
    manifest = verify_v3_local_center_audit(ROOT, OUTPUT)
    rows = [json.loads(line) for line in (OUTPUT / "centers.jsonl").read_text().splitlines()]
    assert len(rows) == manifest["row_count"] == 33
    assert manifest["solver_converged_count"] == 33
    assert manifest["candidate_count"] == 33
    assert manifest["engine_feasible_count"] == 0
    assert manifest["source_v2_2_labels_reused"] is False
    assert manifest["training_allowed"] is False
    assert len({row["design_id"] for row in rows}) == 33
    for row in rows:
        design = Geometry22Design.from_mapping(row["design"])
        assert len(design.coordinates) == 22
        assert design.design_id() == row["design_id"]
        assert row["source_role"] == "coordinate_seed_only_no_v2_2_labels_reused"
        assert row["legacy_partition_development_diagnostic"]["physics_domain_valid"] is True
        assert row["candidate_for_local_pilot"] is True
        assert row["engine_feasible"] is False
        assert row["training_allowed"] is False


def test_v3_local_center_audit_rebuild_is_deterministic_and_rejects_relabeling(tmp_path):
    output = tmp_path / "audit"
    rebuilt = build_v3_local_center_audit(ROOT, output)
    original = verify_v3_local_center_audit(ROOT, OUTPUT)
    assert rebuilt["files"] == original["files"]
    with pytest.raises(V3LocalCenterAuditError, match="Refusing to overwrite"):
        build_v3_local_center_audit(ROOT, output)

    cases = output / "centers.jsonl"
    rows = [json.loads(line) for line in cases.read_text().splitlines()]
    rows[0]["source_v2_2_root_seed_id"] = "0" * 64
    cases.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    manifest_path = output / "complete_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["centers.jsonl"] = hashlib.sha256(cases.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(V3LocalCenterAuditError, match="row identity changed"):
        verify_v3_local_center_audit(ROOT, output)
