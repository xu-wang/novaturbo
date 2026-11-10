"""Tests for the S3 v4.1 V2.2 physics-acquisition protocol lock."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_stable_root_s3_v4_1_physics_protocol import (
    LOCK_PASS,
    Geometry19StableRootS3V41PhysicsProtocolError,
    lock_s3_v4_1_physics_protocol,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config/experiments/geometry19_v2_2_stable_root_s3_v4_1_physics_acquisition_v1.yaml"
EVIDENCE = REPO_ROOT / "data/validation/v2_2_stable_root_forward_fresh_eval_v1"


def test_protocol_freezes_development_scope_and_acquisition_gates() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_protocol(config)
    assert config["claims"]["forbidden"][0] == "fresh_confirmation_accuracy"
    assert config["execution_order"][3] == "seal_complete_physics_ledger_before_any_model_access"
    config["physics_labeling"]["performance_adaptive_top_up"] = True
    with pytest.raises(Geometry19StableRootS3V41PhysicsProtocolError, match="Adaptive top-up"):
        validate_protocol(config)


def test_frozen_lock_has_exact_coordinates_and_no_execution() -> None:
    lock = json.loads((EVIDENCE / "audits/S3_v4_1_Physics_Protocol_Lock.json").read_text(encoding="utf-8"))
    assert lock["status"] == LOCK_PASS
    assert lock["coordinate_rows"] == 3168
    assert lock["root_count"] == 33
    assert lock["rows_per_root"] == 96
    assert lock["coordinate_source_rows"] == {
        "v4_1_boundary_completion": 384,
        "v4_medoid_covariance": 2784,
    }
    assert lock["future_shard_count"] == 33
    assert all(lock["future_artifacts_absent_at_lock"].values())
    assert lock["physics_evaluations_executed"] is False
    assert lock["model_manifest_content_read"] is False
    assert lock["model_predictions_executed"] is False
    assert lock["confirmation_content_read"] is False
    assert lock["physics_acquisition_authorized"] is True
    assert lock["model_scoring_authorized_now"] is False
    assert lock["inverse_design_authorized"] is False
    assert lock["protocol_sha256"] == sha256_file(CONFIG)
    assert lock["implementation_sha256"] == sha256_file(
        REPO_ROOT / lock["implementation_path"]
    )
    assert lock["report_sha256"] == sha256_file(REPO_ROOT / lock["report_path"])


def test_lock_rejects_any_coordinate_hash_change(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["inputs"]["coordinate_ledger"]["sha256"] = "0" * 64
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(Geometry19StableRootS3V41PhysicsProtocolError, match="coordinate_ledger"):
        lock_s3_v4_1_physics_protocol(REPO_ROOT, changed, output_root=tmp_path / "out")


def test_lock_never_parses_models_or_confirmation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    protected = {
        (REPO_ROOT / config["inputs"][role]["path"]).resolve()
        for role in ("s2_model_manifest", "sealed_confirmation")
    }
    original = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.resolve() in protected:
            raise AssertionError("protected content was parsed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    result = lock_s3_v4_1_physics_protocol(REPO_ROOT, CONFIG, output_root=tmp_path)
    assert result["physics_acquisition_authorized"] is True
    assert result["model_manifest_content_read"] is False
    assert result["confirmation_content_read"] is False
