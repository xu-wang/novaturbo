"""Tests for the S3 v4.1 one-shot five-arm scoring protocol lock."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_stable_root_s3_v4_1_scoring_protocol import (
    LOCK_PASS,
    Geometry19StableRootS3V41ScoringProtocolError,
    lock_s3_v4_1_model_scoring_protocol,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config/experiments/geometry19_v2_2_stable_root_s3_v4_1_model_scoring_v1.yaml"
EVIDENCE = REPO_ROOT / "data/validation/v2_2_stable_root_forward_fresh_eval_v1"


def test_protocol_freezes_common_rows_arms_metrics_and_claims() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_protocol(config)
    assert config["evaluation_population"]["exact_rows"] == 3039
    assert config["execution_order"][4] == "write_and_hash_complete_prediction_ledger_before_metrics"
    assert config["claims"]["forbidden"][0] == "fresh_confirmation_accuracy"
    config["scoring"]["prediction_adaptive_changes"] = "allowed"
    with pytest.raises(Geometry19StableRootS3V41ScoringProtocolError, match="Adaptive"):
        validate_protocol(config)


def test_frozen_lock_has_exact_rows_models_and_zero_predictions() -> None:
    lock = json.loads((EVIDENCE / "audits/S3_v4_1_Model_Scoring_Protocol_Lock.json").read_text(encoding="utf-8"))
    assert lock["status"] == LOCK_PASS
    assert lock["eligible_rows"] == 3039
    assert lock["eligible_roots"] == 33
    assert lock["minimum_rows_per_root"] == 67
    assert len(lock["model_arms"]) == 5
    assert all(item["match"] and item["model_arrays_loaded"] is False for item in lock["model_file_audit"])
    assert lock["model_manifest_content_read"] is True
    assert lock["model_files_hash_checked"] is True
    assert lock["model_arrays_loaded"] is False
    assert lock["model_predictions_executed"] is False
    assert lock["confirmation_content_read"] is False
    assert lock["model_scoring_authorized"] is True
    assert lock["inverse_design_authorized"] is False
    assert all(lock["future_artifacts_absent_at_lock"].values())
    assert lock["protocol_sha256"] == sha256_file(CONFIG)
    assert lock["implementation_sha256"] == sha256_file(REPO_ROOT / lock["implementation_path"])
    assert lock["artifacts"]["eligible_row_registry_sha256"] == sha256_file(REPO_ROOT / lock["artifacts"]["eligible_row_registry"])


def test_lock_never_loads_models_or_parses_confirmation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    confirmation = (REPO_ROOT / config["inputs"]["sealed_confirmation"]["path"]).resolve()
    original = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.resolve() == confirmation:
            raise AssertionError("confirmation content was parsed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    result = lock_s3_v4_1_model_scoring_protocol(REPO_ROOT, CONFIG, output_root=tmp_path)
    assert result["model_arrays_loaded"] is False
    assert result["model_predictions_executed"] is False
    assert result["confirmation_content_read"] is False
