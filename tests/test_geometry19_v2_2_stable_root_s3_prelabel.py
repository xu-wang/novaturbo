"""Tests for stable-root S3 prelabel coordinate generation and isolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_stable_root_s3_prelabel import (
    FAIL_STATUS,
    Geometry19StableRootS3PrelabelError,
    build_direction_matrix,
    generate_s3_prelabel_coordinates,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config/experiments/geometry19_v2_2_stable_root_s3_prelabel_v1.yaml"
EVIDENCE = REPO_ROOT / "data/validation/v2_2_stable_root_forward_fresh_eval_v1"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_direction_matrix_is_deterministic_normalized_and_shell_balanced() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    first = build_direction_matrix(config)
    second = build_direction_matrix(config)
    assert np.array_equal(first["directions"], second["directions"])
    assert first["directions"].shape == (48, 19)
    assert np.allclose(np.linalg.norm(first["directions"], axis=1), 1.0)
    assert np.sum(first["direction_family"] == "schema_axis") == 19
    assert np.sum(first["direction_family"] == "sobol_mixed") == 29
    assert np.sum(first["radius_fraction"] == 0.5) == 24
    assert np.sum(first["radius_fraction"] == 0.9) == 24


def test_protocol_forbids_model_prediction_during_prelabel_stage() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_protocol(config)
    config["authorization"]["model_prediction_authorized"] = True
    with pytest.raises(Geometry19StableRootS3PrelabelError, match="cannot be authorized"):
        validate_protocol(config)


def test_frozen_prelabel_failure_is_complete_hashed_and_label_free() -> None:
    summary = json.loads(
        (EVIDENCE / "audits/S3_Prelabel_Coordinate_Summary.json").read_text(
            encoding="utf-8"
        )
    )
    ledger = _read_jsonl(
        EVIDENCE / "fresh_evaluation/S3_Prelabel_Coordinates.jsonl"
    )
    assert summary["status"] == FAIL_STATUS
    assert summary["attempt_rows"] == len(ledger) == 3648
    assert set(summary["attempts_per_root"].values()) == {96}
    assert len(summary["attempts_per_root"]) == 38
    assert summary["coordinate_eligible_rows"] == 3362
    assert summary["coordinate_rejected_rows"] == 286
    assert summary["minimum_coordinate_eligible_per_root_observed"] == 50
    assert summary["rejection_reasons"] == {
        "schema_bounds": 255,
        "within_S3_design_hash_duplicate": 31,
    }
    assert summary["physics_labeling_authorized"] is False
    assert summary["physics_evaluations_executed"] is False
    assert summary["model_files_opened"] is False
    assert summary["model_predictions_executed"] is False
    assert summary["performance_labels_written"] is False
    assert summary["confirmation_content_read"] is False
    assert summary["protocol_sha256"] == sha256_file(CONFIG)
    for key, value in summary["artifacts"].items():
        if key.endswith("_sha256"):
            continue
        assert summary["artifacts"][f"{key}_sha256"] == sha256_file(REPO_ROOT / value)
    forbidden = {"performance", "physics", "prediction", "rho", "solver", "map_states"}
    assert all(not (forbidden & set(row)) for row in ledger)
    assert all(row["label_status"] == "NOT_EVALUATED_PRELABEL_ONLY" for row in ledger)


def test_prelabel_generation_never_text_parses_confirmation_or_opens_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    confirmation = (REPO_ROOT / config["inputs"]["sealed_confirmation"]["path"]).resolve()
    model_paths = {
        (REPO_ROOT / model["model_path"]).resolve()
        for model in json.loads(
            (EVIDENCE / "models/S2_Model_Manifest.json").read_text(encoding="utf-8")
        )["models"]
    }
    original_text = Path.read_text
    original_open = Path.open

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.resolve() == confirmation:
            raise AssertionError("sealed confirmation content was parsed")
        return original_text(path, *args, **kwargs)

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path.resolve() in model_paths:
            raise AssertionError("S2 model file was opened during prelabel generation")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "open", guarded_open)
    result = generate_s3_prelabel_coordinates(REPO_ROOT, CONFIG, output_root=tmp_path)
    assert result["status"] == FAIL_STATUS
    assert result["confirmation_content_read"] is False
    assert result["model_files_opened"] is False
    assert result["physics_evaluations_executed"] is False
