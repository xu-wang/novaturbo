"""Tests for the boundary-aware stable-root S3 v2 coordinate lock."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_stable_root_s3_prelabel_v2 import (
    FAIL_STATUS,
    Geometry19StableRootS3V2Error,
    build_v2_directions,
    generate_s3_v2_prelabel_coordinates,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config/experiments/geometry19_v2_2_stable_root_s3_prelabel_v2.yaml"
EVIDENCE = REPO_ROOT / "data/validation/v2_2_stable_root_forward_fresh_eval_v1"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_v2_direction_pools_are_deterministic_normalized_and_shell_balanced() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    first = build_v2_directions(config)
    second = build_v2_directions(config)
    for key in first:
        assert np.array_equal(first[key], second[key])
    assert first["primary_directions"].shape == (48, 19)
    assert first["reserve_directions"].shape == (64, 19)
    assert np.allclose(np.linalg.norm(first["primary_directions"], axis=1), 1.0)
    assert np.allclose(np.linalg.norm(first["reserve_directions"], axis=1), 1.0)
    assert np.sum(first["primary_shell"] == 0.5) == 24
    assert np.sum(first["primary_shell"] == 0.9) == 24
    assert np.sum(first["reserve_shell"] == 0.5) == 32
    assert np.sum(first["reserve_shell"] == 0.9) == 32


def test_v2_protocol_forbids_physics_and_prediction_during_prelabel() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_protocol(config)
    config["authorization"]["physics_labeling_executed_now"] = True
    with pytest.raises(Geometry19StableRootS3V2Error, match="Physics cannot run"):
        validate_protocol(config)
    config["authorization"]["physics_labeling_executed_now"] = False
    config["authorization"]["model_prediction_authorized"] = True
    with pytest.raises(Geometry19StableRootS3V2Error, match="Prediction cannot be authorized"):
        validate_protocol(config)


def test_frozen_v2_failure_is_hashed_unique_and_label_free() -> None:
    summary = json.loads(
        (EVIDENCE / "audits/S3_v2_Prelabel_Coordinate_Summary.json").read_text(
            encoding="utf-8"
        )
    )
    ledger = _read_jsonl(
        EVIDENCE / "fresh_evaluation_v2/S3_v2_Prelabel_Coordinates.jsonl"
    )
    assert summary["status"] == FAIL_STATUS
    assert summary["selected_pairs"] == 1734
    assert summary["final_prelabel_rows"] == len(ledger) == 3468
    assert summary["minimum_selected_pairs_per_root"] == 17
    assert summary["reserve_pairs_selected"] == 62
    assert summary["screening_pairs_considered"] == 2177
    assert summary["screening_pairs_rejected"] == 435
    assert summary["screening_rejection_reasons"] == {
        "integer_axis_two_sided_step_illegal": 12,
        "pair_design_hashes_identical": 186,
        "reserve_directions_exhausted": 8,
        "schema_bounds_after_scaling": 12,
        "two_sided_safe_step_ratio_below_minimum": 423,
    }
    assert summary["final_design_hashes_unique"] is True
    assert summary["final_development_overlap"] is False
    assert summary["all_final_coordinates_inside_schema"] is True
    assert summary["physics_labeling_authorized"] is False
    assert summary["physics_evaluations_executed"] is False
    assert summary["model_manifest_metadata_read"] is False
    assert summary["model_files_opened"] is False
    assert summary["model_predictions_executed"] is False
    assert summary["performance_labels_written"] is False
    assert summary["confirmation_content_read"] is False
    assert summary["input_hash_audit"]["s2_model_manifest"]["content_parsed"] is False
    assert summary["protocol_sha256"] == sha256_file(CONFIG)
    for key, value in summary["artifacts"].items():
        if not key.endswith("_sha256"):
            assert summary["artifacts"][f"{key}_sha256"] == sha256_file(
                REPO_ROOT / value
            )
    v1_ledger = EVIDENCE / "fresh_evaluation/S3_Prelabel_Coordinates.jsonl"
    assert sha256_file(v1_ledger) == (
        "7b642cd65a0fa44b55d09d6dad05d7a75306aefcd68e4054f3a51cb98f523956"
    )
    forbidden = {"performance", "physics", "prediction", "rho", "solver", "map_states"}
    assert len({row["design_hash"] for row in ledger}) == len(ledger)
    assert all(not (forbidden & set(row)) for row in ledger)
    assert all(row["label_status"] == "NOT_EVALUATED_PRELABEL_ONLY" for row in ledger)


def test_v2_generation_does_not_parse_confirmation_manifest_or_open_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    confirmation = (REPO_ROOT / config["inputs"]["sealed_confirmation"]["path"]).resolve()
    manifest = (REPO_ROOT / config["inputs"]["s2_model_manifest"]["path"]).resolve()
    model_paths = {
        (REPO_ROOT / model["model_path"]).resolve()
        for model in json.loads(manifest.read_text(encoding="utf-8"))["models"]
    }
    original_text = Path.read_text
    original_open = Path.open

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.resolve() in {confirmation, manifest}:
            raise AssertionError("sealed input content was parsed")
        return original_text(path, *args, **kwargs)

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path.resolve() in model_paths:
            raise AssertionError("S2 model file was opened during S3 v2")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "open", guarded_open)
    result = generate_s3_v2_prelabel_coordinates(REPO_ROOT, CONFIG, output_root=tmp_path)
    assert result["status"] == FAIL_STATUS
    assert result["confirmation_content_read"] is False
    assert result["model_manifest_metadata_read"] is False
    assert result["model_files_opened"] is False
    assert result["physics_evaluations_executed"] is False
