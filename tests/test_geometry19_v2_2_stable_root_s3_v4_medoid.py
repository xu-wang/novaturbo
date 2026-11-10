"""Tests for S3 v4 medoid/covariance coordinate generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_stable_root_s3_v4_medoid import (
    FAIL_STATUS,
    Geometry19StableRootS3V4Error,
    build_root_direction_model,
    generate_s3_v4_medoid_coordinates,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config/experiments/geometry19_v2_2_stable_root_s3_v4_medoid_covariance_prelabel_v1.yaml"
EVIDENCE = REPO_ROOT / "data/validation/v2_2_stable_root_forward_fresh_eval_v1"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_v4_protocol_freezes_medoid_covariance_and_small_shells() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_protocol(config)
    assert config["anchor"]["arithmetic_mean_center_forbidden"] is True
    assert config["local_direction_model"]["performance_values_read"] is False
    assert config["scaling"]["shell_fractions"] == [0.10, 0.20]
    config["scaling"]["shell_fractions"] = [0.20, 0.30]
    with pytest.raises(Geometry19StableRootS3V4Error, match="Radius shells"):
        validate_protocol(config)


def test_v4_root_direction_model_is_deterministic_and_covariance_shaped() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    schema = yaml.safe_load(
        (REPO_ROOT / config["inputs"]["design_schema"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    variables = sorted(schema["design_variables"], key=lambda row: row["index"])
    names = [row["name"] for row in variables]
    lower = np.asarray([row["bounds"][0] for row in variables], dtype=float)
    upper = np.asarray([row["bounds"][1] for row in variables], dtype=float)
    development = _read_jsonl(
        REPO_ROOT / config["inputs"]["development_dataset"]["path"]
    )
    anchors = _read_jsonl(EVIDENCE / "audits/S3_v4_Medoid_Anchor_Registry.jsonl")
    root_id = anchors[0]["root_seed_id"]
    root_rows = [row for row in development if row["root_seed_id"] == root_id]
    anchor = np.asarray(anchors[0]["normalized_medoid"])
    first = build_root_direction_model(
        root_id, anchor, root_rows, names, lower, upper - lower, config
    )
    second = build_root_direction_model(
        root_id, anchor, root_rows, names, lower, upper - lower, config
    )
    for key in first:
        assert np.array_equal(first[key], second[key])
    assert first["primary_directions"].shape == (48, 19)
    assert first["reserve_directions"].shape == (256, 19)
    assert np.allclose(np.linalg.norm(first["primary_directions"], axis=1), 1.0)
    assert np.all(first["eigenvalues"] > 0.0)


def test_frozen_v4_failure_preserves_valid_medoid_evidence_without_labels() -> None:
    summary = json.loads(
        (EVIDENCE / "audits/S3_v4_Medoid_Covariance_Prelabel_Summary.json").read_text(
            encoding="utf-8"
        )
    )
    anchors = _read_jsonl(EVIDENCE / "audits/S3_v4_Medoid_Anchor_Registry.jsonl")
    ledger = _read_jsonl(
        EVIDENCE / "fresh_evaluation_v4_medoid/S3_v4_Prelabel_Coordinates.jsonl"
    )
    assert summary["status"] == FAIL_STATUS
    assert summary["eligible_medoid_anchors"] == len(anchors) == 33
    assert all(row["existing_V2_2_training_candidate"] for row in anchors)
    assert summary["selected_pairs"] == 1392
    assert summary["final_prelabel_rows"] == len(ledger) == 2784
    assert len(summary["selected_pairs_per_root"]) == 29
    assert summary["screening_rejection_reasons"]["two_sided_safe_step_ratio_below_minimum"] == 1216
    assert summary["future_physics_protocol_authorized"] is False
    assert summary["physics_evaluations_executed"] is False
    assert summary["model_manifest_content_read"] is False
    assert summary["confirmation_content_read"] is False
    assert summary["protocol_sha256"] == sha256_file(CONFIG)
    for key, value in summary["artifacts"].items():
        if not key.endswith("_sha256"):
            assert summary["artifacts"][f"{key}_sha256"] == sha256_file(REPO_ROOT / value)
    forbidden = {"performance", "physics", "prediction", "rho", "solver", "map_states"}
    assert all(not (forbidden & set(row)) for row in ledger)


def test_v4_does_not_parse_confirmation_or_model_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    protected = {
        (REPO_ROOT / config["inputs"][role]["path"]).resolve()
        for role in ("sealed_confirmation", "s2_model_manifest")
    }
    original = Path.read_text

    def guarded(path: Path, *args: object, **kwargs: object) -> str:
        if path.resolve() in protected:
            raise AssertionError("protected content was parsed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    result = generate_s3_v4_medoid_coordinates(
        REPO_ROOT, CONFIG, output_root=tmp_path
    )
    assert result["status"] == FAIL_STATUS
    assert result["model_manifest_content_read"] is False
    assert result["confirmation_content_read"] is False
    assert result["physics_evaluations_executed"] is False
