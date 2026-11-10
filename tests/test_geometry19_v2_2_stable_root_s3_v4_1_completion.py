"""Tests for S3 v4.1 discrete-boundary tangent completion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_stable_root_s3_v4_1_completion import (
    PASS_STATUS,
    Geometry19StableRootS3V41Error,
    complete_s3_v4_boundary_coordinates,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config/experiments/geometry19_v2_2_stable_root_s3_v4_1_boundary_completion_v1.yaml"
EVIDENCE = REPO_ROOT / "data/validation/v2_2_stable_root_forward_fresh_eval_v1"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_v4_1_protocol_freezes_four_boundary_roots_and_bilateral_projection() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_protocol(config)
    assert len(config["completion_scope"]["missing_root_ids"]) == 4
    assert config["bilateral_tangent_subspace"]["one_sided_variation_allowed"] is False
    config["bilateral_tangent_subspace"]["one_sided_variation_allowed"] = True
    with pytest.raises(Geometry19StableRootS3V41Error, match="One-sided"):
        validate_protocol(config)


def test_frozen_v4_1_pass_is_exact_unique_and_label_free() -> None:
    summary = json.loads(
        (EVIDENCE / "audits/S3_v4_1_Boundary_Completion_Summary.json").read_text(
            encoding="utf-8"
        )
    )
    completion = _read_jsonl(
        EVIDENCE
        / "fresh_evaluation_v4_1_medoid/S3_v4_1_Boundary_Completion_Coordinates.jsonl"
    )
    combined = _read_jsonl(
        EVIDENCE
        / "fresh_evaluation_v4_1_medoid/S3_v4_1_Combined_Prelabel_Coordinates.jsonl"
    )
    assert summary["status"] == PASS_STATUS
    assert summary["carried_forward_roots"] == 29
    assert summary["completed_boundary_roots"] == 4
    assert summary["completion_rows"] == len(completion) == 384
    assert summary["combined_rows"] == len(combined) == 3168
    assert len(summary["combined_rows_per_root"]) == 33
    assert set(summary["combined_rows_per_root"].values()) == {96}
    assert len({row["design_hash"] for row in combined}) == 3168
    assert summary["fixed_boundary_axes_preserved"] is True
    assert summary["future_physics_protocol_authorized"] is True
    assert summary["physics_evaluations_executed"] is False
    assert summary["model_files_opened"] is False
    assert summary["confirmation_content_read"] is False
    assert summary["protocol_sha256"] == sha256_file(CONFIG)
    forbidden = {"performance", "physics", "prediction", "rho", "solver", "map_states"}
    assert all(not (forbidden & set(row)) for row in combined)


def test_completion_holds_each_discrete_boundary_axis_at_medoid() -> None:
    summary = json.loads(
        (EVIDENCE / "audits/S3_v4_1_Boundary_Completion_Summary.json").read_text(
            encoding="utf-8"
        )
    )
    anchors = {
        row["root_seed_id"]: row
        for row in _read_jsonl(EVIDENCE / "audits/S3_v4_Medoid_Anchor_Registry.jsonl")
    }
    completion = _read_jsonl(
        EVIDENCE
        / "fresh_evaluation_v4_1_medoid/S3_v4_1_Boundary_Completion_Coordinates.jsonl"
    )
    for row in completion:
        root_id = row["root_seed_id"]
        assert row["fixed_boundary_axes"] == summary["fixed_boundary_axes_per_root"][root_id]
        for name in row["fixed_boundary_axes"]:
            assert row["geometry19"][name] == anchors[root_id]["medoid_geometry19"][name]


def test_v4_1_does_not_parse_confirmation_or_model_manifest(
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
    result = complete_s3_v4_boundary_coordinates(
        REPO_ROOT, CONFIG, output_root=tmp_path
    )
    assert result["status"] == PASS_STATUS
    assert result["model_manifest_content_read"] is False
    assert result["confirmation_content_read"] is False
    assert result["physics_evaluations_executed"] is False
