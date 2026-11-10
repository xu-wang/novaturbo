"""Tests for the restricted 33-root S3 v3 coordinate lock."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_stable_root_s3_prelabel_v3 import (
    PASS_STATUS,
    Geometry19StableRootS3V3Error,
    build_v3_directions,
    generate_s3_v3_prelabel_coordinates,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config/experiments/geometry19_v2_2_stable_root_s3_prelabel_v3_33root.yaml"
EVIDENCE = REPO_ROOT / "data/validation/v2_2_stable_root_forward_fresh_eval_v1"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_v3_scope_and_direction_pools_are_frozen_and_deterministic() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_protocol(config)
    assert len(config["scope"]["selected_root_ids"]) == 33
    assert len(config["scope"]["excluded_root_ids"]) == 5
    first = build_v3_directions(config)
    second = build_v3_directions(config)
    for key in first:
        assert np.array_equal(first[key], second[key])
    assert first["primary_directions"].shape == (48, 19)
    assert first["reserve_directions"].shape == (256, 19)
    assert np.allclose(np.linalg.norm(first["primary_directions"], axis=1), 1.0)
    assert np.allclose(np.linalg.norm(first["reserve_directions"], axis=1), 1.0)


def test_v3_protocol_rejects_scope_or_authorization_drift() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["scope"]["selected_root_ids"] = config["scope"]["selected_root_ids"][:-1]
    with pytest.raises(Geometry19StableRootS3V3Error, match="Selected root count"):
        validate_protocol(config)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["authorization"]["physics_labeling_executed_now"] = True
    with pytest.raises(Geometry19StableRootS3V3Error, match="Physics cannot run"):
        validate_protocol(config)


def test_frozen_v3_coordinate_pass_is_exact_independent_and_label_free() -> None:
    summary = json.loads(
        (EVIDENCE / "audits/S3_v3_33root_Prelabel_Coordinate_Summary.json").read_text(
            encoding="utf-8"
        )
    )
    ledger = _read_jsonl(
        EVIDENCE / "fresh_evaluation_v3_33root/S3_v3_Prelabel_Coordinates.jsonl"
    )
    assert summary["status"] == PASS_STATUS
    assert summary["selected_root_count"] == 33
    assert summary["excluded_root_count"] == 5
    assert summary["selected_pairs"] == 1584
    assert summary["final_prelabel_rows"] == len(ledger) == 3168
    assert set(summary["selected_pairs_per_root"].values()) == {48}
    assert len(summary["selected_pairs_per_root"]) == 33
    assert summary["minimum_selected_pairs_per_root"] == 48
    assert summary["physics_labeling_authorized"] is True
    assert summary["prior_dataset_or_S3_overlap"] is False
    assert summary["final_design_hashes_unique"] is True
    assert summary["physics_evaluations_executed"] is False
    assert summary["model_manifest_content_read"] is False
    assert summary["model_files_opened"] is False
    assert summary["confirmation_content_read"] is False
    assert summary["protocol_sha256"] == sha256_file(CONFIG)
    for key, value in summary["artifacts"].items():
        if not key.endswith("_sha256"):
            assert summary["artifacts"][f"{key}_sha256"] == sha256_file(REPO_ROOT / value)
    hashes = {row["design_hash"] for row in ledger}
    assert len(hashes) == len(ledger)
    prior = set()
    for relative in (
        "data/validation/v2_2_known_manifold_root_discovery_v1/development_dataset.jsonl",
        "data/validation/v2_2_stable_root_forward_fresh_eval_v1/fresh_evaluation/S3_Prelabel_Coordinates.jsonl",
        "data/validation/v2_2_stable_root_forward_fresh_eval_v1/fresh_evaluation_v2/S3_v2_Prelabel_Coordinates.jsonl",
    ):
        prior.update(row.get("design_hash") for row in _read_jsonl(REPO_ROOT / relative))
    assert not (hashes & prior)
    forbidden = {"performance", "physics", "prediction", "rho", "solver", "map_states"}
    assert all(not (forbidden & set(row)) for row in ledger)
    integer_names = {
        "compressor_blade_count",
        "combustor_injector_count",
        "turbine_blade_count",
    }
    assert all(
        isinstance(row["geometry19"][name], int)
        for row in ledger
        for name in integer_names
    )


def test_v3_generation_does_not_read_confirmation_or_model_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    protected = {
        (REPO_ROOT / config["inputs"][role]["path"]).resolve()
        for role in ("sealed_confirmation", "s2_model_manifest")
    }
    original_text = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.resolve() in protected:
            raise AssertionError("protected input content was parsed")
        return original_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    result = generate_s3_v3_prelabel_coordinates(REPO_ROOT, CONFIG, output_root=tmp_path)
    assert result["status"] == PASS_STATUS
    assert result["confirmation_content_read"] is False
    assert result["model_manifest_content_read"] is False
    assert result["physics_evaluations_executed"] is False
