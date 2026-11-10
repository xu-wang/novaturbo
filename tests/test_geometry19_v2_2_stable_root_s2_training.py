"""Tests for stable-root S2 Ridge selection and model locking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_stable_root_s2_training import (
    MODEL_PASS,
    Geometry19StableRootS2Error,
    _root_balanced_weights,
    run_s2_training,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT
    / "config/experiments/geometry19_v2_2_stable_root_s2_training_v1.yaml"
)
EVIDENCE = REPO_ROOT / "data/validation/v2_2_stable_root_forward_fresh_eval_v1"


def test_root_balanced_weights_have_equal_root_totals_and_mean_one() -> None:
    root_ids = ["a", "a", "b", "b", "b", "b"]
    weights = _root_balanced_weights(root_ids)
    assert np.mean(weights) == pytest.approx(1.0)
    assert np.sum(weights[:2]) == pytest.approx(np.sum(weights[2:]))


def test_s2_protocol_forbids_fresh_evaluation_visibility() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_protocol(config)
    config["inner_selection"]["fresh_evaluation_visible"] = True
    with pytest.raises(Geometry19StableRootS2Error, match="leakage"):
        validate_protocol(config)


def test_frozen_s2_manifest_has_all_models_reload_and_hashes() -> None:
    manifest = json.loads(
        (EVIDENCE / "models/S2_Model_Manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == MODEL_PASS
    assert manifest["all_five_models_locked"] is True
    assert manifest["exact_reload_pass"] is True
    assert manifest["stable_roots"] == 38
    assert manifest["stable_development_rows"] == 2935
    assert manifest["all_root_comparator_rows"] == 4529
    assert manifest["fresh_evaluation_visible_during_selection"] is False
    assert manifest["fresh_coordinates_read_or_generated"] is False
    assert manifest["fresh_physics_labels_read_or_generated"] is False
    assert manifest["confirmation_content_read"] is False
    assert manifest["inverse_design_authorized"] is False
    assert manifest["original_local_chart_p4_authorized"] is False
    assert manifest["s3_coordinate_generation_authorized"] is False
    assert manifest["protocol_sha256"] == sha256_file(CONFIG)
    assert manifest["inner_selection_sha256"] == sha256_file(
        REPO_ROOT / manifest["inner_selection_path"]
    )
    assert manifest["training_report_sha256"] == sha256_file(
        REPO_ROOT / manifest["training_report_path"]
    )
    assert len(manifest["models"]) == 5
    for model in manifest["models"]:
        path = REPO_ROOT / model["model_path"]
        assert model["model_sha256"] == sha256_file(path)
        assert model["exact_reload_equal"] is True
        assert model["exact_prediction_reload_equal"] is True
        assert model["all_coefficients_finite"] is True
        with np.load(path) as bundle:
            assert str(bundle["arm_id"].item()) == model["arm_id"]
            assert float(bundle["alpha"].item()) == model["selected_alpha"]


def test_s2_training_never_text_parses_confirmation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    confirmation = (REPO_ROOT / config["inputs"]["sealed_confirmation"]["path"]).resolve()
    original = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.resolve() == confirmation:
            raise AssertionError("sealed confirmation content was parsed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    result = run_s2_training(REPO_ROOT, CONFIG, output_root=tmp_path)
    assert result["status"] == MODEL_PASS
    assert result["confirmation_content_read"] is False
    assert result["fresh_coordinates_read_or_generated"] is False
    assert result["fresh_physics_labels_read_or_generated"] is False
