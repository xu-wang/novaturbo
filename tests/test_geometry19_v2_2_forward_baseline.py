"""Tests for the V2.2 performance-only forward baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.ai.geometry19_v2_2_forward_baseline import (
    evaluate_predictions,
    load_v2_2_splits,
    select_seed,
)
from src.ai.geometry19_training import PERFORMANCE_NAMES
from src.physics import Geometry19


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config/experiments/geometry19_v2_2_forward_baseline_v1.yaml"


def test_committed_v2_2_splits_pass_training_leakage_audit() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    domain = yaml.safe_load((REPO_ROOT / config["target_domain"]).read_text(encoding="utf-8"))
    segment_names = [item["id"] for item in domain["segments"]]
    splits, audit = load_v2_2_splits(
        REPO_ROOT,
        config,
        list(Geometry19.coordinate_names()),
        segment_names,
    )

    assert {name: len(bucket["x"]) for name, bucket in splits.items()} == {
        "train": 797,
        "validation": 264,
        "test": 237,
    }
    assert audit["design_count"] == 1298
    assert audit["group_count"] == 32
    assert audit["root_neighborhoods_unique_to_one_split"] is True
    assert audit["independent_roots_by_split_segment"]["test"] == {
        "low_mass_low_to_mid": 1,
        "high_mass_low_to_mid": 1,
        "cross_branch_mid": 2,
        "low_mass_mid_to_high": 1,
        "high_mass_mid_to_high": 1,
    }


def test_segment_metrics_are_reported_independently() -> None:
    truth = np.asarray([[100.0, 1.0e-4, 2.0], [50.0, 2.0e-4, 1.0]])
    prediction = np.asarray([[101.0, 1.0e-4, 2.0], [50.0, 2.2e-4, 1.0]])
    records = [{"assigned_segment": "a"}, {"assigned_segment": "b"}]
    result = evaluate_predictions(
        truth,
        prediction,
        records,
        ["a", "b"],
        [0.01, 0.01, 0.01],
        np.ones(3),
        "natural_log",
    )

    assert result["by_segment"]["a"]["sample_count"] == 1
    assert result["by_segment"]["b"]["rho_p95"] == pytest.approx(10.0)
    assert result["worst_segment_by_rho_p95"] == "b"


def test_seed_selection_uses_validation_and_ignores_test() -> None:
    def run(seed: int, validation_worst: float, test_worst: float) -> dict:
        return {
            "seed": seed,
            "metrics": {
                "validation": {
                    "worst_segment_rho_p95": validation_worst,
                    "overall": {"normalized_rmse": validation_worst},
                },
                "test": {"worst_segment_rho_p95": test_worst},
            },
        }

    assert select_seed([run(42, 2.0, 0.1), run(43, 1.0, 100.0)]) == 43


def test_protocol_freezes_performance_only_nonformal_scope() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["formal_use_allowed"] is False
    assert config["model"]["ordered_outputs"] == PERFORMANCE_NAMES
    assert config["model"]["output_transform"] == "natural_log"
    assert "no_pipeline_or_default_model_change" in config["limitations"]
