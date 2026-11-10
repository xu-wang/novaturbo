"""Tests for low-mass corrective retraining contracts."""

from __future__ import annotations

from pathlib import Path
import json

import pytest
import yaml

from src.ai.geometry19_v2_2_corrective_retraining import (
    _confirmation_gate,
    load_fresh_splits,
)
from src.physics import Geometry19


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config/experiments/geometry19_v2_2_low_mass_corrective_retraining_v1.yaml"
SUMMARY = (
    REPO_ROOT
    / "data/validation/v2_2_low_mass_corrective_retraining_v1/formal/summary.json"
)


def test_corrective_loader_preserves_global_root_disjointness() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    splits, audit = load_fresh_splits(REPO_ROOT, config, Geometry19.coordinate_names())
    assert {role: len(bucket["x"]) for role, bucket in splits.items()} == {
        "train": 1645,
        "validation": 244,
        "confirmation": 156,
    }
    assert audit["global_root_disjoint"] is True
    assert audit["roots_by_role"] == {"train": 20, "validation": 2, "confirmation": 2}


def test_confirmation_gate_requires_both_segments_and_tail_limit() -> None:
    baseline = {
        "overall": {"rho_p50": 10.0},
        "by_segment": {"a": {"rho_p50": 10.0}, "b": {"rho_p50": 5.0}},
    }
    challenger = {
        "overall": {"rho_p50": 5.0},
        "by_segment": {"a": {"rho_p50": 5.0}, "b": {"rho_p50": 4.5}},
        "worst_segment_rho_p95": 4.0,
    }
    result = _confirmation_gate(
        baseline,
        challenger,
        {
            "minimum_overall_rho_p50_reduction": 0.3,
            "minimum_each_segment_rho_p50_reduction": 0.2,
            "maximum_worst_segment_rho_p95": 5.0,
        },
        ["a", "b"],
    )
    assert result["passed"] is False
    assert result["checks"]["every_segment_rho_p50_reduction"] is False
    assert result["overall_rho_p50_reduction"] == pytest.approx(0.5)


def test_frozen_corrective_retraining_preserves_failed_confirmation() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["status"] == "CORRECTIVE_CONFIRMATION_FAIL"
    assert summary["selected_seed"] == 42
    assert summary["selection"]["confirmation_used_for_selection"] is False
    assert summary["confirmation_evaluated_once_after_selection"] is True
    assert summary["confirmation_gate"]["passed"] is False
    assert summary["confirmation_gate"]["overall_rho_p50_reduction"] == pytest.approx(
        -0.33935479092929866
    )
    assert summary["confirmation_gate"]["segment_rho_p50_reductions"] == pytest.approx(
        {
            "low_mass_low_to_mid": 0.21261165350808664,
            "low_mass_mid_to_high": -1.0500822628440063,
        }
    )
    assert all(run["weight_reload_equal"] for run in summary["runs"])
