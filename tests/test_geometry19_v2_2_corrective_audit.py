"""Tests for the post-hoc corrective failure audit."""

from __future__ import annotations

import numpy as np
import pytest
import json
from pathlib import Path

from src.ai.geometry19_v2_2_corrective_audit import _nearest, _root_concentration


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT = REPO_ROOT / "data/validation/v2_2_corrective_failure_audit_v1/audit_summary.json"


def test_nearest_distance_reports_l2_and_linf_quantiles() -> None:
    result = _nearest(np.asarray([[1.0, 1.0]]), np.asarray([[0.0, 0.0]]))
    assert result["l2"]["p50"] == pytest.approx(2.0**0.5)
    assert result["linf"]["p95"] == pytest.approx(1.0)


def test_root_concentration_uses_effective_row_weighted_root_count() -> None:
    rows = [
        {"assigned_segment": "a", "root_seed_id": "r1"},
        {"assigned_segment": "a", "root_seed_id": "r1"},
        {"assigned_segment": "a", "root_seed_id": "r1"},
        {"assigned_segment": "a", "root_seed_id": "r2"},
    ]
    result = _root_concentration(rows, "a")
    assert result["roots"] == 2
    assert result["maximum_root_row_share"] == pytest.approx(0.75)
    assert result["effective_root_count_by_row_weight"] == pytest.approx(1.6)


def test_frozen_failure_audit_blocks_model_change_and_confirmation_reuse() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["status"] == "AUDIT_COMPLETE"
    assert audit["decision"] == "NO_MODEL_CHANGE_ROOT_GENERALIZATION_REDESIGN_REQUIRED"
    assert audit["confirmation_reuse_allowed"] is False
    assert audit["deployment_allowed"] is False
    assert audit["generalization_gap"]["validation_to_train_rmse_ratio"] == pytest.approx(
        3.7102808416065005
    )
    assert audit["model_capacity"]["parameter_count"] == 144387
    assert audit["root_concentration"]["confirmation"]["low_mass_low_to_mid"][
        "roots"
    ] == 1
