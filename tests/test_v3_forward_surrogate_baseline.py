"""V3 forward baseline split, fit and selected-only holdout gates."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.ai.v3_forward_surrogate_baseline import (
    V3ForwardSurrogateError,
    _load_split,
    _source,
    _stats,
    _weights,
    evaluate_v3_forward_holdout,
    verify_v3_forward_complete,
    verify_v3_forward_fit,
    verify_v3_forward_protocol,
    _config,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/validation/v3_22d_forward_surrogate_baseline_v1"


def test_protocol_and_normalization_use_only_development_centers():
    lock = verify_v3_forward_protocol(ROOT, OUTPUT)
    fit = verify_v3_forward_fit(ROOT, OUTPUT)
    config = _config(ROOT)
    source, _ = _source(ROOT, config)
    x, y, centers = _load_split(
        source, "development", 11289, 23, lock["design_schema_hash"], lock["physics_bundle_hash"]
    )
    expected = _stats(x, y, _weights(centers))
    actual = json.loads((OUTPUT / "normalization.json").read_text())
    assert actual == expected
    assert fit["normalization_fit_split"] == "development"
    assert len(lock["development_center_ids"]) == 23
    assert len(lock["validation_center_ids"]) == 5
    assert not set(lock["development_center_ids"]) & set(lock["validation_center_ids"])
    assert lock["holdout_content_parsed"] is False
    assert all(asset["checkpoint_reload_equal"] for asset in fit["models"])


def test_selection_precedes_selected_only_holdout():
    complete = verify_v3_forward_complete(ROOT, OUTPUT)
    fit = json.loads((OUTPUT / "fit_manifest.json").read_text())
    selection = json.loads((OUTPUT / "selection.json").read_text())
    holdout = json.loads((OUTPUT / "holdout_report.json").read_text())
    chosen = min(
        fit["models"],
        key=lambda asset: (
            asset["validation"]["worst_center_p95_max_abs_log_error"],
            asset["validation"]["overall_p95_max_abs_log_error"],
            asset["model_id"],
        ),
    )
    assert complete["selected_model_id"] == selection["selected_model_id"] == chosen["model_id"]
    assert holdout["selected_model_id"] == chosen["model_id"]
    assert holdout["model_selection_used_holdout"] is False
    assert holdout["metrics"]["rows"] == 2485
    assert holdout["metrics"]["centers"] == 5
    assert np.isfinite(holdout["metrics"]["worst_center_p95_max_abs_log_error"])
    assert complete["formal_use_allowed"] is False
    with pytest.raises(V3ForwardSurrogateError, match="already exists"):
        evaluate_v3_forward_holdout(ROOT, OUTPUT)
