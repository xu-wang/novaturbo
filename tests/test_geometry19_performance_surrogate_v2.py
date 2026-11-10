from __future__ import annotations

import numpy as np
import pytest

from src.ai.geometry19_performance_surrogate_v2 import (
    Geometry19PerformanceSurrogateV2Error,
    inverse_transform_outputs,
    performance_metrics,
    ranking_metrics,
    transform_outputs,
)


def test_output_transforms_round_trip() -> None:
    values = np.asarray([[10.0, 1.0e-4, 2.0], [20.0, 2.0e-4, 3.0]], dtype=np.float32)
    for mode in ("identity", "natural_log"):
        restored = inverse_transform_outputs(transform_outputs(values, mode), mode)
        np.testing.assert_allclose(restored, values, rtol=1.0e-6)


def test_log_transform_rejects_nonpositive_values() -> None:
    with pytest.raises(Geometry19PerformanceSurrogateV2Error):
        transform_outputs(np.asarray([[1.0, 0.0, 2.0]], dtype=np.float32), "natural_log")


def test_performance_metrics_use_tolerance_normalized_rho() -> None:
    truth = np.asarray([[100.0, 10.0, 1.0], [100.0, 10.0, 1.0]])
    prediction = np.asarray([[101.0, 10.0, 1.0], [102.0, 10.0, 1.0]])
    result = performance_metrics(truth, prediction, [0.01, 0.01, 0.01])
    assert result["rho_p50"] == pytest.approx(1.5)
    assert result["outputs"]["thrust_N"]["relative_error_p50"] == pytest.approx(0.015)


def test_ranking_metrics_report_first_hit_and_nested_coverage() -> None:
    result = ranking_metrics(
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        np.asarray([2.0, 0.8, 3.0, 0.5]),
        ["a", "a", "b", "b"],
        [1, 2],
    )
    assert result["target_coverage"] == {"1": 0, "2": 2}
    assert result["first_hit_ranks"] == [2, 2]
