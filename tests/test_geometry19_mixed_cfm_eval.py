import numpy as np

from src.ai.geometry19_mixed_cfm_eval import _render, _top_tuple_labels


def test_render_compares_historical_projection_baseline():
    metric = {"oracle_hit_rate": 0.6, "median_best_true_rho": 0.9, "p90_best_true_rho": 2.0, "mean_physics_acceptable": 500.0}
    aggregate = {arm: {split: {"512": metric} for split in ("train_diagnostic", "locked_held_out")} for arm in ("a_gumbel", "c_dequantized")}
    baseline_row = {"run_hit_rate": 0.556, "median_best_rho": 0.948, "mean_physics_acceptable": 364.0}
    summary = {
        "generated_at_utc": "now", "arm_ids": ["a_gumbel", "c_dequantized"], "evaluation_key": "512",
        "aggregate": aggregate,
        "historical_projection_baseline": {split: baseline_row for split in ("train_diagnostic", "locked_held_out")},
        "comparison_to_projection": {arm: {"heldout_hit_rate_delta_pp": 4.4, "heldout_median_best_rho_delta": -0.048} for arm in ("a_gumbel", "c_dequantized")},
        "funnel": {arm: {"raw": 512, "INVALID_DISCRETE_COORDINATE": 0, "GEOMETRY_MAPPING_FAILED": 0, "prefilter_valid": 512, "physics_acceptable": 500, "strict_hits": 2} for arm in ("a_gumbel", "c_dequantized")},
    }
    report = _render(summary)
    assert "最近整数投影基线" in report
    assert "55.6%" in report


def test_tuple_retrieval_is_unique_and_round_robin():
    train_conditions = np.asarray([[0.0], [0.1], [0.2], [0.3]])
    train_labels = np.asarray([[1, 1, 1], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    labels = _top_tuple_labels(np.asarray([0.0]), train_conditions, train_labels, 3, 7)
    assert labels.tolist() == [[1,1,1],[2,2,2],[3,3,3],[1,1,1],[2,2,2],[3,3,3],[1,1,1]]
