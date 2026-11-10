"""Target-directed benchmark aggregation tests."""

from __future__ import annotations

import pandas as pd

from src.ai.target_benchmark import _aggregate, _paired_comparison, _read_forward_results


def test_read_forward_results_accepts_zero_candidate_file(tmp_path):
    path = tmp_path / "forward_results.csv"
    path.write_text("", encoding="utf-8")
    frame = _read_forward_results(path)
    assert frame.empty
    assert {"forward_valid", "total_error", "within_tolerance"}.issubset(frame.columns)


def test_paired_comparison_treats_lower_target_error_as_a_generator_win():
    common = {
        "training_seed": 42,
        "target_id": "target-a",
        "target_type": "ID",
        "optimizer_seed": 7,
        "initial_hit_rate": 0.0,
        "initial_has_hit": False,
        "search_has_hit": False,
        "candidate_hit_rate": 0.0,
    }
    metrics = pd.DataFrame(
        [
            {
                **common,
                "method": "random",
                "initial_best_error": 0.2,
                "initial_top10_mean_error": 0.3,
                "search_best_error": 0.1,
            },
            {
                **common,
                "method": "screened_random",
                "initial_best_error": 0.1,
                "initial_top10_mean_error": 0.2,
                "search_best_error": 0.05,
            },
            {
                **common,
                "method": "nflows",
                "initial_best_error": 0.001,
                "initial_top10_mean_error": 0.002,
                "initial_hit_rate": 1.0,
                "initial_has_hit": True,
                "search_best_error": 0.0005,
                "search_has_hit": True,
                "candidate_hit_rate": 0.95,
            },
        ]
    )

    paired = _paired_comparison(metrics)
    screened = paired.loc[paired["baseline"] == "screened_random"].iloc[0]

    assert screened["initial_best_error_improvement"] == 0.099
    assert screened["initial_hit_rate_improvement"] == 1.0
    assert bool(screened["initial_best_win"])
    aggregate = _aggregate(metrics, paired).set_index("method")
    assert aggregate.loc["nflows", "vs_screened_random_initial_best_win_rate"] == 1.0
