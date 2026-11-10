from __future__ import annotations

import numpy as np

from src.ai.geometry19_reranker import build_pair_indices, ranking_metrics, true_utility


def test_invalid_candidate_is_worse_than_any_finite_rho() -> None:
    worst = 25.0
    assert true_utility(None, worst) < true_utility(worst, worst)
    assert true_utility(0.5, worst) > true_utility(2.0, worst)


def test_pair_builder_never_reverses_utility() -> None:
    rows = [
        {"run_id": "a", "rho": 0.5},
        {"run_id": "a", "rho": 2.0},
        {"run_id": "a", "rho": None},
    ]
    utilities = np.asarray([true_utility(0.5, 2.0), true_utility(2.0, 2.0), true_utility(None, 2.0)])
    good, bad, weights = build_pair_indices(
        rows,
        utilities,
        seed=7,
        pairs_per_run=32,
        elite_per_run=2,
        minimum_gap=0.01,
    )
    assert len(good) == len(bad) == len(weights)
    assert np.all(utilities[good] > utilities[bad])
    assert np.all(weights > 0.0)


def test_ranking_metrics_use_only_hit_bearing_runs_for_recall() -> None:
    rows = [
        {"run_id": "a", "target_id": "t1", "rho": 3.0},
        {"run_id": "a", "target_id": "t1", "rho": 0.8},
        {"run_id": "b", "target_id": "t2", "rho": 4.0},
    ]
    metrics = ranking_metrics(rows, np.asarray([2.0, 1.0, 4.0]), [1, 2])
    assert metrics["hit_bearing_run_count"] == 1
    assert metrics["hit_bearing_recall"] == {"1": 0.0, "2": 1.0}
    assert metrics["all_run_success"] == {"1": 0.0, "2": 0.5}
    assert metrics["median_first_hit_rank"] == 2.0
