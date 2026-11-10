from src.ai.geometry19_generator_oracle import _prefix_metrics


def test_prefix_metrics_uses_raw_prefix_and_strict_rho():
    rows = [
        {"rho": None}, {"rho": 2.0}, {"rho": 0.9}, {"rho": 1.1},
    ]
    result = _prefix_metrics(rows, [2, 4])
    assert result["2"]["oracle_hit"] is False
    assert result["2"]["best_true_rho"] == 2.0
    assert result["4"]["oracle_hit"] is True
    assert result["4"]["strict_hits"] == 1
    assert result["4"]["hit_density"] == 0.25
