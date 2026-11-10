from src.ai.geometry19_tuple_diagnostic import _aggregate


def test_tuple_aggregate_counts_hits():
    rows = [
        {"top_m": 4, "target_id": "x", "oracle_hit": hit, "best_true_rho": 0.5 if hit else 2.0, "physics_acceptable": 10}
        for hit in (True, False, False)
    ]
    result = _aggregate(rows, 4)
    assert result["hit_rate"] == 1 / 3
    assert result["coverage"] == 1.0
