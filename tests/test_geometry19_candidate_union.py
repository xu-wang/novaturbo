from src.ai.geometry19_candidate_union import _aggregate


def test_union_aggregate_target_coverage():
    rows = [
        {"target_id": "a", "oracle_hit": hit, "best_true_rho": 0.5 if hit else 2.0, "physics_acceptable": 10}
        for hit in (True, True, False)
    ] + [
        {"target_id": "b", "oracle_hit": False, "best_true_rho": 3.0, "physics_acceptable": 8}
        for _ in range(3)
    ]
    result = _aggregate(rows)
    assert result["any_seed_target_coverage"] == 0.5
    assert result["stable_target_fraction"] == 0.5
