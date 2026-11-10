from src.ai.geometry19_checkpoint_selection import _confirmation_aggregate, _selection_key


def test_checkpoint_selection_is_lexicographic():
    base = {"hit_rate": 0.5, "p90_best_rho": 4.0, "median_best_rho": 1.0, "validation_loss": 1.5}
    assert _selection_key({**base, "hit_rate": 0.6}) > _selection_key(base)
    assert _selection_key({**base, "p90_best_rho": 3.0}) > _selection_key(base)
    assert _selection_key({**base, "median_best_rho": 0.9}) > _selection_key(base)
    assert _selection_key({**base, "validation_loss": 1.4}) > _selection_key(base)


def test_confirmation_aggregate_counts_target_stability():
    rows = [
        {"arm_id": "a", "target_id": "x", "oracle_hit": value, "best_true_rho": 0.5 if value else 2.0, "physics_acceptable": 10}
        for value in (True, True, False)
    ] + [
        {"arm_id": "a", "target_id": "y", "oracle_hit": False, "best_true_rho": 3.0, "physics_acceptable": 8}
        for _ in range(3)
    ]
    result = _confirmation_aggregate(rows, "a")
    assert result["any_seed_target_coverage"] == 0.5
    assert result["stable_target_fraction"] == 0.5
