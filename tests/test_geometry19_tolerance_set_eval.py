from src.ai.geometry19_tolerance_set_eval import _decision, _replication_decision


def test_tolerance_set_gate_requires_all_frozen_checks():
    runs = [
        {"arm_id": "t0_frozen_baseline", "target_id": "a", "source_seed": 1, "oracle_hit": False},
        {"arm_id": "t0_frozen_baseline", "target_id": "b", "source_seed": 1, "oracle_hit": True},
        {"arm_id": "t1_half_tolerance", "target_id": "a", "source_seed": 1, "oracle_hit": True},
        {"arm_id": "t1_half_tolerance", "target_id": "b", "source_seed": 1, "oracle_hit": True},
    ]
    aggregate = {
        "t0_frozen_baseline": {"hit_rate": 0.5, "median_best_rho": 1.0, "p90_best_rho": 2.0},
        "t1_half_tolerance": {"hit_rate": 1.0, "median_best_rho": 0.7, "p90_best_rho": 1.8},
    }
    gates = {
        "minimum_hit_rate_gain": 0.1,
        "minimum_median_rho_reduction_fraction": 0.2,
        "require_p90_no_regression": True,
        "require_net_positive_paired_hits": True,
    }
    decision = _decision(runs, aggregate, gates)["t1_half_tolerance"]
    assert decision["passed"]
    assert decision["paired_wins"] == 1
    assert decision["paired_losses"] == 0


def test_replication_gate_is_distinct_from_strict_promotion_gate():
    decisions = {
        "a": {
            "passed": False,
            "hit_rate_delta_pp": 12.0,
            "median_reduction_fraction": 0.10,
            "checks": {"p90_no_regression": True, "net_positive_paired_hits": True},
        },
        "b": {
            "passed": False,
            "hit_rate_delta_pp": 10.0,
            "median_reduction_fraction": 0.05,
            "checks": {"p90_no_regression": True, "net_positive_paired_hits": True},
        },
        "c": {
            "passed": True,
            "hit_rate_delta_pp": 15.0,
            "median_reduction_fraction": 0.21,
            "checks": {"p90_no_regression": False, "net_positive_paired_hits": False},
        },
    }
    result = _replication_decision(
        decisions,
        {
            "arm_ids": ["a", "b", "c"],
            "minimum_median_hit_rate_gain": 0.10,
            "minimum_seed_arms_with_lower_median": 2,
            "minimum_seed_arms_with_p90_no_regression": 2,
            "minimum_seed_arms_with_net_positive_hits": 2,
            "minimum_strict_passes_for_promotion": 2,
        },
    )
    assert result["reproduced"]
    assert not result["promotion_passed"]
