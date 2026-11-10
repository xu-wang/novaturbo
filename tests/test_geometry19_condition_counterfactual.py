from src.ai.geometry19_condition_counterfactual import _counterfactual_decision


def test_counterfactual_decision_selects_condition_strengthening_when_gate_fails():
    runs = [
        {"condition_mode": "own", "target_id": "a", "source_seed": 1, "oracle_hit": True},
        {"condition_mode": "own", "target_id": "b", "source_seed": 1, "oracle_hit": False},
        {"condition_mode": "shuffled", "target_id": "a", "source_seed": 1, "oracle_hit": False},
        {"condition_mode": "shuffled", "target_id": "b", "source_seed": 1, "oracle_hit": True},
    ]
    aggregate = {"own": {"hit_rate": 0.5}, "shuffled": {"hit_rate": 0.5}}
    gate = {
        "minimum_own_vs_shuffled_hit_gain": 0.1,
        "require_net_positive_paired_hits": True,
        "insufficient_branch": "condition_strengthened_cfg_cfm",
        "sufficient_branch": "tolerance_set_cfm",
    }
    result = _counterfactual_decision(runs, aggregate, gate)
    assert not result["condition_utilization_sufficient"]
    assert result["selected_branch"] == "condition_strengthened_cfg_cfm"


def test_counterfactual_decision_selects_tolerance_set_when_gate_passes():
    runs = [
        {"condition_mode": "own", "target_id": "a", "source_seed": 1, "oracle_hit": True},
        {"condition_mode": "own", "target_id": "b", "source_seed": 1, "oracle_hit": True},
        {"condition_mode": "shuffled", "target_id": "a", "source_seed": 1, "oracle_hit": False},
        {"condition_mode": "shuffled", "target_id": "b", "source_seed": 1, "oracle_hit": True},
    ]
    aggregate = {"own": {"hit_rate": 1.0}, "shuffled": {"hit_rate": 0.5}}
    gate = {
        "minimum_own_vs_shuffled_hit_gain": 0.1,
        "require_net_positive_paired_hits": True,
        "insufficient_branch": "condition_strengthened_cfg_cfm",
        "sufficient_branch": "tolerance_set_cfm",
    }
    result = _counterfactual_decision(runs, aggregate, gate)
    assert result["condition_utilization_sufficient"]
    assert result["selected_branch"] == "tolerance_set_cfm"
