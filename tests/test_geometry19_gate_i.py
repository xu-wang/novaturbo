from pathlib import Path

import yaml

from src.ai.geometry19_gate_i import GATE_I_VERSION, _power_of_two_floor, _report


def test_gate_i_config_has_complete_four_arm_ablation() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "config/experiments/wp19_23_gate_i_hit_alignment_ablation_v1.yaml").read_text()
    )
    assert config["protocol_version"] == GATE_I_VERSION
    assert len(config["targets"]) == 4
    arms = {arm["arm_id"]: arm for arm in config["arms"]}
    assert list(arms) == [
        "i0_legacy", "i1_rho_ranking", "i2_hit_loss", "i3_full_hit_alignment"
    ]
    assert arms["i0_legacy"]["candidate_ranking_mode"] == "candidate_id"
    assert arms["i1_rho_ranking"]["candidate_ranking_mode"] == "predicted_rho"
    assert arms["i2_hit_loss"]["dflow_target_loss_mode"] == "hit_aligned"
    assert arms["i3_full_hit_alignment"]["true_acceptance_mode"] == "hit_aligned"


def test_gate_i_budget_is_common_power_of_two() -> None:
    assert _power_of_two_floor(0) == 0
    assert _power_of_two_floor(31) == 16
    assert _power_of_two_floor(32) == 32
    assert _power_of_two_floor(55) == 32


def test_gate_i_report_handles_missing_rho() -> None:
    config = {
        "candidate_ranking_mode": "predicted_rho",
        "dflow_target_loss_mode": "hit_aligned",
        "refinement_target_loss_mode": "hit_aligned",
        "true_acceptance_mode": "hit_aligned",
    }
    text = _report({
        "target_count": 1,
        "common_c_budget_per_target": 32,
        "arms": {"i3": {
            "configuration": config, "c_target_hits": 0, "d_rescue_hits": 0,
            "final_target_hits": 0, "c_v2_1_calls": 32, "d_v2_1_calls": 24,
            "median_final_true_rho": None,
        }},
    })
    assert "| i3 |" in text
    assert "| n/a |" in text
