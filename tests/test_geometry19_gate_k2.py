from src.ai.geometry19_gate_k2 import (
    _closure_sort_key,
    _power_of_two_floor,
    _render_report,
    select_representative_runs,
)


def test_representative_seed_uses_validation_not_test() -> None:
    summary = {
        "arms": {
            "s1": {
                "loss": {"mode": "tolerance_huber"},
                "runs": [
                    {
                        "seed": 42,
                        "model_tag": "a",
                        "artifacts": {"manifest": "/tmp/a/model_manifest.json"},
                        "validation": {"hit_alignment": {"rho_p50": 2.0}},
                        "test": {"hit_alignment": {"rho_p50": 1.0}},
                    },
                    {
                        "seed": 43,
                        "model_tag": "b",
                        "artifacts": {"manifest": "/tmp/b/model_manifest.json"},
                        "validation": {"hit_alignment": {"rho_p50": 1.0}},
                        "test": {"hit_alignment": {"rho_p50": 9.0}},
                    },
                ],
            }
        }
    }
    selected = select_representative_runs(summary, ["s1"])
    assert selected["s1"]["seed"] == 43


def test_common_budget_is_power_of_two_floor() -> None:
    assert _power_of_two_floor(32) == 32
    assert _power_of_two_floor(31) == 16


def test_closure_sort_prioritizes_hits_before_rho_and_cost() -> None:
    def arm(hits: int, rho: float, calls: int) -> dict:
        return {
            "arm_id": str(hits),
            "final_target_hits": hits,
            "c_target_hits": 0,
            "median_best_true_rho": rho,
            "retained_production_hit_count": 0,
            "refinement": {"rho_improved_rate": 0.5},
            "total_v2_1_calls": calls,
            "frozen_top32_target_hits": 0,
        }
    assert _closure_sort_key(arm(3, 9.0, 300)) < _closure_sort_key(arm(2, 0.5, 100))


def test_report_keeps_full_row_when_refinement_rate_is_missing() -> None:
    summary = {
        "arms": {
            "s0": {
                "loss": {"mode": "legacy_mse"},
                "representative_seed": 42,
                "frozen_top32_target_hits": 0,
                "c_target_hits": 1,
                "d_rescue_hits": 0,
                "final_target_hits": 1,
                "median_best_true_rho": 1.25,
                "total_v2_1_calls": 128,
                "refinement": {"rho_improved_rate": None},
            }
        },
        "selection": {
            "winning_arm": "s0",
            "phase_a_decision": "STOP_LOSS_TUNING_AND_MOVE_TO_GATE_L",
        },
    }
    report = _render_report(summary)
    assert "| s0 | legacy_mse | 42 | 0/4 | 1/4" in report
    assert "| 1.250 | 128 | n/a |" in report
