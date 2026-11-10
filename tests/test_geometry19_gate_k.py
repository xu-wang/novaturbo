from pathlib import Path

import numpy as np
import pytest
import yaml

from src.ai.geometry19_gate_k import (
    GATE_K_VERSION,
    _refinement_diagnostics,
    ranking_benchmark,
)


TOLERANCES = {"thrust_N": 0.1, "tsfc_kg_N_s": 0.1, "mass_kg": 0.1}
EPSILONS = {"thrust_N": 1.0e-12, "tsfc_kg_N_s": 1.0e-15, "mass_kg": 1.0e-15}


def test_gate_k_config_freezes_four_arms_and_three_seeds() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "config/experiments/wp19_22_gate_k_hit_aligned_surrogate_v1.yaml").read_text()
    )
    assert config["protocol_version"] == GATE_K_VERSION
    assert config["seeds"] == [42, 43, 44]
    assert [arm["loss"]["mode"] for arm in config["arms"]] == [
        "legacy_mse", "tolerance_huber", "smooth_minimax", "hybrid"
    ]


def test_ranking_benchmark_excludes_identity_and_finds_alternative_hit() -> None:
    truth = np.asarray([
        [10.0, 1.0, 2.0],
        [10.5, 1.0, 2.0],
        [20.0, 2.0, 4.0],
        [20.5, 2.0, 4.0],
    ])
    prediction = truth.copy()
    metrics = ranking_benchmark(
        truth, prediction, TOLERANCES, EPSILONS, maximum_targets=4, budgets=[1, 2]
    )
    assert metrics["eligible_target_count"] == 4
    assert metrics["target_coverage"]["1"] == pytest.approx(1.0)
    assert metrics["first_hit_rank_p50"] == pytest.approx(1.0)


def test_refinement_diagnostics_only_score_comparable_valid_rows(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    rows = [
        {
            "accepted": True,
            "prior_hit_alignment": {"rho": 2.0, "failed_axis_count": 2},
            "proposed": {"hit_alignment": {"rho": 1.0, "failed_axis_count": 1}},
        },
        {"accepted": False, "prior_hit_alignment": None, "proposed": {}},
    ]
    import json
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    metrics = _refinement_diagnostics([path])
    assert metrics["true_evaluations"] == 2
    assert metrics["accepted_updates"] == 1
    assert metrics["rho_improved_rate"] == pytest.approx(1.0)
    assert metrics["median_rho_delta"] == pytest.approx(1.0)
