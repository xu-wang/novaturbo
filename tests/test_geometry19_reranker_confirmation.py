from __future__ import annotations

import numpy as np

from src.ai.geometry19_reranker_confirmation import (
    cluster_bootstrap_difference,
    select_spread_targets,
)


def _row(sample_id: int, thrust: float) -> dict:
    return {
        "sample_id": sample_id,
        "physics_outputs": {
            "performance": {
                "thrust_N": thrust,
                "tsfc_kg_N_s": 1.0e-4 * (1.0 + thrust / 100.0),
                "mass_kg": 2.0 + thrust / 100.0,
            }
        },
    }


def test_spread_selection_is_deterministic_and_unique() -> None:
    rows = [_row(index, float(index + 1)) for index in range(20)]
    first = select_spread_targets(rows, 7)
    second = select_spread_targets(list(reversed(rows)), 7)
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert len({row["sample_id"] for row in first}) == 7


def test_cluster_bootstrap_uses_target_level_units() -> None:
    baseline = {"a1": {"32": False}, "a2": {"32": False}, "b1": {"32": True}}
    challenger = {"a1": {"32": True}, "a2": {"32": True}, "b1": {"32": True}}
    result = cluster_bootstrap_difference(
        baseline,
        challenger,
        {"a1": "a", "a2": "a", "b1": "b"},
        32,
        replicates=2000,
        seed=9,
        confidence_level=0.95,
    )
    assert result["point"] == 0.5
    assert 0.0 <= result["lower"] <= result["upper"] <= 1.0
