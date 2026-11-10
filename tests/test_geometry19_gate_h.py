from __future__ import annotations

import numpy as np

from src.ai.geometry19_gate_h import (
    _json_safe,
    _power_of_two_floor,
    merge_candidate_pools,
    select_tolerance_ranked_diverse,
)


def _row(key: str, design_hash: str, rho: float, x: list[float], mode: str) -> dict:
    return {
        "candidate_key": key,
        "candidate_id": int(key.split(":")[-1]),
        "source_modes": [mode],
        "source_candidate_ids": [int(key.split(":")[-1])],
        "geometry19": {},
        "design_hash": design_hash,
        "normalized_geometry": x,
        "strict_valid": False,
        "rescued_discrete": True,
        "predicted_rho": rho,
    }


def test_power_of_two_floor() -> None:
    assert _power_of_two_floor(0) == 0
    assert _power_of_two_floor(1) == 1
    assert _power_of_two_floor(31) == 16
    assert _power_of_two_floor(32) == 32


def test_json_safe_replaces_nonfinite_values() -> None:
    assert _json_safe({"values": [1.0, float("inf"), float("nan")]}) == {
        "values": [1.0, None, None]
    }


def test_merge_candidate_pools_preserves_provenance() -> None:
    direct = [_row("direct:0", "same", 2.0, [0.0, 0.0], "direct")]
    dflow = [_row("dflow:0", "same", 1.0, [0.0, 0.0], "dflow")]
    merged = merge_candidate_pools(direct, dflow)
    assert len(merged) == 1
    assert merged[0]["source_modes"] == ["dflow", "direct"]
    assert merged[0]["source_candidate_ids"] == [0]


def test_ranked_diverse_selection_is_deterministic_and_spreads_tail() -> None:
    rows = [
        _row(f"direct:{index}", str(index), float(index), [float(index), 0.0], "direct")
        for index in range(8)
    ]
    first = select_tolerance_ranked_diverse(rows, 4)
    second = select_tolerance_ranked_diverse(rows, 4)
    assert [row["design_hash"] for row in first] == [
        row["design_hash"] for row in second
    ]
    assert {row["design_hash"] for row in first[:2]} == {"0", "1"}
    assert max(float(row["normalized_geometry"][0]) for row in first) >= 6.0
    assert np.isfinite([row["predicted_rho"] for row in first]).all()
