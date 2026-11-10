from __future__ import annotations

from src.ai.geometry19_gate_h_sweep import select_predicted_rho


def _row(index: int, rho: float) -> dict:
    return {
        "candidate_key": f"direct:{index}",
        "predicted_rho": rho,
    }


def test_predicted_rho_selection_is_nested_and_deterministic() -> None:
    rows = [_row(0, 3.0), _row(1, 1.0), _row(2, 2.0), _row(3, 1.0)]
    selected_two = select_predicted_rho(rows, 2)
    selected_three = select_predicted_rho(rows, 3)
    assert [row["candidate_key"] for row in selected_two] == ["direct:1", "direct:3"]
    assert [row["candidate_key"] for row in selected_three[:2]] == [
        row["candidate_key"] for row in selected_two
    ]
