import numpy as np

from src.ai.geometry19_gate_l import _cosine, collect_direction_pairs


def test_cosine_reports_alignment_and_zero_vector() -> None:
    assert _cosine(np.asarray([1.0, 0.0]), np.asarray([2.0, 0.0])) == 1.0
    assert _cosine(np.asarray([0.0, 0.0]), np.asarray([1.0, 0.0])) is None


def test_collect_direction_pairs_rejects_missing_root(tmp_path) -> None:
    fields = [
        {"name": "x", "dtype": "float64", "bounds": [0.0, 1.0]},
        {"name": "z", "dtype": "int64", "bounds": [1, 3]},
    ]
    pairs, counts = collect_direction_pairs(
        tmp_path / "missing", fields, minimum_normalized_direction_norm=1.0e-6
    )
    assert pairs == []
    assert counts["ledger_rows"] == 0
