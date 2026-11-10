import numpy as np
import pytest

from src.ai.candidate_protocol import (
    V21_RAW_CANDIDATE_COUNT,
    apply_candidate_protocol_v21,
    largest_power_of_two_not_above,
)


@pytest.mark.parametrize(
    ("n_valid", "expected"),
    [
        (128, 128),
        (127, 64),
        (65, 64),
        (64, 64),
        (63, 32),
        (33, 32),
        (32, 32),
        (2, 2),
        (1, 1),
        (0, 0),
    ],
)
def test_v21_required_power_of_two_cases(n_valid, expected):
    assert largest_power_of_two_not_above(n_valid) == expected
    mask = np.zeros(V21_RAW_CANDIDATE_COUNT, dtype=bool)
    mask[:n_valid] = True
    result = apply_candidate_protocol_v21(np.arange(128), mask)
    assert result.n_raw == 128
    assert result.n_valid == n_valid
    assert result.n_keep == expected
    assert result.n_trimmed == n_valid - expected
    assert result.failed is (n_valid == 0)


def test_v21_trimming_uses_candidate_id_not_performance():
    candidate_ids = np.arange(128)[::-1]
    valid = np.zeros(128, dtype=bool)
    valid[:100] = True
    performance = np.arange(128, dtype=float)
    performance[candidate_ids == 127] = -1.0e9

    first = apply_candidate_protocol_v21(candidate_ids, valid)
    performance[candidate_ids == 127] = 1.0e9
    second = apply_candidate_protocol_v21(candidate_ids, valid)

    assert first.kept_candidate_ids == second.kept_candidate_ids
    assert first.n_keep == 64
    assert first.kept_candidate_ids == tuple(range(28, 92))
    assert 127 not in first.kept_candidate_ids


def test_v21_rejects_any_raw_count_other_than_128():
    with pytest.raises(ValueError, match="exactly 128"):
        apply_candidate_protocol_v21(np.arange(127), np.ones(127, dtype=bool))
