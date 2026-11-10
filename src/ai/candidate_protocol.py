"""Shared V2.1 candidate-count protocol for target-directed inverse design."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


V21_RAW_CANDIDATE_COUNT = 128


def largest_power_of_two_not_above(count: int) -> int:
    """Return the largest power of two not greater than ``count``."""

    value = int(count)
    if value < 0:
        raise ValueError("candidate count cannot be negative")
    if value == 0:
        return 0
    return 1 << (value.bit_length() - 1)


@dataclass(frozen=True)
class CandidateProtocolV21Result:
    """Candidate IDs retained by the fixed-raw V2.1 protocol."""

    n_raw: int
    n_valid: int
    n_keep: int
    n_trimmed: int
    valid_rate: float
    kept_candidate_ids: tuple[int, ...]
    kept_positions: tuple[int, ...]

    @property
    def failed(self) -> bool:
        return self.n_keep == 0


def apply_candidate_protocol_v21(
    candidate_ids: Sequence[int],
    valid_mask: Sequence[bool],
    *,
    expected_raw_count: int = V21_RAW_CANDIDATE_COUNT,
) -> CandidateProtocolV21Result:
    """Filter then trim candidates solely by their original candidate IDs.

    Performance values are intentionally absent from this interface.  Valid rows are
    sorted by the immutable original ``candidate_id`` and the first ``N_keep`` rows
    are retained, where ``N_keep`` is the greatest power of two not above
    ``N_valid``.
    """

    ids = np.asarray(candidate_ids)
    valid = np.asarray(valid_mask, dtype=bool)
    if ids.ndim != 1 or valid.ndim != 1 or len(ids) != len(valid):
        raise ValueError("candidate_ids and valid_mask must be equal-length vectors")
    if len(ids) != int(expected_raw_count):
        raise ValueError(
            f"V2.1 requires exactly {expected_raw_count} raw candidates, got {len(ids)}"
        )
    if not np.issubdtype(ids.dtype, np.integer):
        if not np.isfinite(ids.astype(float)).all() or not np.equal(
            ids.astype(float), ids.astype(int)
        ).all():
            raise ValueError("candidate_id values must be finite integers")
        ids = ids.astype(int)
    else:
        ids = ids.astype(int, copy=False)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("candidate_id values must be unique")

    valid_positions = np.flatnonzero(valid)
    ordered_positions = sorted(valid_positions.tolist(), key=lambda index: int(ids[index]))
    n_valid = len(ordered_positions)
    n_keep = largest_power_of_two_not_above(n_valid)
    kept_positions = tuple(int(value) for value in ordered_positions[:n_keep])
    kept_ids = tuple(int(ids[value]) for value in kept_positions)
    return CandidateProtocolV21Result(
        n_raw=len(ids),
        n_valid=n_valid,
        n_keep=n_keep,
        n_trimmed=n_valid - n_keep,
        valid_rate=float(n_valid / len(ids)) if len(ids) else math.nan,
        kept_candidate_ids=kept_ids,
        kept_positions=kept_positions,
    )
