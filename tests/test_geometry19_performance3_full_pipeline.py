from __future__ import annotations

import numpy as np
import torch

from src.ai.geometry19_performance3_full_pipeline import (
    _bootstrap_interval,
    _choose_targets,
    _decode_log_outputs,
    _paired_exact_p,
)


def test_decode_log_outputs_inverts_normalized_log_space() -> None:
    physical = torch.tensor([[10.0, 0.25, 3.0]], dtype=torch.float64)
    mean = np.log(np.asarray([5.0, 0.5, 1.5]))
    std = np.asarray([2.0, 0.5, 0.25])
    normalized = (torch.log(physical) - torch.tensor(mean)) / torch.tensor(std)
    decoded = _decode_log_outputs(normalized, {"output_mean": mean, "output_std": std})
    assert torch.allclose(decoded, physical)


def test_target_selection_is_stratified_and_order_invariant() -> None:
    rows = [
        {"target_region": region, "target": {"target_id": f"{region}-{index}"}}
        for region in ("dense", "mid", "sparse")
        for index in range(5)
    ]
    selected = _choose_targets(rows, ["dense", "mid", "sparse"], 2, 17)
    reversed_selected = _choose_targets(list(reversed(rows)), ["dense", "mid", "sparse"], 2, 17)
    assert selected == reversed_selected
    assert [row["target_region"] for row in selected].count("dense") == 2
    assert [row["target_region"] for row in selected].count("mid") == 2
    assert [row["target_region"] for row in selected].count("sparse") == 2


def test_paired_exact_and_cluster_bootstrap() -> None:
    assert _paired_exact_p(0, 0) == 1.0
    assert _paired_exact_p(4, 0) == 0.125
    interval = _bootstrap_interval(np.asarray([1.0, 1.0, -1.0, -1.0]), ["a", "a", "b", "b"], 100, 3)
    assert interval["point"] == 0.0
    assert interval["lower"] <= interval["point"] <= interval["upper"]
