from __future__ import annotations

import pytest
import numpy as np
import torch

from src.ai.geometry19_hit_alignment import (
    evaluation_hit_key,
    metrics_from_relative_errors,
    torch_hit_aligned_terms,
)


TOLERANCES = {
    "thrust_N": 0.03,
    "tsfc_kg_N_s": 0.01,
    "mass_kg": 0.02,
}


def test_exact_rho_is_equivalent_to_three_axis_and_hit() -> None:
    passing = metrics_from_relative_errors(
        {"thrust_N": 0.03, "tsfc_kg_N_s": 0.005, "mass_kg": 0.01},
        TOLERANCES,
    )
    failing = metrics_from_relative_errors(
        {"thrust_N": 0.01, "tsfc_kg_N_s": 0.011, "mass_kg": 0.01},
        TOLERANCES,
    )
    assert passing["rho"] == pytest.approx(1.0)
    assert passing["strict_hit"] is True
    assert failing["rho"] == pytest.approx(1.1)
    assert failing["failed_axis_count"] == 1
    assert failing["strict_hit"] is False


def test_evaluation_key_prioritizes_failed_axis_count_then_rho() -> None:
    fewer_failed = {
        "physics_acceptable": True,
        "hit_alignment": {
            "failed_axis_count": 1,
            "rho": 1.5,
            "sum_axis_ratios": 2.0,
        },
    }
    lower_rho_but_more_failed = {
        "physics_acceptable": True,
        "hit_alignment": {
            "failed_axis_count": 2,
            "rho": 1.1,
            "sum_axis_ratios": 2.1,
        },
    }
    assert evaluation_hit_key(fewer_failed) < evaluation_hit_key(lower_rho_but_more_failed)


def test_torch_smooth_rho_focuses_on_largest_tolerance_ratio() -> None:
    prediction = torch.tensor([[103.0, 2.02e-5, 2.04]], requires_grad=True)
    terms = torch_hit_aligned_terms(
        prediction,
        torch.tensor([100.0, 2.0e-5, 2.0]),
        torch.tensor([0.03, 0.01, 0.02]),
        torch.tensor([1.0e-12, 1.0e-15, 1.0e-15]),
        alpha=12.0,
    )
    np.testing.assert_allclose(
        terms["axis_ratios"].detach().numpy(), [[1.0, 1.0, 1.0]], rtol=1.0e-5
    )
    assert terms["rho"].item() == pytest.approx(1.0)
    assert terms["smooth_rho"].item() >= terms["rho"].item()
    terms["smooth_rho"].sum().backward()
    assert torch.isfinite(prediction.grad).all()
