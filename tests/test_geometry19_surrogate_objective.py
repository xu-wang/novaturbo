import numpy as np
import pytest
import torch

from src.ai.geometry19_surrogate_objective import (
    Geometry19SurrogateObjectiveError,
    hit_aligned_prediction_metrics,
    surrogate_loss_terms,
    validate_surrogate_loss_config,
)


TOLERANCES = torch.tensor([0.02745, 0.00857, 0.01171])
EPSILONS = torch.tensor([1.0e-9, 1.0e-12, 1.0e-9])


def _config(mode: str) -> dict:
    return {
        "mode": mode,
        "performance_weight": 0.5,
        "constraint_weight": 0.5,
        "huber_delta": 1.0,
        "smooth_max_alpha": 8.0,
        "hybrid_mean_weight": 0.5,
        "smooth_abs_epsilon": 1.0e-6,
    }


def _terms(prediction: torch.Tensor, truth: torch.Tensor, mode: str = "hybrid"):
    mean = torch.tensor([100.0, 1.0e-4, 2.0, 0.0])
    std = torch.tensor([20.0, 2.0e-5, 0.5, 1.0])
    return surrogate_loss_terms(
        prediction, truth, mean, std, TOLERANCES, EPSILONS, _config(mode)
    )


def test_hit_aligned_loss_is_zero_at_truth_and_has_finite_gradients() -> None:
    truth = torch.zeros((2, 4))
    prediction = truth.clone().requires_grad_(True)
    terms = _terms(prediction, truth)
    assert float(terms["performance"].detach()) == pytest.approx(0.0, abs=1.0e-6)
    terms["total"].backward()
    assert torch.isfinite(prediction.grad).all()


def test_one_physical_tolerance_maps_to_unit_axis_ratio() -> None:
    truth = torch.zeros((1, 4))
    physical_delta = TOLERANCES[0] * 100.0
    prediction = truth.clone()
    prediction[0, 0] = physical_delta / 20.0
    terms = _terms(prediction, truth, "tolerance_huber")
    assert float(terms["axis_ratios"][0, 0]) == pytest.approx(1.0, rel=1.0e-5)


def test_smooth_minimax_gradient_focuses_worst_axis() -> None:
    truth = torch.zeros((1, 4))
    prediction = torch.tensor([[0.41175, 0.04285, 0.04684, 0.0]], requires_grad=True)
    terms = _terms(prediction, truth, "smooth_minimax")
    terms["performance"].backward()
    # Convert normalized-output gradients back to gradients with respect to
    # tolerance-normalized ratios before comparing axes.
    scale = torch.tensor([
        20.0 / (0.02745 * 100.0),
        2.0e-5 / (0.00857 * 1.0e-4),
        0.5 / (0.01171 * 2.0),
    ])
    ratio_gradient = prediction.grad[0, :3] / scale
    assert abs(float(ratio_gradient[0])) > abs(float(ratio_gradient[1]))
    assert torch.isfinite(prediction.grad).all()


def test_legacy_mode_exactly_matches_normalized_mse() -> None:
    truth = torch.zeros((2, 4))
    prediction = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, 2.0]])
    terms = _terms(prediction, truth, "legacy_mse")
    expected_performance = torch.nn.functional.mse_loss(prediction[:, :3], truth[:, :3])
    expected_constraint = torch.nn.functional.mse_loss(prediction[:, 3:], truth[:, 3:])
    assert torch.equal(terms["performance"], expected_performance)
    assert torch.equal(terms["total"], 0.5 * expected_performance + 0.5 * expected_constraint)


def test_metrics_match_strict_tolerance_semantics() -> None:
    truth = np.asarray([[100.0, 1.0e-4, 2.0], [100.0, 1.0e-4, 2.0]])
    prediction = truth.copy()
    prediction[0, 0] += 100.0 * 0.02745 * 0.9
    prediction[1, 1] += 1.0e-4 * 0.00857 * 1.1
    tolerances = {"thrust_N": 0.02745, "tsfc_kg_N_s": 0.00857, "mass_kg": 0.01171}
    epsilons = {"thrust_N": 1.0e-9, "tsfc_kg_N_s": 1.0e-12, "mass_kg": 1.0e-9}
    metrics = hit_aligned_prediction_metrics(truth, prediction, tolerances, epsilons)
    assert metrics["strict_within_tolerance_count"] == 1
    assert metrics["axis_ratios"]["thrust_N"]["p50"] == pytest.approx(0.45)


def test_invalid_loss_config_is_rejected() -> None:
    with pytest.raises(Geometry19SurrogateObjectiveError):
        validate_surrogate_loss_config({"mode": "unknown", "performance_weight": 1, "constraint_weight": 1})
