"""Hit-aligned training objectives and metrics for the Geometry19 surrogate."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from src.ai.geometry19_hit_alignment import OUTPUT_NAMES


LOSS_MODES = ("legacy_mse", "tolerance_huber", "smooth_minimax", "hybrid")


class Geometry19SurrogateObjectiveError(ValueError):
    """Raised when the surrogate objective contract is invalid."""


def validate_surrogate_loss_config(config: Mapping[str, Any]) -> None:
    mode = str(config.get("mode"))
    if mode not in LOSS_MODES:
        raise Geometry19SurrogateObjectiveError(f"Unsupported surrogate loss mode: {mode}")
    for name in ("performance_weight", "constraint_weight"):
        value = float(config.get(name, math.nan))
        if not math.isfinite(value) or value < 0.0:
            raise Geometry19SurrogateObjectiveError(f"{name} must be finite and nonnegative")
    if mode == "legacy_mse":
        return
    alpha = float(config.get("smooth_max_alpha", math.nan))
    huber_delta = float(config.get("huber_delta", math.nan))
    hybrid_weight = float(config.get("hybrid_mean_weight", math.nan))
    smooth_abs_epsilon = float(config.get("smooth_abs_epsilon", math.nan))
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise Geometry19SurrogateObjectiveError("smooth_max_alpha must be positive")
    if not math.isfinite(huber_delta) or huber_delta <= 0.0:
        raise Geometry19SurrogateObjectiveError("huber_delta must be positive")
    if not math.isfinite(hybrid_weight) or not 0.0 <= hybrid_weight <= 1.0:
        raise Geometry19SurrogateObjectiveError("hybrid_mean_weight must be in [0,1]")
    if not math.isfinite(smooth_abs_epsilon) or smooth_abs_epsilon <= 0.0:
        raise Geometry19SurrogateObjectiveError("smooth_abs_epsilon must be positive")


def surrogate_loss_terms(
    normalized_prediction: torch.Tensor,
    normalized_truth: torch.Tensor,
    output_mean: torch.Tensor,
    output_std: torch.Tensor,
    relative_tolerances: torch.Tensor,
    denominator_epsilons: torch.Tensor,
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Return differentiable performance, constraint, and total losses."""

    validate_surrogate_loss_config(config)
    if normalized_prediction.shape != normalized_truth.shape:
        raise Geometry19SurrogateObjectiveError("Prediction/truth shapes differ")
    if normalized_prediction.ndim != 2 or normalized_prediction.shape[1] < 3:
        raise Geometry19SurrogateObjectiveError("Surrogate tensors must have shape [n, >=3]")
    if output_mean.shape != (normalized_prediction.shape[1],) or output_std.shape != output_mean.shape:
        raise Geometry19SurrogateObjectiveError("Output normalization shape mismatch")
    if relative_tolerances.shape != (3,) or denominator_epsilons.shape != (3,):
        raise Geometry19SurrogateObjectiveError("Tolerance tensors must have shape [3]")

    constraint_loss = (
        F.mse_loss(normalized_prediction[:, 3:], normalized_truth[:, 3:])
        if normalized_prediction.shape[1] > 3
        else normalized_prediction.new_zeros(())
    )
    mode = str(config["mode"])
    if mode == "legacy_mse":
        performance_loss = F.mse_loss(
            normalized_prediction[:, :3], normalized_truth[:, :3]
        )
        axis_ratios = normalized_prediction.new_full(
            (normalized_prediction.shape[0], 3), float("nan")
        )
        mean_loss = performance_loss
        minimax_loss = performance_loss
    else:
        prediction = normalized_prediction[:, :3] * output_std[:3] + output_mean[:3]
        truth = normalized_truth[:, :3] * output_std[:3] + output_mean[:3]
        denominator = relative_tolerances * torch.maximum(
            truth.abs(), denominator_epsilons
        )
        signed_ratios = (prediction - truth) / denominator
        axis_ratios = signed_ratios.abs()
        mean_per_row = F.smooth_l1_loss(
            signed_ratios,
            torch.zeros_like(signed_ratios),
            beta=float(config["huber_delta"]),
            reduction="none",
        ).mean(dim=1)
        epsilon = float(config["smooth_abs_epsilon"])
        smooth_absolute = torch.sqrt(signed_ratios.square() + epsilon**2) - epsilon
        alpha = float(config["smooth_max_alpha"])
        minimax_per_row = (
            torch.logsumexp(alpha * smooth_absolute, dim=1) - math.log(3.0)
        ) / alpha
        mean_loss = mean_per_row.mean()
        minimax_loss = minimax_per_row.mean()
        if mode == "tolerance_huber":
            performance_loss = mean_loss
        elif mode == "smooth_minimax":
            performance_loss = minimax_loss
        else:
            weight = float(config["hybrid_mean_weight"])
            performance_loss = weight * mean_loss + (1.0 - weight) * minimax_loss
    total = (
        float(config["performance_weight"]) * performance_loss
        + float(config["constraint_weight"]) * constraint_loss
    )
    return {
        "total": total,
        "performance": performance_loss,
        "constraint": constraint_loss,
        "mean_axis": mean_loss,
        "smooth_minimax": minimax_loss,
        "axis_ratios": axis_ratios,
    }


def hit_aligned_prediction_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    relative_tolerances: Mapping[str, float],
    denominator_epsilons: Mapping[str, float],
) -> dict[str, Any]:
    """Measure prediction error using the same per-axis scales as strict hits."""

    truth_array = np.asarray(truth, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    if truth_array.shape != prediction_array.shape or truth_array.ndim != 2 or truth_array.shape[1] < 3:
        raise Geometry19SurrogateObjectiveError("Metric arrays must share shape [n, >=3]")
    tolerances = np.asarray([relative_tolerances[name] for name in OUTPUT_NAMES])
    epsilons = np.asarray([denominator_epsilons[name] for name in OUTPUT_NAMES])
    denominator = tolerances * np.maximum(np.abs(truth_array[:, :3]), epsilons)
    ratios = np.abs(prediction_array[:, :3] - truth_array[:, :3]) / denominator
    rho = ratios.max(axis=1)
    return {
        "sample_count": int(len(ratios)),
        "strict_within_tolerance_count": int(np.sum(np.all(ratios <= 1.0, axis=1))),
        "strict_within_tolerance_rate": float(np.mean(np.all(ratios <= 1.0, axis=1))),
        "rho_p50": float(np.quantile(rho, 0.50)),
        "rho_p90": float(np.quantile(rho, 0.90)),
        "rho_p95": float(np.quantile(rho, 0.95)),
        "axis_ratios": {
            name: {
                "p50": float(np.quantile(ratios[:, index], 0.50)),
                "p90": float(np.quantile(ratios[:, index], 0.90)),
                "p95": float(np.quantile(ratios[:, index], 0.95)),
                "within_tolerance_rate": float(np.mean(ratios[:, index] <= 1.0)),
                "worst_axis_count": int(np.sum(np.argmax(ratios, axis=1) == index)),
            }
            for index, name in enumerate(OUTPUT_NAMES)
        },
    }
