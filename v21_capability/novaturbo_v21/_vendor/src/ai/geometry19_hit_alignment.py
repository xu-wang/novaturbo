"""Shared hit-aligned metrics for Geometry19 ranking and optimization."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch


HIT_ALIGNMENT_VERSION = "geometry19_hit_alignment_v1"
OUTPUT_NAMES = ("thrust_N", "tsfc_kg_N_s", "mass_kg")


class Geometry19HitAlignmentError(ValueError):
    """Raised when hit-aligned metric inputs violate their contract."""


def validate_hit_alignment_config(
    config: Mapping[str, Any], relative_tolerances: Mapping[str, float]
) -> None:
    """Validate the shared hit-alignment protocol block."""

    if config.get("version") != HIT_ALIGNMENT_VERSION or config.get("enabled") is not True:
        raise Geometry19HitAlignmentError("Hit alignment must be enabled with v1 semantics")
    if tuple(config.get("ordered_outputs") or ()) != OUTPUT_NAMES:
        raise Geometry19HitAlignmentError("Hit-alignment output order mismatch")
    if config.get("axis_scale") != "relative_error_divided_by_acceptance_tolerance":
        raise Geometry19HitAlignmentError("Hit-alignment axis scale mismatch")
    if config.get("hard_metric") != "rho_max_axis_ratio":
        raise Geometry19HitAlignmentError("Hit-alignment hard metric mismatch")
    if config.get("surrogate_loss") != "smooth_max_axis_ratio":
        raise Geometry19HitAlignmentError("Hit-alignment surrogate loss mismatch")
    if config.get("true_acceptance") != "lexicographic_failed_axes_rho_sum_constraint":
        raise Geometry19HitAlignmentError("Hit-alignment true acceptance mismatch")
    alpha = float(config.get("smooth_max_alpha", math.nan))
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise Geometry19HitAlignmentError("smooth_max_alpha must be finite and positive")
    if set(relative_tolerances) != set(OUTPUT_NAMES) or any(
        not math.isfinite(float(relative_tolerances[name]))
        or float(relative_tolerances[name]) <= 0.0
        for name in OUTPUT_NAMES
    ):
        raise Geometry19HitAlignmentError("Relative tolerances are invalid")


def metrics_from_relative_errors(
    relative_errors: Mapping[str, float], relative_tolerances: Mapping[str, float]
) -> dict[str, Any]:
    """Return the exact non-compensating hit metric from true relative errors."""

    if set(relative_errors) != set(OUTPUT_NAMES):
        raise Geometry19HitAlignmentError("Relative-error output set mismatch")
    ratios = {
        name: float(relative_errors[name]) / float(relative_tolerances[name])
        for name in OUTPUT_NAMES
    }
    if any(not math.isfinite(value) or value < 0.0 for value in ratios.values()):
        raise Geometry19HitAlignmentError("Axis ratios must be finite and nonnegative")
    failed = sum(value > 1.0 for value in ratios.values())
    return {
        "version": HIT_ALIGNMENT_VERSION,
        "axis_ratios": ratios,
        "failed_axis_count": failed,
        "rho": max(ratios.values()),
        "sum_axis_ratios": sum(ratios.values()),
        "strict_hit": failed == 0,
    }


def evaluation_hit_key(evaluation: Mapping[str, Any]) -> tuple[float, ...]:
    """Return invalid/failed/rho/sum ordering fields for a true evaluation."""

    if not evaluation.get("physics_acceptable"):
        return (1.0, math.inf, math.inf, math.inf)
    metrics = evaluation.get("hit_alignment")
    if not isinstance(metrics, Mapping):
        return (0.0, math.inf, math.inf, math.inf)
    return (
        0.0,
        float(metrics["failed_axis_count"]),
        float(metrics["rho"]),
        float(metrics["sum_axis_ratios"]),
    )


def torch_hit_aligned_terms(
    physical_prediction: torch.Tensor,
    target_values: torch.Tensor,
    tolerances: torch.Tensor,
    epsilons: torch.Tensor,
    *,
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Return differentiable tolerance-normalized errors and smooth minimax loss."""

    if physical_prediction.ndim != 2 or physical_prediction.shape[1] < len(OUTPUT_NAMES):
        raise Geometry19HitAlignmentError("Prediction must have shape [n, >=3]")
    if target_values.shape != (len(OUTPUT_NAMES),):
        raise Geometry19HitAlignmentError("Target tensor must have shape [3]")
    if tolerances.shape != target_values.shape or epsilons.shape != target_values.shape:
        raise Geometry19HitAlignmentError("Tolerance and epsilon tensors must have shape [3]")
    denominator = tolerances * torch.maximum(target_values.abs(), epsilons)
    ratios = (physical_prediction[:, :3] - target_values).abs() / denominator
    smooth_rho = torch.logsumexp(float(alpha) * ratios, dim=1) / float(alpha)
    return {
        "axis_ratios": ratios,
        "rho": ratios.max(dim=1).values,
        "sum_axis_ratios": ratios.sum(dim=1),
        "smooth_rho": smooth_rho,
    }


def ordered_tensor(
    values: Mapping[str, float], *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Build a tensor using the frozen output order."""

    return torch.tensor(
        [float(values[name]) for name in OUTPUT_NAMES], dtype=dtype, device=device
    )
