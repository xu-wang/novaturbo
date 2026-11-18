"""Audited smooth transform between normalized design bounds and model space."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Tuple

import torch
from torch.nn import functional as F


class SmoothBoundedTransform:
    """Map ``(-1, 1)`` to the real line and back with explicit diagnostics."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        kind: str = "atanh_tanh",
        epsilon: float = 1.0e-6,
        input_tolerance: float = 1.0e-6,
        saturation_threshold: float = 0.99,
        offset: tuple[float, ...] = (),
        dimension: int | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.kind = str(kind)
        self.epsilon = float(epsilon)
        self.input_tolerance = float(input_tolerance)
        self.saturation_threshold = float(saturation_threshold)
        self.offset = tuple(float(value) for value in offset)
        self.dimension = int(dimension) if dimension is not None else None
        if self.kind not in {"atanh_tanh", "offset_sigmoid"}:
            raise ValueError("bounded_output.kind must be atanh_tanh or offset_sigmoid")
        if not 0.0 < self.epsilon < 0.1:
            raise ValueError("bounded_output.epsilon must be between zero and 0.1")
        if self.input_tolerance < 0.0:
            raise ValueError("bounded_output.input_tolerance must be non-negative")
        if not 0.0 < self.saturation_threshold < 1.0:
            raise ValueError(
                "bounded_output.saturation_threshold must be between zero and one"
            )
        if self.kind == "atanh_tanh" and self.offset:
            raise ValueError("atanh_tanh does not accept bounded_output.offset")
        if self.kind == "offset_sigmoid":
            if not self.offset:
                raise ValueError("offset_sigmoid requires bounded_output.offset")
            if self.dimension is not None and len(self.offset) not in {1, self.dimension}:
                raise ValueError("bounded_output.offset must be scalar or match design_dim")
            if not all(math.isfinite(value) for value in self.offset):
                raise ValueError("bounded_output.offset must contain finite values")

    @classmethod
    def from_config(
        cls, value: Any, *, dimension: int | None = None
    ) -> "SmoothBoundedTransform":
        if value is None or value is False:
            return cls(enabled=False, dimension=dimension)
        if value is True:
            return cls(enabled=True, dimension=dimension)
        if not isinstance(value, Mapping):
            raise ValueError("bounded_output must be a boolean or mapping")
        allowed = {
            "enabled",
            "kind",
            "epsilon",
            "input_tolerance",
            "saturation_threshold",
            "offset",
            "offset_strategy",
        }
        unknown = set(value).difference(allowed)
        if unknown:
            raise ValueError(f"Unknown bounded_output keys: {sorted(unknown)}")
        return cls(
            enabled=bool(value.get("enabled", True)),
            kind=str(value.get("kind", "atanh_tanh")),
            epsilon=float(value.get("epsilon", 1.0e-6)),
            input_tolerance=float(value.get("input_tolerance", 1.0e-6)),
            saturation_threshold=float(value.get("saturation_threshold", 0.99)),
            offset=tuple(value.get("offset", ())),
            dimension=dimension,
        )

    def to_config(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "kind": self.kind,
            "epsilon": self.epsilon,
            "input_tolerance": self.input_tolerance,
            "saturation_threshold": self.saturation_threshold,
            "offset": list(self.offset),
        }

    def _offset_tensor(self, reference: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(self.offset, dtype=reference.dtype, device=reference.device)
        if values.numel() == 1:
            values = values.expand(reference.shape[-1])
        if values.numel() != reference.shape[-1]:
            raise ValueError("offset_sigmoid offset dimension differs from tensor")
        return values

    def to_unconstrained(
        self, bounded: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return model-space value, log|dy/dx|, and audited clip rate."""

        if not self.enabled:
            zeros = torch.zeros(
                bounded.shape[:-1], device=bounded.device, dtype=bounded.dtype
            )
            return bounded, zeros, zeros
        if not torch.isfinite(bounded).all():
            raise ValueError("Bounded transform input contains non-finite values")
        maximum = float(bounded.detach().abs().max())
        if maximum > 1.0 + self.input_tolerance:
            raise ValueError(
                f"Bounded transform input exceeds [-1,1]: maximum absolute value={maximum}"
            )
        if self.kind == "offset_sigmoid":
            probability_raw = (
                0.5 * (bounded + 1.0) - self.epsilon
            ) / (1.0 - 2.0 * self.epsilon)
            probability = probability_raw.clamp(
                min=self.epsilon, max=1.0 - self.epsilon
            )
            clipped = (probability != probability_raw).to(bounded.dtype).mean(dim=-1)
            unconstrained = torch.logit(probability) - self._offset_tensor(bounded)
            log_abs_det = (
                -math.log(2.0 * (1.0 - 2.0 * self.epsilon))
                - torch.log(probability)
                - torch.log1p(-probability)
            ).sum(dim=-1)
            return unconstrained, log_abs_det, clipped
        scale = 1.0 - self.epsilon
        unit_raw = bounded / scale
        unit = unit_raw.clamp(min=-scale, max=scale)
        clipped = (unit != unit_raw).to(bounded.dtype).mean(dim=-1)
        unconstrained = torch.atanh(unit)
        log_abs_det = (
            -math.log(scale) - torch.log1p(-unit.square())
        ).sum(dim=-1)
        return unconstrained, log_abs_det, clipped

    def to_bounded(
        self, unconstrained: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return bounded value and log|dx/dy| using a stable tanh Jacobian."""

        if not self.enabled:
            zeros = torch.zeros(
                unconstrained.shape[:-1],
                device=unconstrained.device,
                dtype=unconstrained.dtype,
            )
            return unconstrained, zeros
        if not torch.isfinite(unconstrained).all():
            raise ValueError("Unconstrained transform input contains non-finite values")
        if self.kind == "offset_sigmoid":
            shifted = unconstrained + self._offset_tensor(unconstrained)
            probability = torch.sigmoid(shifted)
            bounded = 2.0 * (
                self.epsilon + (1.0 - 2.0 * self.epsilon) * probability
            ) - 1.0
            elementwise = (
                math.log(2.0 * (1.0 - 2.0 * self.epsilon))
                - F.softplus(-shifted)
                - F.softplus(shifted)
            )
            return bounded, elementwise.sum(dim=-1)
        scale = 1.0 - self.epsilon
        bounded = scale * torch.tanh(unconstrained)
        log_two = math.log(2.0)
        elementwise = math.log(scale) + 2.0 * (
            log_two - unconstrained - F.softplus(-2.0 * unconstrained)
        )
        return bounded, elementwise.sum(dim=-1)

    def sample_diagnostics(
        self, raw: torch.Tensor, output: torch.Tensor
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {"bounded_output_enabled": False}
        raw_flat = raw.detach().reshape(-1, raw.shape[-1])
        output_flat = output.detach().reshape(-1, output.shape[-1])
        boundary_distance = 1.0 - output_flat.abs()
        return {
            "bounded_output_enabled": True,
            "bounded_transform": self.kind,
            "bounded_epsilon": self.epsilon,
            "raw_pre_transform_outside_unit_rate": float(
                (raw_flat.abs() > 1.0).any(dim=-1).to(torch.float32).mean().cpu()
            ),
            "output_out_of_bounds_rate": float(
                (output_flat.abs() > 1.0).any(dim=-1).to(torch.float32).mean().cpu()
            ),
            "output_near_boundary_rate": float(
                (output_flat.abs() >= self.saturation_threshold)
                .any(dim=-1)
                .to(torch.float32)
                .mean()
                .cpu()
            ),
            "output_min_boundary_distance": float(boundary_distance.min().cpu()),
            "output_max_abs": float(output_flat.abs().max().cpu()),
        }
