"""Independent zero-anchored Diagonal CFM generator.

The state convention follows de Campos, Krebs, and Gottschalk (2026):

    t = 0: [z, y]
    t = 1: [x, 0]

Unlike :mod:`torchcfm_adapter`, this adapter does not condition a design-only
velocity field.  It learns one unconditional velocity field in the augmented
design/label state space and therefore supports both inverse synthesis and
reverse-time label reconstruction.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint

from .base import ConditionalGenerator, register_generator
from ..generator_dataset import GeneratorNormalizer


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


class DiagCFMVelocityMLP(nn.Module):
    """Paper-style MLP velocity field on ``[design, label]`` state plus time."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        if state_dim <= 0 or hidden_dim <= 0 or hidden_layers <= 0:
            raise ValueError("Diag-CFM dimensions and hidden_layers must be positive")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Diag-CFM activation must be one of {sorted(_ACTIVATIONS)}"
            )
        activation_type = _ACTIVATIONS[activation]
        sizes = [state_dim + 1, *([hidden_dim] * hidden_layers), state_dim]
        layers = []
        for index, (source, target) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(source, target))
            if index < len(sizes) - 2:
                layers.append(activation_type())
        self.network = nn.Sequential(*layers)
        self.state_dim = int(state_dim)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(
                f"state must have shape [batch,{self.state_dim}], got {tuple(state.shape)}"
            )
        if t.ndim == 0:
            time_column = t.expand(state.shape[0], 1)
        elif t.ndim == 1:
            time_column = t[:, None]
        elif t.ndim == 2 and t.shape[1] == 1:
            time_column = t
        else:
            raise ValueError("time must be scalar, [batch], or [batch,1]")
        if time_column.shape[0] != state.shape[0]:
            if time_column.shape[0] == 1:
                time_column = time_column.expand(state.shape[0], 1)
            else:
                raise ValueError("time batch dimension differs from state")
        return self.network(torch.cat([state, time_column], dim=-1))


@register_generator
class DiagCFMGenerator(ConditionalGenerator):
    """Zero-anchored Diag-CFM with bidirectional augmented-state ODE flow."""

    generator_name = "diag_cfm"
    upstream_packages = ("torchdiffeq",)

    def __init__(
        self,
        model_config: Mapping[str, Any],
        runtime_config: Mapping[str, Any],
        schema_version: str = "generator_v1",
        normalizer: Optional[GeneratorNormalizer] = None,
    ) -> None:
        super().__init__(model_config, runtime_config, schema_version, normalizer)
        if self.condition_stabilization_enabled:
            raise ValueError(
                "Diag-CFM zero anchoring requires the exact normalized labels; "
                "condition_stabilization must be disabled"
            )
        forbidden = {
            "matcher",
            "sinkhorn_regularization",
            "sinkhorn_iterations",
            "film",
            "condition_encoder_dim",
            "residual_blocks",
            "time_embedding",
        }.intersection(model_config)
        if forbidden:
            raise ValueError(
                "Diag-CFM is an independent plain augmented-state baseline and does "
                f"not accept CFM-4 fields: {sorted(forbidden)}"
            )

        self.state_dim = self.design_dim + self.condition_dim
        self.hidden_dim = int(model_config.get("hidden_dim", 366))
        self.hidden_layers = int(model_config.get("hidden_layers", 4))
        self.activation = str(model_config.get("activation", "leaky_relu"))
        self.velocity = DiagCFMVelocityMLP(
            self.state_dim,
            self.hidden_dim,
            self.hidden_layers,
            self.activation,
        )

        self.base_distribution = str(
            model_config.get("base_distribution", "uniform")
        )
        if self.base_distribution not in {"uniform", "standard_normal"}:
            raise ValueError(
                "Diag-CFM base_distribution must be uniform or standard_normal"
            )
        self.base_low = float(model_config.get("base_low", -1.0))
        self.base_high = float(model_config.get("base_high", 1.0))
        if self.base_distribution == "uniform" and not self.base_high > self.base_low:
            raise ValueError("Diag-CFM uniform base_high must exceed base_low")

        state_scaling = dict(model_config.get("state_scaling", {}))
        self.design_state_scaling = str(state_scaling.get("design", "identity"))
        self.condition_state_scaling = str(
            state_scaling.get("condition", "identity")
        )
        if self.design_state_scaling not in {"identity", "zero_one"}:
            raise ValueError(
                "Diag-CFM state_scaling.design must be identity or zero_one"
            )
        if self.condition_state_scaling not in {"identity", "zero_one"}:
            raise ValueError(
                "Diag-CFM state_scaling.condition must be identity or zero_one"
            )
        condition_min = state_scaling.get("condition_input_min")
        condition_max = state_scaling.get("condition_input_max")
        if self.condition_state_scaling == "zero_one":
            if condition_min is None or condition_max is None:
                raise ValueError(
                    "zero_one condition state scaling requires condition_input_min/max"
                )
            if len(condition_min) != self.condition_dim or len(condition_max) != self.condition_dim:
                raise ValueError(
                    "condition_input_min/max must match Diag-CFM condition_dim"
                )
            if any(float(high) <= float(low) for low, high in zip(condition_min, condition_max)):
                raise ValueError("condition_input_max must exceed condition_input_min")
        self.condition_state_min = tuple(float(value) for value in (condition_min or []))
        self.condition_state_max = tuple(float(value) for value in (condition_max or []))

    def _design_to_state_space(
        self, normalized_design: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        design, log_abs_det, clipped_rate = self._to_model_space(normalized_design)
        if self.design_state_scaling == "zero_one":
            design = 0.5 * (design + 1.0)
        return design, log_abs_det, clipped_rate

    def _design_from_state_space(
        self, state_design: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        raw = state_design
        if self.design_state_scaling == "zero_one":
            raw = 2.0 * raw - 1.0
        return self._from_model_space(raw)

    def _condition_to_state_space(self, condition: torch.Tensor) -> torch.Tensor:
        if self.condition_state_scaling == "identity":
            return condition
        lower = condition.new_tensor(self.condition_state_min)
        upper = condition.new_tensor(self.condition_state_max)
        return (condition - lower) / (upper - lower)

    def _condition_from_state_space(self, condition: torch.Tensor) -> torch.Tensor:
        if self.condition_state_scaling == "identity":
            return condition
        lower = condition.new_tensor(self.condition_state_min)
        upper = condition.new_tensor(self.condition_state_max)
        return lower + condition * (upper - lower)

    def _sample_latent(
        self,
        shape: tuple[int, ...],
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.base_distribution == "standard_normal":
            return torch.randn(
                shape,
                generator=generator,
                device=self.device,
                dtype=self.dtype,
            )
        unit = torch.rand(
            shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        return self.base_low + (self.base_high - self.base_low) * unit

    def _batch_loss(
        self, batch: Mapping[str, Any], training: bool
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        design, _, clipped_rate = self._design_to_state_space(batch["x"])
        condition = self._condition_to_state_space(batch["condition"])
        latent = self._sample_latent(tuple(design.shape))
        zeros = torch.zeros_like(condition)
        source = torch.cat([latent, condition], dim=-1)
        target = torch.cat([design, zeros], dim=-1)
        t = torch.rand(
            design.shape[0], 1, device=self.device, dtype=self.dtype
        )
        state_t = (1.0 - t) * source + t * target
        target_velocity = target - source
        prediction = self.velocity(t, state_t)
        fm_loss = F.mse_loss(prediction, target_velocity)
        design_loss = F.mse_loss(
            prediction[:, : self.design_dim],
            target_velocity[:, : self.design_dim],
        )
        anchor_loss = F.mse_loss(
            prediction[:, self.design_dim :],
            target_velocity[:, self.design_dim :],
        )
        return fm_loss, {
            "flow_matching_loss": fm_loss,
            "design_velocity_loss": design_loss,
            "zero_anchor_velocity_loss": anchor_loss,
            "bounded_input_clip_rate": clipped_rate.mean(),
        }

    def _ode_solve(
        self,
        initial: torch.Tensor,
        start_t: float,
        end_t: float,
    ) -> Tuple[torch.Tensor, int, float]:
        nfe = 0

        def vector_field(t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            nonlocal nfe
            nfe += 1
            return self.velocity(t, state)

        steps = int(self.model_config.get("ode_steps", 30))
        method = str(self.model_config.get("ode_method", "euler"))
        if steps <= 0:
            raise ValueError("Diag-CFM ode_steps must be positive")
        supported = {"euler", "midpoint", "rk4", "dopri5"}
        if method not in supported:
            raise ValueError(f"Diag-CFM ode_method must be one of {sorted(supported)}")
        started = time.perf_counter()
        trajectory = odeint(
            vector_field,
            initial,
            torch.tensor(
                [start_t, end_t], device=self.device, dtype=self.dtype
            ),
            method=method,
            options={"step_size": abs(end_t - start_t) / steps}
            if method in {"euler", "midpoint", "rk4"}
            else None,
            rtol=float(self.model_config.get("ode_rtol", 1.0e-5)),
            atol=float(self.model_config.get("ode_atol", 1.0e-5)),
        )
        return trajectory[-1], nfe, time.perf_counter() - started

    def predict_conditions(self, normalized_design: Any) -> torch.Tensor:
        """Integrate ``[x, 0]`` backward and return normalized label predictions."""

        design = torch.as_tensor(
            normalized_design, device=self.device, dtype=self.dtype
        )
        single = design.ndim == 1
        if single:
            design = design[None, :]
        if design.ndim != 2 or design.shape[1] != self.design_dim:
            raise ValueError(
                "normalized_design must have shape "
                f"[{self.design_dim}] or [batch,{self.design_dim}]"
            )
        design_model, _, _ = self._design_to_state_space(design)
        endpoint = torch.cat(
            [
                design_model,
                torch.zeros(
                    design.shape[0],
                    self.condition_dim,
                    device=self.device,
                    dtype=self.dtype,
                ),
            ],
            dim=-1,
        )
        self.eval()
        with torch.no_grad():
            source, _, _ = self._ode_solve(endpoint, 1.0, 0.0)
        labels = self._condition_from_state_space(source[:, self.design_dim :])
        return labels[0] if single else labels

    def _sample_normalized(
        self, condition: torch.Tensor, n_samples: int, seed: Optional[int]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(int(seed))
        batch = condition.shape[0]
        flat_count = batch * n_samples
        latent = self._sample_latent(
            (flat_count, self.design_dim), generator=generator
        )
        state_condition = self._condition_to_state_space(condition)
        expanded_condition = (
            state_condition[:, None, :]
            .expand(-1, n_samples, -1)
            .reshape(flat_count, self.condition_dim)
        )
        initial = torch.cat([latent, expanded_condition], dim=-1)
        augmented, generation_nfe, generation_seconds = self._ode_solve(
            initial, 0.0, 1.0
        )
        raw_design = augmented[:, : self.design_dim]
        zero_scores = augmented[:, self.design_dim :].square().sum(dim=-1)
        samples, _, bounded_diagnostics = self._design_from_state_space(raw_design)

        self_consistency_scores: Optional[torch.Tensor] = None
        diagnosis_nfe = 0
        diagnosis_seconds = 0.0
        if bool(self.model_config.get("self_consistency_diagnostics", True)):
            design_model, _, _ = self._design_to_state_space(samples)
            exact_zero_endpoint = torch.cat(
                [
                    design_model,
                    torch.zeros(
                        flat_count,
                        self.condition_dim,
                        device=self.device,
                        dtype=self.dtype,
                    ),
                ],
                dim=-1,
            )
            reconstructed, diagnosis_nfe, diagnosis_seconds = self._ode_solve(
                exact_zero_endpoint, 1.0, 0.0
            )
            predicted_condition = reconstructed[:, self.design_dim :]
            self_consistency_scores = (
                predicted_condition - expanded_condition
            ).square().mean(dim=-1)

        samples = samples.reshape(batch, n_samples, self.design_dim)
        candidate_diagnostics: Dict[str, Any] = {
            "zero_deviation": zero_scores.detach().cpu().tolist()
        }
        diagnostics: Dict[str, Any] = {
            "sampler": "torchdiffeq.odeint",
            "ode_method": str(self.model_config.get("ode_method", "euler")),
            "ode_steps": int(self.model_config.get("ode_steps", 30)),
            "nfe": generation_nfe,
            "generation_nfe": generation_nfe,
            "generation_wall_seconds": generation_seconds,
            "zero_deviation_mean": float(zero_scores.mean().detach().cpu()),
            "zero_deviation_max": float(zero_scores.max().detach().cpu()),
            "base_distribution": self.base_distribution,
            "design_state_scaling": self.design_state_scaling,
            "condition_state_scaling": self.condition_state_scaling,
            "candidate_diagnostics": candidate_diagnostics,
            **bounded_diagnostics,
        }
        if self_consistency_scores is not None:
            candidate_diagnostics["self_consistency"] = (
                self_consistency_scores.detach().cpu().tolist()
            )
            diagnostics.update(
                {
                    "self_consistency_mean": float(
                        self_consistency_scores.mean().detach().cpu()
                    ),
                    "self_consistency_max": float(
                        self_consistency_scores.max().detach().cpu()
                    ),
                    "diagnosis_nfe": diagnosis_nfe,
                    "diagnosis_wall_seconds": diagnosis_seconds,
                }
            )
        diagnostics["wall_seconds"] = generation_seconds + diagnosis_seconds
        return samples, diagnostics
