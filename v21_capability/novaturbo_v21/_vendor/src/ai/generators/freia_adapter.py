"""Thin conditional INN adapter around pinned FrEIA invertible blocks."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from .base import ConditionalGenerator, register_generator
from ..generator_dataset import GeneratorNormalizer


@register_generator
class CINNGenerator(ConditionalGenerator):
    generator_name = "cinn"
    upstream_packages = ("FrEIA",)

    def __init__(
        self,
        model_config: Mapping[str, Any],
        runtime_config: Mapping[str, Any],
        schema_version: str = "generator_v1",
        normalizer: Optional[GeneratorNormalizer] = None,
    ) -> None:
        super().__init__(model_config, runtime_config, schema_version, normalizer)
        hidden = int(model_config.get("hidden_features", 64))

        def subnet(input_dim, output_dim):
            return nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, output_dim),
            )

        self.inn = Ff.SequenceINN(self.design_dim)
        for _ in range(int(model_config.get("num_blocks", 6))):
            self.inn.append(
                Fm.AllInOneBlock,
                cond=0,
                cond_shape=(self.condition_dim,),
                subnet_constructor=subnet,
                affine_clamping=float(model_config.get("affine_clamping", 2.0)),
                permute_soft=False,
            )

    def _inn_forward(self, x: torch.Tensor, condition: torch.Tensor):
        condition = self._condition_for_model(condition)
        return self.inn(x, c=[condition], rev=False, jac=True)

    def _inn_inverse(self, z: torch.Tensor, condition: torch.Tensor):
        condition = self._condition_for_model(condition)
        return self.inn(z, c=[condition], rev=True, jac=True)

    def forward_transform(self, x: torch.Tensor, condition: torch.Tensor):
        model_x, transform_log_det, _ = self._to_model_space(x)
        z, inn_log_det = self._inn_forward(model_x, condition)
        return z, inn_log_det + transform_log_det

    def inverse_transform(self, z: torch.Tensor, condition: torch.Tensor):
        raw, inn_log_det = self._inn_inverse(z, condition)
        output, transform_log_det, _ = self._from_model_space(raw)
        return output, inn_log_det + transform_log_det

    def _batch_loss(
        self, batch: Mapping[str, Any], training: bool
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z, log_det = self.forward_transform(batch["x"], batch["condition"])
        nll = (0.5 * z.square().sum(dim=-1) - log_det).mean()
        return nll, {"negative_log_likelihood": nll}

    @property
    def supports_latent_repair(self) -> bool:
        return True

    def sample_latent(
        self, condition: Any, n_samples: int, seed: Optional[int] = None
    ) -> torch.Tensor:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        prepared, _ = self._prepare_condition(condition)
        prepared = prepared.to(device=self.device, dtype=self.dtype)
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(int(seed))
        return torch.randn(
            prepared.shape[0],
            int(n_samples),
            self.design_dim,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

    def decode_latent(self, latent: Any, condition: Any) -> torch.Tensor:
        prepared, _ = self._prepare_condition(condition)
        prepared = prepared.to(device=self.device, dtype=self.dtype)
        value = torch.as_tensor(latent, dtype=self.dtype, device=self.device)
        if value.ndim == 2 and prepared.shape[0] == 1:
            value = value.unsqueeze(0)
        if (
            value.ndim != 3
            or value.shape[0] != prepared.shape[0]
            or value.shape[2] != self.design_dim
        ):
            raise ValueError(
                "latent must have shape [batch,n_samples,design_dim]"
            )
        expanded = (
            prepared[:, None, :]
            .expand(-1, value.shape[1], -1)
            .reshape(-1, self.condition_dim)
        )
        decoded, _ = self.inverse_transform(
            value.reshape(-1, self.design_dim), expanded
        )
        return decoded.reshape(value.shape[0], value.shape[1], self.design_dim)

    def _sample_normalized(
        self, condition: torch.Tensor, n_samples: int, seed: Optional[int]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(int(seed))
        batch = condition.shape[0]
        z = torch.randn(
            batch * n_samples,
            self.design_dim,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        expanded = (
            condition[:, None, :]
            .expand(-1, n_samples, -1)
            .reshape(-1, self.condition_dim)
        )
        raw, _ = self._inn_inverse(z, expanded)
        samples, _, bounded_diagnostics = self._from_model_space(raw)
        return samples.reshape(batch, n_samples, self.design_dim), {
            "sampler": "FrEIA reverse",
            **self._condition_diagnostics(condition),
            **bounded_diagnostics,
        }

    def log_prob(self, x: Any, condition: Any) -> torch.Tensor:
        x_tensor = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        condition_tensor, _ = self._prepare_condition(condition)
        condition_tensor = condition_tensor.to(self.device)
        if x_tensor.ndim == 1:
            x_tensor = x_tensor.unsqueeze(0)
        if condition_tensor.shape[0] == 1 and x_tensor.shape[0] > 1:
            condition_tensor = condition_tensor.expand(x_tensor.shape[0], -1)
        self.eval()
        with torch.no_grad():
            z, log_det = self.forward_transform(x_tensor, condition_tensor)
            constant = 0.5 * self.design_dim * math.log(2.0 * math.pi)
            return (-0.5 * z.square().sum(dim=-1) - constant + log_det).cpu()
