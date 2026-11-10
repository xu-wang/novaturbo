"""Thin conditional-density adapter around the pinned nflows library."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch.nn import functional as F

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import RandomPermutation

from .base import ConditionalGenerator, register_generator
from ..generator_dataset import GeneratorNormalizer


@register_generator
class NFlowsGenerator(ConditionalGenerator):
    generator_name = "nflows"
    upstream_packages = ("nflows",)

    def __init__(
        self,
        model_config: Mapping[str, Any],
        runtime_config: Mapping[str, Any],
        schema_version: str = "generator_v1",
        normalizer: Optional[GeneratorNormalizer] = None,
    ) -> None:
        super().__init__(model_config, runtime_config, schema_version, normalizer)
        transforms = []
        for _ in range(int(model_config.get("num_layers", 4))):
            transforms.extend(
                [
                    RandomPermutation(features=self.design_dim),
                    MaskedAffineAutoregressiveTransform(
                        features=self.design_dim,
                        hidden_features=int(model_config.get("hidden_features", 64)),
                        context_features=self.condition_dim,
                        num_blocks=int(model_config.get("num_blocks", 2)),
                        use_residual_blocks=True,
                        random_mask=False,
                        activation=F.relu,
                        dropout_probability=0.0,
                        use_batch_norm=False,
                    ),
                ]
            )
        self.flow = Flow(
            CompositeTransform(transforms),
            StandardNormal(shape=[self.design_dim]),
        )

    def _batch_loss(
        self, batch: Mapping[str, Any], training: bool
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        inputs, transform_log_det, clipped_rate = self._to_model_space(batch["x"])
        context = self._condition_for_model(batch["condition"])
        log_probability = (
            self.flow.log_prob(inputs=inputs, context=context)
            + transform_log_det
        )
        nll = -log_probability.mean()
        return nll, {
            "negative_log_likelihood": nll,
            "bounded_input_clip_rate": clipped_rate.mean(),
        }

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
        expanded = self._condition_for_model(expanded)
        decoded, _ = self.flow._transform.inverse(
            value.reshape(-1, self.design_dim), context=expanded
        )
        decoded, _, _ = self._from_model_space(decoded)
        return decoded.reshape(value.shape[0], value.shape[1], self.design_dim)

    def _sample_normalized(
        self, condition: torch.Tensor, n_samples: int, seed: Optional[int]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        model_condition = self._condition_for_model(condition)
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if seed is not None:
                torch.manual_seed(int(seed))
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(int(seed))
            raw = self.flow.sample(n_samples, context=model_condition)
        if raw.ndim == 2:
            raw = raw.unsqueeze(0)
        samples, _, bounded_diagnostics = self._from_model_space(raw)
        return samples, {
            "sampler": "nflows.Flow.sample",
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
        if x_tensor.shape != (condition_tensor.shape[0], self.design_dim):
            raise ValueError("x and condition batch shapes are incompatible")
        self.eval()
        with torch.no_grad():
            inputs, transform_log_det, _ = self._to_model_space(x_tensor)
            condition_tensor = self._condition_for_model(condition_tensor)
            return (
                self.flow.log_prob(inputs, context=condition_tensor)
                + transform_log_det
            ).cpu()
