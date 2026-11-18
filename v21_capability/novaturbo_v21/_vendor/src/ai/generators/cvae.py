"""Low-dimensional MLP adaptation of the AntixK ConditionalVAE baseline."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .base import ConditionalGenerator, register_generator
from ..generator_dataset import GeneratorNormalizer


def _mlp(sizes, final_activation=False):
    layers = []
    for index in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2 or final_activation:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


@register_generator
class CVAEGenerator(ConditionalGenerator):
    """Reference ELBO/reparameterization with only CNN-to-MLP adaptation."""

    generator_name = "cvae"
    upstream_packages = ()

    def __init__(
        self,
        model_config: Mapping[str, Any],
        runtime_config: Mapping[str, Any],
        schema_version: str = "generator_v1",
        normalizer: Optional[GeneratorNormalizer] = None,
    ) -> None:
        super().__init__(model_config, runtime_config, schema_version, normalizer)
        hidden = [int(value) for value in model_config.get("hidden_dims", [128, 128])]
        latent_dim = int(model_config.get("latent_dim", 8))
        self.latent_dim = latent_dim
        self.encoder = _mlp(
            [self.design_dim + self.condition_dim, *hidden],
            final_activation=True,
        )
        self.mu = nn.Linear(hidden[-1], latent_dim)
        self.log_var = nn.Linear(hidden[-1], latent_dim)
        self.decoder = _mlp(
            [latent_dim + self.condition_dim, *reversed(hidden), self.design_dim]
        )

    def encode(self, x: torch.Tensor, condition: torch.Tensor):
        hidden = self.encoder(torch.cat([x, condition], dim=-1))
        return self.mu(hidden), self.log_var(hidden)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def decode_raw(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, condition], dim=-1))

    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        raw = self.decode_raw(z, condition)
        output, _ = self.design_transform.to_bounded(raw)
        return output

    def _batch_loss(
        self, batch: Mapping[str, Any], training: bool
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = batch["x"]
        condition = batch["condition"]
        mu, log_var = self.encode(x, condition)
        z = self.reparameterize(mu, log_var) if training else mu
        reconstruction = self.decode(z, condition)
        reconstruction_loss = F.mse_loss(reconstruction, x)
        kl_loss = -0.5 * torch.mean(1.0 + log_var - mu.square() - log_var.exp())
        beta = float(self.model_config.get("beta", 1e-3))
        loss = reconstruction_loss + beta * kl_loss
        return loss, {
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def _sample_normalized(
        self, condition: torch.Tensor, n_samples: int, seed: Optional[int]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(int(seed))
        batch = condition.shape[0]
        z = torch.randn(
            batch,
            n_samples,
            self.latent_dim,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        expanded_condition = condition[:, None, :].expand(-1, n_samples, -1)
        raw = self.decode_raw(
            z.reshape(-1, self.latent_dim),
            expanded_condition.reshape(-1, self.condition_dim),
        ).reshape(batch, n_samples, self.design_dim)
        decoded, _, bounded_diagnostics = self._from_model_space(raw)
        return decoded, {
            "latent_dim": self.latent_dim,
            "sampler": "standard_normal",
            **bounded_diagnostics,
        }
