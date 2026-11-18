"""Thin Conditional Flow Matching adapter using TorchCFM and torchdiffeq."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from .base import ConditionalGenerator, register_generator
from ..generator_dataset import GeneratorNormalizer


class SinusoidalTimeEmbedding(nn.Module):
    """Fixed continuous-time features with no trainable projection."""

    def __init__(
        self,
        embedding_dim: int = 32,
        max_period: float = 10000.0,
        time_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or embedding_dim % 2:
            raise ValueError("time_embedding_dim must be a positive even integer")
        if max_period <= 1.0 or time_scale <= 0.0:
            raise ValueError("time max_period and time_scale must be positive")
        half = embedding_dim // 2
        frequencies = torch.exp(
            -math.log(float(max_period))
            * torch.arange(half, dtype=torch.float32)
            / float(half)
        )
        self.embedding_dim = int(embedding_dim)
        self.max_period = float(max_period)
        self.time_scale = float(time_scale)
        self.register_buffer("frequencies", frequencies, persistent=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None]
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if t.ndim != 1:
            raise ValueError("time must be scalar, [batch], or [batch,1]")
        arguments = (
            t[:, None]
            * self.time_scale
            * self.frequencies[None, :].to(dtype=t.dtype, device=t.device)
        )
        return torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)


class ConditionalVelocityMLP(nn.Module):
    def __init__(
        self,
        hidden_dims,
        design_dim: int = 5,
        condition_dim: int = 2,
        *,
        time_embedding: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        time_config = dict(time_embedding or {})
        self.time_embedding = (
            SinusoidalTimeEmbedding(
                embedding_dim=int(time_config.get("dim", 32)),
                max_period=float(time_config.get("max_period", 10000.0)),
                time_scale=float(time_config.get("time_scale", 1000.0)),
            )
            if bool(time_config.get("enabled", False))
            else None
        )
        time_features = (
            self.time_embedding.embedding_dim if self.time_embedding is not None else 1
        )
        sizes = [
            design_dim + condition_dim + time_features,
            *[int(value) for value in hidden_dims],
            design_dim,
        ]
        layers = []
        for index in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[index], sizes[index + 1]))
            if index < len(sizes) - 2:
                layers.append(nn.SiLU())
        self.network = nn.Sequential(*layers)

    def forward(
        self, t: torch.Tensor, x: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        encoded_time = (
            self.time_embedding(t)
            if self.time_embedding is not None
            else t[:, None] if t.ndim == 1 else t
        )
        return self.network(torch.cat([x, encoded_time, condition], dim=-1))


class ResidualMLPBlock(nn.Module):
    """Two-layer residual block without condition modulation."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = self.linear_2(self.activation(self.linear_1(hidden)))
        return hidden + residual


class ConditionalResidualVelocityMLP(nn.Module):
    """CFM-3 input projection followed by fixed-width residual blocks."""

    def __init__(
        self,
        hidden_dim: int,
        residual_blocks: int,
        design_dim: int = 10,
        condition_dim: int = 3,
        *,
        time_embedding: Mapping[str, Any],
    ) -> None:
        super().__init__()
        time_config = dict(time_embedding)
        self.time_embedding = SinusoidalTimeEmbedding(
            embedding_dim=int(time_config.get("dim", 32)),
            max_period=float(time_config.get("max_period", 10000.0)),
            time_scale=float(time_config.get("time_scale", 1000.0)),
        )
        input_dim = design_dim + condition_dim + self.time_embedding.embedding_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_activation = nn.SiLU()
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(hidden_dim) for _ in range(residual_blocks)
        )
        self.output_projection = nn.Linear(hidden_dim, design_dim)

    def forward(
        self, t: torch.Tensor, x: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        encoded_time = self.time_embedding(t)
        hidden = self.input_activation(
            self.input_projection(torch.cat([x, encoded_time, condition], dim=-1))
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(hidden)


class FiLMResidualMLPBlock(nn.Module):
    """Residual block with explicit, identity-initialized FiLM modulation."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        delta_gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        conditioned = (1.0 + delta_gamma) * hidden + beta
        residual = self.linear_2(self.activation(self.linear_1(conditioned)))
        return hidden + residual


class ConditionalFiLMResidualVelocityMLP(nn.Module):
    """CFM-4 residual velocity field with explicit condition/time modulation."""

    def __init__(
        self,
        hidden_dim: int,
        residual_blocks: int,
        condition_encoder_dim: int,
        design_dim: int = 10,
        condition_dim: int = 3,
        *,
        time_embedding: Mapping[str, Any],
    ) -> None:
        super().__init__()
        time_config = dict(time_embedding)
        self.time_embedding = SinusoidalTimeEmbedding(
            embedding_dim=int(time_config.get("dim", 32)),
            max_period=float(time_config.get("max_period", 10000.0)),
            time_scale=float(time_config.get("time_scale", 1000.0)),
        )
        time_dim = self.time_embedding.embedding_dim
        self.input_projection = nn.Linear(
            design_dim + condition_dim + time_dim, hidden_dim
        )
        self.input_activation = nn.SiLU()
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim + time_dim, condition_encoder_dim),
            nn.SiLU(),
        )
        self.film_projection = nn.Linear(
            condition_encoder_dim, residual_blocks * 2 * hidden_dim
        )
        nn.init.zeros_(self.film_projection.weight)
        nn.init.zeros_(self.film_projection.bias)
        self.blocks = nn.ModuleList(
            FiLMResidualMLPBlock(hidden_dim) for _ in range(residual_blocks)
        )
        self.output_projection = nn.Linear(hidden_dim, design_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_blocks = int(residual_blocks)

    def forward(
        self, t: torch.Tensor, x: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        encoded_time = self.time_embedding(t)
        hidden = self.input_activation(
            self.input_projection(torch.cat([x, encoded_time, condition], dim=-1))
        )
        condition_features = self.condition_encoder(
            torch.cat([condition, encoded_time], dim=-1)
        )
        film = self.film_projection(condition_features).reshape(
            x.shape[0], self.residual_blocks, 2, self.hidden_dim
        )
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, film[:, index, 0], film[:, index, 1])
        return self.output_projection(hidden)


class GatedResidualMLPBlock(nn.Module):
    """GLU-style residual block with independent value and sigmoid gates."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.gate_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value = self.activation(self.value_projection(hidden))
        gate = torch.sigmoid(self.gate_projection(hidden))
        return hidden + self.output_projection(value * gate)


class ConditionalGatedVelocityMLP(nn.Module):
    """CFM-5 parameter-matched GLU-style residual velocity field."""

    def __init__(
        self,
        hidden_dim: int,
        gated_blocks: int,
        design_dim: int = 10,
        condition_dim: int = 3,
        *,
        time_embedding: Mapping[str, Any],
    ) -> None:
        super().__init__()
        time_config = dict(time_embedding)
        self.time_embedding = SinusoidalTimeEmbedding(
            embedding_dim=int(time_config.get("dim", 32)),
            max_period=float(time_config.get("max_period", 10000.0)),
            time_scale=float(time_config.get("time_scale", 1000.0)),
        )
        input_dim = design_dim + condition_dim + self.time_embedding.embedding_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_activation = nn.SiLU()
        self.blocks = nn.ModuleList(
            GatedResidualMLPBlock(hidden_dim) for _ in range(gated_blocks)
        )
        self.output_projection = nn.Linear(hidden_dim, design_dim)

    def forward(
        self, t: torch.Tensor, x: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        encoded_time = self.time_embedding(t)
        hidden = self.input_activation(
            self.input_projection(torch.cat([x, encoded_time, condition], dim=-1))
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(hidden)


class MLPUNetDecoderStage(nn.Module):
    """Dimension reduction followed by a fixed concatenation skip fusion."""

    def __init__(self, input_dim: int, skip_dim: int) -> None:
        super().__init__()
        self.up_projection = nn.Linear(input_dim, skip_dim)
        self.fusion = nn.Linear(2 * skip_dim, skip_dim)
        self.activation = nn.SiLU()

    def forward(self, hidden: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        reduced = self.activation(self.up_projection(hidden))
        return self.activation(self.fusion(torch.cat([reduced, skip], dim=-1)))


class ConditionalMLPUNetVelocity(nn.Module):
    """CFM-6 low-dimensional encoder-decoder MLP with concatenation skips."""

    def __init__(
        self,
        encoder_dims: list[int],
        bottleneck_blocks: int,
        design_dim: int = 10,
        condition_dim: int = 3,
        *,
        time_embedding: Mapping[str, Any],
    ) -> None:
        super().__init__()
        time_config = dict(time_embedding)
        self.time_embedding = SinusoidalTimeEmbedding(
            embedding_dim=int(time_config.get("dim", 32)),
            max_period=float(time_config.get("max_period", 10000.0)),
            time_scale=float(time_config.get("time_scale", 1000.0)),
        )
        input_dim = design_dim + condition_dim + self.time_embedding.embedding_dim
        self.input_projection = nn.Linear(input_dim, encoder_dims[0])
        self.input_activation = nn.SiLU()
        self.encoder_layers = nn.ModuleList(
            nn.Linear(source, target)
            for source, target in zip(encoder_dims[:-1], encoder_dims[1:])
        )
        self.encoder_activation = nn.SiLU()
        self.bottleneck = nn.ModuleList(
            ResidualMLPBlock(encoder_dims[-1])
            for _ in range(bottleneck_blocks)
        )
        self.decoder_stages = nn.ModuleList(
            MLPUNetDecoderStage(source, skip)
            for source, skip in zip(
                reversed(encoder_dims[1:]), reversed(encoder_dims[:-1])
            )
        )
        self.output_projection = nn.Linear(encoder_dims[0], design_dim)

    def forward(
        self, t: torch.Tensor, x: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        encoded_time = self.time_embedding(t)
        hidden = self.input_activation(
            self.input_projection(torch.cat([x, encoded_time, condition], dim=-1))
        )
        skips = [hidden]
        for layer in self.encoder_layers:
            hidden = self.encoder_activation(layer(hidden))
            skips.append(hidden)
        for block in self.bottleneck:
            hidden = block(hidden)
        for stage, skip in zip(self.decoder_stages, reversed(skips[:-1])):
            hidden = stage(hidden, skip)
        return self.output_projection(hidden)


class ConditionalAttentionVelocity(nn.Module):
    """CFM-7 lightweight self-attention over scalar design-variable tokens."""

    def __init__(
        self,
        model_dim: int,
        attention_heads: int,
        transformer_layers: int,
        feedforward_dim: int,
        design_dim: int = 10,
        condition_dim: int = 3,
        *,
        time_embedding: Mapping[str, Any],
    ) -> None:
        super().__init__()
        time_config = dict(time_embedding)
        self.time_embedding = SinusoidalTimeEmbedding(
            embedding_dim=int(time_config.get("dim", 32)),
            max_period=float(time_config.get("max_period", 10000.0)),
            time_scale=float(time_config.get("time_scale", 1000.0)),
        )
        self.design_value_projection = nn.Linear(1, model_dim)
        self.design_identity = nn.Parameter(torch.empty(design_dim, model_dim))
        self.condition_projection = nn.Linear(condition_dim, model_dim)
        self.time_projection = nn.Linear(self.time_embedding.embedding_dim, model_dim)
        self.token_type_embedding = nn.Parameter(torch.empty(3, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.velocity_head = nn.Linear(model_dim, 1)
        self.design_dim = int(design_dim)
        nn.init.normal_(self.design_identity, mean=0.0, std=0.02)
        nn.init.normal_(self.token_type_embedding, mean=0.0, std=0.02)

    def forward(
        self, t: torch.Tensor, x: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        design_tokens = self.design_value_projection(x.unsqueeze(-1))
        design_tokens = (
            design_tokens
            + self.design_identity[None, :, :]
            + self.token_type_embedding[0][None, None, :]
        )
        condition_token = self.condition_projection(condition)
        condition_token = condition_token + self.token_type_embedding[1][None, :]
        time_token = self.time_projection(self.time_embedding(t))
        time_token = time_token + self.token_type_embedding[2][None, :]
        tokens = torch.cat(
            [design_tokens, condition_token[:, None, :], time_token[:, None, :]],
            dim=1,
        )
        encoded = self.encoder(tokens)
        return self.velocity_head(encoded[:, : self.design_dim]).squeeze(-1)


class GPUSinkhornConditionalFlowMatcher(ConditionalFlowMatcher):
    """Entropic minibatch OT-CFM coupling computed entirely with PyTorch."""

    def __init__(
        self,
        sigma: float = 0.0,
        regularization: float = 0.1,
        iterations: int = 50,
    ) -> None:
        super().__init__(sigma)
        if regularization <= 0.0 or iterations <= 0:
            raise ValueError("Sinkhorn regularization and iterations must be positive")
        self.regularization = float(regularization)
        self.iterations = int(iterations)

    def transport_plan(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Return a differentiability-free log-domain entropic OT plan on x0.device."""
        if x0.shape[0] != x1.shape[0]:
            raise ValueError("GPU Sinkhorn OT requires equal source and target batch sizes")
        with torch.no_grad():
            source = x0.detach().flatten(start_dim=1).float()
            target = x1.detach().flatten(start_dim=1).float()
            cost = torch.cdist(source, target).square()
            cost = cost / cost.mean().clamp_min(torch.finfo(cost.dtype).eps)
            log_kernel = -cost / self.regularization
            count = source.shape[0]
            log_marginal = torch.full(
                (count,),
                -math.log(float(count)),
                dtype=log_kernel.dtype,
                device=log_kernel.device,
            )
            log_u = torch.zeros_like(log_marginal)
            log_v = torch.zeros_like(log_marginal)
            for _ in range(self.iterations):
                log_u = log_marginal - torch.logsumexp(
                    log_kernel + log_v[None, :], dim=1
                )
                log_v = log_marginal - torch.logsumexp(
                    log_kernel + log_u[:, None], dim=0
                )
            plan = torch.exp(log_kernel + log_u[:, None] + log_v[None, :])
            plan = plan / plan.sum()
            if not torch.isfinite(plan).all() or float(plan.sum()) <= 0.0:
                raise FloatingPointError("GPU Sinkhorn produced a non-finite OT plan")
            return plan.to(dtype=x0.dtype)

    def guided_sample_location_and_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        y0: Optional[torch.Tensor] = None,
        y1: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        return_noise: bool = False,
    ):
        plan = self.transport_plan(x0, x1)
        count = x0.shape[0]
        choices = torch.multinomial(plan.flatten(), count, replacement=True)
        source_index = torch.div(choices, count, rounding_mode="floor")
        target_index = choices.remainder(count)
        paired_x0 = x0[source_index]
        paired_x1 = x1[target_index]
        values = super().sample_location_and_conditional_flow(
            paired_x0, paired_x1, t=t, return_noise=return_noise
        )
        paired_y0 = y0[source_index] if y0 is not None else None
        paired_y1 = y1[target_index] if y1 is not None else None
        if return_noise:
            sampled_t, location, velocity, noise = values
            return sampled_t, location, velocity, paired_y0, paired_y1, noise
        sampled_t, location, velocity = values
        return sampled_t, location, velocity, paired_y0, paired_y1


@register_generator
class CFMGenerator(ConditionalGenerator):
    generator_name = "cfm"
    upstream_packages = ("torchcfm", "torchdiffeq")

    def __init__(
        self,
        model_config: Mapping[str, Any],
        runtime_config: Mapping[str, Any],
        schema_version: str = "generator_v1",
        normalizer: Optional[GeneratorNormalizer] = None,
    ) -> None:
        super().__init__(model_config, runtime_config, schema_version, normalizer)
        self.architecture = str(model_config.get("architecture", "baseline"))
        if self.architecture not in {
            "baseline",
            "time_embedding",
            "wide_deep",
            "residual",
            "film_residual",
            "gated",
            "mlp_unet",
            "attention",
        }:
            raise ValueError(
                "Only stage-gated baseline, time_embedding, wide_deep, residual, "
                "film_residual, gated, mlp_unet, and attention CFM architectures are "
                "currently available"
            )
        time_embedding = dict(model_config.get("time_embedding", {}))
        expected_time_embedding = self.architecture in {
            "time_embedding",
            "wide_deep",
            "residual",
            "film_residual",
            "gated",
            "mlp_unet",
            "attention",
        }
        if bool(time_embedding.get("enabled", False)) != expected_time_embedding:
            raise ValueError(
                "CFM architecture and time_embedding.enabled are inconsistent"
            )
        hidden_dims = [int(value) for value in model_config.get("hidden_dims", [128] * 3)]
        if self.architecture == "wide_deep" and hidden_dims != [256] * 4:
            raise ValueError("CFM wide_deep architecture requires hidden_dims=[256]*4")
        if self.architecture == "attention":
            model_dim = int(model_config.get("model_dim", 128))
            attention_heads = int(model_config.get("attention_heads", 4))
            transformer_layers = int(model_config.get("transformer_layers", 2))
            feedforward_dim = int(model_config.get("feedforward_dim", 256))
            tokenization = str(model_config.get("tokenization", "design_condition_time"))
            if (
                model_dim != 128
                or attention_heads != 4
                or transformer_layers != 2
                or feedforward_dim != 256
                or tokenization != "design_condition_time"
            ):
                raise ValueError(
                    "CFM attention architecture requires model_dim=128, "
                    "attention_heads=4, transformer_layers=2, feedforward_dim=256, "
                    "and tokenization=design_condition_time"
                )
            self.velocity = ConditionalAttentionVelocity(
                model_dim,
                attention_heads,
                transformer_layers,
                feedforward_dim,
                self.design_dim,
                self.condition_dim,
                time_embedding=time_embedding,
            )
        elif self.architecture == "mlp_unet":
            encoder_dims = [
                int(value)
                for value in model_config.get("encoder_dims", [32, 64, 128, 256])
            ]
            bottleneck_blocks = int(model_config.get("bottleneck_blocks", 1))
            skip_mode = str(model_config.get("skip_mode", "concat"))
            if (
                encoder_dims not in ([32, 64, 128, 256], [64, 128, 256, 512])
                or bottleneck_blocks != 1
                or skip_mode != "concat"
            ):
                raise ValueError(
                    "CFM mlp_unet architecture requires a frozen encoder_dims variant "
                    "([32,64,128,256] or [64,128,256,512]), bottleneck_blocks=1, "
                    "and skip_mode=concat"
                )
            self.velocity = ConditionalMLPUNetVelocity(
                encoder_dims,
                bottleneck_blocks,
                self.design_dim,
                self.condition_dim,
                time_embedding=time_embedding,
            )
        elif self.architecture in {"residual", "film_residual", "gated"}:
            hidden_dim = int(model_config.get("hidden_dim", 256))
            residual_blocks = int(model_config.get("residual_blocks", 2))
            if hidden_dim != 256 or residual_blocks != 2:
                raise ValueError(
                    "CFM residual architectures require hidden_dim=256 and "
                    "residual_blocks=2"
                )
            if self.architecture == "gated":
                gated_blocks = int(model_config.get("gated_blocks", 2))
                if gated_blocks != 2:
                    raise ValueError(
                        "CFM gated architecture requires hidden_dim=256 and "
                        "gated_blocks=2"
                    )
                self.velocity = ConditionalGatedVelocityMLP(
                    hidden_dim,
                    gated_blocks,
                    self.design_dim,
                    self.condition_dim,
                    time_embedding=time_embedding,
                )
            elif self.architecture == "film_residual":
                condition_encoder_dim = int(
                    model_config.get("condition_encoder_dim", 128)
                )
                if condition_encoder_dim != 128:
                    raise ValueError(
                        "CFM film_residual architecture requires "
                        "condition_encoder_dim=128"
                    )
                self.velocity = ConditionalFiLMResidualVelocityMLP(
                    hidden_dim,
                    residual_blocks,
                    condition_encoder_dim,
                    self.design_dim,
                    self.condition_dim,
                    time_embedding=time_embedding,
                )
            else:
                self.velocity = ConditionalResidualVelocityMLP(
                    hidden_dim,
                    residual_blocks,
                    self.design_dim,
                    self.condition_dim,
                    time_embedding=time_embedding,
                )
        else:
            self.velocity = ConditionalVelocityMLP(
                hidden_dims,
                self.design_dim,
                self.condition_dim,
                time_embedding=time_embedding,
            )
        self.matcher_type = str(model_config.get("matcher", "plain"))
        sigma = float(model_config.get("sigma", 0.0))
        if self.matcher_type == "plain":
            self.matcher = ConditionalFlowMatcher(sigma=sigma)
        elif self.matcher_type == "gpu_sinkhorn":
            cost_normalization = str(
                model_config.get("sinkhorn_cost_normalization", "batch_mean")
            )
            regularization = float(model_config.get("sinkhorn_regularization", 0.1))
            iterations = int(model_config.get("sinkhorn_iterations", 50))
            if (
                cost_normalization != "batch_mean"
                or regularization != 0.1
                or iterations != 50
            ):
                raise ValueError(
                    "S2-WP4B GPU Sinkhorn requires batch-mean cost normalization, "
                    "sinkhorn_regularization=0.1, and sinkhorn_iterations=50"
                )
            self.matcher = GPUSinkhornConditionalFlowMatcher(
                sigma=sigma,
                regularization=regularization,
                iterations=iterations,
            )
        else:
            raise ValueError("CFM matcher must be plain or gpu_sinkhorn")
        # Domain-specific auxiliary objectives are training-only.  Bypass
        # nn.Module registration so frozen evaluator weights are neither
        # optimized nor serialized into the generator checkpoint.
        object.__setattr__(self, "_training_auxiliary_loss", None)

    def set_training_auxiliary_loss(
        self,
        loss_function: Callable[
            [torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        ]
        | None,
    ) -> None:
        """Attach a differentiable training-only endpoint objective."""

        object.__setattr__(self, "_training_auxiliary_loss", loss_function)

    def _batch_loss(
        self, batch: Mapping[str, Any], training: bool
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        target, _, clipped_rate = self._to_model_space(batch["x"])
        source = torch.randn_like(target)
        if self.matcher_type == "gpu_sinkhorn":
            (
                t,
                location,
                target_velocity,
                _,
                paired_condition,
            ) = self.matcher.guided_sample_location_and_conditional_flow(
                source, target, y1=batch["condition"]
            )
        else:
            t, location, target_velocity = (
                self.matcher.sample_location_and_conditional_flow(source, target)
            )
            paired_condition = batch["condition"]
        prediction = self.velocity(t, location, paired_condition)
        fm_loss = F.mse_loss(prediction, target_velocity)
        components = {
            "flow_matching_loss": fm_loss,
            "bounded_input_clip_rate": clipped_rate.mean(),
        }
        auxiliary = self._training_auxiliary_loss
        if auxiliary is None:
            return fm_loss, components
        if float(self.model_config.get("sigma", 0.0)) != 0.0:
            raise ValueError("Endpoint auxiliary loss requires deterministic sigma=0 flow")
        endpoint_model_space = location + (1.0 - t.reshape(-1, 1)) * prediction
        endpoint_normalized, _, _ = self._from_model_space(endpoint_model_space)
        auxiliary_loss, auxiliary_components = auxiliary(
            endpoint_normalized, paired_condition
        )
        if auxiliary_loss.ndim != 0 or not torch.isfinite(auxiliary_loss):
            raise ValueError("CFM auxiliary loss must be one finite scalar")
        components.update(auxiliary_components)
        return fm_loss + auxiliary_loss, components

    def _sample_normalized(
        self, condition: torch.Tensor, n_samples: int, seed: Optional[int]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(int(seed))
        batch = condition.shape[0]
        initial = torch.randn(
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
        nfe = 0

        def vector_field(t, state):
            nonlocal nfe
            nfe += 1
            return self.velocity(t, state, expanded)

        steps = int(self.model_config.get("ode_steps", 32))
        method = str(self.model_config.get("ode_method", "rk4"))
        started = time.perf_counter()
        trajectory = odeint(
            vector_field,
            initial,
            torch.tensor([0.0, 1.0], device=self.device, dtype=self.dtype),
            method=method,
            options={"step_size": 1.0 / steps} if method == "rk4" else None,
            rtol=float(self.model_config.get("ode_rtol", 1e-5)),
            atol=float(self.model_config.get("ode_atol", 1e-5)),
        )
        raw = trajectory[-1].reshape(batch, n_samples, self.design_dim)
        samples, _, bounded_diagnostics = self._from_model_space(raw)
        return samples, {
            "sampler": "torchdiffeq.odeint",
            "ode_method": method,
            "ode_steps": steps,
            "nfe": nfe,
            "wall_seconds": time.perf_counter() - started,
            **bounded_diagnostics,
        }
