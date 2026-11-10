"""Model-independent ConditionalGenerator interface and training shell."""

from __future__ import annotations

import abc
import copy
import datetime as dt
import importlib.metadata
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type

import numpy as np
import torch
from torch import nn

from ..generator_dataset import GeneratorNormalizer
from .bounded import SmoothBoundedTransform


GENERATOR_REGISTRY: Dict[str, Type["ConditionalGenerator"]] = {}


@dataclass(frozen=True)
class GeneratedSamples:
    values: torch.Tensor
    space: str
    condition: torch.Tensor
    seed: Optional[int]
    diagnostics: Dict[str, Any]


def register_generator(cls: Type["ConditionalGenerator"]):
    GENERATOR_REGISTRY[cls.generator_name] = cls
    return cls


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


class ConditionalGenerator(nn.Module, abc.ABC):
    """Uniform fit/sample/save/load contract for all WP3 adapters."""

    generator_name = "abstract"
    upstream_packages: Sequence[str] = ()

    def __init__(
        self,
        model_config: Mapping[str, Any],
        runtime_config: Mapping[str, Any],
        schema_version: str = "generator_v1",
        normalizer: Optional[GeneratorNormalizer] = None,
    ) -> None:
        super().__init__()
        _set_seed(int(runtime_config.get("seed", 42)))
        self.model_config = dict(model_config)
        self.runtime_config = dict(runtime_config)
        self.schema_version = str(schema_version)
        self.normalizer = normalizer
        self.design_dim = int(
            model_config.get(
                "design_dim", len(normalizer.design) if normalizer is not None else 5
            )
        )
        self.condition_dim = int(
            model_config.get(
                "condition_dim",
                len(normalizer.conditions) if normalizer is not None else 2,
            )
        )
        if self.design_dim <= 0 or self.condition_dim <= 0:
            raise ValueError("Generator dimensions must be positive")
        if normalizer is not None and (
            self.design_dim != len(normalizer.design)
            or self.condition_dim != len(normalizer.conditions)
        ):
            raise ValueError("Configured generator dimensions differ from normalizer")
        self.device = resolve_device(str(runtime_config.get("device", "auto")))
        self.dtype = torch.float32
        self.design_transform = SmoothBoundedTransform.from_config(
            model_config.get("bounded_output"), dimension=self.design_dim
        )
        stabilization = model_config.get("condition_stabilization", {})
        if stabilization in (None, False):
            stabilization = {}
        if stabilization is True:
            stabilization = {"enabled": True}
        if not isinstance(stabilization, Mapping):
            raise ValueError("condition_stabilization must be a boolean or mapping")
        self.condition_stabilization_enabled = bool(
            stabilization.get("enabled", False)
        )
        self.condition_stabilization_kind = str(
            stabilization.get("kind", "softclip")
        )
        self.condition_stabilization_limit = float(
            stabilization.get("limit", 3.0)
        )
        if self.condition_stabilization_kind != "softclip":
            raise ValueError("condition_stabilization.kind must be softclip")
        if self.condition_stabilization_limit <= 0.0:
            raise ValueError("condition_stabilization.limit must be positive")
        self.training_metadata: Dict[str, Any] = {}
        self._last_state_dict: Optional[Dict[str, torch.Tensor]] = None

    @abc.abstractmethod
    def _batch_loss(
        self, batch: Mapping[str, Any], training: bool
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    @abc.abstractmethod
    def _sample_normalized(
        self, condition: torch.Tensor, n_samples: int, seed: Optional[int]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError

    def _move_batch(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            key: value.to(device=self.device, dtype=self.dtype)
            if torch.is_tensor(value)
            else value
            for key, value in batch.items()
        }

    def _condition_for_model(self, condition: torch.Tensor) -> torch.Tensor:
        if not self.condition_stabilization_enabled:
            return condition
        limit = self.condition_stabilization_limit
        return limit * torch.tanh(condition / limit)

    def _condition_diagnostics(self, condition: torch.Tensor) -> Dict[str, Any]:
        transformed = self._condition_for_model(condition)
        return {
            "condition_stabilization_enabled": self.condition_stabilization_enabled,
            "condition_stabilization_kind": self.condition_stabilization_kind,
            "condition_stabilization_limit": self.condition_stabilization_limit,
            "condition_raw_max_abs": float(condition.detach().abs().max().cpu()),
            "condition_model_max_abs": float(transformed.detach().abs().max().cpu()),
        }

    @property
    def supports_latent_repair(self) -> bool:
        """Whether this adapter exposes a differentiable base-to-design map."""

        return False

    def sample_latent(
        self, condition: Any, n_samples: int, seed: Optional[int] = None
    ) -> torch.Tensor:
        """Draw base-space samples for a later differentiable decode."""

        raise NotImplementedError(
            f"{self.generator_name} does not expose a repairable latent space"
        )

    def decode_latent(self, latent: Any, condition: Any) -> torch.Tensor:
        """Decode base-space samples while preserving gradients on ``latent``."""

        raise NotImplementedError(
            f"{self.generator_name} does not expose a repairable latent space"
        )

    def _to_model_space(
        self, bounded: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.design_transform.to_unconstrained(bounded)

    def _from_model_space(
        self, raw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        output, log_abs_det = self.design_transform.to_bounded(raw)
        diagnostics = self.design_transform.sample_diagnostics(raw, output)
        return output, log_abs_det, diagnostics

    def _run_epoch(self, loader, optimizer=None) -> Dict[str, float]:
        training = optimizer is not None
        self.train(training)
        sums: Dict[str, float] = {}
        count = 0
        for batch in loader:
            moved = self._move_batch(batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                loss, components = self._batch_loss(moved, training=training)
                if not torch.isfinite(loss):
                    sample_ids = list(batch.get("sample_id", []))[:8]
                    raise FloatingPointError(
                        f"{self.generator_name} produced non-finite loss; "
                        f"sample_ids={sample_ids}"
                    )
                if training:
                    loss.backward()
                    max_norm = float(self.runtime_config.get("gradient_clip_norm", 0.0))
                    if max_norm > 0:
                        nn.utils.clip_grad_norm_(self.parameters(), max_norm)
                    optimizer.step()
            batch_size = int(moved["x"].shape[0])
            count += batch_size
            values = {"loss": loss.detach(), **components}
            for name, value in values.items():
                scalar = float(value.detach().cpu())
                sums[name] = sums.get(name, 0.0) + scalar * batch_size
        if count == 0:
            raise ValueError("Generator training received an empty DataLoader")
        return {name: value / count for name, value in sums.items()}

    def fit(self, train_loader, val_loader) -> Dict[str, Any]:
        seed = int(self.runtime_config.get("seed", 42))
        _set_seed(seed)
        self.to(self.device)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(self.model_config.get("learning_rate", 1e-3)),
            weight_decay=float(self.model_config.get("weight_decay", 0.0)),
        )
        epochs = int(self.runtime_config.get("epochs", 10))
        patience = int(self.runtime_config.get("patience", epochs))
        minimum_epochs = int(self.runtime_config.get("minimum_epochs", 1))
        min_delta = float(self.runtime_config.get("min_delta", 0.0))
        if epochs <= 0 or patience <= 0 or minimum_epochs <= 0:
            raise ValueError("epochs, patience, and minimum_epochs must be positive")
        if minimum_epochs > epochs:
            raise ValueError("minimum_epochs cannot exceed epochs")
        if min_delta < 0.0:
            raise ValueError("min_delta must be non-negative")
        scheduler_config = dict(self.runtime_config.get("lr_scheduler", {}))
        scheduler = None
        if bool(scheduler_config.get("enabled", False)):
            scheduler_patience = int(scheduler_config.get("patience", 5))
            scheduler_factor = float(scheduler_config.get("factor", 0.5))
            scheduler_min_lr = float(scheduler_config.get("min_lr", 1.0e-6))
            if scheduler_patience < 0:
                raise ValueError("lr_scheduler.patience must be non-negative")
            if not 0.0 < scheduler_factor < 1.0:
                raise ValueError("lr_scheduler.factor must be between zero and one")
            if scheduler_min_lr < 0.0:
                raise ValueError("lr_scheduler.min_lr must be non-negative")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_factor,
                patience=scheduler_patience,
                min_lr=scheduler_min_lr,
            )
        best_loss = math.inf
        best_epoch = -1
        best_state = None
        history = []
        stale_epochs = 0
        snapshot_every = int(self.runtime_config.get("snapshot_every", 0))
        snapshot_directory_value = self.runtime_config.get("snapshot_directory")
        snapshot_directory = (
            Path(str(snapshot_directory_value))
            if snapshot_directory_value is not None
            else None
        )
        if snapshot_every < 0 or (snapshot_every > 0 and snapshot_directory is None):
            raise ValueError("snapshot_every requires a non-negative interval and snapshot_directory")
        if snapshot_directory is not None:
            snapshot_directory.mkdir(parents=True, exist_ok=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            train_metrics = self._run_epoch(train_loader, optimizer=optimizer)
            with torch.no_grad():
                val_metrics = self._run_epoch(val_loader, optimizer=None)
            learning_rate_before = float(optimizer.param_groups[0]["lr"])
            validation_loss = float(val_metrics["loss"])
            if scheduler is not None:
                scheduler.step(validation_loss)
            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "learning_rate": learning_rate_before,
                "learning_rate_after_scheduler": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.perf_counter() - epoch_started,
            }
            history.append(record)
            if validation_loss < best_loss - min_delta:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if snapshot_every > 0 and epoch % snapshot_every == 0:
                snapshot_path = snapshot_directory / f"epoch_{epoch:04d}.pt"
                payload = self._checkpoint_payload("best")
                payload["checkpoint_state"] = "epoch_snapshot"
                payload["snapshot_epoch"] = epoch
                payload["training_metadata"] = {
                    "snapshot_epoch": epoch,
                    "validation_loss": validation_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                temporary = snapshot_path.with_suffix(".pt.tmp")
                torch.save(payload, temporary)
                temporary.replace(snapshot_path)
            if epoch >= minimum_epochs and stale_epochs >= patience:
                break
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_seconds = time.perf_counter() - started
        peak_gpu_memory_bytes = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0
        )
        self._last_state_dict = copy.deepcopy(self.state_dict())
        if best_state is None:
            raise RuntimeError("No finite best checkpoint was selected")
        self.load_state_dict(best_state)
        self.training_metadata = {
            "generator_name": self.generator_name,
            "seed": seed,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "bounded_output": self.design_transform.to_config(),
            "condition_stabilization": {
                "enabled": self.condition_stabilization_enabled,
                "kind": self.condition_stabilization_kind,
                "limit": self.condition_stabilization_limit,
            },
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "epochs_completed": len(history),
            "early_stopped": len(history) < epochs,
            "termination_reason": (
                "early_stopping" if len(history) < epochs else "max_epochs"
            ),
            "maximum_epochs": epochs,
            "minimum_epochs": minimum_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "initial_learning_rate": float(
                self.model_config.get("learning_rate", 1.0e-3)
            ),
            "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "lr_scheduler": scheduler_config,
            "elapsed_seconds": elapsed_seconds,
            "average_epoch_seconds": float(
                np.mean([record["elapsed_seconds"] for record in history])
            ),
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
            "total_parameters": int(sum(value.numel() for value in self.parameters())),
            "trainable_parameters": int(
                sum(value.numel() for value in self.parameters() if value.requires_grad)
            ),
            "history": history,
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        return copy.deepcopy(self.training_metadata)

    def _prepare_condition(self, condition: Any) -> Tuple[torch.Tensor, bool]:
        tensor = torch.as_tensor(condition, dtype=torch.float32)
        single = tensor.ndim == 1
        if single:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != self.condition_dim:
            raise ValueError(
                "condition must have shape "
                f"[{self.condition_dim}] or [batch,{self.condition_dim}], "
                f"got {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError("condition contains non-finite values")
        return tensor, single

    def sample(
        self,
        condition: Any,
        n_samples: int,
        seed: Optional[int] = None,
        space: str = "normalized",
    ) -> GeneratedSamples:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if space not in {"normalized", "physical"}:
            raise ValueError("space must be normalized or physical")
        prepared, single = self._prepare_condition(condition)
        prepared = prepared.to(device=self.device, dtype=self.dtype)
        self.eval()
        with torch.no_grad():
            normalized, diagnostics = self._sample_normalized(
                prepared, int(n_samples), seed
            )
        expected = (prepared.shape[0], n_samples, self.design_dim)
        if tuple(normalized.shape) != expected:
            raise RuntimeError(
                f"{self.generator_name} returned {tuple(normalized.shape)}, expected {expected}"
            )
        if not torch.isfinite(normalized).all():
            raise FloatingPointError("Generator returned non-finite samples")
        values = normalized.detach().cpu()
        if space == "physical":
            if self.normalizer is None:
                raise RuntimeError("Physical sampling requires an attached normalizer")
            flat = values.reshape(-1, self.design_dim).numpy()
            physical = self.normalizer.inverse_design(flat)
            values = torch.as_tensor(physical, dtype=torch.float32).reshape_as(values)
        if single:
            values = values[0]
        return GeneratedSamples(
            values=values,
            space=space,
            condition=prepared.detach().cpu()[0] if single else prepared.detach().cpu(),
            seed=seed,
            diagnostics=dict(diagnostics),
        )

    def log_prob(self, x: Any, condition: Any) -> torch.Tensor:
        raise NotImplementedError(f"{self.generator_name} does not expose log_prob")

    def _upstream_versions(self) -> Dict[str, str]:
        versions = {}
        for package in self.upstream_packages:
            versions[package] = importlib.metadata.version(package)
        return versions

    def _checkpoint_payload(self, state: str) -> Dict[str, Any]:
        if state not in {"best", "last"}:
            raise ValueError("checkpoint state must be best or last")
        if state == "last":
            if self._last_state_dict is None:
                raise RuntimeError("No last state exists before fit")
            state_dict = self._last_state_dict
        else:
            state_dict = self.state_dict()
        return {
            "format_version": "conditional_generator_checkpoint_v1",
            "generator_name": self.generator_name,
            "checkpoint_state": state,
            "model_config": self.model_config,
            "runtime_config": self.runtime_config,
            "schema_version": self.schema_version,
            "normalizer": self.normalizer.to_dict() if self.normalizer else None,
            "training_metadata": self.training_metadata,
            "upstream_versions": self._upstream_versions(),
            "state_dict": state_dict,
        }

    def save(self, path: Path, state: str = "best") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint_payload(state), path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "ConditionalGenerator":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if payload.get("format_version") != "conditional_generator_checkpoint_v1":
            raise ValueError("Unsupported conditional generator checkpoint format")
        if payload.get("generator_name") != cls.generator_name:
            raise ValueError(
                f"Checkpoint is {payload.get('generator_name')}, not {cls.generator_name}"
            )
        normalizer_payload = payload.get("normalizer")
        normalizer = (
            GeneratorNormalizer.from_dict(normalizer_payload)
            if normalizer_payload is not None
            else None
        )
        runtime = dict(payload["runtime_config"])
        runtime["device"] = device
        model = cls(
            payload["model_config"],
            runtime,
            schema_version=payload["schema_version"],
            normalizer=normalizer,
        )
        installed = model._upstream_versions()
        if installed != payload.get("upstream_versions", {}):
            raise RuntimeError(
                f"Upstream version mismatch: installed={installed}, "
                f"checkpoint={payload.get('upstream_versions', {})}"
            )
        model.load_state_dict(payload["state_dict"])
        model.training_metadata = dict(payload.get("training_metadata", {}))
        model.to(model.device)
        model.eval()
        return model


def load_generator(path: Path, device: str = "cpu") -> ConditionalGenerator:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    name = payload.get("generator_name")
    if name not in GENERATOR_REGISTRY:
        raise ValueError(f"Unknown generator_name in checkpoint: {name}")
    return GENERATOR_REGISTRY[name].load(path, device=device)
