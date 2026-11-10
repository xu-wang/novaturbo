"""Tagged Geometry19 MLP variants derived from the legacy surrogate topology."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


class Geometry19SurrogateError(ValueError):
    """Raised when a Geometry19 network or checkpoint contract is invalid."""


class TaggedGeometry19MLP(nn.Module):
    """Legacy surrogate hidden topology with versioned dynamic I/O dimensions."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: Sequence[int] = (128, 256, 256, 128, 64),
        activation: str = "relu",
        dropout: float = 0.1,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or any(int(width) <= 0 for width in hidden_layers):
            raise Geometry19SurrogateError("Network dimensions must be positive")
        if activation != "relu":
            raise Geometry19SurrogateError("Geometry19 v1 freezes the legacy ReLU activation")
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for width_value in hidden_layers:
            width = int(width_value)
            layers.append(nn.Linear(previous, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            previous = width
        layers.append(nn.Linear(previous, int(output_dim)))
        self.network = nn.Sequential(*layers)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_layers = tuple(int(value) for value in hidden_layers)
        self.activation = activation
        self.dropout = float(dropout)
        self.batch_norm = bool(batch_norm)

    def forward(self, normalized_input: torch.Tensor) -> torch.Tensor:
        if normalized_input.ndim != 2 or normalized_input.shape[1] != self.input_dim:
            raise Geometry19SurrogateError(
                f"Expected [n,{self.input_dim}] input, got {tuple(normalized_input.shape)}"
            )
        return self.network(normalized_input)

    def architecture_mapping(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_layers": list(self.hidden_layers),
            "activation": self.activation,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
        }


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_weight_bundle(
    directory: str | Path,
    model: TaggedGeometry19MLP,
    manifest: Mapping[str, Any],
    normalization: Mapping[str, np.ndarray],
) -> dict[str, str]:
    """Save state_dict, metadata, and normalization as separate artifacts."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    weights_path = target / "weights_best_val.pt"
    normalization_path = target / "normalization.npz"
    manifest_path = target / "model_manifest.json"
    torch.save(model.state_dict(), weights_path)
    np.savez(normalization_path, **normalization)
    payload = {
        **dict(manifest),
        "architecture": model.architecture_mapping(),
        "weights_sha256": sha256_file(weights_path),
        "normalization_sha256": sha256_file(normalization_path),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "weights": str(weights_path),
        "normalization": str(normalization_path),
        "manifest": str(manifest_path),
    }


def load_weight_bundle(
    directory: str | Path,
    *,
    expected_model_tag: str,
    expected_feature_names: Sequence[str],
    expected_target_names: Sequence[str],
    device: str | torch.device = "cpu",
) -> tuple[TaggedGeometry19MLP, dict[str, np.ndarray], dict[str, Any]]:
    """Strictly validate tags/order/hashes before loading weights-only state."""
    source = Path(directory)
    manifest = json.loads((source / "model_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("model_tag") != expected_model_tag:
        raise Geometry19SurrogateError("Model tag mismatch")
    if manifest.get("feature_names") != list(expected_feature_names):
        raise Geometry19SurrogateError("Feature contract mismatch")
    if manifest.get("target_names") != list(expected_target_names):
        raise Geometry19SurrogateError("Target contract mismatch")
    weights_path = source / "weights_best_val.pt"
    normalization_path = source / "normalization.npz"
    if sha256_file(weights_path) != manifest["weights_sha256"]:
        raise Geometry19SurrogateError("Weights hash mismatch")
    if sha256_file(normalization_path) != manifest["normalization_sha256"]:
        raise Geometry19SurrogateError("Normalization hash mismatch")
    model = TaggedGeometry19MLP(**manifest["architecture"]).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    normalization_file = np.load(normalization_path)
    normalization = {name: normalization_file[name] for name in normalization_file.files}
    return model, normalization, manifest
