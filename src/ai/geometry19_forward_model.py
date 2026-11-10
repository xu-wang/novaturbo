"""Forward-surrogate contract resolution and output decoding for WP19-23."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.ai.geometry19_inverse_contract import OUTPUT_NAMES


class Geometry19ForwardModelError(ValueError):
    """Raised when a forward bundle does not satisfy its declared role."""


@dataclass(frozen=True)
class ForwardModelSpec:
    logical_tag: str
    model_tag: str
    directory: Path
    output_transform: str
    output_contract: str
    target_names: tuple[str, ...]
    use_constraint_loss: bool
    use_feasibility_loss: bool

    def as_mapping(self) -> dict[str, Any]:
        return {
            "logical_tag": self.logical_tag,
            "model_tag": self.model_tag,
            "weights": str(self.directory),
            "output_transform": self.output_transform,
            "output_contract": self.output_contract,
            "target_names": list(self.target_names),
            "use_constraint_loss": self.use_constraint_loss,
            "use_feasibility_loss": self.use_feasibility_loss,
        }


def _resolved(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _manifest(directory: Path) -> dict[str, Any]:
    path = directory / "model_manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Geometry19ForwardModelError(f"Cannot load forward manifest: {path}") from exc
    if not isinstance(value, dict):
        raise Geometry19ForwardModelError("Forward manifest must be a mapping")
    return value


def resolve_forward_model_spec(
    root: str | Path,
    protocol: Mapping[str, Any],
    *,
    logical_tag: str | None = None,
    weights_override: str | Path | None = None,
) -> ForwardModelSpec:
    """Resolve a configured tag, retaining legacy override support for audits."""

    repository_root = Path(root).resolve()
    forward = protocol["models"]["forward"]
    if logical_tag is not None and weights_override is not None:
        raise Geometry19ForwardModelError(
            "forward_model_tag and forward_weights_override are mutually exclusive"
        )
    variants = forward["variants"]
    if weights_override is None:
        resolved_tag = str(forward["default_tag"] if logical_tag is None else logical_tag)
        try:
            declared = variants[resolved_tag]
        except KeyError as exc:
            raise Geometry19ForwardModelError(
                f"Unsupported forward model tag: {resolved_tag}"
            ) from exc
        directory = _resolved(repository_root, declared["weights"])
    else:
        directory = _resolved(repository_root, weights_override)
        matches = [
            (str(tag), value)
            for tag, value in variants.items()
            if _resolved(repository_root, value["weights"]) == directory
        ]
        if matches:
            resolved_tag, declared = matches[0]
        else:
            resolved_tag = "development_override"
            declared = None

    manifest = _manifest(directory)
    model_tag = str(manifest.get("model_tag", ""))
    target_names = tuple(str(value) for value in manifest.get("target_names", ()))
    output_transform = str(manifest.get("output_transform", "identity"))
    if output_transform not in {"identity", "natural_log"}:
        raise Geometry19ForwardModelError(
            f"Unsupported forward output transform: {output_transform}"
        )
    if target_names[:3] != OUTPUT_NAMES:
        raise Geometry19ForwardModelError(
            "Forward bundle must begin with thrust, TSFC, and mass"
        )
    if declared is None:
        performance_only = len(target_names) == 3
        output_contract = (
            "performance3_only"
            if performance_only
            else "performance3_plus_constraint_residuals"
        )
        use_constraint_loss = not performance_only
        use_feasibility_loss = not performance_only
    else:
        if model_tag != declared["model_tag"]:
            raise Geometry19ForwardModelError(
                f"Forward model_tag mismatch for tag {resolved_tag}"
            )
        if output_transform != declared["output_transform"]:
            raise Geometry19ForwardModelError(
                f"Forward output transform mismatch for tag {resolved_tag}"
            )
        output_contract = str(declared["output_contract"])
        use_constraint_loss = bool(declared["use_constraint_loss"])
        use_feasibility_loss = bool(declared["use_feasibility_loss"])
    if output_contract == "performance3_only":
        if target_names != OUTPUT_NAMES:
            raise Geometry19ForwardModelError(
                "performance3_only bundles must expose exactly three targets"
            )
        if use_constraint_loss or use_feasibility_loss:
            raise Geometry19ForwardModelError(
                "performance3_only cannot enable surrogate constraint or feasibility loss"
            )
    elif output_contract == "performance3_plus_constraint_residuals":
        if len(target_names) <= 3:
            raise Geometry19ForwardModelError(
                "Constraint-residual bundles must expose more than three targets"
            )
    else:
        raise Geometry19ForwardModelError(
            f"Unsupported forward output contract: {output_contract}"
        )
    return ForwardModelSpec(
        logical_tag=resolved_tag,
        model_tag=model_tag,
        directory=directory,
        output_transform=output_transform,
        output_contract=output_contract,
        target_names=target_names,
        use_constraint_loss=use_constraint_loss,
        use_feasibility_loss=use_feasibility_loss,
    )


def attach_forward_spec(
    normalization: Mapping[str, np.ndarray], spec: ForwardModelSpec
) -> dict[str, Any]:
    """Attach non-array runtime metadata without changing persisted normalization."""

    return {
        **dict(normalization),
        "_forward_model_spec": spec,
    }


def spec_from_normalization(normalization: Mapping[str, Any]) -> ForwardModelSpec:
    spec = normalization.get("_forward_model_spec")
    if not isinstance(spec, ForwardModelSpec):
        raise Geometry19ForwardModelError("Forward runtime spec is missing")
    return spec


def decode_forward_outputs(
    normalized_prediction: torch.Tensor,
    normalization: Mapping[str, Any],
) -> torch.Tensor:
    spec = spec_from_normalization(normalization)
    output_mean = torch.as_tensor(
        normalization["output_mean"],
        dtype=normalized_prediction.dtype,
        device=normalized_prediction.device,
    )
    output_std = torch.as_tensor(
        normalization["output_std"],
        dtype=normalized_prediction.dtype,
        device=normalized_prediction.device,
    )
    transformed = normalized_prediction * output_std + output_mean
    return torch.exp(transformed) if spec.output_transform == "natural_log" else transformed


def transform_physical_targets(
    target_values: torch.Tensor,
    output_transform: str,
) -> torch.Tensor:
    if output_transform == "natural_log":
        if torch.any(target_values <= 0.0):
            raise Geometry19ForwardModelError(
                "Natural-log forward targets must be positive"
            )
        return torch.log(target_values)
    if output_transform == "identity":
        return target_values
    raise Geometry19ForwardModelError(
        f"Unsupported forward output transform: {output_transform}"
    )
