"""Mixed continuous/integer CFM ablations for Geometry19.

The module deliberately lives beside the frozen WP19-23 implementation.  It
does not change the production generator contract or historical checkpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint
import yaml

from src.ai.generator_dataset import GeneratorNormalizer
from src.ai.generators.base import ConditionalGenerator, register_generator
from src.ai.generators.bounded import SmoothBoundedTransform
from src.ai.generators.torchcfm_adapter import (
    CFMGenerator,
    ConditionalFiLMResidualVelocityMLP,
    GPUSinkhornConditionalFlowMatcher,
)
from src.ai.geometry19_cfm import (
    CONDITION_NAMES,
    Geometry19CFMDataset,
    Geometry19CFMError,
    build_geometry19_cfm_loaders,
    build_geometry19_normalizer,
    fit_offset_sigmoid_bias,
    load_geometry19_cfm_splits,
)
from src.ai.geometry19_surrogate import sha256_file


INTEGER_NAMES = (
    "compressor_blade_count",
    "combustor_injector_count",
    "turbine_blade_count",
)
VERSION = "wp19_23_mixed_variable_cfm_v1"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mixed_indices(schema: Mapping[str, Any]) -> tuple[list[int], list[int], list[int]]:
    fields = list(schema["design_variables"])
    integer = [index for index, item in enumerate(fields) if item["dtype"] == "int64"]
    continuous = [index for index in range(len(fields)) if index not in integer]
    classes = [int(fields[index]["bounds"][1] - fields[index]["bounds"][0] + 1) for index in integer]
    names = tuple(fields[index]["name"] for index in integer)
    if names != INTEGER_NAMES or len(continuous) != 16 or classes != [13, 9, 13]:
        raise Geometry19CFMError("Geometry19 mixed-variable schema changed")
    return continuous, integer, classes


def normalized_integer_labels(
    normalized: torch.Tensor, integer_indices: Sequence[int], class_counts: Sequence[int]
) -> list[torch.Tensor]:
    labels = []
    for index, count in zip(integer_indices, class_counts, strict=True):
        values = torch.round(0.5 * (normalized[:, index] + 1.0) * (count - 1))
        labels.append(values.to(dtype=torch.long).clamp(0, count - 1))
    return labels


def labels_to_one_hot(labels: Sequence[torch.Tensor], class_counts: Sequence[int]) -> torch.Tensor:
    return torch.cat(
        [F.one_hot(value, count).to(dtype=torch.float32) for value, count in zip(labels, class_counts, strict=True)],
        dim=-1,
    )


class DiscreteConditionHead(nn.Module):
    def __init__(self, condition_dim: int, hidden_dim: int, class_counts: Sequence[int]) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, count) for count in class_counts)

    def forward(self, condition: torch.Tensor) -> list[torch.Tensor]:
        hidden = self.trunk(condition)
        return [head(hidden) for head in self.heads]


@register_generator
class Geometry19GumbelCFM(ConditionalGenerator):
    """16D bounded CFM conditioned by three ST Gumbel categorical heads."""

    generator_name = "geometry19_gumbel_cfm"
    upstream_packages = ("torchcfm", "torchdiffeq")

    def __init__(self, model_config, runtime_config, schema_version="geometry19_schema_v1", normalizer=None):
        super().__init__(model_config, runtime_config, schema_version, normalizer)
        self.continuous_indices = [int(value) for value in model_config["continuous_indices"]]
        self.integer_indices = [int(value) for value in model_config["integer_indices"]]
        self.class_counts = [int(value) for value in model_config["class_counts"]]
        if self.design_dim != 19 or len(self.continuous_indices) != 16 or self.class_counts != [13, 9, 13]:
            raise ValueError("Invalid Geometry19 Gumbel mixed dimensions")
        self.discrete_dim = sum(self.class_counts)
        self.discrete_head = DiscreteConditionHead(
            self.condition_dim,
            int(model_config.get("discrete_hidden_dim", 128)),
            self.class_counts,
        )
        self.velocity = ConditionalFiLMResidualVelocityMLP(
            hidden_dim=int(model_config.get("hidden_dim", 256)),
            residual_blocks=int(model_config.get("residual_blocks", 2)),
            condition_encoder_dim=int(model_config.get("condition_encoder_dim", 128)),
            design_dim=16,
            condition_dim=self.condition_dim + self.discrete_dim,
            time_embedding=model_config["time_embedding"],
        )
        self.matcher = GPUSinkhornConditionalFlowMatcher(
            sigma=float(model_config.get("sigma", 0.0)),
            regularization=float(model_config.get("sinkhorn_regularization", 0.1)),
            iterations=int(model_config.get("sinkhorn_iterations", 50)),
        )
        bounded = dict(model_config["continuous_bounded_output"])
        self.continuous_transform = SmoothBoundedTransform.from_config(bounded, dimension=16)
        self.gumbel_temperature = float(model_config.get("gumbel_temperature", 0.75))
        self.discrete_loss_weight = float(model_config.get("discrete_loss_weight", 1.0))

    def _straight_through_condition(
        self, condition: torch.Tensor, true_one_hot: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        logits = self.discrete_head(condition)
        soft = torch.cat(
            [
                F.gumbel_softmax(value, tau=self.gumbel_temperature, hard=False, dim=-1)
                if training
                else F.softmax(value / self.gumbel_temperature, dim=-1)
                for value in logits
            ],
            dim=-1,
        )
        # Forward values are the true categories; backward gradients follow the
        # relaxed Gumbel samples into the categorical heads.
        return true_one_hot + soft - soft.detach(), logits

    def _batch_loss(self, batch, training):
        normalized = batch["x"]
        labels = normalized_integer_labels(normalized, self.integer_indices, self.class_counts)
        true_one_hot = labels_to_one_hot(labels, self.class_counts).to(normalized)
        continuous = normalized[:, self.continuous_indices]
        target, _, clipped = self.continuous_transform.to_unconstrained(continuous)
        source = torch.randn_like(target)
        combined = torch.cat([batch["condition"], true_one_hot], dim=-1)
        t, location, target_velocity, _, paired = self.matcher.guided_sample_location_and_conditional_flow(
            source, target, y1=combined
        )
        paired_condition = paired[:, : self.condition_dim]
        paired_true = paired[:, self.condition_dim :]
        discrete_condition, logits = self._straight_through_condition(
            paired_condition, paired_true, training
        )
        prediction = self.velocity(t, location, torch.cat([paired_condition, discrete_condition], dim=-1))
        flow_loss = F.mse_loss(prediction, target_velocity)
        paired_labels = []
        offset = 0
        for count in self.class_counts:
            paired_labels.append(paired_true[:, offset : offset + count].argmax(dim=-1))
            offset += count
        categorical_loss = sum(F.cross_entropy(logit, label) for logit, label in zip(logits, paired_labels, strict=True)) / len(logits)
        total = flow_loss + self.discrete_loss_weight * categorical_loss
        accuracy = torch.stack(
            [(logit.argmax(dim=-1) == label).float().mean() for logit, label in zip(logits, paired_labels, strict=True)]
        ).mean()
        return total, {
            "flow_matching_loss": flow_loss,
            "categorical_loss": categorical_loss,
            "categorical_accuracy": accuracy,
            "bounded_input_clip_rate": clipped.mean(),
        }

    def _draw_categories(self, condition: torch.Tensor, seed: int | None) -> tuple[list[torch.Tensor], torch.Tensor]:
        logits = self.discrete_head(condition)
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(int(seed) + 104729)
        labels = []
        one_hot = []
        for value, count in zip(logits, self.class_counts, strict=True):
            uniform = torch.rand(value.shape, generator=generator, device=self.device, dtype=self.dtype).clamp_(1e-7, 1.0 - 1e-7)
            noise = -torch.log(-torch.log(uniform))
            label = ((value + noise) / self.gumbel_temperature).argmax(dim=-1)
            labels.append(label)
            one_hot.append(F.one_hot(label, count).to(self.dtype))
        return labels, torch.cat(one_hot, dim=-1)

    def _sample_normalized(self, condition, n_samples, seed):
        batch = condition.shape[0]
        expanded_condition = condition[:, None, :].expand(-1, n_samples, -1).reshape(-1, self.condition_dim)
        labels, one_hot = self._draw_categories(expanded_condition, seed)
        augmented = torch.cat([expanded_condition, one_hot], dim=-1)
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(int(seed))
        initial = torch.randn(batch * n_samples, 16, generator=generator, device=self.device, dtype=self.dtype)

        def field(t, state):
            return self.velocity(t, state, augmented)

        steps = int(self.model_config.get("ode_steps", 32))
        trajectory = odeint(
            field,
            initial,
            torch.tensor([0.0, 1.0], device=self.device, dtype=self.dtype),
            method="rk4",
            options={"step_size": 1.0 / steps},
        )
        continuous, _ = self.continuous_transform.to_bounded(trajectory[-1])
        output = torch.empty(batch * n_samples, 19, device=self.device, dtype=self.dtype)
        output[:, self.continuous_indices] = continuous
        for index, label, count in zip(self.integer_indices, labels, self.class_counts, strict=True):
            output[:, index] = 2.0 * label.to(self.dtype) / (count - 1) - 1.0
        diagnostics = self.continuous_transform.sample_diagnostics(trajectory[-1], continuous)
        diagnostics.update({"mixed_strategy": "st_gumbel_softmax", "integer_legal_rate": 1.0})
        return output.reshape(batch, n_samples, 19), diagnostics


@register_generator
class Geometry19DequantizedCFM(CFMGenerator):
    """19D bounded CFM trained on disjoint dequantization bins."""

    generator_name = "geometry19_dequantized_cfm"

    def __init__(self, model_config, runtime_config, schema_version="geometry19_schema_v1", normalizer=None):
        super().__init__(model_config, runtime_config, schema_version, normalizer)
        self.integer_indices = [int(value) for value in model_config["integer_indices"]]
        self.class_counts = [int(value) for value in model_config["class_counts"]]
        if self.design_dim != 19 or self.class_counts != [13, 9, 13]:
            raise ValueError("Invalid Geometry19 dequantization dimensions")

    def _dequantize(self, normalized: torch.Tensor, training: bool) -> torch.Tensor:
        result = normalized.clone()
        labels = normalized_integer_labels(normalized, self.integer_indices, self.class_counts)
        for index, label, count in zip(self.integer_indices, labels, self.class_counts, strict=True):
            noise = torch.rand_like(label, dtype=normalized.dtype) if training else torch.full_like(label, 0.5, dtype=normalized.dtype)
            result[:, index] = 2.0 * (label.to(normalized.dtype) + noise) / count - 1.0
        return result

    def _batch_loss(self, batch, training):
        updated = dict(batch)
        updated["x"] = self._dequantize(batch["x"], training)
        return super()._batch_loss(updated, training)

    def _decode_bins(self, normalized: torch.Tensor) -> torch.Tensor:
        result = normalized.clone()
        for index, count in zip(self.integer_indices, self.class_counts, strict=True):
            label = torch.floor(0.5 * (result[..., index] + 1.0) * count).to(torch.long).clamp(0, count - 1)
            result[..., index] = 2.0 * label.to(result.dtype) / (count - 1) - 1.0
        return result

    def _sample_normalized(self, condition, n_samples, seed):
        values, diagnostics = super()._sample_normalized(condition, n_samples, seed)
        decoded = self._decode_bins(values)
        diagnostics.update({"mixed_strategy": "uniform_dequantization_bins", "integer_legal_rate": 1.0})
        return decoded, diagnostics


@register_generator
class Geometry19TupleCFM(ConditionalGenerator):
    """16D bounded CFM evaluated over an externally selected legal tuple set."""

    generator_name = "geometry19_tuple_cfm"
    upstream_packages = ("torchcfm", "torchdiffeq")

    def __init__(self, model_config, runtime_config, schema_version="geometry19_schema_v1", normalizer=None):
        super().__init__(model_config, runtime_config, schema_version, normalizer)
        self.continuous_indices = [int(value) for value in model_config["continuous_indices"]]
        self.integer_indices = [int(value) for value in model_config["integer_indices"]]
        self.class_counts = [int(value) for value in model_config["class_counts"]]
        self.discrete_dim = sum(self.class_counts)
        self.velocity = ConditionalFiLMResidualVelocityMLP(
            hidden_dim=int(model_config.get("hidden_dim", 256)),
            residual_blocks=int(model_config.get("residual_blocks", 2)),
            condition_encoder_dim=int(model_config.get("condition_encoder_dim", 128)),
            design_dim=16,
            condition_dim=self.condition_dim + self.discrete_dim,
            time_embedding=model_config["time_embedding"],
        )
        self.matcher = GPUSinkhornConditionalFlowMatcher(
            sigma=float(model_config.get("sigma", 0.0)),
            regularization=float(model_config.get("sinkhorn_regularization", 0.1)),
            iterations=int(model_config.get("sinkhorn_iterations", 50)),
        )
        self.continuous_transform = SmoothBoundedTransform.from_config(
            model_config["continuous_bounded_output"], dimension=16
        )

    def _batch_loss(self, batch, training):
        labels = normalized_integer_labels(batch["x"], self.integer_indices, self.class_counts)
        one_hot = labels_to_one_hot(labels, self.class_counts).to(batch["x"])
        continuous = batch["x"][:, self.continuous_indices]
        target, _, clipped = self.continuous_transform.to_unconstrained(continuous)
        source = torch.randn_like(target)
        combined = torch.cat([batch["condition"], one_hot], dim=-1)
        t, location, target_velocity, _, paired = self.matcher.guided_sample_location_and_conditional_flow(
            source, target, y1=combined
        )
        prediction = self.velocity(t, location, paired)
        loss = F.mse_loss(prediction, target_velocity)
        return loss, {"flow_matching_loss": loss, "bounded_input_clip_rate": clipped.mean()}

    def sample_with_labels(self, condition: torch.Tensor, labels: torch.Tensor, seed: int | None) -> torch.Tensor:
        prepared, _ = self._prepare_condition(condition)
        prepared = prepared.to(device=self.device, dtype=self.dtype)
        if prepared.shape[0] != 1 or labels.ndim != 2 or labels.shape[1] != 3:
            raise ValueError("Tuple CFM expects one condition and labels shaped [n,3]")
        labels = labels.to(device=self.device, dtype=torch.long)
        one_hot = labels_to_one_hot([labels[:, i] for i in range(3)], self.class_counts).to(self.device)
        expanded = prepared.expand(len(labels), -1)
        augmented = torch.cat([expanded, one_hot], dim=-1)
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(int(seed))
        initial = torch.randn(len(labels), 16, generator=generator, device=self.device, dtype=self.dtype)

        def field(t, state):
            return self.velocity(t, state, augmented)

        trajectory = odeint(
            field, initial, torch.tensor([0.0, 1.0], device=self.device, dtype=self.dtype),
            method="rk4", options={"step_size": 1.0 / int(self.model_config.get("ode_steps", 32))},
        )
        continuous, _ = self.continuous_transform.to_bounded(trajectory[-1])
        output = torch.empty(len(labels), 19, device=self.device, dtype=self.dtype)
        output[:, self.continuous_indices] = continuous
        for index, values, count in zip(self.integer_indices, labels.T, self.class_counts, strict=True):
            output[:, index] = 2.0 * values.to(self.dtype) / (count - 1) - 1.0
        return output.detach().cpu()

    def _sample_normalized(self, condition, n_samples, seed):
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(int(seed) + 104729)
        labels = torch.stack(
            [torch.randint(count, (condition.shape[0] * n_samples,), generator=generator, device=self.device) for count in self.class_counts],
            dim=-1,
        )
        if condition.shape[0] != 1:
            raise ValueError("Generic tuple sampling currently supports one condition")
        values = self.sample_with_labels(condition[0], labels, seed).to(self.device)
        return values.reshape(1, n_samples, 19), {"mixed_strategy": "uniform_tuple_fallback", "integer_legal_rate": 1.0}


def _dequantized_center_targets(normalized: np.ndarray, integer_indices: Sequence[int], class_counts: Sequence[int]) -> np.ndarray:
    result = np.asarray(normalized, dtype=np.float64).copy()
    for index, count in zip(integer_indices, class_counts, strict=True):
        labels = np.rint(0.5 * (result[:, index] + 1.0) * (count - 1)).clip(0, count - 1)
        result[:, index] = 2.0 * (labels + 0.5) / count - 1.0
    return result


def train_mixed_cfm_arms(
    root: str | Path, config_path: str | Path, *, smoke: bool = False,
    selected_arms: Sequence[str] | None = None,
    fixed_400: bool = False,
    snapshot_every: int = 0,
) -> dict[str, Any]:
    repo = Path(root).resolve()
    source = Path(config_path).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if config.get("protocol_version") != VERSION:
        raise Geometry19CFMError("Mixed CFM protocol mismatch")
    if smoke and fixed_400:
        raise Geometry19CFMError("smoke and fixed_400 are mutually exclusive")
    if snapshot_every and (not fixed_400 or selected_arms != ["c_dequantized"]):
        raise Geometry19CFMError("Snapshot campaign requires fixed_400 and only c_dequantized")
    schema = yaml.safe_load((repo / config["data"]["design_schema"]).read_text(encoding="utf-8"))
    feature_names = [item["name"] for item in schema["design_variables"]]
    continuous, integer, classes = mixed_indices(schema)
    maximum = int(config["smoke"]["maximum_rows_per_split"]) if smoke else None
    splits = load_geometry19_cfm_splits(
        repo / config["data"]["regression_dataset"], feature_names,
        config["data"]["campaign_split"], maximum,
    )
    normalizer = build_geometry19_normalizer(schema, splits["train"], config["data"]["condition_transforms"])
    batch_size = int(config["smoke"]["batch_size"] if smoke else config["data"]["batch_size"])
    train_loader, validation_loader, _ = build_geometry19_cfm_loaders(
        splits, normalizer, batch_size=batch_size,
        num_workers=int(config["data"]["num_workers"]), seed=int(config["training"]["seed"]),
    )
    results = {}
    for arm in config["arms"]:
        arm_id = str(arm["arm_id"])
        if selected_arms is not None and arm_id not in selected_arms:
            continue
        model_config = dict(arm["model"])
        model_config.update(
            design_dim=19, condition_dim=3,
            continuous_indices=continuous, integer_indices=integer, class_counts=classes,
        )
        runtime = dict(config["training"])
        if smoke:
            runtime.update(
                device=config["smoke"]["device"], epochs=config["smoke"]["epochs"],
                minimum_epochs=config["smoke"]["minimum_epochs"], patience=config["smoke"]["patience"],
            )
            model_config["ode_steps"] = int(config["smoke"]["ode_steps"])
        elif fixed_400:
            runtime.update(epochs=400, minimum_epochs=400, patience=400)
        _seed_all(int(runtime["seed"]))
        if arm_id == "a_gumbel":
            normalized_train = normalizer.transform_design(splits["train"]["x"])
            cont_values = normalized_train[:, continuous]
            bounded = dict(model_config["continuous_bounded_output"])
            bounded["offset"] = fit_offset_sigmoid_bias(cont_values, float(bounded["epsilon"])).tolist()
            model_config["continuous_bounded_output"] = bounded
            model = Geometry19GumbelCFM(model_config, runtime, str(schema["schema_version"]), normalizer)
        elif arm_id == "c_dequantized":
            normalized_train = normalizer.transform_design(splits["train"]["x"])
            centers = _dequantized_center_targets(normalized_train, integer, classes)
            bounded = dict(model_config["bounded_output"])
            bounded["offset"] = fit_offset_sigmoid_bias(centers, float(bounded["epsilon"])).tolist()
            model_config["bounded_output"] = bounded
            model = Geometry19DequantizedCFM(model_config, runtime, str(schema["schema_version"]), normalizer)
        elif arm_id == "b_tuple":
            normalized_train = normalizer.transform_design(splits["train"]["x"])
            bounded = dict(model_config["continuous_bounded_output"])
            bounded["offset"] = fit_offset_sigmoid_bias(normalized_train[:, continuous], float(bounded["epsilon"])).tolist()
            model_config["continuous_bounded_output"] = bounded
            model = Geometry19TupleCFM(model_config, runtime, str(schema["schema_version"]), normalizer)
        else:
            raise Geometry19CFMError(f"Unknown mixed CFM arm: {arm_id}")
        suffix = "smoke" if smoke else "checkpoint_campaign" if snapshot_every else "full400" if fixed_400 else "baseline"
        weights = repo / arm["artifacts"]["weights_root"] / suffix
        runtime_root = repo / arm["artifacts"]["runtime_root"] / suffix
        if weights.exists() or runtime_root.exists():
            raise Geometry19CFMError(f"Refusing to overwrite mixed CFM artifacts: {arm_id}/{suffix}")
        weights.mkdir(parents=True)
        runtime_root.mkdir(parents=True)
        if snapshot_every:
            model.runtime_config["snapshot_every"] = int(snapshot_every)
            model.runtime_config["snapshot_directory"] = str(weights / "snapshots")
        normalizer.save(weights / "normalizer.json")
        history = model.fit(train_loader, validation_loader)
        model.save(weights / "best.pt", state="best")
        model.save(weights / "last.pt", state="last")
        _write_json(weights / "training_history.json", history)
        sample = model.sample(validation_loader.dataset.condition[0], int(config["validation"]["sample_count"]), seed=int(config["validation"]["source_seed"]))
        integer_values = sample.values[:, integer].numpy()
        legal = all(np.allclose(0.5 * (integer_values[:, pos] + 1.0) * (count - 1), np.rint(0.5 * (integer_values[:, pos] + 1.0) * (count - 1)), atol=1e-6) for pos, count in enumerate(classes))
        reloaded = type(model).load(weights / "best.pt", device=str(model.device))
        replay = reloaded.sample(validation_loader.dataset.condition[0], int(config["validation"]["sample_count"]), seed=int(config["validation"]["source_seed"]))
        summary = {
            "protocol_version": VERSION,
            "status": "PASS" if legal and torch.equal(sample.values, replay.values) else "AUDIT_FAILED",
            "arm_id": arm_id,
            "run_kind": suffix,
            "training_regimen": "fixed_400_no_early_stop" if fixed_400 else "early_stopping",
            "snapshot_every": int(snapshot_every),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "training": {key: history[key] for key in ("best_epoch", "best_validation_loss", "epochs_completed", "elapsed_seconds", "peak_gpu_memory_bytes", "total_parameters")},
            "validation": {"integer_legal": bool(legal), "reload_exact": bool(torch.equal(sample.values, replay.values)), "sample_count": len(sample.values)},
            "artifacts": {"best_checkpoint": str((weights / "best.pt").relative_to(repo))},
            "hashes": {"config": sha256_file(source)},
        }
        _write_json(runtime_root / "training_summary.json", summary)
        results[arm_id] = summary
    return results
