"""Geometry19 data adaptation and CFM-4 training for WP19-23A."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchdiffeq import odeint
import yaml

from src.ai.generator_dataset import GeneratorNormalizer
from src.ai.generators import CFMGenerator, load_generator
from src.ai.candidate_protocol import largest_power_of_two_not_above
from src.ai.geometry19_inverse_contract import load_wp19_23_contract
from src.ai.geometry19_hit_alignment import ordered_tensor
from src.ai.geometry19_surrogate import load_weight_bundle, sha256_file
from src.physics import ConstraintProfile, ModelConfig


CONDITION_NAMES = ("thrust_N", "tsfc_kg_N_s", "mass_kg")


class Geometry19CFMError(ValueError):
    """Raised when Geometry19 CFM data, training, or artifacts violate contract."""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _campaign_lookup(
    campaign_split: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    if set(campaign_split) != {"train", "validation", "test"}:
        raise Geometry19CFMError("Campaign split must define train/validation/test")
    lookup: dict[str, str] = {}
    for split, campaigns in campaign_split.items():
        for campaign in campaigns:
            campaign_name = str(campaign)
            if campaign_name in lookup:
                raise Geometry19CFMError(
                    f"Campaign appears in multiple splits: {campaign_name}"
                )
            lookup[campaign_name] = str(split)
    return lookup


def load_geometry19_cfm_splits(
    path: str | Path,
    feature_names: Sequence[str],
    campaign_split: Mapping[str, Sequence[str]],
    maximum_rows_per_split: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Load only 19D designs and three conditions from the converged JSONL."""

    lookup = _campaign_lookup(campaign_split)
    buckets = {
        split: {"x": [], "condition": [], "sample_ids": []}
        for split in campaign_split
    }
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            campaign = str(record["campaign_id"])
            split = lookup.get(campaign)
            if split is None:
                raise Geometry19CFMError(
                    f"Regression record has unknown campaign: {campaign}"
                )
            bucket = buckets[split]
            if (
                maximum_rows_per_split is not None
                and len(bucket["x"]) >= maximum_rows_per_split
            ):
                continue
            performance = record["physics_outputs"]["performance"]
            bucket["x"].append([record["x_19"][name] for name in feature_names])
            bucket["condition"].append(
                [performance[name] for name in CONDITION_NAMES]
            )
            bucket["sample_ids"].append(
                f"{campaign}:{record.get('campaign_sample_id', record['sample_id'])}"
            )
    for split, bucket in buckets.items():
        bucket["x"] = np.asarray(bucket["x"], dtype=np.float64)
        bucket["condition"] = np.asarray(bucket["condition"], dtype=np.float64)
        if (
            bucket["x"].ndim != 2
            or bucket["x"].shape[1] != len(feature_names)
            or bucket["condition"].shape != (len(bucket["x"]), 3)
            or not len(bucket["x"])
            or not np.isfinite(bucket["x"]).all()
            or not np.isfinite(bucket["condition"]).all()
            or np.any(bucket["condition"] <= 0.0)
        ):
            raise Geometry19CFMError(f"Invalid or empty Geometry19 CFM split: {split}")
    return buckets


def build_geometry19_normalizer(
    schema: Mapping[str, Any],
    train: Mapping[str, Any],
    condition_transforms: Mapping[str, str],
) -> GeneratorNormalizer:
    """Fit train-only conditions and frozen-bound 19D normalization."""

    feature_names = [item["name"] for item in schema["design_variables"]]
    design = []
    for item in schema["design_variables"]:
        lower, upper = (float(value) for value in item["bounds"])
        if not upper > lower:
            raise Geometry19CFMError(f"Invalid bounds for {item['name']}")
        design.append(
            {
                "name": item["name"],
                "method": "physical_bounds_to_minus_one_plus_one",
                "physical_min": lower,
                "physical_max": upper,
                "normalized_min": -1.0,
                "normalized_max": 1.0,
            }
        )
    if feature_names != [item["name"] for item in design]:
        raise Geometry19CFMError("Geometry19 feature order changed during normalization")

    conditions = []
    matrix = np.asarray(train["condition"], dtype=np.float64)
    for index, name in enumerate(CONDITION_NAMES):
        transform = str(condition_transforms[name])
        values = matrix[:, index]
        if transform == "natural_log":
            if np.any(values <= 0.0):
                raise Geometry19CFMError(f"{name} must be positive for logarithm")
            values = np.log(values)
        elif transform != "identity":
            raise Geometry19CFMError(f"Unsupported condition transform: {transform}")
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if not math.isfinite(std) or std <= 0.0:
            raise Geometry19CFMError(f"Invalid training standard deviation for {name}")
        conditions.append(
            {
                "name": name,
                "method": "standard",
                "transform": transform,
                "mean": mean,
                "std": std,
            }
        )
    sample_ids = sorted(str(value) for value in train["sample_ids"])
    sample_hash = sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()
    return GeneratorNormalizer(
        schema_version=str(schema["schema_version"]),
        design=design,
        conditions=conditions,
        fitted_split="train",
        fitted_sample_count=len(sample_ids),
        fitted_sample_id_sha256=sample_hash,
    )


class Geometry19CFMDataset(Dataset):
    def __init__(
        self,
        bucket: Mapping[str, Any],
        normalizer: GeneratorNormalizer,
    ) -> None:
        x = normalizer.transform_design(bucket["x"])
        condition = normalizer.transform_conditions(bucket["condition"])
        if not np.isfinite(x).all() or not np.isfinite(condition).all():
            raise Geometry19CFMError("Normalized Geometry19 CFM data are non-finite")
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.condition = torch.as_tensor(condition, dtype=torch.float32)
        self.sample_ids = list(bucket["sample_ids"])

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "x": self.x[index],
            "condition": self.condition[index],
            "sample_id": self.sample_ids[index],
        }


class Geometry19ToleranceSetDataset(Geometry19CFMDataset):
    """Pair each design with random targets it satisfies under the frozen tolerances."""

    def __init__(
        self,
        bucket: Mapping[str, Any],
        normalizer: GeneratorNormalizer,
        relative_tolerances: Sequence[float],
        maximum_fraction: float,
    ) -> None:
        super().__init__(bucket, normalizer)
        if len(relative_tolerances) != len(CONDITION_NAMES):
            raise Geometry19CFMError("Tolerance-set training requires three tolerances")
        if not 0.0 < maximum_fraction <= 1.0:
            raise Geometry19CFMError("Tolerance-set maximum_fraction must be in (0,1]")
        tolerances = torch.as_tensor(relative_tolerances, dtype=torch.float32)
        if not torch.isfinite(tolerances).all() or torch.any(tolerances <= 0.0):
            raise Geometry19CFMError("Tolerance-set tolerances must be finite and positive")
        self.physical_condition = torch.as_tensor(
            bucket["condition"], dtype=torch.float32
        )
        self.relative_tolerances = tolerances
        self.maximum_fraction = float(maximum_fraction)
        self.condition_mean = torch.tensor(
            [item["mean"] for item in normalizer.conditions], dtype=torch.float32
        )
        self.condition_std = torch.tensor(
            [item["std"] for item in normalizer.conditions], dtype=torch.float32
        )
        self.condition_log_mask = torch.tensor(
            [item["transform"] == "natural_log" for item in normalizer.conditions],
            dtype=torch.bool,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        delta = (
            2.0 * torch.rand(len(CONDITION_NAMES), dtype=torch.float32) - 1.0
        ) * self.maximum_fraction * self.relative_tolerances
        target = self.physical_condition[index] / (1.0 + delta)
        transformed = torch.where(self.condition_log_mask, torch.log(target), target)
        condition = (transformed - self.condition_mean) / self.condition_std
        return {
            "x": self.x[index],
            "condition": condition,
            "sample_id": self.sample_ids[index],
        }


class Geometry19PerformanceConsistencyLoss:
    """Frozen-S0 endpoint loss for 19D CFM training only."""

    def __init__(
        self,
        root: Path,
        generator: CFMGenerator,
        schema: Mapping[str, Any],
        protocol: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> None:
        self.weight = float(config["weight"])
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise Geometry19CFMError("Performance-consistency weight must be positive")
        if config.get("objective") != "hit_aligned_smooth_rho":
            raise Geometry19CFMError("Unsupported performance-consistency objective")
        if config.get("integer_projection") != "straight_through_nearest_even":
            raise Geometry19CFMError("Performance consistency requires ST nearest projection")
        forward_dir = Path(config["forward_weights"])
        if not forward_dir.is_absolute():
            forward_dir = root / forward_dir
        manifest = json.loads((forward_dir / "model_manifest.json").read_text(encoding="utf-8"))
        feature_names = [item["name"] for item in schema["design_variables"]]
        if manifest["feature_names"] != feature_names:
            raise Geometry19CFMError("S0 and Geometry19 CFM feature contracts differ")
        model, normalization, _ = load_weight_bundle(
            forward_dir,
            expected_model_tag=str(manifest["model_tag"]),
            expected_feature_names=feature_names,
            expected_target_names=list(manifest["target_names"]),
            device=generator.device,
        )
        model.eval().requires_grad_(False)
        self.model = model
        self.normalization = normalization
        self.integer_indices = [
            index
            for index, item in enumerate(schema["design_variables"])
            if item["dtype"] == "int64"
        ]
        dtype = generator.dtype
        device = generator.device
        self.lower = torch.tensor(
            [item["bounds"][0] for item in schema["design_variables"]],
            dtype=dtype,
            device=device,
        )
        self.upper = torch.tensor(
            [item["bounds"][1] for item in schema["design_variables"]],
            dtype=dtype,
            device=device,
        )
        self.input_mean = torch.as_tensor(
            normalization["input_mean"], dtype=dtype, device=device
        )
        self.input_std = torch.as_tensor(
            normalization["input_std"], dtype=dtype, device=device
        )
        self.output_mean = torch.as_tensor(
            normalization["output_mean"], dtype=dtype, device=device
        )
        self.output_std = torch.as_tensor(
            normalization["output_std"], dtype=dtype, device=device
        )
        self.condition_mean = torch.tensor(
            [item["mean"] for item in generator.normalizer.conditions],
            dtype=dtype,
            device=device,
        )
        self.condition_std = torch.tensor(
            [item["std"] for item in generator.normalizer.conditions],
            dtype=dtype,
            device=device,
        )
        self.condition_log_mask = torch.tensor(
            [item["transform"] == "natural_log" for item in generator.normalizer.conditions],
            dtype=torch.bool,
            device=device,
        )
        acceptance = protocol["diagnostic_acceptance"]
        self.tolerances = ordered_tensor(
            acceptance["relative_tolerances"], dtype=dtype, device=device
        )
        self.epsilons = ordered_tensor(
            acceptance["epsilons"], dtype=dtype, device=device
        )
        self.alpha = float(protocol["hit_alignment"]["smooth_max_alpha"])
        self.forward_weights = forward_dir

    def __call__(
        self, normalized_design: torch.Tensor, normalized_condition: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        physical = self.lower + 0.5 * (normalized_design + 1.0) * (
            self.upper - self.lower
        )
        if self.integer_indices:
            projected = physical[:, self.integer_indices]
            straight_through = projected + (torch.round(projected) - projected).detach()
            physical = physical.clone()
            physical[:, self.integer_indices] = straight_through
        surrogate_input = (physical - self.input_mean) / self.input_std
        normalized_prediction = self.model(surrogate_input)
        prediction = normalized_prediction * self.output_std + self.output_mean
        target = normalized_condition * self.condition_std + self.condition_mean
        target = torch.where(self.condition_log_mask, torch.exp(target), target)
        denominator = self.tolerances[None, :] * torch.maximum(
            target.abs(), self.epsilons[None, :]
        )
        ratios = (prediction[:, :3] - target).abs() / denominator
        smooth_rho = torch.logsumexp(self.alpha * ratios, dim=1) / self.alpha
        consistency = smooth_rho.mean()
        weighted = self.weight * consistency
        return weighted, {
            "performance_consistency_loss": consistency,
            "weighted_performance_consistency_loss": weighted,
            "surrogate_endpoint_rho": ratios.max(dim=1).values.mean(),
        }


def fit_offset_sigmoid_bias(
    normalized_designs: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Fit a fixed decoder bias so model-space train targets are centered."""

    values = np.asarray(normalized_designs, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise Geometry19CFMError("Offset fitting requires a finite 2D design matrix")
    if not 0.0 < epsilon < 0.1:
        raise Geometry19CFMError("Offset sigmoid epsilon must be in (0,0.1)")
    mean = values.mean(axis=0)
    lower = -1.0 + 2.0 * epsilon
    upper = 1.0 - 2.0 * epsilon
    stabilized = np.clip(mean, lower, upper)
    probability = (0.5 * (stabilized + 1.0) - epsilon) / (1.0 - 2.0 * epsilon)
    bias = np.log(probability) - np.log1p(-probability)
    if not np.isfinite(bias).all():
        raise Geometry19CFMError("Offset sigmoid bias fitting became non-finite")
    return bias


def build_geometry19_cfm_loaders(
    splits: Mapping[str, Mapping[str, Any]],
    normalizer: GeneratorNormalizer,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    tolerance_set: Mapping[str, Any] | None = None,
) -> tuple[DataLoader, DataLoader, Geometry19CFMDataset]:
    tolerance_config = dict(tolerance_set or {})
    if bool(tolerance_config.get("enabled", False)):
        train = Geometry19ToleranceSetDataset(
            splits["train"],
            normalizer,
            tolerance_config["relative_tolerances"],
            float(tolerance_config["maximum_fraction"]),
        )
    else:
        train = Geometry19CFMDataset(splits["train"], normalizer)
    validation = Geometry19CFMDataset(splits["validation"], normalizer)
    test = Geometry19CFMDataset(splits["test"], normalizer)
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
            drop_last=False,
        ),
        DataLoader(
            validation,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        ),
        test,
    )


def sample_cfm_sources(generator: CFMGenerator, count: int, seed: int) -> torch.Tensor:
    """Draw explicit source values identically to the production sampler."""

    if count <= 0:
        raise Geometry19CFMError("Source count must be positive")
    random_generator = torch.Generator(device=generator.device)
    random_generator.manual_seed(int(seed))
    return torch.randn(
        count,
        generator.design_dim,
        generator=random_generator,
        device=generator.device,
        dtype=generator.dtype,
    )


def integrate_cfm_sources(
    generator: CFMGenerator,
    sources: torch.Tensor,
    normalized_condition: torch.Tensor | np.ndarray,
    *,
    ode_steps: int | None = None,
) -> torch.Tensor:
    """Integrate explicit sources through the frozen CFM-4 RK4 field."""

    if sources.ndim != 2 or sources.shape[1] != generator.design_dim:
        raise Geometry19CFMError("Sources must have shape [n,19]")
    condition = torch.as_tensor(
        normalized_condition,
        dtype=sources.dtype,
        device=sources.device,
    )
    if condition.ndim == 1:
        condition = condition.unsqueeze(0)
    if condition.shape[0] == 1:
        condition = condition.expand(sources.shape[0], -1)
    if condition.shape != (sources.shape[0], generator.condition_dim):
        raise Geometry19CFMError("Condition cannot be expanded to source batch")
    steps = int(ode_steps or generator.model_config.get("ode_steps", 32))
    if steps <= 0 or generator.model_config.get("ode_method", "rk4") != "rk4":
        raise Geometry19CFMError("Geometry19 CFM source integration requires RK4")

    def velocity(time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return generator.velocity(time, state, condition)

    trajectory = odeint(
        velocity,
        sources,
        torch.tensor([0.0, 1.0], dtype=sources.dtype, device=sources.device),
        method="rk4",
        options={"step_size": 1.0 / steps},
        rtol=float(generator.model_config.get("ode_rtol", 1.0e-5)),
        atol=float(generator.model_config.get("ode_atol", 1.0e-5)),
    )
    normalized, _, _ = generator._from_model_space(trajectory[-1])
    return normalized


def _sample_metrics(values: np.ndarray) -> dict[str, Any]:
    finite = bool(np.isfinite(values).all())
    rounded_unique = np.unique(np.round(values, 8), axis=0)
    outside = np.any((values < -1.0) | (values > 1.0), axis=1)
    return {
        "count": len(values),
        "finite": finite,
        "unique_count_at_1e_8": len(rounded_unique),
        "duplicate_rate": float(1.0 - len(rounded_unique) / len(values)),
        "out_of_bounds_count": int(outside.sum()),
        "out_of_bounds_rate": float(outside.mean()),
        "mean_dimension_std": float(values.std(axis=0).mean()),
    }


def _validate_provenance(root: Path, config: Mapping[str, Any]) -> dict[str, str]:
    manifest_path = root / config["data"]["dataset_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_path = root / config["data"]["design_schema"]
    calibration_path = root / config["data"]["calibration_pack"]
    profile_path = root / config["data"]["constraint_profile"]
    current = {
        "schema_hash": sha256_file(schema_path),
        "calibration_hash": ModelConfig.from_yaml(calibration_path).calibration_pack_hash,
        "constraint_profile_hash": ConstraintProfile.from_yaml(profile_path).source_hash,
    }
    provenance = manifest["provenance"]
    for name, value in current.items():
        if provenance[name] != value:
            raise Geometry19CFMError(f"Dataset provenance mismatch: {name}")
    return {
        **current,
        "dataset_manifest_hash": sha256_file(manifest_path),
        "regression_dataset_hash": sha256_file(root / config["data"]["regression_dataset"]),
    }


def train_geometry19_cfm4(
    config_path: str | Path,
    *,
    smoke: bool = False,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Train and validate the isolated Geometry19 CFM-4 seed-43 model."""

    root = Path(repository_root or Path.cwd()).resolve()
    source = Path(config_path).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    version = str(config.get("version"))
    allowed_versions = {
        "wp19_23_geometry19_cfm4_seed43_v1",
        "wp19_23_geometry19_cfm4_bounded_sigmoid_seed43_v1",
        "wp19_23_geometry19_cfm4_performance_consistency_v1",
        "wp19_23_geometry19_cfm4_tolerance_set_v1",
    }
    if version not in allowed_versions:
        raise Geometry19CFMError("Geometry19 CFM config version mismatch")
    protocol = load_wp19_23_contract(root / config["protocol"])
    tolerance_config = dict(config.get("tolerance_set", {}))
    if version == "wp19_23_geometry19_cfm4_tolerance_set_v1":
        if (
            not bool(tolerance_config.get("enabled", False))
            or tolerance_config.get("construction")
            != "guaranteed_relative_tolerance_box"
            or tolerance_config.get("target_formula")
            != "actual_divided_by_one_plus_delta"
            or tolerance_config.get("apply_to") != "train_only"
        ):
            raise Geometry19CFMError("Tolerance-set construction differs from protocol")
        frozen_tolerances = protocol.payload["diagnostic_acceptance"][
            "relative_tolerances"
        ]
        expected_tolerances = [
            float(frozen_tolerances[name]) for name in CONDITION_NAMES
        ]
        if not np.allclose(
            tolerance_config.get("relative_tolerances", []),
            expected_tolerances,
            rtol=0.0,
            atol=0.0,
        ):
            raise Geometry19CFMError("Tolerance-set values differ from acceptance contract")
    model_config = dict(config["model"])
    baseline_tag = protocol.generator_variant("baseline")["model_tag"]
    bounded_ablation = bool(model_config.get("bounded_output", {}).get("enabled"))
    model_identity_valid = (
        model_config.get("model_tag") != baseline_tag
        and model_config.get("ablation_of_model_tag") == baseline_tag
        and model_config.get("bounded_output", {}).get("enabled") is True
        and model_config.get("bounded_output", {}).get("kind") == "offset_sigmoid"
    ) if bounded_ablation else model_config.get("model_tag") == baseline_tag
    if (
        not model_identity_valid
        or model_config.get("design_dim") != 19
        or model_config.get("condition_dim") != 3
        or model_config.get("architecture") != "film_residual"
        or model_config.get("matcher") != "gpu_sinkhorn"
    ):
        raise Geometry19CFMError("Geometry19 CFM model differs from frozen protocol")
    training = dict(config["training"])
    training_seed = int(training["seed"])
    allowed_training_seeds = (
        {42, 43, 44}
        if version == "wp19_23_geometry19_cfm4_tolerance_set_v1"
        else {43}
    )
    if training_seed not in allowed_training_seeds:
        raise Geometry19CFMError(
            "Geometry19 CFM training seed is outside the frozen experiment protocol"
        )
    maximum = None
    batch_size = int(config["data"]["batch_size"])
    run_kind = "baseline"
    if smoke:
        run_kind = "smoke"
        training.update(
            device=config["smoke"]["device"],
            epochs=config["smoke"]["epochs"],
            minimum_epochs=config["smoke"]["minimum_epochs"],
            patience=config["smoke"]["patience"],
        )
        maximum = int(config["smoke"]["maximum_rows_per_split"])
        batch_size = int(config["smoke"]["batch_size"])
        model_config["ode_steps"] = int(config["smoke"]["ode_steps"])
    _seed_all(int(training["seed"]))
    provenance = _validate_provenance(root, config)
    schema = yaml.safe_load((root / config["data"]["design_schema"]).read_text(encoding="utf-8"))
    feature_names = [item["name"] for item in schema["design_variables"]]
    if len(feature_names) != 19:
        raise Geometry19CFMError("Geometry19 schema must contain 19 features")
    splits = load_geometry19_cfm_splits(
        root / config["data"]["regression_dataset"],
        feature_names,
        config["data"]["campaign_split"],
        maximum,
    )
    normalizer = build_geometry19_normalizer(
        schema,
        splits["train"],
        config["data"]["condition_transforms"],
    )
    bounded_config = dict(model_config.get("bounded_output", {}))
    if bounded_ablation:
        if bounded_config.get("offset_strategy") != "train_normalized_mean_logit":
            raise Geometry19CFMError("Bounded ablation offset strategy mismatch")
        normalized_train = normalizer.transform_design(splits["train"]["x"])
        bounded_config["offset"] = fit_offset_sigmoid_bias(
            normalized_train,
            float(bounded_config["epsilon"]),
        ).tolist()
        model_config["bounded_output"] = bounded_config
    train_loader, validation_loader, test_dataset = build_geometry19_cfm_loaders(
        splits,
        normalizer,
        batch_size=batch_size,
        num_workers=int(config["data"]["num_workers"]),
        seed=int(training["seed"]),
        tolerance_set=tolerance_config,
    )
    weights_root = root / config["artifacts"]["weights_root"] / run_kind
    runtime_root = root / config["artifacts"]["runtime_root"] / run_kind
    if weights_root.exists() or runtime_root.exists():
        raise Geometry19CFMError(
            f"Refusing to overwrite Geometry19 CFM artifacts: {run_kind}"
        )
    weights_root.mkdir(parents=True)
    runtime_root.mkdir(parents=True)
    if int(training.get("snapshot_every", 0)) > 0:
        training["snapshot_directory"] = str(weights_root / "snapshots")
    normalizer.save(weights_root / "normalizer.json")
    try:
        model = CFMGenerator(
            model_config,
            training,
            schema_version=str(schema["schema_version"]),
            normalizer=normalizer,
        )
        consistency_config = dict(config.get("performance_consistency", {}))
        consistency = None
        if bool(consistency_config.get("enabled", False)):
            consistency = Geometry19PerformanceConsistencyLoss(
                root,
                model,
                schema,
                protocol.payload,
                consistency_config,
            )
            model.set_training_auxiliary_loss(consistency)
        history = model.fit(train_loader, validation_loader)
        model.set_training_auxiliary_loss(None)
        best_path = weights_root / "best.pt"
        last_path = weights_root / "last.pt"
        model.save(best_path, state="best")
        model.save(last_path, state="last")
        _write_json(weights_root / "training_history.json", history)

        condition = test_dataset.condition[0].to(model.device)
        count = int(
            config["smoke"]["source_consistency_count"]
            if smoke
            else config["validation"]["source_consistency_count"]
        )
        sample_seed = int(config["validation"]["source_seed"])
        sampled = model.sample(condition, count, seed=sample_seed, space="normalized")
        sources = sample_cfm_sources(model, count, sample_seed)
        with torch.no_grad():
            explicit = integrate_cfm_sources(model, sources, condition)
        sampled_tensor = sampled.values.to(explicit.device)
        source_max_error = float((explicit - sampled_tensor).abs().max().cpu())
        source_exact = bool(torch.equal(explicit, sampled_tensor))

        loaded = load_generator(best_path, device=str(model.device))
        reloaded = loaded.sample(condition, count, seed=sample_seed, space="normalized")
        reload_exact = bool(torch.equal(sampled.values, reloaded.values))
        metrics = _sample_metrics(sampled.values.numpy())
        metrics["in_bounds_count"] = metrics["count"] - metrics["out_of_bounds_count"]
        metrics["candidate_protocol_n_keep"] = largest_power_of_two_not_above(
            metrics["in_bounds_count"]
        )
        required = config["validation"]
        minimum_in_bounds = int(
            config["smoke"]["minimum_in_bounds_count"]
            if smoke
            else required["minimum_in_bounds_count"]
        )
        passed = (
            metrics["finite"]
            and metrics["in_bounds_count"] >= minimum_in_bounds
            and (source_exact or not required["require_exact_source_integration_match"])
            and (reload_exact or not required["require_checkpoint_reload_match"])
        )
        summary = {
            "work_package": (
                "WP19-23-PC"
                if version == "wp19_23_geometry19_cfm4_performance_consistency_v1"
                else (
                    "WP19-23-TS"
                    if version == "wp19_23_geometry19_cfm4_tolerance_set_v1"
                    else ("WP19-23-BS" if bounded_ablation else "WP19-23A")
                )
            ),
            "status": "PASS" if passed else "AUDIT_FAILED",
            "run_kind": run_kind,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_tier": "development_bootstrap",
            "formal_use_allowed": False,
            "model_tag": model_config["model_tag"],
            "architecture_id": model_config["architecture_id"],
            "bounded_output": model.design_transform.to_config(),
            "performance_consistency": {
                "enabled": consistency is not None,
                "weight": consistency.weight if consistency is not None else 0.0,
                "objective": consistency_config.get("objective"),
                "integer_projection": consistency_config.get("integer_projection"),
                "forward_weights": (
                    str(consistency.forward_weights.relative_to(root))
                    if consistency is not None
                    else None
                ),
            },
            "tolerance_set": dict(config.get("tolerance_set", {})),
            "training_seed": int(training["seed"]),
            "device": str(model.device),
            "feature_names": feature_names,
            "condition_names": list(CONDITION_NAMES),
            "split_counts": {name: len(bucket["x"]) for name, bucket in splits.items()},
            "training": {
                key: history[key]
                for key in (
                    "best_epoch",
                    "best_validation_loss",
                    "epochs_completed",
                    "early_stopped",
                    "termination_reason",
                    "elapsed_seconds",
                    "peak_gpu_memory_bytes",
                    "total_parameters",
                )
            },
            "validation": {
                "sample_metrics": metrics,
                "source_integration_exact": source_exact,
                "source_integration_max_absolute_error": source_max_error,
                "checkpoint_reload_exact": reload_exact,
                "source_seed": sample_seed,
                "sample_count": count,
            },
            "artifacts": {
                "best_checkpoint": str(best_path.relative_to(root)),
                "last_checkpoint": str(last_path.relative_to(root)),
                "normalizer": str((weights_root / "normalizer.json").relative_to(root)),
            },
            "hashes": {
                **provenance,
                "training_config_hash": sha256_file(source),
                "protocol_hash": sha256_file(root / config["protocol"]),
                "best_checkpoint_hash": sha256_file(best_path),
                "normalizer_hash": sha256_file(weights_root / "normalizer.json"),
                "forward_surrogate_manifest_hash": (
                    sha256_file(consistency.forward_weights / "model_manifest.json")
                    if consistency is not None
                    else None
                ),
                "code_commit": _git_commit(root),
            },
        }
        _write_json(runtime_root / "summary.json", summary)
        if not smoke:
            _write_json(root / config["artifacts"]["summary_json"], summary)
        if not passed:
            raise Geometry19CFMError("Geometry19 CFM validation Gate failed")
        return summary
    except Exception as exc:
        _write_json(
            runtime_root / "failure.json",
            {
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
