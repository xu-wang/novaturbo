"""WP19-23A Geometry19 CFM data and explicit-source contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.ai.generators import CFMGenerator
from src.ai.geometry19_cfm import (
    CONDITION_NAMES,
    Geometry19CFMDataset,
    Geometry19ToleranceSetDataset,
    build_geometry19_normalizer,
    integrate_cfm_sources,
    fit_offset_sigmoid_bias,
    load_geometry19_cfm_splits,
    sample_cfm_sources,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "config" / "physics" / "geometry19_schema_v1.yaml"
CONFIG_PATH = REPO_ROOT / "config" / "experiments" / "geometry19_cfm4_seed43_v1.yaml"


def _write_synthetic_dataset(path: Path, schema: dict) -> None:
    campaigns = [
        "seed_20260826",
        "seed_20260827",
        "seed_20260828",
        "seed_20260829",
        "seed_20260830",
        "seed_20260831",
    ]
    records = []
    for index, campaign in enumerate(campaigns):
        x_19 = {}
        for offset, item in enumerate(schema["design_variables"]):
            lower, upper = item["bounds"]
            fraction = 0.2 + 0.1 * ((index + offset) % 6)
            value = float(lower + fraction * (upper - lower))
            if item["dtype"] == "int64":
                value = int(round(value))
            x_19[item["name"]] = value
        records.append(
            {
                "campaign_id": campaign,
                "campaign_sample_id": index,
                "sample_id": index,
                "x_19": x_19,
                "physics_outputs": {
                    "performance": {
                        "thrust_N": 30.0 + index,
                        "tsfc_kg_N_s": 5.0e-5 + index * 1.0e-6,
                        "mass_kg": 1.0 + 0.1 * index,
                    }
                },
            }
        )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_geometry19_cfm_config_freezes_seed43_sinkhorn_film_contract() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["training"]["seed"] == 43
    assert config["model"]["design_dim"] == 19
    assert config["model"]["condition_dim"] == 3
    assert config["model"]["architecture"] == "film_residual"
    assert config["model"]["matcher"] == "gpu_sinkhorn"
    assert config["model"]["ode_method"] == "rk4"
    assert config["model"]["ode_steps"] == 32


def test_bounded_sigmoid_config_is_independent_and_bias_centers_train_mean() -> None:
    path = (
        REPO_ROOT
        / "config"
        / "experiments"
        / "geometry19_cfm4_bounded_sigmoid_seed43_v1.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    bounded = config["model"]["bounded_output"]
    assert config["model"]["model_tag"] != yaml.safe_load(
        CONFIG_PATH.read_text(encoding="utf-8")
    )["model"]["model_tag"]
    assert bounded["kind"] == "offset_sigmoid"
    values = np.asarray([[-0.5, 0.2], [0.1, 0.6], [0.4, -0.2]])
    bias = fit_offset_sigmoid_bias(values, epsilon=1.0e-4)
    probability = 1.0 / (1.0 + np.exp(-bias))
    decoded_mean = 2.0 * (1.0e-4 + (1.0 - 2.0e-4) * probability) - 1.0
    np.testing.assert_allclose(decoded_mean, values.mean(axis=0), atol=1.0e-12)


def test_campaign_loader_and_train_only_normalizer(tmp_path: Path) -> None:
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    source = tmp_path / "regression.jsonl"
    _write_synthetic_dataset(source, schema)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    feature_names = [item["name"] for item in schema["design_variables"]]
    splits = load_geometry19_cfm_splits(
        source,
        feature_names,
        config["data"]["campaign_split"],
    )
    assert {name: len(bucket["x"]) for name, bucket in splits.items()} == {
        "train": 4,
        "validation": 1,
        "test": 1,
    }
    normalizer = build_geometry19_normalizer(
        schema,
        splits["train"],
        config["data"]["condition_transforms"],
    )
    assert normalizer.fitted_sample_count == 4
    assert normalizer.design_names == feature_names
    assert normalizer.condition_names == list(CONDITION_NAMES)
    dataset = Geometry19CFMDataset(splits["train"], normalizer)
    assert dataset.x.shape == (4, 19)
    assert dataset.condition.shape == (4, 3)
    assert torch.isfinite(dataset.x).all()


def test_explicit_source_integration_matches_production_sampler() -> None:
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    design = [
        {
            "name": item["name"],
            "method": "physical_bounds_to_minus_one_plus_one",
            "physical_min": float(item["bounds"][0]),
            "physical_max": float(item["bounds"][1]),
            "normalized_min": -1.0,
            "normalized_max": 1.0,
        }
        for item in schema["design_variables"]
    ]
    from src.ai.generator_dataset import GeneratorNormalizer

    normalizer = GeneratorNormalizer(
        schema_version=schema["schema_version"],
        design=design,
        conditions=[
            {"name": name, "method": "standard", "transform": "identity", "mean": 0.0, "std": 1.0}
            for name in CONDITION_NAMES
        ],
        fitted_split="train",
        fitted_sample_count=1,
        fitted_sample_id_sha256="0" * 64,
    )
    model_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["model"]
    model_config = dict(model_config, ode_steps=2)
    generator = CFMGenerator(
        model_config,
        {"seed": 43, "device": "cpu"},
        schema_version=schema["schema_version"],
        normalizer=normalizer,
    )
    condition = torch.tensor([0.1, -0.2, 0.3])
    seed = 20260817
    sampled = generator.sample(condition, 4, seed=seed, space="normalized")
    sources = sample_cfm_sources(generator, 4, seed)
    with torch.no_grad():
        explicit = integrate_cfm_sources(generator, sources, condition)
    assert torch.equal(sampled.values, explicit)
    assert np.isfinite(sampled.values.numpy()).all()


def test_cfm_training_auxiliary_loss_is_differentiable_and_not_serialized() -> None:
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    design = [
        {
            "name": item["name"],
            "method": "physical_bounds_to_minus_one_plus_one",
            "physical_min": float(item["bounds"][0]),
            "physical_max": float(item["bounds"][1]),
            "normalized_min": -1.0,
            "normalized_max": 1.0,
        }
        for item in schema["design_variables"]
    ]
    from src.ai.generator_dataset import GeneratorNormalizer

    normalizer = GeneratorNormalizer(
        schema_version=schema["schema_version"],
        design=design,
        conditions=[
            {"name": name, "method": "standard", "transform": "identity", "mean": 0.0, "std": 1.0}
            for name in CONDITION_NAMES
        ],
        fitted_split="train",
        fitted_sample_count=1,
        fitted_sample_id_sha256="0" * 64,
    )
    model_config = dict(
        yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["model"],
        ode_steps=2,
    )
    generator = CFMGenerator(
        model_config,
        {"seed": 43, "device": "cpu"},
        schema_version=schema["schema_version"],
        normalizer=normalizer,
    )

    def auxiliary(endpoint: torch.Tensor, condition: torch.Tensor):
        value = 0.1 * (endpoint.square().mean() + condition.square().mean())
        return value, {"test_auxiliary_loss": value}

    generator.set_training_auxiliary_loss(auxiliary)
    loss, components = generator._batch_loss(
        {"x": torch.zeros(8, 19), "condition": torch.zeros(8, 3)},
        training=True,
    )
    assert torch.isfinite(loss)
    assert torch.isclose(
        loss,
        components["flow_matching_loss"] + components["test_auxiliary_loss"],
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in generator.parameters())
    assert not any(key.startswith("_training_auxiliary_loss") for key in generator.state_dict())


def test_tolerance_set_dataset_generates_only_guaranteed_valid_targets() -> None:
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    bucket = {
        "x": np.asarray(
            [[0.5 * (item["bounds"][0] + item["bounds"][1]) for item in schema["design_variables"]]],
            dtype=np.float64,
        ),
        "condition": np.asarray([[50.0, 7.0e-5, 2.0]], dtype=np.float64),
        "sample_ids": ["sample:0"],
    }
    from src.ai.generator_dataset import GeneratorNormalizer

    normalizer = GeneratorNormalizer(
        schema_version=schema["schema_version"],
        design=[
            {
                "name": item["name"],
                "method": "physical_bounds_to_minus_one_plus_one",
                "physical_min": float(item["bounds"][0]),
                "physical_max": float(item["bounds"][1]),
                "normalized_min": -1.0,
                "normalized_max": 1.0,
            }
            for item in schema["design_variables"]
        ],
        conditions=[
            {"name": name, "method": "standard", "transform": "identity", "mean": 0.0, "std": 1.0}
            for name in CONDITION_NAMES
        ],
        fitted_split="train",
        fitted_sample_count=1,
        fitted_sample_id_sha256="0" * 64,
    )
    tolerances = [0.03, 0.01, 0.02]
    dataset = Geometry19ToleranceSetDataset(bucket, normalizer, tolerances, 1.0)
    torch.manual_seed(123)
    target = dataset[0]["condition"].numpy()
    actual = bucket["condition"][0]
    rho_axes = np.abs(actual - target) / (np.asarray(tolerances) * target)
    assert np.max(rho_axes) <= 1.0 + 1.0e-6
