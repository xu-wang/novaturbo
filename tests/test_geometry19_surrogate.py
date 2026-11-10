"""Contract tests for the independent tagged Geometry19 network family."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.ai.geometry19_surrogate import (
    Geometry19SurrogateError,
    TaggedGeometry19MLP,
    load_weight_bundle,
    save_weight_bundle,
)
from src.ai.geometry19_training import (
    Geometry19TrainingError,
    fit_normalization,
    load_classification_splits,
    load_regression_splits,
)
from src.ai.surrogate import INPUT_FEATURES, OUTPUT_FEATURES, SurrogateConfig, SurrogateModel


FEATURES = [f"x_{index}" for index in range(19)]
CLASSES = [
    "CONVERGED_UNIQUE",
    "CONVERGED_SELECTED_MULTIPLE",
    "NO_FLOW_MATCH",
    "GEOMETRY_INVALID",
    "FLOW_RESIDUAL_DISCONTINUITY",
]
CAMPAIGNS = {
    "train": ["seed_a", "seed_b", "seed_c", "seed_d"],
    "validation": ["seed_e"],
    "test": ["seed_f"],
}


def test_new_models_derive_topology_without_changing_legacy_dimensions() -> None:
    forward = TaggedGeometry19MLP(19, 48)
    feasibility = TaggedGeometry19MLP(19, 5)
    assert forward(torch.zeros(2, 19)).shape == (2, 48)
    assert feasibility(torch.zeros(2, 19)).shape == (2, 5)
    assert forward.hidden_layers == tuple(SurrogateConfig().hidden_layers)

    legacy = SurrogateModel(len(INPUT_FEATURES), len(OUTPUT_FEATURES), SurrogateConfig())
    assert legacy.network[0].in_features == 17
    assert legacy.network[-1].out_features == 9


def test_weight_bundle_is_weights_only_tagged_and_strict(tmp_path: Path) -> None:
    torch.manual_seed(42)
    model = TaggedGeometry19MLP(19, 5)
    model.eval()
    sample = torch.arange(38, dtype=torch.float32).reshape(2, 19)
    expected = model(sample).detach().numpy()
    normalization = {
        "input_mean": np.zeros(19, dtype=np.float32),
        "input_std": np.ones(19, dtype=np.float32),
    }
    manifest = {
        "model_tag": "geometry19_feasibility_test_v1",
        "architecture_tag": "legacy_surrogate_mlp_geometry19_v1",
        "feature_names": FEATURES,
        "target_names": CLASSES,
    }
    save_weight_bundle(tmp_path, model, manifest, normalization)

    state = torch.load(tmp_path / "weights_best_val.pt", weights_only=True)
    assert isinstance(state, dict) and all(isinstance(value, torch.Tensor) for value in state.values())
    loaded, loaded_stats, loaded_manifest = load_weight_bundle(
        tmp_path,
        expected_model_tag=manifest["model_tag"],
        expected_feature_names=FEATURES,
        expected_target_names=CLASSES,
    )
    assert np.array_equal(expected, loaded(sample).detach().numpy())
    assert np.array_equal(loaded_stats["input_mean"], normalization["input_mean"])
    assert loaded_manifest["architecture"]["input_dim"] == 19

    with pytest.raises(Geometry19SurrogateError, match="Model tag mismatch"):
        load_weight_bundle(
            tmp_path,
            expected_model_tag="wrong",
            expected_feature_names=FEATURES,
            expected_target_names=CLASSES,
        )


def _record(campaign: str, index: int) -> dict:
    return {
        "campaign_id": campaign,
        "sample_id": f"{campaign}_{index}",
        "x_19": {name: float(index + offset) for offset, name in enumerate(FEATURES)},
        "physics_outputs": {
            "performance": {
                "thrust_N": 10.0 + index,
                "tsfc_kg_N_s": 0.01 + index,
                "mass_kg": 1.0 + index,
            }
        },
        "constraint_values": {
            "all": {f"c_{constraint:02d}": float(constraint - index) for constraint in range(45)}
        },
        "classification": CLASSES[index % len(CLASSES)],
    }


def test_campaign_loaders_freeze_19d_48d_and_train_only_stats(tmp_path: Path) -> None:
    records = [_record(campaign, index) for index, campaign in enumerate(sum(CAMPAIGNS.values(), []))]
    source = tmp_path / "dataset.jsonl"
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    regression, targets = load_regression_splits(source, FEATURES, CAMPAIGNS)
    classification = load_classification_splits(source, FEATURES, CLASSES, CAMPAIGNS)
    assert regression["train"]["x"].shape == (4, 19)
    assert regression["test"]["y"].shape == (1, 48)
    assert len(targets) == 48 and targets[:3] == ["thrust_N", "tsfc_kg_N_s", "mass_kg"]
    assert classification["validation"]["y"].shape == (1,)

    stats = fit_normalization(regression["train"]["x"], regression["train"]["y"])
    assert np.array_equal(stats["input_mean"], regression["train"]["x"].mean(axis=0))
    assert not np.array_equal(stats["input_mean"], regression["test"]["x"].mean(axis=0))


def test_campaign_overlap_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "dataset.jsonl"
    source.write_text(json.dumps(_record("duplicate", 0)) + "\n", encoding="utf-8")
    with pytest.raises(Geometry19TrainingError, match="multiple splits"):
        load_regression_splits(
            source,
            FEATURES,
            {"train": ["duplicate"], "validation": ["duplicate"], "test": ["other"]},
        )
