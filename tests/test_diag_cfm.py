"""Tests for the independent zero-anchored Diag-CFM adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.ai.generator_dataset import GeneratorNormalizer, load_schema
from src.ai.generators import DiagCFMGenerator, load_generator
from src.ai.generators.diag_cfm_adapter import DiagCFMVelocityMLP


RUNTIME = {
    "seed": 17,
    "device": "cpu",
    "epochs": 1,
    "minimum_epochs": 1,
    "patience": 1,
    "gradient_clip_norm": 10.0,
}


def _config(**overrides):
    config = {
        "design_dim": 10,
        "condition_dim": 3,
        "hidden_dim": 32,
        "hidden_layers": 2,
        "activation": "leaky_relu",
        "base_distribution": "uniform",
        "base_low": -1.0,
        "base_high": 1.0,
        "ode_method": "euler",
        "ode_steps": 2,
        "self_consistency_diagnostics": True,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-5,
        "bounded_output": {"enabled": False},
    }
    config.update(overrides)
    return config


def _normalizer() -> GeneratorNormalizer:
    schema = load_schema(Path("config/generator_active10d_schema.yaml"))
    rows = []
    for index in range(8):
        fraction = (index + 1) / 9.0
        row = {"sample_id": f"diag-{index}"}
        for field in schema["design_variables"]:
            lower, upper = field["bounds"]
            row[field["name"]] = lower + fraction * (upper - lower)
        row.update(
            {
                "thrust_N": 30.0 + 4.0 * index,
                "tsfc_kg_N_s": 2.0e-5 + 1.0e-6 * index,
                "total_mass_kg": 0.4 + 0.03 * index,
            }
        )
        rows.append(row)
    return GeneratorNormalizer.fit(pd.DataFrame(rows), schema)


def test_diag_cfm_formal_mlp_dimensions_and_parameter_match():
    velocity = DiagCFMVelocityMLP(
        state_dim=13,
        hidden_dim=366,
        hidden_layers=4,
        activation="leaky_relu",
    )
    linear = [layer for layer in velocity.network if isinstance(layer, torch.nn.Linear)]
    assert linear[0].in_features == 14
    assert linear[-1].out_features == 13
    assert len(linear) == 5
    assert sum(parameter.numel() for parameter in velocity.parameters()) == 413227
    assert abs(413227 - 414218) / 414218 < 0.003

    output = velocity(torch.tensor(0.5), torch.randn(7, 13))
    assert output.shape == (7, 13)
    output.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in velocity.parameters()
    )


def test_diag_cfm_paper_scale_mlp_and_zero_one_state_scaling():
    config = _config(
        hidden_dim=1024,
        hidden_layers=4,
        base_low=0.0,
        base_high=1.0,
        state_scaling={
            "design": "zero_one",
            "condition": "zero_one",
            "condition_input_min": [-2.0, -3.0, -4.0],
            "condition_input_max": [3.0, 2.0, 4.0],
        },
    )
    model = DiagCFMGenerator(config, RUNTIME)
    assert sum(parameter.numel() for parameter in model.parameters()) == 3177485

    normalized_design = torch.tensor([[-1.0] * 10, [1.0] * 10])
    state_design, _, _ = model._design_to_state_space(normalized_design)
    assert torch.equal(state_design[0], torch.zeros(10))
    assert torch.equal(state_design[1], torch.ones(10))
    restored, _, _ = model._design_from_state_space(state_design)
    assert torch.equal(restored, normalized_design)

    condition = torch.tensor([[-2.0, -3.0, -4.0], [3.0, 2.0, 4.0]])
    state_condition = model._condition_to_state_space(condition)
    assert torch.equal(state_condition[0], torch.zeros(3))
    assert torch.equal(state_condition[1], torch.ones(3))
    assert torch.equal(model._condition_from_state_space(state_condition), condition)


def test_diag_cfm_zero_one_condition_scaling_requires_valid_bounds():
    with pytest.raises(ValueError, match="condition_input_min/max"):
        DiagCFMGenerator(
            _config(state_scaling={"condition": "zero_one"}), RUNTIME
        )
    with pytest.raises(ValueError, match="condition_input_max must exceed"):
        DiagCFMGenerator(
            _config(
                state_scaling={
                    "condition": "zero_one",
                    "condition_input_min": [0.0, 0.0, 0.0],
                    "condition_input_max": [1.0, 0.0, 1.0],
                }
            ),
            RUNTIME,
        )


def test_diag_cfm_rejects_cfm4_specific_components():
    with pytest.raises(ValueError, match="does not accept CFM-4 fields"):
        DiagCFMGenerator(_config(matcher="gpu_sinkhorn"), RUNTIME)
    with pytest.raises(ValueError, match="condition_stabilization must be disabled"):
        DiagCFMGenerator(
            _config(condition_stabilization={"enabled": True}), RUNTIME
        )


def test_diag_cfm_batch_loss_sampling_and_intrinsic_diagnostics_are_finite():
    model = DiagCFMGenerator(_config(), RUNTIME)
    batch = {
        "x": 2.0 * torch.rand(12, 10) - 1.0,
        "condition": torch.randn(12, 3),
        "sample_id": [f"sample-{index}" for index in range(12)],
    }
    loss, components = model._batch_loss(batch, training=True)
    assert torch.isfinite(loss)
    assert set(components) == {
        "flow_matching_loss",
        "design_velocity_loss",
        "zero_anchor_velocity_loss",
        "bounded_input_clip_rate",
    }
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    condition = torch.tensor([0.1, -0.2, 0.3])
    first = model.sample(condition, 6, seed=23)
    second = model.sample(condition, 6, seed=23)
    assert first.values.shape == (6, 10)
    assert torch.equal(first.values, second.values)
    assert len(first.diagnostics["candidate_diagnostics"]["zero_deviation"]) == 6
    assert len(first.diagnostics["candidate_diagnostics"]["self_consistency"]) == 6
    assert first.diagnostics["generation_nfe"] == 2
    assert first.diagnostics["diagnosis_nfe"] == 2
    predicted = model.predict_conditions(first.values)
    assert predicted.shape == (6, 3)
    assert torch.isfinite(predicted).all()


def test_diag_cfm_single_batch_fit_checkpoint_load_and_physical_sampling(tmp_path):
    normalizer = _normalizer()
    samples = [
        {
            "x": 2.0 * torch.rand(10) - 1.0,
            "condition": torch.randn(3),
            "sample_id": f"fit-{index}",
        }
        for index in range(16)
    ]
    loader = torch.utils.data.DataLoader(samples, batch_size=16, shuffle=False)
    model = DiagCFMGenerator(
        _config(),
        RUNTIME,
        schema_version="generator_active10d_v1",
        normalizer=normalizer,
    )
    metadata = model.fit(loader, loader)
    assert metadata["epochs_completed"] == 1
    assert metadata["generator_name"] == "diag_cfm"

    checkpoint = tmp_path / "diag-cfm.pt"
    model.save(checkpoint)
    restored = load_generator(checkpoint, device="cpu")
    assert isinstance(restored, DiagCFMGenerator)
    condition = torch.tensor([0.2, -0.1, 0.4])
    expected = model.sample(condition, 5, seed=31)
    actual = restored.sample(condition, 5, seed=31)
    assert torch.equal(expected.values, actual.values)

    physical = restored.sample(condition, 5, seed=31, space="physical")
    assert physical.values.shape == (5, 10)
    assert torch.isfinite(physical.values).all()
