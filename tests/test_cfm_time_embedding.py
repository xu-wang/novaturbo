import torch

from src.ai.generators.torchcfm_adapter import (
    CFMGenerator,
    ConditionalVelocityMLP,
    SinusoidalTimeEmbedding,
)


def test_fixed_time_embedding_shape_and_determinism():
    layer = SinusoidalTimeEmbedding(32, max_period=10000.0, time_scale=1000.0)
    t = torch.tensor([0.0, 0.25, 1.0])
    first = layer(t)
    second = layer(t[:, None])
    assert first.shape == (3, 32)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert not torch.equal(first[0], first[1])
    assert sum(parameter.numel() for parameter in layer.parameters()) == 0


def test_cfm_1_changes_only_time_representation_and_first_layer_capacity():
    baseline = ConditionalVelocityMLP([128, 128, 128], 10, 3)
    embedded = ConditionalVelocityMLP(
        [128, 128, 128],
        10,
        3,
        time_embedding={
            "enabled": True,
            "dim": 32,
            "max_period": 10000.0,
            "time_scale": 1000.0,
        },
    )
    assert baseline.network[0].in_features == 14
    assert embedded.network[0].in_features == 45
    assert baseline.network[0].out_features == embedded.network[0].out_features == 128
    assert len(baseline.network) == len(embedded.network)
    assert sum(p.numel() for p in baseline.parameters()) == 36234
    assert sum(p.numel() for p in embedded.parameters()) == 40202
    output = embedded(torch.tensor(0.5), torch.randn(7, 10), torch.randn(7, 3))
    assert output.shape == (7, 10)
    output.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in embedded.parameters()
    )


def test_cfm_architecture_gate_rejects_inconsistent_time_configuration():
    runtime = {"seed": 42, "device": "cpu", "epochs": 1, "patience": 1}
    try:
        CFMGenerator(
            {
                "architecture": "time_embedding",
                "design_dim": 10,
                "condition_dim": 3,
                "hidden_dims": [128, 128, 128],
                "time_embedding": {"enabled": False},
            },
            runtime,
        )
    except ValueError as exc:
        assert "inconsistent" in str(exc)
    else:
        raise AssertionError("inconsistent CFM-1 configuration was accepted")
