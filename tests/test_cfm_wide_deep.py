import torch

from src.ai.generators.torchcfm_adapter import CFMGenerator, ConditionalVelocityMLP


TIME_EMBEDDING = {
    "enabled": True,
    "dim": 32,
    "max_period": 10000.0,
    "time_scale": 1000.0,
}


def test_cfm_2_changes_only_plain_mlp_width_and_depth():
    cfm_1 = ConditionalVelocityMLP(
        [128, 128, 128], 10, 3, time_embedding=TIME_EMBEDDING
    )
    cfm_2 = ConditionalVelocityMLP(
        [256, 256, 256, 256], 10, 3, time_embedding=TIME_EMBEDDING
    )
    cfm_1_linears = [layer for layer in cfm_1.network if isinstance(layer, torch.nn.Linear)]
    cfm_2_linears = [layer for layer in cfm_2.network if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in cfm_1_linears] == [
        (45, 128),
        (128, 128),
        (128, 128),
        (128, 10),
    ]
    assert [(layer.in_features, layer.out_features) for layer in cfm_2_linears] == [
        (45, 256),
        (256, 256),
        (256, 256),
        (256, 256),
        (256, 10),
    ]
    assert sum(parameter.numel() for parameter in cfm_2.parameters()) == 211722
    output = cfm_2(torch.tensor(0.5), torch.randn(7, 10), torch.randn(7, 3))
    assert output.shape == (7, 10)
    output.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in cfm_2.parameters()
    )


def test_cfm_2_architecture_gate_requires_exact_width_and_depth():
    runtime = {"seed": 42, "device": "cpu", "epochs": 1, "patience": 1}
    common = {
        "architecture": "wide_deep",
        "design_dim": 10,
        "condition_dim": 3,
        "time_embedding": TIME_EMBEDDING,
    }
    model = CFMGenerator({**common, "hidden_dims": [256] * 4}, runtime)
    assert model.architecture == "wide_deep"
    assert sum(parameter.numel() for parameter in model.velocity.parameters()) == 211722

    try:
        CFMGenerator({**common, "hidden_dims": [256] * 3}, runtime)
    except ValueError as exc:
        assert "hidden_dims=[256]*4" in str(exc)
    else:
        raise AssertionError("non-CFM-2 width/depth was accepted")
