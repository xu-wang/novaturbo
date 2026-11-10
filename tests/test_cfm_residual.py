import torch

from src.ai.generators.torchcfm_adapter import (
    CFMGenerator,
    ConditionalResidualVelocityMLP,
    ResidualMLPBlock,
)


TIME_EMBEDDING = {
    "enabled": True,
    "dim": 32,
    "max_period": 10000.0,
    "time_scale": 1000.0,
}


def test_residual_block_is_exact_two_layer_skip_connection():
    block = ResidualMLPBlock(4)
    with torch.no_grad():
        block.linear_1.weight.zero_()
        block.linear_1.bias.zero_()
        block.linear_2.weight.zero_()
        block.linear_2.bias.zero_()
    hidden = torch.randn(3, 4)
    assert torch.equal(block(hidden), hidden)


def test_cfm_3_has_fixed_width_two_residual_blocks_only():
    model = ConditionalResidualVelocityMLP(
        256, 2, 10, 3, time_embedding=TIME_EMBEDDING
    )
    assert model.input_projection.in_features == 45
    assert model.input_projection.out_features == 256
    assert len(model.blocks) == 2
    assert all(isinstance(block, ResidualMLPBlock) for block in model.blocks)
    assert all(
        block.linear_1.in_features == block.linear_1.out_features == 256
        and block.linear_2.in_features == block.linear_2.out_features == 256
        for block in model.blocks
    )
    assert model.output_projection.in_features == 256
    assert model.output_projection.out_features == 10
    assert sum(parameter.numel() for parameter in model.parameters()) == 277514
    output = model(torch.tensor(0.5), torch.randn(7, 10), torch.randn(7, 3))
    assert output.shape == (7, 10)
    output.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_cfm_3_architecture_gate_freezes_width_and_block_count():
    runtime = {"seed": 42, "device": "cpu", "epochs": 1, "patience": 1}
    common = {
        "architecture": "residual",
        "design_dim": 10,
        "condition_dim": 3,
        "time_embedding": TIME_EMBEDDING,
    }
    model = CFMGenerator(
        {**common, "hidden_dim": 256, "residual_blocks": 2}, runtime
    )
    assert model.architecture == "residual"
    assert sum(parameter.numel() for parameter in model.velocity.parameters()) == 277514

    for invalid in (
        {"hidden_dim": 128, "residual_blocks": 2},
        {"hidden_dim": 256, "residual_blocks": 3},
    ):
        try:
            CFMGenerator({**common, **invalid}, runtime)
        except ValueError as exc:
            assert "hidden_dim=256" in str(exc)
        else:
            raise AssertionError("non-CFM-3 residual architecture was accepted")
