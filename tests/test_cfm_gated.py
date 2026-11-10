import torch

from src.ai.generators.torchcfm_adapter import (
    CFMGenerator,
    ConditionalGatedVelocityMLP,
    GatedResidualMLPBlock,
)


TIME_EMBEDDING = {
    "enabled": True,
    "dim": 32,
    "max_period": 10000.0,
    "time_scale": 1000.0,
}


def test_gated_block_uses_independent_value_gate_and_residual_path():
    block = GatedResidualMLPBlock(4)
    hidden = torch.randn(3, 4)
    expected = hidden + block.output_projection(
        block.activation(block.value_projection(hidden))
        * torch.sigmoid(block.gate_projection(hidden))
    )
    assert torch.equal(block(hidden), expected)
    assert block.value_projection.weight.data_ptr() != block.gate_projection.weight.data_ptr()


def test_cfm_5_structure_is_parameter_matched_glu_residual_mlp():
    model = ConditionalGatedVelocityMLP(
        256, 2, 10, 3, time_embedding=TIME_EMBEDDING
    )
    assert model.input_projection.in_features == 45
    assert model.input_projection.out_features == 256
    assert len(model.blocks) == 2
    assert all(isinstance(block, GatedResidualMLPBlock) for block in model.blocks)
    assert model.output_projection.out_features == 10
    assert sum(parameter.numel() for parameter in model.parameters()) == 409098
    output = model(torch.tensor(0.5), torch.randn(7, 10), torch.randn(7, 3))
    assert output.shape == (7, 10)
    output.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_cfm_5_architecture_gate_freezes_width_and_block_count():
    runtime = {"seed": 42, "device": "cpu", "epochs": 1, "patience": 1}
    common = {
        "architecture": "gated",
        "design_dim": 10,
        "condition_dim": 3,
        "hidden_dim": 256,
        "residual_blocks": 2,
        "time_embedding": TIME_EMBEDDING,
    }
    model = CFMGenerator({**common, "gated_blocks": 2}, runtime)
    assert model.architecture == "gated"
    assert sum(parameter.numel() for parameter in model.velocity.parameters()) == 409098

    try:
        CFMGenerator({**common, "gated_blocks": 3}, runtime)
    except ValueError as exc:
        assert "gated_blocks=2" in str(exc)
    else:
        raise AssertionError("non-CFM-5 gated block count was accepted")
