import torch

from src.ai.generators.torchcfm_adapter import (
    CFMGenerator,
    ConditionalFiLMResidualVelocityMLP,
    ConditionalResidualVelocityMLP,
    FiLMResidualMLPBlock,
)


TIME_EMBEDDING = {
    "enabled": True,
    "dim": 32,
    "max_period": 10000.0,
    "time_scale": 1000.0,
}


def test_film_block_identity_modulation_matches_plain_residual_block_math():
    block = FiLMResidualMLPBlock(4)
    hidden = torch.randn(3, 4)
    zero = torch.zeros_like(hidden)
    expected = hidden + block.linear_2(block.activation(block.linear_1(hidden)))
    assert torch.equal(block(hidden, zero, zero), expected)


def test_cfm_4_zero_initialized_film_starts_as_cfm_3():
    torch.manual_seed(12)
    cfm_3 = ConditionalResidualVelocityMLP(
        256, 2, 10, 3, time_embedding=TIME_EMBEDDING
    )
    cfm_4 = ConditionalFiLMResidualVelocityMLP(
        256, 2, 128, 10, 3, time_embedding=TIME_EMBEDDING
    )
    with torch.no_grad():
        cfm_4.input_projection.load_state_dict(cfm_3.input_projection.state_dict())
        cfm_4.output_projection.load_state_dict(cfm_3.output_projection.state_dict())
        for film_block, residual_block in zip(cfm_4.blocks, cfm_3.blocks):
            film_block.linear_1.load_state_dict(residual_block.linear_1.state_dict())
            film_block.linear_2.load_state_dict(residual_block.linear_2.state_dict())
    x = torch.randn(7, 10)
    condition = torch.randn(7, 3)
    t = torch.rand(7)
    assert torch.equal(cfm_4(t, x, condition), cfm_3(t, x, condition))
    assert torch.count_nonzero(cfm_4.film_projection.weight) == 0
    assert torch.count_nonzero(cfm_4.film_projection.bias) == 0


def test_cfm_4_structure_parameter_count_and_backward():
    model = ConditionalFiLMResidualVelocityMLP(
        256, 2, 128, 10, 3, time_embedding=TIME_EMBEDDING
    )
    assert model.input_projection.in_features == 45
    assert model.condition_encoder[0].in_features == 35
    assert model.condition_encoder[0].out_features == 128
    assert model.film_projection.out_features == 2 * 2 * 256
    assert len(model.blocks) == 2
    assert all(isinstance(block, FiLMResidualMLPBlock) for block in model.blocks)
    assert sum(parameter.numel() for parameter in model.parameters()) == 414218
    output = model(torch.tensor(0.5), torch.randn(7, 10), torch.randn(7, 3))
    assert output.shape == (7, 10)
    output.square().mean().backward()
    assert model.film_projection.weight.grad is not None
    assert torch.isfinite(model.film_projection.weight.grad).all()


def test_cfm_4_architecture_gate_freezes_film_dimensions():
    runtime = {"seed": 42, "device": "cpu", "epochs": 1, "patience": 1}
    common = {
        "architecture": "film_residual",
        "design_dim": 10,
        "condition_dim": 3,
        "hidden_dim": 256,
        "residual_blocks": 2,
        "time_embedding": TIME_EMBEDDING,
    }
    model = CFMGenerator({**common, "condition_encoder_dim": 128}, runtime)
    assert model.architecture == "film_residual"
    assert sum(parameter.numel() for parameter in model.velocity.parameters()) == 414218

    try:
        CFMGenerator({**common, "condition_encoder_dim": 64}, runtime)
    except ValueError as exc:
        assert "condition_encoder_dim=128" in str(exc)
    else:
        raise AssertionError("non-CFM-4 FiLM dimension was accepted")
