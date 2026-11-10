import torch
from torch import nn

from src.ai.generators.torchcfm_adapter import (
    CFMGenerator,
    ConditionalAttentionVelocity,
)


TIME_EMBEDDING = {
    "enabled": True,
    "dim": 32,
    "max_period": 10000.0,
    "time_scale": 1000.0,
}


def test_cfm_7_builds_twelve_lightweight_tokens_and_velocity():
    model = ConditionalAttentionVelocity(
        128, 4, 2, 256, 10, 3, time_embedding=TIME_EMBEDDING
    )
    assert model.design_identity.shape == (10, 128)
    assert model.token_type_embedding.shape == (3, 128)
    assert model.condition_projection.in_features == 3
    assert model.time_projection.in_features == 32
    assert len(model.encoder.layers) == 2
    for layer in model.encoder.layers:
        assert layer.self_attn.embed_dim == 128
        assert layer.self_attn.num_heads == 4
        assert layer.linear1.out_features == 256
        assert layer.dropout.p == 0.0
    output = model(torch.tensor(0.5), torch.randn(7, 10), torch.randn(7, 3))
    assert output.shape == (7, 10)
    output.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_cfm_7_attention_gate_freezes_lightweight_variant():
    runtime = {"seed": 42, "device": "cpu", "epochs": 1, "patience": 1}
    valid = {
        "architecture": "attention",
        "design_dim": 10,
        "condition_dim": 3,
        "model_dim": 128,
        "attention_heads": 4,
        "transformer_layers": 2,
        "feedforward_dim": 256,
        "tokenization": "design_condition_time",
        "time_embedding": TIME_EMBEDDING,
    }
    model = CFMGenerator(valid, runtime)
    assert model.architecture == "attention"
    assert isinstance(model.velocity.encoder.layers[0], nn.TransformerEncoderLayer)
    assert sum(parameter.numel() for parameter in model.velocity.parameters()) == 272001

    for key, value in (
        ("model_dim", 256),
        ("attention_heads", 8),
        ("transformer_layers", 3),
        ("feedforward_dim", 512),
        ("tokenization", "flat"),
    ):
        try:
            CFMGenerator({**valid, key: value}, runtime)
        except ValueError as exc:
            assert "model_dim=128" in str(exc)
        else:
            raise AssertionError(f"non-CFM-7 attention variant accepted: {key}")
