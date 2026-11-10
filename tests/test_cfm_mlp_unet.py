import torch

from src.ai.generators.torchcfm_adapter import (
    CFMGenerator,
    ConditionalMLPUNetVelocity,
    MLPUNetDecoderStage,
    ResidualMLPBlock,
)


TIME_EMBEDDING = {
    "enabled": True,
    "dim": 32,
    "max_period": 10000.0,
    "time_scale": 1000.0,
}


def test_decoder_stage_uses_concatenation_skip():
    stage = MLPUNetDecoderStage(256, 128)
    assert stage.up_projection.in_features == 256
    assert stage.up_projection.out_features == 128
    assert stage.fusion.in_features == 256
    assert stage.fusion.out_features == 128
    output = stage(torch.randn(5, 256), torch.randn(5, 128))
    assert output.shape == (5, 128)


def test_cfm_6_is_scaled_mlp_unet_with_fixed_hierarchy():
    model = ConditionalMLPUNetVelocity(
        [32, 64, 128, 256], 1, 10, 3, time_embedding=TIME_EMBEDDING
    )
    assert model.input_projection.in_features == 45
    assert model.input_projection.out_features == 32
    assert [
        (layer.in_features, layer.out_features) for layer in model.encoder_layers
    ] == [(32, 64), (64, 128), (128, 256)]
    assert len(model.bottleneck) == 1
    assert isinstance(model.bottleneck[0], ResidualMLPBlock)
    assert [
        (stage.up_projection.in_features, stage.up_projection.out_features)
        for stage in model.decoder_stages
    ] == [(256, 128), (128, 64), (64, 32)]
    assert sum(parameter.numel() for parameter in model.parameters()) == 263306
    output = model(torch.tensor(0.5), torch.randn(7, 10), torch.randn(7, 3))
    assert output.shape == (7, 10)
    output.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_cfm_6_architecture_gate_freezes_scaled_variant():
    runtime = {"seed": 42, "device": "cpu", "epochs": 1, "patience": 1}
    common = {
        "architecture": "mlp_unet",
        "design_dim": 10,
        "condition_dim": 3,
        "time_embedding": TIME_EMBEDDING,
    }
    valid = {
        **common,
        "encoder_dims": [32, 64, 128, 256],
        "bottleneck_blocks": 1,
        "skip_mode": "concat",
    }
    model = CFMGenerator(valid, runtime)
    assert model.architecture == "mlp_unet"
    assert sum(parameter.numel() for parameter in model.velocity.parameters()) == 263306

    full = {**valid, "encoder_dims": [64, 128, 256, 512]}
    full_model = CFMGenerator(full, runtime)
    assert sum(parameter.numel() for parameter in full_model.velocity.parameters()) == 1046794

    for invalid in (
        {**valid, "encoder_dims": [48, 96, 192, 384]},
        {**valid, "bottleneck_blocks": 2},
        {**valid, "skip_mode": "add"},
    ):
        try:
            CFMGenerator(invalid, runtime)
        except ValueError as exc:
            assert "frozen encoder_dims variant" in str(exc)
        else:
            raise AssertionError("non-CFM-6 MLP-U-Net variant was accepted")
