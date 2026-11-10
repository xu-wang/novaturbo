import torch

from src.ai.generators.torchcfm_adapter import (
    CFMGenerator,
    GPUSinkhornConditionalFlowMatcher,
)


TIME_EMBEDDING = {
    "enabled": True,
    "dim": 32,
    "max_period": 10000.0,
    "time_scale": 1000.0,
}


def model_config(matcher: str):
    return {
        "architecture": "film_residual",
        "architecture_id": "S2-WP4B-CFM-4-GPU-SINKHORN",
        "design_dim": 10,
        "condition_dim": 3,
        "hidden_dim": 256,
        "residual_blocks": 2,
        "condition_encoder_dim": 128,
        "time_embedding": TIME_EMBEDDING,
        "matcher": matcher,
        "sinkhorn_cost_normalization": "batch_mean",
        "sinkhorn_regularization": 0.1,
        "sinkhorn_iterations": 50,
        "sigma": 0.0,
        "ode_steps": 2,
    }


def test_matcher_gate_selects_plain_or_frozen_gpu_sinkhorn():
    runtime = {"seed": 42, "device": "cpu", "epochs": 1, "patience": 1}
    plain = CFMGenerator(model_config("plain"), runtime)
    sinkhorn = CFMGenerator(model_config("gpu_sinkhorn"), runtime)
    assert not isinstance(plain.matcher, GPUSinkhornConditionalFlowMatcher)
    assert isinstance(sinkhorn.matcher, GPUSinkhornConditionalFlowMatcher)
    assert sinkhorn.matcher.regularization == 0.1
    assert sinkhorn.matcher.iterations == 50
    for invalid in (
        {**model_config("sinkhorn" )},
        {**model_config("gpu_sinkhorn"), "sinkhorn_cost_normalization": "none"},
        {**model_config("gpu_sinkhorn"), "sinkhorn_regularization": 0.2},
        {**model_config("gpu_sinkhorn"), "sinkhorn_iterations": 20},
    ):
        try:
            CFMGenerator(invalid, runtime)
        except ValueError:
            pass
        else:
            raise AssertionError("unfrozen CFM matcher was accepted")


def test_gpu_sinkhorn_plan_has_finite_near_uniform_marginals():
    matcher = GPUSinkhornConditionalFlowMatcher(
        sigma=0.0, regularization=0.1, iterations=50
    )
    source = torch.randn(32, 10)
    target = torch.randn(32, 10)
    plan = matcher.transport_plan(source, target)
    expected = torch.full((32,), 1.0 / 32.0)
    assert plan.shape == (32, 32)
    assert plan.device == source.device
    assert torch.isfinite(plan).all()
    assert torch.allclose(plan.sum(), torch.tensor(1.0), atol=1.0e-6)
    assert torch.allclose(plan.sum(dim=0), expected, atol=2.0e-3)
    assert torch.allclose(plan.sum(dim=1), expected, atol=2.0e-3)


def test_gpu_sinkhorn_loss_keeps_reordered_target_condition_paired():
    runtime = {"seed": 42, "device": "cpu", "epochs": 1, "patience": 1}
    model = CFMGenerator(model_config("gpu_sinkhorn"), runtime)
    batch = {
        "x": torch.linspace(-0.8, 0.8, 80).reshape(8, 10),
        "condition": torch.arange(24, dtype=torch.float32).reshape(8, 3),
    }
    captured = {}
    original = model.matcher.guided_sample_location_and_conditional_flow

    def wrapped(source, target, y0=None, y1=None, t=None, return_noise=False):
        values = original(source, target, y0, y1, t, return_noise)
        captured["paired_condition"] = values[4]
        return values

    model.matcher.guided_sample_location_and_conditional_flow = wrapped
    original_velocity = model.velocity.forward

    def velocity(t, x, condition):
        assert torch.equal(condition, captured["paired_condition"])
        return original_velocity(t, x, condition)

    model.velocity.forward = velocity
    loss, components = model._batch_loss(batch, training=True)
    assert torch.isfinite(loss)
    assert torch.isfinite(components["flow_matching_loss"])
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.velocity.parameters()
    )
