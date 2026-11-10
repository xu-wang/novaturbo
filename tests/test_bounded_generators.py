"""Stage-II WP2 tests for smooth bounded 10D generator outputs."""

from __future__ import annotations

import pytest
import torch

from src.ai.generators import (
    CFMGenerator,
    CINNGenerator,
    CVAEGenerator,
    NFlowsGenerator,
    SmoothBoundedTransform,
    load_generator,
)


RUNTIME = {
    "seed": 71,
    "device": "cpu",
    "epochs": 1,
    "patience": 1,
    "gradient_clip_norm": 10.0,
}

BOUNDED = {
    "enabled": True,
    "epsilon": 1.0e-6,
    "input_tolerance": 1.0e-6,
    "saturation_threshold": 0.99,
}

MODEL_CONFIGS = {
    "cvae": {
        "hidden_dims": [16, 16],
        "latent_dim": 4,
        "beta": 0.001,
        "learning_rate": 0.001,
    },
    "nflows": {
        "num_layers": 2,
        "hidden_features": 16,
        "num_blocks": 1,
        "learning_rate": 0.001,
    },
    "cinn": {
        "num_blocks": 2,
        "hidden_features": 16,
        "affine_clamping": 2.0,
        "learning_rate": 0.001,
    },
    "cfm": {
        "hidden_dims": [16, 16],
        "sigma": 0.0,
        "ode_method": "rk4",
        "ode_steps": 2,
        "ode_rtol": 1.0e-5,
        "ode_atol": 1.0e-5,
        "learning_rate": 0.001,
    },
}

GENERATORS = (
    (CVAEGenerator, "cvae"),
    (NFlowsGenerator, "nflows"),
    (CINNGenerator, "cinn"),
    (CFMGenerator, "cfm"),
)


def _config(name: str):
    return {
        **MODEL_CONFIGS[name],
        "design_dim": 10,
        "condition_dim": 3,
        "bounded_output": BOUNDED,
    }


def test_smooth_bounded_transform_round_trip_and_jacobian():
    transform = SmoothBoundedTransform.from_config(BOUNDED)
    bounded = torch.linspace(-0.95, 0.95, 30).reshape(3, 10)

    unconstrained, forward_log_det, clipped = transform.to_unconstrained(bounded)
    restored, reverse_log_det = transform.to_bounded(unconstrained)

    torch.testing.assert_close(restored, bounded, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(
        forward_log_det + reverse_log_det,
        torch.zeros_like(forward_log_det),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.equal(clipped, torch.zeros_like(clipped))


def test_transform_audits_boundary_stabilization_and_rejects_bad_input():
    transform = SmoothBoundedTransform.from_config(BOUNDED)
    bounded = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])
    unconstrained, log_det, clipped = transform.to_unconstrained(bounded)

    assert torch.isfinite(unconstrained).all()
    assert torch.isfinite(log_det).all()
    assert clipped.item() == pytest.approx(0.4)
    with pytest.raises(ValueError, match="exceeds"):
        transform.to_unconstrained(torch.tensor([[1.01]]))


def test_offset_sigmoid_round_trip_bias_and_strict_margin():
    transform = SmoothBoundedTransform.from_config(
        {
            "enabled": True,
            "kind": "offset_sigmoid",
            "epsilon": 1.0e-4,
            "offset": [-0.7, 0.0, 0.9],
        },
        dimension=3,
    )
    bounded = torch.tensor([[-0.8, 0.1, 0.7], [0.2, -0.4, 0.0]])
    unconstrained, forward_log_det, clipped = transform.to_unconstrained(bounded)
    restored, reverse_log_det = transform.to_bounded(unconstrained)
    torch.testing.assert_close(restored, bounded, atol=2.0e-6, rtol=2.0e-6)
    torch.testing.assert_close(
        forward_log_det + reverse_log_det,
        torch.zeros_like(forward_log_det),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.equal(clipped, torch.zeros_like(clipped))
    zero_decoded, _ = transform.to_bounded(torch.zeros(1, 3))
    assert zero_decoded[0, 0] < zero_decoded[0, 1] < zero_decoded[0, 2]
    assert torch.all(zero_decoded > -1.0 + 1.0e-4)
    assert torch.all(zero_decoded < 1.0 - 1.0e-4)
    boundary_model, _, boundary_clipped = transform.to_unconstrained(
        torch.tensor([[-1.0, 0.0, 1.0]])
    )
    assert torch.isfinite(boundary_model).all()
    assert boundary_clipped.item() == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize(("generator_class", "name"), GENERATORS)
def test_bounded_generators_have_finite_loss_and_strictly_bounded_samples(
    generator_class, name
):
    model = generator_class(_config(name), RUNTIME)
    x = torch.linspace(-1.0, 1.0, 80).reshape(8, 10)
    condition = torch.randn(8, 3, generator=torch.Generator().manual_seed(3))
    loss, components = model._batch_loss(
        {"x": x, "condition": condition}, training=True
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    generated = model.sample(condition[0], 32, seed=91)
    assert generated.values.shape == (32, 10)
    assert torch.all(generated.values > -1.0)
    assert torch.all(generated.values < 1.0)
    assert generated.diagnostics["bounded_output_enabled"] is True
    assert generated.diagnostics["output_out_of_bounds_rate"] == 0.0
    assert generated.diagnostics["output_min_boundary_distance"] > 0.0


@pytest.mark.parametrize(("generator_class", "name"), GENERATORS)
def test_bounded_checkpoint_preserves_samples(generator_class, name, tmp_path):
    model = generator_class(_config(name), RUNTIME)
    condition = torch.tensor([0.1, -0.2, 0.3])
    before = model.sample(condition, 8, seed=17)
    checkpoint = tmp_path / f"bounded-{name}.pt"
    model.save(checkpoint)

    restored = load_generator(checkpoint, device="cpu")
    after = restored.sample(condition, 8, seed=17)

    assert torch.equal(before.values, after.values)
    assert restored.design_transform.enabled is True


def test_bounded_flow_log_prob_and_cinn_inverse_are_finite():
    x = torch.linspace(-1.0, 1.0, 80).reshape(8, 10)
    condition = torch.randn(8, 3, generator=torch.Generator().manual_seed(5))
    nflows = NFlowsGenerator(_config("nflows"), RUNTIME)
    assert torch.isfinite(nflows.log_prob(x, condition)).all()

    cinn = CINNGenerator(_config("cinn"), RUNTIME)
    interior = x.clamp(-0.9, 0.9)
    z, log_det = cinn.forward_transform(interior, condition)
    restored, reverse_log_det = cinn.inverse_transform(z, condition)
    torch.testing.assert_close(restored, interior, atol=2.0e-5, rtol=2.0e-5)
    torch.testing.assert_close(
        log_det + reverse_log_det,
        torch.zeros_like(log_det),
        atol=2.0e-4,
        rtol=2.0e-4,
    )
    assert torch.isfinite(cinn.log_prob(x, condition)).all()
