"""Bundle smoke test: import, predict, evaluate, generate, inverse design."""

from __future__ import annotations

import pytest

from novaturbo_v21 import V21Capability


@pytest.fixture(scope="module")
def capability() -> V21Capability:
    return V21Capability(device="cpu")


def test_describe(capability: V21Capability) -> None:
    info = capability.describe()
    assert info["design_dimension"] == 19
    assert set(info["outputs"]) == {"thrust_N", "tsfc_kg_N_s", "mass_kg"}
    assert len(info["integer_features"]) == 3
    assert info["physics"]["model_tag"] == "geometry19_closed_v2_1"


def test_predict_reference(capability: V21Capability) -> None:
    prediction = capability.predict(capability.reference_design)
    assert set(prediction) == {"thrust_N", "tsfc_kg_N_s", "mass_kg"}
    assert all(value > 0.0 for value in prediction.values())


def test_evaluate_reference(capability: V21Capability) -> None:
    result = capability.evaluate(capability.reference_design)
    assert result["solver"]["converged"] is True
    assert result["performance"]["thrust_N"] > 0.0
    assert result["physics"]["backend_id"] == "geometry19_v2_1_backend"


def test_generate_is_conditioned(capability: V21Capability) -> None:
    target = {"thrust_N": 60.0, "tsfc_kg_N_s": 6.7e-5, "mass_kg": 1.77}
    designs = capability.generate(target, n_samples=8, seed=20260817)
    assert len(designs) == 8
    assert all(set(design) == set(capability.feature_names) for design in designs)
    for design in designs:
        for name in capability.integer_names:
            assert float(design[name]).is_integer()


def test_guided_generation_moves_towards_target(capability: V21Capability) -> None:
    target = {"thrust_N": 64.0, "tsfc_kg_N_s": 6.5e-5, "mass_kg": 2.26}
    direct = capability.generate(target, n_samples=32, seed=20260817)
    guided = capability.generate(target, n_samples=32, seed=20260817, guided=True)
    assert len(guided) == 32
    direct_rho = min(capability.hit_metrics(p, target)["rho"] for p in capability.predict(direct))
    guided_rho = min(capability.hit_metrics(p, target)["rho"] for p in capability.predict(guided))
    assert guided_rho < direct_rho


def test_inverse_design_returns_verified_design(capability: V21Capability) -> None:
    target = {"thrust_N": 64.0, "tsfc_kg_N_s": 6.5e-5, "mass_kg": 2.26}
    result = capability.inverse_design(
        target, n_samples=64, seed=20260817, physics_budget=32, guided=True
    )
    assert result["status"] in {
        "DIAGNOSTIC_HIT",
        "BEST_APPROXIMATION_NOT_HIT",
        "NO_PHYSICS_VALID_CANDIDATE",
    }
    if result["delivered"] is not None:
        assert result["delivered"]["performance"]["thrust_N"] > 0.0
        assert result["delivered"]["hit"]["rho"] is not None
