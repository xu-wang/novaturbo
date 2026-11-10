"""Tests for WP19-18 conservation and profile comparison helpers."""

import math

from scripts.run_wp19_18_physics_regression import (
    audit_conservation,
    canonical_json,
    load_tolerances,
)


def test_conservation_accepts_closed_residuals_and_scales_mass_flow():
    tolerances = load_tolerances()
    physics = {
        "operating_point": {"mass_flow_gas_kg_s": 0.2},
        "conservation": {
            "combustor_energy_W": 1.0e-10,
            "shaft_power_W": 1.0e-10,
            "mass_flow_kg_s": 1.0e-7,
            "normalized_flow": 5.0e-7,
            "nozzle_velocity_m_s": 0.0,
        },
    }
    result = audit_conservation(physics, tolerances)
    assert result["passed"] is True
    assert math.isclose(
        result["checks"]["mass_flow_kg_s"]["absolute_tolerance"], 2.0e-6
    )


def test_conservation_rejects_missing_or_large_residual():
    tolerances = load_tolerances()
    physics = {
        "operating_point": {"mass_flow_gas_kg_s": 0.2},
        "conservation": {"normalized_flow": 1.0},
    }
    result = audit_conservation(physics, tolerances)
    assert result["passed"] is False
    assert result["checks"]["combustor_energy_W"]["passed"] is False
    assert result["checks"]["normalized_flow"]["passed"] is False


def test_canonical_json_compares_nan_payloads_deterministically():
    first = {"solver": {"residual": math.nan}, "value": 1}
    second = {"value": 1, "solver": {"residual": math.nan}}
    assert canonical_json(first) == canonical_json(second)
