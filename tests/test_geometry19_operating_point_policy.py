"""Regression tests for the explicit WP19-18 operating-point policy."""

from dataclasses import replace

import pytest

from src.physics import (
    Environment,
    FailureCode,
    Geometry19,
    ModelConfig,
    OperatingPointSelectionPolicy,
    solve_design_point,
)


def _config() -> ModelConfig:
    return ModelConfig.from_yaml(
        "config/physics/geometry19_calibration_bootstrap_v1.yaml"
    )


def _reference() -> Geometry19:
    return Geometry19(
        compressor_impeller_exit_diameter_m=0.080,
        compressor_inducer_tip_diameter_m=0.050,
        compressor_inducer_hub_tip_ratio=0.40,
        compressor_impeller_exit_width_m=0.005,
        compressor_exit_blade_angle_deg=60.0,
        compressor_blade_count=12,
        combustor_outer_diameter_m=0.110,
        combustor_inner_diameter_m=0.060,
        combustor_length_m=0.080,
        combustor_liner_thickness_m=0.0012,
        combustor_injector_count=6,
        turbine_tip_diameter_m=0.085,
        turbine_hub_tip_ratio=0.70,
        turbine_blade_count=17,
        turbine_inlet_total_temperature_K=1100.0,
        nozzle_throat_diameter_m=0.055,
        shaft_speed_rpm=100000.0,
        lattice_cell_size_m=0.008,
        lattice_relative_density=0.30,
    )


def test_default_solver_still_rejects_multiple_roots():
    design = replace(_reference(), compressor_impeller_exit_width_m=0.012)
    result = solve_design_point(design, Environment(), _config())
    assert result.solver.converged is False
    assert result.solver.failure_code == FailureCode.MULTIPLE_OPERATING_POINTS
    assert result.operating_point["flow_root_count"] == 2
    assert result.operating_point["selected_flow_root_index"] is None


def test_reference_continuation_policy_selects_high_flow_root():
    design = replace(_reference(), compressor_impeller_exit_width_m=0.012)
    result = solve_design_point(
        design,
        Environment(),
        _config(),
        operating_point_policy=(
            OperatingPointSelectionPolicy.REFERENCE_CONTINUATION_HIGH_FLOW_V1
        ),
    )
    roots = result.operating_point["flow_root_mass_flow_air_kg_s"]
    assert result.solver.converged is True
    assert result.solver.failure_code is None
    assert result.operating_point["flow_root_count"] == 2
    assert result.operating_point["selected_flow_root_index"] == 1
    assert result.operating_point["mass_flow_air_kg_s"] == pytest.approx(max(roots))
    assert abs(result.conservation.normalized_flow) <= 1.0e-5
    assert "high-flow branch" in result.solver.message


def test_reference_continuation_policy_does_not_invent_no_flow_match():
    design = replace(
        _reference(), compressor_impeller_exit_diameter_m=0.150
    )
    result = solve_design_point(
        design,
        Environment(),
        _config(),
        operating_point_policy=(
            OperatingPointSelectionPolicy.REFERENCE_CONTINUATION_HIGH_FLOW_V1
        ),
    )
    assert result.solver.converged is False
    assert result.solver.failure_code == FailureCode.NO_FLOW_MATCH
    assert result.operating_point["flow_root_count"] == 0
    assert result.operating_point["selected_flow_root_index"] is None
