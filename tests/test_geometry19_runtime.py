"""Runtime tests for Geometry19 configuration, geometry, and root matching."""

from dataclasses import replace
import math
from pathlib import Path

import pytest
import yaml

from src.physics.engine_match import (
    FailureCode,
    RootSearchConfig,
    TrialEvaluation,
    find_flow_roots,
)
from src.physics.geometry_state import (
    Geometry19,
    GeometryStateError,
    derive_geometry_state,
)
from src.physics.model_config import CalibrationPackError, ModelConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "config" / "physics" / "geometry19_schema_v1.yaml"
CALIBRATION_PATH = (
    REPO_ROOT
    / "config"
    / "physics"
    / "geometry19_calibration_bootstrap_v1.yaml"
)


def _reference_design() -> Geometry19:
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Geometry19.from_mapping(
        {
            item["name"]: item["reference_value"]
            for item in schema["design_variables"]
        }
    )


@pytest.fixture(scope="module")
def model_config() -> ModelConfig:
    return ModelConfig.from_yaml(CALIBRATION_PATH)


def test_model_config_flattens_parameters_and_hashes_exact_pack(model_config):
    assert model_config.pack_version == "geometry19_calibration_bootstrap_v1"
    assert model_config.formal_dataset_generation_ready is False
    assert len(model_config.parameter_values) == 141
    assert model_config.require("compressor_blade_thickness_ratio") == 0.2
    turbine_reference = model_config.reference_states[
        "turbine_corrected_mass_flux"
    ]
    assert turbine_reference["temperature_K"] == pytest.approx(288.15)
    assert turbine_reference["pressure_Pa"] == pytest.approx(101325.0)
    assert turbine_reference["unit"] == "kg/(s*m^2)"
    assert len(model_config.calibration_pack_hash) == 64
    assert len(model_config.source_file_hash) == 64
    assert model_config.calibration_pack_hash == ModelConfig.from_yaml(
        CALIBRATION_PATH
    ).calibration_pack_hash


def test_reference_design_derives_v21_geometry(model_config):
    design = _reference_design()
    state = derive_geometry_state(design, model_config)

    expected_omega = 2.0 * math.pi * design.shaft_speed_rpm / 60.0
    assert state.omega_rad_s == pytest.approx(expected_omega)
    assert state.compressor_inducer_hub_diameter_m == pytest.approx(0.020)
    assert 0.0 < state.compressor_exit_blockage < 1.0
    assert state.compressor_exit_effective_area_m2 == pytest.approx(
        state.compressor_exit_blockage
        * state.compressor_exit_geometric_area_m2
    )
    assert state.diffuser_outlet_area_m2 == pytest.approx(
        2.5 * state.compressor_exit_effective_area_m2
    )
    assert state.nozzle_exit_area_m2 == state.nozzle_throat_area_m2
    assert set(state.assembly_residuals) == {
        "diffuser_to_combustor_area",
        "combustor_to_turbine_outer_diameter",
        "shaft_clearance",
        "turbine_to_nozzle",
    }


def test_geometry19_rejects_wrong_coordinate_count():
    with pytest.raises(GeometryStateError, match="requires 19 coordinates"):
        Geometry19.from_sequence([1.0] * 18)


def test_geometry_state_rejects_noninteger_blade_count(model_config):
    design = replace(_reference_design(), compressor_blade_count=12.5)
    with pytest.raises(GeometryStateError) as exc_info:
        derive_geometry_state(design, model_config)
    assert exc_info.value.failure_code == FailureCode.GEOMETRY_INVALID.value


def test_geometry_schema_bounds_precede_derived_blockage_check(model_config):
    design = replace(
        _reference_design(),
        compressor_blade_count=200,
        compressor_impeller_exit_width_m=0.012,
    )
    with pytest.raises(GeometryStateError) as exc_info:
        derive_geometry_state(design, model_config)
    assert exc_info.value.failure_code == FailureCode.GEOMETRY_INVALID.value


def test_geometry_schema_rejects_integer_valued_float(model_config):
    design = Geometry19.from_mapping(
        {**_reference_design().as_mapping(), "compressor_blade_count": 12.0}
    )
    with pytest.raises(GeometryStateError, match="exact integer type"):
        derive_geometry_state(design, model_config)


def test_geometry_schema_rejects_out_of_bounds_continuous_value(model_config):
    design = replace(_reference_design(), lattice_relative_density=2.0)
    with pytest.raises(GeometryStateError, match="outside frozen bounds"):
        derive_geometry_state(design, model_config)


def test_effective_config_is_immutable_and_override_changes_hash(model_config):
    with pytest.raises(TypeError):
        model_config.parameter_values["nozzle_discharge_coefficient"] = 0.9
    changed = model_config.with_overrides(
        parameter_values={"nozzle_discharge_coefficient": 0.9}
    )
    assert changed.source_file_hash == model_config.source_file_hash
    assert changed.calibration_pack_hash != model_config.calibration_pack_hash
    assert changed.require("nozzle_discharge_coefficient") == pytest.approx(0.9)


def test_effective_config_rejects_out_of_range_override(model_config):
    with pytest.raises(CalibrationPackError, match="outside valid_range"):
        model_config.with_overrides(
            parameter_values={"compressor_phi_design": 0.0}
        )


def _valid_trial(function):
    return lambda mass_flow: TrialEvaluation.valid_state(
        mass_flow, function(mass_flow)
    )


def test_flow_match_finds_one_sign_change_root():
    result = find_flow_roots(
        _valid_trial(lambda value: value - 1.0),
        0.2,
        2.0,
        RootSearchConfig(linear_scan_points=16, logarithmic_scan_points=16),
    )
    assert result.converged is True
    assert result.failure_code is None
    assert len(result.roots) == 1
    assert result.roots[0].mass_flow_air_kg_s == pytest.approx(1.0)


def test_flow_match_accepts_exact_endpoint_root():
    result = find_flow_roots(
        _valid_trial(lambda value: value - 0.2),
        0.2,
        2.0,
        RootSearchConfig(linear_scan_points=12, logarithmic_scan_points=12),
    )
    assert result.converged is True
    assert len(result.roots) == 1
    assert result.roots[0].mass_flow_air_kg_s == pytest.approx(0.2)


def test_flow_match_does_not_bridge_a_hard_failure_domain():
    def evaluator(value):
        if 0.85 <= value <= 1.15:
            return TrialEvaluation.hard_failure(
                value, FailureCode.COMPRESSOR_EFFICIENCY_INVALID
            )
        return TrialEvaluation.valid_state(value, value - 1.0)

    result = find_flow_roots(
        evaluator,
        0.2,
        2.0,
        RootSearchConfig(linear_scan_points=65, logarithmic_scan_points=65),
    )
    assert result.converged is False
    assert result.roots == ()
    assert result.failure_code == FailureCode.NO_FLOW_MATCH


def test_flow_match_handles_hard_failure_encountered_inside_brent():
    def evaluator(value):
        if abs(value - 1.0) < 1.0e-10:
            return TrialEvaluation.hard_failure(
                value, FailureCode.COMPRESSOR_EFFICIENCY_INVALID
            )
        return TrialEvaluation.valid_state(value, value - 1.0)

    result = find_flow_roots(
        evaluator,
        0.2,
        2.0,
        RootSearchConfig(linear_scan_points=16, logarithmic_scan_points=16),
    )
    assert result.converged is False
    assert result.failure_code == FailureCode.NO_FLOW_MATCH


def test_flow_match_rejects_nonfinite_evaluator_state():
    result = find_flow_roots(
        _valid_trial(lambda _value: math.nan),
        0.2,
        2.0,
        RootSearchConfig(linear_scan_points=12, logarithmic_scan_points=12),
    )
    assert result.converged is False
    assert result.failure_code == FailureCode.NO_FLOW_MATCH


def test_flow_match_detects_tangency_root():
    result = find_flow_roots(
        _valid_trial(lambda value: (value - 1.0) ** 2),
        0.5,
        1.5,
        RootSearchConfig(
            linear_scan_points=10,
            logarithmic_scan_points=10,
            residual_tolerance=1.0e-9,
            root_relative_tolerance=1.0e-8,
            tangency_refine_threshold=1.0e-2,
        ),
    )
    assert result.converged is True
    assert len(result.roots) == 1
    assert result.roots[0].mass_flow_air_kg_s == pytest.approx(1.0, rel=1.0e-5)
    assert abs(result.roots[0].flow_residual) <= 1.0e-9


def test_flow_match_rejects_multiple_operating_points():
    result = find_flow_roots(
        _valid_trial(lambda value: (value - 0.7) * (value - 1.3)),
        0.2,
        2.0,
        RootSearchConfig(linear_scan_points=32, logarithmic_scan_points=32),
    )
    assert result.converged is False
    assert len(result.roots) == 2
    assert result.failure_code == FailureCode.MULTIPLE_OPERATING_POINTS


def test_flow_match_returns_no_match_for_nonzero_residual():
    result = find_flow_roots(
        _valid_trial(lambda value: value + 1.0),
        0.2,
        2.0,
        RootSearchConfig(linear_scan_points=12, logarithmic_scan_points=12),
    )
    assert result.converged is False
    assert result.failure_code == FailureCode.NO_FLOW_MATCH


def test_flow_match_rejects_sign_changing_discontinuity():
    result = find_flow_roots(
        _valid_trial(lambda value: -1.0 if value < 1.0 else 1.0),
        0.2,
        2.0,
        RootSearchConfig(linear_scan_points=16, logarithmic_scan_points=16),
    )
    assert result.converged is False
    assert result.roots == ()
    assert result.failure_code == FailureCode.FLOW_RESIDUAL_DISCONTINUITY


def test_invalid_root_search_settings_return_typed_failure():
    result = find_flow_roots(
        _valid_trial(lambda value: value - 1.0),
        0.2,
        2.0,
        RootSearchConfig(linear_scan_points=2, logarithmic_scan_points=2),
    )
    assert result.converged is False
    assert result.failure_code == FailureCode.ROOT_SOLVER_FAILED


@pytest.mark.parametrize(
    "override",
    (
        {"linear_scan_points": 1_000_000_000},
        {"mass_flow_residual_epsilon_kg_s": 1.0e100},
        {"residual_tolerance": 2.0},
        {"root_relative_tolerance": 2.0},
        {"linear_scan_points": 64.0},
        {"residual_tolerance": "1e-5"},
    ),
)
def test_standalone_root_search_rejects_unsafe_settings_before_allocation(override):
    search = replace(RootSearchConfig(), **override)
    result = find_flow_roots(
        _valid_trial(lambda value: value - 1.0), 0.2, 2.0, search
    )
    assert result.converged is False
    assert result.evaluations == 0
    assert result.failure_code == FailureCode.ROOT_SOLVER_FAILED
