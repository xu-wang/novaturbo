"""Equation-level tests for the Geometry-aware V2.1 turbine/shaft model."""

from dataclasses import replace
import math
from pathlib import Path

import pytest
import yaml

from src.physics import (
    FailureCode,
    Geometry19,
    ModelConfig,
    StationState,
    TurbineModelError,
    derive_geometry_state,
    solve_combustor_loworder,
    solve_compressor_meanline,
    solve_turbine_meanline,
)
from src.physics.model_config import CalibrationPackError


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model_config() -> ModelConfig:
    return ModelConfig.from_yaml(
        REPO_ROOT
        / "config"
        / "physics"
        / "geometry19_calibration_bootstrap_v1.yaml"
    )


@pytest.fixture(scope="module")
def cycle_inputs(model_config):
    schema = yaml.safe_load(
        (
            REPO_ROOT / "config" / "physics" / "geometry19_schema_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    design = Geometry19.from_mapping(
        {
            item["name"]: item["reference_value"]
            for item in schema["design_variables"]
        }
    )
    geometry = derive_geometry_state(design, model_config)
    inlet = StationState(total_temperature_K=288.15, total_pressure_Pa=99300.0)
    compressor = solve_compressor_meanline(geometry, 0.15, inlet, model_config)
    combustor = solve_combustor_loworder(
        geometry, 0.15, compressor.outlet, model_config
    )
    return geometry, compressor, combustor


@pytest.fixture(scope="module")
def turbine_result(cycle_inputs, model_config):
    geometry, compressor, combustor = cycle_inputs
    return solve_turbine_meanline(
        geometry, combustor, compressor.shaft_power_W, model_config
    )


def test_reference_turbine_produces_physical_complete_state(turbine_result):
    result = turbine_result
    assert result.outlet.total_temperature_K < result.inlet.total_temperature_K
    assert result.outlet.total_pressure_Pa < result.inlet.total_pressure_Pa
    assert result.expansion_pressure_ratio > 1.0
    assert 0.0 < result.total_to_total_efficiency < 1.0
    assert result.flow_coefficient > 0.0
    assert result.loading_coefficient > 0.0
    assert result.reynolds_number > 0.0
    assert result.corrected_mass_flux_kg_s_m2 > 0.0
    assert result.mean_radius_mach > 0.0
    assert "turbine_mean_radius_mach_max" in result.constraint_residuals
    assert "turbine_tip_mach_max" not in result.constraint_residuals
    assert result.feasible is True


def test_shaft_power_balance_closes(turbine_result):
    result = turbine_result
    extracted_power = (
        result.mechanical_efficiency
        * result.mass_flow_gas_kg_s
        * result.specific_work_J_kg
    )
    required_power = result.compressor_power_W + result.auxiliary_power_W
    assert extracted_power == pytest.approx(required_power)
    assert abs(result.shaft_power_residual_W) < 1.0e-9


def test_turbine_temperature_and_pressure_relations_close(
    turbine_result, model_config
):
    result = turbine_result
    cp = float(model_config.require("cp_hot_J_kg_K"))
    gamma = float(model_config.require("gamma_hot"))
    expected_temperature = (
        result.inlet.total_temperature_K - result.specific_work_J_kg / cp
    )
    assert result.outlet.total_temperature_K == pytest.approx(expected_temperature)
    expected_isentropic = result.inlet.total_temperature_K - (
        result.inlet.total_temperature_K - result.outlet.total_temperature_K
    ) / result.total_to_total_efficiency
    assert result.isentropic_outlet_temperature_K == pytest.approx(
        expected_isentropic
    )
    expected_pressure = result.inlet.total_pressure_Pa * (
        expected_isentropic / result.inlet.total_temperature_K
    ) ** (gamma / (gamma - 1.0))
    assert result.outlet.total_pressure_Pa == pytest.approx(expected_pressure)


def test_corrected_mass_flux_retains_units_and_frozen_reference(
    cycle_inputs, turbine_result
):
    geometry, _, _ = cycle_inputs
    result = turbine_result
    expected = (
        result.mass_flow_gas_kg_s
        * math.sqrt(
            result.inlet.total_temperature_K
            / result.corrected_mass_flux_reference_temperature_K
        )
        / (
            (
                result.inlet.total_pressure_Pa
                / result.corrected_mass_flux_reference_pressure_Pa
            )
            * geometry.turbine_annulus_area_m2
        )
    )
    assert result.corrected_mass_flux_kg_s_m2 == pytest.approx(expected)
    assert result.corrected_mass_flux_reference_temperature_K == pytest.approx(
        288.15
    )
    assert result.corrected_mass_flux_reference_pressure_Pa == pytest.approx(
        101325.0
    )


def test_insufficient_turbine_enthalpy_is_hard_failure(cycle_inputs, model_config):
    geometry, _, combustor = cycle_inputs
    with pytest.raises(TurbineModelError) as exc_info:
        solve_turbine_meanline(geometry, combustor, 1.0e8, model_config)
    assert exc_info.value.failure_code == FailureCode.TURBINE_POWER_INSUFFICIENT


def test_invalid_efficiency_is_rejected_instead_of_clipped(
    cycle_inputs, model_config
):
    geometry, compressor, combustor = cycle_inputs
    invalid_values = dict(model_config.parameter_values)
    invalid_values["turbine_eta_design"] = -0.1
    with pytest.raises(CalibrationPackError, match="outside valid_range"):
        replace(model_config, parameter_values=invalid_values)


def test_constraint_residuals_use_nonpositive_feasible_convention(turbine_result):
    assert turbine_result.constraint_residuals
    assert all(
        math.isfinite(value)
        for value in turbine_result.constraint_residuals.values()
    )
    assert all(value <= 0.0 for value in turbine_result.constraint_residuals.values())
