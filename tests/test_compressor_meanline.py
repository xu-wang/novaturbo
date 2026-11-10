"""Equation-level tests for the Geometry-aware V2.1 compressor model."""

from dataclasses import replace
import math
from pathlib import Path

import pytest
import yaml

from src.physics.compressor_meanline import (
    CompressorModelError,
    solve_compressor_meanline,
    sutherland_dynamic_viscosity,
)
from src.physics.engine_match import FailureCode, StationState
from src.physics.geometry_state import Geometry19, derive_geometry_state
from src.physics.model_config import CalibrationPackError, ModelConfig


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
def geometry(model_config):
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
    return derive_geometry_state(design, model_config)


@pytest.fixture(scope="module")
def inlet():
    return StationState(total_temperature_K=288.15, total_pressure_Pa=99300.0)


def test_sutherland_reference_state_returns_reference_viscosity():
    viscosity = sutherland_dynamic_viscosity(
        273.15, 1.716e-5, 273.15, 110.4
    )
    assert viscosity == pytest.approx(1.716e-5)


def test_reference_compressor_produces_physical_station_state(
    geometry, inlet, model_config
):
    result = solve_compressor_meanline(geometry, 0.15, inlet, model_config)

    assert 0.0 < result.slip_factor < 1.0
    assert result.flow_coefficient_ref == pytest.approx(0.25, rel=0.08)
    assert result.loading_coefficient == pytest.approx(0.72, rel=0.08)
    assert result.total_to_total_efficiency == pytest.approx(0.78, rel=0.02)
    assert result.pressure_ratio > 1.0
    assert result.outlet.total_temperature_K > inlet.total_temperature_K
    assert result.outlet.total_pressure_Pa > inlet.total_pressure_Pa
    assert result.shaft_power_W > 0.0
    assert result.reynolds_number > 1.0e6
    assert result.calibration_domain_feasible is True


def test_euler_work_temperature_pressure_and_power_close_exactly(
    geometry, inlet, model_config
):
    mass_flow = 0.15
    result = solve_compressor_meanline(geometry, mass_flow, inlet, model_config)
    cp = float(model_config.require("cp_cold_J_kg_K"))
    gamma = float(model_config.require("gamma_cold"))

    assert result.specific_work_J_kg == pytest.approx(
        geometry.compressor_tip_speed_m_s
        * result.exit_tangential_velocity_m_s
    )
    assert result.outlet.total_temperature_K - inlet.total_temperature_K == (
        pytest.approx(result.specific_work_J_kg / cp)
    )
    ideal_temperature = (
        inlet.total_temperature_K
        + result.total_to_total_efficiency * result.specific_work_J_kg / cp
    )
    expected_ratio = (ideal_temperature / inlet.total_temperature_K) ** (
        gamma / (gamma - 1.0)
    )
    assert result.pressure_ratio == pytest.approx(expected_ratio)
    assert result.shaft_power_W == pytest.approx(
        mass_flow * result.specific_work_J_kg
    )


def test_mass_flow_changes_phi_and_euler_loading(geometry, inlet, model_config):
    low = solve_compressor_meanline(geometry, 0.10, inlet, model_config)
    high = solve_compressor_meanline(geometry, 0.20, inlet, model_config)
    assert high.flow_coefficient_ref > low.flow_coefficient_ref
    assert high.exit_tangential_velocity_m_s < low.exit_tangential_velocity_m_s
    assert high.loading_coefficient < low.loading_coefficient


def test_nonpositive_mass_flow_is_a_hard_failure(geometry, inlet, model_config):
    with pytest.raises(CompressorModelError) as exc_info:
        solve_compressor_meanline(geometry, 0.0, inlet, model_config)
    assert exc_info.value.failure_code == FailureCode.COMPRESSOR_NONPOSITIVE_WORK


def test_excessive_mass_flow_returns_nonpositive_work_failure(
    geometry, inlet, model_config
):
    with pytest.raises(CompressorModelError) as exc_info:
        solve_compressor_meanline(geometry, 2.0, inlet, model_config)
    assert exc_info.value.failure_code == FailureCode.COMPRESSOR_NONPOSITIVE_WORK


def test_efficiency_is_rejected_instead_of_clipped(geometry, inlet, model_config):
    invalid_values = dict(model_config.parameter_values)
    invalid_values["compressor_eta_design"] = -0.1
    with pytest.raises(CalibrationPackError, match="outside valid_range"):
        replace(model_config, parameter_values=invalid_values)


def test_all_calibration_domain_residuals_use_nonpositive_feasible_convention(
    geometry, inlet, model_config
):
    result = solve_compressor_meanline(geometry, 0.15, inlet, model_config)
    assert result.calibration_domain_residuals
    assert all(
        math.isfinite(value)
        for value in result.calibration_domain_residuals.values()
    )
    assert all(value <= 0.0 for value in result.calibration_domain_residuals.values())
