"""Consistency tests for the Geometry-aware V2.1 convergent nozzle."""

from dataclasses import replace
import math
from pathlib import Path

import pytest
import yaml

from src.physics import (
    Environment,
    FailureCode,
    Geometry19,
    ModelConfig,
    NozzleModelError,
    StationState,
    compressible_mass_flow_function,
    derive_geometry_state,
    solve_combustor_loworder,
    solve_compressor_meanline,
    solve_convergent_nozzle,
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
def turbine_inputs(model_config):
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
    turbine = solve_turbine_meanline(
        geometry, combustor, compressor.shaft_power_W, model_config
    )
    return geometry, turbine


def test_mass_flow_function_is_positive_and_peaks_at_sonic_state():
    gamma = 1.33
    gas_constant = 287.05
    subsonic = compressible_mass_flow_function(0.7, gamma, gas_constant)
    sonic = compressible_mass_flow_function(1.0, gamma, gas_constant)
    assert 0.0 < subsonic < sonic


def test_reference_nozzle_is_subsonic_and_back_pressure_matched(
    turbine_inputs, model_config
):
    geometry, turbine = turbine_inputs
    environment = Environment(temperature_K=288.15, pressure_Pa=101325.0)
    result = solve_convergent_nozzle(
        geometry, turbine, environment, model_config
    )
    assert result.choked is False
    assert 0.0 < result.exit.mach < 1.0
    assert result.exit.static_pressure_Pa == pytest.approx(environment.pressure_Pa)
    assert result.pressure_thrust_N == pytest.approx(0.0)
    assert result.nozzle_total.total_pressure_Pa == pytest.approx(
        0.98 * turbine.outlet.total_pressure_Pa
    )
    assert result.mass_flow_capacity_kg_s > 0.0


def test_choked_nozzle_has_exact_sonic_velocity_consistency(
    turbine_inputs, model_config
):
    geometry, turbine = turbine_inputs
    environment = Environment(temperature_K=250.0, pressure_Pa=50000.0)
    result = solve_convergent_nozzle(
        geometry, turbine, environment, model_config
    )
    gamma = float(model_config.require("gamma_hot"))
    gas_constant = float(model_config.require("gas_constant_hot_J_kg_K"))
    assert result.choked is True
    assert result.exit.mach == 1.0
    assert result.exit.static_temperature_K == pytest.approx(
        result.nozzle_total.total_temperature_K * 2.0 / (gamma + 1.0)
    )
    assert result.exit.velocity_m_s == pytest.approx(
        math.sqrt(gamma * gas_constant * result.exit.static_temperature_K)
    )
    assert result.velocity_consistency_residual_m_s == pytest.approx(0.0)
    assert result.exit.static_pressure_Pa > environment.pressure_Pa
    assert result.pressure_thrust_N > 0.0


def test_back_pressure_equal_to_critical_pressure_is_choked(
    turbine_inputs, model_config
):
    geometry, turbine = turbine_inputs
    gamma = float(model_config.require("gamma_hot"))
    recovered_total_pressure = float(
        model_config.require("nozzle_total_pressure_recovery")
    ) * turbine.outlet.total_pressure_Pa
    critical_pressure = recovered_total_pressure * (
        2.0 / (gamma + 1.0)
    ) ** (gamma / (gamma - 1.0))
    result = solve_convergent_nozzle(
        geometry,
        turbine,
        Environment(pressure_Pa=critical_pressure),
        model_config,
    )
    assert result.choked is True
    assert result.exit.mach == 1.0
    assert result.exit.static_pressure_Pa == pytest.approx(critical_pressure)
    assert result.pressure_thrust_N == pytest.approx(0.0, abs=1.0e-12)


def test_capacity_matches_v21_mass_flow_equation(turbine_inputs, model_config):
    geometry, turbine = turbine_inputs
    result = solve_convergent_nozzle(
        geometry,
        turbine,
        Environment(pressure_Pa=101325.0),
        model_config,
    )
    gamma = float(model_config.require("gamma_hot"))
    gas_constant = float(model_config.require("gas_constant_hot_J_kg_K"))
    expected = (
        result.discharge_coefficient
        * geometry.nozzle_throat_area_m2
        * result.nozzle_total.total_pressure_Pa
        / math.sqrt(result.nozzle_total.total_temperature_K)
        * compressible_mass_flow_function(result.exit.mach, gamma, gas_constant)
    )
    assert result.mass_flow_capacity_kg_s == pytest.approx(expected)
    assert result.mass_flow_residual_kg_s == pytest.approx(
        turbine.mass_flow_gas_kg_s - expected
    )


def test_discharge_coefficient_changes_capacity_not_exit_energy_state(
    turbine_inputs, model_config
):
    geometry, turbine = turbine_inputs
    environment = Environment(pressure_Pa=101325.0)
    baseline = solve_convergent_nozzle(
        geometry, turbine, environment, model_config
    )
    values = dict(model_config.parameter_values)
    values["nozzle_discharge_coefficient"] = 0.90
    changed_config = replace(model_config, parameter_values=values)
    changed = solve_convergent_nozzle(
        geometry, turbine, environment, changed_config
    )
    assert changed.exit == baseline.exit
    assert changed.mass_flow_capacity_kg_s / baseline.mass_flow_capacity_kg_s == (
        pytest.approx(0.90 / 0.98)
    )


def test_no_usable_total_pressure_is_hard_failure(turbine_inputs, model_config):
    geometry, turbine = turbine_inputs
    with pytest.raises(NozzleModelError) as exc_info:
        solve_convergent_nozzle(
            geometry,
            turbine,
            Environment(pressure_Pa=200000.0),
            model_config,
        )
    assert exc_info.value.failure_code == FailureCode.NOZZLE_STATE_INVALID


def test_invalid_pressure_recovery_is_rejected(turbine_inputs, model_config):
    geometry, turbine = turbine_inputs
    values = dict(model_config.parameter_values)
    values["nozzle_total_pressure_recovery"] = 1.01
    with pytest.raises(CalibrationPackError, match="outside valid_range"):
        replace(model_config, parameter_values=values)


def test_normalized_flow_residual_uses_frozen_denominator_epsilon(
    turbine_inputs, model_config
):
    geometry, turbine = turbine_inputs
    result = solve_convergent_nozzle(
        geometry,
        turbine,
        Environment(pressure_Pa=101325.0),
        model_config,
    )
    epsilon = float(
        model_config.solver_candidate["mass_flow_residual_epsilon_kg_s"]
    )
    assert result.normalized_flow_residual == pytest.approx(
        result.mass_flow_residual_kg_s
        / max(result.mass_flow_gas_kg_s, epsilon)
    )
