"""Equation-level tests for the Geometry-aware V2.1 combustor model."""

from dataclasses import replace
import math
from pathlib import Path

import pytest
import yaml

from src.physics import (
    CombustorModelError,
    FailureCode,
    Geometry19,
    GeometryStateError,
    ModelConfig,
    StationState,
    derive_geometry_state,
    solve_combustor_loworder,
    solve_compressor_meanline,
)


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
def reference_design() -> Geometry19:
    schema = yaml.safe_load(
        (
            REPO_ROOT / "config" / "physics" / "geometry19_schema_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    return Geometry19.from_mapping(
        {
            item["name"]: item["reference_value"]
            for item in schema["design_variables"]
        }
    )


@pytest.fixture(scope="module")
def compressor_result(reference_design, model_config):
    geometry = derive_geometry_state(reference_design, model_config)
    inlet = StationState(total_temperature_K=288.15, total_pressure_Pa=99300.0)
    return solve_compressor_meanline(geometry, 0.15, inlet, model_config)


@pytest.fixture(scope="module")
def combustor_result(reference_design, compressor_result, model_config):
    geometry = derive_geometry_state(reference_design, model_config)
    return solve_combustor_loworder(
        geometry, 0.15, compressor_result.outlet, model_config
    )


def test_reference_combustor_produces_positive_complete_state(combustor_result):
    result = combustor_result
    assert result.outlet.total_temperature_K == pytest.approx(1100.0)
    assert result.outlet.total_pressure_Pa < result.inlet.total_pressure_Pa
    assert 0.0 < result.total_pressure_loss_fraction < 0.08
    assert result.mass_flow_fuel_kg_s > 0.0
    assert result.mass_flow_gas_kg_s > result.mass_flow_air_kg_s
    assert result.air_fuel_ratio == pytest.approx(1.0 / result.fuel_air_ratio)
    assert 0.15 <= result.equivalence_ratio <= 0.45
    assert result.primary_zone_residence_time_s >= 0.002
    assert result.fuel_flow_per_injector_kg_s <= (
        result.fuel_flow_capacity_per_injector_kg_s
    )
    assert result.feasible is True


def test_pressure_loss_uses_diffuser_outlet_to_combustor_area(
    reference_design, compressor_result, model_config
):
    geometry = derive_geometry_state(reference_design, model_config)
    result = solve_combustor_loworder(
        geometry, 0.15, compressor_result.outlet, model_config
    )
    ratio = (
        geometry.diffuser_outlet_area_m2
        / geometry.combustor_annulus_area_m2
    )
    expected_k = (
        float(model_config.require("combustor_pressure_loss_K0"))
        + float(model_config.require("combustor_pressure_loss_K_LD"))
        * reference_design.combustor_length_m
        / geometry.combustor_hydraulic_diameter_m
        + float(model_config.require("combustor_pressure_loss_K_area"))
        * (ratio - 1.0) ** 2
    )
    assert result.pressure_loss_coefficient == pytest.approx(expected_k)


def test_common_reference_energy_balance_closes(combustor_result):
    result = combustor_result
    energy_scale = result.mass_flow_gas_kg_s * 1.0e6
    assert abs(result.energy_residual_W) / energy_scale < 1.0e-12


def test_fuel_solution_matches_explicit_enthalpy_formula(
    combustor_result, model_config
):
    result = combustor_result
    reference_temperature = float(
        model_config.require("enthalpy_reference_temperature_K")
    )
    h_air = float(model_config.require("cp_cold_J_kg_K")) * (
        result.inlet.total_temperature_K - reference_temperature
    )
    h_gas = float(model_config.require("cp_hot_J_kg_K")) * (
        result.outlet.total_temperature_K - reference_temperature
    )
    denominator = (
        float(model_config.require("combustion_efficiency"))
        * float(model_config.require("fuel_lower_heating_value_J_kg"))
        + float(model_config.require("fuel_sensible_enthalpy_at_reference_J_kg"))
        - h_gas
    )
    assert result.fuel_air_ratio == pytest.approx((h_gas - h_air) / denominator)


def test_nonzero_fuel_sensible_enthalpy_preserves_energy_balance(
    reference_design, compressor_result, combustor_result, model_config
):
    geometry = derive_geometry_state(reference_design, model_config)
    changed_config = model_config.with_overrides(
        parameter_values={
            "fuel_sensible_enthalpy_at_reference_J_kg": 100000.0
        }
    )
    changed = solve_combustor_loworder(
        geometry, 0.15, compressor_result.outlet, changed_config
    )
    assert changed.fuel_air_ratio < combustor_result.fuel_air_ratio
    assert abs(changed.energy_residual_W) < 1.0e-8


def test_consistent_enthalpy_reference_shift_leaves_fuel_solution_invariant(
    reference_design, compressor_result, combustor_result, model_config
):
    geometry = derive_geometry_state(reference_design, model_config)
    delta_temperature = 10.0
    cp_cold = float(model_config.require("cp_cold_J_kg_K"))
    cp_hot = float(model_config.require("cp_hot_J_kg_K"))
    fuel_air_ratio = combustor_result.fuel_air_ratio
    shifted_fuel_enthalpy = float(
        model_config.require("fuel_sensible_enthalpy_at_reference_J_kg")
    ) - (cp_hot - cp_cold) * delta_temperature / fuel_air_ratio - (
        cp_hot * delta_temperature
    )
    shifted_config = model_config.with_overrides(
        parameter_values={
            "enthalpy_reference_temperature_K": float(
                model_config.require("enthalpy_reference_temperature_K")
            )
            + delta_temperature,
            "fuel_sensible_enthalpy_at_reference_J_kg": shifted_fuel_enthalpy,
        }
    )
    shifted = solve_combustor_loworder(
        geometry, 0.15, compressor_result.outlet, shifted_config
    )
    assert shifted.fuel_air_ratio == pytest.approx(fuel_air_ratio, rel=1.0e-12)
    assert abs(shifted.energy_residual_W) < 1.0e-8


def test_injector_capacity_matches_orifice_relation(combustor_result, model_config):
    expected = (
        float(model_config.require("injector_discharge_coefficient"))
        * float(model_config.require("injector_effective_orifice_area_m2"))
        * math.sqrt(
            2.0
            * float(model_config.require("fuel_density_kg_m3"))
            * float(model_config.require("injector_fuel_pressure_drop_Pa"))
        )
    )
    assert combustor_result.fuel_flow_capacity_per_injector_kg_s == pytest.approx(
        expected
    )


def test_out_of_domain_low_tit_is_rejected_by_geometry_schema(
    reference_design, compressor_result, model_config
):
    low_tit_design = replace(
        reference_design, turbine_inlet_total_temperature_K=300.0
    )
    with pytest.raises(GeometryStateError, match="outside frozen bounds"):
        derive_geometry_state(low_tit_design, model_config)


def test_pressure_collapse_is_hard_failure(reference_design, model_config):
    geometry = derive_geometry_state(reference_design, model_config)
    low_pressure_inlet = StationState(
        total_temperature_K=412.0, total_pressure_Pa=1000.0
    )
    with pytest.raises(CombustorModelError) as exc_info:
        solve_combustor_loworder(geometry, 0.15, low_pressure_inlet, model_config)
    assert exc_info.value.failure_code == FailureCode.COMBUSTOR_PRESSURE_INVALID


def test_constraint_residuals_follow_nonpositive_feasible_convention(
    combustor_result,
):
    assert combustor_result.constraint_residuals
    assert all(
        math.isfinite(value)
        for value in combustor_result.constraint_residuals.values()
    )
    assert all(value <= 0.0 for value in combustor_result.constraint_residuals.values())
