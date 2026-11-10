"""Integration tests for the WP19-7/8 Geometry19 design-point solver."""

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
    NozzleExitState,
    StationState,
    solve_design_point,
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
def reference_result(reference_design, model_config):
    return solve_design_point(reference_design, Environment(), model_config)


def test_reference_design_converges_to_one_whole_engine_flow_root(reference_result):
    result = reference_result
    assert result.solver.converged is True
    assert result.solver.failure_code is None
    assert result.solver.evaluations > 0
    assert result.solver.iterations > 0
    assert result.operating_point["mass_flow_search_min_kg_s"] < result.operating_point[
        "mass_flow_air_kg_s"
    ]
    assert result.operating_point["mass_flow_air_kg_s"] < result.operating_point[
        "mass_flow_search_max_kg_s"
    ]
    assert result.operating_point["mass_flow_search_max_kg_s"] < result.operating_point[
        "inlet_choked_capacity_kg_s"
    ]


def test_reference_design_closes_all_wp19_7_conservation_residuals(
    reference_result, model_config
):
    residuals = reference_result.conservation
    flow_tolerance = float(
        model_config.solver_candidate["flow_residual_tolerance"]
    )
    flow_epsilon = float(
        model_config.solver_candidate["mass_flow_residual_epsilon_kg_s"]
    )
    raw_flow_tolerance = flow_tolerance * max(
        reference_result.operating_point["mass_flow_gas_kg_s"], flow_epsilon
    )
    assert abs(residuals.combustor_energy_W) < 1.0e-8
    assert abs(residuals.shaft_power_W) < 1.0e-8
    assert abs(residuals.normalized_flow) <= flow_tolerance
    assert abs(residuals.mass_flow_kg_s) <= raw_flow_tolerance
    assert residuals.nozzle_velocity_m_s == pytest.approx(0.0)
    assert reference_result.solver.flow_residual == pytest.approx(
        residuals.normalized_flow
    )


def test_engine_result_contains_all_core_stations_and_components(reference_result):
    result = reference_result
    assert set(result.stations) == {
        "t2",
        "t3",
        "t4",
        "t5",
        "nozzle_total",
        "exit",
    }
    assert all(
        isinstance(result.stations[name], StationState)
        for name in ("t2", "t3", "t4", "t5", "nozzle_total")
    )
    assert isinstance(result.stations["exit"], NozzleExitState)
    assert set(result.components) == {
        "compressor",
        "combustor",
        "turbine",
        "nozzle",
    }


def test_matched_performance_uses_complete_momentum_and_pressure_thrust(
    reference_result,
):
    result = reference_result
    mdot_air = result.operating_point["mass_flow_air_kg_s"]
    mdot_gas = result.operating_point["mass_flow_gas_kg_s"]
    exit_state = result.stations["exit"]
    assert isinstance(exit_state, NozzleExitState)
    gamma_cold = 1.4
    gas_constant_cold = 287.05
    flight_velocity = result.environment.mach * math.sqrt(
        gamma_cold * gas_constant_cold * result.environment.temperature_K
    )
    nozzle_area = math.pi * result.design.nozzle_throat_diameter_m**2 / 4.0
    expected = (
        mdot_gas * exit_state.velocity_m_s
        - mdot_air * flight_velocity
        + (exit_state.static_pressure_Pa - result.environment.pressure_Pa)
        * nozzle_area
    )
    assert result.performance["thrust_N"] == pytest.approx(expected)
    assert result.performance["thrust_N"] > 0.0
    assert result.performance["tsfc_kg_N_s"] == pytest.approx(
        result.operating_point["mass_flow_fuel_kg_s"] / expected
    )


def test_wp19_8_reports_complete_formal_feasibility(reference_result):
    result = reference_result
    assert result.feasibility_scope == "geometry19_wp19_8_bootstrap_complete"
    assert result.feasibility_complete is True
    assert result.feasible is False
    assert result.performance["mass_kg"] > 0.0
    assert result.mass_breakdown_kg
    assert result.mass_volume_breakdown_m3
    assert result.formal_diagnostics
    assert result.core_feasible is False
    assert result.constraints["combustor_pressure_loss_fraction_max"] > 0.0
    assert set(result.thermodynamic_constraints) < set(result.constraints)
    assert result.assembly_diagnostics
    assert result.structural_diagnostics
    assert result.assembly_diagnostics_pending is False
    assert result.structural_diagnostics_pending is False
    grouped_constraints = {
        **result.thermodynamic_constraints,
        **result.assembly_constraints,
        **result.structural_constraints,
        **result.thermal_constraints,
        **result.manufacturing_constraints,
    }
    assert grouped_constraints == result.constraints
    assert set(result.assembly_diagnostics) <= set(result.constraints)
    assert set(result.structural_diagnostics) <= set(result.constraints)
    assert result.calibration_status == "bootstrap_candidate"
    assert result.formal_dataset_generation_ready is False


def test_result_provenance_uses_loaded_spec_and_calibration_hash(
    reference_result, model_config
):
    assert reference_result.physics_model_version == "geometry19_closed_v2_1"
    assert reference_result.calibration_pack_hash == (
        model_config.calibration_pack_hash
    )
    assert len(reference_result.calibration_pack_hash) == 64


def test_mapping_and_sequence_inputs_reproduce_reference_result(
    reference_design, reference_result, model_config
):
    mapping_result = solve_design_point(
        reference_design.as_mapping(), Environment(), model_config
    )
    sequence_result = solve_design_point(
        [
            reference_design.as_mapping()[name]
            for name in Geometry19.coordinate_names()
        ],
        Environment(),
        model_config,
    )
    assert mapping_result.operating_point["mass_flow_air_kg_s"] == pytest.approx(
        reference_result.operating_point["mass_flow_air_kg_s"]
    )
    assert sequence_result.performance["thrust_N"] == pytest.approx(
        reference_result.performance["thrust_N"]
    )


def test_invalid_geometry_returns_typed_hard_failure(reference_design, model_config):
    invalid = replace(
        reference_design,
        combustor_outer_diameter_m=0.05,
        combustor_inner_diameter_m=0.06,
    )
    result = solve_design_point(invalid, Environment(), model_config)
    assert result.solver.converged is False
    assert result.solver.failure_code == FailureCode.GEOMETRY_INVALID
    assert result.core_feasible is False
    assert result.feasible is False
    assert result.feasibility_complete is False
    assert result.feasibility_scope == "geometry19_wp19_8_not_evaluated"


def test_invalid_environment_returns_typed_environment_failure(
    reference_design, model_config
):
    result = solve_design_point(
        reference_design,
        Environment(temperature_K=-1.0),
        model_config,
    )
    assert result.solver.converged is False
    assert result.solver.failure_code == FailureCode.INVALID_ENVIRONMENT


def test_nonfinite_environment_returns_nonfinite_failure(
    reference_design, model_config
):
    result = solve_design_point(
        reference_design,
        Environment(temperature_K=math.nan),
        model_config,
    )
    assert result.solver.converged is False
    assert result.solver.failure_code == FailureCode.NONFINITE_STATE


def test_malformed_geometry_returns_typed_result(model_config):
    result = solve_design_point([1.0] * 18, Environment(), model_config)
    assert result.design is None
    assert result.solver.converged is False
    assert result.solver.failure_code == FailureCode.INVALID_GEOMETRY_INPUT


def test_environment_overflow_returns_typed_nonfinite_failure(
    reference_design, model_config
):
    result = solve_design_point(
        reference_design, Environment(mach=1.0e308), model_config
    )
    assert result.solver.converged is False
    assert result.solver.failure_code == FailureCode.NONFINITE_STATE


def test_whole_engine_choked_result_includes_pressure_thrust(
    reference_design, model_config
):
    result = solve_design_point(
        reference_design, Environment(mach=1.5), model_config
    )
    assert result.solver.converged is True
    assert result.components["nozzle"]["choked"] is True
    exit_state = result.stations["exit"]
    assert isinstance(exit_state, NozzleExitState)
    area = math.pi * reference_design.nozzle_throat_diameter_m**2 / 4.0
    pressure_thrust = (
        exit_state.static_pressure_Pa - result.environment.pressure_Pa
    ) * area
    momentum_thrust = (
        result.operating_point["mass_flow_gas_kg_s"] * exit_state.velocity_m_s
        - result.operating_point["mass_flow_air_kg_s"]
        * result.environment.mach
        * math.sqrt(
            1.4 * 287.05 * result.environment.temperature_K
        )
    )
    assert pressure_thrust > 0.0
    assert result.performance["thrust_N"] == pytest.approx(
        momentum_thrust + pressure_thrust
    )


def test_bracket_without_root_returns_no_flow_match(reference_design, model_config):
    solver_candidate = dict(model_config.solver_candidate)
    solver_candidate["mass_flow_lower_fraction"] = 0.001
    solver_candidate["mass_flow_upper_fraction"] = 0.5
    narrow_config = replace(model_config, solver_candidate=solver_candidate)
    result = solve_design_point(reference_design, Environment(), narrow_config)
    assert result.solver.converged is False
    assert result.solver.failure_code == FailureCode.NO_FLOW_MATCH
