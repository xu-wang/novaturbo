"""Variable-property convergent-nozzle closure for the V2.2-C3 shadow."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from scipy.optimize import brentq

from src.physics.engine_match import Environment, FailureCode, NozzleExitState, StationState
from src.physics.geometry_state import GeometryState
from src.physics.maps import ComponentMap, MapDomainError, MapPolicy
from src.physics.model_config import ModelConfig
from src.physics.nozzle import NozzleModelError, NozzleResult
from src.physics.thermo import ThermoModel, ThermoModelError
from src.physics.turbine_meanline import TurbineResult


@dataclass(frozen=True)
class MapEnhancedNozzleResult(NozzleResult):
    """V2.1-compatible nozzle result plus map and variable-property diagnostics."""

    map_id: str
    map_version: str
    map_source_hash: str
    map_in_domain: bool
    map_extrapolation_distance: float
    map_uncertainty: Mapping[str, float]
    inlet_pressure_ratio: float
    reynolds_ratio: float
    reynolds_number: float
    minimum_reynolds_margin: float
    maximum_reynolds_margin: float
    exit_enthalpy_residual_J_kg: float
    thermo_model_id: str


def _coordinate_domain_margin(
    component_map: ComponentMap, query: Mapping[str, float]
) -> float:
    margins = []
    for name in component_map.coordinate_names:
        axis = component_map.coordinates[name]
        span = float(axis[-1] - axis[0])
        value = float(query[name])
        margins.append(min(value - axis[0], axis[-1] - value) / span)
    return min(margins)


def solve_nozzle_map_enhanced(
    geometry: GeometryState,
    turbine: TurbineResult,
    environment: Environment,
    config: ModelConfig,
    *,
    component_map: ComponentMap,
    thermo: ThermoModel,
    map_policy: MapPolicy | str = MapPolicy.DIAGNOSTIC,
) -> MapEnhancedNozzleResult:
    """Solve a map-corrected nozzle using variable gas enthalpy and entropy."""

    turbine_outlet = turbine.outlet
    fuel_air_ratio = getattr(turbine, "fuel_air_ratio", None)
    inputs = (
        turbine_outlet.total_temperature_K,
        turbine_outlet.total_pressure_Pa,
        turbine.mass_flow_gas_kg_s,
        environment.pressure_Pa,
        geometry.nozzle_throat_area_m2,
        geometry.design.nozzle_throat_diameter_m,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in inputs):
        raise NozzleModelError(
            FailureCode.NONFINITE_STATE,
            "Map nozzle requires finite positive state, flow, and geometry",
        )
    if fuel_air_ratio is None or not math.isfinite(fuel_air_ratio) or fuel_air_ratio < 0:
        raise NozzleModelError(
            FailureCode.NOZZLE_STATE_INVALID,
            "Map nozzle requires the upstream fuel-air ratio",
        )
    if component_map.component_type != "nozzle":
        raise NozzleModelError(
            FailureCode.INVALID_CALIBRATION,
            "Selected map is not a nozzle map",
        )

    total_temperature = turbine_outlet.total_temperature_K
    inlet_pressure_ratio = turbine_outlet.total_pressure_Pa / environment.pressure_Pa
    gas_constant = thermo.gas_constant_gas_J_kg_K
    try:
        total_density = turbine_outlet.total_pressure_Pa / (
            gas_constant * total_temperature
        )
        characteristic_velocity = turbine.mass_flow_gas_kg_s / (
            total_density * geometry.nozzle_throat_area_m2
        )
        viscosity = thermo.mu_gas(total_temperature, fuel_air_ratio)
    except ThermoModelError as exc:
        raise NozzleModelError(FailureCode.NONFINITE_STATE, str(exc)) from exc
    reynolds = (
        total_density
        * characteristic_velocity
        * geometry.design.nozzle_throat_diameter_m
        / viscosity
    )
    reference_reynolds = float(
        component_map.metadata["reference_state"]["reynolds_design"]
    )
    query = {
        "inlet_pressure_ratio": inlet_pressure_ratio,
        "reynolds_ratio": reynolds / reference_reynolds,
    }
    try:
        map_result = component_map.evaluate(query, policy=map_policy)
    except MapDomainError as exc:
        raise NozzleModelError(FailureCode.NOZZLE_STATE_INVALID, str(exc)) from exc
    pressure_recovery = float(map_result.values["total_pressure_recovery"])
    discharge_coefficient = float(map_result.values["discharge_coefficient"])
    if not 0.0 < pressure_recovery <= 1.0:
        raise NozzleModelError(
            FailureCode.NOZZLE_STATE_INVALID,
            "Nozzle-map total-pressure recovery must be in (0, 1]",
        )
    if not 0.0 < discharge_coefficient <= 1.0:
        raise NozzleModelError(
            FailureCode.NOZZLE_STATE_INVALID,
            "Nozzle-map discharge coefficient must be in (0, 1]",
        )
    total_pressure = pressure_recovery * turbine_outlet.total_pressure_Pa
    if total_pressure <= environment.pressure_Pa:
        raise NozzleModelError(
            FailureCode.NOZZLE_STATE_INVALID,
            "Recovered nozzle total pressure must exceed ambient pressure",
        )

    try:
        total_enthalpy = thermo.h_gas(total_temperature, fuel_air_ratio)
        lower_temperature = max(
            float(thermo.valid_temperature_range_K[0]), 0.2 * total_temperature
        )

        def sonic_residual(temperature: float) -> float:
            gamma = thermo.gamma_gas(temperature, fuel_air_ratio)
            return (
                total_enthalpy
                - thermo.h_gas(temperature, fuel_air_ratio)
                - 0.5 * gamma * gas_constant * temperature
            )

        critical_temperature = brentq(
            sonic_residual,
            lower_temperature,
            total_temperature * (1.0 - 1.0e-12),
            xtol=1.0e-10,
            rtol=1.0e-12,
            maxiter=64,
        )
        critical_pressure = total_pressure * thermo.gas_isentropic_pressure_ratio(
            total_temperature, critical_temperature, fuel_air_ratio
        )
        choked = environment.pressure_Pa <= critical_pressure
        if choked:
            exit_temperature = critical_temperature
            exit_pressure = critical_pressure
        else:
            target_pressure_ratio = environment.pressure_Pa / total_pressure

            def pressure_residual(temperature: float) -> float:
                return thermo.gas_isentropic_pressure_ratio(
                    total_temperature, temperature, fuel_air_ratio
                ) - target_pressure_ratio

            exit_temperature = brentq(
                pressure_residual,
                lower_temperature,
                total_temperature * (1.0 - 1.0e-12),
                xtol=1.0e-10,
                rtol=1.0e-12,
                maxiter=64,
            )
            exit_pressure = environment.pressure_Pa
        exit_enthalpy = thermo.h_gas(exit_temperature, fuel_air_ratio)
        enthalpy_drop = total_enthalpy - exit_enthalpy
        exit_velocity = math.sqrt(2.0 * enthalpy_drop)
        exit_gamma = thermo.gamma_gas(exit_temperature, fuel_air_ratio)
    except (ThermoModelError, RuntimeError, ValueError) as exc:
        raise NozzleModelError(
            FailureCode.NOZZLE_STATE_INVALID,
            "Variable-property nozzle state solve failed",
        ) from exc

    local_sound_speed = math.sqrt(exit_gamma * gas_constant * exit_temperature)
    exit_mach = exit_velocity / local_sound_speed
    exit_density = exit_pressure / (gas_constant * exit_temperature)
    mass_flow_capacity = (
        discharge_coefficient
        * exit_density
        * exit_velocity
        * geometry.nozzle_throat_area_m2
    )
    mass_flow_residual = turbine.mass_flow_gas_kg_s - mass_flow_capacity
    residual_epsilon = float(
        config.solver_candidate["mass_flow_residual_epsilon_kg_s"]
    )
    normalized_flow_residual = mass_flow_residual / max(
        turbine.mass_flow_gas_kg_s, residual_epsilon
    )
    enthalpy_residual = total_enthalpy - exit_enthalpy - 0.5 * exit_velocity**2
    velocity_consistency_residual = exit_velocity - math.sqrt(2.0 * enthalpy_drop)
    pressure_thrust = (
        (exit_pressure - environment.pressure_Pa) * geometry.nozzle_exit_area_m2
    )

    minimum = float(map_result.margins["minimum_reynolds_ratio"])
    maximum = float(map_result.margins["maximum_reynolds_ratio"])
    span = maximum - minimum
    if not math.isfinite(span) or span <= 0.0:
        raise NozzleModelError(
            FailureCode.INVALID_CALIBRATION,
            "Nozzle map Reynolds boundaries cross",
        )
    reynolds_ratio = query["reynolds_ratio"]
    minimum_margin = (reynolds_ratio - minimum) / span
    maximum_margin = (maximum - reynolds_ratio) / span
    domain_margin = _coordinate_domain_margin(component_map, query)
    state_residuals = {
        "nozzle_total_pressure_usable": environment.pressure_Pa - total_pressure,
        "nozzle_map_coordinate_domain": (
            -domain_margin if map_result.in_domain else map_result.extrapolation_distance
        ),
        "nozzle_map_minimum_reynolds": -minimum_margin,
        "nozzle_map_maximum_reynolds": -maximum_margin,
        "nozzle_exit_enthalpy_closure": abs(enthalpy_residual) - 1.0e-8,
    }
    calculated = (
        critical_pressure,
        exit_temperature,
        exit_pressure,
        exit_mach,
        exit_velocity,
        mass_flow_capacity,
        normalized_flow_residual,
        pressure_thrust,
        reynolds,
    )
    if not all(math.isfinite(value) for value in calculated):
        raise NozzleModelError(
            FailureCode.NONFINITE_STATE,
            "Map nozzle calculation produced a non-finite state",
        )
    if not 0.0 < exit_mach <= 1.0 + 1.0e-8 or mass_flow_capacity <= 0.0:
        raise NozzleModelError(
            FailureCode.NOZZLE_STATE_INVALID,
            "Map nozzle exit Mach or capacity is outside its physical domain",
        )
    return MapEnhancedNozzleResult(
        turbine_outlet=turbine_outlet,
        nozzle_total=StationState(total_temperature, total_pressure),
        exit=NozzleExitState(
            static_temperature_K=exit_temperature,
            static_pressure_Pa=exit_pressure,
            mach=exit_mach,
            velocity_m_s=exit_velocity,
        ),
        choked=choked,
        critical_static_pressure_Pa=critical_pressure,
        total_pressure_recovery=pressure_recovery,
        discharge_coefficient=discharge_coefficient,
        mass_flow_gas_kg_s=turbine.mass_flow_gas_kg_s,
        mass_flow_capacity_kg_s=mass_flow_capacity,
        mass_flow_residual_kg_s=mass_flow_residual,
        normalized_flow_residual=normalized_flow_residual,
        velocity_consistency_residual_m_s=velocity_consistency_residual,
        pressure_thrust_N=pressure_thrust,
        state_constraint_residuals=state_residuals,
        map_id=map_result.map_id,
        map_version=map_result.map_version,
        map_source_hash=map_result.source_hash,
        map_in_domain=map_result.in_domain,
        map_extrapolation_distance=map_result.extrapolation_distance,
        map_uncertainty=map_result.uncertainty,
        inlet_pressure_ratio=inlet_pressure_ratio,
        reynolds_ratio=reynolds_ratio,
        reynolds_number=reynolds,
        minimum_reynolds_margin=minimum_margin,
        maximum_reynolds_margin=maximum_margin,
        exit_enthalpy_residual_J_kg=enthalpy_residual,
        thermo_model_id=thermo.model_id,
    )
