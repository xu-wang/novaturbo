"""Variable-property, map-enhanced inlet closure for the V2.2-C1 shadow."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from src.physics.engine_match import Environment, FailureCode, StationState
from src.physics.geometry_state import GeometryState
from src.physics.maps import ComponentMap, MapDomainError, MapPolicy
from src.physics.model_config import ModelConfig
from src.physics.thermo import ThermoModel, ThermoModelError


class InletModelError(ValueError):
    """A hard inlet failure carrying the common engine failure code."""

    def __init__(self, failure_code: FailureCode, message: str):
        super().__init__(message)
        self.failure_code = failure_code


@dataclass(frozen=True)
class MapEnhancedInletState(StationState):
    """Station-2 state plus auditable inlet-map and total-state diagnostics."""

    flight_velocity_m_s: float
    maximum_air_flow_kg_s: float
    freestream_total_temperature_K: float
    freestream_total_pressure_Pa: float
    total_pressure_recovery: float
    discharge_coefficient: float
    capture_area_ratio: float
    total_enthalpy_residual_J_kg: float
    map_id: str
    map_version: str
    map_source_hash: str
    map_in_domain: bool
    map_extrapolation_distance: float
    map_uncertainty: Mapping[str, float]
    minimum_capture_margin: float
    maximum_capture_margin: float
    constraint_residuals: Mapping[str, float]
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


def solve_inlet_map_enhanced(
    geometry: GeometryState,
    environment: Environment,
    config: ModelConfig,
    *,
    component_map: ComponentMap,
    thermo: ThermoModel,
    map_policy: MapPolicy | str = MapPolicy.DIAGNOSTIC,
) -> tuple[MapEnhancedInletState, float, float]:
    """Evaluate a bounded inlet response and variable-property total state."""

    del config  # The shadow closure is fully identified by thermo/map contracts.
    values = (environment.temperature_K, environment.pressure_Pa, environment.mach)
    if not all(math.isfinite(value) for value in values):
        raise InletModelError(
            FailureCode.NONFINITE_STATE, "Inlet environment contains non-finite values"
        )
    if (
        environment.temperature_K <= 0.0
        or environment.pressure_Pa <= 0.0
        or environment.mach < 0.0
    ):
        raise InletModelError(
            FailureCode.INVALID_ENVIRONMENT,
            "Inlet environment requires positive temperature/pressure and Mach >= 0",
        )
    if component_map.component_type != "inlet":
        raise InletModelError(
            FailureCode.INVALID_CALIBRATION, "Selected map is not an inlet map"
        )

    tip_diameter = geometry.design.compressor_inducer_tip_diameter_m
    tip_disk_area = math.pi * tip_diameter**2 / 4.0
    capture_area_ratio = geometry.compressor_inlet_effective_area_m2 / tip_disk_area
    query = {
        "flight_mach": environment.mach,
        "capture_area_ratio": capture_area_ratio,
    }
    try:
        map_result = component_map.evaluate(query, policy=map_policy)
    except MapDomainError as exc:
        raise InletModelError(FailureCode.INVALID_ENVIRONMENT, str(exc)) from exc

    pressure_recovery = float(map_result.values["total_pressure_recovery"])
    discharge_coefficient = float(map_result.values["discharge_coefficient"])
    if not 0.0 < pressure_recovery <= 1.0:
        raise InletModelError(
            FailureCode.INVALID_CALIBRATION,
            "Inlet-map total-pressure recovery must be in (0, 1]",
        )
    if not 0.0 < discharge_coefficient <= 1.0:
        raise InletModelError(
            FailureCode.INVALID_CALIBRATION,
            "Inlet-map discharge coefficient must be in (0, 1]",
        )

    try:
        static_gamma = thermo.gamma_air(environment.temperature_K)
        gas_constant = thermo.gas_constant_air_J_kg_K
        flight_velocity = environment.mach * math.sqrt(
            static_gamma * gas_constant * environment.temperature_K
        )
        static_enthalpy = thermo.h_air(environment.temperature_K)
        total_enthalpy = static_enthalpy + 0.5 * flight_velocity**2
        total_temperature = thermo.air_temperature_from_h(total_enthalpy)
        isentropic_pressure_ratio = (
            1.0
            if environment.mach == 0.0
            else thermo.air_isentropic_pressure_ratio(
                environment.temperature_K, total_temperature
            )
        )
        total_gamma = thermo.gamma_air(total_temperature)
    except ThermoModelError as exc:
        raise InletModelError(FailureCode.NONFINITE_STATE, str(exc)) from exc

    freestream_total_pressure = environment.pressure_Pa * isentropic_pressure_ratio
    inlet_total_pressure = pressure_recovery * freestream_total_pressure
    choked_factor = math.sqrt(total_gamma / gas_constant) * (
        2.0 / (total_gamma + 1.0)
    ) ** ((total_gamma + 1.0) / (2.0 * (total_gamma - 1.0)))
    maximum_air_flow = (
        discharge_coefficient
        * geometry.compressor_inlet_effective_area_m2
        * inlet_total_pressure
        / math.sqrt(total_temperature)
        * choked_factor
    )
    if not math.isfinite(maximum_air_flow) or maximum_air_flow <= 0.0:
        raise InletModelError(
            FailureCode.NONFINITE_STATE, "Inlet-map choked capacity is invalid"
        )

    minimum = float(map_result.margins["minimum_capture_area_ratio"])
    maximum = float(map_result.margins["maximum_capture_area_ratio"])
    span = maximum - minimum
    if not math.isfinite(span) or span <= 0.0:
        raise InletModelError(
            FailureCode.INVALID_CALIBRATION, "Inlet map boundaries cross"
        )
    minimum_margin = (capture_area_ratio - minimum) / span
    maximum_margin = (maximum - capture_area_ratio) / span
    domain_margin = _coordinate_domain_margin(component_map, query)
    enthalpy_residual = (
        thermo.h_air(total_temperature) - static_enthalpy - 0.5 * flight_velocity**2
    )
    residuals = {
        "inlet_map_coordinate_domain": (
            -domain_margin if map_result.in_domain else map_result.extrapolation_distance
        ),
        "inlet_map_minimum_capture_area": -minimum_margin,
        "inlet_map_maximum_capture_area": -maximum_margin,
        "inlet_total_enthalpy_closure": abs(enthalpy_residual) - 1.0e-8,
    }
    state = MapEnhancedInletState(
        total_temperature_K=total_temperature,
        total_pressure_Pa=inlet_total_pressure,
        flight_velocity_m_s=flight_velocity,
        maximum_air_flow_kg_s=maximum_air_flow,
        freestream_total_temperature_K=total_temperature,
        freestream_total_pressure_Pa=freestream_total_pressure,
        total_pressure_recovery=pressure_recovery,
        discharge_coefficient=discharge_coefficient,
        capture_area_ratio=capture_area_ratio,
        total_enthalpy_residual_J_kg=enthalpy_residual,
        map_id=map_result.map_id,
        map_version=map_result.map_version,
        map_source_hash=map_result.source_hash,
        map_in_domain=map_result.in_domain,
        map_extrapolation_distance=map_result.extrapolation_distance,
        map_uncertainty=map_result.uncertainty,
        minimum_capture_margin=minimum_margin,
        maximum_capture_margin=maximum_margin,
        constraint_residuals=residuals,
        thermo_model_id=thermo.model_id,
    )
    return state, flight_velocity, maximum_air_flow
