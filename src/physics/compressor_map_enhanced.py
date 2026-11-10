"""Map-enhanced centrifugal-compressor closure for the V2.2 shadow backend."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from src.physics.compressor_meanline import CompressorModelError, CompressorResult
from src.physics.engine_match import FailureCode, StationState
from src.physics.geometry_state import GeometryState
from src.physics.maps import ComponentMap, MapDomainError, MapPolicy
from src.physics.model_config import ModelConfig
from src.physics.thermo import ThermoModel, ThermoModelError


@dataclass(frozen=True)
class MapEnhancedCompressorResult(CompressorResult):
    """V2.1-compatible compressor result plus auditable map diagnostics."""

    map_id: str
    map_version: str
    map_source_hash: str
    map_in_domain: bool
    map_extrapolation_distance: float
    map_uncertainty: Mapping[str, float]
    surge_margin: float
    choke_margin: float
    corrected_speed_ratio: float
    flow_coefficient_ratio: float
    work_coefficient: float
    euler_work_factor: float
    euler_specific_work_J_kg: float
    thermo_model_id: str


def _coordinate_domain_margin(component_map: ComponentMap, query: Mapping[str, float]) -> float:
    margins = []
    for name in component_map.coordinate_names:
        axis = component_map.coordinates[name]
        span = float(axis[-1] - axis[0])
        value = float(query[name])
        margins.append(min(value - axis[0], axis[-1] - value) / span)
    return min(margins)


def solve_compressor_map_enhanced(
    geometry: GeometryState,
    mass_flow_air_kg_s: float,
    inlet: StationState,
    config: ModelConfig,
    *,
    component_map: ComponentMap,
    thermo: ThermoModel,
    map_policy: MapPolicy | str = MapPolicy.DIAGNOSTIC,
) -> MapEnhancedCompressorResult:
    """Retain the Euler/slip skeleton and use the map exactly once for closure."""

    trial_values = (
        mass_flow_air_kg_s,
        inlet.total_temperature_K,
        inlet.total_pressure_Pa,
        geometry.compressor_exit_effective_area_m2,
        geometry.compressor_tip_speed_m_s,
    )
    if not all(math.isfinite(value) for value in trial_values):
        raise CompressorModelError(
            FailureCode.NONFINITE_STATE, "Map compressor trial contains non-finite inputs"
        )
    if mass_flow_air_kg_s <= 0.0:
        raise CompressorModelError(
            FailureCode.COMPRESSOR_NONPOSITIVE_WORK,
            "Map compressor air mass flow must be positive",
        )
    if inlet.total_temperature_K <= 0.0 or inlet.total_pressure_Pa <= 0.0:
        raise CompressorModelError(
            FailureCode.NONFINITE_STATE,
            "Map compressor inlet total state must be positive",
        )
    if component_map.component_type != "compressor":
        raise CompressorModelError(
            FailureCode.INVALID_CALIBRATION, "Selected map is not a compressor map"
        )

    gas_constant = thermo.gas_constant_air_J_kg_K
    rho_ref = inlet.total_pressure_Pa / (gas_constant * inlet.total_temperature_K)
    meridional_velocity = mass_flow_air_kg_s / (
        rho_ref * geometry.compressor_exit_effective_area_m2
    )
    tip_speed = geometry.compressor_tip_speed_m_s
    phi_ref = meridional_velocity / tip_speed
    blade_count = int(geometry.design.compressor_blade_count)
    slip_factor = 1.0 - (
        float(config.require("stanitz_slip_coefficient")) * math.pi / blade_count
    )
    if not 0.0 < slip_factor < 1.0:
        raise CompressorModelError(
            FailureCode.COMPRESSOR_SLIP_INVALID,
            f"Compressor slip factor must be in (0, 1), received {slip_factor}",
        )
    beta_rad = math.radians(geometry.design.compressor_exit_blade_angle_deg)
    tangential_velocity = slip_factor * (
        tip_speed - meridional_velocity / math.tan(beta_rad)
    )
    euler_work = tip_speed * tangential_velocity
    if not math.isfinite(euler_work) or euler_work <= 0.0:
        raise CompressorModelError(
            FailureCode.COMPRESSOR_NONPOSITIVE_WORK,
            f"Compressor Euler work must be positive, received {euler_work}",
        )

    reference = component_map.metadata["reference_state"]
    phi_design = float(reference["phi_design"])
    corrected_speed = geometry.design.shaft_speed_rpm * math.sqrt(
        float(reference["temperature_K"]) / inlet.total_temperature_K
    )
    query = {
        "corrected_speed_ratio": corrected_speed
        / float(reference["corrected_speed_design_rpm"]),
        "flow_coefficient_ratio": phi_ref / phi_design,
    }
    try:
        map_result = component_map.evaluate(query, policy=map_policy)
    except MapDomainError as exc:
        raise CompressorModelError(
            FailureCode.COMPRESSOR_EFFICIENCY_INVALID, str(exc)
        ) from exc
    if "work_coefficient" in map_result.values:
        work_coefficient = float(map_result.values["work_coefficient"])
        specific_work = work_coefficient * tip_speed**2
        work_factor = specific_work / euler_work
    else:
        work_factor = float(map_result.values["euler_work_factor"])
        specific_work = work_factor * euler_work
        work_coefficient = specific_work / tip_speed**2
    efficiency = float(map_result.values["total_to_total_efficiency"])
    if (
        not math.isfinite(work_factor)
        or work_factor <= 0.0
        or not math.isfinite(specific_work)
        or specific_work <= 0.0
    ):
        raise CompressorModelError(
            FailureCode.COMPRESSOR_NONPOSITIVE_WORK,
            "Compressor map produced nonpositive work",
        )
    if not math.isfinite(efficiency) or not 0.0 < efficiency < 1.0:
        raise CompressorModelError(
            FailureCode.COMPRESSOR_EFFICIENCY_INVALID,
            f"Compressor map efficiency must be in (0, 1), received {efficiency}",
        )
    surge_boundary = float(map_result.margins["surge_flow_coefficient_ratio"])
    choke_boundary = float(map_result.margins["choke_flow_coefficient_ratio"])
    stable_span = choke_boundary - surge_boundary
    if stable_span <= 0.0:
        raise CompressorModelError(
            FailureCode.INVALID_CALIBRATION, "Compressor map boundaries cross"
        )
    surge_margin = (query["flow_coefficient_ratio"] - surge_boundary) / stable_span
    choke_margin = (choke_boundary - query["flow_coefficient_ratio"]) / stable_span

    try:
        inlet_enthalpy = thermo.h_air(inlet.total_temperature_K)
        outlet_temperature = thermo.air_temperature_from_h(
            inlet_enthalpy + specific_work
        )
        isentropic_temperature = thermo.air_temperature_from_h(
            inlet_enthalpy + efficiency * specific_work
        )
        pressure_ratio = thermo.air_isentropic_pressure_ratio(
            inlet.total_temperature_K, isentropic_temperature
        )
        viscosity = thermo.mu_air(inlet.total_temperature_K)
        gamma = thermo.gamma_air(inlet.total_temperature_K)
    except ThermoModelError as exc:
        raise CompressorModelError(FailureCode.NONFINITE_STATE, str(exc)) from exc
    outlet_pressure = pressure_ratio * inlet.total_pressure_Pa
    shaft_power = mass_flow_air_kg_s * specific_work
    reynolds = (
        rho_ref
        * tip_speed
        * geometry.design.compressor_impeller_exit_diameter_m
        / viscosity
    )
    tip_mach = tip_speed / math.sqrt(
        gamma * gas_constant * inlet.total_temperature_K
    )
    loading_coefficient = specific_work / tip_speed**2
    domain_margin = _coordinate_domain_margin(component_map, query)
    residuals = {
        "compressor_map_coordinate_domain": (
            -domain_margin if map_result.in_domain else map_result.extrapolation_distance
        ),
        "compressor_map_surge_boundary": -surge_margin,
        "compressor_map_choke_boundary": -choke_margin,
        "compressor_tip_mach_max": tip_mach
        - float(config.require("compressor_tip_mach_max")),
        "compressor_efficiency_min": float(config.require("compressor_efficiency_min"))
        - efficiency,
        "compressor_tip_speed_limit": tip_speed
        - float(config.require("compressor_tip_speed_limit_m_s")),
    }
    outlet_values = (
        outlet_temperature,
        pressure_ratio,
        outlet_pressure,
        shaft_power,
        reynolds,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in outlet_values):
        raise CompressorModelError(
            FailureCode.NONFINITE_STATE,
            "Map compressor outlet calculation produced an invalid state",
        )
    return MapEnhancedCompressorResult(
        inlet=inlet,
        outlet=StationState(outlet_temperature, outlet_pressure),
        mass_flow_air_kg_s=mass_flow_air_kg_s,
        reference_density_kg_m3=rho_ref,
        reference_meridional_velocity_m_s=meridional_velocity,
        flow_coefficient_ref=phi_ref,
        slip_factor=slip_factor,
        exit_tangential_velocity_m_s=tangential_velocity,
        specific_work_J_kg=specific_work,
        loading_coefficient=loading_coefficient,
        tip_mach=tip_mach,
        reynolds_number=reynolds,
        dynamic_viscosity_Pa_s=viscosity,
        total_to_total_efficiency=efficiency,
        pressure_ratio=pressure_ratio,
        shaft_power_W=shaft_power,
        calibration_domain_residuals=residuals,
        calibration_domain_feasible=all(value <= 0.0 for value in residuals.values()),
        map_id=map_result.map_id,
        map_version=map_result.map_version,
        map_source_hash=map_result.source_hash,
        map_in_domain=map_result.in_domain,
        map_extrapolation_distance=map_result.extrapolation_distance,
        map_uncertainty=map_result.uncertainty,
        surge_margin=surge_margin,
        choke_margin=choke_margin,
        corrected_speed_ratio=query["corrected_speed_ratio"],
        flow_coefficient_ratio=query["flow_coefficient_ratio"],
        work_coefficient=work_coefficient,
        euler_work_factor=work_factor,
        euler_specific_work_J_kg=euler_work,
        thermo_model_id=thermo.model_id,
    )
