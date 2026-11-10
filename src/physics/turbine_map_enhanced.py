"""Map-enhanced single-stage turbine closure for the V2.2-B shadow."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Mapping

from src.physics.combustor_loworder import CombustorResult
from src.physics.engine_match import FailureCode
from src.physics.geometry_state import GeometryState
from src.physics.maps import ComponentMap, MapDomainError, MapPolicy
from src.physics.model_config import ModelConfig
from src.physics.thermo import ThermoModel, ThermoModelError
from src.physics.turbine_meanline import (
    TurbineModelError,
    TurbineResult,
    solve_turbine_meanline,
)


@dataclass(frozen=True)
class MapEnhancedTurbineResult(TurbineResult):
    """V2.1-compatible turbine result plus auditable map diagnostics."""

    map_id: str
    map_version: str
    map_source_hash: str
    map_in_domain: bool
    map_extrapolation_distance: float
    map_uncertainty: Mapping[str, float]
    minimum_capacity_margin: float
    maximum_capacity_margin: float
    corrected_speed_ratio: float
    corrected_mass_flux_ratio: float
    fuel_air_ratio: float
    thermo_model_id: str | None
    outlet_enthalpy_residual_J_kg: float | None
    isentropic_efficiency_residual_J_kg: float | None


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


def solve_turbine_map_enhanced(
    geometry: GeometryState,
    combustor: CombustorResult,
    compressor_power_W: float,
    config: ModelConfig,
    *,
    component_map: ComponentMap,
    thermo: ThermoModel | None = None,
    map_policy: MapPolicy | str = MapPolicy.DIAGNOSTIC,
) -> MapEnhancedTurbineResult:
    """Keep shaft work primary; use the map once for efficiency and capacity."""

    if component_map.component_type != "turbine":
        raise TurbineModelError(
            FailureCode.INVALID_CALIBRATION,
            "Selected map is not a turbine map",
        )
    inlet = combustor.outlet
    reference = component_map.metadata["reference_state"]
    reference_temperature = float(reference["temperature_K"])
    reference_pressure = float(reference["pressure_Pa"])
    design_flux = float(reference["corrected_mass_flux_design_kg_s_m2"])
    trial_values = (
        combustor.mass_flow_gas_kg_s,
        inlet.total_temperature_K,
        inlet.total_pressure_Pa,
        geometry.turbine_annulus_area_m2,
        geometry.design.shaft_speed_rpm,
        reference_temperature,
        reference_pressure,
        design_flux,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in trial_values):
        raise TurbineModelError(
            FailureCode.NONFINITE_STATE,
            "Map turbine trial requires finite positive inputs",
        )
    corrected_mass_flux = (
        combustor.mass_flow_gas_kg_s
        * math.sqrt(inlet.total_temperature_K / reference_temperature)
        / (
            (inlet.total_pressure_Pa / reference_pressure)
            * geometry.turbine_annulus_area_m2
        )
    )
    corrected_speed = geometry.design.shaft_speed_rpm * math.sqrt(
        reference_temperature / inlet.total_temperature_K
    )
    query = {
        "corrected_speed_ratio": corrected_speed
        / float(reference["corrected_speed_design_rpm"]),
        "corrected_mass_flux_ratio": corrected_mass_flux / design_flux,
    }
    try:
        map_result = component_map.evaluate(query, policy=map_policy)
    except MapDomainError as exc:
        raise TurbineModelError(
            FailureCode.TURBINE_EFFICIENCY_INVALID, str(exc)
        ) from exc
    efficiency = float(map_result.values["total_to_total_efficiency"])
    if not math.isfinite(efficiency) or not 0.0 < efficiency < 1.0:
        raise TurbineModelError(
            FailureCode.TURBINE_EFFICIENCY_INVALID,
            f"Turbine map efficiency must be in (0, 1), received {efficiency}",
        )
    minimum_boundary = float(
        map_result.margins["minimum_corrected_mass_flux_ratio"]
    )
    maximum_boundary = float(
        map_result.margins["maximum_corrected_mass_flux_ratio"]
    )
    stable_span = maximum_boundary - minimum_boundary
    if not math.isfinite(stable_span) or stable_span <= 0.0:
        raise TurbineModelError(
            FailureCode.INVALID_CALIBRATION,
            "Turbine map capacity boundaries cross",
        )
    flux_ratio = query["corrected_mass_flux_ratio"]
    minimum_margin = (flux_ratio - minimum_boundary) / stable_span
    maximum_margin = (maximum_boundary - flux_ratio) / stable_span
    domain_margin = _coordinate_domain_margin(component_map, query)
    capacity_residuals = {
        "turbine_map_coordinate_domain": (
            -domain_margin
            if map_result.in_domain
            else map_result.extrapolation_distance
        ),
        "turbine_map_minimum_capacity": -minimum_margin,
        "turbine_map_maximum_capacity": -maximum_margin,
    }
    base = solve_turbine_meanline(
        geometry,
        combustor,
        compressor_power_W,
        config,
        efficiency_override=efficiency,
        capacity_constraint_residuals=capacity_residuals,
        thermo=thermo,
        fuel_air_ratio=(combustor.fuel_air_ratio if thermo is not None else None),
    )
    outlet_enthalpy_residual = None
    isentropic_efficiency_residual = None
    if thermo is not None:
        try:
            inlet_enthalpy = thermo.h_gas(
                base.inlet.total_temperature_K, combustor.fuel_air_ratio
            )
            outlet_enthalpy = thermo.h_gas(
                base.outlet.total_temperature_K, combustor.fuel_air_ratio
            )
            isentropic_enthalpy = thermo.h_gas(
                base.isentropic_outlet_temperature_K,
                combustor.fuel_air_ratio,
            )
        except ThermoModelError as exc:
            raise TurbineModelError(FailureCode.NONFINITE_STATE, str(exc)) from exc
        outlet_enthalpy_residual = (
            inlet_enthalpy - outlet_enthalpy - base.specific_work_J_kg
        )
        isentropic_efficiency_residual = (
            base.specific_work_J_kg
            - efficiency * (inlet_enthalpy - isentropic_enthalpy)
        )
    base_values = {
        field.name: getattr(base, field.name) for field in fields(TurbineResult)
    }
    return MapEnhancedTurbineResult(
        **base_values,
        map_id=map_result.map_id,
        map_version=map_result.map_version,
        map_source_hash=map_result.source_hash,
        map_in_domain=map_result.in_domain,
        map_extrapolation_distance=map_result.extrapolation_distance,
        map_uncertainty=map_result.uncertainty,
        minimum_capacity_margin=minimum_margin,
        maximum_capacity_margin=maximum_margin,
        corrected_speed_ratio=query["corrected_speed_ratio"],
        corrected_mass_flux_ratio=flux_ratio,
        fuel_air_ratio=combustor.fuel_air_ratio,
        thermo_model_id=(thermo.model_id if thermo is not None else None),
        outlet_enthalpy_residual_J_kg=outlet_enthalpy_residual,
        isentropic_efficiency_residual_J_kg=isentropic_efficiency_residual,
    )
