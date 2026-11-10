"""Variable-property, map-enhanced combustor closure for V2.2-C2 shadow."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from scipy.optimize import brentq

from src.physics.combustor_loworder import CombustorModelError, CombustorResult
from src.physics.engine_match import FailureCode, StationState
from src.physics.geometry_state import GeometryState
from src.physics.maps import ComponentMap, MapDomainError, MapPolicy
from src.physics.model_config import ModelConfig
from src.physics.thermo import ThermoModel, ThermoModelError


@dataclass(frozen=True)
class MapEnhancedCombustorResult(CombustorResult):
    """V2.1-compatible combustor result plus map and local-root diagnostics."""

    combustion_efficiency: float
    mass_flux_ratio: float
    map_id: str
    map_version: str
    map_source_hash: str
    map_in_domain: bool
    map_extrapolation_distance: float
    map_uncertainty: Mapping[str, float]
    lean_stability_margin: float
    rich_stability_margin: float
    fuel_solution_iterations: int
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


def solve_combustor_map_enhanced(
    geometry: GeometryState,
    mass_flow_air_kg_s: float,
    inlet: StationState,
    config: ModelConfig,
    *,
    component_map: ComponentMap,
    thermo: ThermoModel,
    map_policy: MapPolicy | str = MapPolicy.DIAGNOSTIC,
) -> MapEnhancedCombustorResult:
    """Solve a local FAR root while preserving the outer one-flow-root engine."""

    turbine_inlet_temperature = geometry.design.turbine_inlet_total_temperature_K
    trial_values = (
        mass_flow_air_kg_s,
        inlet.total_temperature_K,
        inlet.total_pressure_Pa,
        turbine_inlet_temperature,
        geometry.combustor_annulus_area_m2,
        geometry.combustor_volume_m3,
        geometry.combustor_hydraulic_diameter_m,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in trial_values):
        raise CombustorModelError(
            FailureCode.NONFINITE_STATE,
            "Map combustor requires finite positive flow, state, and geometry",
        )
    if turbine_inlet_temperature <= inlet.total_temperature_K:
        raise CombustorModelError(
            FailureCode.COMBUSTOR_ENERGY_INVALID,
            "Turbine-inlet temperature must exceed combustor-inlet temperature",
        )
    if component_map.component_type != "combustor":
        raise CombustorModelError(
            FailureCode.INVALID_CALIBRATION,
            "Selected map is not a combustor map",
        )

    reference = component_map.metadata["reference_state"]
    design_mass_flux = float(reference["mass_flux_design_kg_s_m2"])
    mass_flux = mass_flow_air_kg_s / geometry.combustor_annulus_area_m2
    mass_flux_ratio = mass_flux / design_mass_flux
    stoichiometric_far = float(config.require("stoichiometric_fuel_air_ratio"))
    lower_heating_value = float(config.require("fuel_lower_heating_value_J_kg"))
    fuel_sensible_enthalpy = float(
        config.require("fuel_sensible_enthalpy_at_reference_J_kg")
    )
    if not (
        math.isfinite(design_mass_flux)
        and design_mass_flux > 0.0
        and math.isfinite(stoichiometric_far)
        and stoichiometric_far > 0.0
        and math.isfinite(lower_heating_value)
        and lower_heating_value > 0.0
    ):
        raise CombustorModelError(
            FailureCode.INVALID_CALIBRATION,
            "Combustor reference mass flux, stoichiometry, and LHV must be positive",
        )
    try:
        air_enthalpy = thermo.h_air(inlet.total_temperature_K)
    except ThermoModelError as exc:
        raise CombustorModelError(FailureCode.NONFINITE_STATE, str(exc)) from exc

    equivalence_axis = component_map.coordinates["equivalence_ratio"]
    lower_far = float(equivalence_axis[0]) * stoichiometric_far
    upper_far = float(equivalence_axis[-1]) * stoichiometric_far

    def evaluate_energy(fuel_air_ratio: float) -> float:
        equivalence_ratio = fuel_air_ratio / stoichiometric_far
        try:
            evaluated = component_map.evaluate(
                {
                    "mass_flux_ratio": mass_flux_ratio,
                    "equivalence_ratio": equivalence_ratio,
                },
                policy=map_policy,
            )
            gas_enthalpy = thermo.h_gas(
                turbine_inlet_temperature, fuel_air_ratio
            )
        except (MapDomainError, ThermoModelError) as exc:
            raise CombustorModelError(
                FailureCode.COMBUSTOR_ENERGY_INVALID, str(exc)
            ) from exc
        efficiency = float(evaluated.values["combustion_efficiency"])
        if not math.isfinite(efficiency) or not 0.0 < efficiency <= 1.0:
            raise CombustorModelError(
                FailureCode.COMBUSTOR_ENERGY_INVALID,
                "Combustor-map efficiency must be in (0, 1]",
            )
        fuel_energy = efficiency * lower_heating_value + fuel_sensible_enthalpy
        return (
            air_enthalpy
            + fuel_air_ratio * fuel_energy
            - (1.0 + fuel_air_ratio) * gas_enthalpy
        )

    lower_residual = evaluate_energy(lower_far)
    upper_residual = evaluate_energy(upper_far)
    if not (
        math.isfinite(lower_residual)
        and math.isfinite(upper_residual)
        and lower_residual * upper_residual < 0.0
    ):
        raise CombustorModelError(
            FailureCode.COMBUSTOR_ENERGY_INVALID,
            "Combustor energy equation has no unique bracketed FAR solution",
        )
    try:
        fuel_air_ratio, root_info = brentq(
            evaluate_energy,
            lower_far,
            upper_far,
            xtol=1.0e-12,
            rtol=1.0e-12,
            maxiter=64,
            full_output=True,
            disp=False,
        )
    except (RuntimeError, ValueError) as exc:
        raise CombustorModelError(
            FailureCode.COMBUSTOR_ENERGY_INVALID,
            "Combustor FAR root solve failed",
        ) from exc
    equivalence_ratio = fuel_air_ratio / stoichiometric_far
    query = {
        "mass_flux_ratio": mass_flux_ratio,
        "equivalence_ratio": equivalence_ratio,
    }
    try:
        map_result = component_map.evaluate(query, policy=map_policy)
        gas_enthalpy = thermo.h_gas(turbine_inlet_temperature, fuel_air_ratio)
    except (MapDomainError, ThermoModelError) as exc:
        raise CombustorModelError(
            FailureCode.COMBUSTOR_ENERGY_INVALID, str(exc)
        ) from exc
    combustion_efficiency = float(map_result.values["combustion_efficiency"])
    pressure_loss_fraction = float(
        map_result.values["total_pressure_loss_fraction"]
    )
    if not 0.0 <= pressure_loss_fraction < 1.0:
        raise CombustorModelError(
            FailureCode.COMBUSTOR_PRESSURE_INVALID,
            "Combustor-map pressure-loss fraction must be in [0, 1)",
        )
    pressure_loss = pressure_loss_fraction * inlet.total_pressure_Pa
    outlet_pressure = inlet.total_pressure_Pa - pressure_loss

    reference_density = inlet.total_pressure_Pa / (
        thermo.gas_constant_air_J_kg_K * inlet.total_temperature_K
    )
    reference_velocity = mass_flow_air_kg_s / (
        reference_density * geometry.combustor_annulus_area_m2
    )
    dynamic_pressure = 0.5 * reference_density * reference_velocity**2
    pressure_loss_coefficient = pressure_loss / dynamic_pressure
    mass_flow_fuel = fuel_air_ratio * mass_flow_air_kg_s
    mass_flow_gas = mass_flow_air_kg_s + mass_flow_fuel
    air_fuel_ratio = 1.0 / fuel_air_ratio

    primary_volume = float(
        config.require("combustor_primary_volume_fraction")
    ) * geometry.combustor_volume_m3
    primary_air_flow = float(
        config.require("combustor_primary_air_fraction")
    ) * mass_flow_air_kg_s
    primary_gas_flow = primary_air_flow + mass_flow_fuel
    primary_mean_temperature = 0.5 * (
        inlet.total_temperature_K + turbine_inlet_temperature
    )
    primary_mean_pressure = 0.5 * (inlet.total_pressure_Pa + outlet_pressure)
    primary_mean_density = primary_mean_pressure / (
        thermo.gas_constant_gas_J_kg_K * primary_mean_temperature
    )
    primary_residence_time = primary_mean_density * primary_volume / primary_gas_flow

    injector_count = int(geometry.design.combustor_injector_count)
    fuel_flow_per_injector = mass_flow_fuel / injector_count
    injector_capacity = (
        float(config.require("injector_discharge_coefficient"))
        * float(config.require("injector_effective_orifice_area_m2"))
        * math.sqrt(
            2.0
            * float(config.require("fuel_density_kg_m3"))
            * float(config.require("injector_fuel_pressure_drop_Pa"))
        )
    )
    fuel_energy = (
        combustion_efficiency * lower_heating_value + fuel_sensible_enthalpy
    )
    energy_residual = mass_flow_air_kg_s * (
        air_enthalpy
        + fuel_air_ratio * fuel_energy
        - (1.0 + fuel_air_ratio) * gas_enthalpy
    )

    lean_boundary = float(map_result.margins["minimum_equivalence_ratio"])
    rich_boundary = float(map_result.margins["maximum_equivalence_ratio"])
    stable_span = rich_boundary - lean_boundary
    if not math.isfinite(stable_span) or stable_span <= 0.0:
        raise CombustorModelError(
            FailureCode.INVALID_CALIBRATION,
            "Combustor map stability boundaries cross",
        )
    lean_margin = (equivalence_ratio - lean_boundary) / stable_span
    rich_margin = (rich_boundary - equivalence_ratio) / stable_span
    domain_margin = _coordinate_domain_margin(component_map, query)
    residuals = {
        "combustor_map_coordinate_domain": (
            -domain_margin if map_result.in_domain else map_result.extrapolation_distance
        ),
        "combustor_map_lean_stability": -lean_margin,
        "combustor_map_rich_stability": -rich_margin,
        "combustor_pressure_loss_fraction_max": pressure_loss_fraction
        - float(config.require("combustor_total_pressure_loss_fraction_max")),
        "combustor_primary_residence_time_min": float(
            config.require("combustor_primary_residence_time_min_s")
        )
        - primary_residence_time,
        "combustor_injector_capacity": fuel_flow_per_injector - injector_capacity,
        "combustor_injector_pitch_min": float(config.require("injector_pitch_min_m"))
        - geometry.combustor_injector_pitch_m,
    }
    calculated = (
        outlet_pressure,
        pressure_loss_coefficient,
        mass_flow_fuel,
        mass_flow_gas,
        primary_mean_density,
        primary_residence_time,
        injector_capacity,
        energy_residual,
    )
    if not all(math.isfinite(value) for value in calculated):
        raise CombustorModelError(
            FailureCode.NONFINITE_STATE,
            "Map combustor calculation produced a non-finite state",
        )
    if outlet_pressure <= 0.0 or primary_residence_time <= 0.0:
        raise CombustorModelError(
            FailureCode.COMBUSTOR_PRESSURE_INVALID,
            "Map combustor produced a nonpositive pressure or residence time",
        )
    return MapEnhancedCombustorResult(
        inlet=inlet,
        outlet=StationState(turbine_inlet_temperature, outlet_pressure),
        mass_flow_air_kg_s=mass_flow_air_kg_s,
        mass_flow_fuel_kg_s=mass_flow_fuel,
        mass_flow_gas_kg_s=mass_flow_gas,
        fuel_air_ratio=fuel_air_ratio,
        air_fuel_ratio=air_fuel_ratio,
        reference_density_kg_m3=reference_density,
        reference_velocity_m_s=reference_velocity,
        pressure_loss_coefficient=pressure_loss_coefficient,
        total_pressure_loss_Pa=pressure_loss,
        total_pressure_loss_fraction=pressure_loss_fraction,
        equivalence_ratio=equivalence_ratio,
        primary_zone_volume_m3=primary_volume,
        primary_zone_air_flow_kg_s=primary_air_flow,
        primary_zone_gas_flow_kg_s=primary_gas_flow,
        primary_zone_mean_density_kg_m3=primary_mean_density,
        primary_zone_residence_time_s=primary_residence_time,
        fuel_flow_per_injector_kg_s=fuel_flow_per_injector,
        fuel_flow_capacity_per_injector_kg_s=injector_capacity,
        energy_residual_W=energy_residual,
        constraint_residuals=residuals,
        feasible=all(value <= 0.0 for value in residuals.values()),
        combustion_efficiency=combustion_efficiency,
        mass_flux_ratio=mass_flux_ratio,
        map_id=map_result.map_id,
        map_version=map_result.map_version,
        map_source_hash=map_result.source_hash,
        map_in_domain=map_result.in_domain,
        map_extrapolation_distance=map_result.extrapolation_distance,
        map_uncertainty=map_result.uncertainty,
        lean_stability_margin=lean_margin,
        rich_stability_margin=rich_margin,
        fuel_solution_iterations=int(root_info.iterations),
        thermo_model_id=thermo.model_id,
    )
