"""Geometry-aware annular-combustor low-order model for V2.1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from src.physics.engine_match import FailureCode, StationState
from src.physics.geometry_state import GeometryState
from src.physics.model_config import ModelConfig


class CombustorModelError(ValueError):
    """A hard combustor failure carrying a frozen V2.1 failure code."""

    def __init__(self, failure_code: FailureCode, message: str):
        super().__init__(message)
        self.failure_code = failure_code


@dataclass(frozen=True)
class CombustorResult:
    """Complete combustor state consumed by the turbine/shaft model."""

    inlet: StationState
    outlet: StationState
    mass_flow_air_kg_s: float
    mass_flow_fuel_kg_s: float
    mass_flow_gas_kg_s: float
    fuel_air_ratio: float
    air_fuel_ratio: float
    reference_density_kg_m3: float
    reference_velocity_m_s: float
    pressure_loss_coefficient: float
    total_pressure_loss_Pa: float
    total_pressure_loss_fraction: float
    equivalence_ratio: float
    primary_zone_volume_m3: float
    primary_zone_air_flow_kg_s: float
    primary_zone_gas_flow_kg_s: float
    primary_zone_mean_density_kg_m3: float
    primary_zone_residence_time_s: float
    fuel_flow_per_injector_kg_s: float
    fuel_flow_capacity_per_injector_kg_s: float
    energy_residual_W: float
    constraint_residuals: Mapping[str, float]
    feasible: bool


def solve_combustor_loworder(
    geometry: GeometryState,
    mass_flow_air_kg_s: float,
    inlet: StationState,
    config: ModelConfig,
) -> CombustorResult:
    """Solve pressure loss and the common-reference combustor energy balance."""
    turbine_inlet_temperature = (
        geometry.design.turbine_inlet_total_temperature_K
    )
    trial_values = (
        mass_flow_air_kg_s,
        inlet.total_temperature_K,
        inlet.total_pressure_Pa,
        turbine_inlet_temperature,
        geometry.combustor_annulus_area_m2,
        geometry.combustor_volume_m3,
        geometry.combustor_hydraulic_diameter_m,
        geometry.diffuser_outlet_area_m2,
    )
    if not all(math.isfinite(value) for value in trial_values):
        raise CombustorModelError(
            FailureCode.NONFINITE_STATE, "Combustor trial contains non-finite inputs"
        )
    if mass_flow_air_kg_s <= 0.0:
        raise CombustorModelError(
            FailureCode.COMBUSTOR_ENERGY_INVALID,
            "Combustor air mass flow must be positive",
        )
    if (
        inlet.total_temperature_K <= 0.0
        or inlet.total_pressure_Pa <= 0.0
        or turbine_inlet_temperature <= 0.0
    ):
        raise CombustorModelError(
            FailureCode.NONFINITE_STATE,
            "Combustor total temperatures and inlet pressure must be positive",
        )
    if (
        geometry.combustor_annulus_area_m2 <= 0.0
        or geometry.combustor_volume_m3 <= 0.0
        or geometry.combustor_hydraulic_diameter_m <= 0.0
    ):
        raise CombustorModelError(
            FailureCode.GEOMETRY_INVALID,
            "Combustor geometry must have positive area, volume, and hydraulic diameter",
        )

    gas_constant_cold = float(config.require("gas_constant_cold_J_kg_K"))
    reference_density = inlet.total_pressure_Pa / (
        gas_constant_cold * inlet.total_temperature_K
    )
    reference_velocity = mass_flow_air_kg_s / (
        reference_density * geometry.combustor_annulus_area_m2
    )
    area_ratio = (
        geometry.diffuser_outlet_area_m2
        / geometry.combustor_annulus_area_m2
    )
    pressure_loss_coefficient = (
        float(config.require("combustor_pressure_loss_K0"))
        + float(config.require("combustor_pressure_loss_K_LD"))
        * geometry.design.combustor_length_m
        / geometry.combustor_hydraulic_diameter_m
        + float(config.require("combustor_pressure_loss_K_area"))
        * (area_ratio - 1.0) ** 2
    )
    if not math.isfinite(pressure_loss_coefficient) or pressure_loss_coefficient < 0.0:
        raise CombustorModelError(
            FailureCode.COMBUSTOR_PRESSURE_INVALID,
            "Combustor pressure-loss coefficient must be finite and nonnegative",
        )
    pressure_loss = (
        pressure_loss_coefficient
        * 0.5
        * reference_density
        * reference_velocity**2
    )
    outlet_pressure = inlet.total_pressure_Pa - pressure_loss
    pressure_loss_fraction = pressure_loss / inlet.total_pressure_Pa
    if (
        not math.isfinite(outlet_pressure)
        or outlet_pressure <= 0.0
        or not math.isfinite(pressure_loss_fraction)
        or pressure_loss_fraction < 0.0
    ):
        raise CombustorModelError(
            FailureCode.COMBUSTOR_PRESSURE_INVALID,
            "Combustor pressure loss leaves no positive turbine-inlet pressure",
        )

    enthalpy_reference_temperature = float(
        config.require("enthalpy_reference_temperature_K")
    )
    cp_cold = float(config.require("cp_cold_J_kg_K"))
    cp_hot = float(config.require("cp_hot_J_kg_K"))
    combustion_efficiency = float(config.require("combustion_efficiency"))
    lower_heating_value = float(
        config.require("fuel_lower_heating_value_J_kg")
    )
    fuel_sensible_enthalpy = float(
        config.require("fuel_sensible_enthalpy_at_reference_J_kg")
    )
    air_enthalpy = cp_cold * (
        inlet.total_temperature_K - enthalpy_reference_temperature
    )
    gas_enthalpy = cp_hot * (
        turbine_inlet_temperature - enthalpy_reference_temperature
    )
    fuel_energy = (
        combustion_efficiency * lower_heating_value + fuel_sensible_enthalpy
    )
    numerator = gas_enthalpy - air_enthalpy
    denominator = fuel_energy - gas_enthalpy
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or numerator <= 0.0
        or denominator <= 0.0
    ):
        raise CombustorModelError(
            FailureCode.COMBUSTOR_ENERGY_INVALID,
            "Combustor energy equation has no positive fuel solution",
        )
    fuel_air_ratio = numerator / denominator
    if not math.isfinite(fuel_air_ratio) or fuel_air_ratio <= 0.0:
        raise CombustorModelError(
            FailureCode.COMBUSTOR_ENERGY_INVALID,
            "Combustor fuel-air ratio must be finite and positive",
        )
    mass_flow_fuel = fuel_air_ratio * mass_flow_air_kg_s
    mass_flow_gas = mass_flow_air_kg_s + mass_flow_fuel
    air_fuel_ratio = 1.0 / fuel_air_ratio
    equivalence_ratio = fuel_air_ratio / float(
        config.require("stoichiometric_fuel_air_ratio")
    )

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
    primary_mean_pressure = 0.5 * (
        inlet.total_pressure_Pa + outlet_pressure
    )
    primary_mean_density = primary_mean_pressure / (
        float(config.require("gas_constant_hot_J_kg_K"))
        * primary_mean_temperature
    )
    primary_residence_time = (
        primary_mean_density * primary_volume / primary_gas_flow
    )

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

    energy_residual = (
        mass_flow_air_kg_s * air_enthalpy
        + mass_flow_fuel * fuel_energy
        - mass_flow_gas * gas_enthalpy
    )
    calculated_values = (
        mass_flow_fuel,
        mass_flow_gas,
        air_fuel_ratio,
        equivalence_ratio,
        primary_mean_density,
        primary_residence_time,
        fuel_flow_per_injector,
        injector_capacity,
        energy_residual,
    )
    if not all(math.isfinite(value) for value in calculated_values):
        raise CombustorModelError(
            FailureCode.NONFINITE_STATE,
            "Combustor calculation produced a non-finite state",
        )
    if primary_residence_time <= 0.0 or injector_capacity <= 0.0:
        raise CombustorModelError(
            FailureCode.NONFINITE_STATE,
            "Combustor residence time and injector capacity must be positive",
        )

    residuals = {
        "combustor_pressure_loss_fraction_max": pressure_loss_fraction
        - float(config.require("combustor_total_pressure_loss_fraction_max")),
        "combustor_primary_residence_time_min": float(
            config.require("combustor_primary_residence_time_min_s")
        )
        - primary_residence_time,
        "combustor_equivalence_ratio_min": float(
            config.require("equivalence_ratio_min")
        )
        - equivalence_ratio,
        "combustor_equivalence_ratio_max": equivalence_ratio
        - float(config.require("equivalence_ratio_max")),
        "combustor_injector_capacity": fuel_flow_per_injector
        - injector_capacity,
        "combustor_injector_pitch_min": float(
            config.require("injector_pitch_min_m")
        )
        - geometry.combustor_injector_pitch_m,
    }

    return CombustorResult(
        inlet=inlet,
        outlet=StationState(
            total_temperature_K=turbine_inlet_temperature,
            total_pressure_Pa=outlet_pressure,
        ),
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
    )
