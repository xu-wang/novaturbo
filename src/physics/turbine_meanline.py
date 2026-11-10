"""Geometry-aware single-stage turbine and shaft-power model for V2.1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from src.physics.combustor_loworder import CombustorResult
from src.physics.compressor_meanline import (
    CompressorModelError,
    sutherland_dynamic_viscosity,
)
from src.physics.engine_match import FailureCode, StationState
from src.physics.geometry_state import GeometryState
from src.physics.model_config import CalibrationPackError, ModelConfig
from src.physics.thermo import ThermoModel, ThermoModelError


class TurbineModelError(ValueError):
    """A hard turbine failure carrying a frozen V2.1 failure code."""

    def __init__(self, failure_code: FailureCode, message: str):
        super().__init__(message)
        self.failure_code = failure_code


@dataclass(frozen=True)
class TurbineResult:
    """Complete turbine state consumed by the V2.1 nozzle model."""

    inlet: StationState
    outlet: StationState
    mass_flow_gas_kg_s: float
    compressor_power_W: float
    auxiliary_power_W: float
    mechanical_efficiency: float
    specific_work_J_kg: float
    shaft_power_residual_W: float
    reference_density_kg_m3: float
    axial_velocity_m_s: float
    flow_coefficient: float
    loading_coefficient: float
    mean_radius_mach: float
    reynolds_number: float
    dynamic_viscosity_Pa_s: float
    solidity: float
    total_to_total_efficiency: float
    isentropic_outlet_temperature_K: float
    expansion_pressure_ratio: float
    corrected_mass_flux_kg_s_m2: float
    corrected_mass_flux_reference_temperature_K: float
    corrected_mass_flux_reference_pressure_Pa: float
    constraint_residuals: Mapping[str, float]
    feasible: bool


def _corrected_mass_flux_reference(config: ModelConfig) -> tuple[float, float]:
    try:
        reference = config.reference_states["turbine_corrected_mass_flux"]
        temperature = float(reference["temperature_K"])
        pressure = float(reference["pressure_Pa"])
        unit = str(reference["unit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationPackError(
            "turbine_corrected_mass_flux reference state is incomplete"
        ) from exc
    if unit != "kg/(s*m^2)":
        raise CalibrationPackError(
            "turbine_corrected_mass_flux reference unit must be kg/(s*m^2)"
        )
    if not (
        math.isfinite(temperature)
        and temperature > 0.0
        and math.isfinite(pressure)
        and pressure > 0.0
    ):
        raise CalibrationPackError(
            "Turbine corrected-mass-flux reference values must be positive"
        )
    return temperature, pressure


def solve_turbine_meanline(
    geometry: GeometryState,
    combustor: CombustorResult,
    compressor_power_W: float,
    config: ModelConfig,
    *,
    efficiency_override: float | None = None,
    capacity_constraint_residuals: Mapping[str, float] | None = None,
    thermo: ThermoModel | None = None,
    fuel_air_ratio: float | None = None,
) -> TurbineResult:
    """Solve shaft balance with the V2.1 or an injected efficiency closure."""
    inlet = combustor.outlet
    mass_flow_gas = combustor.mass_flow_gas_kg_s
    trial_values = (
        mass_flow_gas,
        compressor_power_W,
        inlet.total_temperature_K,
        inlet.total_pressure_Pa,
        geometry.turbine_annulus_area_m2,
        geometry.turbine_mean_speed_m_s,
        geometry.turbine_chord_m,
        geometry.turbine_solidity,
    )
    if not all(math.isfinite(value) for value in trial_values):
        raise TurbineModelError(
            FailureCode.NONFINITE_STATE, "Turbine trial contains non-finite inputs"
        )
    if mass_flow_gas <= 0.0 or inlet.total_temperature_K <= 0.0:
        raise TurbineModelError(
            FailureCode.TURBINE_POWER_INSUFFICIENT,
            "Turbine requires positive gas flow and inlet temperature",
        )
    if inlet.total_pressure_Pa <= 0.0:
        raise TurbineModelError(
            FailureCode.TURBINE_PRESSURE_INVALID,
            "Turbine inlet total pressure must be positive",
        )
    if (
        geometry.turbine_annulus_area_m2 <= 0.0
        or geometry.turbine_mean_speed_m_s <= 0.0
        or geometry.turbine_chord_m <= 0.0
    ):
        raise TurbineModelError(
            FailureCode.GEOMETRY_INVALID,
            "Turbine annulus area, mean speed, and chord must be positive",
        )

    mechanical_efficiency = float(
        config.require("shaft_mechanical_efficiency")
    )
    auxiliary_power = float(config.require("shaft_auxiliary_power_W"))
    required_power = compressor_power_W + auxiliary_power
    if not 0.0 < mechanical_efficiency <= 1.0 or required_power <= 0.0:
        raise TurbineModelError(
            FailureCode.TURBINE_POWER_INSUFFICIENT,
            "Shaft efficiency and required turbine power are invalid",
        )
    specific_work = required_power / (mechanical_efficiency * mass_flow_gas)
    if thermo is None:
        cp_hot = float(config.require("cp_hot_J_kg_K"))
        outlet_temperature = inlet.total_temperature_K - specific_work / cp_hot
    else:
        if (
            fuel_air_ratio is None
            or not math.isfinite(fuel_air_ratio)
            or fuel_air_ratio < 0.0
        ):
            raise TurbineModelError(
                FailureCode.TURBINE_POWER_INSUFFICIENT,
                "Variable-property turbine requires a finite nonnegative FAR",
            )
        try:
            inlet_enthalpy = thermo.h_gas(
                inlet.total_temperature_K, fuel_air_ratio
            )
            outlet_temperature = thermo.gas_temperature_from_h(
                inlet_enthalpy - specific_work, fuel_air_ratio
            )
        except ThermoModelError as exc:
            raise TurbineModelError(
                FailureCode.TURBINE_POWER_INSUFFICIENT,
                "Variable-property turbine enthalpy solve failed",
            ) from exc
    if (
        not math.isfinite(specific_work)
        or specific_work <= 0.0
        or not math.isfinite(outlet_temperature)
        or outlet_temperature <= 0.0
    ):
        raise TurbineModelError(
            FailureCode.TURBINE_POWER_INSUFFICIENT,
            "Available turbine enthalpy cannot supply the required shaft power",
        )

    gas_constant = (
        float(config.require("gas_constant_hot_J_kg_K"))
        if thermo is None
        else thermo.gas_constant_gas_J_kg_K
    )
    try:
        gamma = (
            float(config.require("gamma_hot"))
            if thermo is None
            else thermo.gamma_gas(inlet.total_temperature_K, fuel_air_ratio)
        )
    except ThermoModelError as exc:
        raise TurbineModelError(FailureCode.NONFINITE_STATE, str(exc)) from exc
    reference_density = inlet.total_pressure_Pa / (
        gas_constant * inlet.total_temperature_K
    )
    axial_velocity = mass_flow_gas / (
        reference_density * geometry.turbine_annulus_area_m2
    )
    mean_speed = geometry.turbine_mean_speed_m_s
    flow_coefficient = axial_velocity / mean_speed
    loading_coefficient = specific_work / mean_speed**2
    mean_radius_mach = mean_speed / math.sqrt(
        gamma * gas_constant * inlet.total_temperature_K
    )
    try:
        viscosity = (
            sutherland_dynamic_viscosity(
                inlet.total_temperature_K,
                float(config.require("dynamic_viscosity_hot_reference_Pa_s")),
                float(config.require("sutherland_hot_reference_temperature_K")),
                float(config.require("sutherland_hot_constant_K")),
            )
            if thermo is None
            else thermo.mu_gas(inlet.total_temperature_K, fuel_air_ratio)
        )
    except (CompressorModelError, ThermoModelError) as exc:
        raise TurbineModelError(FailureCode.NONFINITE_STATE, str(exc)) from exc
    reynolds = (
        reference_density
        * axial_velocity
        * geometry.turbine_chord_m
        / viscosity
    )
    if not math.isfinite(reynolds) or reynolds <= 0.0:
        raise TurbineModelError(
            FailureCode.NONFINITE_STATE,
            "Turbine Reynolds number must be finite and positive",
        )

    reference_temperature, reference_pressure = _corrected_mass_flux_reference(
        config
    )
    corrected_mass_flux = (
        mass_flow_gas
        * math.sqrt(inlet.total_temperature_K / reference_temperature)
        / (
            (inlet.total_pressure_Pa / reference_pressure)
            * geometry.turbine_annulus_area_m2
        )
    )

    phi_design = float(config.require("turbine_phi_design"))
    psi_design = float(config.require("turbine_psi_design"))
    solidity_design = float(config.require("turbine_solidity_design"))
    mach_start = float(config.require("turbine_mean_radius_mach_penalty_start"))
    reynolds_design = float(config.require("turbine_reynolds_design"))
    delta_phi = (flow_coefficient - phi_design) / phi_design
    delta_psi = (loading_coefficient - psi_design) / psi_design
    delta_solidity = (
        geometry.turbine_solidity - solidity_design
    ) / solidity_design
    delta_mach = max(0.0, (mean_radius_mach - mach_start) / mach_start)
    delta_reynolds = max(0.0, reynolds_design / reynolds - 1.0)
    if efficiency_override is None:
        efficiency = (
            float(config.require("turbine_eta_design"))
            - float(config.require("turbine_efficiency_phi_penalty"))
            * delta_phi**2
            - float(config.require("turbine_efficiency_psi_penalty"))
            * delta_psi**2
            - float(config.require("turbine_efficiency_solidity_penalty"))
            * delta_solidity**2
            - float(config.require("turbine_efficiency_mach_penalty"))
            * delta_mach**2
            - float(config.require("turbine_efficiency_reynolds_penalty"))
            * delta_reynolds**2
        )
    else:
        efficiency = float(efficiency_override)
    if not math.isfinite(efficiency) or not 0.0 < efficiency < 1.0:
        raise TurbineModelError(
            FailureCode.TURBINE_EFFICIENCY_INVALID,
            f"Turbine efficiency must be in (0, 1), received {efficiency}",
        )

    if thermo is None:
        isentropic_outlet_temperature = inlet.total_temperature_K - (
            inlet.total_temperature_K - outlet_temperature
        ) / efficiency
    else:
        try:
            isentropic_outlet_temperature = thermo.gas_temperature_from_h(
                inlet_enthalpy - specific_work / efficiency,
                fuel_air_ratio,
            )
        except ThermoModelError as exc:
            raise TurbineModelError(
                FailureCode.TURBINE_PRESSURE_INVALID,
                "Variable-property isentropic turbine state solve failed",
            ) from exc
    if (
        not math.isfinite(isentropic_outlet_temperature)
        or isentropic_outlet_temperature <= 0.0
        or isentropic_outlet_temperature >= inlet.total_temperature_K
    ):
        raise TurbineModelError(
            FailureCode.TURBINE_PRESSURE_INVALID,
            "Turbine isentropic outlet temperature is outside its physical domain",
        )
    if thermo is None:
        pressure_ratio = (
            isentropic_outlet_temperature / inlet.total_temperature_K
        ) ** (gamma / (gamma - 1.0))
    else:
        try:
            pressure_ratio = thermo.gas_isentropic_pressure_ratio(
                inlet.total_temperature_K,
                isentropic_outlet_temperature,
                fuel_air_ratio,
            )
        except ThermoModelError as exc:
            raise TurbineModelError(
                FailureCode.TURBINE_PRESSURE_INVALID,
                "Variable-property turbine entropy solve failed",
            ) from exc
    outlet_pressure = inlet.total_pressure_Pa * pressure_ratio
    if (
        not math.isfinite(outlet_pressure)
        or outlet_pressure <= 0.0
        or outlet_pressure >= inlet.total_pressure_Pa
    ):
        raise TurbineModelError(
            FailureCode.TURBINE_PRESSURE_INVALID,
            "Turbine outlet total pressure is outside its physical domain",
        )
    shaft_power_residual = (
        mechanical_efficiency * mass_flow_gas * specific_work - required_power
    )

    residuals = {
        "turbine_phi_min": float(config.require("turbine_phi_min"))
        - flow_coefficient,
        "turbine_phi_max": flow_coefficient
        - float(config.require("turbine_phi_max")),
        "turbine_psi_max": loading_coefficient
        - float(config.require("turbine_psi_max")),
        "turbine_solidity_min": float(config.require("turbine_solidity_min"))
        - geometry.turbine_solidity,
        "turbine_solidity_max": geometry.turbine_solidity
        - float(config.require("turbine_solidity_max")),
        "turbine_mean_radius_mach_max": mean_radius_mach
        - float(config.require("turbine_mean_radius_mach_max")),
        "turbine_efficiency_min": float(
            config.require("turbine_efficiency_min")
        )
        - efficiency,
        "turbine_corrected_mass_flux_min": float(
            config.require("turbine_corrected_mass_flux_min")
        )
        - corrected_mass_flux,
        "turbine_corrected_mass_flux_max": corrected_mass_flux
        - float(config.require("turbine_corrected_mass_flux_max")),
        "turbine_tip_speed_limit": geometry.turbine_tip_speed_m_s
        - float(config.require("turbine_tip_speed_limit_m_s")),
    }
    if capacity_constraint_residuals is not None:
        expected_capacity_keys = {
            "turbine_map_coordinate_domain",
            "turbine_map_minimum_capacity",
            "turbine_map_maximum_capacity",
        }
        if set(capacity_constraint_residuals) != expected_capacity_keys:
            raise TurbineModelError(
                FailureCode.INVALID_CALIBRATION,
                "Turbine map capacity residual contract mismatch",
            )
        residuals.pop("turbine_corrected_mass_flux_min")
        residuals.pop("turbine_corrected_mass_flux_max")
        residuals.update(
            {
                name: float(value)
                for name, value in capacity_constraint_residuals.items()
            }
        )
    calculated_values = (
        specific_work,
        outlet_temperature,
        reference_density,
        axial_velocity,
        flow_coefficient,
        loading_coefficient,
        mean_radius_mach,
        corrected_mass_flux,
        efficiency,
        outlet_pressure,
        shaft_power_residual,
        *residuals.values(),
    )
    if not all(math.isfinite(value) for value in calculated_values):
        raise TurbineModelError(
            FailureCode.NONFINITE_STATE,
            "Turbine calculation produced a non-finite state",
        )

    return TurbineResult(
        inlet=inlet,
        outlet=StationState(
            total_temperature_K=outlet_temperature,
            total_pressure_Pa=outlet_pressure,
        ),
        mass_flow_gas_kg_s=mass_flow_gas,
        compressor_power_W=compressor_power_W,
        auxiliary_power_W=auxiliary_power,
        mechanical_efficiency=mechanical_efficiency,
        specific_work_J_kg=specific_work,
        shaft_power_residual_W=shaft_power_residual,
        reference_density_kg_m3=reference_density,
        axial_velocity_m_s=axial_velocity,
        flow_coefficient=flow_coefficient,
        loading_coefficient=loading_coefficient,
        mean_radius_mach=mean_radius_mach,
        reynolds_number=reynolds,
        dynamic_viscosity_Pa_s=viscosity,
        solidity=geometry.turbine_solidity,
        total_to_total_efficiency=efficiency,
        isentropic_outlet_temperature_K=isentropic_outlet_temperature,
        expansion_pressure_ratio=inlet.total_pressure_Pa / outlet_pressure,
        corrected_mass_flux_kg_s_m2=corrected_mass_flux,
        corrected_mass_flux_reference_temperature_K=reference_temperature,
        corrected_mass_flux_reference_pressure_Pa=reference_pressure,
        constraint_residuals=residuals,
        feasible=all(value <= 0.0 for value in residuals.values()),
    )
