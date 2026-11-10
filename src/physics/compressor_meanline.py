"""Geometry-aware centrifugal-compressor meanline model for V2.1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from src.physics.engine_match import FailureCode, StationState
from src.physics.geometry_state import GeometryState
from src.physics.model_config import ModelConfig


class CompressorModelError(ValueError):
    """A hard compressor failure carrying a frozen V2.1 failure code."""

    def __init__(self, failure_code: FailureCode, message: str):
        super().__init__(message)
        self.failure_code = failure_code


@dataclass(frozen=True)
class CompressorResult:
    """Complete compressor state used by the downstream combustor model."""

    inlet: StationState
    outlet: StationState
    mass_flow_air_kg_s: float
    reference_density_kg_m3: float
    reference_meridional_velocity_m_s: float
    flow_coefficient_ref: float
    slip_factor: float
    exit_tangential_velocity_m_s: float
    specific_work_J_kg: float
    loading_coefficient: float
    tip_mach: float
    reynolds_number: float
    dynamic_viscosity_Pa_s: float
    total_to_total_efficiency: float
    pressure_ratio: float
    shaft_power_W: float
    calibration_domain_residuals: Mapping[str, float]
    calibration_domain_feasible: bool


def sutherland_dynamic_viscosity(
    temperature_K: float,
    reference_viscosity_Pa_s: float,
    reference_temperature_K: float,
    sutherland_constant_K: float,
) -> float:
    """Evaluate Sutherland's law with explicit reference-state parameters."""
    values = (
        temperature_K,
        reference_viscosity_Pa_s,
        reference_temperature_K,
        sutherland_constant_K,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise CompressorModelError(
            FailureCode.NONFINITE_STATE,
            "Sutherland viscosity inputs must be finite and positive",
        )
    viscosity = (
        reference_viscosity_Pa_s
        * (temperature_K / reference_temperature_K) ** 1.5
        * (reference_temperature_K + sutherland_constant_K)
        / (temperature_K + sutherland_constant_K)
    )
    if not math.isfinite(viscosity) or viscosity <= 0.0:
        raise CompressorModelError(
            FailureCode.NONFINITE_STATE,
            "Sutherland viscosity calculation produced an invalid value",
        )
    return viscosity


def solve_compressor_meanline(
    geometry: GeometryState,
    mass_flow_air_kg_s: float,
    inlet: StationState,
    config: ModelConfig,
) -> CompressorResult:
    """Solve V2.1 compressor equations without clipping invalid closures."""
    trial_values = (
        mass_flow_air_kg_s,
        inlet.total_temperature_K,
        inlet.total_pressure_Pa,
        geometry.compressor_exit_effective_area_m2,
        geometry.compressor_tip_speed_m_s,
    )
    if not all(math.isfinite(value) for value in trial_values):
        raise CompressorModelError(
            FailureCode.NONFINITE_STATE, "Compressor trial contains non-finite inputs"
        )
    if mass_flow_air_kg_s <= 0.0:
        raise CompressorModelError(
            FailureCode.COMPRESSOR_NONPOSITIVE_WORK,
            "Compressor air mass flow must be positive",
        )
    if inlet.total_temperature_K <= 0.0 or inlet.total_pressure_Pa <= 0.0:
        raise CompressorModelError(
            FailureCode.NONFINITE_STATE,
            "Compressor inlet total temperature and pressure must be positive",
        )
    if (
        geometry.compressor_exit_effective_area_m2 <= 0.0
        or geometry.compressor_tip_speed_m_s <= 0.0
    ):
        raise CompressorModelError(
            FailureCode.GEOMETRY_INVALID,
            "Compressor effective area and tip speed must be positive",
        )

    gas_constant = float(config.require("gas_constant_cold_J_kg_K"))
    cp = float(config.require("cp_cold_J_kg_K"))
    gamma = float(config.require("gamma_cold"))
    rho_ref = inlet.total_pressure_Pa / (
        gas_constant * inlet.total_temperature_K
    )
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
    specific_work = tip_speed * tangential_velocity
    if not math.isfinite(specific_work) or specific_work <= 0.0:
        raise CompressorModelError(
            FailureCode.COMPRESSOR_NONPOSITIVE_WORK,
            f"Compressor Euler work must be positive, received {specific_work}",
        )
    psi = specific_work / tip_speed**2

    speed_of_sound = math.sqrt(
        gamma * gas_constant * inlet.total_temperature_K
    )
    tip_mach = tip_speed / speed_of_sound
    viscosity = sutherland_dynamic_viscosity(
        inlet.total_temperature_K,
        float(config.require("dynamic_viscosity_cold_reference_Pa_s")),
        float(config.require("sutherland_cold_reference_temperature_K")),
        float(config.require("sutherland_cold_constant_K")),
    )
    reynolds = (
        rho_ref
        * tip_speed
        * geometry.design.compressor_impeller_exit_diameter_m
        / viscosity
    )
    if not math.isfinite(reynolds) or reynolds <= 0.0:
        raise CompressorModelError(
            FailureCode.NONFINITE_STATE,
            "Compressor Reynolds number must be finite and positive",
        )

    phi_design = float(config.require("compressor_phi_design"))
    psi_design = float(config.require("compressor_psi_design"))
    mach_start = float(config.require("compressor_tip_mach_penalty_start"))
    reynolds_design = float(config.require("compressor_reynolds_design"))
    delta_phi = (phi_ref - phi_design) / phi_design
    delta_psi = (psi - psi_design) / psi_design
    delta_mach = max(0.0, (tip_mach - mach_start) / mach_start)
    delta_reynolds = max(0.0, reynolds_design / reynolds - 1.0)
    efficiency = (
        float(config.require("compressor_eta_design"))
        - float(config.require("compressor_efficiency_phi_penalty"))
        * delta_phi**2
        - float(config.require("compressor_efficiency_psi_penalty"))
        * delta_psi**2
        - float(config.require("compressor_efficiency_mach_penalty"))
        * delta_mach**2
        - float(config.require("compressor_efficiency_reynolds_penalty"))
        * delta_reynolds**2
    )
    if not math.isfinite(efficiency) or not 0.0 < efficiency < 1.0:
        raise CompressorModelError(
            FailureCode.COMPRESSOR_EFFICIENCY_INVALID,
            f"Compressor efficiency must be in (0, 1), received {efficiency}",
        )

    outlet_temperature = inlet.total_temperature_K + specific_work / cp
    isentropic_outlet_temperature = (
        inlet.total_temperature_K + efficiency * specific_work / cp
    )
    pressure_ratio = (
        isentropic_outlet_temperature / inlet.total_temperature_K
    ) ** (gamma / (gamma - 1.0))
    outlet_pressure = pressure_ratio * inlet.total_pressure_Pa
    shaft_power = mass_flow_air_kg_s * specific_work
    outlet_values = (
        outlet_temperature,
        isentropic_outlet_temperature,
        pressure_ratio,
        outlet_pressure,
        shaft_power,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in outlet_values):
        raise CompressorModelError(
            FailureCode.NONFINITE_STATE,
            "Compressor outlet calculation produced an invalid state",
        )

    residuals = {
        "compressor_phi_min": float(config.require("compressor_phi_min"))
        - phi_ref,
        "compressor_phi_max": phi_ref
        - float(config.require("compressor_phi_max")),
        "compressor_psi_min": float(config.require("compressor_psi_min")) - psi,
        "compressor_psi_max": psi
        - float(config.require("compressor_psi_max")),
        "compressor_tip_mach_max": tip_mach
        - float(config.require("compressor_tip_mach_max")),
        "compressor_efficiency_min": float(
            config.require("compressor_efficiency_min")
        )
        - efficiency,
        "compressor_tip_speed_limit": tip_speed
        - float(config.require("compressor_tip_speed_limit_m_s")),
    }

    return CompressorResult(
        inlet=inlet,
        outlet=StationState(
            total_temperature_K=outlet_temperature,
            total_pressure_Pa=outlet_pressure,
        ),
        mass_flow_air_kg_s=mass_flow_air_kg_s,
        reference_density_kg_m3=rho_ref,
        reference_meridional_velocity_m_s=meridional_velocity,
        flow_coefficient_ref=phi_ref,
        slip_factor=slip_factor,
        exit_tangential_velocity_m_s=tangential_velocity,
        specific_work_J_kg=specific_work,
        loading_coefficient=psi,
        tip_mach=tip_mach,
        reynolds_number=reynolds,
        dynamic_viscosity_Pa_s=viscosity,
        total_to_total_efficiency=efficiency,
        pressure_ratio=pressure_ratio,
        shaft_power_W=shaft_power,
        calibration_domain_residuals=residuals,
        calibration_domain_feasible=all(value <= 0.0 for value in residuals.values()),
    )
