"""Thermodynamically consistent convergent-nozzle model for V2.1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from src.physics.engine_match import (
    Environment,
    FailureCode,
    NozzleExitState,
    StationState,
)
from src.physics.geometry_state import GeometryState
from src.physics.model_config import ModelConfig
from src.physics.turbine_meanline import TurbineResult


class NozzleModelError(ValueError):
    """A hard nozzle failure carrying the frozen V2.1 failure code."""

    def __init__(self, failure_code: FailureCode, message: str):
        super().__init__(message)
        self.failure_code = failure_code


@dataclass(frozen=True)
class NozzleResult:
    """Complete nozzle state and flow capacity for engine matching."""

    turbine_outlet: StationState
    nozzle_total: StationState
    exit: NozzleExitState
    choked: bool
    critical_static_pressure_Pa: float
    total_pressure_recovery: float
    discharge_coefficient: float
    mass_flow_gas_kg_s: float
    mass_flow_capacity_kg_s: float
    mass_flow_residual_kg_s: float
    normalized_flow_residual: float
    velocity_consistency_residual_m_s: float
    pressure_thrust_N: float
    state_constraint_residuals: Mapping[str, float]


def compressible_mass_flow_function(
    mach: float, gamma: float, gas_constant_J_kg_K: float
) -> float:
    """Return the V2.1 mass-flow function Q(M, gamma)."""
    if not (
        math.isfinite(mach)
        and 0.0 <= mach <= 1.0
        and math.isfinite(gamma)
        and gamma > 1.0
        and math.isfinite(gas_constant_J_kg_K)
        and gas_constant_J_kg_K > 0.0
    ):
        raise NozzleModelError(
            FailureCode.NOZZLE_STATE_INVALID,
            "Nozzle mass-flow function inputs are outside their physical domain",
        )
    return (
        math.sqrt(gamma / gas_constant_J_kg_K)
        * mach
        * (1.0 + 0.5 * (gamma - 1.0) * mach**2)
        ** (-(gamma + 1.0) / (2.0 * (gamma - 1.0)))
    )


def solve_convergent_nozzle(
    geometry: GeometryState,
    turbine: TurbineResult,
    environment: Environment,
    config: ModelConfig,
) -> NozzleResult:
    """Solve choked or subsonic nozzle state without a velocity efficiency."""
    turbine_outlet = turbine.outlet
    inputs = (
        turbine_outlet.total_temperature_K,
        turbine_outlet.total_pressure_Pa,
        turbine.mass_flow_gas_kg_s,
        environment.pressure_Pa,
        geometry.nozzle_throat_area_m2,
    )
    if not all(math.isfinite(value) for value in inputs):
        raise NozzleModelError(
            FailureCode.NONFINITE_STATE, "Nozzle trial contains non-finite inputs"
        )
    if (
        turbine_outlet.total_temperature_K <= 0.0
        or turbine_outlet.total_pressure_Pa <= 0.0
        or turbine.mass_flow_gas_kg_s <= 0.0
        or environment.pressure_Pa <= 0.0
        or geometry.nozzle_throat_area_m2 <= 0.0
    ):
        raise NozzleModelError(
            FailureCode.NOZZLE_STATE_INVALID,
            "Nozzle temperatures, pressures, gas flow, and area must be positive",
        )

    pressure_recovery = float(config.require("nozzle_total_pressure_recovery"))
    discharge_coefficient = float(config.require("nozzle_discharge_coefficient"))
    if not 0.0 < pressure_recovery <= 1.0:
        raise NozzleModelError(
            FailureCode.NOZZLE_STATE_INVALID,
            "Nozzle total-pressure recovery must be in (0, 1]",
        )
    if not 0.0 < discharge_coefficient <= 1.0:
        raise NozzleModelError(
            FailureCode.NOZZLE_STATE_INVALID,
            "Nozzle discharge coefficient must be in (0, 1]",
        )

    total_temperature = turbine_outlet.total_temperature_K
    total_pressure = pressure_recovery * turbine_outlet.total_pressure_Pa
    if not math.isfinite(total_pressure) or total_pressure <= environment.pressure_Pa:
        raise NozzleModelError(
            FailureCode.NOZZLE_STATE_INVALID,
            "Nozzle recovered total pressure must exceed ambient back pressure",
        )

    gamma = float(config.require("gamma_hot"))
    gas_constant = float(config.require("gas_constant_hot_J_kg_K"))
    if gamma <= 1.0 or gas_constant <= 0.0:
        raise NozzleModelError(
            FailureCode.NOZZLE_STATE_INVALID,
            "Nozzle gas properties are outside their physical domain",
        )
    critical_pressure = total_pressure * (
        2.0 / (gamma + 1.0)
    ) ** (gamma / (gamma - 1.0))
    choked = environment.pressure_Pa <= critical_pressure

    if choked:
        exit_mach = 1.0
        exit_temperature = total_temperature * 2.0 / (gamma + 1.0)
        exit_pressure = critical_pressure
    else:
        pressure_term = (
            total_pressure / environment.pressure_Pa
        ) ** ((gamma - 1.0) / gamma) - 1.0
        if not math.isfinite(pressure_term) or pressure_term <= 0.0:
            raise NozzleModelError(
                FailureCode.NOZZLE_STATE_INVALID,
                "Subsonic nozzle pressure ratio cannot produce a positive Mach number",
            )
        exit_mach = math.sqrt(2.0 * pressure_term / (gamma - 1.0))
        exit_pressure = environment.pressure_Pa
        exit_temperature = total_temperature / (
            1.0 + 0.5 * (gamma - 1.0) * exit_mach**2
        )

    if not 0.0 < exit_mach <= 1.0 or exit_temperature <= 0.0:
        raise NozzleModelError(
            FailureCode.NOZZLE_STATE_INVALID,
            "Nozzle exit state lies outside the convergent-nozzle domain",
        )
    local_sound_speed = math.sqrt(gamma * gas_constant * exit_temperature)
    exit_velocity = exit_mach * local_sound_speed
    mass_flow_capacity = (
        discharge_coefficient
        * geometry.nozzle_throat_area_m2
        * total_pressure
        / math.sqrt(total_temperature)
        * compressible_mass_flow_function(exit_mach, gamma, gas_constant)
    )
    mass_flow_residual = turbine.mass_flow_gas_kg_s - mass_flow_capacity
    residual_epsilon = float(
        config.solver_candidate["mass_flow_residual_epsilon_kg_s"]
    )
    if not math.isfinite(residual_epsilon) or residual_epsilon <= 0.0:
        raise NozzleModelError(
            FailureCode.NOZZLE_STATE_INVALID,
            "Mass-flow residual denominator epsilon must be positive",
        )
    normalized_flow_residual = mass_flow_residual / max(
        turbine.mass_flow_gas_kg_s, residual_epsilon
    )
    velocity_consistency_residual = exit_velocity - (
        exit_mach * math.sqrt(gamma * gas_constant * exit_temperature)
    )
    pressure_thrust = (
        (exit_pressure - environment.pressure_Pa)
        * geometry.nozzle_exit_area_m2
    )

    calculated = (
        total_pressure,
        critical_pressure,
        exit_mach,
        exit_temperature,
        exit_pressure,
        exit_velocity,
        mass_flow_capacity,
        mass_flow_residual,
        normalized_flow_residual,
        velocity_consistency_residual,
        pressure_thrust,
    )
    if not all(math.isfinite(value) for value in calculated):
        raise NozzleModelError(
            FailureCode.NONFINITE_STATE,
            "Nozzle calculation produced a non-finite state",
        )
    if mass_flow_capacity <= 0.0 or exit_velocity <= 0.0:
        raise NozzleModelError(
            FailureCode.NOZZLE_STATE_INVALID,
            "Nozzle capacity and exit velocity must be positive",
        )

    return NozzleResult(
        turbine_outlet=turbine_outlet,
        nozzle_total=StationState(
            total_temperature_K=total_temperature,
            total_pressure_Pa=total_pressure,
        ),
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
        state_constraint_residuals={
            "nozzle_total_pressure_usable": environment.pressure_Pa
            - total_pressure,
        },
    )
