"""WP19-8 structural, thermal, lattice, and manufacturing constraints."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from src.physics.geometry_state import (
    GeometryState,
    geometry19_coordinate_bounds,
)
from src.physics.mass_model_v2 import MassBreakdown
from src.physics.model_config import ModelConfig


class FormalConstraintError(ValueError):
    """Hard failure raised when a formal WP19-8 constraint is undefined."""

    failure_code = "FORMAL_CONSTRAINT_INVALID"


@dataclass(frozen=True)
class FormalConstraintEvaluation:
    """All post-cycle formal constraints using the convention g <= 0."""

    structural_residuals: Mapping[str, float]
    thermal_residuals: Mapping[str, float]
    manufacturing_residuals: Mapping[str, float]
    diagnostics: Mapping[str, float | str | bool]
    feasible: bool


def evaluate_formal_constraints_v2(
    geometry: GeometryState,
    compressor_outlet: Any,
    combustor: Any,
    mass: MassBreakdown,
    config: ModelConfig,
) -> FormalConstraintEvaluation:
    """Evaluate V2.1 sections 14 and 17 after a matched cycle exists.

    The bootstrap pack currently supplies room-temperature yield strengths but
    no elevated-temperature curve. They are therefore used as explicit
    screening proxies; independent metal-temperature limits remain formal.
    """
    design = geometry.design
    compressor_metal_temperature = float(
        compressor_outlet.total_temperature_K
    )
    turbine_inlet_temperature = float(combustor.outlet.total_temperature_K)
    turbine_metal_temperature = compressor_metal_temperature + float(
        config.require("turbine_metal_temperature_fraction")
    ) * (turbine_inlet_temperature - compressor_metal_temperature)
    combustor_metal_temperature = compressor_metal_temperature + float(
        config.require("combustor_metal_temperature_fraction")
    ) * (turbine_inlet_temperature - compressor_metal_temperature)

    compressor_stress = (
        float(config.require("compressor_rotating_stress_factor"))
        * float(config.require("compressor_material_density_kg_m3"))
        * geometry.compressor_tip_speed_m_s**2
    )
    turbine_stress = (
        float(config.require("turbine_rotating_stress_factor"))
        * float(config.require("turbine_material_density_kg_m3"))
        * geometry.turbine_tip_speed_m_s**2
    )
    liner_pressure_difference = float(
        config.require("combustor_liner_pressure_factor")
    ) * float(combustor.total_pressure_loss_Pa)
    combustor_hoop_stress = (
        liner_pressure_difference
        * geometry.combustor_mean_diameter_m
        / (2.0 * design.combustor_liner_thickness_m)
    )

    lattice_strength_ratio = float(
        config.require("lattice_strength_coefficient")
    ) * design.lattice_relative_density ** float(
        config.require("lattice_strength_exponent")
    )
    lattice_modulus_ratio = float(
        config.require("lattice_modulus_coefficient")
    ) * design.lattice_relative_density ** float(
        config.require("lattice_modulus_exponent")
    )
    combustor_lattice_fraction = float(
        config.require("combustor_lattice_volume_fraction")
    )
    combustor_strength_mixture = (
        1.0
        - combustor_lattice_fraction
        + combustor_lattice_fraction * lattice_strength_ratio
    )
    compressor_yield_proxy = float(
        config.require("compressor_yield_strength_room_Pa")
    )
    hot_section_yield_proxy = float(
        config.require("hot_section_yield_strength_room_Pa")
    )
    combustor_yield_proxy = (
        hot_section_yield_proxy * combustor_strength_mixture
    )
    if min(
        compressor_yield_proxy,
        hot_section_yield_proxy,
        combustor_yield_proxy,
    ) <= 0.0:
        raise FormalConstraintError("Yield-strength proxy must be positive")

    structural = {
        "compressor_rotating_stress": float(
            config.require("compressor_safety_factor")
        )
        * compressor_stress
        / compressor_yield_proxy
        - 1.0,
        "turbine_rotating_stress": float(
            config.require("turbine_safety_factor")
        )
        * turbine_stress
        / hot_section_yield_proxy
        - 1.0,
        "combustor_liner_hoop_stress": float(
            config.require("combustor_safety_factor")
        )
        * combustor_hoop_stress
        / combustor_yield_proxy
        - 1.0,
    }
    thermal = {
        "compressor_metal_temperature": compressor_metal_temperature
        - float(config.require("compressor_material_temperature_max_K")),
        "combustor_metal_temperature": combustor_metal_temperature
        - float(config.require("combustor_material_temperature_max_K")),
        "turbine_metal_temperature": turbine_metal_temperature
        - float(config.require("turbine_material_temperature_max_K")),
        "turbine_inlet_temperature_min": float(
            config.require("turbine_inlet_temperature_min_K")
        )
        - turbine_inlet_temperature,
        "turbine_inlet_temperature_max": turbine_inlet_temperature
        - float(config.require("turbine_inlet_temperature_max_K")),
    }

    bounds = geometry19_coordinate_bounds(config)
    relative_density_min, relative_density_max = bounds[
        "lattice_relative_density"
    ]
    nozzle_wall_thickness = float(
        config.require("nozzle_thickness_ratio")
    ) * design.nozzle_throat_diameter_m
    lattice_region_minimum_length = min(
        geometry.combustor_hydraulic_diameter_m,
        mass.nozzle_length_m,
    )
    cells_across_region = (
        lattice_region_minimum_length / design.lattice_cell_size_m
    )
    manufacturing = {
        "lattice_relative_density_min": float(relative_density_min)
        - design.lattice_relative_density,
        "lattice_relative_density_max": design.lattice_relative_density
        - float(relative_density_max),
        "lattice_cell_size_min": float(
            config.require("lattice_cell_size_min_m")
        )
        - design.lattice_cell_size_m,
        "lattice_cells_across_region_min": float(
            config.require("lattice_cells_across_region_min")
        )
        - cells_across_region,
    }
    if bool(config.require("lattice_skin_retained")):
        minimum_skin = float(config.require("lattice_min_skin_thickness_m"))
        manufacturing.update(
            {
                "combustor_lattice_skin_thickness_min": minimum_skin
                - design.combustor_liner_thickness_m,
                "nozzle_lattice_skin_thickness_min": minimum_skin
                - nozzle_wall_thickness,
            }
        )

    all_residuals = {**structural, **thermal, **manufacturing}
    if not all(math.isfinite(value) for value in all_residuals.values()):
        raise FormalConstraintError(
            "Formal constraint calculation produced a non-finite residual"
        )
    diagnostics: dict[str, float | str | bool] = {
        "compressor_metal_temperature_K": compressor_metal_temperature,
        "combustor_metal_temperature_K": combustor_metal_temperature,
        "turbine_metal_temperature_K": turbine_metal_temperature,
        "compressor_rotating_stress_Pa": compressor_stress,
        "turbine_rotating_stress_Pa": turbine_stress,
        "combustor_liner_pressure_difference_Pa": liner_pressure_difference,
        "combustor_hoop_stress_Pa": combustor_hoop_stress,
        "compressor_yield_strength_proxy_Pa": compressor_yield_proxy,
        "hot_section_yield_strength_proxy_Pa": hot_section_yield_proxy,
        "combustor_effective_yield_strength_proxy_Pa": combustor_yield_proxy,
        "lattice_modulus_ratio": lattice_modulus_ratio,
        "lattice_strength_ratio": lattice_strength_ratio,
        "lattice_region_minimum_length_m": lattice_region_minimum_length,
        "lattice_cells_across_region": cells_across_region,
        "lattice_topology": str(config.require("lattice_topology")),
        "lattice_skin_retained": bool(config.require("lattice_skin_retained")),
        "lattice_flange_allowed": bool(
            config.require("lattice_flange_allowed")
        ),
        "yield_strength_temperature_model": "room_temperature_bootstrap_proxy",
    }
    return FormalConstraintEvaluation(
        structural_residuals=structural,
        thermal_residuals=thermal,
        manufacturing_residuals=manufacturing,
        diagnostics=diagnostics,
        feasible=all(value <= 0.0 for value in all_residuals.values()),
    )
