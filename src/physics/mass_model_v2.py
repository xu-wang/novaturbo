"""V2.1 analytic mass closure for the Geometry19 engine model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from src.physics.geometry_state import GeometryState
from src.physics.model_config import ModelConfig


class MassModelError(ValueError):
    """Hard failure raised when the analytic mass closure is invalid."""

    failure_code = "MASS_MODEL_INVALID"


@dataclass(frozen=True)
class MassBreakdown:
    """Component and subcomponent masses produced by V2.1 section 13."""

    component_masses_kg: Mapping[str, float]
    component_volumes_m3: Mapping[str, float]
    total_mass_kg: float
    compressor_length_m: float
    turbine_length_m: float
    nozzle_length_m: float
    nozzle_slant_length_m: float
    shaft_length_m: float
    combustor_effective_density_kg_m3: float
    nozzle_effective_density_kg_m3: float


def _require_finite_nonnegative(values: Mapping[str, float], label: str) -> None:
    invalid = {
        name: value
        for name, value in values.items()
        if not math.isfinite(value) or value < 0.0
    }
    if invalid:
        raise MassModelError(f"{label} contains invalid values: {invalid}")


def solve_mass_model_v2(
    geometry: GeometryState, config: ModelConfig
) -> MassBreakdown:
    """Evaluate every mass term frozen by V2.1 section 13."""
    design = geometry.design
    pi = math.pi

    compressor_inlet_mean_radius = 0.25 * (
        design.compressor_inducer_tip_diameter_m
        + geometry.compressor_inducer_hub_diameter_m
    )
    compressor_hub_thickness = float(
        config.require("compressor_hub_thickness_ratio")
    ) * design.compressor_impeller_exit_width_m
    compressor_hub_volume = (
        float(config.require("compressor_hub_fill_factor"))
        * pi
        * (
            geometry.compressor_exit_radius_m**2
            - compressor_inlet_mean_radius**2
        )
        * compressor_hub_thickness
    )
    compressor_blade_length = float(
        config.require("compressor_blade_length_factor")
    ) * math.sqrt(
        (
            geometry.compressor_exit_radius_m
            - compressor_inlet_mean_radius
        )
        ** 2
        + 0.25
        * (
            design.compressor_inducer_tip_diameter_m
            - geometry.compressor_inducer_hub_diameter_m
        )
        ** 2
    )
    compressor_blade_height = float(
        config.require("compressor_blade_height_factor")
    ) * (
        design.compressor_impeller_exit_width_m
        + 0.5
        * (
            design.compressor_inducer_tip_diameter_m
            - geometry.compressor_inducer_hub_diameter_m
        )
    )
    compressor_blade_thickness = float(
        config.require("compressor_mass_blade_thickness_ratio")
    ) * design.compressor_impeller_exit_width_m
    compressor_blade_volume = (
        int(design.compressor_blade_count)
        * compressor_blade_length
        * compressor_blade_height
        * compressor_blade_thickness
    )
    compressor_length = float(
        config.require("compressor_length_ratio")
    ) * design.compressor_impeller_exit_diameter_m
    compressor_casing_volume = (
        float(config.require("compressor_casing_factor"))
        * pi
        * design.compressor_impeller_exit_diameter_m
        * compressor_length
        * float(config.require("compressor_casing_thickness_m"))
    )
    compressor_volume = (
        compressor_hub_volume
        + compressor_blade_volume
        + compressor_casing_volume
    )
    compressor_mass = float(
        config.require("compressor_material_density_kg_m3")
    ) * compressor_volume

    combustor_cylinder_volume = (
        pi
        * (
            design.combustor_outer_diameter_m
            + design.combustor_inner_diameter_m
        )
        * design.combustor_length_m
        * design.combustor_liner_thickness_m
    )
    combustor_shell_volume = float(
        config.require("combustor_dome_factor")
    ) * combustor_cylinder_volume
    combustor_lattice_fraction = float(
        config.require("combustor_lattice_volume_fraction")
    )
    combustor_material_density = float(
        config.require("combustor_material_density_kg_m3")
    )
    combustor_effective_density = combustor_material_density * (
        1.0
        - combustor_lattice_fraction
        + combustor_lattice_fraction * design.lattice_relative_density
    )
    combustor_mass = combustor_effective_density * combustor_shell_volume
    injector_mass = int(design.combustor_injector_count) * float(
        config.require("injector_unit_mass_kg")
    )

    turbine_disk_thickness = float(
        config.require("turbine_disk_thickness_ratio")
    ) * design.turbine_tip_diameter_m
    turbine_disk_volume = (
        float(config.require("turbine_disk_fill_factor"))
        * pi
        * geometry.turbine_hub_radius_m**2
        * turbine_disk_thickness
    )
    turbine_blade_thickness = float(
        config.require("turbine_blade_thickness_ratio")
    ) * geometry.turbine_chord_m
    turbine_blade_volume = (
        int(design.turbine_blade_count)
        * geometry.turbine_chord_m
        * geometry.turbine_blade_height_m
        * turbine_blade_thickness
    )
    turbine_ngv_volume = float(
        config.require("turbine_ngv_volume_factor")
    ) * turbine_blade_volume
    turbine_length = float(
        config.require("turbine_length_ratio")
    ) * geometry.turbine_blade_height_m
    turbine_casing_volume = (
        float(config.require("turbine_casing_factor"))
        * pi
        * design.turbine_tip_diameter_m
        * turbine_length
        * float(config.require("turbine_casing_thickness_m"))
    )
    turbine_volume = (
        turbine_disk_volume
        + turbine_blade_volume
        + turbine_ngv_volume
        + turbine_casing_volume
    )
    turbine_mass = float(
        config.require("turbine_material_density_kg_m3")
    ) * turbine_volume

    nozzle_length = float(
        config.require("nozzle_length_ratio")
    ) * design.nozzle_throat_diameter_m
    nozzle_inlet_radius = 0.5 * geometry.nozzle_inlet_diameter_m
    nozzle_throat_radius = 0.5 * design.nozzle_throat_diameter_m
    nozzle_slant_length = math.sqrt(
        nozzle_length**2
        + (nozzle_inlet_radius - nozzle_throat_radius) ** 2
    )
    nozzle_shell_area = (
        pi
        * (nozzle_inlet_radius + nozzle_throat_radius)
        * nozzle_slant_length
    )
    nozzle_shell_thickness = float(
        config.require("nozzle_thickness_ratio")
    ) * design.nozzle_throat_diameter_m
    nozzle_shell_volume = nozzle_shell_area * nozzle_shell_thickness
    nozzle_lattice_fraction = float(
        config.require("nozzle_lattice_volume_fraction")
    )
    nozzle_material_density = float(
        config.require("nozzle_material_density_kg_m3")
    )
    nozzle_effective_density = nozzle_material_density * (
        1.0
        - nozzle_lattice_fraction
        + nozzle_lattice_fraction * design.lattice_relative_density
    )
    nozzle_mass = nozzle_effective_density * nozzle_shell_volume

    shaft_length = float(config.require("shaft_length_factor")) * (
        compressor_length + design.combustor_length_m + turbine_length
    )
    shaft_volume = pi * geometry.shaft_radius_m**2 * shaft_length
    shaft_mass = float(
        config.require("shaft_material_density_kg_m3")
    ) * shaft_volume
    bearing_accessory_mass = float(
        config.require("fixed_accessory_mass_kg")
    ) + float(config.require("bearing_accessory_mass_factor")) * (
        compressor_mass + turbine_mass + shaft_mass
    )

    volumes = {
        "compressor_hub": compressor_hub_volume,
        "compressor_blades": compressor_blade_volume,
        "compressor_casing": compressor_casing_volume,
        "compressor_total": compressor_volume,
        "combustor_shell": combustor_shell_volume,
        "turbine_disk": turbine_disk_volume,
        "turbine_blades": turbine_blade_volume,
        "turbine_ngv": turbine_ngv_volume,
        "turbine_casing": turbine_casing_volume,
        "turbine_total": turbine_volume,
        "nozzle_shell": nozzle_shell_volume,
        "shaft": shaft_volume,
    }
    masses = {
        "compressor": compressor_mass,
        "combustor": combustor_mass,
        "turbine": turbine_mass,
        "nozzle": nozzle_mass,
        "shaft": shaft_mass,
        "injectors": injector_mass,
        "bearing_accessory": bearing_accessory_mass,
    }
    _require_finite_nonnegative(volumes, "Mass-model volumes")
    _require_finite_nonnegative(masses, "Mass-model component masses")
    total_mass = sum(masses.values())
    lengths = {
        "compressor_length": compressor_length,
        "turbine_length": turbine_length,
        "nozzle_length": nozzle_length,
        "nozzle_slant_length": nozzle_slant_length,
        "shaft_length": shaft_length,
    }
    _require_finite_nonnegative(lengths, "Mass-model lengths")
    if total_mass <= 0.0:
        raise MassModelError("Total engine mass must be positive")

    return MassBreakdown(
        component_masses_kg=masses,
        component_volumes_m3=volumes,
        total_mass_kg=total_mass,
        compressor_length_m=compressor_length,
        turbine_length_m=turbine_length,
        nozzle_length_m=nozzle_length,
        nozzle_slant_length_m=nozzle_slant_length,
        shaft_length_m=shaft_length,
        combustor_effective_density_kg_m3=combustor_effective_density,
        nozzle_effective_density_kg_m3=nozzle_effective_density,
    )
