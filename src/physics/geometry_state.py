"""Analytic geometry derivation for the frozen Geometry-aware 19D design."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from functools import lru_cache
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from src.physics.model_config import ModelConfig


class GeometryStateError(ValueError):
    """A hard geometry failure with a frozen V2.1 failure code."""

    def __init__(self, failure_code: str, message: str):
        super().__init__(message)
        self.failure_code = failure_code


@dataclass(frozen=True)
class Geometry19:
    """Canonical 19D design vector in the order frozen by WP19-A."""

    compressor_impeller_exit_diameter_m: float
    compressor_inducer_tip_diameter_m: float
    compressor_inducer_hub_tip_ratio: float
    compressor_impeller_exit_width_m: float
    compressor_exit_blade_angle_deg: float
    compressor_blade_count: int
    combustor_outer_diameter_m: float
    combustor_inner_diameter_m: float
    combustor_length_m: float
    combustor_liner_thickness_m: float
    combustor_injector_count: int
    turbine_tip_diameter_m: float
    turbine_hub_tip_ratio: float
    turbine_blade_count: int
    turbine_inlet_total_temperature_K: float
    nozzle_throat_diameter_m: float
    shaft_speed_rpm: float
    lattice_cell_size_m: float
    lattice_relative_density: float

    @classmethod
    def coordinate_names(cls) -> tuple[str, ...]:
        return tuple(item.name for item in fields(cls))

    @classmethod
    def from_sequence(cls, values: Sequence[float | int]) -> "Geometry19":
        if len(values) != 19:
            raise GeometryStateError(
                "GEOMETRY_INVALID",
                f"Geometry19 requires 19 coordinates, received {len(values)}",
            )
        return cls(*values)

    @classmethod
    def from_mapping(cls, values: Mapping[str, float | int]) -> "Geometry19":
        expected = set(cls.coordinate_names())
        supplied = set(values)
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        if missing or extra:
            raise GeometryStateError(
                "GEOMETRY_INVALID",
                f"Geometry19 mapping mismatch; missing={missing}, extra={extra}",
            )
        return cls(*(values[name] for name in cls.coordinate_names()))

    def as_mapping(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class GeometryState:
    """All analytic geometry quantities required by V2.1 component models."""

    design: Geometry19
    omega_rad_s: float

    compressor_inducer_hub_diameter_m: float
    compressor_inlet_area_m2: float
    compressor_inlet_effective_area_m2: float
    compressor_exit_radius_m: float
    compressor_exit_geometric_area_m2: float
    compressor_exit_blade_thickness_m: float
    compressor_exit_blockage: float
    compressor_exit_effective_area_m2: float
    compressor_tip_speed_m_s: float

    diffuser_outlet_area_m2: float
    diffuser_equivalent_length_m: float

    combustor_annulus_area_m2: float
    combustor_volume_m3: float
    combustor_hydraulic_diameter_m: float
    combustor_mean_diameter_m: float
    combustor_injector_pitch_m: float

    turbine_tip_radius_m: float
    turbine_hub_radius_m: float
    turbine_mean_radius_m: float
    turbine_blade_height_m: float
    turbine_annulus_area_m2: float
    turbine_mean_speed_m_s: float
    turbine_tip_speed_m_s: float
    turbine_chord_m: float
    turbine_pitch_m: float
    turbine_solidity: float

    nozzle_throat_area_m2: float
    nozzle_exit_area_m2: float
    nozzle_inlet_diameter_m: float

    shaft_radius_m: float
    shaft_diameter_m: float

    assembly_residuals: Mapping[str, float]


@lru_cache(maxsize=None)
def _design_variables(schema_version: str) -> tuple[Mapping[str, object], ...]:
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "physics"
        / f"{schema_version}.yaml"
    )
    try:
        payload = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise GeometryStateError(
            "GEOMETRY_INVALID", f"Cannot load frozen design schema: {schema_path}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != schema_version:
        raise GeometryStateError(
            "GEOMETRY_INVALID", "Frozen design schema version is invalid"
        )
    raw_variables = payload.get("design_variables")
    if not isinstance(raw_variables, list):
        raise GeometryStateError(
            "GEOMETRY_INVALID", "Frozen design schema has no design_variables"
        )
    variables = tuple(raw_variables)
    names = tuple(item.get("name") for item in variables if isinstance(item, dict))
    indices = tuple(item.get("index") for item in variables if isinstance(item, dict))
    if names != Geometry19.coordinate_names() or indices != tuple(range(1, 20)):
        raise GeometryStateError(
            "GEOMETRY_INVALID", "Frozen design schema coordinate order is inconsistent"
        )
    return variables


def _validate_design(design: Geometry19, config: ModelConfig) -> None:
    values = design.as_mapping()
    for variable in _design_variables(config.design_schema_version):
        name = str(variable["name"])
        value = values[name]
        dtype = variable.get("dtype")
        if dtype == "int64":
            # Python and NumPy integer scalars are accepted; booleans and
            # integer-valued floats are deliberately rejected.
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise GeometryStateError(
                    "GEOMETRY_INVALID", f"{name} must have exact integer type"
                )
        elif dtype == "float64":
            if isinstance(value, bool) or not isinstance(value, Real):
                raise GeometryStateError(
                    "GEOMETRY_INVALID", f"{name} must be a real number"
                )
        else:
            raise GeometryStateError(
                "GEOMETRY_INVALID", f"{name} has unsupported schema dtype {dtype!r}"
            )
        if not math.isfinite(float(value)):
            raise GeometryStateError("NONFINITE_STATE", f"{name} must be finite")
        bounds = variable.get("bounds")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise GeometryStateError(
                "GEOMETRY_INVALID", f"{name} has invalid frozen bounds"
            )
        lower, upper = bounds
        if not lower <= value <= upper:
            raise GeometryStateError(
                "GEOMETRY_INVALID",
                f"{name}={value!r} is outside frozen bounds [{lower}, {upper}]",
            )

    positive_names = (
        "compressor_impeller_exit_diameter_m",
        "compressor_inducer_tip_diameter_m",
        "compressor_impeller_exit_width_m",
        "combustor_outer_diameter_m",
        "combustor_inner_diameter_m",
        "combustor_length_m",
        "combustor_liner_thickness_m",
        "turbine_tip_diameter_m",
        "turbine_inlet_total_temperature_K",
        "nozzle_throat_diameter_m",
        "shaft_speed_rpm",
        "lattice_cell_size_m",
        "lattice_relative_density",
    )
    for name in positive_names:
        if getattr(design, name) <= 0.0:
            raise GeometryStateError(
                "GEOMETRY_INVALID", f"{name} must be positive"
            )

    if not 0.0 < design.compressor_inducer_hub_tip_ratio < 1.0:
        raise GeometryStateError(
            "GEOMETRY_INVALID", "compressor inducer hub-tip ratio must be in (0, 1)"
        )
    if not 0.0 < design.turbine_hub_tip_ratio < 1.0:
        raise GeometryStateError(
            "GEOMETRY_INVALID", "turbine hub-tip ratio must be in (0, 1)"
        )
    if not 0.0 < design.compressor_exit_blade_angle_deg <= 90.0:
        raise GeometryStateError(
            "GEOMETRY_INVALID", "compressor exit blade angle must be in (0, 90]"
        )
    if (
        design.compressor_inducer_tip_diameter_m
        >= design.compressor_impeller_exit_diameter_m
    ):
        raise GeometryStateError(
            "GEOMETRY_INVALID", "compressor inducer tip must be inside the exit"
        )
    if (
        design.combustor_outer_diameter_m
        <= design.combustor_inner_diameter_m
        + 2.0 * design.combustor_liner_thickness_m
    ):
        raise GeometryStateError(
            "GEOMETRY_INVALID", "combustor annulus has no positive liner clearance"
        )


def geometry19_coordinate_bounds(
    config: ModelConfig,
) -> Mapping[str, tuple[float | int, float | int]]:
    """Return the executable frozen bounds keyed by canonical coordinate."""
    return {
        str(variable["name"]): (
            variable["bounds"][0],
            variable["bounds"][1],
        )
        for variable in _design_variables(config.design_schema_version)
    }


def derive_geometry_state(design: Geometry19, config: ModelConfig) -> GeometryState:
    """Derive the complete analytic GeometryState from a valid 19D design."""
    _validate_design(design, config)

    pi = math.pi
    omega = 2.0 * pi * design.shaft_speed_rpm / 60.0

    d_c1h = (
        design.compressor_inducer_hub_tip_ratio
        * design.compressor_inducer_tip_diameter_m
    )
    area_c1 = pi / 4.0 * (
        design.compressor_inducer_tip_diameter_m**2 - d_c1h**2
    )
    area_c1_effective = area_c1 * float(
        config.require("compressor_inlet_blockage_factor")
    )
    r_c2 = design.compressor_impeller_exit_diameter_m / 2.0
    area_c2_geometric = (
        2.0 * pi * r_c2 * design.compressor_impeller_exit_width_m
    )
    blade_thickness = float(
        config.require("compressor_blade_thickness_ratio")
    ) * design.compressor_impeller_exit_width_m
    beta_rad = math.radians(design.compressor_exit_blade_angle_deg)
    blockage = 1.0 - (
        int(design.compressor_blade_count) * blade_thickness
    ) / (2.0 * pi * r_c2 * math.sin(beta_rad))
    if not 0.0 < blockage < 1.0:
        raise GeometryStateError(
            "COMPRESSOR_BLOCKAGE_INVALID",
            f"compressor exit blockage must be in (0, 1), received {blockage}",
        )
    area_c2 = blockage * area_c2_geometric
    compressor_tip_speed = omega * r_c2

    diffuser_area = float(config.require("diffuser_area_ratio")) * area_c2
    diffuser_length = (
        float(config.require("diffuser_length_ratio"))
        * design.compressor_impeller_exit_diameter_m
    )

    area_b = pi / 4.0 * (
        design.combustor_outer_diameter_m**2
        - design.combustor_inner_diameter_m**2
    )
    volume_b = area_b * design.combustor_length_m
    hydraulic_diameter_b = (
        design.combustor_outer_diameter_m
        - design.combustor_inner_diameter_m
    )
    mean_diameter_b = 0.5 * (
        design.combustor_outer_diameter_m
        + design.combustor_inner_diameter_m
    )
    injector_pitch = (
        pi * mean_diameter_b / int(design.combustor_injector_count)
    )

    r_t = design.turbine_tip_diameter_m / 2.0
    r_h = design.turbine_hub_tip_ratio * r_t
    r_m = 0.5 * (r_t + r_h)
    h_t = r_t - r_h
    area_t = pi * (r_t**2 - r_h**2)
    u_m = omega * r_m
    u_t_tip = omega * r_t
    chord_t = h_t / float(config.require("turbine_aspect_ratio"))
    pitch_t = 2.0 * pi * r_m / int(design.turbine_blade_count)
    solidity_t = chord_t / pitch_t

    area_star = pi * design.nozzle_throat_diameter_m**2 / 4.0
    nozzle_inlet_diameter = (
        float(config.require("nozzle_inlet_ratio"))
        * design.nozzle_throat_diameter_m
    )
    shaft_radius = float(config.require("shaft_radius_ratio")) * r_h
    shaft_diameter = 2.0 * shaft_radius

    assembly_residuals = {
        "diffuser_to_combustor_area": abs(
            diffuser_area / area_b
            - float(config.require("diffuser_to_combustor_target_area_ratio"))
        )
        - float(config.require("diffuser_to_combustor_area_tolerance")),
        "combustor_to_turbine_outer_diameter": abs(
            design.combustor_outer_diameter_m - design.turbine_tip_diameter_m
        )
        / design.turbine_tip_diameter_m
        - float(config.require("combustor_to_turbine_diameter_tolerance")),
        "shaft_clearance": shaft_diameter
        + 2.0 * float(config.require("shaft_clearance_m"))
        - design.combustor_inner_diameter_m,
        "turbine_to_nozzle": abs(
            nozzle_inlet_diameter - design.turbine_tip_diameter_m
        )
        / design.turbine_tip_diameter_m
        - float(config.require("turbine_to_nozzle_diameter_tolerance")),
    }

    derived_values = (
        omega,
        area_c1,
        area_c1_effective,
        area_c2,
        compressor_tip_speed,
        diffuser_area,
        area_b,
        volume_b,
        area_t,
        u_m,
        chord_t,
        solidity_t,
        area_star,
        shaft_radius,
        *assembly_residuals.values(),
    )
    if not all(math.isfinite(value) for value in derived_values):
        raise GeometryStateError(
            "NONFINITE_STATE", "Geometry derivation produced a non-finite value"
        )

    return GeometryState(
        design=design,
        omega_rad_s=omega,
        compressor_inducer_hub_diameter_m=d_c1h,
        compressor_inlet_area_m2=area_c1,
        compressor_inlet_effective_area_m2=area_c1_effective,
        compressor_exit_radius_m=r_c2,
        compressor_exit_geometric_area_m2=area_c2_geometric,
        compressor_exit_blade_thickness_m=blade_thickness,
        compressor_exit_blockage=blockage,
        compressor_exit_effective_area_m2=area_c2,
        compressor_tip_speed_m_s=compressor_tip_speed,
        diffuser_outlet_area_m2=diffuser_area,
        diffuser_equivalent_length_m=diffuser_length,
        combustor_annulus_area_m2=area_b,
        combustor_volume_m3=volume_b,
        combustor_hydraulic_diameter_m=hydraulic_diameter_b,
        combustor_mean_diameter_m=mean_diameter_b,
        combustor_injector_pitch_m=injector_pitch,
        turbine_tip_radius_m=r_t,
        turbine_hub_radius_m=r_h,
        turbine_mean_radius_m=r_m,
        turbine_blade_height_m=h_t,
        turbine_annulus_area_m2=area_t,
        turbine_mean_speed_m_s=u_m,
        turbine_tip_speed_m_s=u_t_tip,
        turbine_chord_m=chord_t,
        turbine_pitch_m=pitch_t,
        turbine_solidity=solidity_t,
        nozzle_throat_area_m2=area_star,
        nozzle_exit_area_m2=area_star,
        nozzle_inlet_diameter_m=nozzle_inlet_diameter,
        shaft_radius_m=shaft_radius,
        shaft_diameter_m=shaft_diameter,
        assembly_residuals=assembly_residuals,
    )
