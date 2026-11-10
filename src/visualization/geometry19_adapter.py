"""Unique Geometry19 -> GeometryState -> visualization assembly adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.geometry.assembly import EngineAssemblyParams
from src.geometry.combustor import CombustorParams
from src.geometry.compressor import CompressorParams
from src.geometry.inlet import InletParams
from src.geometry.nozzle import NozzleParams
from src.geometry.turbine import TurbineParams
from src.physics.geometry_state import Geometry19, GeometryState, derive_geometry_state
from src.physics.mass_model_v2 import solve_mass_model_v2
from src.physics.model_config import ModelConfig


class VisualizationCalibrationError(ValueError):
    """Raised when the visualization calibration contract is incomplete."""


class VisualizationGeometryError(ValueError):
    """Raised when a valid physics geometry cannot produce required meshes."""


_COMPONENT_TYPES = {
    "inlet": InletParams,
    "compressor": CompressorParams,
    "combustor": CombustorParams,
    "turbine": TurbineParams,
    "nozzle": NozzleParams,
}

_ASSEMBLY_FIELDS = {
    "name",
    "inlet_compressor_gap_mm",
    "compressor_combustor_gap_mm",
    "combustor_turbine_gap_mm",
    "turbine_nozzle_gap_mm",
    "casing_thickness_mm",
    "casing_clearance_mm",
    "shaft_diameter_mm",
    "shaft_material",
}

_DERIVATION_FIELDS = {
    "inlet_length_to_impeller_diameter",
    "inlet_lip_radius_to_impeller_diameter",
    "compressor_inlet_duct_to_impeller_diameter",
    "compressor_splitter_count_to_main_count",
    "combustor_dome_recess_to_length",
    "combustor_primary_hole_count_to_injector_count",
    "combustor_dilution_hole_count_to_injector_count",
    "turbine_ngv_chord_to_rotor_chord",
    "turbine_blade_axial_position_fraction",
    "nozzle_lip_thickness_to_throat_diameter",
    "nozzle_lip_radius_to_throat_diameter",
    "nozzle_flange_outer_to_inlet_diameter",
    "nozzle_flange_thickness_to_throat_diameter",
    "nozzle_bolt_circle_to_flange_outer_diameter",
    "nozzle_divergent_exit_to_throat_diameter",
    "nozzle_divergent_length_to_throat_diameter",
}


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_finite_tree(value: Any, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise VisualizationCalibrationError(f"{path} must be finite")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_tree(item, f"{path}.{key}")
        return
    raise VisualizationCalibrationError(f"{path} has unsupported value {value!r}")


@dataclass(frozen=True)
class VisualizationCalibration:
    version: str
    description: str
    components: Mapping[str, Mapping[str, Any]]
    assembly: Mapping[str, Any]
    derivation: Mapping[str, float]
    lattice: Mapping[str, Any]
    source_hash: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VisualizationCalibration":
        source_path = Path(path)
        try:
            source_bytes = source_path.read_bytes()
            payload = yaml.safe_load(source_bytes)
        except (OSError, yaml.YAMLError) as exc:
            raise VisualizationCalibrationError(
                f"Cannot load visualization calibration: {source_path}"
            ) from exc
        expected_top = {
            "calibration_version",
            "description",
            "components",
            "assembly",
            "derivation",
            "lattice",
        }
        if not isinstance(payload, dict) or set(payload) != expected_top:
            raise VisualizationCalibrationError(
                "Visualization calibration top-level contract mismatch"
            )
        version = payload["calibration_version"]
        if not isinstance(version, str) or source_path.stem != version:
            raise VisualizationCalibrationError(
                "Calibration filename must match calibration_version"
            )
        description = payload["description"]
        if not isinstance(description, str) or not description.strip():
            raise VisualizationCalibrationError("description must be non-empty")

        components = payload["components"]
        if not isinstance(components, dict) or set(components) != set(_COMPONENT_TYPES):
            raise VisualizationCalibrationError("Component calibration keys mismatch")
        for name, component_type in _COMPONENT_TYPES.items():
            values = components[name]
            expected = {item.name for item in fields(component_type)}
            if not isinstance(values, dict) or set(values) != expected:
                raise VisualizationCalibrationError(
                    f"{name} fields mismatch; missing={sorted(expected - set(values or {}))}, "
                    f"extra={sorted(set(values or {}) - expected)}"
                )

        assembly = payload["assembly"]
        if not isinstance(assembly, dict) or set(assembly) != _ASSEMBLY_FIELDS:
            raise VisualizationCalibrationError("Assembly calibration fields mismatch")
        derivation = payload["derivation"]
        if not isinstance(derivation, dict) or set(derivation) != _DERIVATION_FIELDS:
            raise VisualizationCalibrationError("Derivation calibration fields mismatch")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in derivation.values()
        ):
            raise VisualizationCalibrationError(
                "Every derivation coefficient must be finite and positive"
            )
        lattice = payload["lattice"]
        if not isinstance(lattice, dict) or set(lattice) != {
            "topology",
            "skin_retained",
            "combustor_inner_skin_mm",
            "combustor_outer_skin_mm",
            "relative_density_to_wall_threshold",
            "wall_threshold_min",
            "wall_threshold_max",
        }:
            raise VisualizationCalibrationError("Lattice calibration fields mismatch")
        for name in (
            "combustor_inner_skin_mm",
            "combustor_outer_skin_mm",
            "relative_density_to_wall_threshold",
            "wall_threshold_min",
            "wall_threshold_max",
        ):
            value = lattice[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise VisualizationCalibrationError(
                    f"lattice.{name} must be finite and positive"
                )
        if float(lattice["wall_threshold_min"]) >= float(
            lattice["wall_threshold_max"]
        ):
            raise VisualizationCalibrationError(
                "lattice wall-threshold bounds must satisfy min < max"
            )
        _validate_finite_tree(payload, "visualization_calibration")
        return cls(
            version=version,
            description=description,
            components={name: dict(values) for name, values in components.items()},
            assembly=dict(assembly),
            derivation={name: float(value) for name, value in derivation.items()},
            lattice=dict(lattice),
            source_hash=hashlib.sha256(source_bytes).hexdigest(),
        )


@dataclass(frozen=True)
class VisualizationAssembly:
    """Traceable visualization input separated into solid/lattice/overlay paths."""

    geometry_state: GeometryState
    assembly_params: EngineAssemblyParams
    lattice_parameters: Mapping[str, float | str | bool]
    overlay_parameters: Mapping[str, float]
    solid_geometry_hash: str
    lattice_hash: str
    design_hash: str
    calibration_hash: str
    visualization_calibration_version: str


def _default_calibration_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "config"
        / "physics"
        / "geometry19_visualization_calibration_v1.yaml"
    )


def _solid_mapping(params: EngineAssemblyParams) -> dict[str, Any]:
    payload = asdict(params)
    payload["compressor"].pop("rpm")
    payload["compressor"].pop("pressure_ratio")
    payload["compressor"].pop("mass_flow_kg_s")
    payload["combustor"].pop("inlet_temperature_K")
    payload["combustor"].pop("outlet_temperature_K")
    payload["combustor"].pop("air_fuel_ratio")
    payload["combustor"].pop("pressure_Pa")
    payload["turbine"].pop("rpm")
    payload["turbine"].pop("inlet_temperature_K")
    payload["inlet"].pop("pressure_recovery")
    payload["nozzle"].pop("discharge_coefficient")
    return payload


def geometry19_to_assembly_params(
    design: Geometry19,
    model_config: ModelConfig,
    visualization_calibration: VisualizationCalibration | None = None,
) -> VisualizationAssembly:
    """Build the only supported Geometry19 visualization assembly contract."""
    if not isinstance(design, Geometry19):
        raise TypeError("design must be a validated Geometry19")
    if not isinstance(model_config, ModelConfig):
        raise TypeError("model_config must be a validated ModelConfig")
    calibration = visualization_calibration or VisualizationCalibration.from_yaml(
        _default_calibration_path()
    )
    if not isinstance(calibration, VisualizationCalibration):
        raise TypeError("visualization_calibration has invalid type")

    geometry = derive_geometry_state(design, model_config)
    mass = solve_mass_model_v2(geometry, model_config)
    mm = 1000.0
    derive = calibration.derivation

    inlet_values = dict(calibration.components["inlet"])
    inlet_values.update(
        inlet_diameter_mm=design.compressor_impeller_exit_diameter_m * mm,
        outlet_diameter_mm=design.compressor_inducer_tip_diameter_m * mm,
        length_mm=(
            derive["inlet_length_to_impeller_diameter"]
            * design.compressor_impeller_exit_diameter_m
            * mm
        ),
        lip_radius_mm=(
            derive["inlet_lip_radius_to_impeller_diameter"]
            * design.compressor_impeller_exit_diameter_m
            * mm
        ),
        wall_thickness_mm=float(
            model_config.require("compressor_casing_thickness_m")
        )
        * mm,
        pressure_recovery=float(model_config.require("inlet_total_pressure_recovery")),
    )

    compressor_values = dict(calibration.components["compressor"])
    compressor_values.update(
        impeller_tip_diameter_mm=design.compressor_impeller_exit_diameter_m * mm,
        impeller_hub_diameter_mm=geometry.compressor_inducer_hub_diameter_m * mm,
        inducer_tip_diameter_mm=design.compressor_inducer_tip_diameter_m * mm,
        blade_count=int(design.compressor_blade_count),
        splitter_blade_count=round(
            derive["compressor_splitter_count_to_main_count"]
            * int(design.compressor_blade_count)
        ),
        exducer_blade_angle_deg=design.compressor_exit_blade_angle_deg,
        blade_thickness_mm=geometry.compressor_exit_blade_thickness_m * mm,
        axial_length_mm=mass.compressor_length_m * mm,
        diffuser_radius_ratio=math.sqrt(float(model_config.require("diffuser_area_ratio"))),
        diffuser_width_mm=design.compressor_impeller_exit_width_m * mm,
        inlet_duct_length_mm=(
            derive["compressor_inlet_duct_to_impeller_diameter"]
            * design.compressor_impeller_exit_diameter_m
            * mm
        ),
        rpm=design.shaft_speed_rpm,
    )

    combustor_values = dict(calibration.components["combustor"])
    combustor_values.update(
        outer_diameter_mm=design.combustor_outer_diameter_m * mm,
        inner_diameter_mm=design.combustor_inner_diameter_m * mm,
        length_mm=design.combustor_length_m * mm,
        liner_thickness_mm=design.combustor_liner_thickness_m * mm,
        casing_thickness_mm=float(
            model_config.require("turbine_casing_thickness_m")
        )
        * mm,
        dome_recess_mm=(
            derive["combustor_dome_recess_to_length"]
            * design.combustor_length_m
            * mm
        ),
        num_fuel_injectors=int(design.combustor_injector_count),
        num_primary_holes=round(
            derive["combustor_primary_hole_count_to_injector_count"]
            * int(design.combustor_injector_count)
        ),
        num_dilution_holes=round(
            derive["combustor_dilution_hole_count_to_injector_count"]
            * int(design.combustor_injector_count)
        ),
        primary_zone_fraction=float(
            model_config.require("combustor_primary_volume_fraction")
        ),
        outlet_temperature_K=design.turbine_inlet_total_temperature_K,
    )

    turbine_values = dict(calibration.components["turbine"])
    turbine_values.update(
        tip_diameter_mm=design.turbine_tip_diameter_m * mm,
        hub_diameter_mm=2.0 * geometry.turbine_hub_radius_m * mm,
        blade_height_mm=geometry.turbine_blade_height_m * mm,
        axial_chord_mm=geometry.turbine_chord_m * mm,
        stage_axial_length_mm=mass.turbine_length_m * mm,
        ngv_chord_mm=(
            derive["turbine_ngv_chord_to_rotor_chord"]
            * geometry.turbine_chord_m
            * mm
        ),
        blade_count=int(design.turbine_blade_count),
        blade_chord_mm=geometry.turbine_chord_m * mm,
        blade_thickness_ratio=float(
            model_config.require("turbine_blade_thickness_ratio")
        ),
        blade_axial_position_mm=(
            derive["turbine_blade_axial_position_fraction"]
            * mass.turbine_length_m
            * mm
        ),
        hub_tip_ratio=design.turbine_hub_tip_ratio,
        rpm=design.shaft_speed_rpm,
        inlet_temperature_K=design.turbine_inlet_total_temperature_K,
    )

    throat_m = design.nozzle_throat_diameter_m
    nozzle_inlet_m = geometry.nozzle_inlet_diameter_m
    flange_outer_m = (
        derive["nozzle_flange_outer_to_inlet_diameter"] * nozzle_inlet_m
    )
    nozzle_values = dict(calibration.components["nozzle"])
    nozzle_values.update(
        inlet_diameter_mm=nozzle_inlet_m * mm,
        # The legacy convergent nozzle calls its terminal throat "exit".
        # This assignment is deliberately local and must not leak into schema/UI names.
        exit_diameter_mm=throat_m * mm,
        length_mm=mass.nozzle_length_m * mm,
        wall_thickness_mm=float(model_config.require("nozzle_thickness_ratio"))
        * throat_m
        * mm,
        divergent_exit_diameter_mm=(
            derive["nozzle_divergent_exit_to_throat_diameter"] * throat_m * mm
        ),
        divergent_length_mm=(
            derive["nozzle_divergent_length_to_throat_diameter"] * throat_m * mm
        ),
        lip_thickness_mm=(
            derive["nozzle_lip_thickness_to_throat_diameter"] * throat_m * mm
        ),
        lip_radius_mm=(
            derive["nozzle_lip_radius_to_throat_diameter"] * throat_m * mm
        ),
        flange_outer_diameter_mm=flange_outer_m * mm,
        flange_thickness_mm=(
            derive["nozzle_flange_thickness_to_throat_diameter"] * throat_m * mm
        ),
        bolt_circle_diameter_mm=(
            derive["nozzle_bolt_circle_to_flange_outer_diameter"]
            * flange_outer_m
            * mm
        ),
        discharge_coefficient=float(
            model_config.require("nozzle_discharge_coefficient")
        ),
    )

    inlet = InletParams(**inlet_values)
    compressor = CompressorParams(**compressor_values)
    combustor = CombustorParams(**combustor_values)
    turbine = TurbineParams(**turbine_values)
    nozzle = NozzleParams(**nozzle_values)
    assembly_values = dict(calibration.assembly)
    assembly_values.update(
        casing_thickness_mm=max(
            float(model_config.require("compressor_casing_thickness_m")),
            float(model_config.require("turbine_casing_thickness_m")),
        )
        * mm,
        casing_clearance_mm=float(model_config.require("shaft_clearance_m")) * mm,
        shaft_diameter_mm=geometry.shaft_diameter_m * mm,
    )
    params = EngineAssemblyParams(
        inlet=inlet,
        compressor=compressor,
        combustor=combustor,
        turbine=turbine,
        nozzle=nozzle,
        **assembly_values,
    )

    combustor_lattice_clearance_mm = (
        0.5 * (combustor.outer_diameter_mm - combustor.inner_diameter_mm)
        - combustor.liner_thickness_mm
        - combustor.casing_thickness_mm
        - float(calibration.lattice["combustor_inner_skin_mm"])
        - float(calibration.lattice["combustor_outer_skin_mm"])
    )
    if combustor_lattice_clearance_mm <= 0.0:
        raise VisualizationGeometryError(
            "combustor lattice has no positive radial infill clearance after "
            "liner, casing, and retained skins"
        )

    lattice_parameters: dict[str, float | str | bool] = {
        **calibration.lattice,
        "cell_size_m": design.lattice_cell_size_m,
        "relative_density": design.lattice_relative_density,
    }
    overlay_parameters = {
        "turbine_inlet_total_temperature_K": (
            design.turbine_inlet_total_temperature_K
        ),
        "shaft_speed_rpm": design.shaft_speed_rpm,
    }
    combined_calibration_hash = _canonical_hash(
        {
            "physics": model_config.calibration_pack_hash,
            "visualization": calibration.source_hash,
        }
    )
    solid_geometry_hash = _canonical_hash(_solid_mapping(params))
    lattice_hash = _canonical_hash(
        {
            "solid_geometry_hash": solid_geometry_hash,
            "lattice": lattice_parameters,
        }
    )
    design_hash = _canonical_hash(
        {
            "geometry19": design.as_mapping(),
            "physics_calibration": model_config.calibration_pack_hash,
            "visualization_calibration": calibration.source_hash,
        }
    )
    return VisualizationAssembly(
        geometry_state=geometry,
        assembly_params=params,
        lattice_parameters=lattice_parameters,
        overlay_parameters=overlay_parameters,
        solid_geometry_hash=solid_geometry_hash,
        lattice_hash=lattice_hash,
        design_hash=design_hash,
        calibration_hash=combined_calibration_hash,
        visualization_calibration_version=calibration.version,
    )
