"""Geometry19 solve/regenerate service with isolated, atomic mesh publication."""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import yaml

from src.physics import (
    ConstraintProfile,
    Environment,
    Geometry19,
    ModelConfig,
    OperatingPointSelectionPolicy,
    PhysicsRuntime,
    assess_constraint_profile,
    solve_design_point,
)
from src.geometry.assembly import assemble_engine
from src.visualization.geometry19_adapter import geometry19_to_assembly_params
from src.visualization.geometry19_mesh import generate_geometry19_meshes


class Geometry19ServiceError(ValueError):
    """Typed request/storage failure suitable for an HTTP 4xx response."""


_QUALITY_NAMES = {"preview_mesh", "export_mesh"}
_OPERATING_POINT_POLICY = (
    OperatingPointSelectionPolicy.REFERENCE_CONTINUATION_HIGH_FLOW_V1
)
_OPERATING_POINT_POLICY_FILE = (
    "geometry19_operating_point_policy_reference_continuation_v1.yaml"
)
_PROFILE_FILES = {
    "pipeline_debug_v1": "geometry19_constraint_profile_pipeline_debug_v1.yaml",
    "engineering_candidate_v1": "geometry19_constraint_profile_engineering_candidate_v1.yaml",
    "geometry19_constraint_profile_pipeline_debug_v1": (
        "geometry19_constraint_profile_pipeline_debug_v1.yaml"
    ),
    "geometry19_constraint_profile_engineering_candidate_v1": (
        "geometry19_constraint_profile_engineering_candidate_v1.yaml"
    ),
}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        # ``dataclasses.asdict`` deep-copies every leaf before recursing.  V2.2
        # map results deliberately expose immutable MappingProxyType metadata,
        # which cannot be pickled/deep-copied.  Walk fields directly so the
        # normal Mapping branch can serialize those read-only mappings.
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _geometry19_simulation_payload(visualization: Any, result: Any) -> dict[str, Any]:
    """Build viewer heat-map fields from the solved Geometry19 design point."""
    assembly = assemble_engine(visualization.assembly_params)
    params = visualization.assembly_params
    positions = assembly.component_positions
    stations = result.stations
    diagnostics = result.formal_diagnostics

    t2 = float(stations["t2"].total_temperature_K)
    t3 = float(stations["t3"].total_temperature_K)
    t4 = float(stations["t4"].total_temperature_K)
    t5 = float(stations["t5"].total_temperature_K)
    t6 = float(stations["exit"].static_temperature_K)
    metal = {
        "compressor": float(diagnostics["compressor_metal_temperature_K"]),
        "combustor": float(diagnostics["combustor_metal_temperature_K"]),
        "turbine": float(diagnostics["turbine_metal_temperature_K"]),
    }
    lengths = {
        "inlet": float(params.inlet.length_mm),
        "compressor": float(params.compressor.axial_length_mm),
        "combustor": float(params.combustor.length_mm),
        "turbine": float(params.turbine.stage_axial_length_mm),
        "nozzle": float(params.nozzle.length_mm),
        "shaft": float(assembly.total_length_mm),
        "casing": float(assembly.total_length_mm),
        "transitions": float(assembly.total_length_mm),
    }
    wall_ranges = {
        "inlet": (float(result.environment.temperature_K), t2),
        "compressor": (t2, metal["compressor"]),
        "combustor": (metal["compressor"], metal["combustor"]),
        "turbine": (metal["turbine"], max(t5, metal["turbine"])),
        "nozzle": (t5, t6),
        "shaft": (t2, metal["turbine"]),
        "casing": (t2, metal["combustor"]),
        "transitions": (t2, metal["turbine"]),
    }
    material_limits = {
        "inlet": 673.0,
        "compressor": 673.0,
        "combustor": 973.0,
        "turbine": 973.0,
        "nozzle": 1253.0,
        "shaft": 1143.0,
        "casing": 1143.0,
        "transitions": 973.0,
    }
    component_temps = {
        name: {
            "z": [0.0, lengths[name]],
            "wall_temp": [start, end],
            "z_offset": float(positions.get(name, 0.0)),
        }
        for name, (start, end) in wall_ranges.items()
    }

    stress_mpa = {
        "compressor": float(diagnostics["compressor_rotating_stress_Pa"]) / 1.0e6,
        "combustor": float(diagnostics["combustor_hoop_stress_Pa"]) / 1.0e6,
        "turbine": float(diagnostics["turbine_rotating_stress_Pa"]) / 1.0e6,
    }
    yield_mpa = {
        "compressor": float(diagnostics["compressor_yield_strength_proxy_Pa"]) / 1.0e6,
        "combustor": float(diagnostics["combustor_effective_yield_strength_proxy_Pa"]) / 1.0e6,
        "turbine": float(diagnostics["hot_section_yield_strength_proxy_Pa"]) / 1.0e6,
    }
    materials = {
        "compressor": "Ti-6Al-4V proxy",
        "combustor": "hot-section alloy proxy",
        "turbine": "hot-section alloy proxy",
    }
    component_stress = {
        name: {
            "z": [0.0, lengths[name]],
            "total_MPa": [0.65 * value, value],
            "z_offset": float(positions[name]),
            "yield_MPa": yield_mpa[name],
            "material": materials[name],
        }
        for name, value in stress_mpa.items()
    }
    yield_margins = {
        name: 100.0 * (yield_mpa[name] - value) / yield_mpa[name]
        for name, value in stress_mpa.items()
    }
    temperatures = [value for values in wall_ranges.values() for value in values]
    stresses = [value for values in component_stress.values() for value in values["total_MPa"]]
    mass_flow_fuel = float(result.operating_point["mass_flow_fuel_kg_s"])
    thrust = float(result.performance["thrust_N"])
    station_positions = {
        "S0": 0.0,
        "S1": positions["compressor"],
        "S2": positions["combustor"],
        "S3": positions["turbine"],
        "S4": positions["nozzle"],
        "S5": assembly.total_length_mm,
    }
    station_values = {
        "S0": ("Ambient", float(result.environment.temperature_K), float(result.environment.pressure_Pa)),
        "S1": ("Inlet Exit", t2, float(stations["t2"].total_pressure_Pa)),
        "S2": ("Compressor Exit", t3, float(stations["t3"].total_pressure_Pa)),
        "S3": ("Combustor Exit", t4, float(stations["t4"].total_pressure_Pa)),
        "S4": ("Turbine Exit", t5, float(stations["t5"].total_pressure_Pa)),
        "S5": ("Nozzle Exit", t6, float(stations["exit"].static_pressure_Pa)),
    }
    cycle_stations = {
        name: {
            "name": values[0],
            "T_total_K": round(values[1], 1),
            "P_total_kPa": round(values[2] / 1000.0, 2),
            "z_mm": round(float(station_positions[name]), 2),
        }
        for name, values in station_values.items()
    }
    return {
        "source": "geometry19_closed_v2_1",
        "thermal": {
            "component_temps": component_temps,
            "t_min": min(temperatures),
            "t_max": max(temperatures),
            "material_limits": material_limits,
        },
        "stress": {
            "component_stress": component_stress,
            "s_min": min(stresses),
            "s_max": max(stresses),
            "yield_margins": yield_margins,
        },
        "cycle": {
            "thrust_N": round(thrust, 2),
            "thrust_kgf": round(thrust / 9.80665, 2),
            "exhaust_velocity": round(float(stations["exit"].velocity_m_s), 2),
            "exhaust_temp_K": round(t6, 1),
            "fuel_flow_g_hr": round(mass_flow_fuel * 3.6e6, 2),
            "tsfc_g_kNs": round(float(result.performance["tsfc_kg_N_s"]) * 1.0e6, 3),
            "thermal_efficiency_pct": "N/A",
            "compressor_power_kW": round(float(result.components["compressor"]["shaft_power_W"]) / 1000.0, 3),
            "turbine_power_kW": round(float(result.components["compressor"]["shaft_power_W"]) / 1000.0, 3),
            "stations": cycle_stations,
            "warnings": [],
        },
    }


class Geometry19DesignStore:
    """Repository-scoped Geometry19 runtime service and artifact store."""

    def __init__(
        self,
        repository_root: str | Path,
        storage_root: str | Path,
        *,
        physics_runtime: PhysicsRuntime | None = None,
    ):
        self.repository_root = Path(repository_root).resolve()
        self.storage_root = Path(storage_root).resolve()
        self.physics_config_root = self.repository_root / "config" / "physics"
        if (
            physics_runtime is not None
            and physics_runtime.repository_root != self.repository_root
        ):
            raise Geometry19ServiceError(
                "Physics runtime repository differs from the design-store repository"
            )
        self.physics_runtime = physics_runtime

    def _model_config(self) -> ModelConfig:
        if self.physics_runtime is not None:
            return self.physics_runtime.model_config
        return ModelConfig.from_yaml(
            self.physics_config_root / "geometry19_calibration_bootstrap_v1.yaml"
        )

    def _constraint_profile(self, name: str | None) -> ConstraintProfile:
        if self.physics_runtime is not None:
            profile = self.physics_runtime.constraint_profile
            if name is None or name == profile.profile_version:
                return profile
            filename = _PROFILE_FILES.get(name)
            if filename is not None and Path(filename).stem == profile.profile_version:
                return profile
            raise Geometry19ServiceError(
                "Requested constraint profile differs from the injected physics runtime"
            )
        name = "pipeline_debug_v1" if name is None else name
        filename = _PROFILE_FILES.get(name)
        if filename is None:
            raise Geometry19ServiceError(f"Unsupported constraint profile: {name}")
        return ConstraintProfile.from_yaml(self.physics_config_root / filename)

    def schema_payload(self) -> dict[str, Any]:
        if self.physics_runtime is not None:
            return {
                "schema": dict(self.physics_runtime.design_schema),
                "schema_hash": self.physics_runtime.provenance_hashes[
                    "design_schema_hash"
                ],
                "canonical_names": list(Geometry19.coordinate_names()),
                "supported_constraint_profiles": [
                    self.physics_runtime.constraint_profile.profile_version
                ],
                "supported_mesh_qualities": sorted(_QUALITY_NAMES),
                "operating_point_policy": (
                    self.physics_runtime.operating_point_policy.value
                ),
                "operating_point_policy_hash": self.physics_runtime.provenance_hashes[
                    "operating_point_policy_hash"
                ],
            }
        path = self.physics_config_root / "geometry19_schema_v1.yaml"
        try:
            source = path.read_bytes()
            schema = yaml.safe_load(source)
        except (OSError, yaml.YAMLError) as exc:
            raise Geometry19ServiceError("Cannot load Geometry19 schema") from exc
        return {
            "schema": schema,
            "schema_hash": hashlib.sha256(source).hexdigest(),
            "canonical_names": list(Geometry19.coordinate_names()),
            "supported_constraint_profiles": [
                "pipeline_debug_v1",
                "engineering_candidate_v1",
            ],
            "supported_mesh_qualities": sorted(_QUALITY_NAMES),
            "operating_point_policy": _OPERATING_POINT_POLICY.value,
            "operating_point_policy_hash": _sha256_file(
                self.physics_config_root / _OPERATING_POINT_POLICY_FILE
            ),
        }

    def parse_design(self, payload: Mapping[str, Any]) -> Geometry19:
        if not isinstance(payload, Mapping):
            raise Geometry19ServiceError("JSON request must be an object")
        try:
            return Geometry19.from_mapping(payload)
        except (TypeError, ValueError) as exc:
            raise Geometry19ServiceError(str(exc)) from exc

    def solve(
        self,
        design: Geometry19,
        environment: Environment | None = None,
        constraint_profile: str | None = None,
    ) -> dict[str, Any]:
        config = self._model_config()
        result = (
            self.physics_runtime.evaluate(design, environment)
            if self.physics_runtime is not None
            else solve_design_point(
                design,
                environment or Environment(),
                config,
                operating_point_policy=_OPERATING_POINT_POLICY,
            )
        )
        assessment = assess_constraint_profile(
            result,
            self._constraint_profile(constraint_profile),
            conservation_tolerances=(
                self.physics_runtime.conservation_tolerances
                if self.physics_runtime is not None
                else None
            ),
        )
        return {
            "geometry19": design.as_mapping(),
            "physics": _jsonable(result),
            "constraints": assessment.as_mapping(),
            "solver": _jsonable(result.solver),
            "schema_hash": self.schema_payload()["schema_hash"],
            "calibration_hash": config.calibration_pack_hash,
            "operating_point_policy": self.schema_payload()[
                "operating_point_policy"
            ],
            "operating_point_policy_hash": self.schema_payload()[
                "operating_point_policy_hash"
            ],
            "physics_bundle_hash": (
                self.physics_runtime.physics_bundle_hash
                if self.physics_runtime is not None
                else None
            ),
        }

    @staticmethod
    def _validate_design_id(design_id: str) -> str:
        if (
            not isinstance(design_id, str)
            or len(design_id) != 64
            or any(character not in "0123456789abcdef" for character in design_id)
        ):
            raise Geometry19ServiceError("design_id must be a 64-character lowercase hash")
        return design_id

    def design_directory(self, design_id: str) -> Path:
        return self.storage_root / self._validate_design_id(design_id)

    def publish_mesh(
        self,
        design: Geometry19,
        quality: str = "preview_mesh",
    ) -> dict[str, Any]:
        """Publish deterministic mesh assets without running a physics solve."""
        if quality not in _QUALITY_NAMES:
            raise Geometry19ServiceError(f"Unsupported mesh quality: {quality}")
        config = self._model_config()
        try:
            visualization = geometry19_to_assembly_params(design, config)
        except (TypeError, ValueError) as exc:
            raise Geometry19ServiceError(str(exc)) from exc
        design_id = visualization.design_hash
        design_root = self.design_directory(design_id)
        design_root.mkdir(parents=True, exist_ok=True)
        target = design_root / quality

        def publication_complete(directory: Path) -> bool:
            return (
                (directory / "mesh_diagnostics.json").is_file()
                and (directory / "manifest.json").is_file()
            )

        if not publication_complete(target):
            self.storage_root.mkdir(parents=True, exist_ok=True)
            temporary_root = Path(
                tempfile.mkdtemp(prefix=f".{design_id}.", dir=self.storage_root)
            )
            temporary_quality = temporary_root / quality
            try:
                mesh_result = generate_geometry19_meshes(
                    visualization, temporary_quality, quality
                )
                if not mesh_result.passed:
                    raise Geometry19ServiceError(
                        "Generated mesh failed mandatory quality diagnostics"
                    )
                mesh_files = sorted(temporary_quality.glob("*.stl"))
                _write_json_atomic(
                    temporary_quality / "manifest.json",
                    {
                        "design_id": design_id,
                        "mesh_quality": quality,
                        "mesh_profile_version": getattr(
                            mesh_result, "mesh_profile_version", "test_double"
                        ),
                        "solid_geometry_hash": visualization.solid_geometry_hash,
                        "lattice_hash": visualization.lattice_hash,
                        "calibration_hash": visualization.calibration_hash,
                        "components": {
                            path.stem: {
                                "filename": path.name,
                                "sha256": _sha256_file(path),
                                "size_bytes": path.stat().st_size,
                            }
                            for path in mesh_files
                        },
                    },
                )
                try:
                    os.rename(temporary_quality, target)
                except OSError:
                    # A concurrent identical request published first.  Its
                    # immutable hash-addressed result is the desired result.
                    if not publication_complete(target):
                        raise
            finally:
                shutil.rmtree(temporary_root, ignore_errors=True)

        if not publication_complete(target):
            raise Geometry19ServiceError("Mesh publication did not produce diagnostics")

        _write_json_atomic(
            design_root / "geometry19.json",
            {
                "design_id": design_id,
                "geometry19": design.as_mapping(),
                "design_hash": visualization.design_hash,
                "solid_geometry_hash": visualization.solid_geometry_hash,
                "lattice_hash": visualization.lattice_hash,
                "calibration_hash": visualization.calibration_hash,
                "visualization_calibration_version": (
                    visualization.visualization_calibration_version
                ),
            },
        )
        _write_json_atomic(
            design_root / "derived_geometry.json",
            _jsonable(asdict(visualization.geometry_state)),
        )

        component_files = sorted(
            path for path in target.glob("*.stl") if path.is_file()
        )
        component_urls = {
            path.stem: (
                f"/geometry19-files/{design_id}/{quality}/{path.name}"
                f"?h={_sha256_file(path)}"
            )
            for path in component_files
        }
        diagnostics = json.loads(
            (target / "mesh_diagnostics.json").read_text(encoding="utf-8")
        )
        manifest_path = target / "manifest.json"
        return {
            "design_id": design_id,
            "geometry19": design.as_mapping(),
            "derived_geometry": _jsonable(asdict(visualization.geometry_state)),
            "mesh_diagnostics": diagnostics,
            "component_urls": component_urls,
            "geometry_hash": visualization.solid_geometry_hash,
            "lattice_hash": visualization.lattice_hash,
            "overlay_parameters": dict(visualization.overlay_parameters),
            "calibration_hash": visualization.calibration_hash,
            "mesh_quality": quality,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
        }

    def regenerate(
        self,
        design: Geometry19,
        quality: str = "preview_mesh",
        environment: Environment | None = None,
        constraint_profile: str | None = None,
    ) -> dict[str, Any]:
        mesh_payload = self.publish_mesh(design, quality)
        config = self._model_config()

        resolved_environment = environment or Environment()
        result = (
            self.physics_runtime.evaluate(design, resolved_environment)
            if self.physics_runtime is not None
            else solve_design_point(
                design,
                resolved_environment,
                config,
                operating_point_policy=_OPERATING_POINT_POLICY,
            )
        )
        assessment = assess_constraint_profile(
            result,
            self._constraint_profile(constraint_profile),
            conservation_tolerances=(
                self.physics_runtime.conservation_tolerances
                if self.physics_runtime is not None
                else None
            ),
        )
        solve_payload = {
            "geometry19": design.as_mapping(),
            "physics": _jsonable(result),
            "constraints": assessment.as_mapping(),
            "solver": _jsonable(result.solver),
            "schema_hash": self.schema_payload()["schema_hash"],
            "calibration_hash": config.calibration_pack_hash,
            "operating_point_policy": self.schema_payload()[
                "operating_point_policy"
            ],
            "operating_point_policy_hash": self.schema_payload()[
                "operating_point_policy_hash"
            ],
            "physics_bundle_hash": (
                self.physics_runtime.physics_bundle_hash
                if self.physics_runtime is not None
                else None
            ),
        }
        response = {
            **mesh_payload,
            **solve_payload,
            "simulation": (
                _geometry19_simulation_payload(
                    geometry19_to_assembly_params(design, config), result
                )
                if result.solver.converged
                else None
            ),
            "constraint_profile": solve_payload["constraints"]["constraint_profile"],
        }
        return response

    def get_design(self, design_id: str) -> dict[str, Any]:
        path = self.design_directory(design_id) / "geometry19.json"
        if not path.is_file():
            raise Geometry19ServiceError("Unknown design_id")
        return json.loads(path.read_text(encoding="utf-8"))

    def get_components(self, design_id: str, quality: str = "preview_mesh") -> dict[str, Any]:
        if quality not in _QUALITY_NAMES:
            raise Geometry19ServiceError(f"Unsupported mesh quality: {quality}")
        target = self.design_directory(design_id) / quality
        if not target.is_dir():
            raise Geometry19ServiceError("No generated components for design_id/quality")
        components = {}
        for path in sorted(target.glob("*.stl")):
            components[path.stem] = {
                "url": (
                    f"/geometry19-files/{design_id}/{quality}/{path.name}"
                    f"?h={_sha256_file(path)}"
                ),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        return {"design_id": design_id, "mesh_quality": quality, "components": components}
