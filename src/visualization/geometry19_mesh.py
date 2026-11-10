"""Two-level Geometry19 mesh generation and machine-readable diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from src.geometry.assembly import assemble_engine
from src.visualization.geometry19_adapter import VisualizationAssembly


class Geometry19MeshError(RuntimeError):
    """Raised when a mesh profile or Geometry19 mesh generation is invalid."""


@dataclass(frozen=True)
class MeshQualityProfile:
    profile_version: str
    quality: str
    circumferential_segments: int
    lattice_voxels_per_cell: int
    degenerate_face_area_tolerance_mm2: float
    axial_bounds_tolerance_mm: float
    interface_tolerance_mm: float
    expected_watertight_components: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: str | Path, quality: str) -> "MeshQualityProfile":
        source_path = Path(path)
        try:
            payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise Geometry19MeshError(f"Cannot load mesh profile: {source_path}") from exc
        if not isinstance(payload, dict) or set(payload) != {
            "profile_version",
            "description",
            "qualities",
            "diagnostics",
        }:
            raise Geometry19MeshError("Mesh profile top-level contract mismatch")
        if source_path.stem != payload["profile_version"]:
            raise Geometry19MeshError("Mesh profile filename/version mismatch")
        qualities = payload["qualities"]
        if not isinstance(qualities, dict) or set(qualities) != {
            "preview_mesh",
            "export_mesh",
        }:
            raise Geometry19MeshError("Mesh profile must define preview_mesh and export_mesh")
        if quality not in qualities:
            raise Geometry19MeshError(f"Unsupported mesh quality: {quality}")
        selected = qualities[quality]
        if not isinstance(selected, dict) or set(selected) != {
            "circumferential_segments",
            "lattice_voxels_per_cell",
        }:
            raise Geometry19MeshError(f"Invalid {quality} profile")
        diagnostics = payload["diagnostics"]
        expected_diagnostics = {
            "degenerate_face_area_tolerance_mm2",
            "axial_bounds_tolerance_mm",
            "interface_tolerance_mm",
            "expected_watertight_components",
        }
        if not isinstance(diagnostics, dict) or set(diagnostics) != expected_diagnostics:
            raise Geometry19MeshError("Mesh diagnostics contract mismatch")
        n_theta = selected["circumferential_segments"]
        lattice_resolution = selected["lattice_voxels_per_cell"]
        if type(n_theta) is not int or n_theta < 12 or n_theta % 2:
            raise Geometry19MeshError("circumferential_segments must be an even integer >= 12")
        if type(lattice_resolution) is not int or lattice_resolution < 2:
            raise Geometry19MeshError("lattice_voxels_per_cell must be an integer >= 2")
        watertight = diagnostics["expected_watertight_components"]
        if not isinstance(watertight, list) or any(
            not isinstance(name, str) or not name for name in watertight
        ):
            raise Geometry19MeshError("expected_watertight_components is invalid")
        numeric = {
            name: diagnostics[name]
            for name in (
                "degenerate_face_area_tolerance_mm2",
                "axial_bounds_tolerance_mm",
                "interface_tolerance_mm",
            )
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in numeric.values()
        ):
            raise Geometry19MeshError("Mesh diagnostic tolerances must be finite and nonnegative")
        return cls(
            profile_version=payload["profile_version"],
            quality=quality,
            circumferential_segments=n_theta,
            lattice_voxels_per_cell=lattice_resolution,
            degenerate_face_area_tolerance_mm2=float(
                numeric["degenerate_face_area_tolerance_mm2"]
            ),
            axial_bounds_tolerance_mm=float(numeric["axial_bounds_tolerance_mm"]),
            interface_tolerance_mm=float(numeric["interface_tolerance_mm"]),
            expected_watertight_components=tuple(watertight),
        )


@dataclass(frozen=True)
class ComponentMeshDiagnostics:
    name: str
    vertex_count: int
    face_count: int
    body_count: int
    finite_vertices: bool
    removed_degenerate_face_count: int
    degenerate_face_count: int
    winding_consistent: bool
    watertight: bool
    watertight_required: bool
    volume_mm3: float
    positive_volume: bool
    bounds_min_mm: tuple[float, float, float]
    bounds_max_mm: tuple[float, float, float]
    axial_bounds_valid: bool
    expected_body_count: int | None
    body_count_valid: bool
    passed: bool


@dataclass(frozen=True)
class Geometry19MeshResult:
    quality: str
    mesh_profile_version: str
    solid_geometry_hash: str
    lattice_hash: str
    output_directory: str
    component_files: Mapping[str, str]
    component_diagnostics: Mapping[str, ComponentMeshDiagnostics]
    geometry_measurements: Mapping[str, float | int | bool]
    passed: bool

    def as_mapping(self) -> dict[str, Any]:
        return {
            "quality": self.quality,
            "mesh_profile_version": self.mesh_profile_version,
            "solid_geometry_hash": self.solid_geometry_hash,
            "lattice_hash": self.lattice_hash,
            "output_directory": self.output_directory,
            "component_files": dict(self.component_files),
            "component_diagnostics": {
                name: asdict(value)
                for name, value in self.component_diagnostics.items()
            },
            "geometry_measurements": dict(self.geometry_measurements),
            "passed": self.passed,
        }


def _default_mesh_profile_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "config"
        / "physics"
        / "geometry19_mesh_profiles_v1.yaml"
    )


def _normalize_mesh(mesh: Any, area_tolerance_mm2: float) -> tuple[Any, int]:
    original_faces = len(mesh.faces)
    nondegenerate = mesh.nondegenerate_faces()
    areas = np.asarray(mesh.area_faces)
    nondegenerate &= np.isfinite(areas) & (areas > area_tolerance_mm2)
    mesh.update_faces(nondegenerate)
    removed = original_faces - int(np.count_nonzero(nondegenerate))
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    import trimesh

    trimesh.repair.fix_normals(mesh, multibody=True)
    return mesh, removed


def _expected_axial_bounds(assembly: VisualizationAssembly) -> Mapping[str, tuple[float, float]]:
    params = assembly.assembly_params
    built = assemble_engine(params)
    positions = built.component_positions
    return {
        "inlet": (positions["inlet"], positions["inlet"] + params.inlet.length_mm),
        "compressor": (
            positions["compressor"],
            positions["compressor"] + params.compressor.axial_length_mm,
        ),
        "combustor": (
            positions["combustor"],
            positions["combustor"] + params.combustor.length_mm,
        ),
        "turbine": (
            positions["turbine"],
            positions["turbine"] + params.turbine.stage_axial_length_mm,
        ),
        "nozzle": (
            positions["nozzle"],
            positions["nozzle"] + params.nozzle.length_mm,
        ),
        "shaft": (0.0, built.total_length_mm),
        "casing": (0.0, built.total_length_mm),
        "transitions": (
            positions["compressor"] - 3.0,
            positions["nozzle"],
        ),
        "assembly": (0.0, built.total_length_mm),
        "combustor_lattice": (
            positions["combustor"],
            positions["combustor"] + params.combustor.length_mm,
        ),
        "nozzle_lattice": (
            positions["nozzle"],
            positions["nozzle"] + params.nozzle.length_mm,
        ),
    }


def _expected_body_counts(assembly: VisualizationAssembly) -> Mapping[str, int]:
    params = assembly.assembly_params
    return {
        "inlet": 1,
        "compressor": 1 + params.compressor.blade_count + params.compressor.splitter_blade_count,
        "combustor": 2 + params.combustor.num_fuel_injectors,
        "turbine": 2 + params.turbine.ngv_count + params.turbine.blade_count,
        "nozzle": 1,
        "shaft": 1,
        "casing": 1,
        "transitions": 6,
    }


def _diagnose_mesh(
    name: str,
    mesh: Any,
    removed_degenerate_faces: int,
    profile: MeshQualityProfile,
    axial_bounds: tuple[float, float],
    expected_body_count: int | None,
) -> ComponentMeshDiagnostics:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    finite_vertices = bool(vertices.size and np.isfinite(vertices).all())
    areas = np.asarray(mesh.area_faces) if len(faces) else np.asarray([])
    remaining_degenerate = int(
        np.count_nonzero(
            ~np.isfinite(areas)
            | (areas <= profile.degenerate_face_area_tolerance_mm2)
        )
    )
    bounds = np.asarray(mesh.bounds, dtype=float)
    bounds_valid = bounds.shape == (2, 3) and np.isfinite(bounds).all()
    axial_valid = bool(
        bounds_valid
        and abs(bounds[0, 2] - axial_bounds[0]) <= profile.axial_bounds_tolerance_mm
        and abs(bounds[1, 2] - axial_bounds[1]) <= profile.axial_bounds_tolerance_mm
    )
    body_count = int(mesh.body_count)
    body_count_valid = expected_body_count is None or body_count == expected_body_count
    watertight_required = name in profile.expected_watertight_components
    watertight = bool(mesh.is_watertight)
    volume = float(mesh.volume)
    positive_volume = math.isfinite(volume) and volume > 0.0
    passed = bool(
        len(vertices) > 0
        and len(faces) > 0
        and finite_vertices
        and remaining_degenerate == 0
        and bool(mesh.is_winding_consistent)
        and (watertight or not watertight_required)
        and (positive_volume or not watertight_required)
        and axial_valid
        and body_count_valid
    )
    return ComponentMeshDiagnostics(
        name=name,
        vertex_count=len(vertices),
        face_count=len(faces),
        body_count=body_count,
        finite_vertices=finite_vertices,
        removed_degenerate_face_count=removed_degenerate_faces,
        degenerate_face_count=remaining_degenerate,
        winding_consistent=bool(mesh.is_winding_consistent),
        watertight=watertight,
        watertight_required=watertight_required,
        volume_mm3=volume,
        positive_volume=positive_volume,
        bounds_min_mm=tuple(float(value) for value in bounds[0]),
        bounds_max_mm=tuple(float(value) for value in bounds[1]),
        axial_bounds_valid=axial_valid,
        expected_body_count=expected_body_count,
        body_count_valid=body_count_valid,
        passed=passed,
    )


def _generate_lattice_meshes(
    assembly: VisualizationAssembly,
    profile: MeshQualityProfile,
) -> Mapping[str, Any]:
    import trimesh
    from src.geometry.lattice import (
        LatticeParams,
        TPMSType,
        generate_combustor_lattice,
        generate_nozzle_lattice,
    )

    params = assembly.assembly_params
    built = assemble_engine(params)
    lattice = assembly.lattice_parameters
    topology = str(lattice["topology"]).upper()
    try:
        tpms_type = TPMSType[topology]
    except KeyError as exc:
        raise Geometry19MeshError(f"Unsupported TPMS topology: {topology}") from exc
    threshold = np.clip(
        float(lattice["relative_density"])
        * float(lattice["relative_density_to_wall_threshold"]),
        float(lattice["wall_threshold_min"]),
        float(lattice["wall_threshold_max"]),
    )
    lattice_params = LatticeParams(
        tpms_type=tpms_type,
        cell_size_mm=float(lattice["cell_size_m"]) * 1000.0,
        wall_thickness=float(threshold),
        combustor_inner_skin_mm=float(lattice["combustor_inner_skin_mm"]),
        combustor_outer_skin_mm=float(lattice["combustor_outer_skin_mm"]),
        resolution=profile.lattice_voxels_per_cell,
        grade_radially=False,
        grade_axially=False,
    )
    generators = {
        "combustor_lattice": (
            generate_combustor_lattice,
            params.combustor,
            built.component_positions["combustor"],
        ),
        "nozzle_lattice": (
            generate_nozzle_lattice,
            params.nozzle,
            built.component_positions["nozzle"],
        ),
    }
    meshes = {}
    for name, (generator, component_params, z_offset) in generators.items():
        vertices, faces = generator(component_params, z_offset, lattice_params)
        if len(vertices) == 0 or len(faces) == 0:
            raise Geometry19MeshError(f"{name} generation returned an empty mesh")
        meshes[name] = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    return meshes


def generate_geometry19_meshes(
    assembly: VisualizationAssembly,
    output_directory: str | Path,
    quality: str = "preview_mesh",
    profile_path: str | Path | None = None,
) -> Geometry19MeshResult:
    """Generate, normalize, diagnose, and persist one Geometry19 mesh set."""
    if not isinstance(assembly, VisualizationAssembly):
        raise TypeError("assembly must be a VisualizationAssembly")
    profile = MeshQualityProfile.from_yaml(
        profile_path or _default_mesh_profile_path(), quality
    )
    try:
        import trimesh
    except ImportError as exc:
        raise Geometry19MeshError("trimesh is required for mesh generation") from exc
    from src.export.stl_export import export_full_engine

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    exported = export_full_engine(
        output_dir=str(output_path),
        n_theta=profile.circumferential_segments,
        verbose=False,
        assembly_params=assembly.assembly_params,
    )
    if not exported:
        raise Geometry19MeshError("Solid engine mesh export returned no files")

    solid_meshes = {
        name: trimesh.load_mesh(path, process=True, force="mesh")
        for name, path in exported.items()
    }
    lattice_meshes = _generate_lattice_meshes(assembly, profile)
    all_meshes = {**solid_meshes, **lattice_meshes}
    axial_bounds = _expected_axial_bounds(assembly)
    expected_bodies = _expected_body_counts(assembly)
    diagnostics = {}
    component_files = {}
    normalized_meshes = {}
    for name, mesh in all_meshes.items():
        mesh, removed = _normalize_mesh(
            mesh, profile.degenerate_face_area_tolerance_mm2
        )
        normalized_meshes[name] = mesh
        path = output_path / f"{name}.stl"
        mesh.export(path)
        component_files[name] = str(path)
        diagnostics[name] = _diagnose_mesh(
            name,
            mesh,
            removed,
            profile,
            axial_bounds[name],
            expected_bodies.get(name),
        )

    nozzle = normalized_meshes["nozzle"]
    nozzle_vertices = np.asarray(nozzle.vertices)
    nozzle_z_max = float(nozzle_vertices[:, 2].max())
    throat_ring = nozzle_vertices[
        np.isclose(
            nozzle_vertices[:, 2],
            nozzle_z_max,
            atol=profile.interface_tolerance_mm,
        )
    ]
    throat_radii = np.linalg.norm(throat_ring[:, :2], axis=1)
    measured_throat = 2.0 * float(throat_radii.min())
    expected_throat = assembly.geometry_state.design.nozzle_throat_diameter_m * 1000.0
    measurements: dict[str, float | int | bool] = {
        "nozzle_throat_diameter_mesh_mm": measured_throat,
        "nozzle_throat_diameter_state_mm": expected_throat,
        "nozzle_throat_interface_valid": abs(measured_throat - expected_throat)
        <= profile.interface_tolerance_mm,
        "compressor_blade_count": assembly.assembly_params.compressor.blade_count,
        "compressor_splitter_blade_count": (
            assembly.assembly_params.compressor.splitter_blade_count
        ),
        "combustor_injector_count": (
            assembly.assembly_params.combustor.num_fuel_injectors
        ),
        "turbine_blade_count": assembly.assembly_params.turbine.blade_count,
    }
    passed = all(item.passed for item in diagnostics.values()) and bool(
        measurements["nozzle_throat_interface_valid"]
    )
    result = Geometry19MeshResult(
        quality=quality,
        mesh_profile_version=profile.profile_version,
        solid_geometry_hash=assembly.solid_geometry_hash,
        lattice_hash=assembly.lattice_hash,
        output_directory=str(output_path),
        component_files=component_files,
        component_diagnostics=diagnostics,
        geometry_measurements=measurements,
        passed=passed,
    )
    (output_path / "mesh_diagnostics.json").write_text(
        json.dumps(result.as_mapping(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result
