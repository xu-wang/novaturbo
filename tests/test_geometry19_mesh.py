"""WP19-13 two-level mesh and diagnostic regression tests."""

from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("trimesh")
pytest.importorskip("skimage")

from src.physics import Geometry19, ModelConfig
from src.export.stl_export import export_combustor_stl
from src.geometry.combustor import CombustorParams, compute_combustor_geometry
from src.visualization import (
    MeshQualityProfile,
    generate_geometry19_meshes,
    geometry19_to_assembly_params,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PHYSICS_CONFIG = REPO_ROOT / "config" / "physics"


@pytest.fixture(scope="module")
def model_config() -> ModelConfig:
    return ModelConfig.from_yaml(
        PHYSICS_CONFIG / "geometry19_calibration_bootstrap_v1.yaml"
    )


@pytest.fixture(scope="module")
def reference_design() -> Geometry19:
    schema = yaml.safe_load(
        (PHYSICS_CONFIG / "geometry19_schema_v1.yaml").read_text(encoding="utf-8")
    )
    return Geometry19.from_mapping(
        {item["name"]: item["reference_value"] for item in schema["design_variables"]}
    )


def test_mesh_profiles_define_preview_and_export_levels():
    path = PHYSICS_CONFIG / "geometry19_mesh_profiles_v1.yaml"
    preview = MeshQualityProfile.from_yaml(path, "preview_mesh")
    export = MeshQualityProfile.from_yaml(path, "export_mesh")
    assert preview.circumferential_segments < export.circumferential_segments
    assert preview.lattice_voxels_per_cell < export.lattice_voxels_per_cell


def test_reference_preview_mesh_passes_component_diagnostics(
    tmp_path, reference_design, model_config
):
    assembly = geometry19_to_assembly_params(reference_design, model_config)
    result = generate_geometry19_meshes(assembly, tmp_path, "preview_mesh")
    assert result.passed is True
    assert set(result.component_diagnostics) == {
        "inlet",
        "compressor",
        "combustor",
        "turbine",
        "nozzle",
        "shaft",
        "casing",
        "transitions",
        "assembly",
        "combustor_lattice",
        "nozzle_lattice",
    }
    for name in (
        "inlet",
        "compressor",
        "combustor",
        "turbine",
        "nozzle",
        "shaft",
        "casing",
        "transitions",
    ):
        diagnostic = result.component_diagnostics[name]
        assert diagnostic.watertight is True
        assert diagnostic.positive_volume is True
        assert diagnostic.degenerate_face_count == 0
        assert diagnostic.axial_bounds_valid is True
        assert diagnostic.body_count_valid is True
    assert result.geometry_measurements["nozzle_throat_interface_valid"] is True
    payload = json.loads((tmp_path / "mesh_diagnostics.json").read_text())
    assert payload["passed"] is True


def test_combustor_stl_contains_one_body_per_discrete_fuel_injector():
    baseline = CombustorParams(num_fuel_injectors=6)
    changed = replace(baseline, num_fuel_injectors=10)
    baseline_mesh = export_combustor_stl(
        compute_combustor_geometry(baseline), baseline, n_theta=24
    )
    changed_mesh = export_combustor_stl(
        compute_combustor_geometry(changed), changed, n_theta=24
    )
    baseline_mesh.merge_vertices()
    changed_mesh.merge_vertices()

    assert baseline_mesh.body_count == 2 + baseline.num_fuel_injectors
    assert changed_mesh.body_count == 2 + changed.num_fuel_injectors
    assert len(changed_mesh.faces) > len(baseline_mesh.faces)


def test_reference_export_mesh_passes_high_resolution_diagnostics(
    tmp_path, reference_design, model_config
):
    assembly = geometry19_to_assembly_params(reference_design, model_config)
    result = generate_geometry19_meshes(assembly, tmp_path, "export_mesh")
    assert result.passed is True
    assert result.component_diagnostics["inlet"].face_count > 10000
    assert result.component_diagnostics["casing"].face_count > 25000
    assert result.component_diagnostics["combustor_lattice"].degenerate_face_count == 0


def test_lattice_inputs_change_actual_preview_mesh(
    tmp_path, reference_design, model_config
):
    baseline = geometry19_to_assembly_params(reference_design, model_config)
    changed_design = replace(
        reference_design,
        lattice_cell_size_m=0.01,
        lattice_relative_density=0.4,
    )
    changed = geometry19_to_assembly_params(changed_design, model_config)
    baseline_result = generate_geometry19_meshes(
        baseline, tmp_path / "baseline", "preview_mesh"
    )
    changed_result = generate_geometry19_meshes(
        changed, tmp_path / "changed", "preview_mesh"
    )
    assert baseline_result.solid_geometry_hash == changed_result.solid_geometry_hash
    assert baseline_result.lattice_hash != changed_result.lattice_hash
    baseline_lattice = baseline_result.component_diagnostics["combustor_lattice"]
    changed_lattice = changed_result.component_diagnostics["combustor_lattice"]
    assert (baseline_lattice.vertex_count, baseline_lattice.face_count) != (
        changed_lattice.vertex_count,
        changed_lattice.face_count,
    )


def test_relative_density_changes_tpms_sheet_geometry():
    from src.geometry.lattice import (
        AnnularRegion,
        LatticeParams,
        generate_annular_lattice,
    )

    region = AnnularRegion(
        z_start_mm=0.0,
        z_end_mm=16.0,
        r_inner_mm=8.0,
        r_outer_mm=16.0,
        inner_wall_mm=0.5,
        outer_wall_mm=0.5,
    )
    thin_vertices, thin_faces = generate_annular_lattice(
        region,
        LatticeParams(cell_size_mm=8.0, wall_thickness=0.2, resolution=3),
    )
    thick_vertices, thick_faces = generate_annular_lattice(
        region,
        LatticeParams(cell_size_mm=8.0, wall_thickness=0.4, resolution=3),
    )
    assert len(thin_vertices) > 0 and len(thick_vertices) > 0
    assert (len(thin_vertices), len(thin_faces)) != (
        len(thick_vertices),
        len(thick_faces),
    )
