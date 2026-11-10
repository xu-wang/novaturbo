"""V3 mesh geometry mapping, publication integrity, and manual UI API."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.physics.geometry22_design import Geometry22Design
from src.visualization import geometry22_mesh as mesh
from ui import server


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reference():
    schema = yaml.safe_load((ROOT / "config/physics/geometry22_schema_v3_development_v1.yaml").read_text())
    return {row["name"]: row["reference_value"] for row in schema["design_variables"]}


def test_v3_geometric_coordinates_change_assembly_and_air_split_remains_flow_metadata(reference):
    base = mesh.Geometry22MeshStore.assembly(Geometry22Design.from_mapping(reference), ROOT)
    diffuser = mesh.Geometry22MeshStore.assembly(Geometry22Design.from_mapping(
        {**reference, "diffuser_area_ratio": reference["diffuser_area_ratio"] + 0.2}), ROOT)
    clearance = mesh.Geometry22MeshStore.assembly(Geometry22Design.from_mapping(
        {**reference, "compressor_tip_clearance_ratio": 0.004}), ROOT)
    air = mesh.Geometry22MeshStore.assembly(Geometry22Design.from_mapping(
        {**reference, "combustor_primary_air_fraction": 0.3}), ROOT)
    assert diffuser.assembly_params.compressor.diffuser_radius_ratio != base.assembly_params.compressor.diffuser_radius_ratio
    assert clearance.assembly_params.compressor.shroud_clearance_mm != base.assembly_params.compressor.shroud_clearance_mm
    assert diffuser.solid_geometry_hash != base.solid_geometry_hash
    assert clearance.solid_geometry_hash != base.solid_geometry_hash
    assert air.solid_geometry_hash == base.solid_geometry_hash
    assert air.overlay_parameters["combustor_primary_air_fraction"] == 0.3


def test_manual_mesh_api_has_full_22d_identity_and_sealed_components(reference, tmp_path, monkeypatch):
    monkeypatch.setitem(server.app.config, "GEOMETRY22_PREVIEW_ROOT", str(tmp_path))
    def fake_generate(assembly, directory, quality):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "compressor.stl").write_bytes(b"V3 synthetic compressor")
        return SimpleNamespace(passed=True, as_mapping=lambda: {"passed": True})
    monkeypatch.setattr(mesh, "generate_geometry19_meshes", fake_generate)
    client = server.app.test_client()
    response = client.post("/api/geometry22/mesh", json={"geometry22": reference})
    assert response.status_code == 200
    data = response.get_json()
    assert data["geometry22"] == reference and data["mesh_scope"] == "22d_design_driven_visual_mesh"
    assert data["flow_only_parameters"] == {"combustor_primary_air_fraction": reference["combustor_primary_air_fraction"]}
    url = data["component_urls"]["compressor"]
    assert client.get(url).status_code == 200
    assert client.get(url.replace("compressor.stl", "unknown.stl")).status_code == 404
    path = tmp_path / "mesh22" / data["mesh_id"] / "compressor.stl"
    path.write_bytes(b"tampered")
    assert client.get(url).status_code == 409
    assert client.post("/api/geometry22/mesh", json={"geometry22": reference}).status_code == 409


def test_manual_mesh_api_rejects_incomplete_design():
    assert server.app.test_client().post("/api/geometry22/mesh", json={"geometry22": {}}).status_code == 400
