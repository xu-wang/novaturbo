"""WP19-14 storage isolation and Flask API tests."""

from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import yaml

pytest.importorskip("trimesh")
pytest.importorskip("skimage")

from src.physics import Geometry19
from src.visualization import Geometry19DesignStore
from src.visualization.geometry19_service import _jsonable
from ui.server import app


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "config" / "physics" / "geometry19_schema_v1.yaml"


def test_jsonable_serializes_read_only_mapping_inside_dataclass():
    @dataclass(frozen=True)
    class MapResult:
        uncertainty: object

    payload = _jsonable(
        MapResult(uncertainty=MappingProxyType({"pressure_ratio": 0.02}))
    )

    assert payload == {"uncertainty": {"pressure_ratio": 0.02}}
    assert json.loads(json.dumps(payload)) == payload


def test_publish_mesh_does_not_run_physics(tmp_path, reference_design, monkeypatch):
    def fake_generate(assembly, output_directory, quality):
        output = Path(output_directory)
        output.mkdir(parents=True)
        (output / "compressor.stl").write_bytes(b"mesh")
        (output / "mesh_diagnostics.json").write_text(
            json.dumps({"passed": True, "quality": quality}), encoding="utf-8"
        )
        return SimpleNamespace(passed=True, mesh_profile_version="test")

    monkeypatch.setattr(
        "src.visualization.geometry19_service.generate_geometry19_meshes", fake_generate
    )
    monkeypatch.setattr(
        "src.visualization.geometry19_service.solve_design_point",
        lambda *args, **kwargs: pytest.fail("mesh publication ran physics"),
    )
    payload = Geometry19DesignStore(REPO_ROOT, tmp_path).publish_mesh(
        reference_design, "export_mesh"
    )
    assert payload["mesh_quality"] == "export_mesh"
    assert payload["mesh_diagnostics"]["passed"] is True
    assert len(payload["manifest_sha256"]) == 64


@pytest.fixture(scope="module")
def reference_design() -> Geometry19:
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Geometry19.from_mapping(
        {item["name"]: item["reference_value"] for item in schema["design_variables"]}
    )


def test_store_publishes_hash_isolated_designs_and_reuses_identical_mesh(
    tmp_path, reference_design
):
    store = Geometry19DesignStore(REPO_ROOT, tmp_path)
    first = store.regenerate(reference_design)
    first_id = first["design_id"]
    assert len(first_id) == 64
    assert first["mesh_diagnostics"]["passed"] is True
    assert first["simulation"]["source"] == "geometry19_closed_v2_1"
    assert first["overlay_parameters"]["shaft_speed_rpm"] == reference_design.shaft_speed_rpm
    assert first["simulation"]["thermal"]["component_temps"]["turbine"]["wall_temp"]
    assert first["simulation"]["stress"]["component_stress"]["compressor"]["total_MPa"]
    assert (tmp_path / first_id / "preview_mesh" / "mesh_diagnostics.json").is_file()
    assert (tmp_path / first_id / "preview_mesh" / "manifest.json").is_file()
    component = tmp_path / first_id / "preview_mesh" / "compressor.stl"
    first_mtime = component.stat().st_mtime_ns

    repeated = store.regenerate(reference_design)
    assert repeated["design_id"] == first_id
    assert component.stat().st_mtime_ns == first_mtime

    changed = replace(reference_design, lattice_relative_density=0.4)
    second = store.regenerate(changed)
    second_id = second["design_id"]
    assert second_id != first_id
    assert second["geometry_hash"] == first["geometry_hash"]
    assert second["lattice_hash"] != first["lattice_hash"]
    assert (tmp_path / second_id / "preview_mesh").is_dir()
    assert not list(tmp_path.glob(".*.*"))


def test_geometry19_api_schema_solve_lookup_components_and_file(
    tmp_path, reference_design
):
    store = Geometry19DesignStore(REPO_ROOT, tmp_path)
    generated = store.regenerate(reference_design)
    design_id = generated["design_id"]
    app.config.update(TESTING=True, GEOMETRY19_STORE_ROOT=str(tmp_path))
    client = app.test_client()

    schema_response = client.get("/api/geometry19/schema")
    assert schema_response.status_code == 200
    assert len(schema_response.get_json()["canonical_names"]) == 19

    solve_response = client.post(
        "/api/geometry19/solve", json=reference_design.as_mapping()
    )
    assert solve_response.status_code == 200
    solve_payload = solve_response.get_json()
    assert solve_payload["solver"]["converged"] is True
    assert solve_payload["constraints"]["numerically_valid"] is True
    assert solve_payload["physics"]["performance"]["mass_kg"] > 0.0
    assert len(solve_payload["schema_hash"]) == 64

    design_response = client.get(f"/api/geometry19/design/{design_id}")
    assert design_response.status_code == 200
    assert design_response.get_json()["design_id"] == design_id

    components_response = client.get(
        f"/api/geometry19/components/{design_id}?quality=preview_mesh"
    )
    assert components_response.status_code == 200
    components = components_response.get_json()["components"]
    assert "compressor" in components
    assert "?h=" in components["compressor"]["url"]
    file_response = client.get(components["compressor"]["url"])
    assert file_response.status_code == 200
    assert len(file_response.data) == components["compressor"]["size_bytes"]


def test_completed_pipeline_result_api_is_read_only_and_hash_verified(
    tmp_path, reference_design, monkeypatch
):
    run = tmp_path / "run"
    store = tmp_path / "store"
    design_id = "a" * 64
    source_hash = "d" * 64
    quality = store / design_id / "preview_mesh"
    quality.mkdir(parents=True)
    component = quality / "compressor.stl"
    component.write_bytes(b"pipeline mesh")
    component_hash = hashlib.sha256(component.read_bytes()).hexdigest()
    manifest = {
        "design_id": design_id,
        "solid_geometry_hash": "1" * 64,
        "lattice_hash": "2" * 64,
        "calibration_hash": "3" * 64,
        "components": {
            "compressor": {
                "filename": component.name,
                "sha256": component_hash,
                "size_bytes": component.stat().st_size,
            }
        },
    }
    manifest_path = quality / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (quality / "mesh_diagnostics.json").write_text(
        json.dumps({"passed": True, "quality": "preview_mesh"}), encoding="utf-8"
    )
    design_root = store / design_id
    (design_root / "geometry19.json").write_text(
        json.dumps({"geometry19": reference_design.as_mapping()}), encoding="utf-8"
    )
    ledger_path = run / "core" / "F" / "ledger.jsonl"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(json.dumps({"evaluation": {
        "performance": {"thrust_N": 60.0, "tsfc_kg_N_s": 6.7e-5, "mass_kg": 1.7},
        "solver_converged": True,
        "solver_iterations": 4,
        "conservation": {"normalized_flow": 1e-9},
        "constraint_assessment": {"accepted": True, "violated_constraints": []},
    }}) + "\n", encoding="utf-8")
    delivered_geometry_path = run / "core" / "runtime" / "Delivery_Geometry.json"
    delivered_geometry_path.parent.mkdir(parents=True)
    delivered_geometry_path.write_text(
        json.dumps(reference_design.as_mapping()), encoding="utf-8"
    )
    summary = {
        "model": "model V2.2",
        "status": "V2_2_PIPELINE_COMPLETED",
        "formal_use_allowed": False,
        "target_id": "target",
        "paths": {"design_storage": str(store)},
        "mesh_assets": {"preview_mesh": {
            "source_design_hash": source_hash,
            "design_id": design_id,
            "manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }},
    }
    summary_path = run / "Pipeline_Summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    files = {
        str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (summary_path, ledger_path, delivered_geometry_path)
    }
    (run / "Complete_Manifest.json").write_text(
        json.dumps({"files": files}), encoding="utf-8"
    )
    monkeypatch.setitem(app.config, "GEOMETRY19_PIPELINE_RESULT_ROOT", str(run))
    client = app.test_client()
    response = client.get("/api/geometry19/pipeline-result?quality=preview_mesh")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["source_design_hash"] == source_hash
    assert payload["design_id"] == design_id
    assert payload["formal_use_allowed"] is False
    mesh_response = client.get(payload["component_urls"]["compressor"])
    assert mesh_response.status_code == 200
    assert mesh_response.data == b"pipeline mesh"

    component.write_bytes(b"tampered")
    rejected = client.get("/api/geometry19/pipeline-result?quality=preview_mesh")
    assert rejected.status_code == 409
    assert "integrity" in rejected.get_json()["error"]


def test_geometry19_api_rejects_unknown_and_incomplete_fields(reference_design):
    app.config.update(TESTING=True)
    client = app.test_client()
    extra = {**reference_design.as_mapping(), "unexpected": 1}
    response = client.post("/api/geometry19/solve", json=extra)
    assert response.status_code == 400
    assert "Unknown request fields" in response.get_json()["error"]

    response = client.post(
        "/api/geometry19/solve",
        json={"compressor_impeller_exit_diameter_m": 0.08},
    )
    assert response.status_code == 400
    assert "missing=" in response.get_json()["error"]


def test_concurrent_design_publication_is_isolated(
    tmp_path, reference_design, monkeypatch
):
    def fake_generate(assembly, output_directory, quality):
        output = Path(output_directory)
        output.mkdir(parents=True)
        (output / "compressor.stl").write_bytes(
            assembly.design_hash.encode("ascii")
        )
        (output / "mesh_diagnostics.json").write_text(
            json.dumps({"passed": True, "quality": quality}),
            encoding="utf-8",
        )
        return SimpleNamespace(passed=True)

    monkeypatch.setattr(
        "src.visualization.geometry19_service.generate_geometry19_meshes",
        fake_generate,
    )
    store = Geometry19DesignStore(REPO_ROOT, tmp_path)
    designs = (
        reference_design,
        replace(reference_design, compressor_impeller_exit_diameter_m=0.09),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(store.regenerate, designs))

    ids = {response["design_id"] for response in responses}
    assert len(ids) == 2
    for design_id in ids:
        component = tmp_path / design_id / "preview_mesh" / "compressor.stl"
        assert component.read_text(encoding="ascii") == design_id
    assert not list(tmp_path.glob(".*.*"))
