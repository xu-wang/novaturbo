"""Read-only V3 diagnosis, preview provenance and fail-closed file routes."""
from pathlib import Path
import json
import shutil
from types import SimpleNamespace

import pytest

from src.ai.geometry22_pipeline_v3_routed import read, sha, state_files, write
from src.visualization import geometry22_results as v
from src.visualization import geometry22_mesh as mesh
from ui import server

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data/validation/v3_22d_final_holdout_v1/cases"


@pytest.fixture
def result_service(tmp_path, monkeypatch):
    source = sorted(CASES.iterdir())[0]
    results = tmp_path / "results"
    run = results / source.name / "run"
    shutil.copytree(source / "run", run)
    summary = read(run / "Pipeline_Summary.json")
    summary["paths"] = {"runtime_root": str(run), "pipeline_manifest": str(run / "Complete_Manifest.json")}
    write(run / "Pipeline_Summary.json", summary)
    state = read(run / "Run_State.json")
    state["files"] = state_files(run)
    write(run / "Run_State.json", state)
    complete = read(run / "Complete_Manifest.json")
    complete.update(files=state["files"], run_state_sha256=sha(run / "Run_State.json"))
    write(run / "Complete_Manifest.json", complete)
    previews = tmp_path / "previews"
    monkeypatch.setitem(server.app.config, "GEOMETRY22_RESULTS_ROOT", str(results))
    monkeypatch.setitem(server.app.config, "GEOMETRY22_PREVIEW_ROOT", str(previews))
    return v.Geometry22Results(ROOT, results, previews), source.name, run


def test_no_hit_result_exposes_verified_full_22d_without_writes(result_service):
    service, result_id, run = result_service
    before = {str(p): sha(p) for p in run.rglob("*") if p.is_file()}
    data = service.result(result_id)
    assert data["delivery_status"] == "NO_HIT" and data["software_hit_5pct"]
    assert len(data["geometry22"]) == 22 and len(data["component_parameters"]) == 3
    assert data["f_recompute_equal"] and not data["selected"]["feasible"]
    assert data["preview_available"] and data["mesh_scope"] == "22d_design_driven_visual_mesh"
    assert before == {str(p): sha(p) for p in run.rglob("*") if p.is_file()}
    assert not service.previews.exists()
    response = server.app.test_client().get(f"/api/geometry22/results/{result_id}")
    assert response.status_code == 200 and response.get_json()["design_id"] == data["design_id"]


def fake_mesh_store(monkeypatch, tmp_path):
    def fake_generate(assembly, directory, quality):
        assert quality == "preview_mesh"
        assert len(assembly.geometry_state.design.as_mapping()) == 19
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "inlet.stl").write_bytes(b"synthetic STL test artifact")
        return SimpleNamespace(passed=True, as_mapping=lambda: {"passed": True})
    monkeypatch.setattr(mesh, "generate_geometry19_meshes", fake_generate)


def test_preview_separate_identity_file_integrity_and_source_binding(result_service, tmp_path, monkeypatch):
    service, result_id, run = result_service
    fake_mesh_store(monkeypatch, tmp_path)
    before = sha(run / "Complete_Manifest.json")
    client = server.app.test_client()
    data = client.post(f"/api/geometry22/results/{result_id}/preview").get_json()
    assert data["mesh_id"] != data["design_id"]
    url = data["component_urls"]["inlet"]
    assert client.get(url).status_code == 200
    assert client.get(url.replace("inlet.stl", "unknown.stl")).status_code == 404
    assert sha(run / "Complete_Manifest.json") == before
    artifact = service.previews / "mesh22" / data["preview_id"] / "inlet.stl"
    artifact.write_bytes(b"tampered")
    assert client.get(url).status_code == 409
    assert client.post(f"/api/geometry22/results/{result_id}/preview").status_code == 409


def test_source_tamper_blocks_results_and_cached_mesh(result_service, tmp_path, monkeypatch):
    service, result_id, run = result_service
    fake_mesh_store(monkeypatch, tmp_path)
    data, _, _ = service.preview(result_id, generate=True)
    (run / "F/summary.json").write_text("{}")
    client = server.app.test_client()
    assert client.get(f"/api/geometry22/results/{result_id}").status_code == 409
    assert client.get(data["component_urls"]["inlet"]).status_code == 409
    listed = client.get("/api/geometry22/results").get_json()["results"]
    assert listed[0]["integrity_pass"] is False


def test_no_selected_design_is_visible_and_cannot_preview(result_service, monkeypatch):
    service, result_id, _ = result_service
    monkeypatch.setattr(service, "result", lambda value: {"preview_available": False})
    with pytest.raises(v.Geometry22ResultError, match="no selected"):
        service.preview(result_id, generate=True)
    assert not service.previews.exists()


def test_result_id_and_configured_root_guards(result_service):
    service, result_id, _ = result_service
    with pytest.raises(v.Geometry22ResultError, match="Invalid"):
        service.result("../" + result_id)
    assert server.app.test_client().get("/api/geometry22/results/unknown").status_code == 409
    assert server.app.test_client().get("/api/geometry22/results/" + "0" * 64).status_code == 404


def test_preview_manifest_tamper_rejected(result_service, tmp_path, monkeypatch):
    service, result_id, _ = result_service
    fake_mesh_store(monkeypatch, tmp_path)
    payload, directory, _ = service.preview(result_id, generate=True)
    manifest = read(directory / "manifest.json")
    manifest["mesh_diagnostics"]["passed"] = False
    write(directory / "manifest.json", manifest)
    with pytest.raises(mesh.Geometry22MeshError, match="artifact changed"):
        service.preview(result_id, expected_preview_id=payload["preview_id"])
