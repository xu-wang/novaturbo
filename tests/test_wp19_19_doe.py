"""Small, real-physics tests for the resumable WP19-19 pipeline."""

from pathlib import Path

import yaml

from src.physics.wp19_19_doe import _status_stratified_ids, run_wp19_19


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / "config/physics/wp19_19_permissive_doe_v1.yaml"


def test_status_stratified_selection_is_deterministic():
    records = [
        {
            "sample_id": index,
            "classification": "A" if index % 2 else "B",
            "geometry_mapping_valid": True,
        }
        for index in range(20)
    ]
    assert _status_stratified_ids(records, 8) == _status_stratified_ids(records, 8)
    assert len(_status_stratified_ids(records, 8)) == 8


def test_physics_only_smoke_is_resumable(tmp_path):
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    protocol["artifacts"] = {
        "runtime_root": str(tmp_path / "runtime"),
        "mesh_storage_root": str(tmp_path / "meshes"),
        "summary_json": str(tmp_path / "summary.json"),
        "report_markdown": str(tmp_path / "report.md"),
    }
    protocol_path = tmp_path / "wp19_19_test.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    first = run_wp19_19(
        protocol_path,
        REPO_ROOT,
        sample_count_override=8,
        preview_budget_override=2,
        export_budget_override=1,
        physics_only=True,
    )
    second = run_wp19_19(
        protocol_path,
        REPO_ROOT,
        sample_count_override=8,
        preview_budget_override=2,
        export_budget_override=1,
        physics_only=True,
    )
    assert first["status"] == "PHYSICS_COMPLETE_MESH_PENDING"
    assert first["summary"] == second["summary"]
    ledger = tmp_path / "runtime/results/samples.jsonl"
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 8
    assert (tmp_path / "runtime/config_snapshot").is_dir()
