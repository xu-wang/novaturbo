"""R5 software gates for the independent V3 development B--F path."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.ai import geometry19_pipeline_v3 as v3
from src.ai.v3_development_assets import load_v3_fixture_assets
from src.physics.v3_development_runtime import build_v3_development_physics_runtime


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "data/validation/v3_22d_software_fixture_v1"


def _reference_target():
    runtime = build_v3_development_physics_runtime(ROOT)
    assets = load_v3_fixture_assets(ASSET_ROOT, runtime)
    return assets.rows[0]["performance"]


def test_real_v3_bf_run_has_22d_identity_and_strict_resume(tmp_path):
    target = _reference_target()
    output = tmp_path / "run"
    summary = v3.run_wp19_23_pipeline_v3(ROOT, target_values=target, output_root=output)
    assert summary["status"] == "V3_DEVELOPMENT_NO_DELIVERED_HIT"
    assert summary["candidate_count"] == 45
    assert summary["physics_evaluation_count"] == 46
    assert summary["design_dimension"] == 22
    assert summary["formal_use_allowed"] is False
    assert summary["deployment_allowed"] is False
    assert json.loads((output / "B/candidates.json").read_text())[0]["forward_fixture_label_available"]
    assert sum(json.loads((output / stage / "summary.json").read_text())["evaluated_count"] for stage in "CDE") == 45
    f = json.loads((output / "F/summary.json").read_text())
    assert f["status"] == "NO_HIT"
    assert f["independent_recompute"]["design_schema_hash"] == summary["design_schema_hash"]
    assert v3.run_wp19_23_pipeline_v3(ROOT, target_values=target, output_root=output, resume=True) == summary
    with pytest.raises(v3.Geometry19V3PendingError, match="overwrite"):
        v3.run_wp19_23_pipeline_v3(ROOT, target_values=target, output_root=output)
    with pytest.raises(v3.Geometry19V3PendingError, match="identity changed"):
        v3.run_wp19_23_pipeline_v3(
            ROOT, target_values={**target, "thrust_N": target["thrust_N"] * 1.01},
            output_root=output, resume=True,
        )
    f_path = output / "F/summary.json"
    original_f = f_path.read_bytes()
    f_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(v3.Geometry19V3PendingError, match="artifact changed"):
        v3.run_wp19_23_pipeline_v3(ROOT, target_values=target, output_root=output, resume=True)
    f_path.write_bytes(original_f)

    # Rehashing an altered stage file cannot make a mismatched 22D identity valid.
    stage_path = output / "C/evaluations.json"
    rows = json.loads(stage_path.read_text())
    rows[0]["design_id"] = "0" * 64
    stage_path.write_text(json.dumps(rows), encoding="utf-8")
    manifest_path = output / "Complete_Manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["C/evaluations.json"] = hashlib.sha256(stage_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(v3.Geometry19V3PendingError, match="V3 C design identity changed"):
        v3.run_wp19_23_pipeline_v3(ROOT, target_values=target, output_root=output, resume=True)


def test_v3_preflight_rejects_old_options_before_loading_assets(tmp_path):
    with pytest.raises(v3.Geometry19V3PendingError, match="--model-variant"):
        v3.run_wp19_23_pipeline_v3(tmp_path, target_values={}, model_variant="baseline")
    with pytest.raises(v3.Geometry19V3PendingError, match="CPU only"):
        v3.run_wp19_23_pipeline_v3(tmp_path, target_values={}, device="cuda:0")


def test_v3_execution_failure_has_preserved_terminal_artifact(monkeypatch, tmp_path):
    target = _reference_target()
    output = tmp_path / "failed"

    def fail_evaluation(*args):
        raise RuntimeError("injected V3 evaluator failure")

    monkeypatch.setattr(v3, "_evaluate", fail_evaluation)
    with pytest.raises(v3.Geometry19V3PendingError, match="development run failed"):
        v3.run_wp19_23_pipeline_v3(ROOT, target_values=target, output_root=output)
    failure = json.loads((output / "Failure_Summary.json").read_text())
    assert failure["status"] == "V3_DEVELOPMENT_FAILED"
    assert failure["error_type"] == "RuntimeError"
    assert (output / "Failure_Manifest.json").is_file()
    with pytest.raises(v3.Geometry19V3PendingError, match="Failed model V3 run"):
        v3.run_wp19_23_pipeline_v3(ROOT, target_values=target, output_root=output, resume=True)


def test_injected_bf_success_and_no_candidate_terminals(monkeypatch, tmp_path):
    runtime = build_v3_development_physics_runtime(ROOT)
    assets = load_v3_fixture_assets(ASSET_ROOT, runtime)
    target = _reference_target()
    config = yaml.safe_load((ROOT / v3.CONFIG).read_text())
    target_doc = v3._load_target_v3(None, target, "ID", "UNKNOWN")
    reference_id = assets.rows[0]["design_id"]

    def fake_evaluate(design, current_runtime, requested, tolerances):
        is_reference = design.design_id() == reference_id
        values = target if is_reference else {name: value * 2 for name, value in target.items()}
        return {
            "design_id": design.design_id(), "design": design.as_mapping(),
            "design_schema_hash": current_runtime.design_schema_hash,
            "physics_bundle_hash": current_runtime.physics_bundle_hash,
            "physics_model_version": current_runtime.model_version,
            "solver_converged": True, "solver_failure_code": None,
            "feasible": is_reference, "performance": values,
            "rho": v3._rho(values, requested, tolerances),
            "physics_validation_status": "not_completed",
            "formal_use_allowed": False, "deployment_allowed": False,
        }

    monkeypatch.setattr(v3, "_evaluate", fake_evaluate)
    monkeypatch.setattr(v3, "build_v3_development_physics_runtime", lambda root: runtime)
    success = v3._execute(tmp_path / "success", {}, runtime, assets, target_doc, config)
    assert success["delivery_status"] == "PASS"
    assert success["delivered_design_ids"] == [reference_id]
    assert json.loads((tmp_path / "success/F/summary.json").read_text())["independent_recompute"]["design_id"] == reference_id

    empty = SimpleNamespace(rows=[], candidate_designs=lambda: (), manifest=assets.manifest)
    no_candidates = v3._execute(tmp_path / "empty", {}, runtime, empty, target_doc, config)
    assert no_candidates["candidate_count"] == 0
    assert no_candidates["physics_evaluation_count"] == 0
    assert json.loads((tmp_path / "empty/F/summary.json").read_text())["status"] == "NO_CANDIDATES"
