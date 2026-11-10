"""Lifecycle, provenance and compatibility gates for the trained V3 CLI profile."""
from pathlib import Path
import json

import pytest

from src.ai import geometry22_pipeline_v3_cfm as p
from src.ai.geometry19_pipeline_dispatch import run_wp19_23_pipeline
from src.ai.geometry19_pipeline_artifact import verify_completed_artifact

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "data/validation/v3_22d_fresh_validation_l6h_v1"


@pytest.fixture
def fake_flow(monkeypatch):
    target = json.loads((CHECK / "target_labels.jsonl").read_text().splitlines()[0])["target_performance"]
    raw = [json.loads(s) for s in (CHECK / "raw_candidates.jsonl").read_text().splitlines()]
    original_id = raw[0]["target_id"]
    raw = [r for r in raw if r["target_id"] == original_id and r["arm"] == "gated_top2"]
    def generate(root, requested, cfg, local, support, norm, checkpoint):
        rows = [{**r, "target_id": requested["target_id"], "generator":"v3_local_cfm_v1", "generator_checkpoint_sha256":cfg["sources"]["cfm_checkpoint_sha256"]} for r in raw]
        ranked = sorted(rows, key=lambda r: (r["predicted_rho"], r["design_id"], r["raw_slot"]))[:24]
        selected = [{**r, "rank": i} for i, r in enumerate(ranked)]
        route = {"target_id": requested["target_id"], "status": "routable", "eligible_center_count": 2,
                 "arms": {"gated_top2": sorted({r["route_center_id"] for r in rows})}}
        return selected, rows, route
    def evaluate(row, runtime, profile, requested, cfg):
        return {"design_id": row["design_id"], "design": row["design"], "route_center_id": row["route_center_id"],
                "design_schema_hash": runtime.design_schema_hash, "physics_bundle_hash": runtime.physics_bundle_hash,
                "solver_converged": True, "solver_failure_code": None, "diagnostic_domain_valid": True,
                "feasible": False, "performance": target, "rho": p._rho(target, requested, cfg["relative_tolerances"]),
                "software_hit_5pct": True, "formal_use_allowed": False, "deployment_allowed": False}
    monkeypatch.setattr(p, "generate", generate)
    monkeypatch.setattr(p, "evaluate_design", evaluate)
    return target, evaluate


def test_dispatch_complete_resume_and_rehashed_identity_tamper(fake_flow, tmp_path):
    target, _ = fake_flow
    out = tmp_path / "run"
    args = dict(model="model V3", pipeline_profile=p.PROFILE, device="cuda:0", target_values=target, output_root=out)
    summary = run_wp19_23_pipeline(ROOT, **args)
    assert summary["candidate_count"] == 24 and summary["physics_evaluation_count"] == 25
    assert summary["software_hit_5pct"] and summary["delivery_status"] == "NO_HIT"
    assert run_wp19_23_pipeline(ROOT, **args, resume=True) == summary
    report = verify_completed_artifact(out, ROOT)
    assert report["integrity_pass"] and report["summary"] == summary
    archived = verify_completed_artifact(out, tmp_path / "different_code_checkout")
    assert archived["integrity_pass"] and not archived["current_code_identity_matches_lock"]
    with pytest.raises(p.Geometry19V3PendingError, match="overwrite"):
        run_wp19_23_pipeline(ROOT, **args)
    with pytest.raises(p.Geometry19V3PendingError, match="identity changed"):
        p.run_v3_routed_pipeline(ROOT, target_values={**target, "mass_kg": target["mass_kg"] * 1.01}, output_root=out, resume=True)
    path = out / "C/evaluations.json"
    rows = p.read(path)
    rows[0]["design_id"] = "0" * 64
    p.write(path, rows)
    state = p.read(out / "Run_State.json")
    state["files"] = p.state_files(out)
    p.write(out / "Run_State.json", state)
    p.write(out / "Complete_Manifest.json", {"manifest_version": p.VERSION, "stages": p.STAGES,
            "files": state["files"], "run_state_sha256": p.sha(out / "Run_State.json")})
    with pytest.raises(p.Geometry19V3PendingError, match="evaluation design changed"):
        run_wp19_23_pipeline(ROOT, **args, resume=True)


def test_stage_boundary_resume_reuses_completed_physics(fake_flow, monkeypatch, tmp_path):
    target, evaluate = fake_flow
    count = 0
    def interrupt(row, *args):
        nonlocal count
        count += 1
        if count == 9:
            raise KeyboardInterrupt("simulated interruption before D")
        return evaluate(row, *args)
    monkeypatch.setattr(p, "evaluate_design", interrupt)
    out = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        p.run_v3_routed_pipeline(ROOT, target_values=target, output_root=out)
    assert p.read(out / "Run_State.json")["completed_stages"] == ["B", "C"]
    before = p.sha(out / "C/evaluations.json")
    monkeypatch.setattr(p, "evaluate_design", evaluate)
    monkeypatch.setattr(p, "generate", lambda *a: pytest.fail("B must not be repeated"))
    summary = p.run_v3_routed_pipeline(ROOT, target_values=target, output_root=out, resume=True)
    assert p.sha(out / "C/evaluations.json") == before
    assert summary["physics_evaluation_count"] == 25


def test_failure_terminal_and_unsealed_partial_refused(fake_flow, monkeypatch, tmp_path):
    target, _ = fake_flow
    def fail(*args):
        raise RuntimeError("injected solver execution error")
    monkeypatch.setattr(p, "evaluate_design", fail)
    out = tmp_path / "failed"
    with pytest.raises(p.Geometry19V3PendingError, match="run failed"):
        p.run_v3_routed_pipeline(ROOT, target_values=target, output_root=out)
    assert p.read(out / "Run_State.json")["status"] == "FAILED"
    assert p.read(out / "Failure_Summary.json")["last_completed_stage"] == "B"
    with pytest.raises(p.Geometry19V3PendingError, match="Failed"):
        p.run_v3_routed_pipeline(ROOT, target_values=target, output_root=out, resume=True)


def test_abstained_target_seals_all_stages_without_physics(fake_flow, monkeypatch, tmp_path):
    target, _ = fake_flow
    monkeypatch.setattr(p, "generate", lambda root, t, *a: ([], [], {"target_id": t["target_id"],
        "status": "abstained_no_target_support", "eligible_center_count": 0, "arms": {"gated_top2": []}}))
    monkeypatch.setattr(p, "evaluate_design", lambda *a: pytest.fail("Abstained target cannot reach physics"))
    out = tmp_path / "abstained"
    summary = p.run_v3_routed_pipeline(ROOT, target_values=target, output_root=out)
    assert summary["route_status"] == "abstained_no_target_support"
    assert summary["candidate_count"] == summary["physics_evaluation_count"] == 0
    assert summary["delivery_status"] == "NO_CANDIDATES"
    assert p.read(out / "Complete_Manifest.json")["stages"] == list("BCDEF")
    assert p.run_v3_routed_pipeline(ROOT, target_values=target, output_root=out, resume=True) == summary


def test_gpu_preflight_and_options(tmp_path):
    with pytest.raises(p.Geometry19V3PendingError, match="requires --device"):
        p.run_v3_routed_pipeline(tmp_path, target_values={}, device="cpu")
    with pytest.raises(p.Geometry19V3PendingError, match="Unsupported"):
        p.run_v3_routed_pipeline(tmp_path, target_values={}, storage_root=tmp_path)


def test_run_lock_blocks_second_writer(tmp_path):
    with p.run_lock(tmp_path):
        with pytest.raises(p.Geometry19V3PendingError, match="already active"):
            with p.run_lock(tmp_path):
                pass


def test_unsealed_stage_cannot_be_resumed(fake_flow, monkeypatch, tmp_path):
    target, _ = fake_flow
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(p, "evaluate_design", interrupt)
    out = tmp_path / "unsealed"
    with pytest.raises(KeyboardInterrupt):
        p.run_v3_routed_pipeline(ROOT, target_values=target, output_root=out)
    (out / "C").mkdir()
    with pytest.raises(p.Geometry19V3PendingError, match="Unsealed"):
        p.run_v3_routed_pipeline(ROOT, target_values=target, output_root=out, resume=True)


def test_f_recompute_mismatch_preserves_failure(fake_flow, monkeypatch, tmp_path):
    target, evaluate = fake_flow
    count = 0
    def mismatch(row, *args):
        nonlocal count
        count += 1
        result = evaluate(row, *args)
        if count == 25:
            result["rho"] = .5
        return result
    monkeypatch.setattr(p, "evaluate_design", mismatch)
    out = tmp_path / "f_mismatch"
    with pytest.raises(p.Geometry19V3PendingError, match="run failed"):
        p.run_v3_routed_pipeline(ROOT, target_values=target, output_root=out)
    assert p.read(out / "Failure_Summary.json")["error"] == "V3 F independent recompute differs"
