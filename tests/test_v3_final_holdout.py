"""Holdout guards use synthetic labels only; real holdout physics stays sealed until commit."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from src.ai import v3_final_holdout as h

ROOT = Path(__file__).resolve().parents[1]


def test_split_and_training_center_leakage():
    cfg = h.config_at(ROOT)
    registry = h.rows(ROOT / cfg["registry_path"])
    support = h.source(ROOT)[3]
    selected = h.coordinates(registry, support)
    assert len(selected) == 7 and all(t["split"] == "holdout" for t in selected)
    broken = deepcopy(registry)
    broken[0]["root_id"] = broken[-1]["root_id"]
    with pytest.raises(ValueError, match="split leakage"):
        h.coordinates(broken, support)
    with pytest.raises(ValueError, match="training support"):
        h.coordinates(registry, support + [{"center_id": selected[0]["target_source_design_id"]}])


def test_failures_and_abstention_stay_in_denominator():
    cases = [{"source_design_id": str(i), "status": "EVALUATED" if i < 5 else "ABSTAINED" if i == 5 else "TARGET_LABEL_FAILED",
              "software_hit_5pct": i < 5, "domain_valid_software_hit_5pct": i < 4,
              "delivered_hit": False, "physics_evaluations": 25 if i < 5 else 0} for i in range(7)]
    result = h.aggregate(cases)
    assert result["total_target_denominator"] == 7 and result["software_hit_targets_5pct"] == 5
    assert result["status_counts"] == {"ABSTAINED": 1, "EVALUATED": 5, "TARGET_LABEL_FAILED": 1}
    with pytest.raises(ValueError, match="remain seven"):
        h.aggregate(cases[:5])


@pytest.fixture
def synthetic_lock(tmp_path, monkeypatch):
    cfg = h.config_at(ROOT)
    cfg["output_root"] = str(tmp_path / "check")
    runtime = SimpleNamespace(contract_mapping=lambda: {"synthetic_test_runtime": True})
    # Freeze may inspect contracts/coordinates, never evaluate target designs.
    runtime.evaluate = lambda *a: pytest.fail("Target labels must remain unopened during freeze")
    monkeypatch.setattr(h, "build_v3_development_physics_runtime", lambda root: runtime)
    monkeypatch.setattr(h, "assess_constraint_profile", lambda *a, **kw: SimpleNamespace(as_mapping=lambda: {"numerically_valid": True, "physics_domain_valid": True}))
    h.freeze(ROOT, cfg)
    return cfg, runtime


def test_frozen_code_change_rejected(synthetic_lock, monkeypatch):
    cfg, _ = synthetic_lock
    original = h.sha
    monkeypatch.setattr(h, "sha", lambda path: "0" * 64 if str(path).endswith(h.CODE) else original(path))
    with pytest.raises(ValueError, match="frozen model, code"):
        h.verify_lock(ROOT, cfg)


def failed_labels(cfg, runtime):
    runtime.base_runtime = SimpleNamespace(conservation_tolerances={})
    runtime.evaluate = lambda design: SimpleNamespace(design_id=design.design_id(), engine_result=SimpleNamespace(
        solver=SimpleNamespace(converged=False, failure_code=SimpleNamespace(value="synthetic_failure")), feasible=False))
    return h.label(ROOT, cfg)


def test_label_opening_once_and_tamper_rejected(synthetic_lock):
    cfg, runtime = synthetic_lock
    manifest = failed_labels(cfg, runtime)
    assert manifest["total_targets"] == 7 and manifest["usable_targets"] == 0
    assert len(h.verify_labels(ROOT, cfg)) == 7
    with pytest.raises(ValueError, match="already opened"):
        h.label(ROOT, cfg)
    path = Path(cfg["output_root"]) / "target_labels.json"
    labels = h.read(path)
    labels[0]["split"] = "development"
    path.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match="labels changed"):
        h.verify_labels(ROOT, cfg)


def test_complete_verification_rebuilds_metrics_without_cli(synthetic_lock, monkeypatch):
    cfg, runtime = synthetic_lock
    failed_labels(cfg, runtime)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    monkeypatch.setattr(h.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(h.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(h.torch.cuda, "get_device_name", lambda *a: "synthetic_test_device")
    monkeypatch.setattr(h.subprocess, "run", lambda *a, **kw: pytest.fail("Failed labels cannot call CLI"))
    report = h.evaluate(ROOT, cfg)
    assert report["metrics"]["total_target_denominator"] == 7
    assert report["metrics"]["status_counts"] == {"TARGET_LABEL_FAILED": 7}
    assert h.verify_complete(ROOT, cfg) == report
    output = Path(cfg["output_root"])
    report["metrics"]["software_hit_targets_5pct"] = 7
    (output / "report.json").write_text(json.dumps(report))
    manifest = h.read(output / "complete_manifest.json")
    manifest["report_sha256"] = h.sha(output / "report.json")
    (output / "complete_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="result summary changed"):
        h.verify_complete(ROOT, cfg)


def test_label_execution_exception_preserves_seven(synthetic_lock):
    cfg, runtime = synthetic_lock
    def fail(design):
        raise RuntimeError("synthetic backend execution exception")
    runtime.evaluate = fail
    manifest = h.label(ROOT, cfg)
    assert manifest["total_targets"] == 7 and manifest["usable_targets"] == 0
    labels = h.verify_labels(ROOT, cfg)
    assert all(t["failure_code"] == "label_execution_error" for t in labels)
