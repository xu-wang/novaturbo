"""V3 L6b gate source isolation, GPU identity and model selection checks."""

import json
from pathlib import Path
import shutil

import pytest

from src.ai import v3_target_gate_l6b as gate
from src.ai import v3_target_route_protocol as route


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/validation/v3_22d_target_gate_l6b_gpu6_v1"


def test_gpu_gate_and_local_forward_outperform_simple_comparators():
    fit = gate.verify_v3_target_gate_fit(ROOT, OUTPUT)
    selection = json.loads((OUTPUT / "gate_selection.json").read_text())
    local = json.loads((OUTPUT / "local_forward_report.json").read_text())
    chosen = next(item for item in fit["neural_models"] if item["model_id"] == fit["selected_neural_model_id"])
    nearest = selection["baseline_comparison"]["nearest_frozen_center_target_support"]
    assert fit["gpu"]["visible_devices"] == "6"
    assert fit["gpu"]["logical_device"] == "cuda:0"
    assert fit["gpu"]["name"] == "NVIDIA GeForce RTX 4090"
    assert fit["fit_rows"] == 8992 and fit["tuning_rows"] == 2297
    assert chosen["tuning"]["macro_top1"] > nearest["macro_top1"]
    assert chosen["tuning"]["macro_top2"] > nearest["macro_top2"]
    assert local["selected_model_id"] == "ridge_0_01"
    assert local["metrics"]["ridge_0_01"]["worst_center_p95_max_abs_log_error"] < local["metrics"]["center_constant"]["worst_center_p95_max_abs_log_error"]
    assert fit["old_validation_and_holdout_content_parsed"] is False
    assert fit["physics_evaluations_executed"] == 0
    assert fit["formal_use_allowed"] is False


def test_fit_verification_does_not_parse_old_validation_or_holdout(monkeypatch):
    original = route._rows

    def guarded(path):
        if "validation_candidates" in str(path) or "holdout_candidates" in str(path):
            raise AssertionError("Old validation/holdout labels must stay unread in L6b")
        return original(path)

    monkeypatch.setattr(route, "_rows", guarded)
    monkeypatch.setattr(gate, "_rows", guarded)
    gate.verify_v3_target_gate_fit(ROOT, OUTPUT)


def test_gpu_mapping_and_artifact_tampering_are_rejected(monkeypatch, tmp_path):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(gate.V3TargetGateError, match="CUDA_VISIBLE_DEVICES=6"):
        gate._device()
    copy = tmp_path / "fit"
    shutil.copytree(OUTPUT, copy)
    selection = copy / "gate_selection.json"
    value = json.loads(selection.read_text())
    value["selected_neural_model_id"] = "gate_seed_42"
    selection.write_text(json.dumps(value))
    with pytest.raises(gate.V3TargetGateError, match="fit or model selection changed"):
        gate.verify_v3_target_gate_fit(ROOT, copy)
