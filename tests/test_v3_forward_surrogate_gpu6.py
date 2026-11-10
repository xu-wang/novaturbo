"""GPU 6 V3 surrogate artifact and device identity checks."""

import json
from pathlib import Path

import pytest

from src.ai.v3_forward_surrogate_gpu6 import (
    V3ForwardSurrogateError,
    _gpu_device,
    verify_v3_forward_complete,
    verify_v3_forward_fit,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/validation/v3_22d_forward_surrogate_gpu6_v1"


def test_gpu6_training_identity_and_selection():
    fit = verify_v3_forward_fit(ROOT, OUTPUT)
    complete = verify_v3_forward_complete(ROOT, OUTPUT)
    report = json.loads((OUTPUT / "holdout_report.json").read_text())
    assert fit["training_device"] == "cuda:0"
    assert fit["physical_gpu_index"] == 6
    assert fit["visible_gpu_name"] == "NVIDIA GeForce RTX 4090"
    assert {asset["model_id"] for asset in fit["models"] if asset["kind"] == "mlp"} == {
        "mlp_seed_42", "mlp_seed_43"
    }
    assert all(asset["checkpoint_reload_equal"] for asset in fit["models"])
    assert complete["selected_model_id"] == fit["selected_model_id"] == "constant"
    assert report["model_selection_used_holdout"] is False
    assert complete["formal_use_allowed"] is False


def test_gpu6_mapping_must_be_explicit(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(V3ForwardSurrogateError, match="CUDA_VISIBLE_DEVICES=6"):
        _gpu_device()
