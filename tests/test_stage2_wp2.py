from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from src.ai.stage2_wp2 import (
    load_stage2_wp2_config,
    prepare_stage2_wp2,
    validate_training_configs,
)
from src.ai.wp3 import load_wp3_config, run_upstream_smoke


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_CONFIG = REPO_ROOT / "config" / "stage2" / "s2_wp2.yaml"


def _configs():
    stage = load_stage2_wp2_config(STAGE_CONFIG)
    reference = load_wp3_config(REPO_ROOT / stage["reference_training_config"])
    smoke = load_wp3_config(REPO_ROOT / stage["smoke_training_config"])
    full = load_wp3_config(REPO_ROOT / stage["full_training_config"])
    return stage, reference, smoke, full


def test_stage2_wp2_full_config_only_adds_bounded_output():
    stage, reference, smoke, full = _configs()

    validate_training_configs(
        reference, smoke, full, stage["bounded_contract"]
    )

    for model in full["models"].values():
        assert model["bounded_output"]["enabled"] is True
        assert model["bounded_output"]["epsilon"] == 1.0e-6


def test_stage2_wp2_rejects_hidden_architecture_change():
    stage, reference, smoke, full = _configs()
    changed = copy.deepcopy(full)
    changed["models"]["nflows"]["num_layers"] += 1

    with pytest.raises(ValueError, match="architecture differs"):
        validate_training_configs(
            reference, smoke, changed, stage["bounded_contract"]
        )


def test_bounded_upstream_smoke_passes_all_adapters():
    _, _, smoke, _ = _configs()
    result = run_upstream_smoke(smoke)

    assert result["status"].tolist() == ["passed"] * 4


def test_stage2_wp2_prepare_verifies_baseline_and_writes_manifest(tmp_path):
    output = prepare_stage2_wp2(
        STAGE_CONFIG,
        repo_root=REPO_ROOT,
        output_dir_override=tmp_path / "stage2_wp2",
    )
    manifest = yaml.safe_load((output / "manifest.yaml").read_text(encoding="utf-8"))

    assert manifest["status"] == "prepared"
    assert manifest["baseline_id"] == "active10d_baseline_v1"
    assert manifest["bounded_contract"]["kind"] == "atanh_tanh"
