from __future__ import annotations

from pathlib import Path

import yaml

from src.ai.stage2_wp2_rejection import load_config, prepare


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config" / "stage2" / "s2_wp2_rejection.yaml"


def test_rejection_config_freezes_original_candidate_budget():
    config = load_config(CONFIG)

    assert config["rejection_contract"]["policy"] == "bounded_rejection"
    assert config["rejection_contract"]["accepted_candidate_count"] == 256
    assert config["rejection_contract"]["max_draw_multiplier"] == 64


def test_rejection_prepare_verifies_frozen_baseline(tmp_path):
    output = prepare(
        CONFIG,
        repo_root=REPO_ROOT,
        output_dir_override=tmp_path / "rejection",
    )
    manifest = yaml.safe_load((output / "manifest.yaml").read_text(encoding="utf-8"))

    assert manifest["status"] == "prepared"
    assert manifest["baseline_id"] == "active10d_baseline_v1"
    assert manifest["rejection_contract"]["accepted_candidate_count"] == 256
