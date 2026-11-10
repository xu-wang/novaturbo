from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.ai.baseline import sha256_file
from src.ai.inverse_design_baseline import (
    DEFAULT_BASELINE_CONFIG,
    _checkpoint_contract,
    load_inverse_design_baseline_config,
    verify_critical_artifacts,
    verify_inverse_design_baseline,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / DEFAULT_BASELINE_CONFIG


def test_repository_baseline_config_and_critical_hashes_are_valid():
    config = load_inverse_design_baseline_config(CONFIG_PATH)
    records = verify_critical_artifacts(config, REPO_ROOT)
    assert config["baseline_id"] == "active10d_cfm4_v1"
    assert len(records) == 20
    assert {record["category"] for record in records} >= {
        "dataset",
        "normalizer",
        "target_suite",
        "cfm_checkpoint",
        "gradient_surrogate",
    }


def test_completed_repository_freeze_manifest_matches_snapshot():
    freeze_path = (
        REPO_ROOT
        / "config"
        / "inverse_design"
        / "baselines"
        / "active10d_cfm4_v1_freeze.yaml"
    )
    freeze = yaml.safe_load(freeze_path.read_text(encoding="utf-8"))
    assert freeze["status"] == "complete"
    assert freeze["immutable"] is True
    assert freeze["execution_source"]["dirty"] is False
    assert len(freeze["execution_source"]["commit"]) == 40
    for artifact in freeze["artifacts"]:
        path = REPO_ROOT / artifact["path"]
        assert path.is_file()
        assert sha256_file(path) == artifact["sha256"]
    verified = verify_inverse_design_baseline(
        REPO_ROOT / freeze["output"]["directory"]
    )
    assert verified["artifact_count"] == 404
    assert verified["source_commit"] == freeze["execution_source"]["commit"]


def test_executable_contract_freezes_all_cost_and_acceptance_values():
    config = load_inverse_design_baseline_config(CONFIG_PATH)
    contract = _checkpoint_contract(config, REPO_ROOT)
    assert contract["candidate"]["n_raw"] == 128
    assert contract["candidate"]["maximum_total_0d_evaluations"] == 792
    assert contract["candidate"]["frozen_extreme_total_0d_evaluations"] == 728
    assert contract["acceptance"]["thrust_relative_tolerance"] == pytest.approx(
        0.0274514502151586
    )
    assert contract["acceptance"]["tsfc_relative_tolerance"] == pytest.approx(
        0.008573042420144972
    )
    assert contract["acceptance"]["mass_relative_tolerance"] == pytest.approx(
        0.01170710896944147
    )
    assert contract["acceptance"]["total_error_tolerance"] == pytest.approx(
        0.01591053386824835
    )
    assert contract["cfm_checkpoint"]["training_metadata"]["best_epoch"] == 105
    assert contract["gradient_surrogate"]["dataset_sha256"] == (
        "b4784922be044fccff846cacc93cec4133965de0007145fee84545dd60e78637"
    )


def test_critical_hash_drift_is_rejected(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"expected")
    config = {
        "critical_artifacts": [
            {
                "category": "test",
                "path": str(artifact),
                "sha256": hashlib.sha256(b"different").hexdigest(),
            }
        ]
    }
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_critical_artifacts(config, REPO_ROOT)


def test_baseline_verifier_detects_mutated_artifact(tmp_path):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    artifact = frozen / "value.txt"
    artifact.write_text("frozen\n", encoding="utf-8")
    frame = pd.DataFrame(
        [
            {
                "path": "value.txt",
                "size_bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        ]
    )
    artifact_manifest = frozen / "artifact_manifest.csv"
    frame.to_csv(artifact_manifest, index=False)
    (frozen / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "baseline_id": "test",
                "status": "complete",
                "source_commit": "a" * 40,
                "artifact_manifest_sha256": sha256_file(artifact_manifest),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    assert verify_inverse_design_baseline(frozen)["artifact_count"] == 1
    artifact.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_inverse_design_baseline(frozen)
