"""Tests for local-chart MoE P0 protocol preregistration and input locking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.ai.geometry19_v2_2_local_chart_protocol import (
    Geometry19LocalChartProtocolError,
    P0_PASS_STATUS,
    run_protocol_input_lock,
    validate_protocol_schema,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config/experiments/geometry19_v2_2_local_chart_moe_v1.yaml"
EVIDENCE = REPO_ROOT / "data/validation/v2_2_local_chart_moe_v1/audits"


def test_protocol_schema_freezes_models_folds_gates_and_confirmation() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_protocol_schema(config)

    assert config["charts"]["candidate_k"] == [3, 4, 6, 8]
    assert config["inner_selection"]["fold_count"] == 4
    assert config["outer_evaluation"]["fold_count"] == 5
    assert len(config["experiment_arms"]) == 8
    assert config["training"]["physical_gpu"] == 6
    assert config["training"]["process_device"] == "cuda:0"
    assert config["inputs"]["sealed_confirmation"]["access"] == "sha256_only_no_parse"
    assert config["inputs"]["sealed_confirmation"]["confirmation_content_read"] is False
    assert config["gates"]["advancement"]["minimum_reduction_vs_legacy_no_change"] == 0.20
    assert config["gates"]["readiness"]["maximum_cross_fold_worst_root_rho_p95"] == 5.0


def test_protocol_lock_matches_frozen_inputs_and_root_exclusive_folds() -> None:
    audit = json.loads((EVIDENCE / "protocol_input_lock.json").read_text(encoding="utf-8"))
    assert audit["status"] == P0_PASS_STATUS
    assert audit["protocol_schema_validated"] is True
    assert audit["all_input_hashes_match"] is True
    assert audit["confirmation_content_read"] is False
    assert audit["confirmation_reuse_allowed"] is False
    assert audit["model_work_executed"] is False
    assert audit["fold_audit"]["rows"] == 4529
    assert audit["fold_audit"]["global_roots"] == 78
    assert audit["fold_audit"]["root_fold_exclusive"] is True
    assert len(audit["input_hash_audit"]) == 8
    assert all(item["match"] for item in audit["input_hash_audit"])
    assert all(
        list(item["validation_roots_by_fold"].values()) == [8, 8, 8, 8, 8]
        for item in audit["fold_audit"]["target_segments"].values()
    )


def test_protocol_lock_never_text_parses_sealed_confirmation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    confirmation = (REPO_ROOT / config["inputs"]["sealed_confirmation"]["path"]).resolve()
    original = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.resolve() == confirmation:
            raise AssertionError("sealed confirmation content was parsed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    result = run_protocol_input_lock(REPO_ROOT, CONFIG, output_root=tmp_path)
    assert result["status"] == P0_PASS_STATUS
    assert result["confirmation_hash_checked"] is True
    assert result["confirmation_content_read"] is False


def test_protocol_lock_fails_closed_on_hash_mismatch(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["inputs"]["development_dataset"]["sha256"] = "0" * 64
    bad_config = tmp_path / "bad_protocol.yaml"
    bad_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        Geometry19LocalChartProtocolError,
        match="Frozen input hash mismatch: development_dataset",
    ):
        run_protocol_input_lock(REPO_ROOT, bad_config, output_root=tmp_path / "output")
    assert not (tmp_path / "output").exists()
