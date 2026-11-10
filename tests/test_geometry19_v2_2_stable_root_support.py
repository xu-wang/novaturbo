"""Tests for stable-root S0/S1 registry and support auditing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_stable_root_support import (
    PASS_BOTH,
    Geometry19StableRootSupportError,
    run_stable_root_support_audit,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT
    / "config/experiments/geometry19_v2_2_stable_root_conditional_surrogate_v1.yaml"
)
EVIDENCE = REPO_ROOT / "data/validation/v2_2_stable_root_conditional_surrogate_v1"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_protocol_forbids_model_or_fresh_data_work() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_protocol(config)
    config["next_stage_contract"]["train_forward_surrogate_now"] = True
    with pytest.raises(Geometry19StableRootSupportError, match="cannot train"):
        validate_protocol(config)


def test_frozen_stable_registry_exactly_matches_p2_four_of_four_roots() -> None:
    p2 = json.loads(
        (
            REPO_ROOT
            / "data/validation/v2_2_local_chart_moe_v1/audits/jacobian_stability_summary.json"
        ).read_text(encoding="utf-8")
    )
    expected = {
        root_id
        for root_id, value in p2["root_reliability"].items()
        if value["training_fold_estimates"] == 4 and value["reliable_estimates"] == 4
    }
    registry = _read_jsonl(EVIDENCE / "audits/stable_root_registry.jsonl")
    assert len(registry) == 38
    assert {row["root_seed_id"] for row in registry} == expected
    assert sum(row["development_rows"] for row in registry) == 2935
    assert all(row["development_rows"] >= 24 for row in registry)
    assert all(row["normalized_covariance_effective_rank"] >= 6 for row in registry)
    assert all(row["support_radius_p95_normalized_l2"] > 0.0 for row in registry)


def test_frozen_support_audit_passes_but_remains_development_only() -> None:
    summary = json.loads(
        (EVIDENCE / "audits/stable_root_support_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == PASS_BOTH
    assert summary["anchored_support_gate_pass"] is True
    assert summary["blind_support_gate_pass"] is True
    assert summary["s2_anchored_mode_authorized"] is True
    assert summary["s2_blind_mode_authorized"] is True
    assert summary["original_p4_authorized"] is False
    assert summary["fresh_evaluation_required_before_accuracy_claim"] is True
    assert summary["fresh_data_generated"] is False
    assert summary["forward_surrogate_training_executed"] is False
    assert summary["inverse_model_training_executed"] is False
    assert summary["confirmation_content_read"] is False
    assert summary["stable_root_sources"] == {
        "known_manifold_root_discovery_v1_new": 1,
        "pre_root_discovery_development": 37,
    }
    audit = summary["support_audit"]
    assert audit["development_diagnostic_only"] is True
    assert audit["anchored_stable_row_acceptance_fraction"] >= 0.80
    assert audit["blind_stable_row_acceptance_fraction"] >= 0.80
    assert audit["always_unreliable_false_acceptance_fraction"] <= 0.20
    for key, value in summary["artifacts"].items():
        if key.endswith("_sha256"):
            continue
        assert summary["artifacts"][f"{key}_sha256"] == sha256_file(REPO_ROOT / value)
    assert summary["protocol_sha256"] == sha256_file(CONFIG)


def test_s0_s1_never_text_parses_sealed_confirmation(
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
    result = run_stable_root_support_audit(REPO_ROOT, CONFIG, output_root=tmp_path)
    assert result["confirmation_hash_checked"] is True
    assert result["confirmation_content_read"] is False
    assert result["fresh_data_generated"] is False
    assert result["forward_surrogate_training_executed"] is False
