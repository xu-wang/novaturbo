"""Tests for sealed S3 v4.1 V2.2 development acquisition."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_stable_root_s3_v4_1_physics import (
    PASS_STATUS,
    _eligible,
    run_s3_v4_1_physics_acquisition,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config/experiments/geometry19_v2_2_stable_root_s3_v4_1_physics_acquisition_v1.yaml"
EVIDENCE = REPO_ROOT / "data/validation/v2_2_stable_root_forward_fresh_eval_v1"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_evaluation_eligibility_requires_every_frozen_condition() -> None:
    row = {
        "solver": {"converged": True},
        "constraint_assessment": {"numerically_valid": True, "physics_domain_valid": True},
        "performance": {"thrust_N": 70.0, "tsfc_kg_N_s": 6.0e-5, "mass_kg": 1.8},
        "target_domain_assessment": {"inside": True},
    }
    assert _eligible(row) is True
    for outer, inner in (
        ("solver", "converged"),
        ("constraint_assessment", "numerically_valid"),
        ("constraint_assessment", "physics_domain_valid"),
        ("target_domain_assessment", "inside"),
    ):
        changed = json.loads(json.dumps(row))
        changed[outer][inner] = False
        assert _eligible(changed) is False


def test_sealed_acquisition_is_complete_hashed_and_unscored() -> None:
    summary = json.loads((EVIDENCE / "audits/S3_v4_1_Physics_Acquisition_Summary.json").read_text(encoding="utf-8"))
    ledger_path = EVIDENCE / "fresh_evaluation_v4_1_medoid/S3_v4_1_Sealed_V2_2_Evaluations.jsonl"
    ledger = _read_jsonl(ledger_path)
    assert summary["status"] == PASS_STATUS
    assert summary["counts"] == {
        "evaluated": 3168,
        "evaluation_eligible": 3039,
        "inside_target_domain": 3045,
        "numerically_valid": 3111,
        "physics_domain_valid": 3101,
        "solver_converged": 3165,
    }
    assert summary["minimum_evaluation_eligible_per_root_observed"] == 67
    assert summary["source_target_segment_evaluation_eligible_fraction"] == {
        "low_mass_low_to_mid": 0.9510869565217391,
        "low_mass_mid_to_high": 0.9391025641025641,
    }
    assert summary["future_scoring_protocol_authorized"] is True
    assert len(ledger) == 3168
    assert len({row["design_hash"] for row in ledger}) == 3168
    assert summary["coordinate_design_hashes_preserved_in_order"] is True
    assert summary["all_shards_complete_before_gate"] is True
    assert summary["model_manifest_content_read"] is False
    assert summary["model_files_opened"] is False
    assert summary["model_predictions_executed"] is False
    assert summary["confirmation_content_read"] is False
    assert summary["model_scoring_executed"] is False
    assert summary["model_scoring_authorized_now"] is False
    assert summary["artifacts"]["sealed_physics_ledger_sha256"] == sha256_file(ledger_path)
    assert summary["artifacts"]["report_sha256"] == sha256_file(REPO_ROOT / summary["artifacts"]["report"])
    for metadata, relative in zip(summary["artifacts"]["physics_shard_metadata"], summary["artifacts"]["physics_shards"]):
        assert metadata["output_hash"] == sha256_file(REPO_ROOT / relative)


def test_acquisition_never_parses_models_or_confirmation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    protected = {
        (REPO_ROOT / config["inputs"][role]["path"]).resolve()
        for role in ("s2_model_manifest", "sealed_confirmation")
    }
    original = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.resolve() in protected:
            raise AssertionError("protected content was parsed")
        return original(path, *args, **kwargs)

    @dataclass
    class FakeTarget:
        source_hash: str

    fake_context = SimpleNamespace(
        target_domain=FakeTarget(sha256_file(REPO_ROOT / config["inputs"]["target_domain"]["path"])),
        context_hash="fake-context",
        physics_runtime=SimpleNamespace(physics_bundle_hash="fake-bundle"),
    )

    def fake_evaluate(row: dict, context: object) -> dict:
        return {
            **row,
            "solver": {"converged": True, "failure_code": None},
            "constraint_assessment": {"numerically_valid": True, "physics_domain_valid": True},
            "performance": {"thrust_N": 70.0, "tsfc_kg_N_s": 6.0e-5, "mass_kg": 1.8},
            "target_domain_assessment": {"inside": True},
        }

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(
        "src.ai.geometry19_v2_2_stable_root_s3_v4_1_physics.build_pipeline_context",
        lambda *args, **kwargs: fake_context,
    )
    monkeypatch.setattr(
        "src.ai.geometry19_v2_2_stable_root_s3_v4_1_physics._evaluate_row",
        fake_evaluate,
    )
    result = run_s3_v4_1_physics_acquisition(
        REPO_ROOT, CONFIG, output_root=tmp_path, resume=False
    )
    assert result["future_scoring_protocol_authorized"] is True
    assert result["model_scoring_authorized_now"] is False
    assert result["model_manifest_content_read"] is False
    assert result["confirmation_content_read"] is False
