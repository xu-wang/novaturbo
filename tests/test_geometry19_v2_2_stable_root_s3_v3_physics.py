"""Tests for the sealed S3 v3 33-root V2.2 acquisition."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_stable_root_s3_v3_physics import (
    FAIL_STATUS,
    Geometry19StableRootS3V3PhysicsError,
    _eligible,
    run_s3_v3_physics_acquisition,
    validate_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config/experiments/geometry19_v2_2_stable_root_s3_v3_33root_evaluation_v1.yaml"
EVIDENCE = REPO_ROOT / "data/validation/v2_2_stable_root_forward_fresh_eval_v1"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_evaluation_protocol_freezes_order_scope_and_gates() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_protocol(config)
    assert config["execution_order"][2] == "write_hash_and_seal_physics_ledger_before_any_model_prediction"
    assert config["acquisition_gates"]["exact_roots"] == 33
    assert config["claims"]["forbidden"] == [
        "original_38root_accuracy",
        "all_78root_accuracy",
        "excluded_boundary_root_accuracy",
        "unknown_manifold_generalization",
        "inverse_design_readiness",
        "formal_or_deployment_use",
    ]
    config["physics_labeling"]["performance_adaptive_top_up"] = True
    with pytest.raises(Geometry19StableRootS3V3PhysicsError, match="Adaptive top-up"):
        validate_protocol(config)


def test_evaluation_eligibility_requires_every_frozen_condition() -> None:
    row = {
        "solver": {"converged": True},
        "constraint_assessment": {"numerically_valid": True, "physics_domain_valid": True},
        "performance": {"thrust_N": 70.0, "tsfc_kg_N_s": 6.0e-5, "mass_kg": 1.8},
        "target_domain_assessment": {"inside": True},
    }
    assert _eligible(row) is True
    for mutation in (
        ("solver", "converged"),
        ("constraint_assessment", "numerically_valid"),
        ("constraint_assessment", "physics_domain_valid"),
        ("target_domain_assessment", "inside"),
    ):
        changed = json.loads(json.dumps(row))
        changed[mutation[0]][mutation[1]] = False
        assert _eligible(changed) is False


def test_frozen_physics_failure_is_complete_hashed_and_unscored() -> None:
    summary_path = EVIDENCE / "audits/S3_v3_33root_Physics_Acquisition_Summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ledger = _read_jsonl(
        EVIDENCE / "fresh_evaluation_v3_33root/S3_v3_Sealed_V2_2_Evaluations.jsonl"
    )
    assert summary["status"] == FAIL_STATUS
    assert summary["counts"] == {
        "evaluated": 3168,
        "evaluation_eligible": 1147,
        "inside_target_domain": 1164,
        "numerically_valid": 2767,
        "physics_domain_valid": 2670,
        "solver_converged": 2946,
    }
    assert len(ledger) == 3168
    assert len({row["design_hash"] for row in ledger}) == 3168
    assert sum(row["evaluation_eligible"] for row in ledger) == 1147
    assert summary["minimum_evaluation_eligible_per_root_observed"] == 1
    assert summary["model_scoring_authorized"] is False
    assert summary["model_manifest_content_read"] is False
    assert summary["model_files_opened"] is False
    assert summary["model_predictions_executed"] is False
    assert summary["confirmation_content_read"] is False
    assert summary["protocol_sha256"] == sha256_file(CONFIG)
    assert summary["artifacts"]["sealed_physics_ledger_sha256"] == sha256_file(
        EVIDENCE / "fresh_evaluation_v3_33root/S3_v3_Sealed_V2_2_Evaluations.jsonl"
    )
    assert summary["artifacts"]["report_sha256"] == sha256_file(
        EVIDENCE / "audits/S3_v3_33root_Physics_Acquisition_Report.md"
    )
    for metadata, relative in zip(
        summary["artifacts"]["physics_shard_metadata"],
        summary["artifacts"]["physics_shards"],
    ):
        assert metadata["output_hash"] == sha256_file(REPO_ROOT / relative)


def test_label_stage_does_not_parse_manifest_or_confirmation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    protected = {
        (REPO_ROOT / config["inputs"][role]["path"]).resolve()
        for role in ("s2_model_manifest", "sealed_confirmation")
    }
    original_text = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.resolve() in protected:
            raise AssertionError("protected content was parsed during labeling")
        return original_text(path, *args, **kwargs)

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
        "src.ai.geometry19_v2_2_stable_root_s3_v3_physics.build_pipeline_context",
        lambda *args, **kwargs: fake_context,
    )
    monkeypatch.setattr(
        "src.ai.geometry19_v2_2_stable_root_s3_v3_physics._evaluate_row",
        fake_evaluate,
    )
    result = run_s3_v3_physics_acquisition(
        REPO_ROOT, CONFIG, output_root=tmp_path, resume=False
    )
    assert result["model_scoring_authorized"] is True
    assert result["model_manifest_content_read"] is False
    assert result["confirmation_content_read"] is False
