"""Tests for sealed S3 v4.1 one-shot five-arm scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_stable_root_s3_v4_1_scoring import (
    PASS_STATUS,
    _metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = REPO_ROOT / "data/validation/v2_2_stable_root_forward_fresh_eval_v1"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_metrics_use_root_p95_and_all_source_segments() -> None:
    rows = [
        {"root_seed_id": "a", "source_root_segments": ["low_mass_low_to_mid"]},
        {"root_seed_id": "a", "source_root_segments": ["low_mass_low_to_mid"]},
        {"root_seed_id": "b", "source_root_segments": ["low_mass_mid_to_high"]},
    ]
    truth = np.ones((3, 3))
    prediction = np.asarray([[1.0, 1.0, 1.0], [1.1, 1.0, 1.0], [1.0, 1.2, 1.0]])
    metrics, rho = _metrics(rows, truth, prediction, np.ones(3))
    assert metrics["roots"] == 2
    assert metrics["by_target_segment"]["low_mass_low_to_mid"]["roots"] == 1
    assert metrics["by_target_segment"]["low_mass_mid_to_high"]["roots"] == 1
    assert len(rho) == 3


def test_sealed_scoring_is_complete_hashed_and_nonadaptive() -> None:
    summary = json.loads((EVIDENCE / "audits/S3_v4_1_Model_Scoring_Summary.json").read_text(encoding="utf-8"))
    ledger_path = EVIDENCE / "fresh_evaluation_v4_1_medoid/S3_v4_1_Frozen_Model_Predictions.jsonl"
    ledger = _read_jsonl(ledger_path)
    assert summary["status"] == PASS_STATUS
    assert summary["scoring_rows"] == len(ledger) == 3039
    assert len(summary["model_arms"]) == 5
    assert summary["advancing_arms"] == [
        "stable_38_nearest_root_local_ridge",
        "stable_38_oracle_root_local_ridge",
        "stable_38_global_plus_root_residual_ridge",
    ]
    assert summary["nearest_route_accuracy"] == 0.9851924975320829
    assert summary["nearest_vs_oracle_relative_worst_root_gap"] == 0.16939223574102735
    assert {
        item["arm_id"]: item["metrics"]["worst_root_rho_p95"]
        for item in summary["model_arms"]
    } == {
        "all_78_global_root_balanced_ridge": 8.942623272417311,
        "stable_38_pooled_root_balanced_ridge": 6.738862708577817,
        "stable_38_nearest_root_local_ridge": 2.8248559283636627,
        "stable_38_oracle_root_local_ridge": 2.4156616078210846,
        "stable_38_global_plus_root_residual_ridge": 2.206007431229487,
    }
    assert [row["scoring_row_index"] for row in ledger] == list(range(3039))
    assert all(len(row["arms"]) == 5 for row in ledger)
    assert summary["prediction_ledger_sealed_before_metrics"] is True
    assert summary["confirmation_content_read"] is False
    assert summary["inverse_design_authorized"] is False
    assert summary["artifacts"]["prediction_ledger_sha256"] == sha256_file(ledger_path)
    assert summary["artifacts"]["report_sha256"] == sha256_file(
        REPO_ROOT / summary["artifacts"]["report"]
    )
