"""Tests for train-fold-only local Jacobian bootstrap stability."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_local_chart_features import (
    P2_PASS_STATUS,
    _estimate_root,
    run_local_jacobian_stability,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config/experiments/geometry19_v2_2_local_chart_moe_v1.yaml"
EVIDENCE = REPO_ROOT / "data/validation/v2_2_local_chart_moe_v1"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_root_estimate_is_deterministic_and_marks_one_row_unreliable() -> None:
    variable_names = [f"x{index}" for index in range(19)]
    record = {
        "geometry19": {name: 0.5 for name in variable_names},
        "performance": {
            "thrust_N": 50.0,
            "tsfc_kg_N_s": 0.00006,
            "mass_kg": 2.0,
        },
        "assigned_segment": "high_mass_mid_to_high",
    }
    kwargs = {
        "fold": 0,
        "variable_names": variable_names,
        "lower": np.zeros(19),
        "scale": np.ones(19),
        "tolerances": np.asarray([0.02, 0.01, 0.01]),
        "alpha_candidates": [0.0001, 0.001],
        "replicates": 5,
        "base_seed": 7,
        "reliability_gate": {
            "minimum_rows": 24,
            "minimum_effective_rank": 6,
            "minimum_tsfc_gradient_cosine_median": 0.7,
            "maximum_frobenius_coefficient_of_variation": 0.5,
        },
    }
    first_audit, first_arrays = _estimate_root("one-row", [record], **kwargs)
    second_audit, second_arrays = _estimate_root("one-row", [record], **kwargs)
    assert first_audit == second_audit
    assert np.array_equal(first_arrays["jacobian"], second_arrays["jacobian"])
    assert first_audit["jacobian_reliable"] is False
    assert "rows_below_minimum" in first_audit["reliability_failures"]
    assert "effective_rank_below_minimum" in first_audit["reliability_failures"]


def test_frozen_p2_outputs_are_train_fold_only_and_hashed() -> None:
    summary_path = EVIDENCE / "audits/jacobian_stability_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    folds = json.loads(
        (
            REPO_ROOT / config["inputs"]["development_folds"]["path"]
        ).read_text(encoding="utf-8")
    )
    assignments = {str(key): int(value) for key, value in folds["root_assignments"].items()}

    assert summary["status"] == P2_PASS_STATUS
    assert summary["p3_authorized"] is True
    assert summary["root_fold_estimates"] == 312
    assert len(summary["folds"]) == 5
    assert summary["bootstrap_replicates"] == 50
    assert summary["confirmation_content_read"] is False
    assert summary["local_jacobian_fitting_executed"] is True
    assert summary["surrogate_expert_training_executed"] is False
    for fold in summary["folds"]:
        fold_id = fold["fold"]
        npz_path = REPO_ROOT / fold["artifacts"]["root_jacobians"]
        audit_path = REPO_ROOT / fold["artifacts"]["jacobian_audit"]
        assert fold["artifacts"]["root_jacobians_sha256"] == sha256_file(npz_path)
        assert fold["artifacts"]["jacobian_audit_sha256"] == sha256_file(audit_path)
        with np.load(npz_path) as bundle:
            root_ids = {str(value) for value in bundle["root_ids"].tolist()}
            validation_ids = {str(value) for value in bundle["validation_root_ids"].tolist()}
            assert bundle["jacobians"].shape == (fold["training_roots"], 3, 19)
            assert bundle["jacobian_reliable"].shape == (fold["training_roots"],)
        assert root_ids == {root_id for root_id, value in assignments.items() if value != fold_id}
        assert validation_ids == {root_id for root_id, value in assignments.items() if value == fold_id}
        assert not (root_ids & validation_ids)
        audits = _read_jsonl(audit_path)
        assert len(audits) == fold["training_roots"]
        assert all(row["role"] == "outer_training_root" for row in audits)
        assert all(len(row["alpha_metrics"]) == 4 for row in audits)
        assert all(
            metric["bootstrap_replicates"] == 50
            for row in audits
            for metric in row["alpha_metrics"]
        )


def test_p2_summary_hashes_and_reliability_actions() -> None:
    summary = json.loads(
        (EVIDENCE / "audits/jacobian_stability_summary.json").read_text(encoding="utf-8")
    )
    assert summary["protocol_sha256"] == sha256_file(CONFIG)
    assert summary["report_sha256"] == sha256_file(
        EVIDENCE / "audits/P2_Jacobian_Stability_Report.md"
    )
    assert summary["minimum_fold_reliable_fraction"] >= 0.5
    assert summary["minimum_target_segment_reliable_fraction"] == pytest.approx(0.4375)
    assert summary["roots_reliable_in_all_training_folds"] == 38
    assert summary["roots_unreliable_in_all_training_folds"] == 32
    all_audits = [
        row
        for fold in range(5)
        for row in _read_jsonl(EVIDENCE / f"jacobians/fold-{fold}/jacobian_audit.jsonl")
    ]
    assert all(
        row["unreliable_action"]
        == "retain_design_center_and_zero_jacobian_distance_weight"
        for row in all_audits
        if not row["jacobian_reliable"]
    )
    assert not any(
        "validation" in str(row["role"]).lower()
        for row in all_audits
    )


def test_p2_never_text_parses_sealed_confirmation(
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
    source_audits = EVIDENCE / "audits"
    target_audits = tmp_path / "audits"
    target_audits.mkdir(parents=True)
    for name in ("protocol_input_lock.json", "input_audit.json"):
        (target_audits / name).write_bytes((source_audits / name).read_bytes())
    result = run_local_jacobian_stability(REPO_ROOT, CONFIG, output_root=tmp_path)
    assert result["confirmation_hash_checked"] is True
    assert result["confirmation_content_read"] is False
