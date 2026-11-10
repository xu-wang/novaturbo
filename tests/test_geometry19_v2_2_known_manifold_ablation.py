"""Tests for the frozen V2.2 known-manifold five-fold ablation."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.ai.geometry19_v2_2_known_manifold_ablation import (
    _normalized_design_matrix,
    _root_centers_array,
    _root_weights,
    support_decisions,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config/experiments/geometry19_v2_2_known_manifold_ablation_v1.yaml"
EVIDENCE = REPO_ROOT / "data/validation/v2_2_known_manifold_ablation_v1"


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_root_weights_give_every_root_equal_total_weight() -> None:
    records = [
        {"root_seed_id": "a"},
        {"root_seed_id": "a"},
        {"root_seed_id": "a"},
        {"root_seed_id": "b"},
    ]
    weights = _root_weights(records)
    assert float(np.sum(weights[:3])) == pytest.approx(float(weights[3]))
    assert float(np.mean(weights)) == pytest.approx(1.0)


def test_frozen_ablation_uses_all_rows_once_per_variant_and_does_not_advance() -> None:
    summary = json.loads((EVIDENCE / "summary.json").read_text(encoding="utf-8"))
    predictions = _read_jsonl(EVIDENCE / "cross_validation_predictions.jsonl")

    assert summary["status"] == "DEVELOPMENT_ABLATION_COMPLETE_SUPPORT_GATE_PASS"
    assert summary["selected_variant"] == "ridge"
    assert summary["selected_vs_legacy_reduction_fraction"] == pytest.approx(
        0.0004446742870258813
    )
    assert summary["development_accuracy_gate"] == "NOT_PRE_REGISTERED_MODEL_SELECTION_ONLY"
    assert summary["model_advancement_established"] is False
    assert summary["fresh_confirmation_acquisition_ready"] is False
    assert summary["confirmation_content_read"] is False
    assert summary["confirmation_reuse_allowed"] is False
    assert summary["formal_use_allowed"] is False
    assert summary["deployment_allowed"] is False
    assert summary["dataset_rows"] == 4529
    assert summary["global_roots"] == 78
    assert len(summary["variant_results"]) == 5
    assert len(predictions) == 5 * 4529

    designs_by_variant: dict[str, set[str]] = {}
    for variant in summary["variant_results"]:
        variant_id = variant["variant_id"]
        design_hashes = {
            str(row["design_hash"])
            for row in predictions
            if row["variant_id"] == variant_id
        }
        designs_by_variant[variant_id] = design_hashes
        assert len(design_hashes) == 4529
        assert len(variant["folds"]) == 5
        assert all(
            fold["metrics"]["by_segment"][segment]["roots"] == 8
            for fold in variant["folds"]
            for segment in summary["target_segments"]
        )
    assert all(
        values == designs_by_variant["legacy_no_change"]
        for values in designs_by_variant.values()
    )

    selected = next(
        value for value in summary["variant_results"] if value["variant_id"] == "ridge"
    )
    assert selected["aggregate"]["maximum_worst_root_rho_p95"] == pytest.approx(
        15.825529602978083
    )
    error_audit = summary["selected_error_audit"]
    assert error_audit["dominant_axis_counts"] == {
        "thrust_N": 0,
        "tsfc_kg_N_s": 10,
        "mass_kg": 0,
    }
    assert error_audit["worst_segment"] == "low_mass_low_to_mid"
    assert error_audit["new_roots_in_top_worst"] == 3
    assert error_audit["existing_roots_in_top_worst"] == 7


def test_support_gate_is_development_calibrated_and_rejects_out_of_bounds() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    summary = json.loads((EVIDENCE / "summary.json").read_text(encoding="utf-8"))
    support = json.loads(
        (EVIDENCE / "known_manifold_support.json").read_text(encoding="utf-8")
    )
    rows = _read_jsonl(REPO_ROOT / config["development_dataset"]["path"])
    schema = yaml.safe_load(
        (REPO_ROOT / config["design_schema"]).read_text(encoding="utf-8")
    )
    variables = schema["design_variables"]
    _, centers = _root_centers_array(rows, variables)

    assert summary["support_audit"]["gate_pass"] is True
    assert summary["support_audit"]["target_validation_root_acceptance_fraction"] == pytest.approx(
        59 / 72
    )
    assert support["status"] == "KNOWN_MANIFOLD_SUPPORT_GATE_READY"
    assert support["outside_status"] == "OUTSIDE_KNOWN_MANIFOLD_SUPPORT"
    assert support["formal_use_allowed"] is False
    assert support["development_dataset_sha256"] == config["development_dataset"]["sha256"]
    assert float(np.mean(support_decisions(centers, support))) >= 0.98

    outside = np.full((1, 19), -0.01, dtype=float)
    assert support_decisions(outside, support).tolist() == [False]
    assert _normalized_design_matrix(rows[:2], variables).shape == (2, 19)


def test_frozen_ablation_artifact_hashes() -> None:
    summary = json.loads((EVIDENCE / "summary.json").read_text(encoding="utf-8"))
    assert summary["hashes"]["predictions"] == _hash(
        EVIDENCE / "cross_validation_predictions.jsonl"
    )
    assert summary["hashes"]["known_manifold_support"] == _hash(
        EVIDENCE / "known_manifold_support.json"
    )
    assert summary["hashes"]["config"] == _hash(CONFIG)
    assert summary["hashes"]["implementation"] == _hash(
        REPO_ROOT / "src/ai/geometry19_v2_2_known_manifold_ablation.py"
    )
