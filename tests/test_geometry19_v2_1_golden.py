"""Frozen multi-case Geometry19 V2.1 baseline contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.physics.geometry19_baseline_validation import (
    DEFAULT_MANIFEST_PATH,
    Geometry19BaselineError,
    build_v2_1_baseline_manifest,
    canonical_baseline_payload_hash,
    verify_v2_1_baseline,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_committed_v2_1_baseline_reproduces_all_golden_cases() -> None:
    report = verify_v2_1_baseline(REPO_ROOT)

    assert report["passed"] is True
    assert report["case_count"] == 11
    assert report["difference_count"] == 0


def test_baseline_manifest_hash_rejects_tampering(tmp_path: Path) -> None:
    source = REPO_ROOT / DEFAULT_MANIFEST_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["cases"][0]["expected_result"]["performance"]["thrust_N"] += 1.0
    tampered = tmp_path / "tampered_manifest.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(Geometry19BaselineError, match="payload hash mismatch"):
        verify_v2_1_baseline(REPO_ROOT, manifest_path=tampered)


def test_baseline_verifier_reports_field_level_difference(tmp_path: Path) -> None:
    source = REPO_ROOT / DEFAULT_MANIFEST_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["cases"][0]["expected_result"]["performance"]["thrust_N"] += 1.0
    without_hash = {
        key: value
        for key, value in payload.items()
        if key != "manifest_payload_sha256"
    }
    payload["manifest_payload_sha256"] = canonical_baseline_payload_hash(without_hash)
    changed = tmp_path / "changed_manifest.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    report = verify_v2_1_baseline(REPO_ROOT, manifest_path=changed)
    assert report["passed"] is False
    assert report["difference_count"] == 1
    assert report["differences"][0]["path"].endswith(
        "performance.thrust_N"
    )


def test_golden_suite_rejects_solver_branch_drift(tmp_path: Path) -> None:
    source = REPO_ROOT / "config/physics/geometry19_v2_1_golden_cases_v1.yaml"
    changed = source.read_text(encoding="utf-8").replace(
        "root_count: 2, selected_root_index: 1",
        "root_count: 1, selected_root_index: 0",
        1,
    )
    suite = tmp_path / source.name
    suite.write_text(changed, encoding="utf-8")

    with pytest.raises(
        Geometry19BaselineError,
        match="Golden solver expectation failed for shaft_speed_105",
    ):
        build_v2_1_baseline_manifest(REPO_ROOT, suite_path=suite)
