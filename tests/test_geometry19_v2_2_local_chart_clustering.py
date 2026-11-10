"""Tests for reliability-aware P3 local-chart identification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_v2_2_local_chart_clustering import (
    UNSTABLE,
    _cluster_candidate,
    run_chart_identification,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config/experiments/geometry19_v2_2_local_chart_clustering_v1.yaml"
EVIDENCE = REPO_ROOT / "data/validation/v2_2_local_chart_moe_v1"


def _distances(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)


def test_unreliable_roots_do_not_change_reliable_core_linkage() -> None:
    core_points = np.asarray([[0.0], [0.1], [0.2], [0.3], [9.7], [9.8], [9.9], [10.0]])
    attached_points = np.asarray([[0.4], [9.6]])
    all_points = np.vstack([core_points, attached_points])
    core_ids = [f"a{index}" for index in range(len(core_points))]
    all_ids = core_ids + ["z0", "z1"]
    core_result = _cluster_candidate(
        core_ids,
        np.ones(len(core_ids), dtype=np.bool_),
        _distances(core_points),
        _distances(core_points),
        requested_k=2,
        minimum_final_charts=2,
        minimum_core=2,
        minimum_total=1,
    )
    full_result = _cluster_candidate(
        all_ids,
        np.asarray([True] * len(core_ids) + [False, False]),
        _distances(all_points),
        _distances(all_points),
        requested_k=2,
        minimum_final_charts=2,
        minimum_core=2,
        minimum_total=1,
    )
    core_labels = core_result["labels"]
    full_core_labels = full_result["labels"][: len(core_ids)]
    assert np.array_equal(core_labels[:, None] == core_labels[None, :], full_core_labels[:, None] == full_core_labels[None, :])


def test_single_chart_collapse_fails_structural_gate() -> None:
    points = np.asarray([[0.0], [0.1], [0.2], [0.3]])
    result = _cluster_candidate(
        [f"r{index}" for index in range(len(points))],
        np.ones(len(points), dtype=np.bool_),
        _distances(points),
        _distances(points),
        requested_k=1,
        minimum_final_charts=2,
        minimum_core=2,
        minimum_total=1,
    )
    assert result["final_k"] == 1
    assert result["structural_gate_pass"] is False
    assert result["status"] == "CANDIDATE_STRUCTURE_FAIL"


def test_frozen_p3_outputs_fail_closed_and_are_hashed() -> None:
    summary = json.loads(
        (EVIDENCE / "audits/chart_identification_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == UNSTABLE
    assert summary["response_qualified_k"] == []
    assert summary["design_only_qualified_k"] == []
    assert summary["p4_authorized"] is False
    assert summary["p3_selects_final_k"] is False
    assert summary["confirmation_content_read"] is False
    assert summary["expert_training_executed"] is False
    assert summary["router_training_executed"] is False
    assert summary["protocol_sha256"] == sha256_file(CONFIG)
    for key, value in summary["artifacts"].items():
        if key.endswith("_sha256"):
            continue
        assert summary["artifacts"][f"{key}_sha256"] == sha256_file(REPO_ROOT / value)
    for view in ("design_only", "response_aware"):
        for requested_k in (3, 4, 6, 8):
            for item in summary["candidate_artifacts"][view][str(requested_k)].values():
                assert item["assignment_sha256"] == sha256_file(
                    REPO_ROOT / item["assignment_path"]
                )
                assert item["chart_summary_sha256"] == sha256_file(
                    REPO_ROOT / item["chart_summary_path"]
                )

    candidates = summary["response_comparison"]["candidates"]
    assert [item["requested_k"] for item in candidates] == [3, 4, 6, 8]
    assert candidates[0]["design_structural_gate_pass"] is False
    assert candidates[0]["response_structural_gate_pass"] is False
    assert candidates[-1]["response_value_gate_pass"] is True
    assert candidates[-1]["response_stability_gate_pass"] is False

    for fold in range(5):
        for view in ("design_only", "response_aware"):
            for requested_k in (3, 4, 6, 8):
                payload = json.loads(
                    (
                        EVIDENCE
                        / f"charts/fold-{fold}/{view}/k-{requested_k}/chart_assignments.json"
                    ).read_text(encoding="utf-8")
                )
                if "assignments" not in payload:
                    continue
                assert not (
                    set(payload["assignments"]) & set(payload["validation_roots_excluded"])
                )
                assert all(
                    item["role"] in {"reliable_core", "design_only_attachment"}
                    for item in payload["assignments"].values()
                )


def test_p3_never_text_parses_sealed_confirmation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    child = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    parent = yaml.safe_load(
        (REPO_ROOT / child["parent_protocol"]["path"]).read_text(encoding="utf-8")
    )
    confirmation = (REPO_ROOT / parent["inputs"]["sealed_confirmation"]["path"]).resolve()
    original = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.resolve() == confirmation:
            raise AssertionError("sealed confirmation content was parsed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    result = run_chart_identification(REPO_ROOT, CONFIG, output_root=tmp_path)
    assert result["confirmation_hash_checked"] is True
    assert result["confirmation_content_read"] is False
    assert result["p4_authorized"] is False
