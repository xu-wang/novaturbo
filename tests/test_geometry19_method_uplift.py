"""Tests for aligned WP19-23 method-arm uplift reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ai.geometry19_method_uplift import (
    Geometry19MethodUpliftError,
    compare_method_arms,
)


def _summary(path: Path, hits: tuple[bool, bool, bool], calls: tuple[int, int, int]) -> None:
    classes = ("ID", "INTERPOLATION", "EXTRAPOLATION")
    rows = []
    for index, target_class in enumerate(classes):
        rows.append(
            {
                "label": f"target_{index}",
                "target_class": target_class,
                "target": {"target_id": str(index)},
                "project": {
                    "diagnostic_hit_delivered": hits[index],
                    "terminal_search_stage": "WP19-23C" if hits[index] else "WP19-23E",
                    "total_v2_1_calls": calls[index],
                },
            }
        )
    path.write_text(json.dumps({"target_runs": rows}), encoding="utf-8")


def test_uplift_is_paired_by_target_and_reports_call_change(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    enhanced = tmp_path / "enhanced.json"
    _summary(baseline, (False, False, False), (100, 200, 300))
    _summary(enhanced, (True, True, False), (50, 150, 350))
    summary = compare_method_arms(baseline, enhanced, tmp_path / "uplift")
    assert summary["overall"]["baseline_hits"] == 0
    assert summary["overall"]["enhanced_hits"] == 2
    assert summary["overall"]["hit_rate_uplift_percentage_points"] == pytest.approx(
        200 / 3
    )
    assert summary["overall"]["additional_v2_1_calls"] == -50
    assert summary["aggregate"]["ID"]["hit_rate_uplift_percentage_points"] == 100


def test_uplift_rejects_different_target_sets(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    enhanced = tmp_path / "enhanced.json"
    _summary(baseline, (False, False, False), (1, 1, 1))
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload["target_runs"].pop()
    enhanced.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Geometry19MethodUpliftError, match="different target labels"):
        compare_method_arms(baseline, enhanced, tmp_path / "uplift")
