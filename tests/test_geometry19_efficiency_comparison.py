"""Integration tests for the WP19-23 V2.1 call-efficiency comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import src.ai.geometry19_efficiency_comparison as comparison


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_three_target_classes_use_actual_stage_call_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = yaml.safe_load(
        (
            REPO_ROOT
            / "config/experiments/wp19_23_pipeline_vs_cold_nsga2_256_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    config["targets"] = [config["targets"][index] for index in (0, 3, 6)]
    config_path = tmp_path / "comparison.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    project_calls: list[dict] = []
    cold_calls: list[dict] = []

    def project_mock(*args, **kwargs):
        project_calls.append(kwargs)
        output = Path(kwargs["output_root"])
        _write_summary(
            output / "wp19_23c/summary.json",
            {"counts": {"unique_v2_1_evaluations": 32}, "downstream_stop": True},
        )
        _write_summary(
            output / "wp19_23f/summary.json",
            {
                "counts": {
                    "v2_1_recalculation_passed": 1,
                    "development_diagnostic_hits": 1,
                }
            },
        )
        _write_summary(output / "pipeline_summary.json", {"status": "PASS"})
        return {"status": "PASS"}

    def cold_mock(*args, **kwargs):
        cold_calls.append(kwargs)
        summary = {
            "counts": {"unique_v2_1_evaluations": 128},
            "stopped_after_generation": 1,
        }
        _write_summary(Path(kwargs["output_root"]) / "summary.json", summary)
        return summary

    def delivery_mock(*args, **kwargs):
        summary = {
            "counts": {
                "v2_1_recalculation_passed": 1,
                "development_diagnostic_hits": 1,
            }
        }
        _write_summary(Path(kwargs["output_root"]) / "summary.json", summary)
        return summary

    monkeypatch.setattr(comparison, "run_wp19_23_pipeline", project_mock)
    monkeypatch.setattr(comparison, "run_nsga2_stage", cold_mock)
    monkeypatch.setattr(comparison, "run_delivery_stage", delivery_mock)
    summary = comparison.run_efficiency_comparison(
        REPO_ROOT,
        config_path,
        device="cpu",
        output_root=tmp_path / "runtime",
    )
    assert len(project_calls) == 3
    assert all(call["model_variant"] == "baseline" for call in project_calls)
    assert all(call["sampling_mode"] == "direct" for call in project_calls)
    assert len(cold_calls) == 3
    assert all(call["cold_start_comparison"] is True for call in cold_calls)
    assert summary["raw_candidate_count"] == 256
    for target_class in ("ID", "INTERPOLATION", "EXTRAPOLATION"):
        row = summary["aggregate"][target_class]
        assert row["project_total_v2_1_calls"] == 33
        assert row["cold_start_total_v2_1_calls"] == 129
        assert row["saved_v2_1_calls"] == 96
    assert summary["overall"]["saved_v2_1_calls"] == 288


def test_counting_reports_negative_savings_without_clamping() -> None:
    rows = [
        {
            "target_class": "ID",
            "saved_v2_1_calls": -24,
            "project": {"total_v2_1_calls": 665, "diagnostic_hit_delivered": False},
            "cold_start": {"total_v2_1_calls": 641, "diagnostic_hit_delivered": False},
        }
    ]
    value = comparison._aggregate(rows, "ID")
    assert value["saved_v2_1_calls"] == -24
    assert value["saved_fraction"] == pytest.approx(-24 / 641)
