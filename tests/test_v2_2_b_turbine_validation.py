"""Locked software gates for the V2.2-B synthetic turbine shadow."""

from __future__ import annotations

from pathlib import Path

from src.physics.wp_v2_2_b_validation import (
    render_v2_2_b_turbine_report,
    run_v2_2_b_turbine_validation,
    write_v2_2_b_turbine_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v2_2_b_turbine_campaign_passes_software_only() -> None:
    report = run_v2_2_b_turbine_validation(REPO_ROOT)

    assert report["software_gate_pass"] is True
    assert all(report["software_gates"].values())
    assert report["engineering_ready"] is False
    assert report["dense_map_scan"]["sample_shape"] == [81, 121]
    assert report["root_sweep"]["trace_complete"] is True
    assert report["root_sweep"]["enhanced_converged_fraction"] == 1.0
    assert report["root_sweep"]["unexplained_jump_count"] == 0
    assert report["root_sweep"]["state_direction_and_shaft_balance_pass"] is True
    assert report["runtime_benchmark"]["timed_evaluation_count_per_profile"] == 35
    assert report["runtime_benchmark"]["p50_enhanced_to_baseline_ratio"] <= 2.0
    assert report["runtime_benchmark"]["p95_enhanced_to_baseline_ratio"] <= 2.0
    assert "turbine_map_is_synthetic_software_fixture" in report[
        "engineering_blockers"
    ]


def test_v2_2_b_turbine_artifacts_are_machine_and_human_readable(
    tmp_path: Path,
) -> None:
    report = run_v2_2_b_turbine_validation(REPO_ROOT)
    output = tmp_path / "validation"
    write_v2_2_b_turbine_artifacts(report, output)

    assert (output / "validation_report.json").is_file()
    rendered = (output / "Validation_Report.md").read_text(encoding="utf-8")
    assert rendered == render_v2_2_b_turbine_report(report)
    assert "software gate: **PASS**" in rendered
    assert "engineering ready: **false**" in rendered
