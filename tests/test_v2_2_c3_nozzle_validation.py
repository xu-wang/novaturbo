"""Locked software gates for the V2.2-C3 synthetic nozzle shadow."""

from __future__ import annotations

from pathlib import Path

from src.physics.wp_v2_2_c3_validation import (
    render_v2_2_c3_nozzle_report,
    run_v2_2_c3_nozzle_validation,
    write_v2_2_c3_nozzle_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v2_2_c3_nozzle_campaign_passes_software_only() -> None:
    report = run_v2_2_c3_nozzle_validation(REPO_ROOT)

    assert report["software_gate_pass"] is True
    assert all(report["software_gates"].values())
    assert report["engineering_ready"] is False
    assert report["dense_map_scan"]["sample_shape"] == [81, 101]
    assert report["critical_transition"]["branch_change_count"] == 1
    assert (
        report["critical_transition"][
            "maximum_branch_transition_capacity_relative_jump"
        ]
        <= 0.02
    )
    assert report["critical_transition"]["physical_state_and_enthalpy_pass"] is True
    assert report["root_sweep"]["trace_complete"] is True
    assert report["root_sweep"]["root_trace_pass"] is True
    assert report["runtime_benchmark"]["timed_evaluation_count_per_profile"] == 28
    assert report["runtime_benchmark"]["p50_enhanced_to_baseline_ratio"] <= 3.0
    assert report["runtime_benchmark"]["p95_enhanced_to_baseline_ratio"] <= 3.0
    assert "nozzle_map_is_synthetic_software_fixture" in report[
        "engineering_blockers"
    ]


def test_v2_2_c3_nozzle_artifacts_are_machine_and_human_readable(
    tmp_path: Path,
) -> None:
    report = run_v2_2_c3_nozzle_validation(REPO_ROOT)
    output = tmp_path / "validation"
    write_v2_2_c3_nozzle_artifacts(report, output)

    assert (output / "validation_report.json").is_file()
    rendered = (output / "Validation_Report.md").read_text(encoding="utf-8")
    assert rendered == render_v2_2_c3_nozzle_report(report)
    assert "software gate: **PASS**" in rendered
    assert "engineering ready: **false**" in rendered
