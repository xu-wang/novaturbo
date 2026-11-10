"""Locked software gates for V2.2-C4 hot-side variable properties."""

from __future__ import annotations

from pathlib import Path

from src.physics.wp_v2_2_c4_validation import (
    render_v2_2_c4_report,
    run_v2_2_c4_validation,
    write_v2_2_c4_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v2_2_c4_campaign_passes_software_only() -> None:
    report = run_v2_2_c4_validation(REPO_ROOT)

    assert report["software_gate_pass"] is True
    assert all(report["software_gates"].values())
    assert report["engineering_ready"] is False
    assert report["c3_reference_isolation"]["pass"] is True
    assert report["hot_side_handoff"]["pass"] is True
    assert report["root_sweep"]["enhanced_case_count"] == 12
    assert report["root_sweep"]["state_pass"] is True
    assert report["root_sweep"]["root_trace_pass"] is True
    assert report["runtime_benchmark"]["timed_evaluation_count_per_profile"] == 28
    assert report["runtime_benchmark"]["p50_enhanced_to_baseline_ratio"] <= 2.0
    assert report["runtime_benchmark"]["p95_enhanced_to_baseline_ratio"] <= 2.0
    assert "no_nominal_conservative_uq_acceptance_contract" in report[
        "engineering_blockers"
    ]


def test_v2_2_c4_artifacts_are_machine_and_human_readable(tmp_path: Path) -> None:
    report = run_v2_2_c4_validation(REPO_ROOT)
    output = tmp_path / "validation"
    write_v2_2_c4_artifacts(report, output)

    assert (output / "validation_report.json").is_file()
    rendered = (output / "Validation_Report.md").read_text(encoding="utf-8")
    assert rendered == render_v2_2_c4_report(report)
    assert "software gate: **PASS**" in rendered
    assert "engineering ready: **false**" in rendered
