"""WP-V3-2 HECC map software and evidence gates."""

from __future__ import annotations

from pathlib import Path

from src.physics.wp_v3_2_validation import (
    render_wp_v3_2_report,
    run_wp_v3_2_validation,
    write_wp_v3_2_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_locked_wp_v3_2_campaign_passes_software_but_not_engineering_gates() -> None:
    report = run_wp_v3_2_validation(REPO_ROOT)

    assert report["software_gate_pass"] is True
    assert all(report["software_gates"].values())
    assert report["engineering_ready"] is False
    assert report["dense_map_scan"]["sample_shape"] == [81, 121]
    assert report["root_sweep"]["trace_complete"] is True
    assert report["root_sweep"]["unexplained_jump_count"] == 0
    assert report["root_sweep"]["enhanced_converged_fraction"] == 1.0
    assert report["runtime_benchmark"]["timed_evaluation_count_per_profile"] == 35
    assert report["runtime_benchmark"]["enhanced_to_baseline_ratio"] <= 2.0
    assert "only_one_independent_measured_design_point" in report[
        "engineering_blockers"
    ]


def test_wp_v3_2_artifacts_are_machine_and_human_readable(tmp_path: Path) -> None:
    report = run_wp_v3_2_validation(REPO_ROOT)
    output = tmp_path / "validation"
    write_wp_v3_2_artifacts(report, output)

    assert (output / "validation_report.json").is_file()
    rendered = (output / "Validation_Report.md").read_text(encoding="utf-8")
    assert rendered == render_wp_v3_2_report(report)
    assert "software gate: **PASS**" in rendered
    assert "engineering ready: **false**" in rendered
