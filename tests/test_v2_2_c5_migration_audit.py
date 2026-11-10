"""V2.2-C5 cross-version migration audit tests."""

from pathlib import Path

from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.physics import Environment, Geometry19
from src.physics.flow_scan_diagnostics import diagnose_v2_2_c4_flow_scan
from src.physics.wp_v2_2_c5_migration_audit import (
    INCREMENTAL_PROFILES,
    render_v2_2_c5_migration_report,
    run_v2_2_c5_migration_audit,
    write_v2_2_c5_migration_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_c5_migration_audit_is_reproducible_and_not_training_ready(
    tmp_path: Path,
) -> None:
    report = run_v2_2_c5_migration_audit(REPO_ROOT, sample_limit=8)

    assert report["counts"]["evaluated"] == 8
    assert report["counts"].get("converged", 0) <= 8
    assert report["training_ready"] is False
    assert len(report["audit_code_hash"]) == 64
    assert len(report["physics_bundle_hash"]) == 64
    assert tuple(report["incremental_profile_migration"]) == INCREMENTAL_PROFILES
    assert sum(
        report["failed_flow_scan_summary"]["design_classification_counts"].values()
    ) == report["failure_codes"].get("NO_FLOW_MATCH", 0)
    assert report["failed_flow_scan_summary"]["failure_reason_trial_counts"]
    assert report["software_pipeline_migration"]["profile"] == "v2_2_development"
    assert report["software_pipeline_migration"]["counts"]["converged"] > (
        report["counts"].get("converged", 0)
    )

    output = tmp_path / "audit"
    write_v2_2_c5_migration_artifacts(report, output)
    assert (output / "migration_audit.json").is_file()
    assert (output / "Migration_Audit.md").read_text(encoding="utf-8") == (
        render_v2_2_c5_migration_report(report)
    )


def test_c4_flow_scan_diagnostic_replays_component_chain() -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_full_variable_thermo_shadow"
    )
    design = Geometry19.from_mapping(
        {
            item["name"]: item["reference_value"]
            for item in context.physics_runtime.design_schema["design_variables"]
        }
    )

    diagnostic = diagnose_v2_2_c4_flow_scan(context, design, Environment())

    assert diagnostic["scan_point_count"] > 100
    assert diagnostic["valid_trial_count"] + diagnostic["failure_trial_count"] == (
        diagnostic["scan_point_count"]
    )
    assert diagnostic["sign_change_bracket_count"] >= 1
    assert set(diagnostic["compressor_query_envelope"]) == {
        "corrected_speed_ratio",
        "flow_coefficient_ratio",
        "domain_distance",
    }
