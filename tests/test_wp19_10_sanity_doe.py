"""Tests for WP19-10 sampling, summaries, and physical-path audit."""

import csv
import json
from pathlib import Path

from src.physics import Geometry19
from src.physics.sanity_doe_v2 import (
    run_wp19_10_doe,
    sample_geometry19_designs,
    write_wp19_10_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = REPO_ROOT / "config" / "physics" / "geometry19_schema_v1.yaml"
PROTOCOL = REPO_ROOT / "config" / "physics" / "wp19_10_sanity_doe_v1.yaml"


def test_mixed_geometry19_lhs_is_reproducible_and_typed():
    first = sample_geometry19_designs(SCHEMA, 32, 77)
    second = sample_geometry19_designs(SCHEMA, 32, 77)
    assert first == second
    assert len(first) == 32
    assert all(isinstance(design, Geometry19) for design in first)
    for design in first:
        assert type(design.compressor_blade_count) is int
        assert type(design.turbine_blade_count) is int
        assert type(design.combustor_injector_count) is int


def test_smoke_doe_closes_counts_and_audits_all_19_dimensions():
    result = run_wp19_10_doe(PROTOCOL, REPO_ROOT, sample_count_override=32)
    report = result.report
    summary = report["summary"]
    assert report["effective_sample_count"] == 32
    assert len(result.sample_records) == 32
    assert summary["converged_count"] + summary["hard_failure_count"] == 32
    assert 0.0 <= summary["convergence_rate"] <= 1.0
    oat = report["sensitivity_audit"]["oat_physical_path"]
    assert set(oat["variables"]) == set(Geometry19.coordinate_names())
    assert oat["dead_dimensions"] == []
    assert report["status"] == "PASS_WITH_CALIBRATION_ACTIONS"
    assert report["wp19_10_numerical_audit_passed"] is True
    assert report["wp19_11_entry_authorized"] is True
    assert report["calibration_actions_required"] is True
    assert report["formal_dataset_generation_ready"] is False


def test_sample_ledger_preserves_hard_failure_and_soft_feasibility_semantics():
    result = run_wp19_10_doe(PROTOCOL, REPO_ROOT, sample_count_override=32)
    for record in result.sample_records:
        if record["solver_converged"]:
            assert record["failure_code"] == ""
            assert record["violated_constraint_count"] is not None
        else:
            assert record["failure_code"]
            assert record["feasible"] is False
            assert record["thrust_N"] is None


def test_doe_artifacts_are_strict_and_machine_readable(tmp_path):
    result = run_wp19_10_doe(PROTOCOL, REPO_ROOT, sample_count_override=16)
    report_path = tmp_path / "report.json"
    csv_path = tmp_path / "samples.csv"
    markdown_path = tmp_path / "report.md"
    write_wp19_10_artifacts(result, report_path, csv_path, markdown_path)
    assert json.loads(report_path.read_text(encoding="utf-8")) == result.report
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16
    assert "WP19-10 小规模 DOE 报告" in markdown_path.read_text(
        encoding="utf-8"
    )
