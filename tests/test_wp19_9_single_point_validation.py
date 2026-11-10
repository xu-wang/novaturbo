"""Tests for the frozen WP19-9 single-point validation gate."""

import json
from pathlib import Path

import yaml

from src.physics.single_point_validation_v2 import (
    render_wp19_9_markdown,
    run_wp19_9_validation,
    write_wp19_9_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPO_ROOT
    / "config"
    / "physics"
    / "wp19_9_single_point_validation_v1.yaml"
)


def test_frozen_wp19_9_protocol_passes_all_representative_cases():
    report = run_wp19_9_validation(PROTOCOL, REPO_ROOT)
    assert report["status"] == "PASS"
    assert report["wp19_9_gate_passed"] is True
    assert report["wp19_10_entry_authorized"] is True
    assert report["passed_case_count"] == report["case_count"] == 3
    assert report["calibration_status"] == "bootstrap_candidate"
    assert report["formal_dataset_generation_ready"] is False


def test_protocol_covers_subsonic_choked_and_boundary_semantics():
    report = run_wp19_9_validation(PROTOCOL, REPO_ROOT)
    cases = {case["id"]: case for case in report["cases"]}
    assert cases["reference_ground_subsonic"]["nozzle_state_consistency"][
        "choked"
    ] is False
    choked = cases["reference_flight_choked"]
    assert choked["nozzle_state_consistency"]["choked"] is True
    assert choked["nozzle_state_consistency"]["exit_mach"] == 1.0
    boundary = cases["lattice_cell_upper_boundary"]
    assert boundary["design_overrides"] == {"lattice_cell_size_m": 0.014}
    assert boundary["solver"]["failure_code"] is None
    assert boundary["classification"]["feasible"] is False
    assert boundary["checks"]["soft_infeasible_semantics"] is True


def test_report_contains_complete_station_component_and_residual_records():
    report = run_wp19_9_validation(PROTOCOL, REPO_ROOT)
    for case in report["cases"]:
        assert tuple(case["stations"]) == (
            "t2",
            "t3",
            "t4",
            "t5",
            "nozzle_total",
            "exit",
        )
        assert tuple(case["components"]) == (
            "compressor",
            "combustor",
            "turbine",
            "nozzle",
        )
        assert case["conservation"]["passed"] is True
        assert case["nozzle_state_consistency"]["passed"] is True
        assert case["constraints"]["group_union_matches"] is True


def test_tighter_than_observed_energy_tolerance_fails_gate(tmp_path):
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    protocol["tolerances"]["combustor_energy_abs_W"] = 1.0e-13
    path = tmp_path / "strict_protocol.yaml"
    path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    report = run_wp19_9_validation(path, REPO_ROOT)
    assert report["status"] == "FAIL"
    assert report["wp19_10_entry_authorized"] is False
    assert report["passed_case_count"] < report["case_count"]


def test_artifact_writers_emit_strict_json_and_markdown(tmp_path):
    report = run_wp19_9_validation(PROTOCOL, REPO_ROOT)
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    write_wp19_9_artifacts(report, json_path, markdown_path)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded == report
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown == render_wp19_9_markdown(report)
    assert "WP19-9 单点守恒验证报告" in markdown
