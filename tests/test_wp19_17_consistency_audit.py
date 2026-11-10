"""Tests for WP19-17 mesh measurement and tolerance contracts."""

import numpy as np

from scripts.run_wp19_17_consistency_audit import (
    infer_combustor_lattice_threshold,
    load_tolerances,
    measure_diameter_mm,
    metric,
)


def test_consistency_tolerances_are_quality_specific_and_positive():
    _, preview = load_tolerances("preview_mesh")
    _, export = load_tolerances("export_mesh")
    assert preview["lattice_threshold_absolute_tolerance"] > export["lattice_threshold_absolute_tolerance"]
    assert all(value > 0.0 for value in preview.values())


def test_diameter_probe_selects_expected_ring_at_axial_station():
    theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    ring_40 = np.column_stack((20.0 * np.cos(theta), 20.0 * np.sin(theta), np.full_like(theta, 10.0)))
    ring_60 = np.column_stack((30.0 * np.cos(theta), 30.0 * np.sin(theta), np.full_like(theta, 20.0)))
    vertices = np.vstack((ring_40, ring_60))
    assert np.isclose(measure_diameter_mm(vertices, 60.0, 20.0), 60.0)


def test_inverse_gyroid_probe_recovers_mesh_iso_threshold():
    design = {
        "combustor_inner_diameter_m": 0.06,
        "combustor_outer_diameter_m": 0.11,
        "combustor_liner_thickness_m": 0.0012,
        "lattice_cell_size_m": 0.008,
    }
    r_inner = 31.7
    r_outer = 53.0
    r_average = 0.5 * (r_inner + r_outer)
    scale = 2.0 * np.pi / 8.0
    u = np.linspace(0.0, 2.0 * np.pi * r_average, 4000, endpoint=False)
    v = np.zeros_like(u)
    w = np.zeros_like(u)
    field = np.sin(u * scale)
    target = 0.3
    chosen = np.argsort(np.abs(np.abs(field) - target))[:200]
    theta = u[chosen] / r_average
    vertices = np.column_stack((r_inner * np.cos(theta), r_inner * np.sin(theta), w[chosen]))
    inferred = infer_combustor_lattice_threshold(vertices, design, 0.0, 1.5)
    assert abs(inferred - target) < 0.01


def test_metric_reports_absolute_and_relative_error():
    result = metric("diameter", 100.0, 100.02, absolute_tolerance=0.05, unit="mm")
    assert result["passed"] is True
    assert result["relative_error"] > 0.0
