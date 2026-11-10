#!/usr/bin/env python3
"""Audit GeometryState, generated STL measurements, and V2.1 physics identity."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.physics import ModelConfig  # noqa: E402


DEFAULT_REGRESSION_ROOT = (
    REPO_ROOT / "results" / "visualization" / "wp19_16_geometry19_regression_20260825"
)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "physics" / "wp19_17_geometry_physics_consistency_v1.json"
TOLERANCE_PATH = REPO_ROOT / "config" / "physics" / "geometry19_consistency_tolerances_v1.yaml"
CALIBRATION_PATH = REPO_ROOT / "config" / "physics" / "geometry19_calibration_bootstrap_v1.yaml"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_tolerances(quality: str) -> tuple[dict[str, Any], dict[str, float]]:
    payload = yaml.safe_load(TOLERANCE_PATH.read_text(encoding="utf-8"))
    if payload.get("profile_version") != TOLERANCE_PATH.stem:
        raise ValueError("Consistency tolerance filename/version mismatch")
    try:
        selected = payload["qualities"][quality]
    except KeyError as exc:
        raise ValueError(f"Unsupported consistency quality: {quality}") from exc
    required = {
        "radial_absolute_tolerance_mm",
        "axial_absolute_tolerance_mm",
        "lattice_threshold_absolute_tolerance",
    }
    if set(selected) != required:
        raise ValueError("Consistency tolerance quality contract mismatch")
    values = {name: float(value) for name, value in selected.items()}
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError("Consistency tolerances must be finite and positive")
    return payload, values


def _load_vertices(path: Path) -> np.ndarray:
    import trimesh

    mesh = trimesh.load_mesh(path, process=False, force="mesh")
    return np.asarray(mesh.vertices, dtype=float)


def measure_diameter_mm(
    vertices: np.ndarray,
    expected_diameter_mm: float,
    z_target_mm: float | None = None,
    z_tolerance_mm: float = 0.051,
) -> float:
    selected = vertices
    if z_target_mm is not None:
        mask = np.abs(vertices[:, 2] - z_target_mm) <= z_tolerance_mm
        if not np.any(mask):
            nearest_z = vertices[np.argmin(np.abs(vertices[:, 2] - z_target_mm)), 2]
            mask = np.isclose(vertices[:, 2], nearest_z, atol=1.0e-6)
        selected = vertices[mask]
    radii = np.linalg.norm(selected[:, :2], axis=1)
    expected_radius = expected_diameter_mm / 2.0
    return 2.0 * float(radii[np.argmin(np.abs(radii - expected_radius))])


def infer_combustor_lattice_threshold(
    vertices: np.ndarray,
    design: Mapping[str, Any],
    combustor_z_min_mm: float,
    casing_thickness_mm: float,
) -> float:
    r_fill_inner = (
        float(design["combustor_inner_diameter_m"]) * 500.0
        + float(design["combustor_liner_thickness_m"]) * 1000.0
        + 0.5
    )
    r_fill_outer = (
        float(design["combustor_outer_diameter_m"]) * 500.0
        - casing_thickness_mm
        - 0.5
    )
    r_average = 0.5 * (r_fill_inner + r_fill_outer)
    radii = np.linalg.norm(vertices[:, :2], axis=1)
    theta = np.mod(np.arctan2(vertices[:, 1], vertices[:, 0]), 2.0 * np.pi)
    u = theta * r_average
    v = radii - r_fill_inner
    w = vertices[:, 2] - combustor_z_min_mm
    scale = 2.0 * np.pi / (float(design["lattice_cell_size_m"]) * 1000.0)
    field = (
        np.sin(u * scale) * np.cos(v * scale)
        + np.sin(v * scale) * np.cos(w * scale)
        + np.sin(w * scale) * np.cos(u * scale)
    )
    return float(np.median(np.abs(field)))


def metric(
    name: str,
    expected: float,
    measured: float,
    *,
    absolute_tolerance: float | None = None,
    relative_tolerance: float | None = None,
    unit: str = "",
) -> dict[str, Any]:
    absolute_error = abs(measured - expected)
    relative_error = absolute_error / max(abs(expected), 1.0e-12)
    checks = []
    if absolute_tolerance is not None:
        checks.append(absolute_error <= absolute_tolerance)
    if relative_tolerance is not None:
        checks.append(relative_error <= relative_tolerance)
    if not checks:
        raise ValueError(f"Metric {name} has no tolerance")
    return {
        "name": name,
        "expected": expected,
        "measured": measured,
        "unit": unit,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "passed": all(checks),
    }


def _annulus_area_m2(outer_diameter_mm: float, inner_diameter_mm: float) -> float:
    return math.pi * 0.25 * (outer_diameter_mm**2 - inner_diameter_mm**2) * 1.0e-6


def audit_case(
    case_root: Path,
    quality: str,
    tolerances: Mapping[str, float],
    casing_thickness_mm: float,
) -> dict[str, Any]:
    design = _read_json(case_root / "geometry19.json")
    derived = _read_json(case_root / "derived_geometry.json")
    physics_result = _read_json(case_root / "physics_result.json")
    physics = physics_result["physics"]
    diagnostics = _read_json(case_root / "mesh_diagnostics.json")
    component_diagnostics = diagnostics["component_diagnostics"]
    stl_root = case_root / "component_stl"
    radial_tolerance = tolerances["radial_absolute_tolerance_mm"]
    axial_tolerance = tolerances["axial_absolute_tolerance_mm"]

    compressor_vertices = _load_vertices(stl_root / "compressor.stl")
    combustor_vertices = _load_vertices(stl_root / "combustor.stl")
    turbine_vertices = _load_vertices(stl_root / "turbine.stl")
    nozzle_vertices = _load_vertices(stl_root / "nozzle.stl")
    lattice_vertices = _load_vertices(stl_root / "combustor_lattice.stl")

    compressor_z_max = component_diagnostics["compressor"]["bounds_max_mm"][2]
    combustor_z_min = component_diagnostics["combustor"]["bounds_min_mm"][2]
    nozzle_z_max = component_diagnostics["nozzle"]["bounds_max_mm"][2]
    dc2 = measure_diameter_mm(
        compressor_vertices,
        design["compressor_impeller_exit_diameter_m"] * 1000.0,
        compressor_z_max,
        radial_tolerance,
    )
    dbo = measure_diameter_mm(
        combustor_vertices,
        design["combustor_outer_diameter_m"] * 1000.0,
        combustor_z_min,
        radial_tolerance,
    )
    dbi = measure_diameter_mm(
        combustor_vertices,
        design["combustor_inner_diameter_m"] * 1000.0,
        combustor_z_min,
        radial_tolerance,
    )
    dt = measure_diameter_mm(
        turbine_vertices, design["turbine_tip_diameter_m"] * 1000.0
    )
    dh = measure_diameter_mm(
        turbine_vertices,
        2.0 * derived["turbine_hub_radius_m"] * 1000.0,
    )
    throat = measure_diameter_mm(
        nozzle_vertices,
        design["nozzle_throat_diameter_m"] * 1000.0,
        nozzle_z_max,
        radial_tolerance,
    )

    metrics = []
    for name, expected, measured in (
        ("compressor_exit_diameter", design["compressor_impeller_exit_diameter_m"] * 1000.0, dc2),
        ("combustor_outer_diameter", design["combustor_outer_diameter_m"] * 1000.0, dbo),
        ("combustor_inner_diameter", design["combustor_inner_diameter_m"] * 1000.0, dbi),
        ("turbine_tip_diameter", design["turbine_tip_diameter_m"] * 1000.0, dt),
        ("turbine_hub_diameter", 2.0 * derived["turbine_hub_radius_m"] * 1000.0, dh),
        ("nozzle_throat_diameter", design["nozzle_throat_diameter_m"] * 1000.0, throat),
    ):
        metrics.append(metric(name, expected, measured, absolute_tolerance=2.0 * radial_tolerance, unit="mm"))

    combustor_area = _annulus_area_m2(dbo, dbi)
    area_tolerance = (
        math.pi * 0.5 * (dbo + dbi) * (2.0 * radial_tolerance) * 1.0e-6
        / max(abs(derived["combustor_annulus_area_m2"]), 1.0e-12)
    )
    metrics.append(metric("combustor_annulus_area", derived["combustor_annulus_area_m2"], combustor_area, relative_tolerance=area_tolerance, unit="m^2"))

    compressor_geometric_area = math.pi * (dc2 / 1000.0) * design["compressor_impeller_exit_width_m"]
    compressor_effective_area = compressor_geometric_area * derived["compressor_exit_blockage"]
    metrics.append(metric("compressor_exit_effective_area", derived["compressor_exit_effective_area_m2"], compressor_effective_area, relative_tolerance=(2.0 * radial_tolerance) / dc2, unit="m^2"))

    turbine_area = _annulus_area_m2(dt, dh)
    turbine_area_tolerance = (
        math.pi * 0.5 * (dt + dh) * (2.0 * radial_tolerance) * 1.0e-6
        / max(abs(derived["turbine_annulus_area_m2"]), 1.0e-12)
    )
    metrics.append(metric("turbine_annulus_area", derived["turbine_annulus_area_m2"], turbine_area, relative_tolerance=turbine_area_tolerance, unit="m^2"))
    metrics.append(metric("turbine_mean_radius", derived["turbine_mean_radius_m"], (dt + dh) * 0.00025, absolute_tolerance=radial_tolerance * 0.001, unit="m"))
    metrics.append(metric("nozzle_throat_area", derived["nozzle_throat_area_m2"], math.pi * 0.25 * (throat * 0.001) ** 2, relative_tolerance=(4.0 * radial_tolerance) / throat, unit="m^2"))

    combustor_length = component_diagnostics["combustor"]["bounds_max_mm"][2] - combustor_z_min
    metrics.append(metric("combustor_axial_length", design["combustor_length_m"] * 1000.0, combustor_length, absolute_tolerance=axial_tolerance, unit="mm"))
    for upstream, downstream in (("inlet", "compressor"), ("compressor", "combustor"), ("combustor", "turbine"), ("turbine", "nozzle")):
        gap = component_diagnostics[downstream]["bounds_min_mm"][2] - component_diagnostics[upstream]["bounds_max_mm"][2]
        metrics.append(metric(f"interface_{upstream}_to_{downstream}", 0.0, gap, absolute_tolerance=axial_tolerance, unit="mm"))

    expected_threshold = min(max(float(design["lattice_relative_density"]), 0.1), 0.5)
    inferred_threshold = infer_combustor_lattice_threshold(
        lattice_vertices, design, combustor_z_min, casing_thickness_mm
    )
    metrics.append(metric("lattice_iso_threshold", expected_threshold, inferred_threshold, absolute_tolerance=tolerances["lattice_threshold_absolute_tolerance"], unit="dimensionless"))

    count_checks = {
        "compressor_all_blade_bodies": component_diagnostics["compressor"]["body_count"] == component_diagnostics["compressor"]["expected_body_count"],
        "combustor_injector_bodies": component_diagnostics["combustor"]["body_count"] - 2 == design["combustor_injector_count"],
        "turbine_all_blade_bodies": component_diagnostics["turbine"]["body_count"] == component_diagnostics["turbine"]["expected_body_count"],
    }
    turbine_physics = physics.get("components", {}).get("turbine")
    physics_assembly = physics.get("assembly_constraints", {})
    identity_checks = {
        "request_equals_geometry_state_design": design == derived["design"],
        "request_equals_physics_design": design == physics["design"],
        "assembly_residuals_equal_physics_when_available": (
            not physics_assembly
            or derived["assembly_residuals"] == physics_assembly
        ),
        "turbine_solidity_equal_physics_when_available": (
            turbine_physics is None
            or math.isclose(
                derived["turbine_solidity"],
                turbine_physics["solidity"],
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ),
        "physics_calibration_hash_current": physics["calibration_pack_hash"] == ModelConfig.from_yaml(CALIBRATION_PATH).calibration_pack_hash,
    }
    passed = all(item["passed"] for item in metrics) and all(count_checks.values()) and all(identity_checks.values())
    return {
        "case": case_root.name,
        "quality": quality,
        "metrics": metrics,
        "count_checks": count_checks,
        "identity_checks": identity_checks,
        "passed": passed,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    regression = _read_json(args.regression_root / "summary.json")
    tolerance_profile, tolerances = load_tolerances(args.quality)
    model_config = ModelConfig.from_yaml(CALIBRATION_PATH)
    casing_thickness_mm = float(model_config.require("turbine_casing_thickness_m")) * 1000.0
    records = []
    expected_hard_failures = []
    for source in regression["cases"]:
        if source.get("status") == "expected_numerical_hard_failure":
            expected_hard_failures.append(source["case"])
            continue
        case_root = args.regression_root / "cases" / source["case"]
        records.append(audit_case(case_root, args.quality, tolerances, casing_thickness_mm))
    failed_metrics = {
        record["case"]: {
            "metrics": [item["name"] for item in record["metrics"] if not item["passed"]],
            "counts": [name for name, passed in record["count_checks"].items() if not passed],
            "identity": [name for name, passed in record["identity_checks"].items() if not passed],
        }
        for record in records
        if not record["passed"]
    }
    result = {
        "work_package": "WP19-17",
        "source_work_package": regression["work_package"],
        "source_schema_version": regression["schema_version"],
        "quality": args.quality,
        "tolerance_profile": tolerance_profile["profile_version"],
        "tolerance_methodology": tolerance_profile["methods"],
        "tolerances": tolerances,
        "audited_case_count": len(records),
        "expected_hard_failure_cases": expected_hard_failures,
        "passed_count": sum(record["passed"] for record in records),
        "failed_cases": failed_metrics,
        "passed": all(record["passed"] for record in records),
        "cases": records,
    }
    _write_json(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression-root", type=Path, default=DEFAULT_REGRESSION_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quality", choices=("preview_mesh", "export_mesh"), default="preview_mesh")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    print(f"WP19-17: {result['passed_count']}/{result['audited_case_count']} passed; output={args.output}")
    if result["failed_cases"]:
        print(json.dumps(result["failed_cases"], indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
