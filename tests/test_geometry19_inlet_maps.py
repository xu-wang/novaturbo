"""Software-only V2.2-C1 inlet-map and total-state closure tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ai.geometry19_pipeline import _run_paired_physics_shadow
from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.physics import Environment, Geometry19
from src.physics.maps import MapDomainError, MapPolicy, load_component_map
from src.physics.maps.validation import validate_component_map_package


REPO_ROOT = Path(__file__).resolve().parents[1]
INLET_MAP_PATH = (
    REPO_ROOT / "config/physics/maps/inlet/synthetic_subsonic_inlet_v1"
)


def _reference_design(context) -> Geometry19:
    return Geometry19.from_mapping(
        {
            item["name"]: item["reference_value"]
            for item in context.physics_runtime.design_schema["design_variables"]
        }
    )


def test_synthetic_inlet_map_is_structurally_shadow_ready_only() -> None:
    component_map = load_component_map(INLET_MAP_PATH)
    report = validate_component_map_package(INLET_MAP_PATH)

    assert component_map.component_type == "inlet"
    assert component_map.formal_use_allowed is False
    assert report["structural_pass"] is True
    assert report["shadow_ready"] is True
    assert report["engineering_ready"] is False
    assert report["checks"]["minimum_capture_area_below_maximum"] is True
    assert report["checks"]["pressure_recovery_in_zero_one"] is True
    assert report["checks"]["discharge_coefficient_in_zero_one"] is True
    with pytest.raises(MapDomainError, match="not approved for formal use"):
        component_map.evaluate(
            {"flight_mach": 0.2, "capture_area_ratio": 0.8},
            policy=MapPolicy.FORMAL,
        )


def test_inlet_boundaries_and_mach_trends_have_expected_signs() -> None:
    component_map = load_component_map(INLET_MAP_PATH)
    low_mach = component_map.evaluate(
        {"flight_mach": 0.0, "capture_area_ratio": 0.8}
    )
    high_mach = component_map.evaluate(
        {"flight_mach": 0.6, "capture_area_ratio": 0.8}
    )
    minimum = low_mach.margins["minimum_capture_area_ratio"]
    maximum = low_mach.margins["maximum_capture_area_ratio"]

    assert minimum < 0.8 < maximum
    assert low_mach.values["total_pressure_recovery"] > (
        high_mach.values["total_pressure_recovery"]
    )
    assert low_mach.values["discharge_coefficient"] > (
        high_mach.values["discharge_coefficient"]
    )


def test_inlet_shadow_uses_variable_property_total_state_and_one_root() -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_comp_turb_inlet_shadow"
    )
    runtime = context.physics_runtime
    result = runtime.evaluate(_reference_design(context), Environment(mach=0.2))

    assert runtime.backend_id == "geometry19_v2_2_comp_turb_inlet_shadow_backend"
    assert set(runtime.component_maps) == {"compressor", "turbine", "inlet"}
    assert result.solver.converged is True
    assert result.operating_point["flow_root_count"] == 1
    inlet = result.components["inlet"]
    assert inlet["map_id"] == "synthetic_subsonic_inlet_v1"
    assert inlet["map_in_domain"] is True
    assert inlet["freestream_total_temperature_K"] > 288.15
    assert inlet["freestream_total_pressure_Pa"] > 101325.0
    assert result.stations["t2"].total_pressure_Pa < (
        inlet["freestream_total_pressure_Pa"]
    )
    assert abs(inlet["total_enthalpy_residual_J_kg"]) < 1.0e-8
    assert inlet["minimum_capture_margin"] > 0.0
    assert inlet["maximum_capture_margin"] > 0.0


def test_inlet_shadow_subsonic_sweep_has_traceable_single_roots() -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_comp_turb_inlet_shadow"
    )
    design = _reference_design(context)
    mass_flows = []
    recoveries = []
    for mach in (0.0, 0.1, 0.2, 0.3):
        result = context.physics_runtime.evaluate(design, Environment(mach=mach))
        assert result.solver.converged is True
        assert result.operating_point["flow_root_count"] == 1
        assert result.components["inlet"]["map_in_domain"] is True
        mass_flows.append(result.operating_point["mass_flow_air_kg_s"])
        recoveries.append(result.components["inlet"]["total_pressure_recovery"])

    assert mass_flows == sorted(mass_flows)
    assert recoveries == sorted(recoveries, reverse=True)
    assert max(
        (right - left) / left
        for left, right in zip(mass_flows[:-1], mass_flows[1:], strict=True)
    ) < 0.1


def test_inlet_paired_shadow_ledger_includes_inlet_map(tmp_path: Path) -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_comp_turb_inlet_shadow"
    )
    design = _reference_design(context)
    generation = tmp_path / "generation"
    generation.mkdir()
    candidate = {
        "candidate_id": 13,
        "design_hash": "c" * 64,
        "geometry19": design.as_mapping(),
        "valid": True,
    }
    (generation / "candidate_filter_ledger.jsonl").write_text(
        json.dumps(candidate) + "\n", encoding="utf-8"
    )
    output = tmp_path / "shadow"
    summary = _run_paired_physics_shadow(
        REPO_ROOT,
        generation,
        {
            "target_id": "v2-2-c1-shadow-test",
            "candidate_protocol": {"kept_candidate_ids": [13]},
        },
        context,
        output,
    )
    paired = json.loads(
        (output / "paired_physics_ledger.jsonl").read_text(encoding="utf-8")
    )

    assert paired["enhanced"]["inlet"]["map_id"] == "synthetic_subsonic_inlet_v1"
    assert summary["enhanced_inlet_map_evaluated_count"] == 1
    assert summary["enhanced_inlet_map_in_domain_count"] == 1
    assert len(summary["inlet_map_hash"]) == 64
