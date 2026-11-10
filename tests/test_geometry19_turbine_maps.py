"""Software-only V2.2-B turbine-map closure tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.ai.geometry19_pipeline import _run_paired_physics_shadow
from src.physics import Environment, Geometry19
from src.physics.maps import MapDomainError, MapPolicy, load_component_map
from src.physics.maps.boundary_probe import probe_axis_boundaries
from src.physics.maps.validation import validate_component_map_package


REPO_ROOT = Path(__file__).resolve().parents[1]
TURBINE_MAP_PATH = (
    REPO_ROOT
    / "config/physics/maps/turbine/synthetic_single_stage_turbine_v1"
)


def _reference_design(context) -> Geometry19:
    return Geometry19.from_mapping(
        {
            item["name"]: item["reference_value"]
            for item in context.physics_runtime.design_schema["design_variables"]
        }
    )


def test_synthetic_turbine_map_is_structurally_shadow_ready_only() -> None:
    component_map = load_component_map(TURBINE_MAP_PATH)
    report = validate_component_map_package(TURBINE_MAP_PATH)

    assert component_map.component_type == "turbine"
    assert component_map.formal_use_allowed is False
    assert report["structural_pass"] is True
    assert report["shadow_ready"] is True
    assert report["engineering_ready"] is False
    assert report["checks"]["minimum_capacity_below_maximum_capacity"] is True
    assert report["checks"]["efficiency_strictly_between_zero_and_one"] is True
    with pytest.raises(MapDomainError, match="not approved for formal use"):
        component_map.evaluate(
            {
                "corrected_speed_ratio": 1.0,
                "corrected_mass_flux_ratio": 1.0,
            },
            policy=MapPolicy.FORMAL,
        )


def test_turbine_capacity_boundaries_have_expected_signs() -> None:
    component_map = load_component_map(TURBINE_MAP_PATH)
    probe = probe_axis_boundaries(
        component_map,
        fixed_coordinates={"corrected_speed_ratio": 1.0},
        axis_name="corrected_mass_flux_ratio",
        center_value=1.0,
        minimum_margin_name="minimum_corrected_mass_flux_ratio",
        maximum_margin_name="maximum_corrected_mass_flux_ratio",
        fraction=0.01,
    )

    assert probe["minimum_boundary"] < probe["maximum_boundary"]
    assert probe["pass"] is True
    assert set(probe["signed_margins"]) == {
        "below_minimum",
        "inside_near_minimum",
        "inside_near_maximum",
        "above_maximum",
    }
def test_comp_turb_shadow_keeps_one_root_and_exact_shaft_balance() -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_comp_turb_shadow"
    )
    runtime = context.physics_runtime
    result = runtime.evaluate(_reference_design(context), Environment())

    assert context.profile.mode == "paired_physics_shadow"
    assert context.protocol["delivery"]["enabled"] is False
    assert context.protocol["shadow"]["forbidden_stages"] == ["D", "E", "F"]
    assert runtime.backend_id == "geometry19_v2_2_comp_turb_shadow_backend"
    assert set(runtime.component_maps) == {"compressor", "turbine"}
    assert result.solver.converged is True
    assert result.operating_point["flow_root_count"] == 1
    assert result.conservation.shaft_power_W == pytest.approx(0.0, abs=1.0e-9)
    assert result.stations["t5"].total_temperature_K < (
        result.stations["t4"].total_temperature_K
    )
    assert result.stations["t5"].total_pressure_Pa < (
        result.stations["t4"].total_pressure_Pa
    )
    turbine = result.components["turbine"]
    assert turbine["map_id"] == "synthetic_single_stage_turbine_v1"
    assert turbine["map_in_domain"] is True
    assert 0.0 < turbine["total_to_total_efficiency"] < 1.0
    assert turbine["minimum_capacity_margin"] > 0.0
    assert turbine["maximum_capacity_margin"] > 0.0


def test_comp_turb_shadow_speed_sweep_has_traceable_single_roots() -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_comp_turb_shadow"
    )
    reference = _reference_design(context)
    mass_flows = []
    for ratio in (0.85, 0.90, 0.95, 1.00, 1.05):
        design = replace(
            reference, shaft_speed_rpm=reference.shaft_speed_rpm * ratio
        )
        result = context.physics_runtime.evaluate(design, Environment())
        assert result.solver.converged is True
        assert result.operating_point["flow_root_count"] == 1
        assert result.components["turbine"]["map_in_domain"] is True
        mass_flows.append(result.operating_point["mass_flow_air_kg_s"])

    assert mass_flows == sorted(mass_flows)
    assert max(
        (right - left) / left
        for left, right in zip(mass_flows[:-1], mass_flows[1:], strict=True)
    ) < 0.2


def test_comp_turb_paired_shadow_ledger_includes_turbine_map(
    tmp_path: Path,
) -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_comp_turb_shadow"
    )
    design = _reference_design(context)
    generation = tmp_path / "generation"
    generation.mkdir()
    candidate = {
        "candidate_id": 11,
        "design_hash": "b" * 64,
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
            "target_id": "v2-2-b-shadow-test",
            "candidate_protocol": {"kept_candidate_ids": [11]},
        },
        context,
        output,
    )
    paired = json.loads(
        (output / "paired_physics_ledger.jsonl").read_text(encoding="utf-8")
    )

    assert paired["enhanced"]["turbine"]["map_id"] == (
        "synthetic_single_stage_turbine_v1"
    )
    assert summary["enhanced_turbine_map_evaluated_count"] == 1
    assert summary["enhanced_turbine_map_in_domain_count"] == 1
    assert len(summary["turbine_map_hash"]) == 64
