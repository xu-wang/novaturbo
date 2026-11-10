"""Software-only V2.2-C2 combustor-map and local FAR closure tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.ai.geometry19_pipeline import _run_paired_physics_shadow
from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.physics import Environment, Geometry19
from src.physics.maps import MapDomainError, MapPolicy, load_component_map
from src.physics.maps.validation import validate_component_map_package


REPO_ROOT = Path(__file__).resolve().parents[1]
COMBUSTOR_MAP_PATH = (
    REPO_ROOT / "config/physics/maps/combustor/synthetic_annular_combustor_v1"
)


def _reference_design(context) -> Geometry19:
    return Geometry19.from_mapping(
        {
            item["name"]: item["reference_value"]
            for item in context.physics_runtime.design_schema["design_variables"]
        }
    )


def test_synthetic_combustor_map_is_structurally_shadow_ready_only() -> None:
    component_map = load_component_map(COMBUSTOR_MAP_PATH)
    report = validate_component_map_package(COMBUSTOR_MAP_PATH)

    assert component_map.component_type == "combustor"
    assert component_map.formal_use_allowed is False
    assert report["structural_pass"] is True
    assert report["shadow_ready"] is True
    assert report["engineering_ready"] is False
    assert report["checks"]["lean_boundary_below_rich_boundary"] is True
    assert report["checks"]["combustion_efficiency_in_zero_one"] is True
    assert report["checks"]["pressure_loss_fraction_in_zero_one"] is True
    with pytest.raises(MapDomainError, match="not approved for formal use"):
        component_map.evaluate(
            {"mass_flux_ratio": 1.0, "equivalence_ratio": 0.35},
            policy=MapPolicy.FORMAL,
        )


def test_combustor_map_stability_boundaries_and_loss_trends() -> None:
    component_map = load_component_map(COMBUSTOR_MAP_PATH)
    low_flux = component_map.evaluate(
        {"mass_flux_ratio": 0.5, "equivalence_ratio": 0.35}
    )
    high_flux = component_map.evaluate(
        {"mass_flux_ratio": 1.5, "equivalence_ratio": 0.35}
    )

    assert low_flux.margins["minimum_equivalence_ratio"] < 0.35
    assert low_flux.margins["maximum_equivalence_ratio"] > 0.35
    assert high_flux.values["total_pressure_loss_fraction"] > (
        low_flux.values["total_pressure_loss_fraction"]
    )
    assert 0.0 < high_flux.values["combustion_efficiency"] <= 1.0


def test_combustor_shadow_has_local_far_root_and_exact_energy_closure() -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_comp_turb_inlet_comb_shadow"
    )
    result = context.physics_runtime.evaluate(_reference_design(context), Environment())

    assert context.physics_runtime.backend_id == (
        "geometry19_v2_2_comp_turb_inlet_comb_shadow_backend"
    )
    assert set(context.physics_runtime.component_maps) == {
        "compressor",
        "turbine",
        "inlet",
        "combustor",
    }
    assert result.solver.converged is True
    assert result.operating_point["flow_root_count"] == 1
    combustor = result.components["combustor"]
    assert combustor["map_id"] == "synthetic_annular_combustor_v1"
    assert combustor["map_in_domain"] is True
    assert combustor["fuel_solution_iterations"] > 0
    assert 0.0 < combustor["combustion_efficiency"] <= 1.0
    assert combustor["lean_stability_margin"] > 0.0
    assert combustor["rich_stability_margin"] > 0.0
    assert abs(result.conservation.combustor_energy_W) < 1.0e-6
    assert result.operating_point["mass_flow_gas_kg_s"] == pytest.approx(
        result.operating_point["mass_flow_air_kg_s"]
        + result.operating_point["mass_flow_fuel_kg_s"],
        abs=1.0e-14,
    )


def test_combustor_shadow_speed_sweep_retains_single_traceable_roots() -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_comp_turb_inlet_comb_shadow"
    )
    reference = _reference_design(context)
    mass_flows = []
    fars = []
    for ratio in (0.90, 0.95, 1.00, 1.05):
        design = replace(
            reference, shaft_speed_rpm=reference.shaft_speed_rpm * ratio
        )
        result = context.physics_runtime.evaluate(design, Environment())
        assert result.solver.converged is True
        assert result.operating_point["flow_root_count"] == 1
        assert result.components["combustor"]["map_in_domain"] is True
        assert abs(result.conservation.combustor_energy_W) < 1.0e-6
        mass_flows.append(result.operating_point["mass_flow_air_kg_s"])
        fars.append(result.operating_point["fuel_air_ratio"])

    assert mass_flows == sorted(mass_flows)
    assert all(0.0 < value < 0.1 for value in fars)


def test_combustor_paired_shadow_ledger_includes_combustor_map(
    tmp_path: Path,
) -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_comp_turb_inlet_comb_shadow"
    )
    design = _reference_design(context)
    generation = tmp_path / "generation"
    generation.mkdir()
    candidate = {
        "candidate_id": 17,
        "design_hash": "d" * 64,
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
            "target_id": "v2-2-c2-shadow-test",
            "candidate_protocol": {"kept_candidate_ids": [17]},
        },
        context,
        output,
    )
    paired = json.loads(
        (output / "paired_physics_ledger.jsonl").read_text(encoding="utf-8")
    )

    assert paired["enhanced"]["combustor"]["map_id"] == (
        "synthetic_annular_combustor_v1"
    )
    assert summary["enhanced_combustor_map_evaluated_count"] == 1
    assert summary["enhanced_combustor_map_in_domain_count"] == 1
    assert len(summary["combustor_map_hash"]) == 64
