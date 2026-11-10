"""Software-only V2.2-C3 nozzle-map and variable-property tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.ai.geometry19_pipeline import _run_paired_physics_shadow
from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.physics import Environment, Geometry19, derive_geometry_state
from src.physics.combustor_map_enhanced import solve_combustor_map_enhanced
from src.physics.compressor_map_enhanced import solve_compressor_map_enhanced
from src.physics.inlet_map_enhanced import solve_inlet_map_enhanced
from src.physics.maps import MapDomainError, MapPolicy, load_component_map
from src.physics.maps.validation import validate_component_map_package
from src.physics.nozzle_map_enhanced import solve_nozzle_map_enhanced
from src.physics.turbine_map_enhanced import solve_turbine_map_enhanced


REPO_ROOT = Path(__file__).resolve().parents[1]
NOZZLE_MAP_PATH = (
    REPO_ROOT / "config/physics/maps/nozzle/synthetic_convergent_nozzle_v1"
)


def _reference_design(context) -> Geometry19:
    return Geometry19.from_mapping(
        {
            item["name"]: item["reference_value"]
            for item in context.physics_runtime.design_schema["design_variables"]
        }
    )


def _reference_turbine(context, design: Geometry19, mass_flow_air: float):
    runtime = context.physics_runtime
    geometry = derive_geometry_state(design, runtime.model_config)
    inlet, _, _ = solve_inlet_map_enhanced(
        geometry,
        Environment(),
        runtime.model_config,
        component_map=runtime.component_maps["inlet"],
        thermo=runtime.thermo_model,
    )
    compressor = solve_compressor_map_enhanced(
        geometry,
        mass_flow_air,
        inlet,
        runtime.model_config,
        component_map=runtime.component_maps["compressor"],
        thermo=runtime.thermo_model,
    )
    combustor = solve_combustor_map_enhanced(
        geometry,
        mass_flow_air,
        compressor.outlet,
        runtime.model_config,
        component_map=runtime.component_maps["combustor"],
        thermo=runtime.thermo_model,
    )
    turbine = solve_turbine_map_enhanced(
        geometry,
        combustor,
        compressor.shaft_power_W,
        runtime.model_config,
        component_map=runtime.component_maps["turbine"],
    )
    return geometry, turbine


def test_synthetic_nozzle_map_is_structurally_shadow_ready_only() -> None:
    component_map = load_component_map(NOZZLE_MAP_PATH)
    report = validate_component_map_package(NOZZLE_MAP_PATH)

    assert component_map.component_type == "nozzle"
    assert component_map.formal_use_allowed is False
    assert report["structural_pass"] is True
    assert report["shadow_ready"] is True
    assert report["engineering_ready"] is False
    assert report["checks"]["minimum_reynolds_below_maximum"] is True
    assert report["checks"]["pressure_recovery_in_zero_one"] is True
    assert report["checks"]["discharge_coefficient_in_zero_one"] is True
    with pytest.raises(MapDomainError, match="not approved for formal use"):
        component_map.evaluate(
            {"inlet_pressure_ratio": 1.6, "reynolds_ratio": 1.0},
            policy=MapPolicy.FORMAL,
        )


def test_nozzle_map_reynolds_boundaries_and_coefficients_are_physical() -> None:
    component_map = load_component_map(NOZZLE_MAP_PATH)
    result = component_map.evaluate(
        {"inlet_pressure_ratio": 1.6, "reynolds_ratio": 1.0}
    )

    assert result.margins["minimum_reynolds_ratio"] < 1.0
    assert result.margins["maximum_reynolds_ratio"] > 1.0
    assert 0.0 < result.values["total_pressure_recovery"] <= 1.0
    assert 0.0 < result.values["discharge_coefficient"] <= 1.0


def test_full_component_shadow_has_variable_property_nozzle_and_one_root() -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_full_component_shadow"
    )
    result = context.physics_runtime.evaluate(_reference_design(context), Environment())

    assert context.physics_runtime.backend_id == (
        "geometry19_v2_2_full_component_shadow_backend"
    )
    assert set(context.physics_runtime.component_maps) == {
        "compressor",
        "turbine",
        "inlet",
        "combustor",
        "nozzle",
    }
    assert result.solver.converged is True
    assert result.operating_point["flow_root_count"] == 1
    nozzle = result.components["nozzle"]
    assert nozzle["map_id"] == "synthetic_convergent_nozzle_v1"
    assert nozzle["map_in_domain"] is True
    assert nozzle["minimum_reynolds_margin"] > 0.0
    assert nozzle["maximum_reynolds_margin"] > 0.0
    assert abs(nozzle["exit_enthalpy_residual_J_kg"]) < 1.0e-8
    assert abs(result.conservation.normalized_flow) < 1.0e-5


def test_variable_property_nozzle_resolves_choked_and_unchoked_branches() -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_full_component_shadow"
    )
    design = _reference_design(context)
    matched = context.physics_runtime.evaluate(design, Environment())
    geometry, turbine = _reference_turbine(
        context, design, matched.operating_point["mass_flow_air_kg_s"]
    )
    kwargs = {
        "component_map": context.physics_runtime.component_maps["nozzle"],
        "thermo": context.physics_runtime.thermo_model,
    }
    choked = solve_nozzle_map_enhanced(
        geometry,
        turbine,
        Environment(pressure_Pa=60000.0),
        context.physics_runtime.model_config,
        **kwargs,
    )
    unchoked = solve_nozzle_map_enhanced(
        geometry,
        turbine,
        Environment(pressure_Pa=110000.0),
        context.physics_runtime.model_config,
        **kwargs,
    )

    assert choked.choked is True
    assert choked.exit.mach == pytest.approx(1.0, abs=1.0e-10)
    assert choked.pressure_thrust_N > 0.0
    assert unchoked.choked is False
    assert 0.0 < unchoked.exit.mach < 1.0
    assert unchoked.pressure_thrust_N == pytest.approx(0.0, abs=1.0e-12)
    assert abs(choked.exit_enthalpy_residual_J_kg) < 1.0e-8
    assert abs(unchoked.exit_enthalpy_residual_J_kg) < 1.0e-8


def test_nozzle_shadow_speed_sweep_retains_traceable_roots() -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_full_component_shadow"
    )
    reference = _reference_design(context)
    flows = []
    for ratio in (0.90, 0.95, 1.00, 1.05):
        result = context.physics_runtime.evaluate(
            replace(reference, shaft_speed_rpm=reference.shaft_speed_rpm * ratio),
            Environment(),
        )
        assert result.solver.converged is True
        assert result.operating_point["flow_root_count"] == 1
        assert result.components["nozzle"]["map_in_domain"] is True
        flows.append(result.operating_point["mass_flow_air_kg_s"])
    assert flows == sorted(flows)


def test_nozzle_paired_shadow_ledger_includes_nozzle_map(tmp_path: Path) -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_full_component_shadow"
    )
    design = _reference_design(context)
    generation = tmp_path / "generation"
    generation.mkdir()
    candidate = {
        "candidate_id": 19,
        "design_hash": "e" * 64,
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
            "target_id": "v2-2-c3-shadow-test",
            "candidate_protocol": {"kept_candidate_ids": [19]},
        },
        context,
        output,
    )
    paired = json.loads(
        (output / "paired_physics_ledger.jsonl").read_text(encoding="utf-8")
    )

    assert paired["enhanced"]["nozzle"]["map_id"] == (
        "synthetic_convergent_nozzle_v1"
    )
    assert summary["enhanced_nozzle_map_evaluated_count"] == 1
    assert summary["enhanced_nozzle_map_in_domain_count"] == 1
    assert len(summary["nozzle_map_hash"]) == 64
