"""V2.2-C4 turbine hot-side variable-property software tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.ai.geometry19_pipeline import _run_paired_physics_shadow
from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.ai.geometry19_selection import assess_constraint_dominance
from src.physics import Environment, Geometry19


REPO_ROOT = Path(__file__).resolve().parents[1]


def _reference_design(context) -> Geometry19:
    return Geometry19.from_mapping(
        {
            item["name"]: item["reference_value"]
            for item in context.physics_runtime.design_schema["design_variables"]
        }
    )


def test_c4_profile_is_structurally_shadow_only() -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_full_variable_thermo_shadow"
    )

    assert context.profile.mode == "paired_physics_shadow"
    assert context.protocol["delivery"]["enabled"] is False
    assert context.protocol["shadow"]["forbidden_stages"] == ["D", "E", "F"]
    assert context.protocol["physics_evaluation"]["formal_use_allowed"] is False
    assert context.physics_runtime.backend_id == (
        "geometry19_v2_2_full_variable_thermo_shadow_backend"
    )
    assert set(context.physics_runtime.component_maps) == {
        "compressor",
        "turbine",
        "inlet",
        "combustor",
        "nozzle",
    }


def test_c4_reference_closes_hot_side_enthalpy_and_keeps_one_root() -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_full_variable_thermo_shadow"
    )
    result = context.physics_runtime.evaluate(_reference_design(context), Environment())

    assert result.solver.converged is True
    assert result.operating_point["flow_root_count"] == 1
    assert result.conservation.shaft_power_W == pytest.approx(0.0, abs=1.0e-9)
    assert result.stations["t5"].total_temperature_K < result.stations["t4"].total_temperature_K
    assert result.stations["t5"].total_pressure_Pa < result.stations["t4"].total_pressure_Pa
    turbine = result.components["turbine"]
    nozzle = result.components["nozzle"]
    assert turbine["thermo_model_id"] == nozzle["thermo_model_id"]
    assert abs(turbine["outlet_enthalpy_residual_J_kg"]) <= 1.0e-8
    assert abs(turbine["isentropic_efficiency_residual_J_kg"]) <= 1.0e-8
    assert abs(nozzle["exit_enthalpy_residual_J_kg"]) <= 1.0e-8


def test_c4_constraint_dominance_contract_matches_result_schema() -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_full_variable_thermo_shadow"
    )
    result = context.physics_runtime.evaluate(_reference_design(context), Environment())
    groups = {
        "thermodynamic": result.thermodynamic_constraints,
        "assembly": result.assembly_constraints,
        "structural": result.structural_constraints,
        "thermal": result.thermal_constraints,
        "manufacturing": result.manufacturing_constraints,
    }

    dominance = assess_constraint_dominance(
        {
            "physics_acceptable": True,
            "constraint_assessment": {
                "numerically_valid": True,
                "physics_domain_valid": True,
            },
            "constraint_groups": groups,
        },
        context.constraint_dominance,
    )

    assert dominance["contract_version"] == (
        "geometry19_constraint_dominance_v2_2_c4_v1"
    )
    assert dominance["hard_valid"] is True


def test_c3_profile_retains_its_fixed_property_turbine_path() -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_full_component_shadow"
    )
    result = context.physics_runtime.evaluate(_reference_design(context), Environment())

    assert result.solver.converged is True
    assert result.operating_point["mass_flow_air_kg_s"] == pytest.approx(
        0.26146989825311734, rel=1.0e-12
    )
    assert "thermo_model_id" not in result.components["turbine"]
    assert result.components["nozzle"]["thermo_model_id"] == (
        "geometry19_thermo_variable_property_shadow_v1"
    )


def test_c4_speed_sweep_remains_single_root_and_in_domain() -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_full_variable_thermo_shadow"
    )
    reference = _reference_design(context)
    flows = []
    for ratio in (0.90, 0.95, 1.00, 1.05):
        design = replace(reference, shaft_speed_rpm=reference.shaft_speed_rpm * ratio)
        result = context.physics_runtime.evaluate(design, Environment())
        assert result.solver.converged is True
        assert result.operating_point["flow_root_count"] == 1
        assert result.components["turbine"]["map_in_domain"] is True
        assert result.components["nozzle"]["map_in_domain"] is True
        flows.append(result.operating_point["mass_flow_air_kg_s"])
    assert flows == sorted(flows)


def test_c4_paired_ledger_carries_turbine_thermo_closure(tmp_path: Path) -> None:
    context = build_pipeline_context(
        REPO_ROOT, pipeline_profile="v2_2_full_variable_thermo_shadow"
    )
    generation = tmp_path / "generation"
    generation.mkdir()
    candidate = {
        "candidate_id": 41,
        "design_hash": "d" * 64,
        "geometry19": _reference_design(context).as_mapping(),
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
            "target_id": "v2-2-c4-shadow-test",
            "candidate_protocol": {"kept_candidate_ids": [41]},
        },
        context,
        output,
    )
    paired = json.loads(
        (output / "paired_physics_ledger.jsonl").read_text(encoding="utf-8")
    )

    turbine = paired["enhanced"]["turbine"]
    assert turbine["thermo_model_id"] == (
        "geometry19_thermo_variable_property_shadow_v1"
    )
    assert abs(turbine["outlet_enthalpy_residual_J_kg"]) <= 1.0e-8
    assert summary["enhanced_turbine_map_in_domain_count"] == 1
    assert summary["enhanced_turbine_thermo_closure_evaluated_count"] == 1
    assert summary["enhanced_turbine_max_abs_enthalpy_residual_J_kg"] <= 1.0e-8
