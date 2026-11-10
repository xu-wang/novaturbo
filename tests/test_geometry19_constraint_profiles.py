"""Regression tests for the new-route WP19-11 constraint profiles."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import yaml

from src.physics import (
    ConstraintProfile,
    Environment,
    Geometry19,
    ModelConfig,
    assess_constraint_profile,
    solve_design_point,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PHYSICS_CONFIG = REPO_ROOT / "config" / "physics"


@pytest.fixture(scope="module")
def model_config() -> ModelConfig:
    return ModelConfig.from_yaml(
        PHYSICS_CONFIG / "geometry19_calibration_bootstrap_v1.yaml"
    )


@pytest.fixture(scope="module")
def reference_design() -> Geometry19:
    schema = yaml.safe_load(
        (PHYSICS_CONFIG / "geometry19_schema_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    return Geometry19.from_mapping(
        {
            item["name"]: item["reference_value"]
            for item in schema["design_variables"]
        }
    )


@pytest.fixture(scope="module")
def profiles() -> tuple[ConstraintProfile, ConstraintProfile]:
    return (
        ConstraintProfile.from_yaml(
            PHYSICS_CONFIG
            / "geometry19_constraint_profile_pipeline_debug_v1.yaml"
        ),
        ConstraintProfile.from_yaml(
            PHYSICS_CONFIG
            / "geometry19_constraint_profile_engineering_candidate_v1.yaml"
        ),
    )


def test_profile_files_are_versioned_and_have_distinct_hashes(profiles):
    debug, engineering = profiles
    assert debug.profile_version.endswith("pipeline_debug_v1")
    assert engineering.profile_version.endswith("engineering_candidate_v1")
    assert debug.source_hash != engineering.source_hash
    assert debug.require_engineering_feasible is False
    assert engineering.require_engineering_feasible is True


def test_profile_switch_changes_acceptance_only(
    reference_design, model_config, profiles
):
    result = solve_design_point(reference_design, Environment(), model_config)
    physics_before = asdict(result)

    debug = assess_constraint_profile(result, profiles[0])
    engineering = assess_constraint_profile(result, profiles[1])

    assert asdict(result) == physics_before
    assert debug.numerically_valid is True
    assert debug.physics_domain_valid is True
    assert debug.engineering_feasible is False
    assert debug.violated_constraints == engineering.violated_constraints
    assert debug.accepted is True
    assert engineering.accepted is False
    assert any(
        name.endswith("combustor_pressure_loss_fraction_max")
        for name in debug.violated_constraints
    )
    assert any(
        name.endswith("turbine_rotating_stress")
        for name in debug.violated_constraints
    )


def test_hard_geometry_failure_is_never_accepted(
    reference_design, model_config, profiles
):
    invalid = replace(
        reference_design,
        combustor_outer_diameter_m=0.05,
        combustor_inner_diameter_m=0.06,
    )
    result = solve_design_point(invalid, Environment(), model_config)
    for profile in profiles:
        assessment = assess_constraint_profile(result, profile)
        assert assessment.numerically_valid is False
        assert assessment.physics_domain_valid is False
        assert assessment.engineering_feasible is False
        assert assessment.accepted is False
        assert assessment.violated_constraints == (
            "numerical_hard.solver.GEOMETRY_INVALID",
        )


def test_assessment_mapping_exposes_required_contract(
    reference_design, model_config, profiles
):
    result = solve_design_point(reference_design, Environment(), model_config)
    payload = assess_constraint_profile(result, profiles[0]).as_mapping()
    assert {
        "numerically_valid",
        "physics_domain_valid",
        "engineering_feasible",
        "violated_constraints",
    } <= set(payload)
    assert isinstance(payload["violated_constraints"], list)


def test_map_boundary_violation_is_physics_domain_hard(
    reference_design, model_config, profiles
):
    result = solve_design_point(reference_design, Environment(), model_config)
    result = replace(
        result,
        thermodynamic_constraints={
            **result.thermodynamic_constraints,
            "compressor_map_coordinate_domain": -0.1,
            "compressor_map_choke_boundary": 0.02,
        },
    )

    assessment = assess_constraint_profile(result, profiles[0])

    assert assessment.numerically_valid is True
    assert assessment.physics_domain_valid is False
    assert assessment.accepted is False
    assert "physics_domain.thermodynamic.compressor_map_choke_boundary" in (
        assessment.violated_constraints
    )
    assert not any(
        value.endswith("compressor_map_choke_boundary")
        and value.startswith("engineering_soft")
        for value in assessment.violated_constraints
    )


def test_conservation_tolerances_are_numerical_hard(
    reference_design, model_config, profiles
):
    result = solve_design_point(reference_design, Environment(), model_config)
    result = replace(
        result,
        conservation=replace(result.conservation, shaft_power_W=2.0e-6),
    )
    tolerances = {
        "profile_version": "test",
        "description": "test",
        "combustor_energy_absolute_W": 1.0e-6,
        "shaft_power_absolute_W": 1.0e-6,
        "normalized_flow_absolute": 1.0e-5,
        "nozzle_velocity_absolute_m_s": 1.0e-9,
        "mass_flow_rule": (
            "abs(residual_kg_s) <= normalized_flow_absolute * "
            "max(abs(mass_flow_gas_kg_s), 1.0e-12)"
        ),
    }

    assessment = assess_constraint_profile(
        result, profiles[0], conservation_tolerances=tolerances
    )

    assert assessment.numerically_valid is False
    assert assessment.physics_domain_valid is False
    assert assessment.accepted is False
    assert "numerical_hard.conservation.shaft_power" in assessment.violated_constraints


def test_nonfinite_engineering_residual_is_physics_domain_hard(
    reference_design, model_config, profiles
):
    result = solve_design_point(reference_design, Environment(), model_config)
    result = replace(
        result,
        thermodynamic_constraints={
            **result.thermodynamic_constraints,
            "combustor_pressure_loss_fraction_max": float("nan"),
        },
    )

    assessment = assess_constraint_profile(result, profiles[0])

    assert assessment.physics_domain_valid is False
    assert assessment.accepted is False
    assert (
        "physics_domain.thermodynamic."
        "combustor_pressure_loss_fraction_max.nonfinite"
    ) in assessment.violated_constraints
