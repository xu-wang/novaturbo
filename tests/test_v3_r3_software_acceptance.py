"""R3 diagnostic acceptance of the V3 single-design software backend."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from src.physics.engine_match import Environment, FailureCode
from src.physics.geometry22_design import Geometry22Design, Geometry22DesignError
from src.physics.v3_development_runtime import build_v3_development_physics_runtime


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "config/physics/geometry22_schema_v3_development_v1.yaml"


@pytest.fixture(scope="module")
def runtime():
    return build_v3_development_physics_runtime(ROOT)


@pytest.fixture(scope="module")
def reference_values():
    payload = yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))
    return {row["name"]: row["reference_value"] for row in payload["design_variables"]}


def _design(values, **changes):
    return Geometry22Design.from_mapping({**values, **changes})


@pytest.mark.parametrize(
    ("field", "endpoint"),
    [
        ("diffuser_area_ratio", 1.5),
        ("diffuser_area_ratio", 4.0),
        ("combustor_primary_air_fraction", 0.1),
        ("combustor_primary_air_fraction", 0.5),
        ("compressor_tip_clearance_ratio", 0.002),
        ("compressor_tip_clearance_ratio", 0.005),
    ],
)
def test_development_coordinate_endpoints_are_explicit(runtime, reference_values, field, endpoint):
    design = _design(reference_values, **{field: endpoint})
    result = runtime.evaluate(design)
    assert result.design.as_mapping()[field] == endpoint
    assert result.input_parameters[field] == endpoint
    assert result.engine_result.solver.converged
    assert result.engine_result.operating_point["flow_root_count"] == 1
    assert result.engine_result.operating_point["selected_flow_root_index"] == 0
    assert result.physics_validation_status == "not_completed"
    assert result.formal_use_allowed is False


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("diffuser_area_ratio", 1.49),
        ("diffuser_area_ratio", 4.01),
        ("combustor_primary_air_fraction", 0.09),
        ("combustor_primary_air_fraction", 0.51),
        ("compressor_tip_clearance_ratio", 0.00199),
        ("compressor_tip_clearance_ratio", 0.00501),
        ("compressor_tip_clearance_ratio", float("nan")),
    ],
)
def test_development_coordinate_outside_bounds_is_rejected(reference_values, field, invalid):
    with pytest.raises(Geometry22DesignError, match=field):
        _design(reference_values, **{field: invalid})


def test_numeric_convergence_does_not_hide_assembly_violation(runtime, reference_values):
    result = runtime.evaluate(_design(reference_values, diffuser_area_ratio=4.0))
    assert result.engine_result.solver.converged
    assert result.engine_result.assembly_constraints["diffuser_to_combustor_area"] > 0.0
    assert result.engine_result.feasible is False


def test_root_diagnostics_and_no_root_failure_preserve_22d_identity(runtime, reference_values):
    unique = runtime.evaluate(_design(reference_values))
    assert unique.engine_result.operating_point["flow_root_selection_policy"] == (
        "reference_continuation_high_flow_v1"
    )
    assert unique.engine_result.operating_point["flow_root_count"] == 1
    assert unique.engine_result.operating_point["selected_flow_root_index"] == 0

    no_root = runtime.evaluate(
        _design(reference_values, compressor_impeller_exit_diameter_m=0.15)
    )
    assert no_root.engine_result.solver.converged is False
    assert no_root.engine_result.solver.failure_code == FailureCode.NO_FLOW_MATCH
    assert no_root.engine_result.operating_point["flow_root_count"] == 0
    assert no_root.engine_result.operating_point["selected_flow_root_index"] is None
    assert no_root.engine_result.components == {}
    assert no_root.design.coordinates[19:] == unique.design.coordinates[19:]
    assert no_root.design_id != unique.design_id
    assert no_root.engine_result.physics_model_version == unique.engine_result.physics_model_version


def test_injected_multiple_root_diagnostics_are_preserved(runtime, reference_values):
    """A software fixture checks the wrapper while a V3 multi-root case is unavailable."""

    base_backend = runtime.base_runtime.backend

    def branch_fixture_solver(x19, environment, config, **solvers):
        result = base_backend.solver(x19, environment, config, **solvers)
        assert result.solver.converged
        actual_root = result.operating_point["selected_flow_root_mass_flow_air_kg_s"]
        return replace(
            result,
            operating_point={
                **result.operating_point,
                "flow_root_count": 2,
                "flow_root_mass_flow_air_kg_s": [actual_root * 0.5, actual_root],
                "flow_root_normalized_residuals": [0.0, 0.0],
                "selected_flow_root_index": 1,
                "selected_flow_root_mass_flow_air_kg_s": actual_root,
            },
        )

    injected_backend = replace(base_backend, solver=branch_fixture_solver)
    injected_runtime = replace(
        runtime,
        base_runtime=replace(runtime.base_runtime, backend=injected_backend),
    )
    result = injected_runtime.evaluate(_design(reference_values)).engine_result
    assert result.operating_point["flow_root_count"] == 2
    assert result.operating_point["selected_flow_root_index"] == 1
    assert len(result.operating_point["flow_root_normalized_residuals"]) == 2
    assert result.operating_point["selected_flow_root_mass_flow_air_kg_s"] == max(
        result.operating_point["flow_root_mass_flow_air_kg_s"]
    )


def test_invalid_environment_has_failure_without_fake_component_result(runtime, reference_values):
    design = _design(reference_values)
    failure = runtime.evaluate(design, Environment(pressure_Pa=-1.0))
    assert failure.engine_result.solver.failure_code == FailureCode.INVALID_ENVIRONMENT
    assert failure.engine_result.components == {}
    assert failure.design == design
    assert failure.input_parameters == design.component_parameters()


def test_repeated_evaluation_is_deterministic(runtime, reference_values):
    design = _design(reference_values, compressor_tip_clearance_ratio=0.004)
    first = runtime.evaluate(design)
    second = runtime.evaluate(design)
    assert first.design_id == second.design_id
    assert first.physics_bundle_hash == second.physics_bundle_hash
    assert first.engine_result.solver == second.engine_result.solver
    assert first.engine_result.operating_point == second.engine_result.operating_point
    assert first.engine_result.performance == second.engine_result.performance
    assert first.engine_result.components == second.engine_result.components
    assert first.engine_result.conservation == second.engine_result.conservation


def test_gap_correction_is_monotone_across_development_envelope(runtime, reference_values):
    low = runtime.evaluate(_design(reference_values, compressor_tip_clearance_ratio=0.002))
    reference = runtime.evaluate(_design(reference_values))
    high = runtime.evaluate(_design(reference_values, compressor_tip_clearance_ratio=0.005))
    values = [
        result.engine_result.components["compressor"]["total_to_total_efficiency"]
        for result in (low, reference, high)
    ]
    assert values[0] > values[1] > values[2]
    assert all(result.engine_result.solver.converged for result in (low, reference, high))


def test_other_new_coordinates_reach_distinct_boundary_states(runtime, reference_values):
    diffuser_low = runtime.evaluate(_design(reference_values, diffuser_area_ratio=1.5))
    diffuser_high = runtime.evaluate(_design(reference_values, diffuser_area_ratio=4.0))
    low_residual = diffuser_low.engine_result.assembly_constraints[
        "diffuser_to_combustor_area"
    ]
    high_residual = diffuser_high.engine_result.assembly_constraints[
        "diffuser_to_combustor_area"
    ]
    assert low_residual < 0.0 < high_residual

    primary_low = runtime.evaluate(
        _design(reference_values, combustor_primary_air_fraction=0.1)
    )
    primary_high = runtime.evaluate(
        _design(reference_values, combustor_primary_air_fraction=0.5)
    )
    residence_low = primary_low.engine_result.components["combustor"][
        "primary_residence_time_s"
    ]
    residence_high = primary_high.engine_result.components["combustor"][
        "primary_residence_time_s"
    ]
    assert residence_low > residence_high > 0.0
