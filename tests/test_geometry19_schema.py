"""Contract tests for the frozen Geometry-aware 19D schema."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "config" / "physics" / "geometry19_schema_v1.yaml"
SPEC_PATH = (
    REPO_ROOT
    / "docs"
    / "physics"
    / "NovaTurbo_GeometryAware_19D_Closed_LowOrder_Engine_Model_V2_1.md"
)

EXPECTED_ORDER = [
    "compressor_impeller_exit_diameter_m",
    "compressor_inducer_tip_diameter_m",
    "compressor_inducer_hub_tip_ratio",
    "compressor_impeller_exit_width_m",
    "compressor_exit_blade_angle_deg",
    "compressor_blade_count",
    "combustor_outer_diameter_m",
    "combustor_inner_diameter_m",
    "combustor_length_m",
    "combustor_liner_thickness_m",
    "combustor_injector_count",
    "turbine_tip_diameter_m",
    "turbine_hub_tip_ratio",
    "turbine_blade_count",
    "turbine_inlet_total_temperature_K",
    "nozzle_throat_diameter_m",
    "shaft_speed_rpm",
    "lattice_cell_size_m",
    "lattice_relative_density",
]

DERIVED_NOT_DESIGN = {
    "mass_flow_air_kg_s",
    "compressor_pressure_ratio",
    "compressor_total_to_total_efficiency",
    "turbine_total_to_total_efficiency",
    "mass_flow_fuel_kg_s",
    "air_fuel_ratio",
}


def _schema() -> dict:
    return yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))


def _reference_design(schema: dict) -> dict:
    return {
        variable["name"]: variable["reference_value"]
        for variable in schema["design_variables"]
    }


def test_schema_and_canonical_spec_are_present():
    assert SCHEMA_PATH.is_file()
    assert SPEC_PATH.is_file()
    schema = _schema()
    assert schema["source_document"] == str(SPEC_PATH.relative_to(REPO_ROOT))


def test_schema_freezes_exactly_nineteen_ordered_design_variables():
    schema = _schema()
    variables = schema["design_variables"]
    assert len(variables) == 19
    assert [item["index"] for item in variables] == list(range(1, 20))
    assert [item["name"] for item in variables] == EXPECTED_ORDER
    assert len({item["name"] for item in variables}) == 19
    assert schema["coordinate_order_frozen"] is True


def test_all_bounds_are_finite_ordered_and_contain_reference_design():
    for variable in _schema()["design_variables"]:
        lower, upper = variable["bounds"]
        reference = variable["reference_value"]
        assert isinstance(lower, (int, float))
        assert isinstance(upper, (int, float))
        assert lower < upper
        assert lower <= reference <= upper
        assert variable["unit"]
        assert variable["dtype"] in {"float64", "int64"}


def test_integer_variables_use_balanced_discrete_sampling():
    variables = {item["name"]: item for item in _schema()["design_variables"]}
    expected = {
        "compressor_blade_count",
        "combustor_injector_count",
        "turbine_blade_count",
    }
    actual = {name for name, item in variables.items() if item["dtype"] == "int64"}
    assert actual == expected
    for name in expected:
        item = variables[name]
        assert item["sampling"]["method"] == "balanced_discrete"
        assert item["sampling"]["step"] == 1
        assert all(isinstance(value, int) for value in item["bounds"])
        assert isinstance(item["reference_value"], int)


def test_inverse_conditions_are_three_outputs_not_design_coordinates():
    schema = _schema()
    condition_names = [item["name"] for item in schema["inverse_conditions"]]
    assert condition_names == ["thrust_N", "tsfc_kg_N_s", "mass_kg"]
    design_names = {item["name"] for item in schema["design_variables"]}
    assert design_names.isdisjoint(condition_names)


def test_cycle_states_removed_from_design_vector_and_declared_derived():
    schema = _schema()
    design_names = {item["name"] for item in schema["design_variables"]}
    derived_names = set(schema["derived_states"])
    assert design_names.isdisjoint(DERIVED_NOT_DESIGN)
    assert DERIVED_NOT_DESIGN <= derived_names


def test_reference_design_satisfies_direct_geometry_priors():
    reference = _reference_design(_schema())
    d_c1t = reference["compressor_inducer_tip_diameter_m"]
    lambda_c1 = reference["compressor_inducer_hub_tip_ratio"]
    d_c1h = lambda_c1 * d_c1t
    d_c2 = reference["compressor_impeller_exit_diameter_m"]
    assert 0.0 < d_c1h < d_c1t < d_c2

    d_bo = reference["combustor_outer_diameter_m"]
    d_bi = reference["combustor_inner_diameter_m"]
    t_b = reference["combustor_liner_thickness_m"]
    assert d_bo > d_bi + 2.0 * t_b

    lambda_t = reference["turbine_hub_tip_ratio"]
    assert 0.0 < lambda_t < 1.0
    assert reference["turbine_tip_diameter_m"] > 0.0
    assert reference["nozzle_throat_diameter_m"] > 0.0


def test_schema_is_blocked_from_formal_dataset_generation_until_calibrated():
    schema = _schema()
    assert schema["bounds_status"] == "frozen_wp19_a_v1"
    assert schema["formal_dataset_generation_ready"] is False
    required_quality = {
        "converged",
        "feasible",
        "failure_code",
        "failure_class",
        "flow_residual",
        "physics_model_version",
        "calibration_pack_hash",
    }
    assert required_quality <= set(schema["quality_fields"])
