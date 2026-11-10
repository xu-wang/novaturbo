"""Unit tests for the WP19-16 39-case regression harness."""

from dataclasses import replace

from scripts.run_wp19_16_geometry_regression import (
    EXPECTED_HARD_FAILURES,
    LATTICE_COMPONENTS,
    SOLID_COMPONENTS,
    RegressionCase,
    build_cases,
    evaluate_channel,
    load_schema,
    select_cases,
)


def _response(geometry="geometry-ref", lattice="lattice-ref", thrust=10.0):
    return {
        "geometry_hash": geometry,
        "lattice_hash": lattice,
        "physics": {"performance": {"thrust_N": thrust}},
        "mesh_diagnostics": {"passed": True},
        "solver": {"converged": True},
    }


def test_wp19_16_matrix_is_reference_plus_two_boundaries_per_coordinate():
    cases = build_cases(load_schema())
    assert len(cases) == 39
    assert cases[0].name == "reference"
    assert len({case.name for case in cases}) == 39
    for index in range(1, 20):
        matching = [case for case in cases if case.name.startswith(f"{index:02d}_")]
        assert {case.boundary for case in matching} == {"low", "high"}


def test_case_selection_automatically_includes_reference():
    cases = build_cases(load_schema())
    selected = select_cases(cases, [cases[5].name])
    assert [case.name for case in selected] == ["reference", cases[5].name]


def test_frozen_cross_bound_conflict_is_explicitly_versioned():
    rule = EXPECTED_HARD_FAILURES["02_compressor_inducer_tip_diameter_m_high"]
    assert rule["constraint"] == "compressor_inducer_order"
    assert "D_c1t=0.100" in rule["reason"]


def test_channel_audit_distinguishes_solid_overlay_and_lattice():
    schema = load_schema()
    variables = {item["name"]: item for item in schema["design_variables"]}
    reference_design = build_cases(schema)[0].design
    reference_response = _response()
    reference_hashes = {name: f"ref-{name}" for name in SOLID_COMPONENTS | LATTICE_COMPONENTS}

    solid_name = "compressor_impeller_exit_diameter_m"
    solid_case = RegressionCase("solid", solid_name, "high", replace(reference_design, **{solid_name: 0.15}))
    solid_hashes = {**reference_hashes, "compressor": "changed"}
    assert evaluate_channel(
        solid_case,
        variables[solid_name],
        _response(geometry="changed"),
        solid_hashes,
        reference_response,
        reference_hashes,
    )["passed"]

    overlay_name = "shaft_speed_rpm"
    overlay_case = RegressionCase("overlay", overlay_name, "high", replace(reference_design, shaft_speed_rpm=150000.0))
    assert evaluate_channel(
        overlay_case,
        variables[overlay_name],
        _response(thrust=11.0),
        reference_hashes,
        reference_response,
        reference_hashes,
    )["passed"]

    lattice_name = "lattice_relative_density"
    lattice_case = RegressionCase("lattice", lattice_name, "high", replace(reference_design, lattice_relative_density=0.55))
    lattice_hashes = {**reference_hashes, "combustor_lattice": "changed"}
    assert evaluate_channel(
        lattice_case,
        variables[lattice_name],
        _response(lattice="changed"),
        lattice_hashes,
        reference_response,
        reference_hashes,
    )["passed"]
