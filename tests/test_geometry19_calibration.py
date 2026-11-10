"""Contract tests for the Geometry19 bootstrap calibration pack."""

import copy
from pathlib import Path

import pytest
import yaml

from src.physics.model_config import CalibrationPackError, ModelConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_SCHEMA_PATH = (
    REPO_ROOT / "config" / "physics" / "geometry19_calibration_schema_v1.yaml"
)
PACK_PATH = (
    REPO_ROOT / "config" / "physics" / "geometry19_calibration_bootstrap_v1.yaml"
)
RATIONALE_PATH = (
    REPO_ROOT
    / "docs"
    / "physics"
    / "geometry19_calibration_bootstrap_rationale_v1.md"
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _records(pack: dict) -> list[dict]:
    return [
        record
        for records in pack["parameter_groups"].values()
        for record in records
    ]


def _write_pack(tmp_path: Path, pack: dict) -> Path:
    path = tmp_path / "candidate.yaml"
    path.write_text(yaml.safe_dump(pack, sort_keys=False), encoding="utf-8")
    return path


def test_calibration_artifacts_exist_and_parse():
    assert CALIBRATION_SCHEMA_PATH.is_file()
    assert PACK_PATH.is_file()
    assert RATIONALE_PATH.is_file()
    assert _load(CALIBRATION_SCHEMA_PATH)["schema_version"] == (
        "geometry19_calibration_schema_v1"
    )


def test_pack_is_explicitly_bootstrap_and_not_formal_ready():
    pack = _load(PACK_PATH)
    assert pack["status"] == "bootstrap_candidate"
    assert pack["formal_dataset_generation_ready"] is False
    assert pack["design_schema_version"] == "geometry19_schema_v1"
    assert pack["physics_spec_version"] == "geometry19_closed_v2_1"


def test_all_required_groups_and_record_metadata_are_present():
    schema = _load(CALIBRATION_SCHEMA_PATH)
    pack = _load(PACK_PATH)
    assert set(schema["required_parameter_groups"]) <= set(pack["parameter_groups"])
    required = set(schema["required_parameter_fields"])
    records = _records(pack)
    assert records
    assert all(required <= set(record) for record in records)
    assert len({record["name"] for record in records}) == len(records)
    assert schema["optional_parameter_fields"] == ["notes"]


def test_candidate_contains_no_null_or_placeholder_values():
    forbidden = {None, "", "TBD", "...", "null"}
    for record in _records(_load(PACK_PATH)):
        for field in (
            "value",
            "unit",
            "valid_range",
            "source_type",
            "source_reference",
            "physical_meaning",
            "calibration_target",
        ):
            value = record[field]
            if isinstance(value, list):
                assert all(item not in forbidden for item in value)
            else:
                assert value not in forbidden


def test_values_obey_ranges_and_source_types():
    schema = _load(CALIBRATION_SCHEMA_PATH)
    allowed_sources = set(schema["allowed_source_types"])
    for record in _records(_load(PACK_PATH)):
        assert record["source_type"] in allowed_sources
        lower, upper = record["valid_range"]
        value = record["value"]
        if isinstance(value, bool):
            assert isinstance(lower, bool) and isinstance(upper, bool)
        elif isinstance(value, (int, float)):
            assert lower <= value <= upper
        else:
            assert value in record["valid_range"]


def test_literature_screening_rollback_is_explicit_and_definition_safe():
    records = {item["name"]: item for item in _records(_load(PACK_PATH))}
    assert records["compressor_psi_min"]["value"] == 0.35
    assert records["compressor_psi_max"]["value"] == 0.95
    assert records["turbine_phi_design"]["value"] == 0.14
    assert records["turbine_psi_max"]["value"] == 1.8
    assert records["hot_section_yield_strength_room_Pa"]["value"] == 1.034e9
    assert records["lattice_min_skin_thickness_m"]["value"] == 0.0005
    assert records["compressor_phi_min"]["value"] == 0.08
    assert "different area/diameter normalization" in records[
        "compressor_phi_min"
    ]["notes"]
    assert "Global total-air" in records["equivalence_ratio_max"]["notes"]
    assert "Not a burst-speed" in records["turbine_safety_factor"]["notes"]


def test_hot_thermodynamic_constants_are_consistent_within_one_percent():
    records = {item["name"]: item["value"] for item in _records(_load(PACK_PATH))}
    cp = records["cp_hot_J_kg_K"]
    gamma = records["gamma_hot"]
    gas_constant = records["gas_constant_hot_J_kg_K"]
    implied_cp = gamma * gas_constant / (gamma - 1.0)
    assert abs(cp - implied_cp) / cp < 0.01


def test_sutherland_viscosity_parameters_are_explicit_for_both_gas_sides():
    records = {item["name"]: item for item in _records(_load(PACK_PATH))}
    expected = {
        "dynamic_viscosity_cold_reference_Pa_s",
        "sutherland_cold_reference_temperature_K",
        "sutherland_cold_constant_K",
        "dynamic_viscosity_hot_reference_Pa_s",
        "sutherland_hot_reference_temperature_K",
        "sutherland_hot_constant_K",
    }
    assert expected <= records.keys()
    assert records["dynamic_viscosity_cold_reference_Pa_s"]["unit"] == "Pa*s"
    assert records["dynamic_viscosity_hot_reference_Pa_s"]["unit"] == "Pa*s"
    assert records["dynamic_viscosity_hot_reference_Pa_s"]["source_type"] == (
        "PROJECT_BOOTSTRAP_ASSUMPTION"
    )


def test_corrected_turbine_mass_flux_is_named_and_unitized():
    pack = _load(PACK_PATH)
    records = {item["name"]: item for item in _records(pack)}
    assert pack["reference_states"]["turbine_corrected_mass_flux"]["unit"] == (
        "kg/(s*m^2)"
    )
    for name in (
        "turbine_corrected_mass_flux_min",
        "turbine_corrected_mass_flux_max",
    ):
        assert records[name]["unit"] == "kg/(s*m^2)"
    assert records["turbine_corrected_mass_flux_min"]["value"] < records[
        "turbine_corrected_mass_flux_max"
    ]["value"]


def test_material_density_names_are_unambiguous():
    names = {item["name"] for item in _records(_load(PACK_PATH))}
    expected = {
        "compressor_material_density_kg_m3",
        "combustor_material_density_kg_m3",
        "turbine_material_density_kg_m3",
        "nozzle_material_density_kg_m3",
        "shaft_material_density_kg_m3",
    }
    assert expected <= names
    assert not {"rho_c", "rho_b", "rho_t", "rho_n", "rho_s"} & names


def test_reference_diffuser_assembly_matches_candidate_target():
    design_schema = _load(
        REPO_ROOT / "config" / "physics" / "geometry19_schema_v1.yaml"
    )
    values = {
        item["name"]: item["reference_value"]
        for item in design_schema["design_variables"]
    }
    records = {item["name"]: item["value"] for item in _records(_load(PACK_PATH))}
    d_c2 = values["compressor_impeller_exit_diameter_m"]
    b_c2 = values["compressor_impeller_exit_width_m"]
    beta = values["compressor_exit_blade_angle_deg"]
    blade_count = values["compressor_blade_count"]
    thickness = records["compressor_blade_thickness_ratio"] * b_c2

    import math

    r_c2 = d_c2 / 2.0
    blockage = 1.0 - blade_count * thickness / (
        2.0 * math.pi * r_c2 * math.sin(math.radians(beta))
    )
    a_c2 = blockage * math.pi * d_c2 * b_c2
    a_d_out = records["diffuser_area_ratio"] * a_c2
    d_bo = values["combustor_outer_diameter_m"]
    d_bi = values["combustor_inner_diameter_m"]
    a_b = math.pi * (d_bo**2 - d_bi**2) / 4.0
    ratio = a_d_out / a_b
    target = records["diffuser_to_combustor_target_area_ratio"]
    tolerance = records["diffuser_to_combustor_area_tolerance"]
    assert abs(ratio - target) <= tolerance


def test_runtime_loader_rejects_string_boolean(tmp_path):
    pack = _load(PACK_PATH)
    pack["formal_dataset_generation_ready"] = "false"
    with pytest.raises(CalibrationPackError, match="must be a boolean"):
        ModelConfig.from_yaml(_write_pack(tmp_path, pack))


def test_runtime_loader_rejects_unknown_source_type(tmp_path):
    pack = _load(PACK_PATH)
    pack["parameter_groups"]["thermo"][0]["source_type"] = "UNVERIFIED"
    with pytest.raises(CalibrationPackError, match="source_type is not allowed"):
        ModelConfig.from_yaml(_write_pack(tmp_path, pack))


def test_runtime_loader_rejects_out_of_range_value(tmp_path):
    pack = _load(PACK_PATH)
    pack["parameter_groups"]["theta_c"][4]["value"] = 0.0
    with pytest.raises(CalibrationPackError, match="outside valid_range"):
        ModelConfig.from_yaml(_write_pack(tmp_path, pack))


def test_runtime_loader_rejects_wrong_corrected_flux_unit(tmp_path):
    pack = _load(PACK_PATH)
    pack["reference_states"]["turbine_corrected_mass_flux"]["unit"] = (
        "dimensionless"
    )
    with pytest.raises(CalibrationPackError, match="must be kg/\\(s\\*m\\^2\\)"):
        ModelConfig.from_yaml(_write_pack(tmp_path, pack))


def test_runtime_loader_enforces_multiple_root_policy(tmp_path):
    pack = _load(PACK_PATH)
    pack["solver_candidate"]["reject_multiple_roots"] = False
    with pytest.raises(
        CalibrationPackError, match="freezes reject_multiple_roots=true"
    ):
        ModelConfig.from_yaml(_write_pack(tmp_path, pack))


def test_runtime_loader_requires_top_level_contract_fields(tmp_path):
    pack = _load(PACK_PATH)
    del pack["notes"]
    with pytest.raises(CalibrationPackError, match="missing top-level fields"):
        ModelConfig.from_yaml(_write_pack(tmp_path, pack))


def test_runtime_loader_rejects_unknown_status(tmp_path):
    pack = _load(PACK_PATH)
    pack["status"] = "experimental"
    with pytest.raises(CalibrationPackError, match="Unsupported calibration status"):
        ModelConfig.from_yaml(_write_pack(tmp_path, pack))


def test_runtime_loader_requires_all_parameter_groups(tmp_path):
    pack = _load(PACK_PATH)
    del pack["parameter_groups"]["theta_b"]
    with pytest.raises(CalibrationPackError, match="missing parameter groups"):
        ModelConfig.from_yaml(_write_pack(tmp_path, pack))


def test_runtime_loader_requires_all_record_fields(tmp_path):
    pack = _load(PACK_PATH)
    del pack["parameter_groups"]["thermo"][0]["unit"]
    with pytest.raises(CalibrationPackError, match="is missing"):
        ModelConfig.from_yaml(_write_pack(tmp_path, pack))


def test_runtime_loader_rejects_duplicate_parameter_names(tmp_path):
    pack = _load(PACK_PATH)
    duplicate = copy.deepcopy(pack["parameter_groups"]["thermo"][0])
    pack["parameter_groups"]["inlet"].append(duplicate)
    with pytest.raises(CalibrationPackError, match="Duplicate calibration parameter"):
        ModelConfig.from_yaml(_write_pack(tmp_path, pack))


def test_runtime_loader_enforces_frozen_readiness_consistency(tmp_path):
    pack = _load(PACK_PATH)
    pack["status"] = "frozen"
    with pytest.raises(
        CalibrationPackError, match="Only frozen packs may be formal-ready"
    ):
        ModelConfig.from_yaml(_write_pack(tmp_path, pack))


def test_runtime_loader_rejects_invalid_solver_field_type(tmp_path):
    pack = _load(PACK_PATH)
    pack["solver_candidate"]["linear_scan_points"] = 64.0
    with pytest.raises(CalibrationPackError, match="must be an integer"):
        ModelConfig.from_yaml(_write_pack(tmp_path, pack))


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (
        ("linear_scan_points", 1_000_000_000, "exceeds solver maximum"),
        ("mass_flow_residual_epsilon_kg_s", 1.0e100, "exceeds solver maximum"),
        ("flow_residual_tolerance", 2.0, "exceeds solver maximum"),
        ("root_relative_tolerance", 2.0, "exceeds solver maximum"),
    ),
)
def test_effective_override_rejects_unsafe_solver_magnitude(name, value, message):
    config = ModelConfig.from_yaml(PACK_PATH)
    with pytest.raises(CalibrationPackError, match=message):
        config.with_overrides(solver_candidate={name: value})


def test_solver_contract_enforces_tolerance_ordering():
    config = ModelConfig.from_yaml(PACK_PATH)
    with pytest.raises(
        CalibrationPackError,
        match="flow_residual_tolerance must not exceed tangency_refine_threshold",
    ):
        config.with_overrides(
            solver_candidate={
                "flow_residual_tolerance": 1.0e-3,
                "tangency_refine_threshold": 1.0e-4,
            }
        )


def test_schema_freezes_every_solver_field_domain():
    schema = _load(CALIBRATION_SCHEMA_PATH)
    solver_contract = schema["solver_candidate_contract"]
    expected_fields = set(_load(PACK_PATH)["solver_candidate"])
    assert set(solver_contract) == expected_fields
    for name, contract in solver_contract.items():
        assert contract["type"] in {"integer", "number", "boolean"}
        if contract["type"] in {"integer", "number"}:
            assert contract["minimum"] < contract["maximum"]
        else:
            assert name == "reject_multiple_roots"
            assert contract["const"] is True


def test_runtime_loader_requires_material_density_contract(tmp_path):
    pack = _load(PACK_PATH)
    records = pack["parameter_groups"]["theta_s"]
    density = next(
        item for item in records if item["name"] == "compressor_material_density_kg_m3"
    )
    density["unit"] = "g/cm^3"
    with pytest.raises(CalibrationPackError, match="must have one kg/m\\^3 record"):
        ModelConfig.from_yaml(_write_pack(tmp_path, pack))
