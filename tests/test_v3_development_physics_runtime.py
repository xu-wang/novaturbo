"""R2 software gates for the isolated 22D development backend."""

from pathlib import Path

import pytest
import yaml

from src.physics.geometry22_design import Geometry22Design
from src.physics.v3_development_runtime import (
    MODEL_VERSION,
    V3DevelopmentRuntimeError,
    build_v3_development_physics_runtime,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "config/physics/geometry22_schema_v3_development_v1.yaml"


@pytest.fixture(scope="module")
def runtime():
    return build_v3_development_physics_runtime(ROOT)


@pytest.fixture(scope="module")
def reference_design():
    schema = yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))
    return Geometry22Design.from_mapping(
        {row["name"]: row["reference_value"] for row in schema["design_variables"]}
    )


def _changed(design, field, value):
    return Geometry22Design.from_mapping({**design.as_mapping(), field: value})


def test_reference_design_has_distinct_22d_identity_and_base_parity(runtime, reference_design):
    result = runtime.evaluate(reference_design)
    base = runtime.base_runtime.evaluate(reference_design.geometry19_projection())
    assert result.engine_result.solver.converged
    assert base.solver.converged
    assert result.engine_result.performance == base.performance
    assert result.design == reference_design
    assert result.engine_result.design == reference_design.geometry19_projection()
    assert result.design_id == reference_design.design_id()
    assert result.physics_model_version == MODEL_VERSION
    assert result.engine_result.physics_model_version == MODEL_VERSION
    assert base.physics_model_version.startswith("geometry19_map_enhanced_v2_2")
    assert result.physics_bundle_hash != runtime.base_runtime.physics_bundle_hash
    assert result.physics_validation_status == "not_completed"
    assert result.formal_use_allowed is False
    assert result.deployment_allowed is False
    assert runtime.contract_mapping()["design_schema_hash"] == result.design_schema_hash
    tolerances = runtime.base_runtime.conservation_tolerances
    assert abs(result.engine_result.conservation.combustor_energy_W) <= float(
        tolerances["combustor_energy_absolute_W"]
    )
    assert abs(result.engine_result.conservation.shaft_power_W) <= float(
        tolerances["shaft_power_absolute_W"]
    )
    assert abs(result.engine_result.conservation.normalized_flow) <= float(
        tolerances["normalized_flow_absolute"]
    )


def test_each_new_coordinate_changes_its_declared_physics_path(runtime, reference_design):
    baseline = runtime.evaluate(reference_design).engine_result
    diffuser = runtime.evaluate(_changed(reference_design, "diffuser_area_ratio", 2.6)).engine_result
    primary = runtime.evaluate(_changed(reference_design, "combustor_primary_air_fraction", 0.26)).engine_result
    clearance = runtime.evaluate(_changed(reference_design, "compressor_tip_clearance_ratio", 0.004)).engine_result
    assert all(x.solver.converged for x in (baseline, diffuser, primary, clearance))
    assert diffuser.assembly_constraints["diffuser_to_combustor_area"] != baseline.assembly_constraints["diffuser_to_combustor_area"]
    assert primary.components["combustor"]["primary_residence_time_s"] != baseline.components["combustor"]["primary_residence_time_s"]
    assert clearance.components["compressor"]["total_to_total_efficiency"] < baseline.components["compressor"]["total_to_total_efficiency"]
    assert clearance.components["compressor"]["pressure_ratio"] < baseline.components["compressor"]["pressure_ratio"]
    assert clearance.components["compressor"]["tip_clearance_correction_status"] == "software_fixture_unvalidated"
    assert clearance.performance["thrust_N"] != baseline.performance["thrust_N"]


def test_v3_rejects_19d_input_and_keeps_base_runtime_immutable(runtime, reference_design):
    base_hash = runtime.base_runtime.physics_bundle_hash
    base_calibration = runtime.base_runtime.model_config.calibration_pack_hash
    with pytest.raises(V3DevelopmentRuntimeError, match="Geometry22Design"):
        runtime.evaluate(reference_design.geometry19_projection())
    runtime.evaluate(_changed(reference_design, "diffuser_area_ratio", 2.6))
    assert runtime.base_runtime.physics_bundle_hash == base_hash
    assert runtime.base_runtime.model_config.calibration_pack_hash == base_calibration
    assert runtime.base_runtime.model_config.require("diffuser_area_ratio") == 2.5
