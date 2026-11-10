"""WP19-12 tests for the unique Geometry19 visualization adapter."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.geometry.assembly import assemble_engine
from src.physics import Geometry19, ModelConfig
from src.visualization import (
    VisualizationCalibration,
    VisualizationGeometryError,
    geometry19_to_assembly_params,
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
def visualization_calibration() -> VisualizationCalibration:
    return VisualizationCalibration.from_yaml(
        PHYSICS_CONFIG / "geometry19_visualization_calibration_v1.yaml"
    )


def _adapt(design, model_config, visualization_calibration):
    return geometry19_to_assembly_params(
        design, model_config, visualization_calibration
    )


def test_reference_mapping_uses_geometry_state_and_canonical_semantics(
    reference_design, model_config, visualization_calibration
):
    adapted = _adapt(reference_design, model_config, visualization_calibration)
    params = adapted.assembly_params
    state = adapted.geometry_state
    assert params.compressor.impeller_tip_diameter_mm == pytest.approx(80.0)
    assert params.compressor.impeller_hub_diameter_mm == pytest.approx(
        state.compressor_inducer_hub_diameter_m * 1000.0
    )
    assert params.compressor.blade_count == reference_design.compressor_blade_count
    assert params.combustor.num_fuel_injectors == (
        reference_design.combustor_injector_count
    )
    assert params.turbine.blade_count == reference_design.turbine_blade_count
    assert params.nozzle.inlet_diameter_mm == pytest.approx(
        state.nozzle_inlet_diameter_m * 1000.0
    )
    assert params.nozzle.exit_diameter_mm == pytest.approx(
        reference_design.nozzle_throat_diameter_m * 1000.0
    )
    assert params.shaft_diameter_mm == pytest.approx(
        state.shaft_diameter_m * 1000.0
    )


def test_adapter_is_deterministic_and_records_both_calibrations(
    reference_design, model_config, visualization_calibration
):
    first = _adapt(reference_design, model_config, visualization_calibration)
    second = _adapt(reference_design, model_config, visualization_calibration)
    assert first.solid_geometry_hash == second.solid_geometry_hash
    assert first.lattice_hash == second.lattice_hash
    assert first.design_hash == second.design_hash
    assert first.calibration_hash == second.calibration_hash
    assert len(first.calibration_hash) == 64
    assert first.visualization_calibration_version == (
        "geometry19_visualization_calibration_v1"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("compressor_impeller_exit_diameter_m", 0.09),
        ("compressor_inducer_tip_diameter_m", 0.055),
        ("compressor_inducer_hub_tip_ratio", 0.45),
        ("compressor_impeller_exit_width_m", 0.006),
        ("compressor_exit_blade_angle_deg", 65.0),
        ("compressor_blade_count", 13),
        ("combustor_outer_diameter_m", 0.12),
        ("combustor_inner_diameter_m", 0.065),
        ("combustor_length_m", 0.09),
        ("combustor_liner_thickness_m", 0.0015),
        ("combustor_injector_count", 7),
        ("turbine_tip_diameter_m", 0.09),
        ("turbine_hub_tip_ratio", 0.74),
        ("turbine_blade_count", 19),
        ("nozzle_throat_diameter_m", 0.06),
    ],
)
def test_every_solid_coordinate_changes_solid_contract(
    field, value, reference_design, model_config, visualization_calibration
):
    baseline = _adapt(reference_design, model_config, visualization_calibration)
    changed = _adapt(
        replace(reference_design, **{field: value}),
        model_config,
        visualization_calibration,
    )
    assert changed.solid_geometry_hash != baseline.solid_geometry_hash


@pytest.mark.parametrize(
    "field,value,overlay_key",
    [
        ("turbine_inlet_total_temperature_K", 1200.0, "turbine_inlet_total_temperature_K"),
        ("shaft_speed_rpm", 110000.0, "shaft_speed_rpm"),
    ],
)
def test_tit_and_rpm_change_overlay_not_solid_geometry(
    field,
    value,
    overlay_key,
    reference_design,
    model_config,
    visualization_calibration,
):
    baseline = _adapt(reference_design, model_config, visualization_calibration)
    changed = _adapt(
        replace(reference_design, **{field: value}),
        model_config,
        visualization_calibration,
    )
    assert changed.solid_geometry_hash == baseline.solid_geometry_hash
    assert changed.overlay_parameters[overlay_key] != baseline.overlay_parameters[overlay_key]

    baseline_mesh = assemble_engine(baseline.assembly_params)
    changed_mesh = assemble_engine(changed.assembly_params)
    assert np.array_equal(
        baseline_mesh.compressor_geo.impeller_points,
        changed_mesh.compressor_geo.impeller_points,
    )
    assert np.array_equal(
        baseline_mesh.turbine_geo.rotor_points,
        changed_mesh.turbine_geo.rotor_points,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("lattice_cell_size_m", 0.01),
        ("lattice_relative_density", 0.4),
    ],
)
def test_lattice_coordinates_change_lattice_contract_not_solid_envelope(
    field, value, reference_design, model_config, visualization_calibration
):
    baseline = _adapt(reference_design, model_config, visualization_calibration)
    changed = _adapt(
        replace(reference_design, **{field: value}),
        model_config,
        visualization_calibration,
    )
    assert changed.solid_geometry_hash == baseline.solid_geometry_hash
    assert changed.lattice_hash != baseline.lattice_hash


def test_adapter_rejects_combustor_without_lattice_infill_clearance(
    reference_design, model_config, visualization_calibration
):
    design = replace(
        reference_design,
        combustor_outer_diameter_m=0.08356719484354438,
        combustor_inner_diameter_m=0.07485572017699574,
        combustor_liner_thickness_m=0.00196753917389886,
    )
    with pytest.raises(
        VisualizationGeometryError, match="no positive radial infill clearance"
    ):
        _adapt(design, model_config, visualization_calibration)
