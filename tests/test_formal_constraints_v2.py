"""Equation-level tests for WP19-8 formal constraints."""

from dataclasses import replace
import math
from pathlib import Path

import pytest
import yaml

from src.physics import (
    Geometry19,
    ModelConfig,
    StationState,
    derive_geometry_state,
    evaluate_formal_constraints_v2,
    solve_combustor_loworder,
    solve_compressor_meanline,
    solve_mass_model_v2,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model_config() -> ModelConfig:
    return ModelConfig.from_yaml(
        REPO_ROOT
        / "config"
        / "physics"
        / "geometry19_calibration_bootstrap_v1.yaml"
    )


@pytest.fixture(scope="module")
def reference_design() -> Geometry19:
    schema = yaml.safe_load(
        (
            REPO_ROOT / "config" / "physics" / "geometry19_schema_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    return Geometry19.from_mapping(
        {
            item["name"]: item["reference_value"]
            for item in schema["design_variables"]
        }
    )


def _evaluate(design, config):
    geometry = derive_geometry_state(design, config)
    inlet = StationState(total_temperature_K=288.15, total_pressure_Pa=99300.0)
    compressor = solve_compressor_meanline(geometry, 0.15, inlet, config)
    combustor = solve_combustor_loworder(
        geometry, 0.15, compressor.outlet, config
    )
    mass = solve_mass_model_v2(geometry, config)
    formal = evaluate_formal_constraints_v2(
        geometry, compressor.outlet, combustor, mass, config
    )
    return geometry, compressor, combustor, mass, formal


def test_reference_formal_constraints_are_finite_and_complete(
    reference_design, model_config
):
    _, _, _, _, formal = _evaluate(reference_design, model_config)
    assert set(formal.structural_residuals) == {
        "compressor_rotating_stress",
        "turbine_rotating_stress",
        "combustor_liner_hoop_stress",
    }
    assert set(formal.thermal_residuals) == {
        "compressor_metal_temperature",
        "combustor_metal_temperature",
        "turbine_metal_temperature",
        "turbine_inlet_temperature_min",
        "turbine_inlet_temperature_max",
    }
    assert set(formal.manufacturing_residuals) >= {
        "lattice_relative_density_min",
        "lattice_relative_density_max",
        "lattice_cell_size_min",
        "lattice_cells_across_region_min",
    }
    assert all(
        math.isfinite(value)
        for residuals in (
            formal.structural_residuals,
            formal.thermal_residuals,
            formal.manufacturing_residuals,
        )
        for value in residuals.values()
    )


def test_rotating_stress_residuals_match_v21_equations(
    reference_design, model_config
):
    geometry, _, _, _, formal = _evaluate(reference_design, model_config)
    compressor_stress = (
        float(model_config.require("compressor_rotating_stress_factor"))
        * float(model_config.require("compressor_material_density_kg_m3"))
        * geometry.compressor_tip_speed_m_s**2
    )
    expected_compressor = (
        float(model_config.require("compressor_safety_factor"))
        * compressor_stress
        / float(model_config.require("compressor_yield_strength_room_Pa"))
        - 1.0
    )
    assert formal.structural_residuals[
        "compressor_rotating_stress"
    ] == pytest.approx(expected_compressor)

    turbine_stress = (
        float(model_config.require("turbine_rotating_stress_factor"))
        * float(model_config.require("turbine_material_density_kg_m3"))
        * geometry.turbine_tip_speed_m_s**2
    )
    expected_turbine = (
        float(model_config.require("turbine_safety_factor"))
        * turbine_stress
        / float(model_config.require("hot_section_yield_strength_room_Pa"))
        - 1.0
    )
    assert formal.structural_residuals[
        "turbine_rotating_stress"
    ] == pytest.approx(expected_turbine)


def test_metal_temperature_and_tit_relations_close(reference_design, model_config):
    _, compressor, combustor, _, formal = _evaluate(
        reference_design, model_config
    )
    t3 = compressor.outlet.total_temperature_K
    t4 = combustor.outlet.total_temperature_K
    expected_turbine_metal = t3 + float(
        model_config.require("turbine_metal_temperature_fraction")
    ) * (t4 - t3)
    assert formal.diagnostics["turbine_metal_temperature_K"] == pytest.approx(
        expected_turbine_metal
    )
    assert formal.thermal_residuals[
        "turbine_inlet_temperature_min"
    ] == pytest.approx(
        float(model_config.require("turbine_inlet_temperature_min_K")) - t4
    )
    assert formal.thermal_residuals[
        "turbine_inlet_temperature_max"
    ] == pytest.approx(
        t4 - float(model_config.require("turbine_inlet_temperature_max_K"))
    )


def test_lattice_cell_size_has_manufacturing_path_not_mass_path(
    reference_design, model_config
):
    _, _, _, baseline_mass, baseline = _evaluate(reference_design, model_config)
    large_cell_design = replace(reference_design, lattice_cell_size_m=0.014)
    _, _, _, large_mass, large = _evaluate(large_cell_design, model_config)
    assert large_mass.total_mass_kg == pytest.approx(baseline_mass.total_mass_kg)
    assert large.manufacturing_residuals[
        "lattice_cells_across_region_min"
    ] > baseline.manufacturing_residuals["lattice_cells_across_region_min"]


def test_higher_lattice_density_improves_effective_liner_strength(
    reference_design, model_config
):
    low = replace(reference_design, lattice_relative_density=0.15)
    high = replace(reference_design, lattice_relative_density=0.55)
    _, _, _, low_mass, low_formal = _evaluate(low, model_config)
    _, _, _, high_mass, high_formal = _evaluate(high, model_config)
    assert high_mass.total_mass_kg > low_mass.total_mass_kg
    assert high_formal.structural_residuals[
        "combustor_liner_hoop_stress"
    ] < low_formal.structural_residuals["combustor_liner_hoop_stress"]


def test_bootstrap_yield_temperature_assumption_is_explicit(
    reference_design, model_config
):
    _, _, _, _, formal = _evaluate(reference_design, model_config)
    assert formal.diagnostics["yield_strength_temperature_model"] == (
        "room_temperature_bootstrap_proxy"
    )
