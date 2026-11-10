"""Equation and sensitivity tests for the WP19-8 V2.1 mass closure."""

from dataclasses import replace
import math
from pathlib import Path

import pytest
import yaml

from src.physics import (
    Geometry19,
    ModelConfig,
    derive_geometry_state,
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


@pytest.fixture(scope="module")
def mass(reference_design, model_config):
    geometry = derive_geometry_state(reference_design, model_config)
    return solve_mass_model_v2(geometry, model_config)


def test_reference_mass_is_positive_complete_and_additive(mass):
    assert set(mass.component_masses_kg) == {
        "compressor",
        "combustor",
        "turbine",
        "nozzle",
        "shaft",
        "injectors",
        "bearing_accessory",
    }
    assert all(value > 0.0 for value in mass.component_masses_kg.values())
    assert mass.total_mass_kg == pytest.approx(
        sum(mass.component_masses_kg.values())
    )
    assert mass.total_mass_kg == pytest.approx(1.3285481772955765)


def test_compressor_hub_and_casing_follow_v21_equations(
    reference_design, mass, model_config
):
    geometry = derive_geometry_state(reference_design, model_config)
    inlet_mean_radius = 0.25 * (
        reference_design.compressor_inducer_tip_diameter_m
        + geometry.compressor_inducer_hub_diameter_m
    )
    hub_thickness = float(
        model_config.require("compressor_hub_thickness_ratio")
    ) * reference_design.compressor_impeller_exit_width_m
    expected_hub = (
        float(model_config.require("compressor_hub_fill_factor"))
        * math.pi
        * (geometry.compressor_exit_radius_m**2 - inlet_mean_radius**2)
        * hub_thickness
    )
    expected_casing = (
        float(model_config.require("compressor_casing_factor"))
        * math.pi
        * reference_design.compressor_impeller_exit_diameter_m
        * mass.compressor_length_m
        * float(model_config.require("compressor_casing_thickness_m"))
    )
    assert mass.component_volumes_m3["compressor_hub"] == pytest.approx(
        expected_hub
    )
    assert mass.component_volumes_m3["compressor_casing"] == pytest.approx(
        expected_casing
    )


def test_turbine_mass_retains_disk_term_at_high_hub_ratio(
    reference_design, model_config
):
    high_hub = replace(reference_design, turbine_hub_tip_ratio=0.85)
    result = solve_mass_model_v2(
        derive_geometry_state(high_hub, model_config), model_config
    )
    assert result.component_volumes_m3["turbine_disk"] > 0.0
    assert result.component_masses_kg["turbine"] > 0.0


def test_lattice_density_changes_latticed_mass_but_cell_size_does_not(
    reference_design, model_config
):
    low_density = replace(reference_design, lattice_relative_density=0.15)
    high_density = replace(reference_design, lattice_relative_density=0.55)
    low = solve_mass_model_v2(
        derive_geometry_state(low_density, model_config), model_config
    )
    high = solve_mass_model_v2(
        derive_geometry_state(high_density, model_config), model_config
    )
    assert high.component_masses_kg["combustor"] > low.component_masses_kg[
        "combustor"
    ]
    assert high.component_masses_kg["nozzle"] > low.component_masses_kg[
        "nozzle"
    ]
    assert high.total_mass_kg > low.total_mass_kg

    large_cell = replace(reference_design, lattice_cell_size_m=0.014)
    large_cell_mass = solve_mass_model_v2(
        derive_geometry_state(large_cell, model_config), model_config
    )
    reference_mass = solve_mass_model_v2(
        derive_geometry_state(reference_design, model_config), model_config
    )
    assert large_cell_mass.total_mass_kg == pytest.approx(
        reference_mass.total_mass_kg
    )


def test_discrete_counts_have_direct_mass_paths(reference_design, model_config):
    more_blades = replace(reference_design, compressor_blade_count=20)
    more_injectors = replace(reference_design, combustor_injector_count=12)
    baseline = solve_mass_model_v2(
        derive_geometry_state(reference_design, model_config), model_config
    )
    blade_result = solve_mass_model_v2(
        derive_geometry_state(more_blades, model_config), model_config
    )
    injector_result = solve_mass_model_v2(
        derive_geometry_state(more_injectors, model_config), model_config
    )
    assert (
        blade_result.component_masses_kg["compressor"]
        > baseline.component_masses_kg["compressor"]
    )
    assert injector_result.component_masses_kg["injectors"] == pytest.approx(
        2.0 * baseline.component_masses_kg["injectors"]
    )
