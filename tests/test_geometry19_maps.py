"""Contracts for synthetic component maps and variable-property thermo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.physics.maps import MapDomainError, MapPolicy, load_component_map
from src.physics.maps.hecc import build_hecc_map_data, load_hecc_points
from src.physics.maps.validation import validate_component_map_package
from src.physics.thermo import load_variable_property_thermo


REPO_ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = (
    REPO_ROOT
    / "config/physics/maps/compressor/synthetic_centrifugal_compressor_v1"
)
THERMO_PATH = (
    REPO_ROOT / "config/physics/geometry19_thermo_variable_property_shadow_v1.yaml"
)
HECC_MAP_PATH = REPO_ROOT / "config/physics/maps/compressor/nasa_hecc_post_test_cfd_v1"


def test_map_loads_with_complete_identity_and_exact_grid_value() -> None:
    component_map = load_component_map(MAP_PATH)
    result = component_map.evaluate(
        {"corrected_speed_ratio": 1.0, "flow_coefficient_ratio": 1.0}
    )

    assert component_map.component_type == "compressor"
    assert len(component_map.source_hash) == 64
    assert result.values["euler_work_factor"] == pytest.approx(1.01)
    assert result.values["total_to_total_efficiency"] == pytest.approx(0.81)
    assert result.in_domain is True
    assert result.extrapolation_distance == 0.0


def test_map_interpolates_and_enforces_explicit_domain_policies() -> None:
    component_map = load_component_map(MAP_PATH)
    midpoint = component_map.evaluate(
        {"corrected_speed_ratio": 0.875, "flow_coefficient_ratio": 0.875}
    )
    assert midpoint.values["total_to_total_efficiency"] == pytest.approx(0.795)

    query = {"corrected_speed_ratio": 0.48, "flow_coefficient_ratio": 1.0}
    diagnostic = component_map.evaluate(query, policy=MapPolicy.DIAGNOSTIC)
    assert diagnostic.in_domain is False
    assert diagnostic.extrapolation_distance > 0.0
    assert diagnostic.uncertainty["total_to_total_efficiency"] > 0.025
    with pytest.raises(MapDomainError, match="engineering_candidate"):
        component_map.evaluate(query, policy=MapPolicy.ENGINEERING_CANDIDATE)
    with pytest.raises(MapDomainError, match="not approved for formal use"):
        component_map.evaluate(
            {"corrected_speed_ratio": 1.0, "flow_coefficient_ratio": 1.0},
            policy=MapPolicy.FORMAL,
        )
    with pytest.raises(MapDomainError, match="normalized_distance"):
        component_map.evaluate(
            {"corrected_speed_ratio": 0.2, "flow_coefficient_ratio": 1.0}
        )


def test_map_treats_roundoff_at_coordinate_boundary_as_in_domain() -> None:
    component_map = load_component_map(HECC_MAP_PATH)
    result = component_map.evaluate(
        {
            "corrected_speed_ratio": 100000.0 * 1.05 / 100000.0,
            "flow_coefficient_ratio": 1.0,
        }
    )

    assert result.in_domain is True
    assert result.extrapolation_distance == 0.0


def test_variable_thermo_enthalpy_is_monotone_and_invertible() -> None:
    thermo = load_variable_property_thermo(THERMO_PATH)
    temperatures = (220.0, 288.15, 600.0, 1100.0, 1800.0)
    enthalpies = [thermo.h_air(value) for value in temperatures]

    assert enthalpies == sorted(enthalpies)
    for temperature, enthalpy in zip(temperatures, enthalpies, strict=True):
        assert thermo.air_temperature_from_h(enthalpy) == pytest.approx(
            temperature, abs=1.0e-9
        )
    assert 1.2 < thermo.gamma_air(1100.0) < 1.4
    assert thermo.mu_air(1100.0) > thermo.mu_air(288.15)
    assert thermo.air_isentropic_pressure_ratio(288.15, 400.0) > 1.0
    gas_enthalpy = thermo.h_gas(900.0, 0.02)
    assert thermo.gas_temperature_from_h(gas_enthalpy, 0.02) == pytest.approx(
        900.0, abs=1.0e-9
    )
    assert thermo.gas_isentropic_pressure_ratio(900.0, 700.0, 0.02) < 1.0
    assert thermo.mu_gas(900.0, 0.02) > 0.0


def test_hecc_source_build_matches_checked_in_map_and_passes_shadow_gate() -> None:
    points = load_hecc_points(HECC_MAP_PATH / "source_points.csv")
    generated = build_hecc_map_data(points)
    component_map = load_component_map(HECC_MAP_PATH)

    assert len(points) == 62
    for name in component_map.coordinate_names:
        assert component_map.coordinates[name] == pytest.approx(
            generated["coordinates"][name]
        )
    for name in component_map.output_names:
        assert np.allclose(
            component_map.outputs[name], generated["outputs"][name], atol=1.0e-9
        )
    report = validate_component_map_package(HECC_MAP_PATH)
    assert report["structural_pass"] is True
    assert report["shadow_ready"] is True
    assert report["engineering_ready"] is False
    evidence = report["evidence_validation"]
    assert evidence["validation_point_count"] == 1
    assert evidence["metrics"]["pressure_ratio_relative"][
        "max_absolute_error"
    ] < 0.05
    assert evidence["metrics"]["efficiency_absolute"][
        "max_absolute_error"
    ] < 0.06


def test_hecc_map_uses_direct_work_coefficient_and_traceable_source() -> None:
    component_map = load_component_map(HECC_MAP_PATH)
    result = component_map.evaluate(
        {"corrected_speed_ratio": 1.0, "flow_coefficient_ratio": 1.0}
    )

    assert 0.7 < result.values["work_coefficient"] < 0.9
    assert "euler_work_factor" not in result.values
    assert component_map.metadata["source"]["document_id"] == "20180001471"
    assert component_map.formal_use_allowed is False
