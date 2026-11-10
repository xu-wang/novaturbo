"""Contract and migration tests for the Geometry19 component-map registry."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.physics import Environment, Geometry19
from src.physics.maps import (
    ComponentMapRegistryError,
    load_component_map,
    load_component_map_registry,
)
from src.physics.runtime import PhysicsRuntimeError, build_physics_runtime


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "config/physics/geometry19_component_maps_v1.yaml"
EXPECTED_REFERENCE_RESULTS = {
    "v2_2_shadow": {
        "map_id": "synthetic_centrifugal_compressor_v1",
        "thrust_N": 62.0390199637639,
        "tsfc_kg_N_s": 7.448741693950262e-05,
        "mass_flow_air_kg_s": 0.224210095082749,
        "pressure_ratio": 2.364011006069505,
        "efficiency": 0.7343901399453603,
    },
    "v2_2_hecc_shadow": {
        "map_id": "nasa_hecc_post_test_cfd_v1",
        "thrust_N": 79.57619253274157,
        "tsfc_kg_N_s": 6.485442250717271e-05,
        "mass_flow_air_kg_s": 0.2579193197357817,
        "pressure_ratio": 2.7709354349500095,
        "efficiency": 0.7292033416429852,
    },
}


def _write_registry(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / REGISTRY_PATH.name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _reference_design(context) -> Geometry19:
    return Geometry19.from_mapping(
        {
            item["name"]: item["reference_value"]
            for item in context.physics_runtime.design_schema["design_variables"]
        }
    )


def test_registry_loads_each_package_with_matching_identity() -> None:
    registry = load_component_map_registry(REPO_ROOT, REGISTRY_PATH)

    assert registry.registry_version == "geometry19_component_maps_v1"
    assert len(registry.source_hash) == 64
    assert set(registry.registrations) == {
        "synthetic_centrifugal_compressor_v1",
        "nasa_hecc_post_test_cfd_v1",
        "synthetic_single_stage_turbine_v1",
        "synthetic_subsonic_inlet_v1",
        "synthetic_annular_combustor_v1",
        "synthetic_convergent_nozzle_v1",
    }
    for registration in registry.registrations.values():
        direct = load_component_map(registration.package_path)
        assert registration.component_type == direct.component_type
        assert registration.component_map.source_hash == direct.source_hash
        assert registration.map_id == direct.map_id
        assert registration.map_version == direct.map_version
        assert registration.source_hash == direct.source_hash
        assert registration.formal_use_allowed is direct.formal_use_allowed


@pytest.mark.parametrize("profile_name", tuple(EXPECTED_REFERENCE_RESULTS))
def test_shadow_profiles_select_registered_map_ids_without_paths(
    profile_name: str,
) -> None:
    expected = EXPECTED_REFERENCE_RESULTS[profile_name]
    context = build_pipeline_context(REPO_ROOT, pipeline_profile=profile_name)
    runtime = context.physics_runtime

    assert "compressor_map" not in context.protocol["contracts"]
    assert context.protocol["physics_evaluation"]["component_maps"] == {
        "compressor": expected["map_id"]
    }
    assert runtime.component_map_registry is not None
    assert runtime.component_map_registry.registry_version == (
        "geometry19_component_maps_v1"
    )
    assert len(runtime.provenance_hashes["component_map_registry_hash"]) == 64
    map_contract = runtime.contract_mapping()["component_maps"]["compressor"]
    assert map_contract["usage_level"] in {
        "software_fixture",
        "validation_shadow",
    }
    assert map_contract["formal_use_allowed"] is False
    assert runtime.source_paths["compressor_map"].name == expected["map_id"]
    assert runtime.component_maps["compressor"].map_id == expected["map_id"]

    result = runtime.evaluate(_reference_design(context), Environment())
    compressor = result.components["compressor"]
    assert result.performance["thrust_N"] == pytest.approx(
        expected["thrust_N"], rel=1.0e-12
    )
    assert result.performance["tsfc_kg_N_s"] == pytest.approx(
        expected["tsfc_kg_N_s"], rel=1.0e-12
    )
    assert result.operating_point["mass_flow_air_kg_s"] == pytest.approx(
        expected["mass_flow_air_kg_s"], rel=1.0e-12
    )
    assert compressor["pressure_ratio"] == pytest.approx(
        expected["pressure_ratio"], rel=1.0e-12
    )
    assert compressor["total_to_total_efficiency"] == pytest.approx(
        expected["efficiency"], rel=1.0e-12
    )


def test_registry_rejects_duplicate_map_ids(tmp_path: Path) -> None:
    payload = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    payload["maps"].append(dict(payload["maps"][0]))

    with pytest.raises(ComponentMapRegistryError, match="Duplicate registered map ID"):
        load_component_map_registry(REPO_ROOT, _write_registry(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("map_version", "9.9.9", "identity mismatch"),
        ("source_hash", "0" * 64, "identity mismatch"),
        ("component_type", "turbine", "identity mismatch"),
        ("formal_use_allowed", True, "Only formal registry entries"),
        ("package", "/tmp/outside-map", "escapes the repository root"),
    ),
)
def test_registry_rejects_registration_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    payload["maps"][0][field] = value

    with pytest.raises(ComponentMapRegistryError, match=message):
        load_component_map_registry(REPO_ROOT, _write_registry(tmp_path, payload))


def test_runtime_rejects_unregistered_protocol_selection() -> None:
    context = build_pipeline_context(REPO_ROOT, pipeline_profile="v2_2_shadow")
    protocol = deepcopy(context.protocol)
    protocol["physics_evaluation"]["component_maps"]["compressor"] = "unknown_map"

    with pytest.raises(PhysicsRuntimeError, match="Unregistered component map"):
        build_physics_runtime(REPO_ROOT, protocol)
