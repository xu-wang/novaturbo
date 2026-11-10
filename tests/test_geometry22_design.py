from pathlib import Path

import pytest
import yaml

from src.physics.geometry22_design import (
    Geometry22Design,
    Geometry22DesignError,
    SCHEMA_PATH,
    geometry22_schema_hash,
    geometry22_schema_metadata,
)
from src.physics.geometry22_data import (
    Geometry22DataError,
    Geometry22DesignStore,
    Geometry22Normalizer,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_SCHEMA = ROOT / "config/physics/geometry19_schema_v1.yaml"


def _reference_values():
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    return {row["name"]: row["reference_value"] for row in schema["design_variables"]}


def test_schema_preserves_19d_prefix_and_freezes_three_extensions():
    base = yaml.safe_load(BASE_SCHEMA.read_text(encoding="utf-8"))
    extended = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert extended["design_variables"][:19] == base["design_variables"]
    assert [row["index"] for row in extended["design_variables"]] == list(range(1, 23))
    assert [row["name"] for row in extended["design_variables"][19:]] == [
        "diffuser_area_ratio",
        "combustor_primary_air_fraction",
        "compressor_tip_clearance_ratio",
    ]
    assert extended["runtime_enabled"] is False
    assert extended["formal_dataset_generation_ready"] is False
    assert len(geometry22_schema_hash()) == 64


def test_mapping_roundtrip_and_geometry_projection():
    values = _reference_values()
    design = Geometry22Design.from_mapping(values)
    assert design.as_mapping() == values
    assert design.geometry19_projection().as_mapping() == {
        name: values[name] for name in Geometry22Design.coordinate_names()[:19]
    }
    assert design.component_parameters() == {
        name: values[name] for name in Geometry22Design.coordinate_names()[19:]
    }


def test_implicit_19d_extension_and_invalid_prototype_coordinate_are_rejected():
    values = _reference_values()
    with pytest.raises(Geometry22DesignError, match="missing"):
        Geometry22Design.from_mapping({k: v for k, v in values.items() if k != "diffuser_area_ratio"})
    with pytest.raises(Geometry22DesignError, match="compressor_tip_clearance_ratio"):
        Geometry22Design.from_mapping({**values, "compressor_tip_clearance_ratio": 0.01})
    with pytest.raises(Geometry22DesignError, match="compressor_blade_count"):
        Geometry22Design.from_mapping({**values, "compressor_blade_count": 12.0})


def test_normalization_roundtrip_and_explicit_integer_policy():
    design = Geometry22Design.from_mapping(_reference_values())
    normalizer = Geometry22Normalizer()
    encoded = normalizer.transform(design)
    assert len(encoded) == 22
    assert normalizer.schema_hash == geometry22_schema_hash()
    assert normalizer.inverse(encoded).as_mapping() == design.as_mapping()
    blade_index = Geometry22Design.coordinate_names().index("compressor_blade_count")
    proposal = list(encoded)
    proposal[blade_index] += 0.03
    with pytest.raises(Geometry22DataError, match="explicit integer rounding"):
        normalizer.inverse(proposal)
    assert isinstance(normalizer.inverse(proposal, round_integers=True).coordinates[blade_index], int)


def test_design_records_keep_22d_identity_separate_from_geometry_projection(tmp_path):
    values = _reference_values()
    design = Geometry22Design.from_mapping(values)
    changed = Geometry22Design.from_mapping({**values, "diffuser_area_ratio": 2.6})
    assert design.design_id() != changed.design_id()
    assert design.geometry19_projection_hash() == changed.geometry19_projection_hash()
    store = Geometry22DesignStore(tmp_path)
    first = store.publish(design)
    second = store.publish(changed)
    assert first == store.read(design.design_id())
    assert first["design_id"] != second["design_id"]
    assert first["geometry_asset_id"] is None
    assert first["formal_use_allowed"] is False
    assert len(list(tmp_path.rglob("design.json"))) == 2
    metadata = geometry22_schema_metadata()
    assert len(metadata["design_variables"]) == 22
    assert metadata["runtime_enabled"] is False


def test_tampered_design_record_is_rejected(tmp_path):
    design = Geometry22Design.from_mapping(_reference_values())
    store = Geometry22DesignStore(tmp_path)
    store.publish(design)
    path = tmp_path / "geometry22_schema_v3_development_v1" / design.design_id() / "design.json"
    source = path.read_text(encoding="utf-8")
    path.write_text(source.replace('"formal_use_allowed": false', '"formal_use_allowed": true'))
    with pytest.raises(Geometry22DataError, match="mismatch"):
        store.read(design.design_id())


def test_schema_metadata_endpoint_is_read_only_and_marks_runtime_disabled():
    pytest.importorskip("flask")
    from ui.server import app

    client = app.test_client()
    response = client.get("/api/geometry22/schema")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["schema_hash"] == geometry22_schema_hash()
    assert payload["prototype_dimension"] == 22
    assert payload["geometry_projection_dimension"] == 19
    assert payload["runtime_enabled"] is False
    assert payload["physics_validation_status"] == "not_completed"
