"""R4 gates for the isolated 22D V3 software fixture assets."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.ai.v3_development_assets import (
    V3FixtureAssetError,
    build_v3_fixture_assets,
    load_v3_fixture_assets,
)
from src.physics.geometry22_design import Geometry22Design
from src.physics.v3_development_runtime import build_v3_development_physics_runtime


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "data/validation/v3_22d_software_fixture_v1"


@pytest.fixture(scope="module")
def runtime():
    return build_v3_development_physics_runtime(ROOT)


def test_fixture_has_22d_coverage_and_explicit_failure_rows(runtime):
    assets = load_v3_fixture_assets(ASSET_ROOT, runtime)
    designs = assets.candidate_designs()
    assert len(designs) == 45
    assert len({design.design_id() for design in designs}) == 45
    assert assets.manifest["converged_count"] == 41
    assert assets.manifest["failure_count"] == 4
    assert assets.manifest["axis_oat_coverage"] == 22
    assert assets.manifest["joint_coverage"] == "not_established"
    assert assets.manifest["formal_use_allowed"] is False
    assert assets.manifest["deployment_allowed"] is False
    reference = designs[0].as_mapping()
    for axis in range(22):
        for design in designs[1 + 2 * axis : 3 + 2 * axis]:
            changed = [key for key in reference if design.as_mapping()[key] != reference[key]]
            assert len(changed) == 1
        assert designs[1 + 2 * axis].design_id() != designs[2 + 2 * axis].design_id()


def test_forward_fixture_only_returns_stored_converged_labels(runtime):
    assets = load_v3_fixture_assets(ASSET_ROOT, runtime)
    known = assets.candidate_designs(1)[0]
    result = assets.predict_known(known)
    assert result["prediction_kind"] == "exact_stored_label_fixture_not_general_surrogate"
    assert result["performance"] == assets.rows[0]["performance"]
    assert result["physics_bundle_hash"] == runtime.physics_bundle_hash
    failed = next(row for row in assets.rows if row["performance"] is None)
    with pytest.raises(V3FixtureAssetError, match="No converged fixture label"):
        assets.predict_known(Geometry22Design.from_mapping(failed["design"]))
    unseen = Geometry22Design.from_mapping(
        {**known.as_mapping(), "compressor_tip_clearance_ratio": 0.0038}
    )
    with pytest.raises(V3FixtureAssetError, match="No converged fixture label"):
        assets.predict_known(unseen)
    with pytest.raises(V3FixtureAssetError, match="22D design"):
        assets.predict_known(known.geometry19_projection())


def test_fixture_is_reproducible_and_rejects_mismatch_or_tampering(runtime, tmp_path):
    output = tmp_path / "fixture"
    rebuilt = build_v3_fixture_assets(output, runtime)
    committed = load_v3_fixture_assets(ASSET_ROOT, runtime).manifest
    assert rebuilt["files"] == committed["files"]
    with pytest.raises(V3FixtureAssetError, match="already exists"):
        build_v3_fixture_assets(output, runtime)
    with pytest.raises(V3FixtureAssetError, match="identity or content mismatch"):
        load_v3_fixture_assets(output, replace(runtime, physics_bundle_hash="other"))
    case_file = output / "cases.jsonl"
    case_file.write_text(case_file.read_text(encoding="utf-8") + json.dumps({"extra": True}) + "\n")
    with pytest.raises(V3FixtureAssetError, match="Cannot load complete"):
        load_v3_fixture_assets(output, runtime)
