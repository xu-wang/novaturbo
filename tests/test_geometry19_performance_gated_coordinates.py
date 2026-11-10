"""G0b prelabel geometry, immutable inputs, split isolation and reproducibility."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.ai import geometry19_performance_gated_coordinates as coords
from src.ai.geometry19_performance_gated_protocol import ProtocolError, file_hash, read_rows

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/experiments/geometry19_performance_gated_g0b_coordinates_v1.yaml"


@pytest.fixture
def config():
    return yaml.safe_load(CONFIG.read_text())


@pytest.fixture
def geometry(config):
    parent = yaml.safe_load((ROOT / config["parent_protocol"]["path"]).read_text())
    source = parent["inputs"]
    anchors = {r["root_seed_id"]: r for r in read_rows(ROOT / source["anchor_registry"]["path"])}
    stable = {r["root_seed_id"]: r for r in read_rows(ROOT / source["stable_root_registry"]["path"])}
    rows = read_rows(ROOT / source["development_dataset"]["path"])
    variables = yaml.safe_load((ROOT / source["schema"]["path"]).read_text())["design_variables"]
    return anchors, stable, rows, variables


def test_full_lock_reproducibility_and_protected_inputs(tmp_path, config, monkeypatch):
    parent = yaml.safe_load((ROOT / config["parent_protocol"]["path"]).read_text())
    forbidden = {str((ROOT / parent["inputs"][k]["path"]).resolve()) for k in
                 ("sealed_confirmation", "physics_ledger", "s2_model_manifest")}
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        assert str(path.resolve()) not in forbidden
        assert path.suffix != ".npz"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    one = coords.run(ROOT, CONFIG, output_root=tmp_path / "one")
    two = coords.run(ROOT, CONFIG, output_root=tmp_path / "two")
    assert one == two
    assert one["status"] == coords.PASS
    assert one["audit"]["rows"] == 16896 and one["audit"]["pairs"] == 8448
    assert one["audit"]["split_rows"] == {"train": 13464, "validation": 3432}
    assert all(v == [128] * 4 for v in one["audit"]["per_root_band"].values())
    assert one["audit"]["exclusion_overlap"] == one["audit"]["pair_split_violations"] == 0
    assert not one["physics_evaluations_executed"] and not one["confirmation_content_read"]
    for artifact in one["artifacts"].values():
        assert file_hash(tmp_path / "one" / artifact["path"]) == artifact["sha256"]
    with pytest.raises(ProtocolError, match="overwrite"):
        coords.run(ROOT, CONFIG, output_root=tmp_path / "one")


@pytest.mark.parametrize("section,key,value", [
    ("algorithm", "integer_policy", "round"), ("algorithm", "expected_rows", 100),
    ("split", "before_physics_labels", False), ("authorization", "physics_acquisition", True),
    ("environment", "numpy", "different"),
])
def test_contract_changes_rejected(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ProtocolError):
        coords.validate_protocol(config)


def test_parent_hash_tamper_no_output(tmp_path, config):
    config["parent_lock"]["sha256"] = "0" * 64
    path = tmp_path / "altered.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ProtocolError, match="Input hash mismatch"):
        coords.run(ROOT, path, output_root=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_directions_do_not_use_performance_and_ignore_input_order(config, geometry):
    anchors, stable, rows, variables = geometry
    root = sorted(anchors)[0]
    root_rows = [r for r in rows if r["root_seed_id"] == root]
    first = coords.direction_model(root, root_rows, anchors[root]["medoid_geometry19"], variables, config["algorithm"])
    changed = deepcopy(root_rows[::-1])
    for row in changed:
        row["performance"] = {"thrust_N": float("nan")}
    second = coords.direction_model(root, changed, anchors[root]["medoid_geometry19"], variables, config["algorithm"])
    assert first[2] == second[2]
    assert np.array_equal(first[1][0], second[1][0])


def test_no_capacity_preserves_empty_band_without_forced_repair(config, geometry):
    anchors, stable, rows, variables = geometry
    root = sorted(anchors)[0]
    anchor = dict(anchors[root]["medoid_geometry19"])
    anchor["lattice_relative_density"] = 0.15  # continuous boundary: no symmetric step
    algorithm = dict(config["algorithm"], maximum_attempts_per_band=2)
    result = coords.generate_root(root, anchor, stable[root]["support_radius_p95_normalized_l2"],
                                  [r for r in rows if r["root_seed_id"] == root], variables, algorithm, set())
    assert not result[0] and not result[1]
    assert len(result[2]) == 8
    assert all(r["rejection_reason"] == "shrunk_step_outside_band" for r in result[2])


def test_independent_audit_rejects_pair_split_leak(config, geometry):
    anchors, stable, rows, variables = geometry
    root = sorted(anchors)[0]
    generated, pairs, _, _ = coords.generate_root(root, anchors[root]["medoid_geometry19"],
        stable[root]["support_radius_p95_normalized_l2"], [r for r in rows if r["root_seed_id"] == root],
        variables, dict(config["algorithm"], pairs_per_root_band=1), set())
    generated[0]["split"] = "train"
    with pytest.raises(ProtocolError, match="Split rule"):
        coords.audit_coordinates(generated, pairs, anchors, stable, variables, set(), config["algorithm"])


def test_hard_geometry_checks(config, geometry):
    anchors, _, _, variables = geometry
    g = dict(next(iter(anchors.values()))["medoid_geometry19"])
    assert coords.hard_failure(g, variables, .2) is None
    g["combustor_outer_diameter_m"] = g["combustor_inner_diameter_m"]
    assert coords.hard_failure(g, variables, .2) in {"schema_bounds_types", "combustor_annulus"}
