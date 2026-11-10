from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest

from src.ai import geometry19_performance_gated_training_preparation as m
from src.ai.geometry19_performance_gated_protocol import ProtocolError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Preparation must not evaluate physics or load model arrays")
    monkeypatch.setattr("src.physics.runtime.PhysicsRuntime.evaluate", forbidden)
    monkeypatch.setattr(np, "load", forbidden)


def fixture_rows():
    return [{"split": "train", "training_eligible": True, "root_seed_id": root,
             "design_hash": str(i), "performance": dict(zip(m.OUTPUTS, np.exp([x, x+1, 2*x])))}
            for i, (root, x) in enumerate([( "a", 1.), ("b", 2.), ("b", 4.)])]


def test_root_equal_transform():
    rows = fixture_rows()
    transform = m.fit_condition(rows, ["a", "b"])
    assert transform["log_mean"] == pytest.approx([2., 3., 4.])
    assert transform["log_std"] == pytest.approx(np.sqrt([1.5, 1.5, 6.]))
    conditions = np.array([m.transform_condition(r["performance"], transform) for r in rows])
    assert np.array([.5, .25, .25]) @ conditions == pytest.approx([0., 0., 0.])
    with pytest.raises(ProtocolError):
        m.transform_condition({**rows[0]["performance"], "root_id": "a"}, transform)


@pytest.mark.parametrize("mutation", ["validation", "ineligible", "missing_root", "zero_variance"])
def test_invalid_fit_rejected(mutation):
    rows = fixture_rows()
    if mutation == "validation": rows[0]["split"] = "validation"
    elif mutation == "ineligible": rows[0]["training_eligible"] = False
    elif mutation == "missing_root": rows = rows[:1]
    else:
        for r in rows: r["performance"] = dict.fromkeys(m.OUTPUTS, 1.)
    with pytest.raises(ProtocolError): m.fit_condition(rows, ["a", "b"])


def test_validation_bank_reproducible():
    rows = [{"design_hash": str(i)} for i in range(5)]
    first = m.validation_noise(rows, 42)
    assert first == m.validation_noise(rows, 42)
    assert first != m.validation_noise(rows, 43)
    assert all(len(r["epsilon"]) == 16 and 0 <= r["time"] < 1 for r in first)


def test_validation_values_never_enter_fit():
    rows = fixture_rows()
    validation = deepcopy(rows[0])
    validation.update(split="validation", training_eligible=False)
    all_rows = rows + [validation]
    before = m.fit_condition([r for r in all_rows if r["training_eligible"]], ["a", "b"])
    validation["performance"] = dict.fromkeys(m.OUTPUTS, 1e30)
    after = m.fit_condition([r for r in all_rows if r["training_eligible"]], ["a", "b"])
    assert before == after


def test_real_preparation_deterministic_no_execution(tmp_path):
    first, second = tmp_path/"first", tmp_path/"second"
    a = m.run(ROOT, output_override=first)
    b = m.run(ROOT, output_override=second)
    assert a == b
    assert a["rows"] == {"train": 12153, "validation": 3110, "excluded": 1633}
    assert not a["training_executed"] and a["physical_evaluations"] == 0
    for name, spec in a["artifacts"].items():
        assert (first/name).read_bytes() == (second/name).read_bytes()
        assert m.file_hash(first/name) == spec["sha256"]
    train = m.read_rows(first/"Train_Registry.jsonl")
    val = m.read_rows(first/"Validation_Registry.jsonl")
    assert set(r["pair_id"] for r in train).isdisjoint(r["pair_id"] for r in val)
    assert all(len(r["condition"]) == 3 and len(r["local_state"]) == 16 for r in train+val)
    assert len(set(r["supervision"]["root_seed_id"] for r in train)) == 33
    with pytest.raises(ProtocolError): m.run(ROOT, output_override=first)


def test_project_roundtrip_and_support_rejection():
    variables = [{"name": f"x{i}", "dtype": "float64", "bounds": [0, 10]} for i in range(16)]
    variables += [{"name": f"n{i}", "dtype": "int64", "bounds": [1, 3]} for i in range(3)]
    geometry = {v["name"]: (2 if v["dtype"] == "int64" else 5.) for v in variables}
    decoders = m.make_decoders(["a"], {"a": {"medoid_geometry19": geometry, "medoid_design_hash": "anchor"}},
                               {"a": {"support_radius_p95_normalized_l2": .5}}, variables)
    row = {**fixture_rows()[0], "geometry19": deepcopy(geometry), "pair_id": "pair", "band_index": 0}
    transform = m.fit_condition(fixture_rows(), ["a", "b"])
    row["geometry19"]["x0"] += .5
    result = m.project([row], transform, decoders)[0]
    assert result["local_state"][0] == pytest.approx(.1)
    assert result["integer_class_indices"] == [1, 1, 1]
    row["geometry19"]["x0"] = 7.
    with pytest.raises(ProtocolError): m.project([row], transform, decoders)
    row["geometry19"]["x0"] = 5.
    row["geometry19"]["n0"] = 1
    with pytest.raises(ProtocolError): m.project([row], transform, decoders)
