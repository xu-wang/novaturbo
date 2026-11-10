"""G3 scientific contracts, failure preservation, and sealed resume checks."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.ai import geometry19_performance_gated_g3 as g
from src.ai.geometry19_performance_gated_protocol import ProtocolError
from src.ai.geometry19_pipeline_context import build_pipeline_context

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "data/validation/v2_2_performance_gated_local_cfm_v1/training_preparation/g1_g2_v1"


@pytest.fixture(scope="module")
def decoder_data():
    decoders = json.loads((PREP / "Expert_Decoders.json").read_text())
    schema = yaml.safe_load((ROOT / "config/physics/geometry19_schema_v1.yaml").read_text())
    context = build_pipeline_context(ROOT, pipeline_profile="v2_2_development")
    return decoders, schema, context.physics_runtime.model_config


def test_frozen_budget_and_no_physics_config():
    config = yaml.safe_load((ROOT / g.CONFIG).read_text())
    g.validate_config(config)
    changed = deepcopy(config)
    changed["authorization"]["G4_physics"] = True
    with pytest.raises(ProtocolError):
        g.validate_config(changed)


def test_source_matrix_shared_and_distinct_across_targets():
    for seed in (20261001, 20261002, 20261003):
        a, u = g.source_bank(seed, 5)
        b, v = g.source_bank(seed, 5)
        assert a.shape == (256, 16) and a.dtype == np.float32
        assert np.array_equal(a, b) and np.array_equal(u, v)
        assert not np.array_equal(a, g.source_bank(seed, 6)[0])


def test_top2_largest_remainder_extremes_and_ties():
    assert g.dispatch_for(g.ARMS[1], np.array([.5, .5]), ["b", "a"]).tolist().count(0) == 128
    dispatch = g.dispatch_for(g.ARMS[1], np.array([1., 0.]), ["a", "b"])
    assert dispatch.tolist().count(0) == 255 and dispatch.tolist().count(1) == 1
    dispatch = g.dispatch_for(g.ARMS[1], np.array([.5, .5]), ["b", "a"], count=255)
    assert dispatch.tolist().count(1) == 128


@pytest.mark.parametrize("arm", [g.ARMS[0], g.ARMS[1], g.ARMS[3]])
def test_non_oracle_cannot_receive_source_root(arm):
    with pytest.raises(ProtocolError, match="Oracle root forbidden"):
        g.dispatch_for(arm, np.ones(33) / 33, [str(i) for i in range(33)], oracle_root="0")


def test_inference_rejects_private_target_fields_before_model_use():
    target = {"target_id": "x", "performance": {}, "units": g.UNITS, "root_seed_id": "secret"}
    with pytest.raises(ProtocolError, match="public target"):
        g.generate_raw(target, 0, 42, g.ARMS[0], {}, {})


def test_overlapping_supports_score_same_geometry_same_root(decoder_data):
    decoders, schema, calibration = decoder_data
    decoders = deepcopy(decoders)
    # Duplicate the first center into a second expert to force a support tie.
    e = deepcopy(decoders["experts"][0])
    e.update(root_index=1, root_seed_id="zzzz")
    decoders["experts"] = [decoders["experts"][0], e]
    states = np.zeros((1, 16))
    a = g.decode_rows(states, np.array([0]), None, decoders, schema, calibration)[0]
    b = g.decode_rows(states, np.array([1]), None, decoders, schema, calibration)[0]
    assert a["geometry19"] == b["geometry19"] and a["design_hash"] == b["design_hash"]
    assert a["support_root_id"] == b["support_root_id"] == decoders["experts"][0]["root_seed_id"]
    assert a["valid"] and b["valid"] and a["geometry_hard_valid"]


def test_nonfinite_and_outside_slots_retained_without_projection(decoder_data):
    decoders, schema, calibration = decoder_data
    rows = g.decode_rows(np.array([[np.nan] * 16, [1.] * 16]), np.array([0, 0]), None, decoders, schema, calibration)
    assert len(rows) == 2
    assert rows[0]["rejection_reason"] == "nonfinite_generation" and rows[0]["geometry19"] is None
    assert rows[0]["generated_state"] == [None] * 16
    assert rows[1]["rejection_reason"] == "outside_local_support"
    assert rows[1]["generator_support_distance"] == 4.
    json.dumps(rows, allow_nan=False)


def test_existing_geometry_hard_check_is_enforced(decoder_data, monkeypatch):
    decoders, schema, calibration = decoder_data
    def failed(*args):
        raise g.GeometryStateError("synthetic hard failure")
    monkeypatch.setattr(g, "derive_geometry_state", failed)
    row = g.decode_rows(np.zeros((1, 16)), np.array([0]), None, decoders, schema, calibration)[0]
    assert not row["valid"] and row["rejection_reason"] == "geometry_hard_rejection"


def test_ranking_dedup_ties_full_rank_and_missing_slots(monkeypatch):
    raw = [dict(raw_slot_id=i, target_id="t", target_index=0, arm=g.ARMS[0], training_seed=42,
                generation_seed=20261001, valid=True, design_hash=h, geometry19={"x": i}, support_root_id="r")
           for i, h in enumerate(["b", "a", "a"])]
    def score(rows, *args):
        for row in rows:
            row.update(predicted_rho=1., predicted_performance={"x": 1.})
    monkeypatch.setattr(g.adapter, "_score", score)
    inputs = {"parent": {"evaluation": {"relative_tolerances": {k: 1. for k in g.adapter.OUTPUT_NAMES}}}, "decoders": {}, "schema": {}}
    prediction, selected = g.rank_rows(raw, {}, inputs, [], {})
    assert [r["design_hash"] for r in selected[:2]] == ["a", "b"]
    assert prediction[2]["duplicate_of_raw_slot"] == 1 and prediction[2]["rank"] is None
    assert len(prediction) == 3 and len(selected) == 8 and sum(r["missing"] for r in selected) == 6
    assert all(r["geometry19"] is None for r in selected[2:])


def test_shard_verification_rejects_tamper_wrong_context_and_partial(tmp_path):
    folder = tmp_path / "a"
    identity = {"context": "new"}
    payload = {"Raw.jsonl": [{"slot": 0}], "Task.json": {"target": "x"}}
    names = list(payload)
    expected = g.shard_write(folder, identity, payload)
    assert g.shard_read(folder, identity, names) == expected
    with pytest.raises(ProtocolError, match="identity"):
        g.shard_read(folder, {"context": "old"}, names)
    (folder / "Raw.jsonl").write_text('{"slot":1}\n')
    with pytest.raises(ProtocolError, match="hash"):
        g.shard_read(folder, identity, names)
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    (orphan / "Raw.jsonl").write_text("{}\n")
    with pytest.raises(ProtocolError, match="Orphan"):
        g.shard_read(orphan, identity, names)


def test_classical_rk4_solves_constant_velocity_without_mutating_source():
    class Constant:
        def __call__(self, state, time, condition):
            return torch.ones_like(state)
    source = torch.zeros(2, 16)
    result = g.adapter.rk4_integrate(Constant(), source, torch.zeros(2, 3), steps=32)
    assert torch.equal(result, torch.ones_like(source)) and torch.equal(source, torch.zeros_like(source))


def test_preflight_constructs_exact_frozen_runtime_without_physics(monkeypatch):
    monkeypatch.setattr(g.PhysicsRuntime, "evaluate", g.no_physics)
    data = g.preflight(ROOT)
    assert data["context"].physics_runtime.model_version == "geometry19_v2_2_far_precision_v1"
    assert data["context"].context_hash == "49e9b09a047e096f59bd65132fcfaef5b8dc363d563d2be638247e7eaf55eb72"
    assert len(data["runtime_targets"]) == 99
