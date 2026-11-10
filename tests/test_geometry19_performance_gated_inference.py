"""Tests for the N1 performance-gated inference adapter."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.ai.geometry19_performance_gated_inference import (
    _assign_pooled_support,
    _decode_local,
    largest_remainder_top2,
    load_seed_models,
    validate_protocol,
)
from src.ai.geometry19_performance_gated_protocol import ProtocolError


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/experiments/geometry19_v2_2_performance_gated_inference_v1.yaml"
PREP = ROOT / "data/validation/v2_2_performance_gated_local_cfm_v1/training_preparation/g1_g2_v1"


def test_protocol_locks_far_runtime_budget_and_no_physics() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    validate_protocol(config)
    assert config["artifacts"]["evidence_manifest"] == "Evidence_Manifest.json"
    for mutation in (
        lambda value: value["physics"].update(model_version="geometry19_closed_v2_1"),
        lambda value: value["inference"].update(raw_candidates_per_task=128),
        lambda value: value["inference"]["ode"].update(steps=16),
        lambda value: value["authorization"].update(physics_evaluation_authorized=True),
    ):
        changed = deepcopy(config)
        mutation(changed)
        with pytest.raises(ProtocolError):
            validate_protocol(changed)


def test_top2_allocation_is_exact_and_root_id_breaks_ties() -> None:
    probabilities = np.asarray([0.1, 0.45, 0.45])
    allocation = largest_remainder_top2(probabilities, ["z", "b", "a"], 256)
    assert sum(allocation.values()) == 256
    assert set(allocation) == {1, 2}
    assert allocation[2] == allocation[1] == 128


def test_local_decoder_and_pooled_support_use_same_frozen_geometry_contract() -> None:
    decoders = json.loads((PREP / "Expert_Decoders.json").read_text())
    schema = yaml.safe_load((ROOT / "config/physics/geometry19_schema_v1.yaml").read_text())
    local = _decode_local(np.zeros((1, 16)), np.asarray([0]), decoders, schema, 0.20)[0]
    assert local["valid"] is True and local["local_distance"] == 0
    expert, distance = _assign_pooled_support(local["geometry19"], decoders, 0.20)
    assert expert["root_index"] == 0 and distance == pytest.approx(0.0)
    outside = _decode_local(np.ones((1, 16)), np.asarray([0]), decoders, schema, 0.20)[0]
    assert outside["valid"] is False and outside["rejection_reason"] == "outside_local_support"


def test_seed42_checkpoints_reload_strictly_on_cpu() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    decoders = json.loads((PREP / "Expert_Decoders.json").read_text())
    models, audit = load_seed_models(ROOT, config, 42, decoders, "cpu")
    assert set(models) == {"gate", "local_cfm", "pooled_cfm"}
    assert set(audit) == set(models)
    assert all(item["epoch"] > 0 for item in audit.values())
