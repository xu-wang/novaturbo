"""V3 L6a target routing split and source isolation checks."""

import json
from pathlib import Path
import shutil

import pytest

from src.ai import v3_target_route_protocol as route


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/validation/v3_22d_target_route_l6a_v1"


def test_frozen_route_population_and_pair_isolation():
    lock = route.verify_v3_target_route_freeze(ROOT, OUTPUT)
    route.verify_v3_target_route_complete(ROOT, OUTPUT)
    rows = route._rows(OUTPUT / "target_registry.jsonl")
    support = json.loads((OUTPUT / "center_support.json").read_text())["centers"]
    audit = json.loads((OUTPUT / "cross_center_audit.json").read_text())
    assert len(lock["route_center_ids"]) == len(support) == 23
    assert len(rows) == 11289
    assert {r["internal_split"] for r in rows} == {"fit", "tuning"}
    pair_splits = {}
    for row in rows:
        key = (row["source_center_id"], row["pair_key"])
        pair_splits.setdefault(key, row["internal_split"])
        assert pair_splits[key] == row["internal_split"]
    assert lock["runtime_condition"] == ["thrust_N", "tsfc_kg_N_s", "mass_kg"]
    assert lock["training_executed"] is False
    assert lock["holdout_content_parsed"] is False
    assert audit["all_validation_targets"]["targets"] == 2394
    assert audit["all_validation_targets"]["eligible_any"] == 1373
    assert sum(v["eligible_any"] == 0 for v in audit["all_validation_targets"]["per_source_center"].values()) == 2
    assert audit["physics_evaluations_executed"] == 0


def test_complete_verification_never_parses_old_holdout(monkeypatch):
    original = route._rows

    def guarded(path):
        if "holdout_candidates" in str(path):
            raise AssertionError("Old holdout content must not be parsed")
        return original(path)

    monkeypatch.setattr(route, "_rows", guarded)
    route.verify_v3_target_route_complete(ROOT, OUTPUT)


def test_changed_routing_normalization_is_rejected(tmp_path):
    copy = tmp_path / "audit"
    shutil.copytree(OUTPUT, copy)
    normalization = copy / "normalization.json"
    data = json.loads(normalization.read_text())
    data["log_mean"][0] += 0.1
    normalization.write_text(json.dumps(data))
    with pytest.raises(route.V3TargetRouteProtocolError, match="normalization.json"):
        route.verify_v3_target_route_complete(ROOT, copy)
