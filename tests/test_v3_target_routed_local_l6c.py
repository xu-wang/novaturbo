"""V3 L6c target privacy, frozen selection and backend result checks."""

import json
from pathlib import Path

import numpy as np

from src.ai.v3_target_routed_local_l6c import (
    _runtime_target,
    verify_v3_routed_local_complete,
    verify_v3_routed_local_selection,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/validation/v3_22d_target_routed_local_l6c_v1"


def rows(name):
    return [json.loads(line) for line in (OUTPUT / name).read_text().splitlines()]


def test_runtime_target_hides_source_center_and_selection_precedes_physics():
    selected_manifest = verify_v3_routed_local_selection(ROOT, OUTPUT)
    original = json.loads((ROOT / "data/validation/v3_22d_target_route_l6a_v1/cross_center_targets.jsonl").read_text().splitlines()[0])
    assert _runtime_target(original) == _runtime_target({**original, "source_center_id_for_offline_audit_only": "different"})
    assert set(_runtime_target(original)) == {"target_id", "target_performance"}
    routes = rows("route_ledger.jsonl")
    selected = rows("selection_ledger.jsonl")
    assert len(routes) == 15 and len(selected) == 216
    assert sum(row["status"] == "abstained_no_target_support" for row in routes) == 6
    assert selected_manifest["physics_evaluations_executed"] == 0
    assert all("source_center_id" not in row and "source_center_id_for_offline_audit_only" not in row
               for row in routes + selected)
    assert all(len([row for row in selected if row["target_id"] == route["target_id"] and row["arm"] == arm]) == 8
               for route in routes if route["status"] == "routable" for arm in route["arms"])


def test_selected_designs_keep_local_support_and_integer_state():
    selected = rows("selection_ledger.jsonl")
    support = json.loads((ROOT / "data/validation/v3_22d_target_route_l6a_v1/center_support.json").read_text())["centers"]
    centers = {item["center_id"]: item for item in support}
    continuous = json.loads((ROOT / "data/validation/v3_22d_target_gate_l6b_gpu6_v1/local_forward_models.json").read_text())["input_continuous_indices_zero_based"]
    for row in selected:
        center = centers[row["route_center_id"]]
        distance = np.linalg.norm(np.asarray(row["normalized_design"])[continuous] - np.asarray(center["center_normalized_design"])[continuous])
        assert distance <= 0.08 + 1e-10
        assert all(row["design"][name] == value for name, value in center["integer_state"].items())


def test_v3_physics_and_f_recompute_are_sealed():
    complete = verify_v3_routed_local_complete(ROOT, OUTPUT)
    physics = rows("physics_ledger.jsonl")
    f_rows = rows("f_recompute.jsonl")
    assert len(physics) == 216 and len(f_rows) == 27
    assert all(row["solver_converged"] for row in physics)
    assert all(row["independent_v3_recompute_equal"] for row in f_rows)
    assert all(arm["software_hit_targets_5pct"] == 9 for arm in complete["arms"].values())
    assert all(arm["engine_feasible"] == 0 for arm in complete["arms"].values())
    assert complete["old_validation_reused_for_diagnostic"] is True
    assert complete["old_holdout_labels_parsed"] is False
    assert complete["formal_use_allowed"] is False
