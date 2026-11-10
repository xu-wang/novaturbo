import json
from pathlib import Path
import shutil

import pytest
import torch

from src.ai.v3_fresh_validation_l6h import allocation, config_at, validation_coordinates
from src.ai.v3_target_routed_local_l6c import _route, _runtime_target
from src.ai.v3_validation_integrity import dependency_hashes, sha, verify_complete

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/validation/v3_22d_extended_route_l6g_v2"


def test_registered_split_and_runtime_target_privacy(tmp_path):
    cfg = config_at(ROOT)
    coordinates = validation_coordinates(ROOT, cfg)
    assert len(coordinates) == 6
    public = _runtime_target({**coordinates[0], "target_performance": {"thrust_N": 1., "tsfc_kg_N_s": 1., "mass_kg": 1.}})
    assert set(public) == {"target_id", "target_performance"}
    registry = [json.loads(s) for s in (ROOT / cfg["sources"]["registry_path"]).read_text().splitlines()]
    registry[0]["root_id"] = registry[-1]["root_id"]
    path = tmp_path / "registry.jsonl"
    path.write_text("\n".join(json.dumps(t) for t in registry))
    cfg["sources"]["registry_path"] = str(path)
    with pytest.raises(ValueError, match="split leakage"):
        validation_coordinates(ROOT, cfg)


def test_abstain_and_single_center_budget():
    # Synthetic logits only: tests support masking, not CUDA model inference.
    model = lambda x: torch.zeros((1, 1))
    public = {"target_id": "test", "target_performance": {"thrust_N": 1., "tsfc_kg_N_s": 1., "mass_kg": 1.}}
    norm = {"log_mean": [0, 0, 0], "log_std": [1, 1, 1]}
    support = [{"center_id": "a", "target_centroid_normalized_log": [0, 0, 0], "target_radius_p95_normalized_log": 1}]
    route = _route(public, norm, support, model, torch.device("cpu"))
    assert route["arms"]["gated_top2"] == ["a"]
    assert allocation(route["arms"]["gated_top2"]) == 256
    assert allocation(["a", "b"]) == 128
    support[0]["target_centroid_normalized_log"] = [2, 2, 2]
    route = _route(public, norm, support, model, torch.device("cpu"))
    assert route["status"] == "abstained_no_target_support"
    assert all(not c for c in route["arms"].values())
    with pytest.raises(ValueError):
        allocation([])


def test_dependency_change_detected(tmp_path):
    before = dependency_hashes(ROOT)
    assert "src/ai/v3_target_gate_l6b.py" in before
    assert "src/ai/v3_target_routed_local_l6c.py" in before
    for name in before:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    path = tmp_path / "src/ai/v3_target_routed_local_l6c.py"
    path.write_text(path.read_text() + "\n# simulated dependency change\n")
    assert dependency_hashes(tmp_path) != before


def test_existing_final_results_verify():
    selection = json.loads((OLD / "selection_manifest.json").read_text())
    assert verify_complete(OLD, selection)["physics_evaluations"] == 360


@pytest.mark.parametrize("filename", ["physics_ledger.jsonl", "f_recompute.jsonl"])
def test_final_ledger_tamper_rejected(tmp_path, filename):
    for name in ("complete_manifest.json", "selection_manifest.json", "selection_ledger.jsonl", "physics_ledger.jsonl", "f_recompute.jsonl"):
        shutil.copyfile(OLD / name, tmp_path / name)
    path = tmp_path / filename
    path.write_text(path.read_text() + "{}\n")
    with pytest.raises(ValueError, match="hash changed"):
        verify_complete(tmp_path, json.loads((tmp_path / "selection_manifest.json").read_text()))


def test_summary_tamper_rejected(tmp_path):
    for name in ("complete_manifest.json", "selection_manifest.json", "selection_ledger.jsonl", "physics_ledger.jsonl", "f_recompute.jsonl"):
        shutil.copyfile(OLD / name, tmp_path / name)
    path = tmp_path / "complete_manifest.json"
    complete = json.loads(path.read_text())
    complete["arms"]["gated_top1"]["software_hit_targets_5pct"] = 0
    path.write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="summary changed"):
        verify_complete(tmp_path, json.loads((tmp_path / "selection_manifest.json").read_text()))
