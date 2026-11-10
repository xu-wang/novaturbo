"""Run the frozen L6f 55-center route through local V3 software diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml

from src.ai.v3_extended_gate_l6f import network
from src.ai.v3_target_routed_local_l6c import (
    _candidate, _draws, _physics_result, _route, _runtime_target, _score, _selection,
)
from src.physics.constraint_profiles import ConstraintProfile
from src.physics.geometry22_data import Geometry22Normalizer
from src.physics.v3_development_runtime import build_v3_development_physics_runtime


CONFIG = "config/experiments/geometry22_v3_extended_route_l6g_v2.yaml"
CODE = "src/ai/v3_extended_route_l6g_v2.py"
ARMS = ("gated_top1", "gated_top2", "nearest_center_without_gate")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write_new(path: Path, content: Any) -> None:
    with path.open("xb") as stream:
        stream.write(content if isinstance(content, bytes) else dump(content))


def config_at(root: Path) -> dict[str, Any]:
    cfg = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if (cfg["version"] != "geometry22_v3_extended_route_l6g_v2"
            or cfg["scope"] != "old_validation_15_target_equal_budget_55_center_software_diagnostic"
            or cfg["runtime_target_fields"] != ["target_id", "target_performance"]
            or cfg["source_center_id_runtime_available"] is not False
            or cfg["arms"] != list(ARMS)
            or cfg["route_policy"] != "mask_to_l6f_eligible_centers_then_rank_else_abstain"
            or cfg["raw_candidates_per_target_per_arm"] != 256
            or cfg["physics_evaluations_per_target_per_arm"] != 8
            or cfg["top2_allocation"] != "equal_128_each_if_two_else_256_single"
            or cfg["local_generation"]["radial_bands"] != [[0.0, 0.02], [0.02, 0.04], [0.04, 0.06], [0.06, 0.08]]
            or cfg["local_generation"]["sobol_power"] != 8
            or cfg["local_generation"]["invalid_policy"] != "retain_raw_slot_no_resample"
            or cfg["scoring"]["model"] != "l6f_selected_center_local_ridge"
            or cfg["selected_ledger_sealed_before_physics"] is not True
            or cfg["gate_inference_device"] != "cuda:0"
            or cfg["physical_gpu_index"] != 6
            or cfg["new_validation_or_holdout_labels_parsed"] is not False
            or cfg["formal_use_allowed"] is not False
            or cfg["deployment_allowed"] is not False):
        raise ValueError("L6g protocol scope changed")
    source = cfg["sources"]
    model_root = root / source["l6f_path"]
    checks = ((root / source["target_path"], "target_sha256"),
              (model_root / "fit_manifest.json", "fit_sha256"),
              (model_root / "gate_selection.json", "gate_selection_sha256"),
              (model_root / "local_forward_models.json", "local_forward_sha256"),
              (model_root / "normalization.json", "normalization_sha256"),
              (model_root / "center_support.json", "center_support_sha256"),
              (root / source["diagnostic_profile_path"], "diagnostic_profile_sha256"))
    if any(sha(path) != source[key] for path, key in checks):
        raise ValueError("L6g frozen source changed")
    return cfg


def source(root: Path, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    spec = cfg["sources"]
    targets = rows(root / spec["target_path"])
    model_root = root / spec["l6f_path"]
    fit = json.loads((model_root / "fit_manifest.json").read_text())
    normalization = json.loads((model_root / "normalization.json").read_text())
    support = json.loads((model_root / "center_support.json").read_text())["centers"]
    local = json.loads((model_root / "local_forward_models.json").read_text())
    if (len(targets) != 15 or len({t["target_id"] for t in targets}) != 15
            or len(support) != 55
            or fit["selected_neural_model_id"] != "gate_seed_44"
            or fit["selected_local_forward_model_id"] != local["selected_model_id"]
            or fit["new_validation_or_holdout_labels_parsed"] is not False):
        raise ValueError("L6g source population or model changed")
    return targets, normalization, support, fit, local


def freeze(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    output = root / cfg["output_root"]
    if output.exists():
        raise FileExistsError(output)
    targets, _, _, fit, local = source(root, cfg)
    runtime = build_v3_development_physics_runtime(root)
    if runtime.design_schema_hash != cfg["identity"]["design_schema_hash"] or runtime.physics_bundle_hash != cfg["identity"]["physics_bundle_hash"]:
        raise ValueError("L6g V3 runtime identity changed")
    output.mkdir(parents=True)
    lock = {
        "status": "l6g_55_center_route_selection_protocol_frozen_before_physics",
        "config_sha256": sha(root / CONFIG), "code_sha256": sha(root / CODE),
        "l6f_fit_sha256": cfg["sources"]["fit_sha256"],
        "selected_gate_model_id": fit["selected_neural_model_id"],
        "selected_local_model_id": local["selected_model_id"],
        "target_ids": [item["target_id"] for item in targets],
        "runtime_contract": runtime.contract_mapping(),
        "source_center_id_runtime_available": False,
        "new_validation_or_holdout_labels_parsed": False,
        "physics_evaluations_executed": 0,
        "formal_use_allowed": False, "deployment_allowed": False,
    }
    write_new(output / "protocol_lock.json", lock)
    return lock


def verify_lock(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    output = root / cfg["output_root"]
    lock = json.loads((output / "protocol_lock.json").read_text())
    if (lock["status"] != "l6g_55_center_route_selection_protocol_frozen_before_physics"
            or lock["config_sha256"] != sha(root / CONFIG)
            or lock["code_sha256"] != sha(root / CODE)
            or lock["l6f_fit_sha256"] != cfg["sources"]["fit_sha256"]
            or lock["source_center_id_runtime_available"] is not False
            or lock["new_validation_or_holdout_labels_parsed"] is not False
            or lock["physics_evaluations_executed"] != 0):
        raise ValueError("L6g protocol lock changed")
    return lock


def select(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    lock = verify_lock(root, cfg)
    output = root / cfg["output_root"]
    if (output / "selection_manifest.json").exists():
        raise FileExistsError(output / "selection_manifest.json")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("L6g route and score inference require physical GPU 6")
    device = torch.device("cuda:0")
    targets, normalization, support, fit, local = source(root, cfg)
    checkpoint = next(item for item in fit["neural_models"] if item["model_id"] == fit["selected_neural_model_id"])
    path = root / cfg["sources"]["l6f_path"] / checkpoint["path"]
    if sha(path) != checkpoint["sha256"]:
        raise ValueError("L6g checkpoint changed")
    model = network().to(device)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
    model.eval()
    normalizer = Geometry22Normalizer()
    continuous = local["input_continuous_indices_zero_based"]
    by_center = {item["center_id"]: item for item in support}
    route_rows, raw = [], []
    for target in targets:
        public = _runtime_target(target)
        route = _route(public, normalization, support, model, device)
        route_rows.append(route)
        for arm in ARMS:
            choices = route["arms"][arm]
            if not choices:
                continue
            if arm == "gated_top2" and len(choices) not in (1, 2):
                raise ValueError("L6g Top-2 allocation requires one or two eligible centers")
            budget = cfg["raw_candidates_per_target_per_arm"] // len(choices)
            arm_rows = []
            for center_id in choices:
                draws = _draws(public["target_id"], center_id, cfg)
                for local_slot in range(budget):
                    arm_rows.append(_candidate(public, arm, len(arm_rows), by_center[center_id], draws[local_slot], normalizer, continuous, cfg))
            if len(arm_rows) != cfg["raw_candidates_per_target_per_arm"]:
                raise ValueError("L6g raw candidate budget changed")
            _score(arm_rows, public, local, continuous, by_center, device)
            raw.extend(arm_rows)
        print(f"L6g routed {public['target_id'][:12]}: {route['status']}", flush=True)
    selected = _selection(raw)
    if len(route_rows) != 15 or sum(item["status"] == "routable" for item in route_rows) != 15:
        raise ValueError("L6g 15-target support count changed")
    if len(raw) != 15 * len(ARMS) * 256 or len(selected) != 15 * len(ARMS) * 8:
        raise ValueError("L6g fixed candidate or selected budget changed")
    write_new(output / "route_ledger.jsonl", b"".join(dump(item) for item in route_rows))
    write_new(output / "raw_candidates.jsonl", b"".join(dump(item) for item in raw))
    write_new(output / "selection_ledger.jsonl", b"".join(dump(item) for item in selected))
    manifest = {
        "status": "l6g_55_center_selection_sealed_before_physics",
        "protocol_lock_sha256": sha(output / "protocol_lock.json"),
        "route_ledger_sha256": sha(output / "route_ledger.jsonl"),
        "raw_candidates_sha256": sha(output / "raw_candidates.jsonl"),
        "selection_ledger_sha256": sha(output / "selection_ledger.jsonl"),
        "selected_gate_checkpoint_sha256": checkpoint["sha256"],
        "selected_local_model_id": local["selected_model_id"],
        "targets": 15, "routable_targets": 15, "abstained_targets": 0,
        "raw_slots": len(raw), "geometry_valid_slots": sum(item["status"] == "geometric_candidate" for item in raw),
        "selected_slots": len(selected), "raw_status_counts": dict(Counter(item["status"] for item in raw)),
        "source_center_id_runtime_available": False,
        "new_validation_or_holdout_labels_parsed": False,
        "physics_evaluations_executed": 0,
        "formal_use_allowed": False, "deployment_allowed": False,
    }
    write_new(output / "selection_manifest.json", manifest)
    return manifest


def verify_selection(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    output = root / cfg["output_root"]
    verify_lock(root, cfg)
    manifest = json.loads((output / "selection_manifest.json").read_text())
    route_rows, raw, selected = (rows(output / filename) for filename in ("route_ledger.jsonl", "raw_candidates.jsonl", "selection_ledger.jsonl"))
    if (manifest["status"] != "l6g_55_center_selection_sealed_before_physics"
            or any(manifest[key] != sha(output / filename) for key, filename in (
                ("route_ledger_sha256", "route_ledger.jsonl"),
                ("raw_candidates_sha256", "raw_candidates.jsonl"),
                ("selection_ledger_sha256", "selection_ledger.jsonl")))
            or manifest["targets"] != len(route_rows) or len(route_rows) != 15
            or manifest["raw_slots"] != len(raw) or len(raw) != 11520
            or manifest["selected_slots"] != len(selected) or len(selected) != 360
            or manifest["physics_evaluations_executed"] != 0
            or any("source_center_id" in item for item in route_rows)
            or selected != _selection(raw)):
        raise ValueError("L6g sealed candidate selection changed")
    return manifest


def evaluate(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    selection = verify_selection(root, cfg)
    output = root / cfg["output_root"]
    if (output / "complete_manifest.json").exists():
        raise FileExistsError(output / "complete_manifest.json")
    targets, _, _, _, _ = source(root, cfg)
    public = {item["target_id"]: _runtime_target(item) for item in targets}
    selected = rows(output / "selection_ledger.jsonl")
    runtime = build_v3_development_physics_runtime(root)
    profile = ConstraintProfile.from_yaml(root / cfg["sources"]["diagnostic_profile_path"])
    physics = [_physics_result(runtime, profile, item, public[item["target_id"]]) for item in selected]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in physics:
        grouped[(item["target_id"], item["arm"])].append(item)
    selected_by_id = {(item["target_id"], item["arm"], item["design_id"]): item for item in selected}
    f_rows = []
    for (target_id, arm), group in sorted(grouped.items()):
        valid = [item for item in group if item["actual_rho_5pct"] is not None]
        if not valid:
            continue
        best = min(valid, key=lambda item: (item["actual_rho_5pct"], item["design_id"]))
        repeated = _physics_result(runtime, profile, selected_by_id[(target_id, arm, best["design_id"])], public[target_id])
        f_rows.append({"target_id": target_id, "arm": arm, "design_id": best["design_id"],
                       "independent_v3_recompute_equal": repeated == best,
                       "original_rho_5pct": best["actual_rho_5pct"],
                       "recomputed_rho_5pct": repeated["actual_rho_5pct"],
                       "formal_use_allowed": False, "deployment_allowed": False})
    write_new(output / "physics_ledger.jsonl", b"".join(dump(item) for item in physics))
    write_new(output / "f_recompute.jsonl", b"".join(dump(item) for item in f_rows))
    arms = {}
    for arm in ARMS:
        group = [item for item in physics if item["arm"] == arm]
        hits = {item["target_id"] for item in group if item["software_hit_5pct"]}
        arms[arm] = {"selected_candidates": len(group),
                     "targets_evaluated": len({item["target_id"] for item in group}),
                     "solver_converged": sum(item["solver_converged"] for item in group),
                     "diagnostic_domain_valid": sum(item["diagnostic_domain_valid"] for item in group),
                     "engine_feasible": sum(item["engine_feasible"] for item in group),
                     "software_hit_targets_5pct": len(hits),
                     "software_hit_target_ids_5pct": sorted(hits),
                     "f_recompute_count": sum(item["arm"] == arm for item in f_rows),
                     "f_recompute_equal_count": sum(item["arm"] == arm and item["independent_v3_recompute_equal"] for item in f_rows)}
    complete = {"status": "l6g_55_center_old_validation_software_diagnostic_complete",
                "selection_manifest_sha256": sha(output / "selection_manifest.json"),
                "physics_ledger_sha256": sha(output / "physics_ledger.jsonl"),
                "f_recompute_sha256": sha(output / "f_recompute.jsonl"),
                "targets": 15, "routable_targets": 15, "raw_slots": selection["raw_slots"],
                "physics_evaluations": len(physics), "f_recomputations": len(f_rows),
                "arms": arms, "old_validation_reused_for_diagnostic": True,
                "new_validation_or_holdout_labels_parsed": False,
                "hit_tolerance_relative_each_output": .05,
                "formal_use_allowed": False, "deployment_allowed": False}
    write_new(output / "complete_manifest.json", complete)
    return complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "select", "evaluate", "verify"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cfg = config_at(root)
    result = (freeze(root, cfg) if args.command == "freeze" else
              select(root, cfg) if args.command == "select" else
              evaluate(root, cfg) if args.command == "evaluate" else
              verify_selection(root, cfg))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
