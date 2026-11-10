"""Preregister and evaluate six fresh validation center targets without refitting."""

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
from src.physics.geometry22_design import Geometry22Design
from src.ai.v3_validation_integrity import dependency_hashes, verify_complete
from src.physics.v3_development_runtime import build_v3_development_physics_runtime


CONFIG = "config/experiments/geometry22_v3_fresh_validation_l6h_v1.yaml"
CODE = "src/ai/v3_fresh_validation_l6h.py"
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
    if (cfg["version"] != "geometry22_v3_fresh_validation_l6h_v1"
            or cfg["scope"] != "fresh_six_center_validation_no_refit"
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
            or cfg["holdout_labels_parsed"] is not False
            or cfg["formal_use_allowed"] is not False
            or cfg["deployment_allowed"] is not False):
        raise ValueError("L6h protocol scope changed")
    source = cfg["sources"]
    model_root = root / source["l6f_path"]
    checks = ((root / source["registry_path"], "registry_sha256"),
              (model_root / "fit_manifest.json", "fit_sha256"),
              (model_root / "gate_selection.json", "gate_selection_sha256"),
              (model_root / "local_forward_models.json", "local_forward_sha256"),
              (model_root / "normalization.json", "normalization_sha256"),
              (model_root / "center_support.json", "center_support_sha256"),
              (root / source["diagnostic_profile_path"], "diagnostic_profile_sha256"))
    if any(sha(path) != source[key] for path, key in checks):
        raise ValueError("L6h frozen source changed")
    validation_coordinates(root, cfg)
    return cfg


def source(root: Path, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    spec = cfg["sources"]
    verify_lock(root, cfg)
    targets = rows(root / cfg["output_root"] / "target_labels.jsonl")
    label_manifest = json.loads((root / cfg["output_root"] / "target_label_manifest.json").read_text())
    if label_manifest["protocol_lock_sha256"] != sha(root / cfg["output_root"] / "protocol_lock.json") or label_manifest["target_labels_sha256"] != sha(root / cfg["output_root"] / "target_labels.jsonl") or [t["target_id"] for t in targets] != [t["target_id"] for t in validation_coordinates(root, cfg)]:
        raise ValueError("Fresh validation targets changed")
    coordinates = validation_coordinates(root, cfg)
    if any(t["design"] != c["design"] or t["root_id"] != c["root_id"] or t["split"] != "validation" or t["training_allowed"] is not False for t, c in zip(targets, coordinates)):
        raise ValueError("Validation coordinate or split changed")
    targets = [t for t in targets if t["solver_converged"]]
    model_root = root / spec["l6f_path"]
    fit = json.loads((model_root / "fit_manifest.json").read_text())
    normalization = json.loads((model_root / "normalization.json").read_text())
    support = json.loads((model_root / "center_support.json").read_text())["centers"]
    local = json.loads((model_root / "local_forward_models.json").read_text())
    if (len({t["target_id"] for t in targets}) != len(targets)
            or len(support) != 55
            or fit["selected_neural_model_id"] != "gate_seed_44"
            or fit["selected_local_forward_model_id"] != local["selected_model_id"]
            or fit["new_validation_or_holdout_labels_parsed"] is not False):
        raise ValueError("L6h source population or model changed")
    return targets, normalization, support, fit, local


def freeze(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    output = root / cfg["output_root"]
    if output.exists():
        raise FileExistsError(output)
    targets = validation_coordinates(root, cfg)
    model_root = root / cfg["sources"]["l6f_path"]
    fit = json.loads((model_root / "fit_manifest.json").read_text())
    local = json.loads((model_root / "local_forward_models.json").read_text())
    runtime = build_v3_development_physics_runtime(root)
    if runtime.design_schema_hash != cfg["identity"]["design_schema_hash"] or runtime.physics_bundle_hash != cfg["identity"]["physics_bundle_hash"]:
        raise ValueError("L6h V3 runtime identity changed")
    output.mkdir(parents=True)
    lock = {
        "status": "l6h_55_center_route_selection_protocol_frozen_before_physics",
        "config_sha256": sha(root / CONFIG), "code_sha256": sha(root / CODE),
        "dependency_sha256": dependency_hashes(root),
        "target_rule": "all_six_registered_validation_center_points_no_filter",
        "total_targets": 6, "holdout_labels_parsed": False,
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
    if (lock["status"] != "l6h_55_center_route_selection_protocol_frozen_before_physics"
            or lock["config_sha256"] != sha(root / CONFIG)
            or lock["code_sha256"] != sha(root / CODE)
            or lock["dependency_sha256"] != dependency_hashes(root)
            or lock["runtime_contract"] != build_v3_development_physics_runtime(root).contract_mapping()
            or lock["l6f_fit_sha256"] != cfg["sources"]["fit_sha256"]
            or lock["source_center_id_runtime_available"] is not False
            or lock["new_validation_or_holdout_labels_parsed"] is not False
            or lock["physics_evaluations_executed"] != 0):
        raise ValueError("L6h protocol lock changed")
    return lock


def select(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    lock = verify_lock(root, cfg)
    output = root / cfg["output_root"]
    if (output / "selection_manifest.json").exists():
        raise FileExistsError(output / "selection_manifest.json")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("L6h route and score inference require physical GPU 6")
    device = torch.device("cuda:0")
    targets, normalization, support, fit, local = source(root, cfg)
    checkpoint = next(item for item in fit["neural_models"] if item["model_id"] == fit["selected_neural_model_id"])
    path = root / cfg["sources"]["l6f_path"] / checkpoint["path"]
    if sha(path) != checkpoint["sha256"]:
        raise ValueError("L6h checkpoint changed")
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
                raise ValueError("L6h Top-2 allocation requires one or two eligible centers")
            budget = allocation(choices, cfg["raw_candidates_per_target_per_arm"])
            arm_rows = []
            for center_id in choices:
                draws = _draws(public["target_id"], center_id, cfg)
                for local_slot in range(budget):
                    arm_rows.append(_candidate(public, arm, len(arm_rows), by_center[center_id], draws[local_slot], normalizer, continuous, cfg))
            if len(arm_rows) != cfg["raw_candidates_per_target_per_arm"]:
                raise ValueError("L6h raw candidate budget changed")
            _score(arm_rows, public, local, continuous, by_center, device)
            raw.extend(arm_rows)
        print(f"L6h routed {public['target_id'][:12]}: {route['status']}", flush=True)
    selected = _selection(raw)
    routable = sum(item["status"] == "routable" for item in route_rows)
    if len(route_rows) != len(targets) or len(raw) != routable * len(ARMS) * 256 or len(selected) != routable * len(ARMS) * 8:
        raise ValueError("L6h fixed candidate or selected budget changed")
    write_new(output / "route_ledger.jsonl", b"".join(dump(item) for item in route_rows))
    write_new(output / "raw_candidates.jsonl", b"".join(dump(item) for item in raw))
    write_new(output / "selection_ledger.jsonl", b"".join(dump(item) for item in selected))
    manifest = {
        "status": "l6h_55_center_selection_sealed_before_physics",
        "protocol_lock_sha256": sha(output / "protocol_lock.json"),
        "route_ledger_sha256": sha(output / "route_ledger.jsonl"),
        "raw_candidates_sha256": sha(output / "raw_candidates.jsonl"),
        "selection_ledger_sha256": sha(output / "selection_ledger.jsonl"),
        "selected_gate_checkpoint_sha256": checkpoint["sha256"],
        "selected_local_model_id": local["selected_model_id"],
        "targets": 6, "labeled_converged_targets": len(targets), "routable_targets": routable, "abstained_targets": len(targets) - routable,
        "target_labels_sha256": sha(output / "target_labels.jsonl"), "holdout_labels_parsed": False,
        "raw_slots": len(raw), "geometry_valid_slots": sum(item["status"] == "geometric_candidate" for item in raw),
        "selected_slots": len(selected), "raw_status_counts": dict(Counter(item["status"] for item in raw)),
        "source_center_id_runtime_available": False,
        "new_validation_labels_parsed": True,
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
    if (manifest["status"] != "l6h_55_center_selection_sealed_before_physics"
            or any(manifest[key] != sha(output / filename) for key, filename in (
                ("route_ledger_sha256", "route_ledger.jsonl"),
                ("raw_candidates_sha256", "raw_candidates.jsonl"),
                ("selection_ledger_sha256", "selection_ledger.jsonl")))
            or manifest["targets"] != 6 or manifest["labeled_converged_targets"] != len(route_rows)
            or manifest["raw_slots"] != len(raw) or len(raw) != manifest["routable_targets"] * 768
            or manifest["selected_slots"] != len(selected) or len(selected) != manifest["routable_targets"] * 24
            or manifest["target_labels_sha256"] != sha(output / "target_labels.jsonl")
            or manifest["protocol_lock_sha256"] != sha(output / "protocol_lock.json")
            or manifest["physics_evaluations_executed"] != 0
            or any("source_center_id" in item for item in route_rows)
            or selected != _selection(raw)):
        raise ValueError("L6h sealed candidate selection changed")
    targets, _, _, fit, _ = source(root, cfg)
    checkpoint = next(t for t in fit["neural_models"] if t["model_id"] == fit["selected_neural_model_id"])
    if sha(root / cfg["sources"]["l6f_path"] / checkpoint["path"]) != manifest["selected_gate_checkpoint_sha256"] or manifest["selected_gate_checkpoint_sha256"] != checkpoint["sha256"]:
        raise ValueError("Selected checkpoint changed")
    if {t["target_id"] for t in route_rows} != {t["target_id"] for t in targets} or manifest["routable_targets"] != sum(t["status"] == "routable" for t in route_rows) or manifest["abstained_targets"] != sum(t["status"] == "abstained_no_target_support" for t in route_rows):
        raise ValueError("Routing population changed")
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
                       "independent_v3_recompute_equal": repeated == best, "recomputed_result": repeated,
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
    complete = {"status": "l6h_55_center_fresh_validation_software_diagnostic_complete",
                "selection_manifest_sha256": sha(output / "selection_manifest.json"),
                "physics_ledger_sha256": sha(output / "physics_ledger.jsonl"),
                "f_recompute_sha256": sha(output / "f_recompute.jsonl"),
                "targets": 6, "routable_targets": selection["routable_targets"],
                "abstained_targets": selection["abstained_targets"],
                "target_label_failures": 6 - selection["labeled_converged_targets"], "raw_slots": selection["raw_slots"],
                "physics_evaluations": len(physics), "f_recomputations": len(f_rows),
                "arms": arms, "old_validation_reused_for_diagnostic": False,
                "model_refitted": False, "holdout_labels_parsed": False,
                "new_validation_labels_parsed": True,
                "hit_tolerance_relative_each_output": .05,
                "formal_use_allowed": False, "deployment_allowed": False}
    write_new(output / "complete_manifest.json", complete)
    return complete


def allocation(choices: list[str], total: int = 256) -> int:
    if len(choices) not in (1, 2):
        raise ValueError("Route must contain one or two eligible centers")
    return total // len(choices)


def validation_coordinates(root: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    registry = rows(root / cfg["sources"]["registry_path"])
    if Counter(t["split"] for t in registry) != {"development": 32, "validation": 6, "holdout": 7} or len({t["root_id"] for t in registry}) != 45 or len({t["design_id"] for t in registry}) != 45:
        raise ValueError("Registry split leakage or population changed")
    selected = [t for t in registry if t["split"] == "validation"]
    return [{"target_id": t["design_id"], "design": t["design"], "root_id": t["root_id"]} for t in selected]


def label_targets(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    verify_lock(root, cfg)
    output = root / cfg["output_root"]
    if (output / "target_labels.jsonl").exists():
        raise FileExistsError(output / "target_labels.jsonl")
    runtime = build_v3_development_physics_runtime(root)
    labeled = []
    for item in validation_coordinates(root, cfg):
        result = runtime.evaluate(Geometry22Design.from_mapping(item["design"]))
        engine = result.engine_result
        labeled.append({**item, "split": "validation", "solver_converged": bool(engine.solver.converged),
                        "failure_code": engine.solver.failure_code.value if engine.solver.failure_code else None,
                        "target_performance": {name: float(engine.performance[name]) for name in ("thrust_N", "tsfc_kg_N_s", "mass_kg")} if engine.solver.converged else None,
                        "runtime_contract": runtime.contract_mapping(), "training_allowed": False,
                        "formal_use_allowed": False, "deployment_allowed": False})
    write_new(output / "target_labels.jsonl", b"".join(dump(t) for t in labeled))
    manifest = {"target_labels_sha256": sha(output / "target_labels.jsonl"),
                "protocol_lock_sha256": sha(output / "protocol_lock.json"), "targets": 6,
                "solver_converged": sum(t["solver_converged"] for t in labeled),
                "holdout_labels_parsed": False, "model_refitted": False}
    write_new(output / "target_label_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "label", "select", "evaluate", "verify"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cfg = config_at(root)
    result = (freeze(root, cfg) if args.command == "freeze" else
              label_targets(root, cfg) if args.command == "label" else
              select(root, cfg) if args.command == "select" else
              evaluate(root, cfg) if args.command == "evaluate" else
              verify_complete(root / cfg["output_root"], verify_selection(root, cfg)))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
