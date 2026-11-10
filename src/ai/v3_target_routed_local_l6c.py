"""Route frozen V3 targets, rank new local designs, then evaluate them with V3."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
import torch
import yaml

from src.ai.geometry19_performance_gated_coordinates import hard_failure
from src.ai.v3_target_gate_l6b import _network, verify_v3_target_gate_fit
from src.ai.v3_target_route_protocol import verify_v3_target_route_freeze
from src.physics.constraint_profiles import ConstraintProfile, assess_constraint_profile
from src.physics.geometry22_data import Geometry22Normalizer
from src.physics.geometry22_design import Geometry22Design
from src.physics.v3_development_runtime import build_v3_development_physics_runtime


CONFIG = "config/experiments/geometry22_v3_target_routed_local_l6c_v1.yaml"
CODE = "src/ai/v3_target_routed_local_l6c.py"
NAMES = ("thrust_N", "tsfc_kg_N_s", "mass_kg")
ARMS = ("gated_top1", "gated_top2", "nearest_center_without_gate")


class V3RoutedLocalError(ValueError):
    """An L6c source, candidate selection, or V3 physics result is invalid."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_new(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(value if isinstance(value, bytes) else _bytes(value))


def _config(root: Path) -> dict[str, Any]:
    cfg = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if (
        cfg.get("version") != "geometry22_v3_target_routed_local_l6c_v1"
        or cfg.get("scope") != "old_validation_covered_targets_software_diagnostic_no_model_selection"
        or cfg.get("target_population") != "old_l6a_selected_15_targets_diagnostic_only"
        or cfg.get("runtime_target_fields") != ["target_id", "target_performance"]
        or cfg.get("source_center_id_runtime_available") is not False
        or cfg.get("arms") != list(ARMS)
        or cfg.get("route_policy") != "mask_to_l6a_eligible_centers_then_rank_else_abstain"
        or cfg.get("raw_candidates_per_target_per_arm") != 256
        or cfg.get("physics_evaluations_per_target_per_arm") != 8
        or cfg.get("top2_allocation") != "equal_128_each"
        or cfg.get("local_generation", {}).get("radial_bands") != [[0.0, 0.02], [0.02, 0.04], [0.04, 0.06], [0.06, 0.08]]
        or cfg["local_generation"].get("sobol_dimension") != 20
        or cfg["local_generation"].get("sobol_power") != 8
        or cfg["local_generation"].get("integer_policy") != "fixed_from_routed_center"
        or cfg["local_generation"].get("invalid_policy") != "retain_raw_slot_no_resample"
        or cfg.get("scoring", {}).get("model") != "l6b_selected_center_local_ridge"
        or cfg["scoring"].get("unique_design_id_before_physics") is not True
        or cfg.get("software_hit_relative_tolerance_each_output") != 0.05
        or cfg.get("selected_ledger_sealed_before_physics") is not True
        or cfg.get("gate_inference_device") != "cuda:0"
        or cfg.get("physical_gpu_index") != 6
        or cfg.get("formal_use_allowed") is not False
        or cfg.get("deployment_allowed") is not False
    ):
        raise V3RoutedLocalError("L6c protocol changed scope")
    return cfg


def _source(root: Path, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    spec = cfg["sources"]
    l6a, l6b = root / spec["l6a_path"], root / spec["l6b_path"]
    checks = ((l6a / "complete_manifest.json", "l6a_complete_sha256"),
              (l6a / "cross_center_targets.jsonl", "targets_sha256"),
              (l6b / "fit_manifest.json", "l6b_fit_sha256"),
              (l6b / "gate_selection.json", "gate_selection_sha256"),
              (l6b / "local_forward_models.json", "local_forward_sha256"),
              (root / spec["diagnostic_profile_path"], "diagnostic_profile_sha256"))
    if any(_sha(path) != spec[key] for path, key in checks):
        raise V3RoutedLocalError("L6c source artifact changed")
    lock = verify_v3_target_route_freeze(root, l6a)
    fit = verify_v3_target_gate_fit(root, l6b)
    if (lock["design_schema_hash"] != cfg["identity"]["design_schema_hash"]
            or lock["physics_bundle_hash"] != cfg["identity"]["physics_bundle_hash"]
            or fit["selected_neural_model_id"] != "gate_seed_44"):
        raise V3RoutedLocalError("L6c route/physics/model identity changed")
    targets = _rows(l6a / "cross_center_targets.jsonl")
    if len(targets) != 15 or len({t["target_id"] for t in targets}) != 15:
        raise V3RoutedLocalError("L6c target population changed")
    normalization = json.loads((l6a / "normalization.json").read_text())
    support = json.loads((l6a / "center_support.json").read_text())["centers"]
    local = json.loads((l6b / "local_forward_models.json").read_text())
    if local["selected_model_id"] != fit["selected_local_forward_model_id"] or len(support) != 23:
        raise V3RoutedLocalError("L6c local scorer identity changed")
    return targets, normalization, support, fit, local


def _device() -> torch.device:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise V3RoutedLocalError("L6c gate and Ridge inference require physical GPU 6")
    return torch.device("cuda:0")


def _runtime_target(target: dict[str, Any]) -> dict[str, Any]:
    public = {key: target[key] for key in ("target_id", "target_performance")}
    values = [public["target_performance"][name] for name in NAMES]
    if len(public) != 2 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise V3RoutedLocalError("L6c runtime target is invalid")
    return public


def _route(public: dict[str, Any], normalization: dict[str, Any], support: list[dict[str, Any]],
           model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    values = np.asarray([public["target_performance"][name] for name in NAMES], dtype=np.float64)
    condition = (np.log(values) - np.asarray(normalization["log_mean"])) / np.asarray(normalization["log_std"])
    with torch.no_grad():
        logits = model(torch.as_tensor(condition[None, :], dtype=torch.float32, device=device))[0].cpu().numpy()
    centroids = np.asarray([s["target_centroid_normalized_log"] for s in support])
    radii = np.asarray([s["target_radius_p95_normalized_log"] for s in support])
    distances = np.linalg.norm(centroids - condition[None, :], axis=1) / radii
    eligible = [i for i, distance in enumerate(distances) if distance <= 1.0]
    gate_order = sorted(eligible, key=lambda i: (-float(logits[i]), support[i]["center_id"]))
    nearest_order = sorted(eligible, key=lambda i: (float(distances[i]), support[i]["center_id"]))
    return {"target_id": public["target_id"], "status": "routable" if eligible else "abstained_no_target_support",
            "eligible_center_count": len(eligible),
            "arms": {
                "gated_top1": [support[i]["center_id"] for i in gate_order[:1]],
                "gated_top2": [support[i]["center_id"] for i in gate_order[:2]],
                "nearest_center_without_gate": [support[i]["center_id"] for i in nearest_order[:1]],
            }}


def _draws(target_id: str, center_id: str, cfg: dict[str, Any]) -> np.ndarray:
    salt = cfg["local_generation"]["seed_salt"]
    seed = int(hashlib.sha256(f"{salt}:{target_id}:{center_id}".encode()).hexdigest()[:8], 16)
    return qmc.Sobol(d=20, scramble=True, seed=seed).random_base2(m=8)


def _candidate(public: dict[str, Any], arm: str, slot: int, center: dict[str, Any], draw: np.ndarray,
               normalizer: Geometry22Normalizer, continuous: list[int], cfg: dict[str, Any]) -> dict[str, Any]:
    center_id = center["center_id"]
    row = {"target_id": public["target_id"], "arm": arm, "raw_slot": slot,
           "route_center_id": center_id, "status": None}
    direction = ndtri(np.clip(draw[:19], 1e-12, 1 - 1e-12))
    direction /= np.linalg.norm(direction)
    lower, upper = cfg["local_generation"]["radial_bands"][slot % 4]
    radius = float(lower + (upper - lower) * draw[19])
    normalized = np.asarray(center["center_normalized_design"], dtype=np.float64).copy()
    normalized[continuous] += radius * direction
    row["radius_band"] = slot % 4
    row["normalized_radius"] = radius
    if np.any(normalized < -1) or np.any(normalized > 1):
        row["status"] = "outside_schema"
        return row
    try:
        design = normalizer.inverse(normalized.tolist())
    except ValueError:
        row["status"] = "invalid_22d_design"
        return row
    if any(design.as_mapping()[name] != value for name, value in center["integer_state"].items()):
        raise V3RoutedLocalError("L6c routed integer state changed")
    actual = math.dist([normalizer.transform(design)[i] for i in continuous],
                       [center["center_normalized_design"][i] for i in continuous])
    if actual > 0.08 + 1e-10 or not math.isclose(actual, radius, rel_tol=1e-8, abs_tol=1e-10):
        raise V3RoutedLocalError("L6c local support or serialized radius changed")
    failure = hard_failure(design.as_mapping(), list(normalizer.fields), 0.2)
    if failure is not None:
        row["status"] = f"geometry_rejected:{failure}"
        return row
    row.update(status="geometric_candidate", design_id=design.design_id(), design=design.as_mapping(),
               normalized_design=list(normalizer.transform(design)),
               design_schema_hash=cfg["identity"]["design_schema_hash"],
               physics_bundle_hash=cfg["identity"]["physics_bundle_hash"])
    return row


def _score(rows: list[dict[str, Any]], public: dict[str, Any], local: dict[str, Any],
           continuous: list[int], support: dict[str, dict[str, Any]], device: torch.device) -> None:
    model_id = local["selected_model_id"]
    valid = [row for row in rows if row["status"] == "geometric_candidate"]
    if not valid:
        return
    by_center: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_center[row["route_center_id"]].append(row)
    target = np.asarray([public["target_performance"][name] for name in NAMES])
    for center_id, group in by_center.items():
        asset = local["models"][model_id][center_id]
        anchor = np.asarray(support[center_id]["center_normalized_design"])[continuous]
        x = np.asarray([np.asarray(row["normalized_design"])[continuous] for row in group])
        with torch.no_grad():
            tensor = torch.as_tensor((x - anchor) / 0.08, dtype=torch.float64, device=device)
            coef = torch.as_tensor(asset["coefficients"], dtype=torch.float64, device=device)
            intercept = torch.as_tensor(asset["intercept_log"], dtype=torch.float64, device=device)
            prediction = torch.exp(tensor @ coef + intercept).cpu().numpy()
        for row, values in zip(group, prediction, strict=True):
            row["predicted_performance"] = {name: float(value) for name, value in zip(NAMES, values, strict=True)}
            row["predicted_rho"] = float(np.max(np.abs(values / target - 1.0) / 0.05))


def _selection(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        if row["status"] == "geometric_candidate":
            groups[(row["target_id"], row["arm"])].append(row)
    selected = []
    for (target_id, arm), rows in sorted(groups.items()):
        seen = set()
        for row in sorted(rows, key=lambda item: (item["predicted_rho"], item["design_id"], item["raw_slot"])):
            if row["design_id"] in seen:
                continue
            seen.add(row["design_id"])
            selected.append({"target_id": target_id, "arm": arm, "selection_rank": len([s for s in selected if s["target_id"] == target_id and s["arm"] == arm]) + 1,
                             "raw_slot": row["raw_slot"], "route_center_id": row["route_center_id"],
                             "design_id": row["design_id"], "design": row["design"],
                             "normalized_design": row["normalized_design"],
                             "predicted_performance": row["predicted_performance"],
                             "predicted_rho": row["predicted_rho"],
                             "design_schema_hash": row["design_schema_hash"],
                             "physics_bundle_hash": row["physics_bundle_hash"]})
            if sum(s["target_id"] == target_id and s["arm"] == arm for s in selected) == 8:
                break
    return selected


def freeze_v3_routed_local(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    cfg = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / cfg["output_root"]
    if output.exists():
        raise V3RoutedLocalError("L6c output exists; refusing to overwrite")
    targets, _, _, fit, local = _source(root, cfg)
    runtime = build_v3_development_physics_runtime(root)
    if (runtime.design_schema_hash != cfg["identity"]["design_schema_hash"]
            or runtime.physics_bundle_hash != cfg["identity"]["physics_bundle_hash"]):
        raise V3RoutedLocalError("L6c V3 runtime identity changed")
    lock = {"status": "v3_l6c_route_generation_selection_protocol_frozen_before_physics",
            "config_sha256": _sha(root / CONFIG), "code_sha256": _sha(root / CODE),
            "l6a_complete_sha256": cfg["sources"]["l6a_complete_sha256"],
            "l6b_fit_sha256": cfg["sources"]["l6b_fit_sha256"],
            "selected_gate_model_id": fit["selected_neural_model_id"],
            "selected_local_model_id": local["selected_model_id"],
            "target_ids": [t["target_id"] for t in targets],
            "runtime_contract": runtime.contract_mapping(),
            "source_center_id_runtime_available": False, "selection_before_physics": True,
            "physics_evaluations_executed": 0, "formal_use_allowed": False, "deployment_allowed": False}
    output.mkdir(parents=True)
    _write_new(output / "protocol_lock.json", lock)
    return verify_v3_routed_local_freeze(root, output)


def verify_v3_routed_local_freeze(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root, output = Path(repository_root).resolve(), Path(output_root).resolve()
    cfg = _config(root)
    targets, _, _, fit, local = _source(root, cfg)
    runtime = build_v3_development_physics_runtime(root)
    lock = json.loads((output / "protocol_lock.json").read_text())
    if (lock["status"] != "v3_l6c_route_generation_selection_protocol_frozen_before_physics"
            or lock["config_sha256"] != _sha(root / CONFIG)
            or lock["code_sha256"] != _sha(root / CODE)
            or lock["l6a_complete_sha256"] != cfg["sources"]["l6a_complete_sha256"]
            or lock["l6b_fit_sha256"] != cfg["sources"]["l6b_fit_sha256"]
            or lock["selected_gate_model_id"] != fit["selected_neural_model_id"]
            or lock["selected_local_model_id"] != local["selected_model_id"]
            or lock["target_ids"] != [t["target_id"] for t in targets]
            or lock["runtime_contract"] != runtime.contract_mapping()
            or lock["source_center_id_runtime_available"] is not False
            or lock["selection_before_physics"] is not True
            or lock["physics_evaluations_executed"] != 0
            or lock["formal_use_allowed"] is not False or lock["deployment_allowed"] is not False):
        raise V3RoutedLocalError("L6c protocol lock changed")
    return lock


def prepare_v3_routed_local(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    cfg = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / cfg["output_root"]
    verify_v3_routed_local_freeze(root, output)
    if any((output / name).exists() for name in ("route_ledger.jsonl", "raw_candidates.jsonl", "selection_ledger.jsonl", "selection_manifest.json")):
        raise V3RoutedLocalError("L6c selection already exists; refusing to overwrite")
    targets, normalization, support, fit, local = _source(root, cfg)
    device = _device()
    checkpoint = next(item for item in fit["neural_models"] if item["model_id"] == fit["selected_neural_model_id"])
    payload = torch.load(root / cfg["sources"]["l6b_path"] / checkpoint["path"], map_location="cpu", weights_only=True)
    model = _network().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    normalizer = Geometry22Normalizer()
    continuous = local["input_continuous_indices_zero_based"]
    centers = {item["center_id"]: item for item in support}
    routes, raw = [], []
    for target in targets:
        public = _runtime_target(target)
        route = _route(public, normalization, support, model, device)
        routes.append(route)
        for arm in ARMS:
            choices = route["arms"][arm]
            if not choices:
                continue
            count = 256 // len(choices)
            arm_rows = []
            for center_id in choices:
                draws = _draws(public["target_id"], center_id, cfg)
                for local_slot in range(count):
                    slot = len(arm_rows)
                    arm_rows.append(_candidate(public, arm, slot, centers[center_id], draws[local_slot], normalizer, continuous, cfg))
            if len(arm_rows) != 256:
                raise V3RoutedLocalError("L6c raw candidate budget changed")
            _score(arm_rows, public, local, continuous, centers, device)
            raw.extend(arm_rows)
        print(f"V3 L6c target routed: {public['target_id'][:12]} {route['status']}", flush=True)
    selected = _selection(raw)
    if len(routes) != 15 or sum(row["status"] == "routable" for row in routes) != 9:
        raise V3RoutedLocalError("L6c frozen target support count changed")
    if len(raw) != 9 * len(ARMS) * 256:
        raise V3RoutedLocalError("L6c raw slot count changed")
    _write_new(output / "route_ledger.jsonl", b"".join(_bytes(row) for row in routes))
    _write_new(output / "raw_candidates.jsonl", b"".join(_bytes(row) for row in raw))
    _write_new(output / "selection_ledger.jsonl", b"".join(_bytes(row) for row in selected))
    manifest = {"status": "v3_l6c_candidate_selection_sealed_before_physics",
                "protocol_lock_sha256": _sha(output / "protocol_lock.json"),
                "route_ledger_sha256": _sha(output / "route_ledger.jsonl"),
                "raw_candidates_sha256": _sha(output / "raw_candidates.jsonl"),
                "selection_ledger_sha256": _sha(output / "selection_ledger.jsonl"),
                "selected_gate_checkpoint_sha256": checkpoint["sha256"],
                "selected_local_model_id": local["selected_model_id"],
                "targets": len(routes), "routable_targets": 9, "abstained_targets": 6,
                "raw_slots": len(raw), "geometry_valid_slots": sum(row["status"] == "geometric_candidate" for row in raw),
                "selected_slots": len(selected),
                "raw_status_counts": dict(sorted(Counter(row["status"] for row in raw).items())),
                "physics_evaluations_executed": 0, "source_center_id_runtime_available": False,
                "formal_use_allowed": False, "deployment_allowed": False}
    _write_new(output / "selection_manifest.json", manifest)
    return verify_v3_routed_local_selection(root, output)


def verify_v3_routed_local_selection(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root, output = Path(repository_root).resolve(), Path(output_root).resolve()
    verify_v3_routed_local_freeze(root, output)
    manifest = json.loads((output / "selection_manifest.json").read_text())
    routes, raw, selected = (_rows(output / filename) for filename in (
        "route_ledger.jsonl", "raw_candidates.jsonl", "selection_ledger.jsonl"))
    if (manifest["status"] != "v3_l6c_candidate_selection_sealed_before_physics"
            or manifest["protocol_lock_sha256"] != _sha(output / "protocol_lock.json")
            or manifest["route_ledger_sha256"] != _sha(output / "route_ledger.jsonl")
            or manifest["raw_candidates_sha256"] != _sha(output / "raw_candidates.jsonl")
            or manifest["selection_ledger_sha256"] != _sha(output / "selection_ledger.jsonl")
            or manifest["targets"] != 15 or manifest["routable_targets"] != 9
            or manifest["abstained_targets"] != 6 or manifest["raw_slots"] != 6912
            or manifest["selected_slots"] != len(selected)
            or manifest["physics_evaluations_executed"] != 0
            or manifest["source_center_id_runtime_available"] is not False
            or manifest["formal_use_allowed"] is not False or manifest["deployment_allowed"] is not False
            or any("source_center_id" in row for row in routes)
            or selected != _selection(raw)):
        raise V3RoutedLocalError("L6c prephysics selection changed")
    return manifest


def _physics_result(runtime: Any, profile: Any, row: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    design = Geometry22Design.from_mapping(row["design"])
    result = runtime.evaluate(design)
    if (result.design_id != row["design_id"] or result.design_schema_hash != row["design_schema_hash"]
            or result.physics_bundle_hash != row["physics_bundle_hash"]):
        raise V3RoutedLocalError("L6c V3 physics result identity changed")
    engine = result.engine_result
    diagnostic = assess_constraint_profile(engine, profile, conservation_tolerances=runtime.base_runtime.conservation_tolerances).as_mapping()
    performance = ({name: float(engine.performance[name]) for name in NAMES} if engine.solver.converged else None)
    if performance is not None and any(not math.isfinite(v) or v <= 0 for v in performance.values()):
        performance = None
    target_values = np.asarray([target["target_performance"][name] for name in NAMES])
    rho = (float(np.max(np.abs(np.asarray([performance[name] for name in NAMES]) / target_values - 1.0) / 0.05))
           if performance is not None else None)
    return {"target_id": row["target_id"], "arm": row["arm"], "selection_rank": row["selection_rank"],
            "design_id": row["design_id"], "route_center_id": row["route_center_id"],
            "solver_converged": engine.solver.converged,
            "solver_failure_code": engine.solver.failure_code.value if engine.solver.failure_code else None,
            "diagnostic_domain_valid": bool(diagnostic["numerically_valid"] and diagnostic["physics_domain_valid"]),
            "engine_feasible": bool(engine.feasible), "performance": performance,
            "actual_rho_5pct": rho, "software_hit_5pct": bool(rho is not None and rho <= 1.0),
            "physics_bundle_hash": result.physics_bundle_hash,
            "design_schema_hash": result.design_schema_hash,
            "formal_use_allowed": False, "deployment_allowed": False}


def evaluate_v3_routed_local(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    cfg = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / cfg["output_root"]
    selection = verify_v3_routed_local_selection(root, output)
    if any((output / name).exists() for name in ("physics_ledger.jsonl", "f_recompute.jsonl", "complete_manifest.json")):
        raise V3RoutedLocalError("L6c physics evaluation exists; refusing to repeat")
    targets, _, _, _, _ = _source(root, cfg)
    target_map = {row["target_id"]: _runtime_target(row) for row in targets}
    selected = _rows(output / "selection_ledger.jsonl")
    runtime = build_v3_development_physics_runtime(root)
    profile = ConstraintProfile.from_yaml(root / cfg["sources"]["diagnostic_profile_path"])
    physics = [_physics_result(runtime, profile, row, target_map[row["target_id"]]) for row in selected]
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in physics:
        by_key[(row["target_id"], row["arm"])].append(row)
    selected_by_id = {(row["target_id"], row["arm"], row["design_id"]): row for row in selected}
    f_rows = []
    for (target_id, arm), group in sorted(by_key.items()):
        valid = [row for row in group if row["actual_rho_5pct"] is not None]
        if not valid:
            continue
        best = min(valid, key=lambda row: (row["actual_rho_5pct"], row["design_id"]))
        source = selected_by_id[(target_id, arm, best["design_id"])]
        repeated = _physics_result(runtime, profile, source, target_map[target_id])
        f_rows.append({"target_id": target_id, "arm": arm, "design_id": best["design_id"],
                       "independent_v3_recompute_equal": repeated == best,
                       "original_rho_5pct": best["actual_rho_5pct"],
                       "recomputed_rho_5pct": repeated["actual_rho_5pct"],
                       "formal_use_allowed": False, "deployment_allowed": False})
    _write_new(output / "physics_ledger.jsonl", b"".join(_bytes(row) for row in physics))
    _write_new(output / "f_recompute.jsonl", b"".join(_bytes(row) for row in f_rows))
    arms = {}
    for arm in ARMS:
        group = [row for row in physics if row["arm"] == arm]
        hits = {row["target_id"] for row in group if row["software_hit_5pct"]}
        arms[arm] = {"selected_candidates": len(group), "targets_evaluated": len({row["target_id"] for row in group}),
                     "solver_converged": sum(row["solver_converged"] for row in group),
                     "diagnostic_domain_valid": sum(row["diagnostic_domain_valid"] for row in group),
                     "engine_feasible": sum(row["engine_feasible"] for row in group),
                     "software_hit_targets_5pct": len(hits),
                     "software_hit_target_ids_5pct": sorted(hits),
                     "f_recompute_count": sum(row["arm"] == arm for row in f_rows),
                     "f_recompute_equal_count": sum(row["arm"] == arm and row["independent_v3_recompute_equal"] for row in f_rows)}
    complete = {"status": "v3_l6c_old_validation_covered_target_software_diagnostic_complete",
                "selection_manifest_sha256": _sha(output / "selection_manifest.json"),
                "physics_ledger_sha256": _sha(output / "physics_ledger.jsonl"),
                "f_recompute_sha256": _sha(output / "f_recompute.jsonl"),
                "targets": 15, "routable_targets": 9, "abstained_targets": 6,
                "raw_slots": selection["raw_slots"], "physics_evaluations": len(physics),
                "f_recomputations": len(f_rows), "arms": arms,
                "old_validation_reused_for_diagnostic": True,
                "old_holdout_labels_parsed": False,
                "hit_tolerance_relative_each_output": 0.05,
                "formal_use_allowed": False, "deployment_allowed": False}
    _write_new(output / "complete_manifest.json", complete)
    return verify_v3_routed_local_complete(root, output)


def verify_v3_routed_local_complete(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root, output = Path(repository_root).resolve(), Path(output_root).resolve()
    selection = verify_v3_routed_local_selection(root, output)
    complete = json.loads((output / "complete_manifest.json").read_text())
    physics, f_rows = _rows(output / "physics_ledger.jsonl"), _rows(output / "f_recompute.jsonl")
    if (complete["status"] != "v3_l6c_old_validation_covered_target_software_diagnostic_complete"
            or complete["selection_manifest_sha256"] != _sha(output / "selection_manifest.json")
            or complete["physics_ledger_sha256"] != _sha(output / "physics_ledger.jsonl")
            or complete["f_recompute_sha256"] != _sha(output / "f_recompute.jsonl")
            or complete["targets"] != 15 or complete["routable_targets"] != 9
            or complete["abstained_targets"] != 6 or complete["raw_slots"] != selection["raw_slots"]
            or complete["physics_evaluations"] != len(physics)
            or complete["f_recomputations"] != len(f_rows)
            or any(not row["independent_v3_recompute_equal"] for row in f_rows)
            or complete["old_validation_reused_for_diagnostic"] is not True
            or complete["old_holdout_labels_parsed"] is not False
            or complete["formal_use_allowed"] is not False or complete["deployment_allowed"] is not False):
        raise V3RoutedLocalError("L6c physics result or F recompute changed")
    return complete
