"""N1 inference adapter for performance-gated local and pooled CFM models."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from src.ai.geometry19_performance_gated_models import PerformanceGate, LocalCFM, PooledCFM
from src.ai.geometry19_performance_gated_protocol import canonical, file_hash, read_rows, require, within
from src.ai.geometry19_performance_gated_training_preparation import transform_condition
from src.ai.geometry19_v2_2_stable_root_s2_training import OUTPUT_NAMES, _local_predict, _schema_arrays
from src.ai.geometry19_v2_2_stable_root_s3_v4_1_scoring import _load_model


VERSION = "geometry19_v2_2_performance_gated_inference_v1"
PASS = "N1_COMMON_INFERENCE_ADAPTER_SMOKE_PASS"
ARMS = (
    "gated_top1_local_cfm",
    "gated_top2_local_cfm",
    "oracle_source_root_local_cfm",
    "pooled_33_global_cfm",
)


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write_once(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_once(path, b"".join(canonical(dict(row)) + b"\n" for row in rows))


def validate_protocol(config: Mapping[str, Any]) -> None:
    require(config["protocol_version"] == VERSION, "Inference protocol version changed")
    require(config["stage"] == "N1_common_inference_adapter_and_gpu_smoke", "Wrong inference stage")
    require(config["physics"] == {
        "model_version": "geometry19_v2_2_far_precision_v1",
        "context_sha256": "49e9b09a047e096f59bd65132fcfaef5b8dc363d563d2be638247e7eaf55eb72",
        "bundle_sha256": "2ecf2664a024ccc7ce7bad33d0ccf755eb72d1131fe01b773dff9a64b71ee5be",
        "evaluation_in_N1": "forbidden",
    }, "FAR precision physics identity changed")
    inference = config["inference"]
    require(inference["condition_fields"] == list(OUTPUT_NAMES) and inference["external_fields"] == ["target_id", "performance", "units"], "External condition contract changed")
    require(inference["arms"] == list(ARMS) and inference["primary_arm"] == ARMS[0], "Inference arms changed")
    require(inference["raw_candidates_per_task"] == 256 and inference["physics_slots_per_task"] == 8, "Candidate budget changed")
    require(inference["ode"] == {"method": "rk4", "steps": 32}, "ODE contract changed")
    require(inference["outside_support"] == "reject_retain_raw_slot_no_projection_no_resampling", "Support policy changed")
    require(inference["scorer_target_root_access"] is False, "Scorer target-root leakage")
    require(config["models"]["training_seeds"] == [42, 43, 44], "Training seeds changed")
    require(config["smoke"]["tasks"] == 12 and config["smoke"]["expected_raw_slots"] == 3072, "Smoke scope changed")
    require(config["authorization"] == {
        "model_prediction_authorized": True,
        "smoke_candidate_generation_authorized": True,
        "full_99_target_generation_authorized": False,
        "physics_evaluation_authorized": False,
        "gradient_enhancement_authorized": False,
        "nsga2_authorized": False,
        "confirmation_content_access": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }, "Inference authorization changed")


def largest_remainder_top2(probabilities: np.ndarray, root_order: Sequence[str], total: int = 256) -> dict[int, int]:
    require(probabilities.shape == (len(root_order),) and total >= 2, "Bad top2 allocation inputs")
    order = sorted(range(len(root_order)), key=lambda index: (-float(probabilities[index]), str(root_order[index])))[:2]
    weights = probabilities[order] / probabilities[order].sum()
    exact = weights * (total - 2)
    base = np.floor(exact).astype(int)
    remaining = total - 2 - int(base.sum())
    fractional_order = sorted(range(2), key=lambda k: (-float(exact[k] - base[k]), str(root_order[order[k]])))
    for k in fractional_order[:remaining]:
        base[k] += 1
    result = {order[k]: int(base[k] + 1) for k in range(2)}
    require(sum(result.values()) == total and all(value >= 1 for value in result.values()), "Top2 allocation failed")
    return result


def rk4_integrate(model, source: torch.Tensor, condition: torch.Tensor, *, steps: int, dispatch: torch.Tensor | None = None) -> torch.Tensor:
    require(steps > 0 and source.shape[1] == 16 and condition.shape == (len(source), 3), "Bad CFM integration inputs")
    state = source
    dt = 1.0 / steps
    for step in range(steps):
        t0 = torch.full((len(state),), step * dt, dtype=state.dtype, device=state.device)
        tm = t0 + 0.5 * dt
        t1 = t0 + dt
        def velocity(value, time):
            return model(value, time, condition, dispatch) if dispatch is not None else model(value, time, condition)
        k1 = velocity(state, t0)
        k2 = velocity(state + 0.5 * dt * k1, tm)
        k3 = velocity(state + 0.5 * dt * k2, tm)
        k4 = velocity(state + dt * k3, t1)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return state


def _load_checkpoint(model: torch.nn.Module, path: Path, expected_hash: str, arm: str, seed: int, device: str) -> dict[str, Any]:
    require(file_hash(path) == expected_hash, f"Checkpoint hash changed: {seed}/{arm}")
    payload = torch.load(path, map_location=device, weights_only=True)
    require(payload["identity"]["arm"] == arm and int(payload["identity"]["seed"]) == seed, "Checkpoint identity changed")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return {"path": str(path), "sha256": expected_hash, "epoch": int(payload["epoch"]), "validation_objective": float(payload["validation_objective"])}


def load_seed_models(root: Path, config: Mapping[str, Any], seed: int, decoders: Mapping[str, Any], device: str) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    counts = [len(decoders["integer_categories"][axis]) for axis in decoders["integer_axes"]]
    models = {"gate": PerformanceGate(), "local_cfm": LocalCFM(), "pooled_cfm": PooledCFM(counts)}
    audit = {}
    specs = config["models"]["checkpoints"][seed]
    for arm, model in models.items():
        model.to(device)
        spec = specs[arm]
        audit[arm] = _load_checkpoint(model, within(root, spec["path"]), spec["sha256"], arm, seed, device)
    return models, audit


def _geometry_hash(geometry: Mapping[str, Any]) -> str:
    return sha256(canonical(dict(geometry))).hexdigest()


def _decode_local(states: np.ndarray, dispatch: np.ndarray, decoders: Mapping[str, Any], schema: Mapping[str, Any], radius_limit: float) -> list[dict[str, Any]]:
    experts = decoders["experts"]
    lower = np.asarray(decoders["continuous_lower"], dtype=float)
    span = np.asarray(decoders["continuous_span"], dtype=float)
    variables = sorted(schema["design_variables"], key=lambda item: int(item["index"]))
    rows = []
    for state, root_index in zip(states, dispatch):
        expert = experts[int(root_index)]
        normalized = np.asarray(expert["normalized_medoid_continuous"]) + float(expert["radius"]) * state
        support_distance = float(np.linalg.norm(state))
        support_ok = support_distance <= radius_limit + 1e-12
        schema_ok = bool(np.all(normalized >= 0.0) and np.all(normalized <= 1.0))
        continuous = lower + span * normalized
        values = {name: float(value) for name, value in zip(decoders["continuous_axes"], continuous)}
        values.update({name: int(value) for name, value in zip(decoders["integer_axes"], expert["integer_values"])})
        geometry = {str(variable["name"]): values[str(variable["name"])] for variable in variables}
        valid = support_ok and schema_ok
        rows.append({
            "geometry19": geometry,
            "design_hash": _geometry_hash(geometry),
            "support_root_index": int(root_index),
            "support_root_id": expert["root_seed_id"],
            "local_distance": support_distance,
            "schema_valid": schema_ok,
            "support_valid": support_ok,
            "valid": valid,
            "rejection_reason": None if valid else "outside_local_support" if not support_ok else "outside_schema",
        })
    return rows


def _assign_pooled_support(geometry: Mapping[str, Any], decoders: Mapping[str, Any], radius_limit: float) -> tuple[dict[str, Any] | None, float | None]:
    lower = np.asarray(decoders["continuous_lower"], dtype=float)
    span = np.asarray(decoders["continuous_span"], dtype=float)
    normalized = (np.asarray([float(geometry[name]) for name in decoders["continuous_axes"]]) - lower) / span
    candidates = []
    integer_values = [int(geometry[name]) for name in decoders["integer_axes"]]
    for expert in decoders["experts"]:
        if integer_values != expert["integer_values"]:
            continue
        state = (normalized - np.asarray(expert["normalized_medoid_continuous"])) / float(expert["radius"])
        distance = float(np.linalg.norm(state))
        if distance <= radius_limit + 1e-12:
            candidates.append((distance, str(expert["root_seed_id"]), expert))
    if not candidates:
        return None, None
    distance, _, expert = min(candidates, key=lambda item: (item[0], item[1]))
    return expert, distance


def _decode_pooled(states: np.ndarray, integer_indices: np.ndarray, decoders: Mapping[str, Any], schema: Mapping[str, Any], radius_limit: float) -> list[dict[str, Any]]:
    lower = np.asarray(decoders["continuous_lower"], dtype=float)
    span = np.asarray(decoders["continuous_span"], dtype=float)
    variables = sorted(schema["design_variables"], key=lambda item: int(item["index"]))
    rows = []
    for state, classes in zip(states, integer_indices):
        schema_ok = bool(np.all(state >= 0.0) and np.all(state <= 1.0))
        continuous = lower + span * state
        values = {name: float(value) for name, value in zip(decoders["continuous_axes"], continuous)}
        for axis, class_index in zip(decoders["integer_axes"], classes):
            values[axis] = int(decoders["integer_categories"][axis][int(class_index)])
        geometry = {str(variable["name"]): values[str(variable["name"])] for variable in variables}
        expert, distance = _assign_pooled_support(geometry, decoders, radius_limit) if schema_ok else (None, None)
        valid = schema_ok and expert is not None
        rows.append({
            "geometry19": geometry,
            "design_hash": _geometry_hash(geometry),
            "support_root_index": None if expert is None else int(expert["root_index"]),
            "support_root_id": None if expert is None else expert["root_seed_id"],
            "local_distance": distance,
            "schema_valid": schema_ok,
            "support_valid": expert is not None,
            "valid": valid,
            "rejection_reason": None if valid else "outside_schema" if not schema_ok else "outside_all_local_support",
        })
    return rows


def _sample_integer_classes(model: PooledCFM, condition: torch.Tensor, uniforms: np.ndarray) -> np.ndarray:
    logits = model.integer_logits(condition[:1])
    result = []
    for axis, values in enumerate(logits):
        probabilities = values.softmax(dim=1)[0].detach().cpu().numpy()
        cumulative = np.cumsum(probabilities)
        result.append(np.searchsorted(cumulative, uniforms[:, axis], side="right").clip(max=len(probabilities) - 1))
    return np.stack(result, axis=1)


def _score(rows: list[dict[str, Any]], target: Mapping[str, Any], decoders: Mapping[str, Any], schema: Mapping[str, Any], scorer_ids: Sequence[str], scorer: Mapping[str, np.ndarray], tolerances: np.ndarray) -> None:
    valid_indices = [index for index, row in enumerate(rows) if row["valid"]]
    if not valid_indices:
        return
    variables = sorted(schema["design_variables"], key=lambda item: int(item["index"]))
    names, lower, upper = _schema_arrays(schema)
    require(names == [str(variable["name"]) for variable in variables], "Schema order changed")
    x = np.asarray([[float(rows[index]["geometry19"][name]) for name in names] for index in valid_indices])
    x = (x - lower[None, :]) / (upper - lower)[None, :]
    route = {root: index for index, root in enumerate(scorer_ids)}
    route_indices = np.asarray([route[str(rows[index]["support_root_id"])] for index in valid_indices], dtype=int)
    prediction = np.exp(_local_predict(scorer, x, route_indices))
    target_values = np.asarray([float(target["performance"][name]) for name in OUTPUT_NAMES])
    rho = np.max(np.abs(prediction / target_values[None, :] - 1.0) / tolerances[None, :], axis=1)
    for ledger_index, values, value in zip(valid_indices, prediction.tolist(), rho.tolist()):
        rows[ledger_index]["predicted_performance"] = {name: float(axis) for name, axis in zip(OUTPUT_NAMES, values)}
        rows[ledger_index]["predicted_rho"] = float(value)


def generate_task(
    *, target: Mapping[str, Any], arm: str, train_seed: int, generation_seed: int,
    models: Mapping[str, torch.nn.Module], transform: Mapping[str, Any], decoders: Mapping[str, Any],
    schema: Mapping[str, Any], scorer_ids: Sequence[str], scorer: Mapping[str, np.ndarray],
    tolerances: np.ndarray, device: str, count: int = 256, steps: int = 32,
    oracle_root_id: str | None = None,
) -> list[dict[str, Any]]:
    require(arm in ARMS and count > 0, "Unknown arm or empty task")
    runtime_target = {key: target[key] for key in ("target_id", "performance", "units")}
    require(set(runtime_target) == {"target_id", "performance", "units"}, "Runtime target leakage")
    condition_values = transform_condition(runtime_target["performance"], transform)
    condition = torch.tensor([condition_values] * count, dtype=torch.float32, device=device)
    rng = np.random.Generator(np.random.PCG64(int(generation_seed) + int(target.get("target_index", 0))))
    source_array = rng.standard_normal((count, 16), dtype=np.float32)
    source = torch.from_numpy(source_array).to(device)
    with torch.no_grad():
        gate_probabilities = models["gate"](condition[:1]).softmax(dim=1)[0].detach().cpu().numpy()
        root_order = decoders["root_order"]
        if arm == "gated_top1_local_cfm":
            selected = sorted(range(33), key=lambda index: (-float(gate_probabilities[index]), root_order[index]))[0]
            dispatch_array = np.full(count, selected, dtype=np.int64)
        elif arm == "gated_top2_local_cfm":
            allocation = largest_remainder_top2(gate_probabilities, root_order, count)
            dispatch_array = np.concatenate([np.full(amount, index, dtype=np.int64) for index, amount in sorted(allocation.items(), key=lambda item: root_order[item[0]])])
        elif arm == "oracle_source_root_local_cfm":
            require(oracle_root_id is not None and oracle_root_id in root_order, "Oracle diagnostic requires an audited source root")
            dispatch_array = np.full(count, root_order.index(oracle_root_id), dtype=np.int64)
        else:
            dispatch_array = None
        if dispatch_array is not None:
            dispatch = torch.from_numpy(dispatch_array).to(device=device, dtype=torch.long)
            states = rk4_integrate(models["local_cfm"], source, condition, steps=steps, dispatch=dispatch).cpu().numpy()
            decoded = _decode_local(states, dispatch_array, decoders, schema, 0.20)
        else:
            states = rk4_integrate(models["pooled_cfm"], source, condition, steps=steps).cpu().numpy()
            integer_indices = _sample_integer_classes(models["pooled_cfm"], condition, rng.random((count, 3)))
            decoded = _decode_pooled(states, integer_indices, decoders, schema, 0.20)
    top2 = sorted(range(33), key=lambda index: (-float(gate_probabilities[index]), root_order[index]))[:2]
    for slot, row in enumerate(decoded):
        row.update({
            "target_id": runtime_target["target_id"], "target_index": int(target.get("target_index", 0)),
            "arm": arm, "training_seed": train_seed, "generation_seed": generation_seed,
            "raw_slot_id": slot, "noise_id": f"{generation_seed + int(target.get('target_index', 0))}:{slot}",
            "source_sha256": sha256(source_array[slot].tobytes()).hexdigest(),
            "gate_top2": [{"root_index": index, "root_id": root_order[index], "probability": float(gate_probabilities[index])} for index in top2],
            "selected_expert_index": None if dispatch_array is None else int(dispatch_array[slot]),
            "raw_slot_retained": True,
        })
    _score(decoded, runtime_target, decoders, schema, scorer_ids, scorer, tolerances)
    return decoded


def _select(rows: Sequence[Mapping[str, Any]], keep: int) -> list[dict[str, Any]]:
    best_by_hash = {}
    for row in rows:
        if not row["valid"] or "predicted_rho" not in row:
            continue
        key = row["design_hash"]
        current = best_by_hash.get(key)
        if current is None or (row["predicted_rho"], row["raw_slot_id"]) < (current["predicted_rho"], current["raw_slot_id"]):
            best_by_hash[key] = row
    ranked = sorted(best_by_hash.values(), key=lambda row: (row["predicted_rho"], row["design_hash"]))[:keep]
    return [{"selection_rank": rank + 1, "target_id": row["target_id"], "arm": row["arm"], "training_seed": row["training_seed"],
             "raw_slot_id": row["raw_slot_id"], "design_hash": row["design_hash"], "support_root_id": row["support_root_id"],
             "predicted_rho": row["predicted_rho"], "geometry19": row["geometry19"]} for rank, row in enumerate(ranked)]


def _cuda_manifest() -> dict[str, Any]:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "6", "Set CUDA_VISIBLE_DEVICES=6")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Physical GPU 6 is not visible")
    physical = subprocess.check_output(["nvidia-smi", "-i", "6", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"], text=True).strip()
    require(physical.split(",")[0].strip() == "6", "Wrong physical GPU")
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "visible_devices": "6", "logical_device": "cuda:0", "physical_device": physical}


def _report(summary: Mapping[str, Any]) -> str:
    return "\n".join([
        "# N1 Performance-Gated Inference Adapter Smoke",
        "",
        f"- Status: `{summary['status']}`",
        f"- Tasks: `{summary['tasks']}`",
        f"- Raw slots: `{summary['raw_slots']}`",
        f"- Valid slots: `{summary['valid_slots']}`",
        f"- Selected rows: `{summary['selected_rows']}`",
        f"- Checkpoints reloaded: `{summary['checkpoints_reloaded']}`",
        "- Physics evaluations: `0`",
        "- Candidate generation scope: one frozen smoke target only",
        "",
        "This smoke qualifies model reload, three-performance routing, RK4/32 generation, local/pooled decoding, support rejection, frozen S2 scoring, and top-8 ledger selection. It is not the 99-target G3 experiment and contains no physical closure.",
        "",
    ])


def run_smoke(root: Path, config_path: Path) -> dict[str, Any]:
    root, config_path = root.resolve(), config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_protocol(config)
    paths = {}
    input_hashes = {}
    for role, spec in config["inputs"].items():
        path = within(root, spec["path"])
        require(file_hash(path) == spec["sha256"], f"Inference input changed: {role}")
        paths[role] = path
        input_hashes[role] = spec["sha256"]
    compatibility = json.loads(paths["n0_compatibility"].read_text())
    require(compatibility["status"] == "V2_2_CROSS_VERSION_COMPATIBILITY_PASS", "N0 compatibility did not pass")
    require(json.loads(paths["training_summary"].read_text())["status"] == "G1_G2_TRAINING_COMPLETE", "Training incomplete")
    require(json.loads(paths["training_audit"].read_text())["status"] == "G1_G2_TRAINING_AUDIT_PASS", "Training audit incomplete")
    transform = json.loads(paths["condition_transform"].read_text())
    decoders = json.loads(paths["expert_decoders"].read_text())
    schema = yaml.safe_load(paths["schema"].read_text())
    targets = read_rows(paths["target_registry"])
    smoke_targets = [targets[index] for index in config["smoke"]["target_indices"]]
    scorer_ids, scorer = _load_model(paths["s2_oracle_local_ridge"], "stable_38_oracle_root_local_ridge")
    tolerances = np.asarray([float(value) for value in (0.0274514502151586, 0.008573042420144972, 0.01170710896944147)])
    environment = _cuda_manifest()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    output = within(root, config["artifacts"]["output_root"])
    require(not output.exists(), "Refusing to overwrite N1 smoke")
    output.mkdir(parents=True)
    all_rows, all_selected, reload_audit = [], [], {}
    for seed in config["models"]["training_seeds"]:
        models, audit = load_seed_models(root, config, seed, decoders, "cuda:0")
        reload_audit[str(seed)] = audit
        generation_seed = int(config["inference"]["generation_seed_by_training_seed"][seed])
        for target in smoke_targets:
            for arm in ARMS:
                rows = generate_task(target=target, arm=arm, train_seed=seed, generation_seed=generation_seed,
                    models=models, transform=transform, decoders=decoders, schema=schema, scorer_ids=scorer_ids,
                    scorer=scorer, tolerances=tolerances, device="cuda:0", count=256, steps=32,
                    oracle_root_id=target["root_seed_id"] if arm == "oracle_source_root_local_cfm" else None)
                all_rows.extend(rows)
                all_selected.extend(_select(rows, 8))
        del models
        torch.cuda.empty_cache()
    require(len(all_rows) == config["smoke"]["expected_raw_slots"], "Smoke raw slot count changed")
    require(len(reload_audit) == 3 and sum(len(value) for value in reload_audit.values()) == 9, "Checkpoint reload incomplete")
    require(all(row["raw_slot_retained"] for row in all_rows), "Raw slot dropped")
    summary = {
        "status": PASS,
        "protocol_sha256": file_hash(config_path),
        "implementation_sha256": file_hash(Path(__file__)),
        "cli_sha256": file_hash(root / "scripts/run_geometry19_performance_gated_inference_smoke.py"),
        "input_hashes": input_hashes,
        "physics_identity": config["physics"],
        "environment": environment,
        "tasks": len(smoke_targets) * len(ARMS) * 3,
        "raw_slots": len(all_rows),
        "valid_slots": sum(row["valid"] for row in all_rows),
        "selected_rows": len(all_selected),
        "checkpoints_reloaded": 9,
        "reload_audit": reload_audit,
        "by_arm": {arm: {"raw": sum(row["arm"] == arm for row in all_rows),
                          "valid": sum(row["arm"] == arm and row["valid"] for row in all_rows),
                          "selected": sum(row["arm"] == arm for row in all_selected)} for arm in ARMS},
        "physics_evaluations": 0,
        "candidate_generation_scope": "one_frozen_target_smoke_only",
        "full_99_target_generation_authorized": False,
        "confirmation_content_read": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "next_stage": "freeze_and_execute_original_G3_candidate_generation_then_G4_far_precision_physics",
    }
    _write_jsonl(output / config["artifacts"]["candidate_ledger"], all_rows)
    _write_jsonl(output / config["artifacts"]["selection_ledger"], all_selected)
    _write_json(output / config["artifacts"]["summary"], summary)
    _write_once(output / config["artifacts"]["report"], _report(summary).encode())
    evidence = {
        "status": PASS,
        "protocol_sha256": file_hash(config_path),
        "implementation_sha256": summary["implementation_sha256"],
        "cli_sha256": summary["cli_sha256"],
        "artifacts": {
            role: {
                "path": name,
                "sha256": file_hash(output / name),
            }
            for role, name in config["artifacts"].items()
            if role != "output_root" and role != "evidence_manifest"
        },
        "counts": {"tasks": summary["tasks"], "raw_slots": summary["raw_slots"], "selected_rows": summary["selected_rows"]},
        "physics_evaluations": 0,
    }
    _write_json(output / config["artifacts"]["evidence_manifest"], evidence)
    return summary
