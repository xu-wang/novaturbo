"""Frozen G3 campaign: raw generation barrier, geometry-only scoring, sealed slots.

The N1 implementation and its smoke evidence remain immutable. This adapter fixes
geometry-associated scorer routing and applies existing geometry hard checks.
It does not add optimization, projection, resampling, or physics evaluations.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from unittest.mock import patch

import numpy as np
import scipy
import torch
import yaml

from src.ai import geometry19_performance_gated_inference as adapter
from src.ai.geometry19_performance_gated_protocol import canonical, file_hash, read_rows, require, within
from src.ai.geometry19_performance_gated_training import verified_data, verify_implementation
from src.ai.geometry19_performance_gated_training_preparation import transform_condition
from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.ai.geometry19_v2_2_stable_root_s3_v4_1_scoring import _load_model
from src.physics import Geometry19, GeometryStateError, derive_geometry_state
from src.physics.geometry19_far_precision_v1 import build_candidate
from src.physics.runtime import PhysicsRuntime

CONFIG = "config/experiments/geometry19_performance_gated_g3_v1.yaml"
VERSION = "geometry19_performance_gated_g3_v1"
ARMS = adapter.ARMS
UNITS = {"thrust_N": "N", "tsfc_kg_N_s": "kg/(N*s)", "mass_kg": "kg"}


def digest(value):
    return sha256(canonical(value)).hexdigest()


def seal_json(path, value):
    payload = canonical(value) + b"\n"
    if path.exists():
        require(path.read_bytes() == payload, f"Sealed JSON changed: {path}")
    else:
        adapter._write_once(path, payload)


def no_physics(*args, **kwargs):
    raise RuntimeError("G3 forbids PhysicsRuntime.evaluate")


def validate_config(config):
    require(config["protocol_version"] == VERSION, "G3 version changed")
    require(config["population"] == {
        "targets": 99, "seeds": [42, 43, 44], "arms": list(ARMS), "tasks": 1188,
        "raw_per_task": 256, "raw_total": 304128, "selection_per_task": 8,
        "selection_slots_total": 9504,
    }, "G3 frozen population changed")
    require(config["authorization"] == {
        "full_G3_generation_and_scoring": True, "G4_physics": False, "dflow": False,
        "gradient_enhancement": False, "nsga2": False, "training": False,
        "confirmation_content_access": False, "formal_use_allowed": False, "deployment_allowed": False,
    }, "G3 authorization changed")
    require(config["execution"]["physics_calls"] == 0, "G3 cannot evaluate physics")
    require(config["filtering"]["source_root_used_for_scoring"] is False, "Source-root scoring forbidden")
    require(config["generation"]["generation_seeds"] == {42: 20261001, 43: 20261002, 44: 20261003}, "Seed pairing changed")


def preflight(root):
    """Verify the frozen graph and construct (but never evaluate) the FAR runtime."""
    config = yaml.safe_load((root / CONFIG).read_text())
    validate_config(config)
    hashes = {}

    def verify(role, spec):
        path = within(root, spec["path"])
        require(file_hash(path) == spec["sha256"], f"Input hash changed: {role}")
        hashes[spec["path"]] = spec["sha256"]
        return path

    parent = yaml.safe_load(verify("parent_protocol", config["parent_protocol"]).read_text())
    inference = yaml.safe_load(verify("inference_protocol", config["inference_protocol"]).read_text())
    verify("inference_implementation", config["inference_implementation"])
    runtime_targets = read_rows(verify("runtime_targets", config["runtime_targets"]))
    paths = {role: verify(role, spec) for role, spec in inference["inputs"].items()}
    adapter.validate_protocol(inference)
    require(json.loads(paths["n0_compatibility"].read_text())["status"] == "V2_2_CROSS_VERSION_COMPATIBILITY_PASS", "N0 not passed")
    audit = json.loads(paths["training_audit"].read_text())
    require(audit["status"] == "G1_G2_TRAINING_AUDIT_PASS", "Training audit not passed")
    train_config, _, train_protocol, decoders, _, _ = verified_data(root)
    implementation_hash = verify_implementation(root, train_config)
    require(implementation_hash == audit["implementation_lock_sha256"], "Training implementation differs")
    require(decoders == json.loads(paths["expert_decoders"].read_text()), "Decoder identity differs")
    require(decoders["root_order"] == sorted(decoders["root_order"]) and len(set(decoders["root_order"])) == 33, "Root order invalid")
    require([e["root_seed_id"] for e in decoders["experts"]] == decoders["root_order"] and
            [e["root_index"] for e in decoders["experts"]] == list(range(33)), "Expert order invalid")
    for run in audit["runs"]:
        spec = inference["models"]["checkpoints"][run["seed"]][run["arm"]]
        require(spec == {"path": run["checkpoint_path"], "sha256": run["checkpoint_sha256"]}, "Selected checkpoint differs")
        verify("checkpoint", spec)
    targets = read_rows(paths["target_registry"])
    require(len(targets) == len(runtime_targets) == 99 and
            [t["target_index"] for t in targets] == list(range(99)), "Target order differs")
    require(len({t["target_id"] for t in targets}) == 99 and
            {t["root_seed_id"] for t in targets} == set(decoders["root_order"]), "Target identities differ")
    for t, public in zip(targets, runtime_targets):
        require(public == {k: t[k] for k in ("target_id", "performance", "units")}, "Target runtime view differs")
        require(public["units"] == UNITS, "Target units differ")
    context, _, _ = build_candidate(build_pipeline_context(root, pipeline_profile="v2_2_development"), root)
    require(canonical(context.contract_mapping()) + b"\n" == paths["far_precision_runtime"].read_bytes(), "Actual FAR runtime differs from training")
    require(context.context_hash == inference["physics"]["context_sha256"] and
            context.physics_runtime.physics_bundle_hash == inference["physics"]["bundle_sha256"], "FAR identity differs")
    transform = json.loads(paths["condition_transform"].read_text())
    require(transform["dimension"] == 3 and transform["fit_split"] == "train", "Condition transform differs")
    schema = yaml.safe_load(paths["schema"].read_text())
    require(schema == dict(context.physics_runtime.design_schema), "Runtime schema differs")
    # Lock imported implementation, tests, and transitive frozen training code too.
    code = {}
    for name in (
        "src/ai/geometry19_performance_gated_g3.py", "scripts/run_geometry19_performance_gated_g3.py",
        "tests/test_geometry19_performance_gated_g3.py", "src/ai/geometry19_performance_gated_inference.py",
        "src/ai/geometry19_performance_gated_models.py", "src/ai/geometry19_performance_gated_training.py",
        "src/ai/geometry19_performance_gated_training_preparation.py", "src/ai/geometry19_performance_gated_protocol.py",
        "src/ai/geometry19_v2_2_stable_root_s2_training.py", "src/ai/geometry19_v2_2_stable_root_s3_v4_1_scoring.py",
        "src/physics/geometry19_far_precision_v1.py",
    ):
        code[name] = file_hash(root / name)
    return dict(config=config, inference=inference, parent=parent, context=context, paths=paths,
                targets=targets, runtime_targets=runtime_targets, decoders=decoders, schema=schema,
                transform=transform, inputs=hashes, code=code,
                train_protocol=train_protocol, implementation_hash=implementation_hash,
                training_audit=audit)


def source_bank(seed, target_index, count=256):
    rng = np.random.Generator(np.random.PCG64(seed + target_index))
    return rng.standard_normal((count, 16), dtype=np.float32), rng.random((count, 3))


def dispatch_for(arm, probabilities, roots, count=256, *, oracle_root=None):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    require(probabilities.shape == (len(roots),) and np.isfinite(probabilities).all() and
            (probabilities >= 0).all() and probabilities.sum() > 0, "Invalid gate probabilities")
    if arm != ARMS[2]:
        require(oracle_root is None, "Oracle root forbidden for non-oracle arm")
    if arm == ARMS[3]:
        return None
    if arm == ARMS[2]:
        require(oracle_root in roots, "Missing oracle audit root")
        return np.full(count, roots.index(oracle_root), dtype=np.int64)
    if arm == ARMS[0]:
        index = min(range(len(roots)), key=lambda i: (-probabilities[i], roots[i]))
        return np.full(count, index, dtype=np.int64)
    require(arm == ARMS[1], "Unknown generation arm")
    allocation = adapter.largest_remainder_top2(probabilities, roots, count)
    return np.concatenate([np.full(n, i, dtype=np.int64) for i, n in sorted(allocation.items(), key=lambda v: roots[v[0]])])


def finite_list(values):
    return [float(v) if math.isfinite(float(v)) else None for v in values]


def decode_rows(states, dispatch, integer_classes, decoders, schema, model_config):
    """Keep every slot; canonical scorer root is determined only by geometry."""
    output = []
    for i, state in enumerate(states):
        if not np.isfinite(state).all():
            row = dict(geometry19=None, design_hash=None, valid=False, schema_valid=False,
                       support_valid=False, support_root_id=None, support_root_index=None,
                       local_distance=None, rejection_reason="nonfinite_generation")
        elif dispatch is None:
            row = adapter._decode_pooled(states[i:i+1], integer_classes[i:i+1], decoders, schema, .20)[0]
        else:
            row = adapter._decode_local(states[i:i+1], dispatch[i:i+1], decoders, schema, .20)[0]
        row["generator_support_distance"] = row["local_distance"] if dispatch is not None else None
        row["scoring_support_distance"] = None
        row["geometry_hard_valid"] = False
        row["geometry_error"] = None
        # Local validity still requires the generator's own selected support.
        # Scoring root is canonical for every arm, including overlapping supports.
        if row["valid"]:
            expert, distance = adapter._assign_pooled_support(row["geometry19"], decoders, .20)
            if expert is None:
                row.update(valid=False, support_valid=False, rejection_reason="serialized_geometry_outside_support")
            else:
                row.update(support_root_id=expert["root_seed_id"], support_root_index=expert["root_index"],
                           scoring_support_distance=distance)
            try:
                design = Geometry19.from_mapping(row["geometry19"])
                derive_geometry_state(design, model_config)
                row["geometry_hard_valid"] = True
            except (ValueError, TypeError, GeometryStateError) as exc:
                row.update(valid=False, rejection_reason="geometry_hard_rejection", geometry_error=str(exc))
        if row["geometry19"] is not None:
            require(digest(row["geometry19"]) == row["design_hash"], "Decoded geometry hash differs")
        row["generated_state"] = finite_list(state)
        row["generated_state_float32_sha256"] = sha256(np.asarray(state, dtype="<f4").tobytes()).hexdigest()
        output.append(row)
    return output


@torch.no_grad()
def generate_raw(public_target, target_index, seed, arm, models, inputs, *, oracle_root=None):
    require(set(public_target) == {"target_id", "performance", "units"} and public_target["units"] == UNITS,
            "Normal inference accepts only public target fields")
    gseed = inputs["config"]["generation"]["generation_seeds"][seed]
    source, uniforms = source_bank(gseed, target_index)
    condition = torch.tensor([transform_condition(public_target["performance"], inputs["transform"])] * 256,
                             dtype=torch.float32, device="cuda:0")
    probabilities = models["gate"](condition[:1]).softmax(dim=1)[0].cpu().numpy()
    dispatch = dispatch_for(arm, probabilities, inputs["decoders"]["root_order"], oracle_root=oracle_root)
    tensor = torch.from_numpy(source).to("cuda:0")
    routing = None if dispatch is None else torch.from_numpy(dispatch).to("cuda:0")
    model = models["pooled_cfm" if dispatch is None else "local_cfm"]
    states = adapter.rk4_integrate(model, tensor, condition, steps=32, dispatch=routing).cpu().numpy()
    integer_classes = None
    if dispatch is None:
        samples = []
        for k, logits in enumerate(model.integer_logits(condition[:1])):
            p = logits.softmax(dim=1)[0].cpu().numpy().astype(np.float64)
            require(np.isfinite(p).all() and (p >= 0).all() and p.sum() > 0, "Invalid pooled integer probabilities")
            cdf = np.cumsum(p / p.sum())
            cdf[-1] = 1.0
            samples.append(np.searchsorted(cdf, uniforms[:, k], side="right"))
        integer_classes = np.stack(samples, axis=1)
    rows = decode_rows(states, dispatch, integer_classes, inputs["decoders"], inputs["schema"],
                       inputs["context"].physics_runtime.model_config)
    for slot, row in enumerate(rows):
        row.update(target_id=public_target["target_id"], target_index=target_index,
                   training_seed=seed, generation_seed=gseed, arm=arm, raw_slot_id=slot,
                   noise_id=f"{gseed + target_index}:{slot}", initial_source=source[slot].tolist(),
                   source_sha256=sha256(source[slot].tobytes()).hexdigest(),
                   selected_expert_index=None if dispatch is None else int(dispatch[slot]),
                   raw_slot_retained=True)
    return rows, {"gate_probabilities": probabilities.tolist(), "root_order": inputs["decoders"]["root_order"],
                  "source_matrix_sha256": sha256(source.tobytes()).hexdigest(), "runtime_target": public_target,
                  "selected_expert_counts": {} if dispatch is None else dict(Counter(map(str, dispatch.tolist())))}


def rank_rows(raw, target, inputs, scorer_ids, scorer):
    """Produce full predictions/ranks plus eight explicit selection slots."""
    rows = deepcopy(raw)
    tolerances = np.asarray([inputs["parent"]["evaluation"]["relative_tolerances"][n] for n in adapter.OUTPUT_NAMES])
    adapter._score(rows, target, inputs["decoders"], inputs["schema"], scorer_ids, scorer, tolerances)
    seen, candidates = {}, []
    for row in rows:
        row["rank"] = None
        row["duplicate_of_raw_slot"] = None
        if not row["valid"]:
            continue
        if not math.isfinite(row["predicted_rho"]) or not all(math.isfinite(v) and v > 0 for v in row["predicted_performance"].values()):
            row.update(valid=False, rejection_reason="nonfinite_surrogate_prediction")
            row.pop("predicted_rho", None)
            row.pop("predicted_performance", None)
            continue
        if row["design_hash"] in seen:
            row["duplicate_of_raw_slot"] = seen[row["design_hash"]]
        else:
            seen[row["design_hash"]] = row["raw_slot_id"]
            candidates.append(row)
    candidates.sort(key=lambda r: (r["predicted_rho"], r["design_hash"]))
    for rank, row in enumerate(candidates, 1):
        row["rank"] = rank
    selection = []
    common = {k: raw[0][k] for k in ("target_id", "target_index", "arm", "training_seed", "generation_seed")}
    for rank in range(1, 9):
        if rank <= len(candidates):
            row = candidates[rank - 1]
            selection.append({**common, "selection_rank": rank, "missing": False,
                              **{k: row[k] for k in ("raw_slot_id", "geometry19", "design_hash", "support_root_id", "predicted_rho")}})
        else:
            selection.append({**common, "selection_rank": rank, "missing": True, "raw_slot_id": None,
                              "geometry19": None, "design_hash": None, "support_root_id": None, "predicted_rho": None,
                              "failure_reason": "insufficient_valid_unique_candidates"})
    predictions = [{k: v for k, v in row.items() if k not in {"geometry19", "generated_state", "initial_source"}} for row in rows]
    return predictions, selection


def shard_read(folder, identity, names):
    """Never regenerate an incomplete or tampered shard."""
    paths = [folder / n for n in names] + [folder / "Seal.json"]
    if not any(p.exists() for p in paths):
        require(not folder.exists(), f"Orphan directory: {folder}")
        return None
    require(all(p.is_file() for p in paths), f"Orphan partial shard: {folder}")
    require(set(p.name for p in folder.iterdir()) == set(names) | {"Seal.json"}, "Unexpected shard files")
    seal = json.loads(paths[-1].read_text())
    require(seal["identity"] == identity, "Shard identity changed")
    require(set(seal["files"]) == set(names), "Shard file set changed")
    for name in names:
        require(file_hash(folder / name) == seal["files"][name], f"Shard file hash changed: {name}")
    return seal


def shard_write(folder, identity, payloads):
    folder.mkdir(parents=True, exist_ok=False)
    for name, value in payloads.items():
        if name.endswith(".jsonl"):
            adapter._write_jsonl(folder / name, value)
        else:
            seal_json(folder / name, value)
    seal = {"identity": identity, "files": {n: file_hash(folder / n) for n in payloads}}
    seal_json(folder / "Seal.json", seal)
    return seal


def task_list(indices):
    return [(seed, index, arm) for seed in (42, 43, 44) for index in indices for arm in ARMS]


def task_key(task):
    seed, index, arm = task
    return f"seed{seed}/target{index:03d}/{arm}"


def concatenate(output, name, paths):
    destination = output / name
    expected = sha256()
    for path in paths:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                expected.update(block)
    if destination.exists():
        require(file_hash(destination) == expected.hexdigest(), "Aggregate ledger changed")
    else:
        with destination.open("xb") as stream:
            for path in paths:
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        require(file_hash(destination) == expected.hexdigest(), "Aggregate write differs")
    return {"path": name, "sha256": expected.hexdigest()}


def verify_complete(output, lock):
    require((output / "Complete_Manifest.json").is_file(), "Campaign not fully sealed")
    manifest = json.loads((output / "Complete_Manifest.json").read_text())
    require(manifest["lock_sha256"] == digest(lock), "Complete manifest identity changed")
    for name, expected in manifest["files"].items():
        require(file_hash(within(output, name)) == expected, f"Completed output changed: {name}")
    summary = json.loads((output / "G3_Summary.json").read_text())
    require(summary["raw_slots"] == len(lock["tasks"]) * 256 and summary["selection_slots"] == len(lock["tasks"]) * 8,
            "Completed population differs")
    return summary


def model_reload_probes(models, seed, inputs, root):
    """Compare fixed non-target probes through fresh checkpoint instances exactly."""
    replay, _ = adapter.load_seed_models(root, inputs["inference"], seed, inputs["decoders"], "cuda:0")
    condition = torch.zeros((33, 3), device="cuda:0")
    state = torch.zeros((33, 16), device="cuda:0")
    times = torch.linspace(0, 1, 33, device="cuda:0")
    dispatch = torch.arange(33, device="cuda:0")
    with torch.no_grad():
        for arm in models:
            a, b = models[arm], replay[arm]
            require(all(torch.equal(a.state_dict()[key], b.state_dict()[key]) for key in a.state_dict()), "Reload weights differ")
            args = (condition,) if arm == "gate" else (state, times, condition, dispatch) if arm == "local_cfm" else (state, times, condition)
            require(torch.equal(a(*args), b(*args)), "Reload probe prediction differs")
            if arm == "pooled_cfm":
                require(all(torch.equal(x, y) for x, y in zip(a.integer_logits(condition), b.integer_logits(condition))), "Reload integer logits differ")
    return {"seed": seed, "models": 3, "strict_weights_equal": True, "fixed_probe_predictions_equal": True}


def report(summary):
    lines = ["# G3 full candidate generation and selection", "", f"Status: `{summary['status']}`.", "",
             f"Tasks: {summary['tasks']}; raw slots: {summary['raw_slots']}; selection slots: {summary['selection_slots']}.",
             f"Selected designs: {summary['selected_candidates']}; missing slots: {summary['missing_slots']}; physics calls: 0.", "",
             "| Arm | Raw slots | Valid after hard checks | Unique valid | Selected | Missing |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, counts in summary["by_arm"].items():
        lines.append(f"| {arm} | {counts['raw']} | {counts['valid']} | {counts['unique_valid']} | {counts['selected']} | {counts['missing']} |")
    lines.extend(["", "All raw shards were sealed before loading the scorer. Geometry hard checks use the existing frozen runtime calibration. Scoring-root association depends only on candidate geometry for all arms, including overlapping supports. Missing slots remain failures in the fixed denominator.", "",
                  "This is generation/selection evidence only. No physics hit rate or generator-success PASS is claimed. The 99 targets are a development benchmark; target 0 was previously observed in N1 smoke. Model weights, generation budgets and acceptance thresholds were not tuned using smoke results.", ""])
    return "\n".join(lines)


def execute(root, *, smoke=False, resume=False, verify_only=False):
    root = root.resolve()
    with patch.object(PhysicsRuntime, "evaluate", no_physics):
        inputs = preflight(root)
        config = inputs["config"]
        indices = [0] if smoke else list(range(99))
        tasks = task_list(indices)
        output = within(root, config["artifacts"]["smoke_root" if smoke else "output_root"])
        if verify_only:
            require(output.is_dir(), "No campaign to verify")
            lock = json.loads((output / "Protocol_Lock.json").read_text())
            require(lock["code"] == inputs["code"] and lock["inputs"] == inputs["inputs"] and
                    lock["protocol_sha256"] == file_hash(root / CONFIG), "Verification input/code identity changed")
            return verify_complete(output, lock)
        require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "Set deterministic CUBLAS workspace")
        environment = adapter._cuda_manifest()
        environment.update(numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform())
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        identity = {"protocol_sha256": file_hash(root / CONFIG), "code": inputs["code"], "inputs": inputs["inputs"],
                    "environment": environment, "physics_context_hash": inputs["context"].context_hash,
                    "physics_bundle_hash": inputs["context"].physics_runtime.physics_bundle_hash,
                    "tasks": [list(t) for t in tasks], "scope": "smoke" if smoke else "full_G3"}
        if resume:
            require(output.is_dir(), "Missing campaign for resume")
            lock = json.loads((output / "Protocol_Lock.json").read_text())
            require(all(lock[k] == v for k, v in identity.items()), "Resume identity changed")
        else:
            output.mkdir(parents=True, exist_ok=False)
            lock = {**identity, "created_at": datetime.now(timezone.utc).isoformat()}
            seal_json(output / "Protocol_Lock.json", lock)
        with (output / ".runner.lock").open("a") as mutex:
            fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (output / "Complete_Manifest.json").exists():
                return verify_complete(output, lock)
            return run_stages(root, output, inputs, lock, tasks)


def run_stages(root, output, inputs, lock, tasks):
    lock_hash = digest(lock)
    raw_names = ["Raw.jsonl", "Task.json"]
    rank_names = ["Predictions.jsonl", "Selection.jsonl"]
    # Check every existing shard before doing any new work.
    existing = {}
    for task in tasks:
        ident = {"lock_sha256": lock_hash, "task": list(task)}
        folder = output / "raw" / task_key(task)
        existing[task] = shard_read(folder, ident, raw_names)
        ranked = output / "ranked" / task_key(task)
        if ranked.exists():
            require((output / "Raw_Manifest.json").exists(), "Ranking began before raw barrier")
            require(existing[task] is not None, "Ranked task lacks raw evidence")
            shard_read(ranked, {**ident, "raw_seal_sha256": digest(existing[task])}, rank_names)
        if (output / "Raw_Manifest.json").exists():
            require(existing[task] is not None, "Sealed raw campaign missing shard")
    raw_seals, files = [], {}
    for seed in (42, 43, 44):
        seed_tasks = [task for task in tasks if task[0] == seed]
        models = None
        if any(existing[task] is None for task in seed_tasks):
            models, _ = adapter.load_seed_models(root, inputs["inference"], seed, inputs["decoders"], "cuda:0")
            seal_json(output / f"Reload_seed{seed}.json", model_reload_probes(models, seed, inputs, root))
        for _, index, arm in seed_tasks:
            task = (seed, index, arm)
            folder = output / "raw" / task_key(task)
            ident = {"lock_sha256": lock_hash, "task": list(task)}
            seal = existing[task]
            if seal is None:
                start = time.perf_counter()
                target = inputs["runtime_targets"][index]
                oracle = inputs["targets"][index]["root_seed_id"] if arm == ARMS[2] else None
                rows, meta = generate_raw(target, index, seed, arm, models, inputs, oracle_root=oracle)
                meta.update(elapsed_seconds=time.perf_counter() - start, raw_slots=256,
                            physics_context_hash=lock["physics_context_hash"], physics_calls=0)
                require(len(rows) == 256 and [r["raw_slot_id"] for r in rows] == list(range(256)), "Raw slots differ")
                seal = shard_write(folder, ident, {"Raw.jsonl": rows, "Task.json": meta})
            raw_seals.append({"task": list(task), "seal_sha256": digest(seal)})
            for name in raw_names + ["Seal.json"]:
                files[str((folder / name).relative_to(output))] = file_hash(folder / name)
            if len(raw_seals) % 36 == 0:
                print(json.dumps({"stage": "raw", "completed": len(raw_seals), "total": len(tasks)}), flush=True)
        if models is not None:
            del models
        torch.cuda.empty_cache()
    # Aggregate and seal ALL raw generation before loading the scorer.
    raw_artifact = concatenate(output, "Raw_Candidates.jsonl", [output / "raw" / task_key(t) / "Raw.jsonl" for t in tasks])
    raw_manifest = {"lock_sha256": lock_hash, "tasks": raw_seals, "raw_slots": 256 * len(tasks), "aggregate": raw_artifact}
    seal_json(output / "Raw_Manifest.json", raw_manifest)
    scorer_ids, scorer = _load_model(inputs["paths"]["s2_oracle_local_ridge"], "stable_38_oracle_root_local_ridge")
    task_metrics, ranking_seals = [], []
    source_by_task = {}
    for task in tasks:
        seed, index, arm = task
        raw_folder = output / "raw" / task_key(task)
        raw_seal = json.loads((raw_folder / "Seal.json").read_text())
        raw = read_rows(raw_folder / "Raw.jsonl")
        meta = json.loads((raw_folder / "Task.json").read_text())
        pair = (seed, index)
        if pair in source_by_task:
            require(source_by_task[pair] == meta["source_matrix_sha256"], "Paired source mismatch")
        source_by_task[pair] = meta["source_matrix_sha256"]
        ident = {"lock_sha256": lock_hash, "task": list(task), "raw_seal_sha256": digest(raw_seal)}
        folder = output / "ranked" / task_key(task)
        seal = shard_read(folder, ident, rank_names)
        if seal is None:
            prediction, selection = rank_rows(raw, inputs["runtime_targets"][index], inputs, scorer_ids, scorer)
            seal = shard_write(folder, ident, {"Predictions.jsonl": prediction, "Selection.jsonl": selection})
        else:
            prediction, selection = [read_rows(folder / n) for n in rank_names]
        require(len(prediction) == 256 and len(selection) == 8, "Ranking/selection denominator differs")
        selected = [r for r in selection if not r["missing"]]
        for row in selected:
            origin = raw[row["raw_slot_id"]]
            require(origin["valid"] and origin["geometry_hard_valid"] and origin["support_valid"] and
                    origin["geometry19"] == row["geometry19"] and origin["design_hash"] == row["design_hash"], "Selected candidate invalid")
        counts = dict(raw=256, valid=sum(r["valid"] for r in raw),
                      unique_valid=len({r["design_hash"] for r in prediction if r["valid"]}),
                      selected=len(selected), missing=8-len(selected),
                      scoring_root_reassigned=sum(r["valid"] and r["selected_expert_index"] is not None and
                                                  r["selected_expert_index"] != r["support_root_index"] for r in raw))
        task_metrics.append(dict(seed=seed, target_index=index, arm=arm,
                                 target_class=inputs["targets"][index]["target_class"],
                                 source_root_id=inputs["targets"][index]["root_seed_id"],
                                 generation_seconds=meta["elapsed_seconds"], **counts,
                                 rejection_counts=dict(Counter(r["rejection_reason"] for r in raw if not r["valid"]))))
        ranking_seals.append({"task": list(task), "seal_sha256": digest(seal)})
        for name in rank_names + ["Seal.json"]:
            files[str((folder / name).relative_to(output))] = file_hash(folder / name)
        if len(ranking_seals) % 99 == 0:
            print(json.dumps({"stage": "ranking", "completed": len(ranking_seals), "total": len(tasks)}), flush=True)
    aggregates = {}
    for name in rank_names:
        aggregates[name] = concatenate(output, name, [output / "ranked" / task_key(t) / name for t in tasks])
    seal_json(output / "Selection_Manifest.json", {"lock_sha256": lock_hash, "raw_manifest_sha256": file_hash(output / "Raw_Manifest.json"),
                                                 "tasks": ranking_seals, "aggregates": aggregates, "physics_calls": 0})
    keys = ("raw", "valid", "unique_valid", "selected", "missing", "scoring_root_reassigned")
    by_arm = {arm: {k: sum(r[k] for r in task_metrics if r["arm"] == arm) for k in keys} for arm in ARMS}
    by_seed_arm = {f"{seed}/{arm}": {k: sum(r[k] for r in task_metrics if r["seed"] == seed and r["arm"] == arm) for k in keys}
                   for seed in (42, 43, 44) for arm in ARMS}
    summary = {"status": "G3_SMOKE_SEALED" if lock["scope"] == "smoke" else "G3_CANDIDATES_AND_SELECTION_SEALED",
               "tasks": len(tasks), "raw_slots": len(tasks)*256, "selection_slots": len(tasks)*8,
               "selected_candidates": sum(r["selected"] for r in task_metrics),
               "missing_slots": sum(r["missing"] for r in task_metrics), "by_arm": by_arm, "by_seed_arm": by_seed_arm,
               "physics_calls": 0, "source_pairing_verified": True, "all_raw_sealed_before_scoring": True,
               "physics_context_hash": lock["physics_context_hash"], "physics_bundle_hash": lock["physics_bundle_hash"],
               "generator_quality_or_physics_hit_pass_claimed": False,
               "top1_raw_support_gate_each_seed": {str(seed): by_seed_arm[f"{seed}/{ARMS[0]}"]["valid"] / by_seed_arm[f"{seed}/{ARMS[0]}"]["raw"] >= .90 for seed in (42, 43, 44)},
               "next_stage": "G4_frozen_selection_far_precision_physics", "confirmation_content_read": False,
               "formal_use_allowed": False, "deployment_allowed": False}
    seal_json(output / "G3_Summary.json", summary)
    task_payload = b"".join(canonical(r) + b"\n" for r in task_metrics)
    if (output / "Task_Metrics.jsonl").exists():
        require((output / "Task_Metrics.jsonl").read_bytes() == task_payload, "Task metrics changed")
    else:
        adapter._write_once(output / "Task_Metrics.jsonl", task_payload)
    text = report(summary).encode()
    if (output / "G3_Report.md").exists():
        require((output / "G3_Report.md").read_bytes() == text, "Report changed")
    else:
        adapter._write_once(output / "G3_Report.md", text)
    for name in ["Protocol_Lock.json", "Raw_Candidates.jsonl", "Raw_Manifest.json", "Selection_Manifest.json",
                 "Predictions.jsonl", "Selection.jsonl", "Task_Metrics.jsonl", "G3_Summary.json", "G3_Report.md",
                 "Reload_seed42.json", "Reload_seed43.json", "Reload_seed44.json"]:
        files[name] = file_hash(output / name)
    seal_json(output / "Complete_Manifest.json", {"lock_sha256": lock_hash, "files": files, "physics_calls": 0})
    return verify_complete(output, lock)
