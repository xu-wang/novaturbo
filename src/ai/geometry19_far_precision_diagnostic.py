"""Isolated, post-selection FAR tolerance diagnostic; never promotes training data."""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import threading
from types import SimpleNamespace

import numpy as np
import scipy

from src.ai.geometry19_performance_gated_protocol import canonical, file_hash, read_rows, require, OUTPUTS
from src.ai.geometry19_performance_gated_physics import verify_inputs, write_json, write_once
from src.ai.geometry19_performance_gated_physics_protocol import eligibility_flags
from src.physics.geometry19_v2_2_dataset import _evaluate_row
from src.physics import combustor_map_enhanced as combustor

ROOT_ID = "4cb10f6935baf51117f3cdbabf4ec2840ea23713994e76b3d8140538fd5d05f4"
BASE = "data/validation/v2_2_performance_gated_local_cfm_v1"
OUTPUT = "data/validation/v2_2_g0c_far_precision_diagnostic_v1"
PLAN = "config/experiments/geometry19_g0c_far_precision_diagnostic_v1.md"
LEDGER_HASH = "d7baa82b9323926c4b1f04876e30d7a45ab1ba9be7681b65ae9a1b4d0df46b4e"
COMBUSTOR_HASH = "8456246bceecf0599c3f3984e55dffa9548e17ded9bf4b8d1043d772c91a84bc"
PARAMETERS = {"A": {"xtol": 1e-12, "rtol": 1e-12, "maxiter": 64},
              "B": {"xtol": 1e-15, "rtol": 1e-15, "maxiter": 64}}
_ADAPTER_LOCK = threading.Lock()


def digest(value):
    return sha256(canonical(value)).hexdigest()


def finite_diagnostics(value):
    if isinstance(value, dict):
        return {k: finite_diagnostics(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_diagnostics(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@contextmanager
def far_adapter(arm, trace):
    """Patch ONLY the combustor module's imported name, restore even on failure."""
    require(arm in PARAMETERS, "Unknown arm")
    require(_ADAPTER_LOCK.acquire(blocking=False), "Concurrent FAR adapter forbidden")
    original = combustor.brentq
    def instrument(function, lower, upper, **kwargs):
        require(kwargs == {**PARAMETERS["A"], "full_output": True, "disp": False}, "Unexpected FAR solver contract")
        event = {"lower_far": float(lower), "upper_far": float(upper), **PARAMETERS[arm],
                 "converged": None, "iterations": None, "function_calls": 0, "root": None, "exception": None}
        trace.append(event)
        def counted(x):
            event["function_calls"] += 1
            return function(x)
        try:
            value, info = original(counted, lower, upper, **{**kwargs, **PARAMETERS[arm]})
            event.update(root=float(value), converged=bool(info.converged), iterations=int(info.iterations))
            return value, info
        except Exception as exc:
            event["exception"] = type(exc).__name__
            raise
    combustor.brentq = instrument
    try:
        yield
    finally:
        combustor.brentq = original
        _ADAPTER_LOCK.release()


def load_inputs(root):
    _, coordinates, _, context, parent_identity = verify_inputs(
        root, root / "config/experiments/geometry19_performance_gated_g0c_physics_v1.yaml")
    ledger = root / BASE / "training_physics/g0c_v1/Sealed_Evaluations.jsonl"
    require(file_hash(ledger) == LEDGER_HASH, "Frozen acquisition ledger changed")
    require(file_hash(root / "src/physics/combustor_map_enhanced.py") == COMBUSTOR_HASH, "Combustor source changed")
    samples = [r for r in coordinates if r["root_seed_id"] == ROOT_ID]
    baseline = [r for r in read_rows(ledger) if r["root_seed_id"] == ROOT_ID]
    require(len(samples) == len(baseline) == 512, "Wrong sample count")
    require(Counter(r["split"] for r in samples) == {"train": 408, "validation": 104}, "Split mismatch")
    require([r["design_hash"] for r in samples] == [r["design_hash"] for r in baseline], "Baseline order mismatch")
    require(all(all(old[k] == v for k, v in source.items() if k != "label_status")
                for source, old in zip(samples, baseline)), "Baseline coordinates/split mismatch")
    implementation = {"module": file_hash(Path(__file__)),
                      "cli": file_hash(root / "scripts/run_geometry19_far_precision_diagnostic.py")}
    arms = {arm: {"parameters": parameters, "parent_context_sha256": context.context_hash,
                  "parent_bundle_sha256": context.physics_runtime.physics_bundle_hash,
                  "combustor_source_sha256": COMBUSTOR_HASH, "adapter_sha256": implementation["module"]}
            for arm, parameters in PARAMETERS.items()}
    for arm, identity in arms.items():
        identity["effective_implementation_sha256"] = digest({"arm": arm, **identity})
    lock = {"version": "geometry19_g0c_far_precision_diagnostic_v1", "plan_sha256": file_hash(root / PLAN),
            "implementation": implementation, "parent_identity": parent_identity, "ledger_sha256": LEDGER_HASH,
            "samples_sha256": digest(samples), "baseline_projection_sha256": digest(baseline),
            "runtime_contract_sha256": digest(context.contract_mapping()), "arms": arms,
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "budget": {"A": 512, "B": 512, "total": 1024}, "physics_execution_authorized": False,
            "training_authorized": False, "post_selection_diagnostic_only": True}
    return samples, baseline, context, lock


def prepare(root):
    samples, _, context, lock = load_inputs(root)
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=False)
    write_once(output / "Sample_Registry.jsonl", b"".join(canonical(r) + b"\n" for r in samples))
    write_json(output / "Parent_Runtime.json", context.contract_mapping())
    write_json(output / "Protocol_Lock.json", lock)
    return lock


class CaptureRuntime:
    def __init__(self, runtime):
        self.runtime, self.result = runtime, None

    def __getattr__(self, key):
        return getattr(self.runtime, key)

    def evaluate(self, *args, **kwargs):
        self.result = self.runtime.evaluate(*args, **kwargs)
        return self.result


def physical_evaluator(source, arm, context, identity):
    capture, trace = CaptureRuntime(context.physics_runtime), []
    proxy = SimpleNamespace(physics_runtime=capture, context_hash=context.context_hash, target_domain=context.target_domain)
    with far_adapter(arm, trace):
        result = _evaluate_row(deepcopy(source), proxy)
    result["label_status"] = "V2_2_EVALUATED"
    result.update(eligibility_flags(result))
    if arm == "B":
        # Do not label altered physics as the unmodified parent runtime.
        result["parent_pipeline_context_hash"] = result["pipeline_context_hash"]
        result["parent_physics_bundle_hash"] = result["physics_bundle_hash"]
        result["parent_physics_model_version"] = result["physics_model_version"]
        result["pipeline_context_hash"] = digest({"kind": "diagnostic_context", **identity})
        result["physics_bundle_hash"] = identity["effective_implementation_sha256"]
        result["physics_model_version"] = "geometry19_far_precision_diagnostic_v1_B_not_training_labels"
    op = capture.result.operating_point
    diagnostics = {k: op.get(k) for k in ("fuel_air_ratio", "mass_flow_air_kg_s", "mass_flow_fuel_kg_s")}
    diagnostics.update(far_calls=trace, far_solver_invocations=len(trace),
                       far_function_evaluations=sum(e["function_calls"] for e in trace))
    return {"arm": arm, "arm_identity": identity, "evaluation": result, "diagnostics": finite_diagnostics(diagnostics)}


def validate_record(record, source, arm, identity):
    require(record["arm"] == arm and record["arm_identity"] == identity, "Arm identity mismatch")
    result = record["evaluation"]
    require(all(result.get(k) == v for k, v in source.items() if k != "label_status"), "Frozen coordinate/split changed")
    require(result["label_status"] == "V2_2_EVALUATED", "Label status changed")
    require(all(result.get(k) == v for k, v in eligibility_flags(result).items()), "Eligibility changed")
    expected_context = identity["parent_context_sha256"] if arm == "A" else digest({"kind": "diagnostic_context", **identity})
    expected_bundle = identity["parent_bundle_sha256"] if arm == "A" else identity["effective_implementation_sha256"]
    require(result["pipeline_context_hash"] == expected_context and result["physics_bundle_hash"] == expected_bundle,
            "Effective runtime identity mismatch")


def sealed_json(path, value):
    if path.exists():
        require(path.read_bytes() == canonical(value) + b"\n", f"Sealed artifact changed: {path.name}")
    else:
        write_json(path, value)


def run_arm(output, arm, samples, identity, execution_hash, evaluate):
    records, seals = [], []
    for band in range(4):
        batch = samples[band * 128:(band + 1) * 128]
        prefix = output / f"{arm}-band-{band}"
        data, journal, seal_path = [prefix.with_suffix(s) for s in (".jsonl", ".calls.jsonl", ".seal.json")]
        expected = {"arm": identity, "execution_sha256": execution_hash, "inputs_sha256": digest(batch)}
        exists = sum(p.exists() for p in (data, journal, seal_path))
        require(exists in (0, 3), "Orphan partial shard; preserve and require explicit recovery")
        calls = [{"design_hash": r["design_hash"], "arm": arm, "event": "whole_engine_call_started"} for r in batch]
        if exists:
            seal = json.loads(seal_path.read_text())
            require(seal["identity"] == expected and seal["output_sha256"] == file_hash(data) and
                    seal["journal_sha256"] == file_hash(journal), "Shard hash/identity mismatch")
            require(read_rows(journal) == calls and seal["actual_calls"] == 128, "Call budget mismatch")
            result = read_rows(data)
        else:
            result = []
            with data.open("xb") as stream, journal.open("xb") as call_file:
                for source, event in zip(batch, calls):
                    call_file.write(canonical(event) + b"\n")
                    call_file.flush()
                    os.fsync(call_file.fileno())
                    record = evaluate(deepcopy(source), arm, identity)
                    validate_record(record, source, arm, identity)
                    stream.write(canonical(record) + b"\n")
                    stream.flush()
                    result.append(record)
                os.fsync(stream.fileno())
            seal = {"identity": expected, "actual_calls": 128, "infrastructure_replays": 0,
                    "output_sha256": file_hash(data), "journal_sha256": file_hash(journal)}
            write_json(seal_path, seal)
        require(len(result) == 128, "Shard count mismatch")
        for source, record in zip(batch, result):
            validate_record(record, source, arm, identity)
        records.extend(result)
        seals.append(seal)
        print(json.dumps({"arm": arm, "sealed_rows": len(records)}), flush=True)
    sealed_json(output / f"{arm}_Manifest.json", {"shards": seals, "records_sha256": digest(records), "actual_calls": 512})
    return records


def comparison(a_records, b_records):
    a = [r["evaluation"] for r in a_records]
    b = [r["evaluation"] for r in b_records]
    def energy(r):
        value = (r.get("conservation") or {}).get("combustor_energy_W")
        return abs(value) if type(value) in (int, float) and math.isfinite(value) else None
    def closed(r):
        value = energy(r)
        return value is not None and value <= 1e-6
    def domain(r):
        d = r.get("target_domain_assessment")
        return "unassessed" if d is None else "inside" if d["inside"] else "outside"
    originally_bad = [i for i, r in enumerate(a) if not closed(r)]
    recovered = sum(closed(b[i]) for i in originally_bad)
    regressions = [r["design_hash"] for old, r in zip(a, b) if closed(old) and not closed(r)]
    new_failures = [r["design_hash"] for old, r in zip(a, b) if
                    (old["solver"]["converged"] and not r["solver"]["converged"]) or
                    (old["constraint_assessment"]["numerically_valid"] and not r["constraint_assessment"]["numerically_valid"]) or
                    (old["constraint_assessment"]["physics_domain_valid"] and not r["constraint_assessment"]["physics_domain_valid"])]
    remaining_invalid = [r["design_hash"] for r in b if not (r["solver"]["converged"] and
                         r["constraint_assessment"]["numerically_valid"] and r["constraint_assessment"]["physics_domain_valid"])]
    changes = []
    for old, new in zip(a, b):
        relative = {}
        for k in OUTPUTS:
            x, y = (old.get("performance") or {}).get(k), (new.get("performance") or {}).get(k)
            relative[k] = abs(y - x) / abs(x) if type(x) in (int, float) and type(y) in (int, float) and x > 0 and math.isfinite(x) and math.isfinite(y) else None
        if any(v is None or v > 1e-6 for v in relative.values()):
            changes.append({"design_hash": old["design_hash"], "relative_changes": relative})
    counts = {}
    for arm, rows in (("A", a), ("B", b)):
        bands = [{"band": band, "train": sum(r["training_eligible"] for r in rows if r["band_index"] == band),
                  "validation": sum(r["validation_eligible"] for r in rows if r["band_index"] == band)} for band in range(4)]
        train, validation = sum(r["training_eligible"] for r in rows), sum(r["validation_eligible"] for r in rows)
        residuals = [energy(r) for r in rows if energy(r) is not None]
        counts[arm] = {"train": train, "validation": validation, "bands": bands,
                       "root_and_band_gate_pass": train >= 256 and validation >= 64 and all(v["train"] >= 32 and v["validation"] >= 8 for v in bands),
                       "absolute_energy_quantiles": np.quantile(residuals, [0, .5, .95, 1]).tolist() if residuals else None}
    return {"status": "DIAGNOSTIC_COMPLETE_NOT_TRAINING_AUTHORIZATION", "original_energy_failures": len(originally_bad),
            "recovered_energy_rows": recovered, "energy_regressions": regressions, "new_validity_failures": new_failures,
            "remaining_b_numerical_or_domain_failures": remaining_invalid,
            "numerical_issue_resolved": bool(originally_bad) and recovered == len(originally_bad) and not regressions and not new_failures and not remaining_invalid,
            "performance_stability_pass": not changes, "performance_alerts": changes,
            "target_transitions": dict(Counter(f"{domain(old)}->{domain(new)}" for old, new in zip(a, b))),
            "counts": counts, "per_row_energy_changes": [{"design_hash": old["design_hash"], "A_abs_W": energy(old), "B_abs_W": energy(new)} for old, new in zip(a, b)],
            "training_authorized": False, "old_g0c_status": "G0C_DATA_GATE_FAIL"}


def execute_locked(output, lock, samples, baseline, evaluate, *, authorize=False, resume=False):
    """Core transaction; production caller must verify inputs before entry."""
    require(authorize != resume, "Choose explicit authorization OR resume")
    require(len(samples) == len(baseline) == 512 and digest(samples) == lock["samples_sha256"] and
            digest(baseline) == lock["baseline_projection_sha256"], "Samples/baseline mismatch")
    require(json.loads((output / "Protocol_Lock.json").read_text()) == lock, "Lock changed")
    require(read_rows(output / "Sample_Registry.jsonl") == samples, "Sample registry changed")
    with (output / ".runner.lock").open("a") as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execution = {"lock_sha256": file_hash(output / "Protocol_Lock.json"), "physics_execution_authorized": True,
                     "budget": {"A": 512, "B": 512, "total": 1024}, "training_authorized": False}
        ep = output / "Execution_Manifest.json"
        initial = {"Protocol_Lock.json", "Sample_Registry.jsonl", "Parent_Runtime.json", ".runner.lock"}
        shard_names = {f"{arm}-band-{band}{suffix}" for arm in ("A", "B") for band in range(4)
                       for suffix in (".jsonl", ".calls.jsonl", ".seal.json")}
        allowed = initial | shard_names | {"Execution_Manifest.json", "A_Manifest.json", "B_Manifest.json",
                                          "A_Reproduction_Gate.json", "Complete_Manifest.json", "Summary.json"}
        require({p.name for p in output.iterdir()} <= (initial if authorize else allowed), "Unexpected existing execution artifacts")
        if authorize:
            write_json(ep, execution)
        else:
            require(json.loads(ep.read_text()) == execution, "Execution authorization/identity changed")
        if (output / "Complete_Manifest.json").exists():
            require(all((output / n).is_file() for n in ("A_Manifest.json", "B_Manifest.json", "A_Reproduction_Gate.json")),
                    "Completed run missing seals; no replay")
        if (output / "Summary.json").exists():
            saved = json.loads((output / "Summary.json").read_text())
            require((output / "A_Manifest.json").is_file() and (output / "A_Reproduction_Gate.json").is_file(), "Summary missing A seals")
            if saved["status"] != "A_REPRODUCTION_FAIL_B_NOT_STARTED":
                require((output / "Complete_Manifest.json").is_file(), "Summary missing complete seal")
        if any((output / n).exists() for n in shard_names if n.startswith("B-")):
            require((output / "A_Manifest.json").is_file() and (output / "A_Reproduction_Gate.json").is_file(), "B exists before A seal")
            require(json.loads((output / "A_Reproduction_Gate.json").read_text())["baseline_reproduced"] is True, "B exists after failed A")
        # Preflight all partial files before any new call, including later B shards.
        for arm in ("A", "B"):
            for band in range(4):
                paths = [(output / f"{arm}-band-{band}").with_suffix(s) for s in (".jsonl", ".calls.jsonl", ".seal.json")]
                require(sum(p.exists() for p in paths) in (0, 3), "Orphan partial shard; explicit recovery required")
                if (output / f"{arm}_Manifest.json").exists():
                    require(all(p.exists() for p in paths), "Sealed arm missing shard; no replay")
        a = run_arm(output, "A", samples, lock["arms"]["A"], file_hash(ep), evaluate)
        differences = [{"design_hash": old["design_hash"], "fields": [k for k in old if record["evaluation"].get(k) != old[k]]}
                       for record, old in zip(a, baseline) if any(record["evaluation"].get(k) != old[k] for k in old)]
        gate = {"baseline_reproduced": not differences, "differences": differences, "baseline_sha256": digest(baseline)}
        sealed_json(output / "A_Reproduction_Gate.json", gate)
        if differences:
            failure = {"status": "A_REPRODUCTION_FAIL_B_NOT_STARTED", "actual_calls": 512, "training_authorized": False}
            sealed_json(output / "Summary.json", failure)
            return failure
        b = run_arm(output, "B", samples, lock["arms"]["B"], file_hash(ep), evaluate)
        sealed_json(output / "Complete_Manifest.json", {"A_sha256": file_hash(output / "A_Manifest.json"),
                    "B_sha256": file_hash(output / "B_Manifest.json"), "actual_calls": 1024, "infrastructure_replays": 0})
        summary = comparison(a, b)
        summary.update(actual_calls=1024, complete_manifest_sha256=file_hash(output / "Complete_Manifest.json"))
        sealed_json(output / "Summary.json", summary)
        return summary


def execute(root, *, authorize=False, resume=False):
    samples, baseline, context, lock = load_inputs(root)
    output = root / OUTPUT
    require((output / "Parent_Runtime.json").read_bytes() == canonical(context.contract_mapping()) + b"\n", "Runtime snapshot changed")
    return execute_locked(output, lock, samples, baseline,
                          lambda row, arm, identity: physical_evaluator(row, arm, context, identity),
                          authorize=authorize, resume=resume)
