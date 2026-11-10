"""Staged, immutable precision regression and uniform label migration."""
from collections import Counter
from copy import deepcopy
import fcntl
import json
import math
from pathlib import Path

import yaml

from src.ai import geometry19_far_precision_diagnostic as diagnostic
from src.ai.geometry19_performance_gated_protocol import canonical, file_hash, read_rows, require, within, OUTPUTS
from src.ai.geometry19_performance_gated_physics import verify_inputs, acquire_shard, check_row, data_gates, write_json, write_once
from src.ai.geometry19_performance_gated_physics_protocol import eligibility_flags
from src.physics.geometry19_v2_2_dataset import _evaluate_row
from src.physics.geometry19_far_precision_v1 import build_candidate, tracked_parent

CONFIG = "config/experiments/geometry19_far_precision_relabel_v1.yaml"
OUTPUT = "data/validation/v2_2_far_precision_relabel_v1"
EXEMPT = {"physics_model_version", "physics_bundle_hash", "pipeline_context_hash",
          "parent_physics_model_version", "parent_physics_bundle_hash", "parent_pipeline_context_hash", "far_call_counts"}


def exact_difference(expected, actual, *, allow_provenance=False):
    require(len(expected) == len(actual), "Comparison count mismatch")
    ignored = EXEMPT if allow_provenance else {"far_call_counts"}
    return [{"design_hash": old["design_hash"], "fields": [k for k in old if k not in ignored and old[k] != new.get(k)]}
            for old, new in zip(expected, actual) if any(old[k] != new.get(k) for k in old if k not in ignored)]


def migration_gate(old, new):
    require(len(old) == len(new) and [r["design_hash"] for r in old] == [r["design_hash"] for r in new], "Migration order mismatch")
    violations, incomparable = [], []
    maxima = {k: 0.0 for k in OUTPUTS}
    transitions = Counter()
    recovered = 0
    def valid(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    def domain(row):
        value = row.get("target_domain_assessment")
        return "unassessed" if value is None else "inside" if value["inside"] else "outside"
    for a, b in zip(old, new):
        reasons = []
        if a["solver"]["converged"] and not b["solver"]["converged"]:
            reasons.append("new_solver_failure")
        for key in ("numerically_valid", "physics_domain_valid"):
            if a["constraint_assessment"][key] and not b["constraint_assessment"][key]:
                reasons.append("new_" + key + "_failure")
        if a["physics_eligible"] and not b["physics_eligible"]:
            reasons.append("lost_eligibility")
        if b["solver"]["converged"]:
            residual = (b.get("conservation") or {}).get("combustor_energy_W")
            if type(residual) not in (int, float) or not math.isfinite(residual) or abs(residual) > 1e-6:
                reasons.append("candidate_energy_closure")
        for key in OUTPUTS:
            x, y = (a.get("performance") or {}).get(key), (b.get("performance") or {}).get(key)
            if not valid(x):
                incomparable.append({"design_hash": a["design_hash"], "axis": key})
            elif not valid(y):
                reasons.append("missing_candidate_" + key)
            else:
                change = abs(y-x)/abs(x)
                maxima[key] = max(maxima[key], change)
                if change > 1e-6:
                    reasons.append("performance_change_" + key)
        transitions[f"{domain(a)}->{domain(b)}"] += 1
        recovered += int(not a["physics_eligible"] and b["physics_eligible"])
        if reasons:
            violations.append({"design_hash": a["design_hash"], "reasons": reasons})
    return {"pass": not violations, "violations": violations, "incomparable": incomparable,
            "maximum_relative_changes": maxima, "target_transitions": dict(transitions), "newly_eligible": recovered}


def verify_diagnostic(root):
    """Read-only traversal of the frozen diagnostic hierarchy; never invokes its runner."""
    samples, baseline, context, lock = diagnostic.load_inputs(root)
    out = root / diagnostic.OUTPUT
    require((out / "Protocol_Lock.json").read_bytes() == canonical(lock)+b"\n", "Diagnostic lock changed")
    require(read_rows(out / "Sample_Registry.jsonl") == samples, "Diagnostic samples changed")
    require((out / "Parent_Runtime.json").read_bytes() == canonical(context.contract_mapping())+b"\n", "Diagnostic runtime changed")
    execution = json.loads((out / "Execution_Manifest.json").read_text())
    require(execution == {"lock_sha256": file_hash(out / "Protocol_Lock.json"), "physics_execution_authorized": True,
                          "budget": {"A": 512, "B": 512, "total": 1024}, "training_authorized": False}, "Diagnostic execution changed")
    arms = {}
    for arm in ("A", "B"):
        records, seals = [], []
        for band in range(4):
            base = out / f"{arm}-band-{band}"
            data, journal, seal_path = [base.with_suffix(s) for s in (".jsonl", ".calls.jsonl", ".seal.json")]
            seal = json.loads(seal_path.read_text())
            batch = samples[band*128:(band+1)*128]
            require(seal == {"identity": {"arm": lock["arms"][arm], "execution_sha256": file_hash(out / "Execution_Manifest.json"),
                           "inputs_sha256": diagnostic.digest(batch)}, "actual_calls": 128, "infrastructure_replays": 0,
                           "output_sha256": file_hash(data), "journal_sha256": file_hash(journal)}, "Diagnostic shard changed")
            require(read_rows(journal) == [{"design_hash": r["design_hash"], "arm": arm, "event": "whole_engine_call_started"} for r in batch], "Diagnostic calls changed")
            part = read_rows(data)
            require(len(part) == 128, "Diagnostic count changed")
            for source, record in zip(batch, part):
                diagnostic.validate_record(record, source, arm, lock["arms"][arm])
            records.extend(part)
            seals.append(seal)
        require(json.loads((out / f"{arm}_Manifest.json").read_text()) == {"shards": seals, "records_sha256": diagnostic.digest(records), "actual_calls": 512}, "Diagnostic arm manifest changed")
        arms[arm] = records
    differences = exact_difference(baseline, [r["evaluation"] for r in arms["A"]])
    require(not differences, "Diagnostic A no longer reproduces baseline")
    require(json.loads((out / "A_Reproduction_Gate.json").read_text()) == {"baseline_reproduced": True, "differences": [],
                    "baseline_sha256": diagnostic.digest(baseline)}, "Diagnostic reproduction gate changed")
    require(json.loads((out / "Complete_Manifest.json").read_text()) == {"A_sha256": file_hash(out / "A_Manifest.json"),
                    "B_sha256": file_hash(out / "B_Manifest.json"), "actual_calls": 1024, "infrastructure_replays": 0}, "Diagnostic complete manifest changed")
    summary = diagnostic.comparison(arms["A"], arms["B"])
    summary.update(actual_calls=1024, complete_manifest_sha256=file_hash(out / "Complete_Manifest.json"))
    require(json.loads((out / "Summary.json").read_text()) == summary and summary["numerical_issue_resolved"] and summary["performance_stability_pass"], "Diagnostic summary invalid")
    return [r["evaluation"] for r in arms["B"]]


def load(root):
    config_path = root / CONFIG
    c = yaml.safe_load(config_path.read_text())
    require(c["candidate"]["far_parameters"] == {"xtol": 1e-15, "rtol": 1e-15, "maxiter": 64}, "Candidate parameters changed")
    require(c["budget"] == {"regression_r1": 512, "regression_r2_A": 128, "regression_r2_C": 128,
        "full_relabel": 16896, "maximum_new_whole_engine_calls": 17664, "prior_diagnostic_1024_calls": "excluded_already_completed",
        "far_internal_calls": "counted_separately_not_whole_engine_calls", "unregistered_physics_smokes_or_test_calls": "forbidden"}, "Budget changed")
    for role, spec in c["inputs"].items():
        require(file_hash(within(root, spec["path"])) == spec["sha256"], f"Input changed: {role}")
    _, rows, shards, parent, parent_identity = verify_inputs(root, root / "config/experiments/geometry19_performance_gated_g0c_physics_v1.yaml")
    old = read_rows(root / c["inputs"]["original_labels"]["path"])
    require(len(rows) == len(old) == 16896, "Full population changed")
    for source, result in zip(rows, old):
        check_row(source, result, parent)
    # Follow pinned acquisition seal into every original shard and call journal.
    old_out = root / diagnostic.BASE / "training_physics/g0c_v1"
    require(file_hash(old_out / "Acquisition_Manifest.json") == "21283d781ef78ebaf4632acdf6c90d9b549aaee27514cd275c1661db49f748e7", "Original acquisition manifest changed")
    original_manifest = json.loads((old_out / "Acquisition_Manifest.json").read_text())
    require(file_hash(old_out / "execution_manifest.json") == original_manifest["identity"]["execution_manifest_sha256"], "Original execution changed")
    require(len(original_manifest["shards"]) == len(shards), "Original shard count changed")
    for shard, expected in zip(shards, original_manifest["shards"]):
        path = old_out / shard["future_output"]
        require(json.loads(path.with_suffix(".seal.json").read_text()) == expected and
                file_hash(path) == expected["shard_output_sha256"] and
                file_hash(path.with_suffix(".calls.jsonl")) == expected["call_journal_sha256"], "Original shard integrity failed")
    b = verify_diagnostic(root)
    candidate, counters, implementation = build_candidate(parent, root)
    r1 = [r for r in rows if r["root_seed_id"] == diagnostic.ROOT_ID]
    r2 = [r for r in rows if r["root_seed_id"] != diagnostic.ROOT_ID and r["pair_index"] == 1 and r["sign"] == 1]
    require(len(r1) == 512 and len(r2) == 128 and len({(r["root_seed_id"], r["band_index"]) for r in r2}) == 128 and
            len({r["root_seed_id"] for r in r2}) == 32 and all(r["split"] == "train" for r in r2), "Regression registry invalid")
    implementation.update(runner_sha256=file_hash(Path(__file__)), cli_sha256=file_hash(root / "scripts/run_geometry19_far_precision_relabel.py"))
    artifacts = {"Implementation_Manifest.json": canonical(implementation)+b"\n",
                 "Regression_Registry.jsonl": b"".join(canonical({"stage": name, **r})+b"\n" for name, group in (("R1", r1), ("R2", r2)) for r in group),
                 "Coordinate_Registry.jsonl": b"".join(canonical(r)+b"\n" for r in rows),
                 "Runtime_Contract.json": canonical(candidate.contract_mapping())+b"\n"}
    lock = {"version": c["protocol_version"], "config_sha256": file_hash(config_path), "plan_sha256": file_hash(config_path.with_suffix(".md")),
            "inputs": c["inputs"], "parent_identity": parent_identity, "implementation": implementation,
            "artifacts": {k: diagnostic.sha256(v).hexdigest() for k,v in artifacts.items()},
            "candidate_context": candidate.context_hash, "candidate_bundle": candidate.physics_runtime.physics_bundle_hash,
            "regression_authorized": False, "relabel_authorized": False, "training_authorized": False}
    return SimpleInputs(c, rows, old, b, r1, r2, parent, candidate, counters, lock, artifacts)


class SimpleInputs:
    def __init__(self, config, rows, old, b, r1, r2, parent, candidate, counters, lock, artifacts):
        self.__dict__.update(locals())
        self.__dict__.pop("self")


def prepare(root):
    inputs = load(root)
    out = root / OUTPUT
    out.mkdir(parents=True, exist_ok=False)
    for name, payload in inputs.artifacts.items():
        write_once(out / name, payload)
    write_json(out / "Protocol_Lock.json", inputs.lock)
    return {"status": "PRECISION_RELABEL_PREPARED", "rows": len(inputs.rows), "regression_calls": 768, "physics_calls": 0}


def verify_prepared(out, inputs):
    require((out / "Protocol_Lock.json").read_bytes() == canonical(inputs.lock)+b"\n", "Preparation lock changed")
    for name, payload in inputs.artifacts.items():
        require((out / name).read_bytes() == payload, f"Prepared artifact changed: {name}")


def stage(out, name, rows, context, execution_hash, evaluate, *, resume_only=False):
    directory = out / name
    if not directory.exists():
        require(not resume_only, "Completed stage missing; no replay")
        directory.mkdir()
        (directory / "shards").mkdir()
    if resume_only:
        require((directory / "Manifest.json").exists(), "Completed stage manifest missing; no replay")
    expected_files = set()
    for i in range(len(rows)//128):
        p = directory / f"shards/part-{i:05d}.jsonl"
        paths = (p, p.with_suffix(".seal.json"), p.with_suffix(".calls.jsonl"))
        expected_files.update(paths)
        require(sum(p.exists() for p in paths) in (0, 3), "Orphan partial shard; preserve and stop")
        if resume_only or (directory / "Manifest.json").exists() or (directory / "Sealed_Evaluations.jsonl").exists():
            require(all(p.exists() for p in paths), "Sealed stage missing shard; no replay")
    require(set((directory / "shards").iterdir()) <= expected_files, "Unexpected shard files")
    results, seals = [], []
    for start in range(0, len(rows), 128):
        batch = rows[start:start+128]
        spec = {"future_output": f"shards/part-{start//128:05d}.jsonl", "shard_input_sha256": diagnostic.digest(batch)}
        identity = {"execution_sha256": execution_hash, "stage": name, "context": context.context_hash,
                    "bundle": context.physics_runtime.physics_bundle_hash}
        part, seal = acquire_shard(directory, spec, batch, context, identity, evaluator=evaluate)
        results.extend(part)
        seals.append(seal)
        print(json.dumps({"stage": name, "sealed_rows": len(results), "total": len(rows)}), flush=True)
    ledger = directory / "Sealed_Evaluations.jsonl"
    payload = b"".join(canonical(r)+b"\n" for r in results)
    if ledger.exists():
        require(ledger.read_bytes() == payload, "Stage ledger changed")
    else:
        require(not resume_only and not (directory / "Manifest.json").exists(), "Sealed ledger missing")
        write_once(ledger, payload)
    manifest = {"stage": name, "execution_sha256": execution_hash, "ledger_sha256": file_hash(ledger),
                "rows": len(rows), "actual_calls": len(rows), "infrastructure_replays": 0, "shards": seals,
                "far_call_counts": {k: sum(r.get("far_call_counts", {}).get(k, 0) for r in results)
                                    for k in ("far_solver_invocations", "far_function_evaluations")}}
    diagnostic.sealed_json(directory / "Manifest.json", manifest)
    return results


def evaluator(context, counters):
    def evaluate(row, _context):
        for k in counters:
            counters[k] = 0
        result = _evaluate_row(deepcopy(row), context)
        result["far_call_counts"] = dict(counters)
        return result
    return evaluate


def execute(root, mode, *, resume=False):
    require(mode in {"regression", "relabel"}, "Unknown execution stage")
    inputs = load(root)
    out = root / OUTPUT
    verify_prepared(out, inputs)
    with (out / ".runner.lock").open("a") as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return execute_stages(out, inputs, mode, resume=resume)


def execute_stages(out, inputs, mode, *, resume=False, candidate_eval=None, parent_eval=None):
    """Production entry verifies complete input graph; tests may supply synthetic evaluators."""
    # Check every already-created stage before permitting any new evaluation.
    for name, count in (("R1", 4), ("R2_A", 1), ("R2_C", 1), ("L", 132)):
        directory = out / name
        if directory.exists():
            require((directory / "shards").is_dir(), "Partial stage directory; explicit recovery required")
            for i in range(count):
                path = directory / f"shards/part-{i:05d}.jsonl"
                triple = (path, path.with_suffix(".seal.json"), path.with_suffix(".calls.jsonl"))
                require(sum(p.exists() for p in triple) in (0, 3), "Orphan partial shard; explicit recovery required")
                if (directory / "Manifest.json").exists() or (directory / "Sealed_Evaluations.jsonl").exists():
                    require(all(p.exists() for p in triple), "Sealed stage missing shards; no replay")
    if (out / "R2_A").exists():
        require((out / "R1_Gate.json").exists() and json.loads((out / "R1_Gate.json").read_text())["pass"], "R2 exists before R1 pass")
    if (out / "R2_C").exists():
        require((out / "R2_A_Gate.json").exists() and json.loads((out / "R2_A_Gate.json").read_text())["pass"], "R2 C exists before A pass")
    old_by_hash = {r["design_hash"]: r for r in inputs.old}
    r2_old = [old_by_hash[r["design_hash"]] for r in inputs.r2]
    parent, parent_counters = tracked_parent(inputs.parent)
    candidate_eval = candidate_eval or evaluator(inputs.candidate, inputs.counters)
    parent_eval = parent_eval or evaluator(parent, parent_counters)
    def verify_stage(name, rows, context, evaluate, execution_hash, complete=False):
        return stage(out, name, rows, context, execution_hash, evaluate, resume_only=complete)
    def regression_checks(execution_hash, complete=False):
        r1 = verify_stage("R1", inputs.r1, inputs.candidate, candidate_eval, execution_hash, complete)
        differences = exact_difference(inputs.b, r1, allow_provenance=True)
        diagnostic.sealed_json(out / "R1_Gate.json", {"pass": not differences, "differences": differences})
        if differences:
            return {"status": "REGRESSION_FAIL_R1", "actual_calls": 512, "training_authorized": False}
        a = verify_stage("R2_A", inputs.r2, parent, parent_eval, execution_hash, complete)
        differences = exact_difference(r2_old, a)
        diagnostic.sealed_json(out / "R2_A_Gate.json", {"pass": not differences, "differences": differences})
        if differences:
            return {"status": "REGRESSION_FAIL_R2_A", "actual_calls": 640, "training_authorized": False}
        c = verify_stage("R2_C", inputs.r2, inputs.candidate, candidate_eval, execution_hash, complete)
        gate = migration_gate(r2_old, c)
        diagnostic.sealed_json(out / "R2_C_Gate.json", gate)
        return {"status": "REGRESSION_PASS" if gate["pass"] else "REGRESSION_FAIL_R2_C", "actual_calls": 768,
                "training_authorized": False, "migration_gate": gate}
    ep = out / ("Regression_Execution_Manifest.json" if mode == "regression" else "Relabel_Execution_Manifest.json")
    if mode == "relabel":
        rp = out / "Regression_Summary.json"
        require(rp.exists() and json.loads(rp.read_text())["status"] == "REGRESSION_PASS", "Regression not passed; relabel forbidden")
        require(all((out/n/"Manifest.json").exists() for n in ("R1", "R2_A", "R2_C")), "Regression seals missing")
        verified = regression_checks(file_hash(out / "Regression_Execution_Manifest.json"), complete=True)
        require(rp.read_bytes() == canonical(verified)+b"\n", "Regression summary changed")
    execution = {"mode": mode, "authorized": True, "preparation_sha256": file_hash(out / "Protocol_Lock.json"),
                 "budget": 768 if mode == "regression" else 16896, "training_authorized": False}
    if mode == "relabel":
        execution["regression_summary_sha256"] = file_hash(out / "Regression_Summary.json")
    if resume:
        require(ep.read_bytes() == canonical(execution)+b"\n", "Execution identity changed")
    else:
        require(not any((out/n).exists() for n in (("R1", "R2_A", "R2_C", "Regression_Summary.json") if mode == "regression" else ("L", "Relabel_Summary.json"))), "Execution artifacts already exist")
        write_json(ep, execution)
    if mode == "regression":
        complete = (out / "Regression_Summary.json").exists()
        if complete:
            status = json.loads((out / "Regression_Summary.json").read_text())["status"]
            required = ["R1"] if status == "REGRESSION_FAIL_R1" else ["R1", "R2_A"] if status == "REGRESSION_FAIL_R2_A" else ["R1", "R2_A", "R2_C"]
            require(all((out/n/"Manifest.json").exists() for n in required), "Completed regression seals missing")
        summary = regression_checks(file_hash(ep), complete=complete)
        diagnostic.sealed_json(out / "Regression_Summary.json", summary)
    else:
        results = stage(out, "L", inputs.rows, inputs.candidate, file_hash(ep), candidate_eval,
                        resume_only=(out / "Relabel_Summary.json").exists())
        # Both ledger and manifest exist before any aggregate qualification.
        gate = migration_gate(inputs.old, results)
        summary = data_gates(results, inputs.config["relabel"]["gates"])
        summary.update(status="RELABEL_PASS" if gate["pass"] and not summary["failures"] else "RELABEL_FAIL",
                       migration_gate=gate, actual_calls=16896, total_new_calls_including_regression=17664,
                       ledger_sha256=file_hash(out / "L/Sealed_Evaluations.jsonl"), manifest_sha256=file_hash(out / "L/Manifest.json"),
                       training_authorized=False, old_g0c_status="G0C_DATA_GATE_FAIL")
        summary["next_stage"] = "separate_training_protocol_and_cross_version_compatibility_audit" if summary["status"] == "RELABEL_PASS" else "seal_failure_and_stop_no_training"
        diagnostic.sealed_json(out / "Relabel_Summary.json", summary)
    return summary
