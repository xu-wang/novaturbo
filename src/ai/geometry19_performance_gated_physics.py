"""Frozen G0c acquisition. No training, prediction, resampling or solver retries."""
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import fcntl
import json
import os
from pathlib import Path
import time

import yaml

from src.ai.geometry19_performance_gated_protocol import canonical, file_hash, read_rows, require, within
from src.ai.geometry19_performance_gated_physics_protocol import validate_protocol, eligibility_flags, make_shards
from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.physics.geometry19_v2_2_dataset import _evaluate_row

LOCK_HASH = "d0e5e4a3c941132fe4a4b9a3846bed92722f05beb1d05ee6d805eb7756db671e"


def write_once(path, payload):
    """Exclusive, durable write; incomplete files are never silently replaced."""
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path, value):
    write_once(path, canonical(value) + b"\n")


def verify_inputs(root, config_path):
    c = yaml.safe_load(config_path.read_text())
    validate_protocol(c)
    base = within(root, c["artifacts"]["output_root"])
    lock_path = base / c["artifacts"]["lock"]
    require(file_hash(lock_path) == LOCK_HASH, "G0c lock changed")
    lock = json.loads(lock_path.read_text())
    require(file_hash(config_path) == lock["protocol_sha256"], "G0c protocol changed")
    for name, field in (("physics_protocol", "implementation_sha256"),
                        ("coordinates", "g0b_implementation_sha256"), ("protocol", "g0a_implementation_sha256")):
        require(file_hash(root / f"src/ai/geometry19_performance_gated_{name}.py") == lock[field], "Frozen code changed")
    def verify(spec):
        path = within(root, spec["path"])
        require(file_hash(path) == spec["sha256"], f"Input changed: {path}")
        return path
    paths = {k: verify(v) for k, v in c["inputs"].items()}
    b = yaml.safe_load(paths["coordinate_protocol"].read_text())
    parent_path = verify(b["parent_protocol"])
    parent = yaml.safe_load(parent_path.read_text())
    parent_lock = json.loads(verify(b["parent_lock"]).read_text())
    upstream = {"parent_protocol": b["parent_protocol"], "parent_lock": b["parent_lock"], **parent["inputs"], **b["additional_inputs"]}
    runtime = within(root, parent["artifacts"]["runtime_root"])
    for k, v in parent_lock["artifacts"].items():
        upstream["g0a_" + k] = {"path": str(within(runtime, v["path"]).relative_to(root)), "sha256": v["sha256"]}
    for model in parent_lock["frozen_forward_models"]:
        upstream[model["arm_id"]] = {"path": model["model_path"], "sha256": model["model_sha256"]}
    require({k: v["sha256"] for k, v in upstream.items()} == lock["upstream_input_hashes"], "Upstream set changed")
    # Hash only: no model arrays, frozen target records or confirmation labels parsed.
    for spec in upstream.values():
        verify(spec)
    coordinate_base = within(root, b["artifacts"]["output_root"])
    artifacts = {}
    for k, v in lock["coordinate_artifacts"].items():
        artifacts[k] = within(coordinate_base, v["path"])
        require(file_hash(artifacts[k]) == v["sha256"], "Coordinate artifact changed")
    for v in lock["artifacts"].values():
        require(file_hash(within(base, v["path"])) == v["sha256"], "G0c artifact changed")
    rows = read_rows(artifacts["coordinates"])
    shards = read_rows(base / c["artifacts"]["shard_registry"])
    require(make_shards(rows) == shards, "Shard registry mismatch")
    context = build_pipeline_context(root, pipeline_profile=c["physics"]["profile"])
    require(context.context_hash == c["physics"]["context_sha256"] and
            context.physics_runtime.physics_bundle_hash == c["physics"]["bundle_sha256"] and
            context.physics_runtime.model_version == c["physics"]["model_version"], "Runtime mismatch")
    require(canonical(context.contract_mapping()) + b"\n" == (base / c["artifacts"]["runtime_context"]).read_bytes(), "Runtime contract changed")
    identity = {"protocol_sha256": file_hash(config_path), "protocol_lock_sha256": LOCK_HASH,
        "coordinate_ledger_sha256": file_hash(artifacts["coordinates"]),
        "split_registry_sha256": file_hash(artifacts["splits"]), "pipeline_context_sha256": context.context_hash,
        "runner_sha256": file_hash(Path(__file__)),
        "cli_sha256": file_hash(root / "scripts/run_geometry19_performance_gated_physics.py")}
    return c, rows, shards, context, identity


def check_row(source, result, context):
    require(all(result.get(k) == v for k, v in source.items() if k != "label_status"), "Immutable row field changed")
    require(result["label_status"] == "V2_2_EVALUATED", "Wrong label status")
    require(all(result.get(k) == v for k, v in eligibility_flags(result).items()), "Eligibility changed")
    require(result["pipeline_context_hash"] == context.context_hash and
            result["physics_bundle_hash"] == context.physics_runtime.physics_bundle_hash and
            result["physics_model_version"] == context.physics_runtime.model_version and
            result["target_domain_hash"] == context.target_domain.source_hash, "Row runtime mismatch")


def acquire_shard(output, shard, batch, context, identity, evaluator=_evaluate_row):
    path = within(output, shard["future_output"])
    meta_path = path.with_suffix(".seal.json")
    journal = path.with_suffix(".calls.jsonl")
    expected = {**identity, "shard_input_sha256": sha256(canonical(batch)).hexdigest()}
    require(expected["shard_input_sha256"] == shard["shard_input_sha256"], "Shard input changed")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        require(meta["identity"] == expected and file_hash(path) == meta["shard_output_sha256"] and
                file_hash(journal) == meta["call_journal_sha256"], "Shard seal mismatch")
        results = read_rows(path)
        require(len(results) == len(batch) == meta["actual_calls"], "Shard count mismatch")
        require(read_rows(journal) == [{"row_index": i, "design_hash": r["design_hash"], "event": "call_started"}
                                     for i, r in enumerate(batch)], "Call journal mismatch")
        for source, result in zip(batch, results):
            check_row(source, result, context)
        return results, meta
    require(not path.exists() and not journal.exists(), "Orphan partial shard: preserve files; explicit recovery required")
    results = []
    # Every attempted call is durably recorded BEFORE invoking physics. Any exception stops the run.
    with journal.open("xb") as calls, path.open("xb") as stream:
        for i, row in enumerate(batch):
            calls.write(canonical({"row_index": i, "design_hash": row["design_hash"], "event": "call_started"}) + b"\n")
            calls.flush()
            os.fsync(calls.fileno())
            result = evaluator(deepcopy(row), context)
            result["label_status"] = "V2_2_EVALUATED"
            result.update(eligibility_flags(result))
            check_row(row, result, context)
            stream.write(canonical(result) + b"\n")
            stream.flush()
            results.append(result)
        os.fsync(stream.fileno())
    meta = {"identity": expected, "actual_calls": len(batch), "infrastructure_replays": 0,
            "shard_output_sha256": file_hash(path), "call_journal_sha256": file_hash(journal)}
    write_json(meta_path, meta)
    return results, meta


def data_gates(rows, gates):
    roots, bands = defaultdict(Counter), defaultdict(Counter)
    for row in rows:
        for counts in (roots[row["root_seed_id"]], bands[(row["root_seed_id"], row["band_index"])]):
            counts["rows"] += 1
            counts["valid_train"] += int(row["training_eligible"])
            counts["valid_validation"] += int(row["validation_eligible"])
    require(len(rows) == 16896 and len(roots) == 33 and len(bands) == 132 and
            all(c["rows"] == 512 for c in roots.values()) and all(c["rows"] == 128 for c in bands.values()), "Incomplete population")
    failures = []
    for group, values, prefix in (("root", roots, "each_root"), ("root_band", bands, "each_root_band")):
        for key, counts in sorted(values.items()):
            for split in ("train", "validation"):
                minimum = gates[f"{prefix}_minimum_valid_{split}"]
                if counts[f"valid_{split}"] < minimum:
                    failures.append({"group": group, "key": key, "split": split, "valid": counts[f"valid_{split}"], "minimum": minimum})
    return {"status": "G0C_DATA_GATE_PASS" if not failures else "G0C_DATA_GATE_FAIL", "rows": len(rows),
            "valid_train": sum(r["training_eligible"] for r in rows),
            "valid_validation": sum(r["validation_eligible"] for r in rows), "failures": failures,
            "per_root": dict(roots), "per_root_band": [{"root_seed_id": k[0], "band_index": k[1], **v} for k, v in sorted(bands.items())],
            "solver_failure_counts": dict(Counter(str((r.get("solver") or {}).get("failure_code")) for r in rows if not (r.get("solver") or {}).get("converged"))),
            "training_authorized": False, "inverse_design_executed": False,
            "next_stage": "separate_gate_and_generator_training_protocol_required" if not failures else "seal_failure_and_stop_no_training"}


def run(root, config_path, *, authorize=False, resume=False):
    require(authorize != resume, "Choose first-run authorization OR resume")
    c, rows, shards, context, identity = verify_inputs(root, config_path)
    output = within(root, c["artifacts"]["future_physics_root"])
    execution = {"identity": identity, "authorization": {"physics_acquisition": True, "model_training": False,
        "inverse_design": False, "confirmation_content_access": False},
        "authorization_source": "user_continue_after_G0c_protocol_lock_and_runner_implementation", "logical_budget": 16896,
        "worker_policy": "sequential_cpu_one_context", "nonconvergence_retry": False, "adaptive_top_up": False}
    if authorize:
        output.mkdir(parents=True, exist_ok=False)
    else:
        require(output.is_dir(), "No acquisition to resume")
    with (output / ".runner.lock").open("a") as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output / "execution_manifest.json"
        if authorize:
            write_json(manifest_path, execution)
            (output / "shards").mkdir()
        else:
            require(json.loads(manifest_path.read_text()) == execution, "Execution identity/authorization changed")
        shard_identity = {**identity, "execution_manifest_sha256": file_hash(manifest_path)}
        # Refuse ANY partial shard before executing subsequent rows.
        expected_files = set()
        for s in shards:
            p = within(output, s["future_output"])
            paths = [p, p.with_suffix(".seal.json"), p.with_suffix(".calls.jsonl")]
            expected_files.update(paths)
            require(sum(p.exists() for p in paths) in (0, 3), "Orphan partial shard: explicit recovery required")
        require(set((output / "shards").iterdir()) <= expected_files, "Unexpected shard files")
        sealed = (output / "Acquisition_Manifest.json").exists()
        if sealed:
            require(all(p.exists() for p in expected_files), "Sealed run missing shards; no replay allowed")
        all_rows, seals = [], []
        started = time.monotonic()
        for s in shards:
            batch = rows[s["start_row_inclusive"]:s["stop_row_exclusive"]]
            result, seal = acquire_shard(output, s, batch, context, shard_identity)
            all_rows.extend(result)
            seals.append(seal)
            print(json.dumps({"sealed_shards": len(seals), "total_shards": 132, "sealed_rows": len(all_rows),
                              "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
        ledger = output / "Sealed_Evaluations.jsonl"
        payload = b"".join(canonical(r) + b"\n" for r in all_rows)
        if ledger.exists():
            require(ledger.read_bytes() == payload, "Complete ledger changed or partial; preserve and stop")
        else:
            require(not sealed, "Sealed ledger missing")
            write_once(ledger, payload)
        acquisition = {"identity": shard_identity, "rows": len(all_rows), "shards": seals,
                       "ledger_sha256": file_hash(ledger), "actual_calls": sum(s["actual_calls"] for s in seals),
                       "infrastructure_replays": 0, "all_results_sealed_before_gates": True}
        acquisition_path = output / "Acquisition_Manifest.json"
        if sealed:
            require(json.loads(acquisition_path.read_text()) == acquisition, "Acquisition manifest changed")
        else:
            write_json(acquisition_path, acquisition)
        # Only now may performance-dependent gates be computed.
        summary = data_gates(all_rows, c["gates"])
        summary.update({"acquisition_manifest_sha256": file_hash(acquisition_path), "ledger_sha256": file_hash(ledger),
                        "actual_calls": acquisition["actual_calls"], "infrastructure_replays": 0})
        report = ("# G0c frozen V2.2 acquisition\n\n" + f"Status: `{summary['status']}`\n\n"
                  f"Sealed: {len(all_rows)} rows, {len(seals)} shards; calls: {acquisition['actual_calls']}; replays: 0.\n"
                  f"Eligible train: {summary['valid_train']}/13464; validation: {summary['valid_validation']}/3432.\n"
                  f"Failed root/split or root-band/split gates: {len(summary['failures'])}.\n\n"
                  "All evaluations sealed before data gates. Validation rows excluded from training candidates.\n"
                  "No top-up, solver retries, target/model use, confirmation parsing, training or inverse design.\n"
                  "Passing this data gate is not engineering validation or training authorization.\n"
                  f"Next: {summary['next_stage']}.\n").encode()
        for name, content in (("Acquisition_Summary.json", canonical(summary) + b"\n"), ("Acquisition_Report.md", report)):
            path = output / name
            if path.exists():
                require(path.read_bytes() == content, f"Final artifact changed: {name}")
            else:
                write_once(path, content)
        return summary
