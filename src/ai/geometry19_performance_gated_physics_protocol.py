"""G0c acquisition contract and shard lock. Construction never evaluates physics."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from hashlib import sha256

import yaml

from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.ai.geometry19_performance_gated_protocol import canonical, file_hash, read_rows, require, within, OUTPUTS
from src.ai.geometry19_performance_gated_coordinates import (
    PASS as COORDINATE_PASS, validate_protocol as validate_coordinates, prepare_inputs, audit_coordinates,
)

VERSION = "geometry19_performance_gated_g0c_physics_v1"
PASS = "PERFORMANCE_GATED_LOCAL_CFM_G0C_PHYSICS_PROTOCOL_LOCK_PASS"


def validate_protocol(c):
    require(c["protocol_version"] == VERSION and c["stage"] == "G0c_physics_acquisition_protocol_lock", "Wrong stage/version")
    require(c["authorization"] == {"protocol_lock_authorized": True, "physics_acquisition_authorized_now": False,
        "model_training_authorized": False, "model_prediction_authorized": False, "inverse_design_authorized": False,
        "confirmation_content_access": False, "formal_use_allowed": False, "deployment_allowed": False}, "Authorization changed")
    require(c["population"] == {"rows": 16896, "roots": 33, "bands_per_root": 4, "rows_per_root": 512,
        "train_rows_per_root": 408, "validation_rows_per_root": 104, "shards": 132, "rows_per_shard": 128,
        "train_rows_per_shard": 102, "validation_rows_per_shard": 26,
        "order": "frozen_G0b_root_band_pair_plus_minus_sequence"}, "Population changed")
    require(c["eligibility"] == {
        "physics_eligible_requires": ["solver_converged", "numerically_valid", "physics_domain_valid", "finite_positive_three_performance", "inside_target_domain"],
        "training_eligible": "physics_eligible_and_frozen_split_train",
        "validation_eligible": "physics_eligible_and_frozen_split_validation",
        "override_legacy_training_candidate": "equal_training_eligible", "preserve_all_rejected_rows": True,
        "split_reassignment": "forbidden"}, "Eligibility/split policy changed")
    require(c["gates"] == {"each_root_minimum_valid_train": 256, "each_root_minimum_valid_validation": 64,
        "each_root_band_minimum_valid_train": 32, "each_root_band_minimum_valid_validation": 8,
        "require_all_33_roots_and_132_shards": True, "failure_action": "seal_failure_and_stop_no_training",
        "pass_action": "separate_gate_and_generator_training_protocol_required"}, "Data gates changed")
    e = c["execution"]
    require(e["adaptive_top_up"] is False and e["nonconvergence_retry"] is False and
        e["physics_result_based_coordinate_changes"] is False and e["models_or_targets_used_during_evaluation"] is False,
        "Adaptive execution/leakage")
    require(e["order"] == ["verify_execution_authorization_and_all_hashes", "evaluate_frozen_rows_in_order", "seal_all_shards",
        "seal_complete_ledger_and_manifest", "apply_data_gates", "write_pass_or_failure_report"], "Seal-before-gate order changed")
    require(e["resume_identity"] == ["protocol_sha256", "execution_manifest_sha256", "coordinate_ledger_sha256",
        "split_registry_sha256", "pipeline_context_sha256", "shard_input_sha256", "shard_output_sha256"], "Resume identity changed")
    require(e["orphan_partial_shard"] == "abort_preserve_files_require_explicit_recovery_no_overwrite" and
        e["future_runner_implementation_hash_required_before_execution"] is True, "Recovery/runner lock changed")
    require(c["physics"]["profile"] == "v2_2_development" and c["physics"]["worker_policy"] == "sequential_cpu_one_context", "Physics policy changed")


def eligibility_flags(row):
    """Future evaluator must replace legacy training_candidate, including on validation rows."""
    require(row["split"] in {"train", "validation"}, "Unknown frozen split")
    p, a = row.get("performance") or {}, row.get("constraint_assessment") or {}
    finite = all(type(p.get(k)) in (float, int) and math.isfinite(p[k]) and p[k] > 0 for k in OUTPUTS)
    eligible = bool((row.get("solver") or {}).get("converged") is True and a.get("numerically_valid") is True and
        a.get("physics_domain_valid") is True and (row.get("target_domain_assessment") or {}).get("inside") is True and finite)
    training = eligible and row["split"] == "train"
    return {"physics_eligible": eligible, "training_eligible": training,
            "validation_eligible": eligible and row["split"] == "validation", "training_candidate": training}


def make_shards(rows):
    require(len(rows) == 16896, "Wrong coordinate count")
    ordered = sorted(rows, key=lambda r: (r["root_seed_id"], r["band_index"], r["pair_index"], -r["sign"]))
    require(rows == ordered, "Frozen coordinate order changed")
    result = []
    for start in range(0, len(rows), 128):
        batch = rows[start:start + 128]
        root, band = batch[0]["root_seed_id"], batch[0]["band_index"]
        require(all(r["root_seed_id"] == root and r["band_index"] == band for r in batch), "Shard crosses root/band")
        counts = Counter(r["split"] for r in batch)
        require(counts == {"train": 102, "validation": 26}, "Shard split mismatch")
        result.append({"shard_index": len(result), "root_seed_id": root, "band_index": band,
            "start_row_inclusive": start, "stop_row_exclusive": start + 128, "row_count": 128,
            "split_counts": dict(counts), "shard_input_sha256": sha256(canonical(batch)).hexdigest(),
            "design_hash_sequence_sha256": sha256(canonical([r["design_hash"] for r in batch])).hexdigest(),
            "future_output": f"shards/part-{len(result):05d}.jsonl"})
    require(len({(s["root_seed_id"], s["band_index"]) for s in result}) == 132, "Shard identities duplicated")
    return result


def lock_protocol(root: Path, config_path: Path, *, output_root: Path | None = None):
    root, config_path = root.resolve(), config_path.resolve()
    c = yaml.safe_load(config_path.read_text())
    validate_protocol(c)
    output = output_root.resolve() if output_root is not None else within(root, c["artifacts"]["output_root"])
    require(not output.exists(), "Refusing to overwrite G0c lock")
    paths, hashes = {}, {}
    for role, spec in c["inputs"].items():
        paths[role] = within(root, spec["path"])
        require(file_hash(paths[role]) == spec["sha256"], f"Input hash mismatch: {role}")
        hashes[role] = spec["sha256"]
    g0b = yaml.safe_load(paths["coordinate_protocol"].read_text())
    validate_coordinates(g0b)
    locked = json.loads(paths["coordinate_lock"].read_text())
    require(locked["status"] == COORDINATE_PASS and locked["protocol_sha256"] == hashes["coordinate_protocol"], "G0b gate invalid")
    require(locked["implementation_sha256"] == file_hash(root / "src/ai/geometry19_performance_gated_coordinates.py"), "G0b code changed")
    source_paths, source_hashes, parent = prepare_inputs(root, g0b)
    require(source_hashes == locked["input_hashes"], "G0b upstream input set changed")
    base = within(root, g0b["artifacts"]["output_root"])
    for role, spec in locked["artifacts"].items():
        path = within(base, spec["path"])
        require(file_hash(path) == spec["sha256"], f"G0b artifact changed: {role}")
    rows = read_rows(base / locked["artifacts"]["coordinates"]["path"])
    splits = read_rows(base / locked["artifacts"]["splits"]["path"])
    anchors = {r["root_seed_id"]: r for r in read_rows(source_paths["anchor_registry"])}
    stable = {r["root_seed_id"]: r for r in read_rows(source_paths["stable_root_registry"])}
    variables = sorted(yaml.safe_load(source_paths["schema"].read_text())["design_variables"], key=lambda v: v["index"])
    excluded = set()
    for role in ("development_dataset", "s3_v1", "s3_v2", "s3_v3", "s3_v4_combined"):
        excluded.update(r["design_hash"] for r in read_rows(source_paths[role]))
    excluded.update(r["source_design_hash"] for r in read_rows(source_paths["g0a_target_registry"]))
    coordinate_audit = audit_coordinates(rows, splits, anchors, stable, variables, excluded, g0b["algorithm"])
    require(coordinate_audit == locked["audit"], "Recomputed coordinate audit differs")
    shards = make_shards(rows)
    # Only construct the context: no evaluate call, row evaluator import or solver execution.
    context = build_pipeline_context(root, pipeline_profile=c["physics"]["profile"])
    require(context.context_hash == c["physics"]["context_sha256"] == parent["physics"]["context_sha256"], "Physics context mismatch")
    require(context.physics_runtime.physics_bundle_hash == c["physics"]["bundle_sha256"] == parent["physics"]["bundle_sha256"] and
        context.physics_runtime.model_version == c["physics"]["model_version"] == parent["physics"]["model_version"], "Physics bundle/model mismatch")
    require(context.target_domain.source_hash == parent["inputs"]["target_domain"]["sha256"], "Target domain changed")
    future = within(root, c["artifacts"]["future_physics_root"])
    require(not future.exists(), "Future acquisition output exists")
    result = {"status": PASS, "protocol_version": VERSION, "protocol_sha256": file_hash(config_path),
        "implementation_sha256": file_hash(Path(__file__)), "g0b_implementation_sha256": locked["implementation_sha256"],
        "g0a_implementation_sha256": locked["parent_helper_sha256"], "input_hashes": hashes,
        "upstream_input_hashes": source_hashes, "coordinate_artifacts": locked["artifacts"],
        "coordinate_audit": coordinate_audit, "shards": len(shards), "rows_per_shard": 128,
        "context_sha256": context.context_hash, "physics_bundle_sha256": context.physics_runtime.physics_bundle_hash,
        "authorization": c["authorization"], "runtime_context_constructed": True,
        "physics_evaluations_executed": False, "model_arrays_loaded": False, "model_predictions_executed": False,
        "confirmation_content_read": False, "training_executed": False, "future_physics_output_absent": True,
        "next_stage": "implement_and_hash_runner_then_explicitly_authorize_frozen_acquisition",
        "acquisition_gates": c["gates"], "artifacts": {}}
    files = {k: within(output, c["artifacts"][k]) for k in ("shard_registry", "runtime_context", "report", "lock")}
    require(len(set(files.values())) == len(files), "Artifact paths collide")
    output.mkdir(parents=True, exist_ok=False)
    payloads = {"shard_registry": b"".join(canonical(s) + b"\n" for s in shards),
                "runtime_context": canonical(context.contract_mapping()) + b"\n"}
    for key, content in payloads.items():
        with files[key].open("xb") as stream:
            stream.write(content)
        result["artifacts"][key] = {"path": c["artifacts"][key], "sha256": file_hash(files[key])}
    report = (f"# G0c V2.2 training physics protocol lock\n\nStatus: `{PASS}`\n\n"
        "Frozen: 16,896 rows; 33 roots; 132 root/band shards, 128 rows each.\n"
        "Each shard: 102 train + 26 validation rows; each root: 408 train + 104 validation rows.\n\n"
        "All shards and the complete evaluation ledger must be sealed before gates.\n"
        "Required valid rows per root: train >=256, validation >=64; per band: train >=32, validation >=8.\n"
        "Validation rows never become training candidates, even when physically valid.\n\n"
        "Resume verifies protocol, execution, coordinates, split, context and shard hashes; orphan files cause a stop.\n"
        "Runtime context matches existing V2.2 labels. Context constructed without physics evaluation.\n"
        "No physics evaluations, model loads, training or confirmation parsing executed.\n"
        "Acquisition execution remains unauthorized by this protocol-only lock.\n")
    with files["report"].open("x") as stream:
        stream.write(report)
    result["artifacts"]["report"] = {"path": c["artifacts"]["report"], "sha256": file_hash(files["report"])}
    with files["lock"].open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    return result
