"""Freeze G0a inputs and targets using only existing evidence (no model/runtime imports)."""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import yaml

VERSION = "geometry19_v2_2_performance_gated_local_cfm_v1"
PASS = "PERFORMANCE_GATED_LOCAL_CFM_G0A_PROTOCOL_TARGET_LOCK_PASS"
OUTPUTS = ["thrust_N", "tsfc_kg_N_s", "mass_kg"]
INPUTS = {"development_dataset", "anchor_registry", "stable_root_registry", "physics_ledger",
          "scoring_summary", "s2_model_manifest", "schema", "target_domain", "sealed_confirmation"}


class ProtocolError(ValueError):
    """Invalid evidence or attempted alteration of the G0a boundary."""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ProtocolError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Artifact/input path escapes root")
    return path


def validate_protocol(c: dict) -> None:
    require(c["protocol_version"] == VERSION, "Protocol version mismatch")
    require(c["stage"] == "G0a_protocol_and_target_registry_lock", "Wrong stage")
    require(c["scope"] == "post_selected_33root_development_closure", "Scope changed")
    require(set(c["inputs"]) == INPUTS, "Input roles changed")
    require(c["inputs"]["sealed_confirmation"]["access"] == "sha256_only_no_parse", "Confirmation access changed")
    a = c["authorization"]
    require(a == {"protocol_and_targets_authorized": True, "physics_acquisition_authorized": False,
                  "training_authorized": False, "candidate_generation_authorized": False,
                  "closure_evaluation_authorized": False, "confirmation_content_access": False,
                  "formal_use_allowed": False, "deployment_allowed": False}, "Authorization changed")
    t = c["targets"]
    require((t["roots"], t["count"], t["classes"], t["nominal_radii"]) ==
            (33, 99, ["center", "typical", "outer"], [0.0, 0.1, 0.2]), "Target population changed")
    require(t["runtime_fields"] == ["target_id", "performance", "units"] and
            t["source_root_in_runtime_view"] is False and t["source_geometry_in_runtime_view"] is False,
            "Runtime source leakage")
    require(t["performance_order"] == OUTPUTS, "Output order changed")
    require(t["support_radius"] == 0.2 and t["distance_numeric_tolerance"] == 1e-12 and
            t["same_medoid_integer_state_required"] is True, "Support/discrete rule changed")
    require(t["rule_version"] == "medoid_and_actual_serialized_distance_v1" and
            t["shell_tie_break"] == ["absolute_actual_distance_minus_nominal", "design_hash", "attempt_id"],
            "Selection rule changed")
    require(t["root_order"] == "root_seed_id_lexicographic" and
            t["center_source"] == "frozen_medoid_in_existing_development" and
            t["shell_source"] == "evaluation_eligible_v4_1_only" and
            t["distance"] == "schema_range_normalized_l2_from_medoid_divided_by_frozen_root_p95" and
            t["units"] == {"thrust_N": "N", "tsfc_kg_N_s": "kg/(N*s)", "mass_kg": "kg"},
            "Target source, distance or units changed")
    m = c["models"]
    require(m["condition_dim"] == 3 and m["root_embedding"] is False and m["medoid_as_condition"] is False,
            "Model condition leakage")
    require(m["seeds"] == [42, 43, 44], "Training seeds changed")
    d = c["training_data_plan"]
    require(d["v4_1_training_allowed"] is False and d["targets_used_for_model_selection"] is False,
            "Evaluation leakage into training")
    require(d["coordinates_generated_in_G0a"] is False and d["total_coordinates"] == 16896,
            "Coordinate stage changed")
    e = c["evaluation"]
    require(e["target_root_for_scoring"] is False and e["physical_feedback_or_resampling"] is False and
            e["complete_selection_ledger_before_physics"] is True, "Closure leakage")
    require(e["raw_slots_total"] == 4 * 99 * 3 * e["raw_candidates_per_task"] == 304128 and
            e["physics_slots_total"] == 4 * 99 * 3 * e["physics_slots_per_task"] == 9504,
            "Evaluation budget mismatch")
    require(c["artifacts"] == {
        "runtime_root": "data/validation/v2_2_performance_gated_local_cfm_v1",
        "target_registry": "targets/Target_Registry.jsonl", "runtime_targets": "targets/Runtime_Targets.jsonl",
        "root_audit": "audits/Root_Target_Selection_Audit.json", "report": "audits/G0a_Protocol_Target_Lock_Report.md",
        "lock": "audits/G0a_Protocol_Target_Lock.json",
        "future_directories": ["training_coordinates", "training_physics", "splits", "models", "candidates", "closure"],
    }, "Artifact destinations changed")


def geometry_distance(row: dict, anchor: dict, variables: list[dict], radius: float) -> float:
    g, medoid = row["geometry19"], anchor["medoid_geometry19"]
    require(set(g) == {v["name"] for v in variables}, "Geometry schema keys changed")
    for v in variables:
        x = g[v["name"]]
        require(type(x) in (int, float) and math.isfinite(x) and v["bounds"][0] <= x <= v["bounds"][1],
                "Invalid geometry bounds/type")
        if v["dtype"] == "int64":
            require(type(x) is int, "Integer axis is not serialized as integer")
    require(sha256(canonical(g)).hexdigest() == row["design_hash"], "Source geometry hash mismatch")
    require(math.isfinite(radius) and radius > 0, "Invalid root radius")
    return math.sqrt(sum(((g[v["name"]] - medoid[v["name"]]) /
                          (v["bounds"][1] - v["bounds"][0])) ** 2 for v in variables)) / radius


def validate_physics(row: dict, c: dict) -> None:
    p = c["physics"]
    require(row["solver"]["converged"] is True and
            row["constraint_assessment"]["numerically_valid"] is True and
            row["constraint_assessment"]["physics_domain_valid"] is True and
            row["constraint_assessment"]["accepted"] is True and
            row["target_domain_assessment"]["inside"] is True, "Source physics not eligible")
    for name in OUTPUTS:
        x = row["performance"][name]
        require(type(x) in (int, float) and math.isfinite(x) and x > 0, "Invalid target performance")
    require(row["physics_model_version"] == p["model_version"] and
            row["physics_bundle_hash"] == p["bundle_sha256"] and
            row["pipeline_context_hash"] == p["context_sha256"] and
            row["constraint_assessment"]["constraint_profile_hash"] == p["constraint_profile_sha256"] and
            row["target_domain_hash"] == c["inputs"]["target_domain"]["sha256"], "Source physics context changed")


def select_targets(c: dict, anchors: list[dict], stable: list[dict], development: list[dict],
                   ledger: list[dict], variables: list[dict]) -> tuple[list[dict], list[dict]]:
    eligible = [r for r in ledger if r.get("evaluation_eligible") is True]
    sequence = [r["design_hash"] for r in eligible]
    require(len(ledger) == 3168 and len(eligible) == c["targets"]["shell_eligible_rows"] == 3039,
            "Physics population changed")
    require(sha256(canonical(sequence)).hexdigest() == c["targets"]["eligible_design_hash_sequence_sha256"],
            "Eligible sequence changed")
    require(len(set(sequence)) == len(sequence), "Duplicate evaluation designs")
    roots = sorted({r["root_seed_id"] for r in eligible})
    anchor_map = {a["root_seed_id"]: a for a in anchors}
    stable_map = {r["root_seed_id"]: r for r in stable}
    require(len(anchors) == len(anchor_map) == len(roots) == 33 and set(anchor_map) == set(roots),
            "Root scope changed")
    require(len(stable_map) == len(stable) and set(roots) <= set(stable_map), "Stable roots missing/duplicated")
    by_hash: dict[str, list] = defaultdict(list)
    by_root: dict[str, list] = defaultdict(list)
    for r in development:
        by_hash[r["design_hash"]].append(r)
    for r in eligible:
        validate_physics(r, c)
        by_root[r["root_seed_id"]].append(r)
    integer_names = [v["name"] for v in variables if v["dtype"] == "int64"]
    require(integer_names == c["models"]["local_cfm"]["integer_axes"] and len(variables) == 19,
            "Frozen schema axis order changed")
    targets, audits = [], []
    for root in roots:
        anchor, registry = anchor_map[root], stable_map[root]
        radius = registry["support_radius_p95_normalized_l2"]
        medoid_rows = by_hash[anchor["medoid_design_hash"]]
        require(len(medoid_rows) == 1, "Missing/ambiguous center source")
        center = medoid_rows[0]
        require(center["root_seed_id"] == root and center["sample_id"] == anchor["medoid_sample_id"] and
                center["training_candidate"] is True and center["geometry19"] == anchor["medoid_geometry19"],
                "Center identity changed")
        validate_physics(center, c)
        require(geometry_distance(center, anchor, variables, radius) == 0, "Center distance changed")
        state = {k: anchor["medoid_geometry19"][k] for k in integer_names}
        selected = [("center", 0.0, center, 0.0, "development_dataset")]
        counts = {}
        for kind, nominal in (("typical", 0.1), ("outer", 0.2)):
            candidates = []
            for row in by_root[root]:
                if row["radius_fraction"] != nominal or any(row["geometry19"][k] != v for k, v in state.items()):
                    continue
                require(row["anchor_design_hash"] == anchor["medoid_design_hash"], "Shell anchor changed")
                distance = geometry_distance(row, anchor, variables, radius)
                if distance <= c["targets"]["support_radius"] + c["targets"]["distance_numeric_tolerance"]:
                    candidates.append((abs(distance - nominal), row["design_hash"], row["attempt_id"], distance, row))
            counts[kind] = len(candidates)
            require(bool(candidates), f"No legal {kind} target for root {root}")
            choice = min(candidates, key=lambda x: x[:3])
            selected.append((kind, nominal, choice[4], choice[3], "physics_ledger"))
        audits.append({"root_seed_id": root, "eligible_shell_counts": counts,
                       "anchor_design_hash": anchor["medoid_design_hash"], "integer_state": state,
                       "radius_p95": radius})
        for kind, nominal, row, distance, source in selected:
            target_id = sha256(canonical([VERSION, root, kind, row["design_hash"]])).hexdigest()
            targets.append({
                "target_index": len(targets), "target_id": target_id, "root_seed_id": root,
                "target_class": kind, "source_root_segments": registry["segments"],
                "source_role": source, "source_sample_id": row.get("sample_id", row.get("attempt_id")),
                "source_design_hash": row["design_hash"], "anchor_design_hash": anchor["medoid_design_hash"],
                "source_split": "existing_development" if kind == "center" else "sealed_v4_1_evaluation",
                "center_is_known_anchor_diagnostic": kind == "center",
                "performance": {k: row["performance"][k] for k in OUTPUTS}, "units": c["targets"]["units"],
                "nominal_radius": nominal, "actual_local_distance": distance, "support_pass": True,
                "support_radius_p95": radius, "integer_state": state,
                "physics_model_version": row["physics_model_version"],
                "physics_bundle_hash": row["physics_bundle_hash"], "pipeline_context_hash": row["pipeline_context_hash"],
                "target_domain_hash": row["target_domain_hash"], "rule_version": c["targets"]["rule_version"],
                "source_file_sha256": c["inputs"][source]["sha256"],
                "protocol_version": VERSION,
            })
    require(len(targets) == 99 and len({r["source_design_hash"] for r in targets}) == 99, "Target count/uniqueness failed")
    return targets, audits


def lock_protocol(repo_root: Path, config_path: Path, *, output_root: Path | None = None) -> dict:
    repo_root, config_path = repo_root.resolve(), config_path.resolve()
    c = yaml.safe_load(config_path.read_text())
    validate_protocol(c)
    output = output_root.resolve() if output_root is not None else within(repo_root, c["artifacts"]["runtime_root"])
    require(not output.exists(), "Refusing to overwrite existing lock/output directory")
    paths, inputs = {}, {}
    for role, item in c["inputs"].items():
        path = within(repo_root, item["path"])
        require(path.is_file() and file_hash(path) == item["sha256"], f"Input hash mismatch: {role}")
        paths[role] = path
        inputs[role] = {**item, "hash_match": True, "content_parsed": role not in {"sealed_confirmation", "target_domain"}}
    summary = json.loads(paths["scoring_summary"].read_text())
    require(summary["status"] == "STABLE_ROOT_S3_V4_1_FORWARD_SURROGATE_PASS", "Forward gate not passed")
    model_manifest = json.loads(paths["s2_model_manifest"].read_text())
    require(model_manifest["status"] == "STABLE_ROOT_S2_MODEL_MANIFEST_LOCK_PASS", "S2 model gate not passed")
    models = []
    for model in model_manifest["models"]:
        path = within(repo_root, model["model_path"])
        require(file_hash(path) == model["model_sha256"], "Frozen model hash changed")
        models.append({k: model[k] for k in ("arm_id", "model_path", "model_sha256")})
    variables = sorted(yaml.safe_load(paths["schema"].read_text())["design_variables"], key=lambda v: v["index"])
    targets, root_audit = select_targets(c, read_rows(paths["anchor_registry"]), read_rows(paths["stable_root_registry"]),
                                         read_rows(paths["development_dataset"]), read_rows(paths["physics_ledger"]), variables)
    protocol_hash = file_hash(config_path)
    for row in targets:
        row["protocol_sha256"] = protocol_hash
        row["input_hashes"] = {k: v["sha256"] for k, v in c["inputs"].items()}
    runtime = [{k: r[k] for k in ("target_id", "performance", "units")} for r in targets]
    files = {key: within(output, c["artifacts"][key]) for key in
             ("target_registry", "runtime_targets", "root_audit", "report", "lock")}
    require(len(set(files.values())) == len(files), "Artifact paths collide")
    result = {
        "status": PASS, "stage": c["stage"], "protocol_version": VERSION,
        "protocol_sha256": protocol_hash, "implementation_sha256": file_hash(Path(__file__)),
        "scope": c["scope"], "input_audit": inputs, "frozen_forward_models": models,
        "roots": len(root_audit), "targets": len(targets),
        "target_class_counts": dict(Counter(r["target_class"] for r in targets)),
        "target_id_sequence_sha256": sha256(canonical([r["target_id"] for r in targets])).hexdigest(),
        "source_design_hash_sequence_sha256": sha256(canonical([r["source_design_hash"] for r in targets])).hexdigest(),
        "runtime_target_fields": ["target_id", "performance", "units"],
        "confirmation_content_read": False, "model_arrays_loaded": False,
        "model_predictions_executed": False, "new_physics_evaluations": 0,
        "training_coordinates_generated": False, "training_executed": False,
        "authorization": c["authorization"],
        "next_stage": "G0b_training_coordinate_and_split_lock_only",
        "limitations": ["development_closure_not_fresh_confirmation", "center_is_known_anchor_diagnostic",
                        "runtime_view_is_data_minimization_not_os_access_control",
                        "training_coordinate_algorithm_and_environment_require_separate_lock"],
    }
    # Validate everything before any writes; exclusive creation prevents silently overwriting evidence.
    output.mkdir(parents=True, exist_ok=False)
    payloads = {"target_registry": b"".join(canonical(r) + b"\n" for r in targets),
                "runtime_targets": b"".join(canonical(r) + b"\n" for r in runtime),
                "root_audit": canonical(root_audit) + b"\n"}
    result["artifacts"] = {}
    for key, payload in payloads.items():
        files[key].parent.mkdir(parents=True, exist_ok=True)
        with files[key].open("xb") as stream:
            stream.write(payload)
        result["artifacts"][key] = {"path": c["artifacts"][key], "sha256": file_hash(files[key])}
    result["future_artifacts_absent"] = {name: not within(output, name).exists() for name in c["artifacts"]["future_directories"]}
    require(all(result["future_artifacts_absent"].values()), "Future execution artifact exists")
    report = (f"# Performance-Gated Local CFM G0a protocol and target lock\n\n"
              f"Status: `{PASS}`\n\n"
              "33 roots; 99 unique targets; center/typical/outer: 33 each.\n\n"
              "Actual serialized geometry distances, integer states and source physics contexts verified.\n"
              "Runtime view contains only opaque target ID, three performance values and units.\n"
              "Source roots and hashes remain in the audit registry, never in generator conditions.\n\n"
              "Existing model files hashed, not loaded. Confirmation hash checked, contents not parsed.\n"
              "No new physics, training, coordinate generation or inverse candidate generation executed.\n\n"
              "Next: separately lock new training coordinates and pair-group splits (G0b).\n"
              "This lock does not authorize physics acquisition or model training.\n"
              "Center targets are known-anchor diagnostics; all results remain development closure.\n")
    with files["report"].open("x") as stream:
        stream.write(report)
    result["artifacts"]["report"] = {"path": c["artifacts"]["report"], "sha256": file_hash(files["report"])}
    with files["lock"].open("xb") as stream:
        stream.write(canonical(result) + b"\n")
    return result
