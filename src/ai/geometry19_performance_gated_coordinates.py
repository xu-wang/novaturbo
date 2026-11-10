"""G0b geometry-only Sobol acquisition coordinates; never calls a physics solver."""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.stats import qmc
import yaml

from src.ai.geometry19_performance_gated_protocol import (
    PASS as PARENT_PASS, canonical, file_hash, read_rows, require, validate_protocol as validate_parent, within,
)

VERSION = "geometry19_performance_gated_g0b_coordinates_v1"
PASS = "PERFORMANCE_GATED_LOCAL_CFM_G0B_COORDINATE_SPLIT_LOCK_PASS"
FAIL = "PERFORMANCE_GATED_LOCAL_CFM_G0B_COORDINATE_SPLIT_LOCK_FAIL"
BANDS = [[0.0, 0.05], [0.05, 0.10], [0.10, 0.15], [0.15, 0.20]]


def validate_protocol(c: dict) -> None:
    require(c["protocol_version"] == VERSION and c["stage"] == "G0b_prelabel_training_coordinates_and_pair_split_lock", "Wrong G0b version/stage")
    a, s = c["algorithm"], c["split"]
    expected = {"roots": 33, "bands": BANDS, "pairs_per_root_band": 64, "rows_per_root": 512,
                "expected_rows": 16896, "primary_seed": 20260923, "reserve_seed": 20260924,
                "sobol_dimensions": 17, "pool_base2_exponent": 13, "pool_rows_per_band": 2048,
                "maximum_attempts_per_band": 4096, "covariance_dimensions": 16,
                "shrinkage": .05, "bounds_safety_factor": .95, "radial_endpoint_margin": 1e-6,
                "blockage_thickness_ratio": .2,
                "integer_policy": "exact_medoid_values_never_perturbed_or_rounded",
                "band_membership": "actual_serialized_geometry_distance_strict_lower_inclusive_upper",
                "selection": "first_64_legal_pairs_per_band_primary_then_reserve",
                "failure": "seal_partial_ledger_and_screening_then_stop_no_top_up"}
    require(all(a.get(k) == v for k, v in expected.items()), "Coordinate algorithm changed")
    descriptions = {
        "version": "continuous16_medoid_sobol_pair_bands_v1",
        "seed_mix": "xor_first_8_hex_root_id",
        "direction_transform": "first_16_sobol_axes_signed_uniform_times_symmetric_covariance_sqrt_then_unit_l2",
        "radial_transform": "band_lower_plus_width_times_1e_6_plus_1_minus_2e_6_times_sobol_axis_17",
        "covariance": "same_root_unique_development_geometry_sorted_by_design_hash_medoid_second_moment_n_minus_1",
        "square_root": "numpy_eigh_symmetric_v_sqrt_eigenvalues_v_transpose",
        "shrink": "min_desired_step_and_common_two_sided_schema_limit_times_safety",
        "cross_constraint_reference": "geometry_state_py_pre_solver_geometry_only",
        "exclusion": "development_all_designs_all_prior_s3_selected_coordinates_targets_and_this_run",
        "hard_checks": ["schema_bounds_types", "compressor_inducer_order", "combustor_annulus", "turbine_annulus", "compressor_exit_blockage"],
    }
    require(all(a.get(k) == v for k, v in descriptions.items()), "Algorithm description/implementation mismatch")
    require(s == {"unit": "accepted_positive_negative_pair",
                  "validation": "accepted_pair_index_within_root_band_modulo_5_equals_0",
                  "training_pairs_per_root_band": 51, "validation_pairs_per_root_band": 13,
                  "expected_training_rows": 13464, "expected_validation_rows": 3432, "before_physics_labels": True}, "Split changed")
    require(c["authorization"] == {"coordinate_generation": True, "physics_acquisition": False,
                "training": False, "model_prediction": False, "inverse_candidate_generation": False,
                "confirmation_parse": False}, "Authorization changed")
    require(c["environment"] == {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}, "Numerical environment mismatch")


def hard_failure(g: dict, variables: list[dict], thickness: float) -> str | None:
    """Only schema and analytic pre-solver geometry, not post-physics engineering gates."""
    for v in variables:
        value = g[v["name"]]
        if type(value) not in (int, float) or not math.isfinite(value) or not v["bounds"][0] <= value <= v["bounds"][1]:
            return "schema_bounds_types"
        if v["dtype"] == "int64" and type(value) is not int:
            return "integer_type"
    if not 0 < g["compressor_inducer_hub_tip_ratio"] < 1 or not g["compressor_inducer_tip_diameter_m"] < g["compressor_impeller_exit_diameter_m"]:
        return "compressor_inducer_order"
    if not g["combustor_outer_diameter_m"] > g["combustor_inner_diameter_m"] + 2 * g["combustor_liner_thickness_m"]:
        return "combustor_annulus"
    if not 0 < g["turbine_hub_tip_ratio"] < 1 or not g["turbine_tip_diameter_m"] > 0:
        return "turbine_annulus"
    denominator = math.pi * g["compressor_impeller_exit_diameter_m"] * math.sin(math.radians(g["compressor_exit_blade_angle_deg"]))
    if denominator <= 0:
        return "compressor_exit_blockage"
    blockage = 1 - g["compressor_blade_count"] * thickness * g["compressor_impeller_exit_width_m"] / denominator
    return None if 0 < blockage < 1 else "compressor_exit_blockage"


def direction_model(root: str, rows: list[dict], anchor: dict, variables: list[dict], algorithm: dict) -> tuple[np.ndarray, list[np.ndarray], dict]:
    cont = [v for v in variables if v["dtype"] != "int64"]
    lower = np.array([v["bounds"][0] for v in cont], dtype=float)
    span = np.array([v["bounds"][1] - v["bounds"][0] for v in cont], dtype=float)
    center = (np.array([anchor[v["name"]] for v in cont]) - lower) / span
    unique = {r["design_hash"]: r for r in rows}
    require(len(unique) >= 24, f"Too few geometry rows: {root}")
    x = np.array([[unique[h]["geometry19"][v["name"]] for v in cont] for h in sorted(unique)])
    delta = (x - lower) / span - center
    covariance = delta.T @ delta / (len(delta) - 1)
    trace = np.trace(covariance) / len(cont)
    require(np.isfinite(covariance).all() and trace > 0, "Degenerate covariance")
    covariance = (1 - algorithm["shrinkage"]) * covariance + algorithm["shrinkage"] * trace * np.eye(len(cont))
    values, vectors = np.linalg.eigh(covariance)
    require(np.all(values > 0), "Covariance not positive definite")
    square_root = (vectors * np.sqrt(values)) @ vectors.T
    pools = []
    for base in (algorithm["primary_seed"], algorithm["reserve_seed"]):
        seed = base ^ int(root[:8], 16)
        latent = qmc.Sobol(d=17, scramble=True, seed=seed).random_base2(m=algorithm["pool_base2_exponent"])
        directions = (2 * latent[:, :16] - 1) @ square_root.T
        norms = np.linalg.norm(directions, axis=1)
        require(np.isfinite(norms).all() and np.all(norms > 0), "Invalid Sobol direction")
        pools.append(np.column_stack([directions / norms[:, None], latent[:, 16]]))
    audit = {"root_seed_id": root, "unique_geometry_rows": len(unique), "continuous_axes": [v["name"] for v in cont],
             "covariance": covariance.tolist(), "symmetric_square_root": square_root.tolist(),
             "eigenvalues_ascending": values.tolist(),
             "geometry_hash_sequence_sha256": sha256(canonical(sorted(unique))).hexdigest(),
             "pool_sha256": [sha256(np.asarray(p, dtype="<f8").tobytes()).hexdigest() for p in pools]}
    return center, pools, audit


def generate_root(root: str, anchor: dict, radius: float, rows: list[dict], variables: list[dict],
                  algorithm: dict, excluded: set[str]) -> tuple[list[dict], list[dict], list[dict], dict]:
    center, pools, audit = direction_model(root, rows, anchor, variables, algorithm)
    cont = [v for v in variables if v["dtype"] != "int64"]
    lower = np.array([v["bounds"][0] for v in cont], dtype=float)
    span = np.array([v["bounds"][1] - v["bounds"][0] for v in cont], dtype=float)
    require(radius > 0 and math.isfinite(radius), "Invalid p95 radius")
    require(hard_failure(anchor, variables, algorithm["blockage_thickness_ratio"]) is None, "Invalid anchor geometry")
    coordinates, splits, screening = [], [], []
    for band, (lo, hi) in enumerate(algorithm["bands"]):
        accepted = 0
        for attempt in range(algorithm["maximum_attempts_per_band"]):
            width = algorithm["pool_rows_per_band"]
            pool_id, index = divmod(attempt, width)
            vector = pools[pool_id][band * width + index]
            direction, radial = vector[:16], vector[16]
            margin = algorithm["radial_endpoint_margin"]
            desired_fraction = lo + (hi - lo) * (margin + (1 - 2 * margin) * radial)
            active = np.abs(direction) > 0
            limit = np.min(np.minimum(center[active], 1 - center[active]) / np.abs(direction[active]))
            step = min(desired_fraction * radius, float(limit) * algorithm["bounds_safety_factor"])
            event = {"root_seed_id": root, "band_index": band, "attempt_index": attempt,
                     "pool": "primary" if pool_id == 0 else "reserve", "pool_index": band * width + index,
                     "desired_radius_fraction": desired_fraction, "two_sided_step_normalized_l2": step}
            pair, reason = [], None
            if step <= 0 or not lo < step / radius <= hi:
                reason = "shrunk_step_outside_band"
            else:
                for sign in (1, -1):
                    physical = lower + (center + sign * step * direction) * span
                    g = dict(anchor)
                    g.update({v["name"]: float(physical[j]) for j, v in enumerate(cont)})
                    actual_delta = (physical - np.array([anchor[v["name"]] for v in cont])) / span
                    distance = float(np.linalg.norm(actual_delta) / radius)
                    design_hash = sha256(canonical(g)).hexdigest()
                    reason = hard_failure(g, variables, algorithm["blockage_thickness_ratio"])
                    if reason is None and not lo < distance <= hi:
                        reason = "serialized_distance_outside_band"
                    if reason is None and (design_hash in excluded or any(p["design_hash"] == design_hash for p in pair)):
                        reason = "design_hash_overlap"
                    pair.append({"geometry19": g, "design_hash": design_hash, "sign": sign, "actual_radius_fraction": distance})
                    if reason:
                        break
            event.update({"accepted": reason is None, "rejection_reason": reason,
                          "proposed_design_hashes": [p["design_hash"] for p in pair]})
            if reason is None:
                pair_id = f"{root}:b{band}:p{accepted:03d}"
                split = "validation" if accepted % 5 == 0 else "train"
                event["pair_id"] = pair_id
                for p in pair:
                    p.update({"root_seed_id": root, "band_index": band, "pair_index": accepted,
                              "pair_id": pair_id, "split": split, "anchor_design_hash": sha256(canonical(anchor)).hexdigest(),
                              "support_radius_p95": radius, "source_attempt_index": attempt,
                              "label_status": "PRELABEL_GEOMETRY_ONLY", "protocol_version": VERSION})
                    coordinates.append(p)
                    excluded.add(p["design_hash"])
                splits.append({"pair_id": pair_id, "root_seed_id": root, "band_index": band,
                               "pair_index": accepted, "split": split, "design_hashes": [p["design_hash"] for p in pair]})
                accepted += 1
            screening.append(event)
            if accepted == algorithm["pairs_per_root_band"]:
                break
    return coordinates, splits, screening, audit


def prepare_inputs(root: Path, config: dict) -> tuple[dict, dict, dict]:
    """Hash all frozen inputs; project development records to geometry fields only."""
    paths, hashes = {}, {}
    def verify(role: str, spec: dict) -> Path:
        path = within(root, spec["path"])
        require(path.is_file() and file_hash(path) == spec["sha256"], f"Input hash mismatch: {role}")
        paths[role], hashes[role] = path, spec["sha256"]
        return path
    parent_path = verify("parent_protocol", config["parent_protocol"])
    parent = yaml.safe_load(parent_path.read_text())
    validate_parent(parent)
    locked = json.loads(verify("parent_lock", config["parent_lock"]).read_text())
    require(locked["status"] == PARENT_PASS and locked["protocol_sha256"] == file_hash(parent_path), "Parent lock invalid")
    require(locked["implementation_sha256"] == file_hash(root / "src/ai/geometry19_performance_gated_protocol.py"), "G0a implementation changed")
    require(parent["training_data_plan"]["radius_bands"] == config["algorithm"]["bands"] and
            parent["training_data_plan"]["total_coordinates"] == config["algorithm"]["expected_rows"], "Parent acquisition plan changed")
    runtime = within(root, parent["artifacts"]["runtime_root"])
    for name, entry in locked["artifacts"].items():
        verify("g0a_" + name, {"path": str((runtime / entry["path"]).relative_to(root)), "sha256": entry["sha256"]})
    for role, spec in parent["inputs"].items():
        verify(role, spec)
    for model in locked["frozen_forward_models"]:
        verify(model["arm_id"], {"path": model["model_path"], "sha256": model["model_sha256"]})
    for role, spec in config["additional_inputs"].items():
        verify(role, spec)
    for name in ("training_physics", "splits", "models", "candidates", "closure"):
        require(not (runtime / name).exists(), f"Future stage already exists: {name}")
    return paths, hashes, parent


def audit_coordinates(coordinates: list[dict], splits: list[dict], anchors: dict, stable: dict,
                      variables: list[dict], excluded: set[str], algorithm: dict) -> dict:
    """Recompute invariants from final serialized geometry, independently of proposal steps."""
    seen, by_pair = set(), defaultdict(list)
    counts, split_counts = Counter(), Counter()
    for row in coordinates:
        require(set(row).isdisjoint({"performance", "prediction", "solver", "rho", "training_candidate"}), "Prelabel leakage")
        g, root = row["geometry19"], row["root_seed_id"]
        anchor = anchors[root]["medoid_geometry19"]
        h = sha256(canonical(g)).hexdigest()
        require(h == row["design_hash"] and h not in seen and h not in excluded, "Design overlap/hash failure")
        seen.add(h)
        require(hard_failure(g, variables, algorithm["blockage_thickness_ratio"]) is None, "Hard geometry constraint failed")
        require(all(g[v["name"]] == anchor[v["name"]] for v in variables if v["dtype"] == "int64"), "Integer state changed")
        distance = math.sqrt(sum(((g[v["name"]] - anchor[v["name"]]) / (v["bounds"][1] - v["bounds"][0])) ** 2 for v in variables)) / stable[root]["support_radius_p95_normalized_l2"]
        lo, hi = algorithm["bands"][row["band_index"]]
        require(lo < distance <= hi and abs(distance - row["actual_radius_fraction"]) <= 1e-12, "Actual radius outside assigned band")
        require(row["split"] == ("validation" if row["pair_index"] % 5 == 0 else "train"), "Split rule broken")
        by_pair[row["pair_id"]].append(row)
        counts[(root, row["band_index"]) ] += 1
        split_counts[row["split"]] += 1
    require(len(splits) == len(by_pair) and len({s["pair_id"] for s in splits}) == len(splits), "Pair ledger mismatch")
    for split in splits:
        pair = by_pair[split["pair_id"]]
        require(len(pair) == 2 and {p["sign"] for p in pair} == {-1, 1}, "Incomplete signed pair")
        require([p["design_hash"] for p in pair] == split["design_hashes"] and
                all(p["split"] == split["split"] and p["root_seed_id"] == split["root_seed_id"] and
                    p["band_index"] == split["band_index"] and p["pair_index"] == split["pair_index"] for p in pair), "Pair split leakage")
        anchor = anchors[split["root_seed_id"]]["medoid_geometry19"]
        require(all(abs((pair[0]["geometry19"][v["name"]] + pair[1]["geometry19"][v["name"]]) / 2 - anchor[v["name"]]) /
                    (v["bounds"][1] - v["bounds"][0]) < 1e-12 for v in variables), "Pair not symmetric")
    complete = len(anchors) == 33 and all(counts[(r, band)] == 128 for r in anchors for band in range(4))
    return {"complete": complete, "rows": len(coordinates), "pairs": len(splits), "split_rows": dict(split_counts),
            "unique_hashes": len(seen), "exclusion_overlap": 0, "integer_violations": 0,
            "support_band_violations": 0, "pair_split_violations": 0,
            "per_root_band": {r: [counts[(r, b)] for b in range(4)] for r in sorted(anchors)}}


def run(root: Path, config_path: Path, *, output_root: Path | None = None) -> dict:
    root, config_path = root.resolve(), config_path.resolve()
    c = yaml.safe_load(config_path.read_text())
    validate_protocol(c)
    output = output_root.resolve() if output_root else within(root, c["artifacts"]["output_root"])
    require(not output.exists(), "Refusing to overwrite G0b coordinate artifacts")
    paths, hashes, parent = prepare_inputs(root, c)
    variables = sorted(yaml.safe_load(paths["schema"].read_text())["design_variables"], key=lambda v: v["index"])
    require(len(variables) == 19 and sum(v["dtype"] != "int64" for v in variables) == 16, "Schema dimensions changed")
    calibration = yaml.safe_load(paths["calibration"].read_text())
    def find_values(obj):
        if isinstance(obj, dict):
            if obj.get("name") == "compressor_blade_thickness_ratio":
                yield obj["value"]
            for value in obj.values():
                yield from find_values(value)
        elif isinstance(obj, list):
            for value in obj:
                yield from find_values(value)
    require(list(find_values(calibration)) == [c["algorithm"]["blockage_thickness_ratio"]], "Blockage calibration changed")
    anchors = {r["root_seed_id"]: r for r in read_rows(paths["anchor_registry"])}
    stable = {r["root_seed_id"]: r for r in read_rows(paths["stable_root_registry"])}
    # Source files contain labels; only these geometry/identity fields enter this calculation.
    development = [{k: row[k] for k in ("geometry19", "design_hash", "root_seed_id")} for row in read_rows(paths["development_dataset"])]
    groups = defaultdict(list)
    for row in development:
        require(sha256(canonical(row["geometry19"])).hexdigest() == row["design_hash"], "Development geometry hash changed")
        groups[row["root_seed_id"]].append(row)
    excluded = {r["design_hash"] for r in development}
    for role in ("s3_v1", "s3_v2", "s3_v3", "s3_v4_combined"):
        excluded.update(r["design_hash"] for r in read_rows(paths[role]))
    targets = read_rows(paths["g0a_target_registry"])
    excluded.update(r["source_design_hash"] for r in targets)
    require(len(anchors) == 33 and {r["root_seed_id"] for r in targets} == set(anchors), "Frozen 33-root scope changed")
    initial_excluded = set(excluded)
    coordinates, pairs, screening, directions = [], [], [], []
    for r in sorted(anchors):
        anchor = anchors[r]["medoid_geometry19"]
        require(sha256(canonical(anchor)).hexdigest() == anchors[r]["medoid_design_hash"], "Anchor hash changed")
        rows, split, screened, audit = generate_root(r, anchor, stable[r]["support_radius_p95_normalized_l2"],
                                                   groups[r], variables, c["algorithm"], excluded)
        coordinates.extend(rows)
        pairs.extend(split)
        screening.extend(screened)
        directions.append(audit)
    audit = audit_coordinates(coordinates, pairs, anchors, stable, variables, initial_excluded, c["algorithm"])
    if audit["complete"]:
        require(audit["split_rows"] == {"train": 13464, "validation": 3432}, "Final split counts changed")
    artifact_paths = {k: within(output, c["artifacts"][k]) for k in ("coordinates", "splits", "screening", "direction_audit", "report", "lock")}
    require(len(set(artifact_paths.values())) == len(artifact_paths), "Artifact path collision")
    summary = {"status": PASS if audit["complete"] else FAIL, "protocol_version": VERSION,
               "protocol_sha256": file_hash(config_path), "implementation_sha256": file_hash(Path(__file__)),
               "parent_helper_sha256": file_hash(root / "src/ai/geometry19_performance_gated_protocol.py"),
               "input_hashes": hashes, "environment": c["environment"], "machine": platform.machine(),
               "audit": audit, "screening_pairs": len(screening),
               "rejection_counts": dict(Counter(r["rejection_reason"] for r in screening if not r["accepted"])),
               "reserve_pairs_selected": sum(r["accepted"] and r["pool"] == "reserve" for r in screening),
               "excluded_design_hashes": len(initial_excluded),
               "actual_radius_by_band": {str(b): {"min": min((r["actual_radius_fraction"] for r in coordinates if r["band_index"] == b), default=None),
                                                   "max": max((r["actual_radius_fraction"] for r in coordinates if r["band_index"] == b), default=None)} for b in range(4)},
               "authorization": c["authorization"], "physics_evaluations_executed": False, "model_arrays_loaded": False,
               "model_predictions_executed": False, "confirmation_content_read": False, "performance_values_used_for_selection": False,
               "next_stage": "separate_G0c_physics_acquisition_protocol_and_authorization" if audit["complete"] else "seal_failure_stop",
               "limitations": ["geometry_compliance_does_not_establish_physics_validity", "known_root_development_only",
                               "geometry_statistics_from_existing_development", "same_root_train_validation_are_not_independent_roots"]}
    output.mkdir(parents=True, exist_ok=False)
    summary["artifacts"] = {}
    for key, rows in (("coordinates", coordinates), ("splits", pairs), ("screening", screening), ("direction_audit", directions)):
        payload = canonical(rows) + b"\n" if key == "direction_audit" else b"".join(canonical(r) + b"\n" for r in rows)
        with artifact_paths[key].open("xb") as stream:
            stream.write(payload)
        summary["artifacts"][key] = {"path": c["artifacts"][key], "sha256": file_hash(artifact_paths[key])}
    report = (f"# G0b training coordinate and pair-split lock\n\nStatus: `{summary['status']}`\n\n"
              f"Rows: {len(coordinates)}; pairs: {len(pairs)}; splits: {audit['split_rows']}.\n"
              f"Screened pairs: {len(screening)}; rejection counts: {summary['rejection_counts']}.\n\n"
              "Actual serialized distances satisfy their assigned radial bands; integer axes remain medoid values.\n"
              "Positive/negative pairs share one split. Existing development, prior S3 and frozen target designs excluded.\n"
              "No physics calls, training, model predictions or confirmation parsing.\n\n"
              "Geometry coverage does not establish V2.2 physical validity. Physics acquisition needs its own protocol and authorization.\n")
    with artifact_paths["report"].open("x") as stream:
        stream.write(report)
    summary["artifacts"]["report"] = {"path": c["artifacts"]["report"], "sha256": file_hash(artifact_paths["report"])}
    with artifact_paths["lock"].open("xb") as stream:
        stream.write(canonical(summary) + b"\n")
    return summary
