"""Generate development-only S3 v4 coordinates around physical medoids."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import qmc
import yaml

from src.ai.geometry19_surrogate import sha256_file


PROTOCOL_VERSION = "geometry19_v2_2_stable_root_s3_v4_medoid_covariance_prelabel_v1"
PASS_STATUS = "STABLE_ROOT_S3_V4_MEDOID_COVARIANCE_COORDINATE_LOCK_PASS"
FAIL_STATUS = "STABLE_ROOT_S3_V4_MEDOID_COVARIANCE_COORDINATE_LOCK_FAIL"


class Geometry19StableRootS3V4Error(ValueError):
    """Raised when the frozen v4 coordinate contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Geometry19StableRootS3V4Error(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_text(path, "".join(_canonical_bytes(row).decode("utf-8") + "\n" for row in rows))


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def validate_protocol(config: Mapping[str, Any]) -> None:
    _require(config.get("protocol_version") == PROTOCOL_VERSION, "S3 v4 protocol mismatch")
    _require(config.get("stage") == "S3_v4_33root_medoid_covariance_development_coordinate_generation", "S3 v4 stage mismatch")
    _require(config.get("scope_definition") == "post_selected_33root_development_only", "S3 v4 scope changed")
    _require(config.get("formal_use_allowed") is False, "Formal use must remain forbidden")
    _require(config.get("deployment_allowed") is False, "Deployment must remain forbidden")
    anchor = config.get("anchor", {})
    _require(anchor.get("source") == "frozen_stable_root_registry_medoid_design_hash", "Anchor source changed")
    _require(anchor.get("arithmetic_mean_center_forbidden") is True, "Arithmetic-mean anchor cannot be enabled")
    model = config.get("local_direction_model", {})
    _require(model.get("performance_values_read") is False, "Performance cannot shape v4 directions")
    _require(model.get("shrinkage_fraction_to_isotropic_trace") == 0.05, "Covariance shrinkage changed")
    _require(model.get("primary_count_per_root") == 48, "Primary count changed")
    _require(model.get("primary_seed_base") == 20260919, "Primary seed changed")
    _require(model.get("reserve_count_per_root") == 256, "Reserve count changed")
    _require(model.get("reserve_seed_base") == 20260920, "Reserve seed changed")
    scaling = config.get("scaling", {})
    _require(scaling.get("shell_fractions") == [0.10, 0.20], "Radius shells changed")
    _require(scaling.get("target_pairs_per_shell_per_root") == 24, "Shell target changed")
    _require(scaling.get("minimum_actual_to_desired_step_ratio") == 0.25, "Minimum step ratio changed")
    _require(scaling.get("clipping_allowed") is False, "Clipping cannot be allowed")
    selection = config.get("selection", {})
    _require(selection.get("selected_roots") == 33, "Root count changed")
    _require(selection.get("selected_pairs_per_root") == 48, "Pair count changed")
    _require(selection.get("final_rows_total") == 3168, "Row count changed")
    _require(selection.get("performance_adaptive_selection") is False, "Performance adaptation cannot be allowed")
    authorization = config.get("authorization", {})
    _require(authorization.get("coordinate_generation_authorized_now") is True, "Coordinate generation not authorized")
    _require(authorization.get("physics_evaluation_executed_now") is False, "Physics cannot run now")
    _require(authorization.get("model_prediction_authorized") is False, "Prediction cannot run now")
    _require(config["inputs"]["sealed_confirmation"]["access"] == "sha256_only_no_parse", "Confirmation access changed")
    _require(config["inputs"]["s2_model_manifest"]["access"] == "sha256_only_no_parse", "Model access changed")


def _schema_arrays(schema: Mapping[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray, set[str]]:
    variables = sorted(schema["design_variables"], key=lambda row: int(row["index"]))
    names = [str(row["name"]) for row in variables]
    lower = np.asarray([float(row["bounds"][0]) for row in variables])
    upper = np.asarray([float(row["bounds"][1]) for row in variables])
    integer_names = {str(row["name"]) for row in variables if str(row.get("dtype")) == "int64"}
    _require(len(names) == 19 and len(integer_names) == 3, "Frozen schema differs")
    return names, lower, upper, integer_names


def _normalized(row: Mapping[str, Any], names: Sequence[str], lower: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = np.asarray([float(row["geometry19"][name]) for name in names])
    return (values - lower) / scale


def _geometry(normalized: np.ndarray, names: Sequence[str], lower: np.ndarray, scale: np.ndarray, integer_names: set[str]) -> dict[str, Any]:
    physical = lower + normalized * scale
    return {
        name: int(np.rint(physical[index])) if name in integer_names else float(physical[index])
        for index, name in enumerate(names)
    }


def _hashes(path: Path) -> set[str]:
    return {str(row["design_hash"]) for row in _read_jsonl(path) if row.get("design_hash")}


def _root_seed(base: int, root_id: str) -> int:
    return (int(base) ^ int(root_id[:8], 16)) % (2**32)


def build_root_direction_model(
    root_id: str,
    anchor: np.ndarray,
    root_rows: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    lower: np.ndarray,
    scale: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Fit the frozen local covariance and transform root-specific Sobol pools."""

    unique = {str(row["design_hash"]): row for row in root_rows}
    x = np.asarray([_normalized(row, names, lower, scale) for row in unique.values()])
    _require(x.shape[0] >= 24 and x.shape[1] == 19, f"Insufficient root rows: {root_id}")
    deltas = x - anchor[None, :]
    covariance = deltas.T @ deltas / float(len(deltas) - 1)
    trace_mean = float(np.trace(covariance) / 19.0)
    _require(np.isfinite(trace_mean) and trace_mean > 0.0, f"Degenerate root covariance: {root_id}")
    fraction = float(config["local_direction_model"]["shrinkage_fraction_to_isotropic_trace"])
    covariance = (1.0 - fraction) * covariance + fraction * trace_mean * np.eye(19)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    square_root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T

    definition = config["local_direction_model"]
    primary_engine = qmc.Sobol(d=19, scramble=True, seed=_root_seed(int(definition["primary_seed_base"]), root_id))
    primary_latent = 2.0 * primary_engine.random_base2(m=6)[:48] - 1.0
    reserve_engine = qmc.Sobol(d=19, scramble=True, seed=_root_seed(int(definition["reserve_seed_base"]), root_id))
    reserve_latent = 2.0 * reserve_engine.random_base2(m=8) - 1.0
    primary = primary_latent @ square_root.T
    reserve = reserve_latent @ square_root.T
    primary /= np.linalg.norm(primary, axis=1)[:, None]
    reserve /= np.linalg.norm(reserve, axis=1)[:, None]
    return {
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "primary_directions": primary,
        "primary_shell": np.asarray([0.10 if index % 2 == 0 else 0.20 for index in range(48)]),
        "reserve_directions": reserve,
        "reserve_shell": np.asarray([0.10 if index % 2 == 0 else 0.20 for index in range(256)]),
    }


def _common_limit(center: np.ndarray, direction: np.ndarray) -> float:
    active = np.abs(direction) > 1.0e-15
    return float(np.min(np.minimum(center[active], 1.0 - center[active]) / np.abs(direction[active])))


def _report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Stable-Root S3 v4 Medoid/Covariance Prelabel Coordinates",
            "",
            f"- Status: `{summary['status']}`",
            f"- Roots: {summary['selected_root_count']}",
            f"- Anchors verified as existing V2.2-eligible medoids: {summary['eligible_medoid_anchors']}",
            f"- Screening pairs considered: {summary['screening_pairs_considered']}",
            f"- Screening pairs rejected: {summary['screening_pairs_rejected']}",
            f"- Reserve pairs selected: {summary['reserve_pairs_selected']}",
            f"- Selected pairs: {summary['selected_pairs']}",
            f"- Final rows: {summary['final_prelabel_rows']}",
            f"- Future physics protocol authorized: `{str(summary['future_physics_protocol_authorized']).lower()}`",
            "- Physics/model predictions executed: `false`",
            "- Confirmation/model manifest content read: `false`",
            "",
            "This is adaptive development work after the observed S3 v3 physics failure. The coordinates are not a fresh confirmation set and do not establish target-domain retention.",
            "",
        ]
    )


def generate_s3_v4_medoid_coordinates(
    repo_root: Path,
    config_path: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Generate new medoid-centered, covariance-shaped coordinates only."""

    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_protocol(config)
    parsed_roles = {
        "v3_physics_failure_summary",
        "v3_scope_registry",
        "stable_root_registry",
        "design_schema",
        "development_dataset",
        "v1_coordinate_ledger",
        "v2_coordinate_ledger",
        "v3_coordinate_ledger",
    }
    resolved: dict[str, Path] = {}
    input_audit: dict[str, Any] = {}
    for role, item in config["inputs"].items():
        path = (repo_root / str(item["path"])).resolve()
        _require(path.is_file(), f"Missing v4 input: {role}")
        actual = sha256_file(path)
        _require(actual == str(item["sha256"]), f"v4 input hash mismatch: {role}")
        resolved[role] = path
        input_audit[role] = {
            "path": str(item["path"]),
            "expected_sha256": str(item["sha256"]),
            "actual_sha256": actual,
            "match": True,
            "content_parsed": role in parsed_roles,
        }
    failure = json.loads(resolved["v3_physics_failure_summary"].read_text(encoding="utf-8"))
    _require(failure.get("status") == config["inputs"]["v3_physics_failure_summary"]["required_status"], "v3 failure status mismatch")
    _require(failure.get("model_scoring_authorized") is False, "v3 unexpectedly authorized scoring")
    scope = json.loads(resolved["v3_scope_registry"].read_text(encoding="utf-8"))
    selected_ids = sorted(str(value) for value in scope["selected_root_ids"])
    _require(len(selected_ids) == len(set(selected_ids)) == 33, "Frozen v3 scope differs")
    registry = {str(row["root_seed_id"]): row for row in _read_jsonl(resolved["stable_root_registry"])}
    _require(set(selected_ids) <= set(registry), "Selected root missing from registry")
    schema = yaml.safe_load(resolved["design_schema"].read_text(encoding="utf-8"))
    names, lower, upper, integer_names = _schema_arrays(schema)
    scale = upper - lower
    development = _read_jsonl(resolved["development_dataset"])
    development_by_hash = {str(row["design_hash"]): row for row in development}
    development_by_root: dict[str, list[dict[str, Any]]] = {}
    for row in development:
        root_id = str(row["root_seed_id"])
        if root_id in selected_ids:
            development_by_root.setdefault(root_id, []).append(row)
    prior_hashes = set(development_by_hash)
    prior_hashes.update(_hashes(resolved["v1_coordinate_ledger"]))
    prior_hashes.update(_hashes(resolved["v2_coordinate_ledger"]))
    prior_hashes.update(_hashes(resolved["v3_coordinate_ledger"]))

    anchors: list[dict[str, Any]] = []
    models: dict[str, dict[str, np.ndarray]] = {}
    anchor_vectors: dict[str, np.ndarray] = {}
    for root_id in selected_ids:
        root = registry[root_id]
        medoid_hash = str(root["medoid"]["design_hash"])
        _require(medoid_hash in development_by_hash, f"Medoid absent from development: {root_id}")
        medoid = development_by_hash[medoid_hash]
        _require(str(medoid["root_seed_id"]) == root_id, f"Medoid root mismatch: {root_id}")
        _require(bool(medoid.get("training_candidate")), f"Medoid is not V2.2 eligible: {root_id}")
        _require(bool(medoid.get("constraint_assessment", {}).get("physics_domain_valid")), f"Medoid physics invalid: {root_id}")
        _require(bool(medoid.get("target_domain_assessment", {}).get("inside")), f"Medoid outside target domain: {root_id}")
        anchor = _normalized(medoid, names, lower, scale)
        model = build_root_direction_model(root_id, anchor, development_by_root[root_id], names, lower, scale, config)
        anchor_vectors[root_id] = anchor
        models[root_id] = model
        anchors.append(
            {
                "root_seed_id": root_id,
                "medoid_design_hash": medoid_hash,
                "medoid_sample_id": medoid["sample_id"],
                "medoid_geometry19": medoid["geometry19"],
                "normalized_medoid": anchor.tolist(),
                "development_rows": len(development_by_root[root_id]),
                "unique_development_designs": len({row["design_hash"] for row in development_by_root[root_id]}),
                "covariance_eigenvalues": model["eigenvalues"].tolist(),
                "covariance_effective_rank": int(np.linalg.matrix_rank(model["covariance"])),
                "existing_V2_2_training_candidate": True,
            }
        )

    eta = float(config["scaling"]["common_two_sided_schema_safety_factor"])
    minimum_ratio = float(config["scaling"]["minimum_actual_to_desired_step_ratio"])
    selected_hashes: set[str] = set()
    screening: list[dict[str, Any]] = []
    selected_pairs: list[dict[str, Any]] = []
    for root_id in selected_ids:
        root = registry[root_id]
        center = anchor_vectors[root_id]
        radius = float(root["support_radius_p95_normalized_l2"])
        model = models[root_id]
        for shell in (0.10, 0.20):
            candidates: list[tuple[str, int, np.ndarray]] = []
            for index in np.flatnonzero(model["primary_shell"] == shell).tolist():
                candidates.append(("primary", index, model["primary_directions"][index]))
            for index in np.flatnonzero(model["reserve_shell"] == shell).tolist():
                candidates.append(("reserve", index, model["reserve_directions"][index]))
            shell_selected = 0
            for pool, pool_index, direction in candidates:
                if shell_selected >= 24:
                    break
                desired_step = radius * shell
                actual_step = min(desired_step, eta * _common_limit(center, direction))
                ratio = actual_step / desired_step if desired_step > 0.0 else 0.0
                plus_raw = center + actual_step * direction
                minus_raw = center - actual_step * direction
                plus_geometry = _geometry(plus_raw, names, lower, scale, integer_names)
                minus_geometry = _geometry(minus_raw, names, lower, scale, integer_names)
                plus_hash = sha256(_canonical_bytes(plus_geometry)).hexdigest()
                minus_hash = sha256(_canonical_bytes(minus_geometry)).hexdigest()
                reasons: list[str] = []
                if ratio < minimum_ratio:
                    reasons.append("two_sided_safe_step_ratio_below_minimum")
                if not (np.all((plus_raw >= 0.0) & (plus_raw <= 1.0)) and np.all((minus_raw >= 0.0) & (minus_raw <= 1.0))):
                    reasons.append("schema_bounds_after_scaling")
                if plus_hash == minus_hash:
                    reasons.append("pair_design_hashes_identical")
                if plus_hash in prior_hashes or minus_hash in prior_hashes:
                    reasons.append("development_or_S3_v1_v2_v3_design_hash_duplicate")
                if plus_hash in selected_hashes or minus_hash in selected_hashes:
                    reasons.append("previously_selected_v4_design_hash_duplicate")
                accepted = not reasons
                identity = {
                    "root_seed_id": root_id,
                    "shell": shell,
                    "pool": pool,
                    "pool_index": pool_index,
                    "plus_hash": plus_hash,
                    "minus_hash": minus_hash,
                }
                row = {
                    "screening_id": sha256(_canonical_bytes(identity)).hexdigest(),
                    "root_seed_id": root_id,
                    "anchor_design_hash": str(root["medoid"]["design_hash"]),
                    "source_root_segments": root["segments"],
                    "shell_fraction": shell,
                    "direction_pool": pool,
                    "pool_index": pool_index,
                    "direction_family": "root_local_covariance_sobol",
                    "desired_step_normalized_l2": desired_step,
                    "actual_step_normalized_l2": actual_step,
                    "actual_to_desired_step_ratio": ratio,
                    "screening_status": "PAIR_SELECTED" if accepted else "PAIR_REJECTED",
                    "rejection_reasons": reasons,
                    "plus_design_hash": plus_hash,
                    "minus_design_hash": minus_hash,
                    "label_status": "NOT_EVALUATED_DEVELOPMENT_COORDINATE_ONLY",
                }
                screening.append(row)
                if accepted:
                    selected_hashes.update((plus_hash, minus_hash))
                    selected_pairs.append({**row, "plus_raw": plus_raw, "minus_raw": minus_raw, "plus_geometry": plus_geometry, "minus_geometry": minus_geometry})
                    shell_selected += 1
            if shell_selected < 24:
                screening.append(
                    {
                        "screening_id": sha256(_canonical_bytes({"root_seed_id": root_id, "shell": shell, "failure": "reserve_exhausted"})).hexdigest(),
                        "root_seed_id": root_id,
                        "anchor_design_hash": str(root["medoid"]["design_hash"]),
                        "source_root_segments": root["segments"],
                        "shell_fraction": shell,
                        "direction_pool": "none",
                        "pool_index": None,
                        "direction_family": "none",
                        "desired_step_normalized_l2": radius * shell,
                        "actual_step_normalized_l2": None,
                        "actual_to_desired_step_ratio": None,
                        "screening_status": "SHELL_SELECTION_INCOMPLETE",
                        "rejection_reasons": ["reserve_directions_exhausted"],
                        "plus_design_hash": None,
                        "minus_design_hash": None,
                        "label_status": "NOT_EVALUATED_DEVELOPMENT_COORDINATE_ONLY",
                    }
                )

    by_root: dict[str, list[dict[str, Any]]] = {}
    for pair in selected_pairs:
        by_root.setdefault(str(pair["root_seed_id"]), []).append(pair)
    final_rows: list[dict[str, Any]] = []
    for root_id in selected_ids:
        for pair_index, pair in enumerate(by_root.get(root_id, [])):
            for sign, raw_key, geometry_key, hash_key in (
                (1, "plus_raw", "plus_geometry", "plus_design_hash"),
                (-1, "minus_raw", "minus_geometry", "minus_design_hash"),
            ):
                raw = pair[raw_key]
                proposal_hash = sha256(_canonical_bytes({"protocol_version": PROTOCOL_VERSION, "screening_id": pair["screening_id"], "root_seed_id": root_id, "pair_index": pair_index, "sign": sign, "raw": raw.tolist()})).hexdigest()
                final_rows.append(
                    {
                        "attempt_id": f"s3v4:{root_id[:12]}:{pair_index:02d}:{'p' if sign > 0 else 'm'}:{proposal_hash[:12]}",
                        "proposal_hash": proposal_hash,
                        "screening_id": pair["screening_id"],
                        "root_seed_id": root_id,
                        "anchor_design_hash": pair["anchor_design_hash"],
                        "source_root_segments": pair["source_root_segments"],
                        "selected_pair_index": pair_index,
                        "direction_pool": pair["direction_pool"],
                        "pool_index": pair["pool_index"],
                        "direction_family": pair["direction_family"],
                        "radius_fraction": pair["shell_fraction"],
                        "desired_step_normalized_l2": pair["desired_step_normalized_l2"],
                        "actual_step_normalized_l2": pair["actual_step_normalized_l2"],
                        "actual_to_desired_step_ratio": pair["actual_to_desired_step_ratio"],
                        "sign": sign,
                        "raw_normalized_geometry19": {name: float(raw[index]) for index, name in enumerate(names)},
                        "geometry19": pair[geometry_key],
                        "design_hash": pair[hash_key],
                        "coordinate_status": "COORDINATE_ELIGIBLE",
                        "label_status": "NOT_EVALUATED_DEVELOPMENT_COORDINATE_ONLY",
                    }
                )

    pair_counts = Counter(str(row["root_seed_id"]) for row in selected_pairs)
    shell_counts = Counter((str(row["root_seed_id"]), str(row["shell_fraction"])) for row in selected_pairs)
    hashes = [str(row["design_hash"]) for row in final_rows]
    exact_pairs = set(pair_counts) == set(selected_ids) and set(pair_counts.values()) == {48}
    exact_shells = all(shell_counts[(root_id, "0.1")] == 24 and shell_counts[(root_id, "0.2")] == 24 for root_id in selected_ids)
    unique = len(hashes) == len(set(hashes))
    no_prior_overlap = not (set(hashes) & prior_hashes)
    inside = all(all(0.0 <= float(value) <= 1.0 for value in row["raw_normalized_geometry19"].values()) for row in final_rows)
    integers = all(isinstance(row["geometry19"][name], int) for row in final_rows for name in integer_names)
    gate_pass = all((exact_pairs, exact_shells, len(final_rows) == 3168, unique, no_prior_overlap, inside, integers))
    status = PASS_STATUS if gate_pass else FAIL_STATUS

    runtime_root = output_root.resolve() if output_root is not None else (repo_root / str(config["artifacts"]["runtime_root"])).resolve()
    anchor_path = runtime_root / str(config["artifacts"]["anchor_registry"])
    direction_path = runtime_root / str(config["artifacts"]["direction_matrix"])
    screening_path = runtime_root / str(config["artifacts"]["screening_ledger"])
    final_path = runtime_root / str(config["artifacts"]["final_prelabel_ledger"])
    summary_path = runtime_root / str(config["artifacts"]["summary"])
    report_path = runtime_root / str(config["artifacts"]["report"])
    _write_jsonl(anchor_path, anchors)
    _write_npz(
        direction_path,
        root_ids=np.asarray(selected_ids),
        primary_directions=np.asarray([models[root_id]["primary_directions"] for root_id in selected_ids]),
        primary_shell=models[selected_ids[0]]["primary_shell"],
        reserve_directions=np.asarray([models[root_id]["reserve_directions"] for root_id in selected_ids]),
        reserve_shell=models[selected_ids[0]]["reserve_shell"],
        covariance=np.asarray([models[root_id]["covariance"] for root_id in selected_ids]),
        eigenvalues=np.asarray([models[root_id]["eigenvalues"] for root_id in selected_ids]),
    )
    _write_jsonl(screening_path, screening)
    _write_jsonl(final_path, final_rows)
    rejection_reasons = Counter(reason for row in screening for reason in row["rejection_reasons"])
    summary: dict[str, Any] = {
        "status": status,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope_definition": config["scope_definition"],
        "adaptive_development_after_v3_failure": True,
        "not_fresh_confirmation": True,
        "selected_root_count": len(selected_ids),
        "eligible_medoid_anchors": len(anchors),
        "protocol_path": _relative(config_path, repo_root),
        "protocol_sha256": sha256_file(config_path),
        "implementation_path": _relative(Path(__file__), repo_root),
        "implementation_sha256": sha256_file(Path(__file__)),
        "input_hash_audit": input_audit,
        "all_input_hashes_match": True,
        "v3_physics_values_used_for_direction_selection": False,
        "development_performance_values_used_for_direction_selection": False,
        "screening_pairs_considered": len(screening),
        "screening_pairs_rejected": sum(row["screening_status"] == "PAIR_REJECTED" for row in screening),
        "screening_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "primary_pairs_selected": sum(row["screening_status"] == "PAIR_SELECTED" and row["direction_pool"] == "primary" for row in screening),
        "reserve_pairs_selected": sum(row["screening_status"] == "PAIR_SELECTED" and row["direction_pool"] == "reserve" for row in screening),
        "selected_pairs": len(selected_pairs),
        "selected_pairs_per_root": dict(sorted(pair_counts.items())),
        "selected_pairs_per_root_shell": {f"{root_id}:{shell}": count for (root_id, shell), count in sorted(shell_counts.items())},
        "minimum_selected_pairs_per_root": min(pair_counts.values()) if pair_counts else 0,
        "final_prelabel_rows": len(final_rows),
        "final_design_hashes_unique": unique,
        "prior_dataset_or_S3_overlap": not no_prior_overlap,
        "all_final_coordinates_inside_schema": inside,
        "all_integer_variables_stored_as_integers": integers,
        "physics_evaluations_executed": False,
        "performance_labels_written": False,
        "model_manifest_content_read": False,
        "model_files_opened": False,
        "model_predictions_executed": False,
        "confirmation_content_read": False,
        "future_physics_protocol_authorized": gate_pass,
        "inverse_design_authorized": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "next_authorized_stage": "freeze_v4_development_physics_acquisition_protocol" if gate_pass else "stop_and_review_v4_coordinate_failure",
        "artifacts": {
            "anchor_registry": _relative(anchor_path, repo_root),
            "anchor_registry_sha256": sha256_file(anchor_path),
            "direction_matrix": _relative(direction_path, repo_root),
            "direction_matrix_sha256": sha256_file(direction_path),
            "screening_ledger": _relative(screening_path, repo_root),
            "screening_ledger_sha256": sha256_file(screening_path),
            "final_prelabel_ledger": _relative(final_path, repo_root),
            "final_prelabel_ledger_sha256": sha256_file(final_path),
        },
    }
    _write_text(report_path, _report(summary))
    summary["artifacts"]["report"] = _relative(report_path, repo_root)
    summary["artifacts"]["report_sha256"] = sha256_file(report_path)
    _write_json(summary_path, summary)
    return summary
