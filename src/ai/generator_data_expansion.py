"""WP2: expand, clean, version, and freeze generator evaluation targets."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

from .baseline import (
    collect_environment,
    collect_git_state,
    collect_source_manifest,
    sha256_file,
    write_json,
    write_yaml,
)
from .dataset import (
    DesignSpaceBounds,
    evaluate_design,
    latin_hypercube_sample,
    scale_samples,
)
from .generator_dataset import (
    GeneratorNormalizer,
    assign_splits,
    build_generator_views,
    condition_names,
    design_names,
    engine_inputs_from_cycle_row,
    load_schema,
    run_parity_check,
    validate_schema_against_design_bounds,
)
from ..physics.brayton import FlightConditions, solve_brayton_cycle


INTEGER_DESIGN_FIELDS = {
    "compressor_blade_count",
    "combustor_num_injectors",
    "turbine_blade_count",
}


def load_wp2_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "version",
        "dataset_version",
        "schema",
        "sampling",
        "output",
        "cleaning",
        "split",
        "coverage",
        "targets",
        "parity",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"WP2 config is missing keys: {sorted(missing)}")
    sampling = config["sampling"]
    if sampling.get("method") != "global_latin_hypercube":
        raise ValueError("WP2 only supports global_latin_hypercube sampling")
    if int(sampling["n_samples"]) <= 0 or int(sampling["shard_size"]) <= 0:
        raise ValueError("sampling.n_samples and shard_size must be positive")
    fractions = sum(
        float(config["split"][name])
        for name in ("train_fraction", "val_fraction", "test_fraction")
    )
    if not math.isclose(fractions, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("WP2 split fractions must sum to one")
    for key in (
        "id_count",
        "interpolation_count",
        "ood_feasible_unknown_count",
        "infeasible_count",
    ):
        if int(config["targets"][key]) < 0:
            raise ValueError(f"targets.{key} cannot be negative")
    return config


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _design_space_contract() -> Dict[str, Any]:
    bounds = DesignSpaceBounds()
    fields = []
    for name in bounds.__dataclass_fields__:
        lower, upper = getattr(bounds, name)
        fields.append(
            {
                "name": name,
                "bounds": [float(lower), float(upper)],
                "kind": "integer" if name in INTEGER_DESIGN_FIELDS else "continuous",
            }
        )
    return {"class": "DesignSpaceBounds", "fields": fields}


def _raw_sample_id(
    dataset_version: str,
    global_index: int,
    row: Mapping[str, Any],
    names: Sequence[str],
) -> str:
    payload = {
        "dataset_version": dataset_version,
        "global_sample_index": int(global_index),
        "design": [[name, float(row[name]).hex()] for name in names],
    }
    return _canonical_hash(payload)


def create_global_lhs_inputs(
    n_samples: int,
    seed: int,
    dataset_version: str,
) -> pd.DataFrame:
    """Create one global LHS before slicing it into restartable shards."""
    bounds = DesignSpaceBounds()
    names = list(bounds.__dataclass_fields__)
    unit = latin_hypercube_sample(n_samples, len(names), seed=seed)
    inputs = scale_samples(unit, bounds)
    inputs.insert(0, "global_sample_index", np.arange(n_samples, dtype=np.int64))
    inputs.insert(
        1,
        "raw_sample_id",
        [
            _raw_sample_id(dataset_version, index, inputs.iloc[index], names)
            for index in range(n_samples)
        ],
    )
    return inputs


def _evaluate_input_shard(inputs: pd.DataFrame) -> pd.DataFrame:
    design_columns = list(DesignSpaceBounds().__dataclass_fields__)
    outputs = []
    for _, row in inputs.iterrows():
        result = evaluate_design(row)
        result["evaluation_status"] = "success" if result["n_warnings"] >= 0 else "exception"
        if result["n_warnings"] < 0:
            result["evaluation_failure_code"] = "EVALUATION_EXCEPTION_UNKNOWN"
        elif not bool(result["is_valid"]):
            result["evaluation_failure_code"] = "PHYSICS_INVALID"
        else:
            result["evaluation_failure_code"] = "OK"
        outputs.append(result)
    output_frame = pd.DataFrame(outputs)
    return pd.concat(
        [inputs[["global_sample_index", "raw_sample_id"] + design_columns].reset_index(drop=True),
         output_frame.reset_index(drop=True)],
        axis=1,
    )


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _valid_existing_shard(
    csv_path: Path,
    meta_path: Path,
    config_hash: str,
    start: int,
    stop: int,
) -> bool:
    if not csv_path.is_file() or not meta_path.is_file():
        return False
    try:
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        return bool(
            meta["status"] == "complete"
            and meta["config_sha256"] == config_hash
            and int(meta["start_index"]) == start
            and int(meta["stop_index_exclusive"]) == stop
            and int(meta["row_count"]) == stop - start
            and meta["csv_sha256"] == sha256_file(csv_path)
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError):
        return False


def generate_raw_shards(
    inputs: pd.DataFrame,
    raw_dir: Path,
    config: Mapping[str, Any],
    config_hash: str,
    schema_hash: str,
    bounds_hash: str,
    git_commit: str,
) -> Tuple[List[Path], int]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    n_samples = len(inputs)
    shard_size = int(config["sampling"]["shard_size"])
    resume = bool(config["output"].get("resume", True))
    paths: List[Path] = []
    reused = 0
    for shard_index, start in enumerate(range(0, n_samples, shard_size)):
        stop = min(start + shard_size, n_samples)
        csv_path = raw_dir / f"part-{shard_index:05d}.csv"
        meta_path = raw_dir / f"part-{shard_index:05d}.meta.yaml"
        if resume and _valid_existing_shard(
            csv_path, meta_path, config_hash, start, stop
        ):
            paths.append(csv_path)
            reused += 1
            continue
        shard = _evaluate_input_shard(inputs.iloc[start:stop].copy())
        _atomic_csv(shard, csv_path)
        write_yaml(
            meta_path,
            {
                "status": "complete",
                "shard_index": shard_index,
                "start_index": start,
                "stop_index_exclusive": stop,
                "row_count": len(shard),
                "sampling_seed": int(config["sampling"]["seed"]),
                "config_sha256": config_hash,
                "schema_sha256": schema_hash,
                "design_bounds_sha256": bounds_hash,
                "git_commit": git_commit,
                "columns": list(shard.columns),
                "dtypes": {name: str(dtype) for name, dtype in shard.dtypes.items()},
                "csv_sha256": sha256_file(csv_path),
            },
        )
        paths.append(csv_path)
    return paths, reused


def merge_raw_shards(
    shard_paths: Sequence[Path],
    expected_count: int,
    duplicate_key: str,
    output_path: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not shard_paths:
        raise ValueError("No WP2 shards were produced")
    frames = []
    expected_columns: Optional[List[str]] = None
    for path in shard_paths:
        frame = pd.read_csv(path)
        columns = list(frame.columns)
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise ValueError(f"Raw shard schema mismatch: {path}")
        frames.append(frame)
    merged_before = pd.concat(frames, ignore_index=True)
    duplicate_count = int(merged_before.duplicated(duplicate_key).sum())
    merged = merged_before.drop_duplicates(duplicate_key, keep="first").copy()
    merged.sort_values("global_sample_index", inplace=True, ignore_index=True)
    expected_indices = np.arange(expected_count, dtype=np.int64)
    actual_indices = merged["global_sample_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError("Merged shards do not cover each global sample index exactly once")
    if merged[duplicate_key].duplicated().any():
        raise ValueError(f"Duplicate {duplicate_key} remains after merge")
    _atomic_csv(merged, output_path)
    return merged, {
        "rows_before_deduplication": len(merged_before),
        "duplicate_rows_removed": duplicate_count,
        "rows_after_deduplication": len(merged),
        "unique_sample_ids": int(merged[duplicate_key].nunique()),
        "merged_csv_sha256": sha256_file(output_path),
    }


def apply_cleaning_rules(
    raw: pd.DataFrame,
    schema: Mapping[str, Any],
    rules: Mapping[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    x_names = design_names(schema)
    c_names = condition_names(schema)
    cleaned = raw.copy()
    valid_flags: List[bool] = []
    failure_codes: List[str] = []
    for _, row in cleaned.iterrows():
        code = "OK"
        if bool(rules["require_source_valid"]) and not bool(row["is_valid"]):
            code = "SOURCE_INVALID"
        elif bool(rules["require_finite_design"]) and not np.isfinite(
            row[x_names].to_numpy(dtype=float)
        ).all():
            code = "NONFINITE_DESIGN"
        elif bool(rules["require_finite_conditions"]) and not np.isfinite(
            row[c_names].to_numpy(dtype=float)
        ).all():
            code = "NONFINITE_CONDITION"
        elif float(row["thrust_N"]) <= float(rules["thrust_min_exclusive"]):
            code = "THRUST_BELOW_MIN"
        elif rules.get("thrust_max_inclusive") is not None and float(
            row["thrust_N"]
        ) > float(rules["thrust_max_inclusive"]):
            code = "THRUST_ABOVE_MAX"
        elif float(row["tsfc_kg_N_s"]) <= float(rules["tsfc_min_exclusive"]):
            code = "TSFC_BELOW_MIN"
        elif rules.get("tsfc_max_inclusive") is not None and float(
            row["tsfc_kg_N_s"]
        ) > float(rules["tsfc_max_inclusive"]):
            code = "TSFC_ABOVE_MAX"
        valid_flags.append(code == "OK")
        failure_codes.append(code)
    cleaned["clean_is_valid"] = valid_flags
    cleaned["cleaning_failure_code"] = failure_codes
    counts = cleaned["cleaning_failure_code"].value_counts().to_dict()
    report = {
        "rules": dict(rules),
        "all_rows": len(cleaned),
        "valid_rows": int(cleaned["clean_is_valid"].sum()),
        "invalid_rows": int((~cleaned["clean_is_valid"]).sum()),
        "failure_codes": {str(key): int(value) for key, value in counts.items()},
        "invalid_rows_retained": True,
    }
    return cleaned, report


def _expected_integer_bins(lower: float, upper: float, bins: int) -> set[int]:
    values = np.arange(math.ceil(lower), math.floor(upper) + 1, dtype=float)
    indices = np.floor((values - lower) / (upper - lower) * bins).astype(int)
    return set(np.clip(indices, 0, bins - 1).tolist())


def build_coverage_report(
    raw: pd.DataFrame,
    valid_view: pd.DataFrame,
    bins: int,
    output_dir: Path,
) -> Dict[str, Any]:
    bounds = DesignSpaceBounds()
    bin_records = []
    dimensions: Dict[str, Any] = {}
    all_pass = True
    for name in bounds.__dataclass_fields__:
        lower, upper = (float(value) for value in getattr(bounds, name))
        counts, edges = np.histogram(
            raw[name].to_numpy(dtype=float), bins=bins, range=(lower, upper)
        )
        occupied = set(np.flatnonzero(counts > 0).tolist())
        if name in INTEGER_DESIGN_FIELDS:
            expected = _expected_integer_bins(lower, upper, bins)
        else:
            expected = set(range(bins))
        unexpected_empty = sorted(expected.difference(occupied))
        passed = not unexpected_empty
        all_pass = all_pass and passed
        dimensions[name] = {
            "kind": "integer" if name in INTEGER_DESIGN_FIELDS else "continuous",
            "minimum": float(raw[name].min()),
            "maximum": float(raw[name].max()),
            "unique_count": int(raw[name].nunique()),
            "expected_occupied_bins": len(expected),
            "occupied_bins": len(occupied),
            "unexpected_empty_bins": unexpected_empty,
            "pass": passed,
        }
        for index, count in enumerate(counts):
            bin_records.append(
                {
                    "variable": name,
                    "bin_index": index,
                    "left": float(edges[index]),
                    "right": float(edges[index + 1]),
                    "count": int(count),
                    "expected_occupied": index in expected,
                }
            )
    pd.DataFrame(bin_records).to_csv(output_dir / "coverage_bins.csv", index=False)
    generator_names = list(valid_view.columns.intersection([
        "compressor_pressure_ratio",
        "compressor_efficiency",
        "turbine_inlet_temp_K",
        "turbine_efficiency",
        "mass_flow_kg_s",
        "thrust_N",
        "tsfc_kg_N_s",
    ]))
    correlation = valid_view[generator_names].corr().to_dict()
    report = {
        "bins": bins,
        "all_dimensions_pass": all_pass,
        "dimensions": dimensions,
        "generator_view_pearson_correlation": correlation,
    }
    write_json(output_dir / "coverage_report.json", report)
    return report


def _pareto_rows(valid_view: pd.DataFrame) -> pd.DataFrame:
    ordered = valid_view.sort_values(
        ["thrust_N", "tsfc_kg_N_s"], ascending=[False, True]
    )
    keep = []
    best_tsfc = math.inf
    for index, row in ordered.iterrows():
        tsfc = float(row["tsfc_kg_N_s"])
        if tsfc < best_tsfc:
            keep.append(index)
            best_tsfc = tsfc
    return valid_view.loc[keep].sort_values("thrust_N").reset_index(drop=True)


def build_feasible_domain_report(
    valid_view: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Any]:
    thrust = valid_view["thrust_N"].to_numpy(dtype=float)
    tsfc = valid_view["tsfc_kg_N_s"].to_numpy(dtype=float)
    pareto = _pareto_rows(valid_view)
    pareto.to_csv(output_dir / "pareto_reference.csv", index=False)
    histogram, x_edges, y_edges = np.histogram2d(thrust, tsfc, bins=50)
    np.savez_compressed(
        output_dir / "condition_density.npz",
        counts=histogram,
        thrust_edges=x_edges,
        tsfc_edges=y_edges,
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(thrust, bins=50)
    axes[0].set_xlabel("thrust_N")
    axes[0].set_ylabel("count")
    axes[1].hist(tsfc, bins=50)
    axes[1].set_xlabel("tsfc_kg_N_s")
    axes[2].hexbin(thrust, tsfc, gridsize=50, bins="log", mincnt=1)
    axes[2].plot(pareto["thrust_N"], pareto["tsfc_kg_N_s"], color="red")
    axes[2].set_xlabel("thrust_N")
    axes[2].set_ylabel("tsfc_kg_N_s")
    fig.tight_layout()
    fig.savefig(output_dir / "feasible_domain.png", dpi=150)
    plt.close(fig)
    quantiles = [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]
    report = {
        "valid_count": len(valid_view),
        "thrust_N_quantiles": {
            str(q): float(np.quantile(thrust, q)) for q in quantiles
        },
        "tsfc_kg_N_s_quantiles": {
            str(q): float(np.quantile(tsfc, q)) for q in quantiles
        },
        "condition_correlation": float(np.corrcoef(thrust, tsfc)[0, 1]),
        "pareto_count": len(pareto),
        "density_grid_shape": list(histogram.shape),
    }
    write_json(output_dir / "feasible_domain_report.json", report)
    return report


def _target_id(
    suite_version: str,
    target_type: str,
    thrust: float,
    tsfc: float,
    source_ids: str,
) -> str:
    return _canonical_hash(
        {
            "suite_version": suite_version,
            "target_type": target_type,
            "thrust_N": float(thrust).hex(),
            "tsfc_kg_N_s": float(tsfc).hex(),
            "source_sample_ids": source_ids,
        }
    )


def _target_record(
    suite_version: str,
    target_type: str,
    expected_status: str,
    construction: str,
    thrust: float,
    tsfc: float,
    source_ids: str,
    source_splits: str,
    x_names: Sequence[str],
    design: Optional[Mapping[str, Any]] = None,
    known_forward_solution: bool = False,
    reason: str = "",
    local_condition_distance: float = math.nan,
    interpolation_alpha: float = math.nan,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "target_id": _target_id(
            suite_version, target_type, thrust, tsfc, source_ids
        ),
        "target_suite_version": suite_version,
        "target_type": target_type,
        "expected_status": expected_status,
        "construction": construction,
        "independent_condition_mix": False,
        "thrust_N": float(thrust),
        "tsfc_kg_N_s": float(tsfc),
        "known_forward_solution": bool(known_forward_solution),
        "source_sample_ids": source_ids,
        "source_splits": source_splits,
        "local_condition_distance": float(local_condition_distance),
        "interpolation_alpha": float(interpolation_alpha),
        "reason": reason,
    }
    for name in x_names:
        record[name] = float(design[name]) if design is not None else math.nan
    return record


def _condition_signatures(frame: pd.DataFrame) -> set[Tuple[str, str]]:
    return {
        (float(row.thrust_N).hex(), float(row.tsfc_kg_N_s).hex())
        for row in frame[["thrust_N", "tsfc_kg_N_s"]].itertuples(index=False)
    }


def build_target_suite(
    valid_view: pd.DataFrame,
    schema: Mapping[str, Any],
    normalizer: GeneratorNormalizer,
    target_config: Mapping[str, Any],
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    x_names = design_names(schema)
    suite_version = str(target_config["version"])
    rng = np.random.RandomState(int(target_config["seed"]))
    train = valid_view[valid_view["split"] == "train"].reset_index(drop=True)
    test = valid_view[valid_view["split"] == "test"].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Target construction requires non-empty train and test splits")
    train_ids = set(train["sample_id"])
    train_condition_signatures = _condition_signatures(train)
    records_id: List[Dict[str, Any]] = []
    records_interp: List[Dict[str, Any]] = []
    records_ood: List[Dict[str, Any]] = []

    id_count = min(int(target_config["id_count"]), len(test))
    id_indices = rng.choice(len(test), size=id_count, replace=False)
    id_source_ids = set()
    for index in id_indices:
        row = test.iloc[int(index)]
        source_id = str(row["sample_id"])
        id_source_ids.add(source_id)
        records_id.append(
            _target_record(
                suite_version,
                "ID",
                "known_feasible",
                "held_out_forward_sample",
                row["thrust_N"],
                row["tsfc_kg_N_s"],
                source_id,
                "test",
                x_names,
                design=row,
                known_forward_solution=True,
                reason="held-out sample with an exact Brayton 0D solution",
            )
        )

    normalized_test = normalizer.transform_conditions(test)
    test_tree = cKDTree(normalized_test)
    distances, neighbors = test_tree.query(normalized_test, k=min(2, len(test)))
    alpha_lower, alpha_upper = (
        float(value) for value in target_config["interpolation_alpha"]
    )
    desired_interp = int(target_config["interpolation_count"])
    base_order = rng.permutation(len(test))
    attempts = 0
    while len(records_interp) < desired_interp and attempts < max(100, desired_interp * 20):
        base_index = int(base_order[attempts % len(base_order)])
        if len(test) < 2:
            break
        neighbor_index = int(neighbors[base_index, 1])
        left = test.iloc[base_index]
        right = test.iloc[neighbor_index]
        alpha = float(rng.uniform(alpha_lower, alpha_upper))
        design = {
            name: (1.0 - alpha) * float(left[name]) + alpha * float(right[name])
            for name in x_names
        }
        result = solve_brayton_cycle(
            FlightConditions(), engine_inputs_from_cycle_row(design, schema)
        )
        source_ids = f"{left['sample_id']}|{right['sample_id']}"
        if result.is_valid and result.thrust_N > 0 and result.tsfc_kg_N_s > 0:
            candidate = _target_record(
                suite_version,
                "Interpolation",
                "known_feasible",
                "local_design_interpolation_then_brayton_0d",
                result.thrust_N,
                result.tsfc_kg_N_s,
                source_ids,
                "test|test",
                x_names,
                design=design,
                known_forward_solution=True,
                reason="5D interpolation between nearest held-out condition neighbors",
                local_condition_distance=float(distances[base_index, 1]),
                interpolation_alpha=alpha,
            )
            signature = (
                float(candidate["thrust_N"]).hex(),
                float(candidate["tsfc_kg_N_s"]).hex(),
            )
            if (
                candidate["target_id"] not in {r["target_id"] for r in records_interp}
                and signature not in train_condition_signatures
            ):
                records_interp.append(candidate)
        attempts += 1
    if len(records_interp) != desired_interp:
        raise RuntimeError(
            f"Built {len(records_interp)}/{desired_interp} interpolation targets"
        )

    normalized_train = normalizer.transform_conditions(train)
    train_tree = cKDTree(normalized_train)
    nearest_train_distance, _ = train_tree.query(normalized_test, k=1)
    train_min = normalized_train.min(axis=0)
    train_max = normalized_train.max(axis=0)
    span = np.maximum(train_max - train_min, 1e-12)
    center = 0.5 * (train_min + train_max)
    edge_score = np.max(np.abs((normalized_test - center) / span), axis=1)
    sparse_score = nearest_train_distance + edge_score
    candidate_order = np.argsort(-sparse_score)
    ood_count = min(int(target_config["ood_feasible_unknown_count"]), len(test))
    chosen = []
    for index in candidate_order:
        if str(test.iloc[int(index)]["sample_id"]) not in id_source_ids:
            chosen.append(int(index))
        if len(chosen) == ood_count:
            break
    if len(chosen) < ood_count:
        chosen.extend(
            int(index) for index in candidate_order if int(index) not in chosen
        )
        chosen = chosen[:ood_count]
    for index in chosen:
        row = test.iloc[index]
        records_ood.append(
            _target_record(
                suite_version,
                "OOD-feasible-unknown",
                "feasible_sparse_low_confidence",
                "held_out_sparse_boundary_sample",
                row["thrust_N"],
                row["tsfc_kg_N_s"],
                str(row["sample_id"]),
                "test",
                x_names,
                design=None,
                known_forward_solution=False,
                reason="held-out condition with large train-neighbor/boundary score",
                local_condition_distance=float(nearest_train_distance[index]),
            )
        )

    infeasible_count = int(target_config["infeasible_count"])
    margin_lower, margin_upper = (
        float(value) for value in target_config["infeasible_empirical_margin"]
    )
    max_thrust = float(valid_view["thrust_N"].max())
    min_tsfc = float(valid_view["tsfc_kg_N_s"].min())
    for index in range(infeasible_count):
        anchor = test.iloc[int(rng.randint(0, len(test)))]
        margin = float(rng.uniform(margin_lower, margin_upper))
        mode = index % 3
        if mode == 0:
            thrust = -margin * max_thrust
            tsfc = float(anchor["tsfc_kg_N_s"])
            reason = "non-positive thrust violates the target domain"
            construction = "domain_invalid_nonpositive_thrust"
        elif mode == 1:
            thrust = float(anchor["thrust_N"])
            tsfc = -margin * min_tsfc
            reason = "non-positive TSFC violates the target domain"
            construction = "domain_invalid_nonpositive_tsfc"
        else:
            thrust = max_thrust * (1.0 + margin)
            tsfc = min_tsfc * (1.0 - 0.5 * margin)
            reason = "joint target lies beyond the observed 5D feasible envelope"
            construction = "joint_empirical_frontier_extrapolation"
        records_ood.append(
            _target_record(
                suite_version,
                "Infeasible",
                "reject",
                construction,
                thrust,
                tsfc,
                str(anchor["sample_id"]),
                "test",
                x_names,
                design=None,
                known_forward_solution=False,
                reason=reason,
            )
        )

    id_frame = pd.DataFrame(records_id)
    interp_frame = pd.DataFrame(records_interp)
    ood_frame = pd.DataFrame(records_ood)
    suite = pd.concat([id_frame, interp_frame, ood_frame], ignore_index=True)
    if suite["target_id"].duplicated().any():
        raise ValueError("Target suite contains duplicate target_id values")
    if bool(suite["independent_condition_mix"].any()):
        raise ValueError("Independent thrust/TSFC mixing is forbidden")
    provenance_ids = {
        item
        for value in suite["source_sample_ids"].dropna().astype(str)
        for item in value.split("|")
        if item
    }
    leaked_source_ids = sorted(provenance_ids.intersection(train_ids))
    ordinary = suite[suite["target_type"].isin(["ID", "Interpolation"])]
    leaked_conditions = [
        row.target_id
        for row in ordinary.itertuples(index=False)
        if (float(row.thrust_N).hex(), float(row.tsfc_kg_N_s).hex())
        in train_condition_signatures
    ]
    if leaked_source_ids or leaked_conditions:
        raise ValueError("Target suite leaks train samples or exact train conditions")
    target_dir = output_dir / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    id_frame.to_csv(target_dir / "targets_id.csv", index=False)
    interp_frame.to_csv(target_dir / "targets_interp.csv", index=False)
    ood_frame.to_csv(target_dir / "targets_ood.csv", index=False)
    suite.to_csv(target_dir / "target_suite_v1.csv", index=False)
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(target_dir.glob("*.csv"))
    }
    manifest = {
        "target_suite_version": suite_version,
        "counts": {
            str(key): int(value)
            for key, value in suite["target_type"].value_counts().items()
        },
        "total_count": len(suite),
        "source_splits": sorted(suite["source_splits"].unique().tolist()),
        "train_source_id_overlap": len(leaked_source_ids),
        "ordinary_exact_train_condition_overlap": len(leaked_conditions),
        "independent_condition_mix_count": int(
            suite["independent_condition_mix"].sum()
        ),
        "ood_excluded_from_id_metrics": True,
        "hashes": hashes,
    }
    write_yaml(target_dir / "target_suite_manifest.yaml", manifest)
    return suite, manifest


def _prepare_run_directory(
    output_dir: Path,
    config_hash: str,
    schema_hash: str,
    config: Mapping[str, Any],
) -> None:
    completed = output_dir / "wp2_manifest.yaml"
    if completed.exists() and bool(config["output"].get("refuse_completed_run", True)):
        raise FileExistsError(f"Completed WP2 output will not be overwritten: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "resume_contract.json"
    contract = {
        "config_sha256": config_hash,
        "schema_sha256": schema_hash,
        "dataset_version": config["dataset_version"],
        "n_samples": int(config["sampling"]["n_samples"]),
    }
    if state_path.exists():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError("Existing WP2 partial output has a different resume contract")
    else:
        write_json(state_path, contract)


def run_wp2(
    config_path: Path,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    config_path = config_path.resolve()
    config = load_wp2_config(config_path)
    schema_path = (repo_root / config["schema"]).resolve()
    schema = load_schema(schema_path)
    validate_schema_against_design_bounds(schema)
    output_dir = (
        output_dir_override.resolve()
        if output_dir_override is not None
        else (repo_root / config["output"]["directory"]).resolve()
    )
    config_hash = sha256_file(config_path)
    schema_hash = sha256_file(schema_path)
    bounds_contract = _design_space_contract()
    bounds_hash = _canonical_hash(bounds_contract)
    _prepare_run_directory(output_dir, config_hash, schema_hash, config)
    started = dt.datetime.now(dt.timezone.utc)
    try:
        write_yaml(output_dir / "build_config.yaml", config)
        write_yaml(output_dir / "design_space_contract.yaml", bounds_contract)
        inputs = create_global_lhs_inputs(
            int(config["sampling"]["n_samples"]),
            int(config["sampling"]["seed"]),
            str(config["dataset_version"]),
        )
        shard_paths, reused_shards = generate_raw_shards(
            inputs,
            output_dir / "raw_shards",
            config,
            config_hash,
            schema_hash,
            bounds_hash,
            _git_commit(repo_root),
        )
        raw_path = output_dir / f"dataset_{len(inputs)}_raw.csv"
        merged, merge_report = merge_raw_shards(
            shard_paths,
            len(inputs),
            str(config["cleaning"]["duplicate_key"]),
            raw_path,
        )
        write_yaml(
            output_dir / "raw_dataset_manifest.yaml",
            {
                "dataset_version": config["dataset_version"],
                "sampling": config["sampling"],
                "config_sha256": config_hash,
                "schema_sha256": schema_hash,
                "design_bounds_sha256": bounds_hash,
                "shard_count": len(shard_paths),
                "reused_shard_count": reused_shards,
                "shards": [
                    {
                        "path": path.relative_to(output_dir).as_posix(),
                        "sha256": sha256_file(path),
                    }
                    for path in shard_paths
                ],
                "merge": merge_report,
            },
        )

        cleaned, cleaning_report = apply_cleaning_rules(
            merged, schema, config["cleaning"]
        )
        cleaned_path = output_dir / "dataset_cleaning_ledger.csv"
        _atomic_csv(cleaned, cleaned_path)
        write_yaml(output_dir / "data_quality_report.yaml", cleaning_report)

        source_for_view = cleaned.copy()
        source_for_view["is_valid"] = source_for_view["clean_is_valid"]
        all_view, valid_view, _ = build_generator_views(
            source_for_view,
            schema,
            str(config["dataset_version"]),
            raw_path.name,
        )
        source_failure = cleaned["cleaning_failure_code"].to_dict()
        all_view["failure_code"] = all_view.apply(
            lambda row: "OK"
            if bool(row["is_valid"])
            else source_failure[int(row["source_row"])],
            axis=1,
        )
        all_view, valid_view, split_manifest = assign_splits(
            all_view,
            all_view[all_view["is_valid"]].copy().reset_index(drop=True),
            seed=int(config["split"]["seed"]),
            train_fraction=float(config["split"]["train_fraction"]),
            val_fraction=float(config["split"]["val_fraction"]),
            test_fraction=float(config["split"]["test_fraction"]),
        )
        invalid_view = all_view[~all_view["is_valid"]].copy()
        all_path = output_dir / "generator_dataset_v1_all.csv"
        valid_path = output_dir / "generator_dataset_v1_valid.csv"
        invalid_path = output_dir / "invalid_samples.csv"
        split_path = output_dir / "generator_v1_split.csv"
        all_view.to_csv(all_path, index=False)
        valid_view.to_csv(valid_path, index=False)
        invalid_view.to_csv(invalid_path, index=False)
        split_manifest.to_csv(split_path, index=False)

        train = valid_view[valid_view["split"] == "train"]
        normalizer = GeneratorNormalizer.fit(train, schema)
        normalizer.save(output_dir / "normalizer_v1.json")
        parity_summary: Dict[str, Any] = {"enabled": False}
        if bool(config["parity"].get("enabled", True)):
            parity_report, parity_summary = run_parity_check(
                valid_view, schema, config["parity"]
            )
            parity_report.to_csv(output_dir / "parity_report.csv", index=False)
            parity_report[~parity_report["parity_pass"]].to_csv(
                output_dir / "parity_failures.csv", index=False
            )
            write_json(output_dir / "parity_summary.json", parity_summary)
            if int(parity_summary["failure_count"]) != 0:
                raise ValueError("WP2 Brayton parity check failed")

        coverage_report = build_coverage_report(
            merged,
            valid_view,
            int(config["coverage"]["bins"]),
            output_dir,
        )
        if not bool(coverage_report["all_dimensions_pass"]):
            raise ValueError("WP2 LHS coverage has unexpected empty bins")
        feasible_report = build_feasible_domain_report(valid_view, output_dir)
        suite, target_manifest = build_target_suite(
            valid_view,
            schema,
            normalizer,
            config["targets"],
            output_dir,
        )

        environment = collect_environment()
        write_json(output_dir / "environment.json", environment)
        git_state = collect_git_state(repo_root, output_dir)
        write_json(
            output_dir / "source_manifest.json", collect_source_manifest(repo_root)
        )
        output_files = [
            raw_path,
            cleaned_path,
            all_path,
            valid_path,
            invalid_path,
            split_path,
            output_dir / "normalizer_v1.json",
            output_dir / "coverage_report.json",
            output_dir / "feasible_domain_report.json",
            output_dir / "targets" / "target_suite_v1.csv",
        ]
        output_hashes = {path.name: sha256_file(path) for path in output_files}
        completed = dt.datetime.now(dt.timezone.utc)
        split_counts = {
            str(key): int(value)
            for key, value in valid_view["split"].value_counts().items()
        }
        manifest = {
            "status": "complete",
            "wp_version": config["version"],
            "dataset_version": config["dataset_version"],
            "schema_version": schema["schema_version"],
            "started_utc": started,
            "completed_utc": completed,
            "elapsed_seconds": (completed - started).total_seconds(),
            "counts": {
                "raw": len(merged),
                "valid": len(valid_view),
                "invalid": len(invalid_view),
                "splits": split_counts,
                "targets": len(suite),
            },
            "merge": merge_report,
            "cleaning": cleaning_report,
            "coverage_pass": coverage_report["all_dimensions_pass"],
            "feasible_domain": feasible_report,
            "parity": parity_summary,
            "target_suite": target_manifest,
            "output_hashes": output_hashes,
            "git": git_state,
            "environment_file": "environment.json",
            "source_manifest_file": "source_manifest.json",
        }
        write_yaml(output_dir / "wp2_manifest.yaml", manifest)
        return output_dir
    except Exception as exc:
        write_json(
            output_dir / "failure.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "timestamp_utc": dt.datetime.now(dt.timezone.utc),
                "resume_supported": True,
            },
        )
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run WP2 data expansion, cleaning, and target-suite construction."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config/generator_wp2.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    output_dir = run_wp2(
        args.config,
        repo_root=Path.cwd(),
        output_dir_override=args.output_dir,
    )
    print(f"WP2 data expansion complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
