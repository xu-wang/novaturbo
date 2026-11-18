"""
WP1 generator dataset construction for the NovaTurbo inverse problem.

The generated view contains only the five Brayton-cycle design variables,
the two performance conditions, stable identifiers, split assignments, and
explicit quality metadata.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from .baseline import (
    collect_environment,
    collect_git_state,
    collect_source_manifest,
    sha256_file,
    write_json,
    write_yaml,
)
from .dataset import DesignSpaceBounds
from ..physics.brayton import EngineInputs, FlightConditions, solve_brayton_cycle


OK = "OK"
SOURCE_INVALID = "SOURCE_INVALID"
NONFINITE_DESIGN = "NONFINITE_DESIGN"
NONFINITE_THRUST = "NONFINITE_THRUST"
NONPOSITIVE_THRUST = "NONPOSITIVE_THRUST"
NONFINITE_TSFC = "NONFINITE_TSFC"
NONPOSITIVE_TSFC = "NONPOSITIVE_TSFC"


def load_schema(path: Path) -> Dict[str, Any]:
    schema = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "design_variables",
        "conditions",
        "quality_fields",
        "identifier_fields",
        "normalization",
    }
    missing = required.difference(schema)
    if missing:
        raise ValueError(f"Generator schema is missing keys: {sorted(missing)}")
    if not schema["design_variables"]:
        raise ValueError("Generator schema must define at least one design variable")
    if not schema["conditions"]:
        raise ValueError("Generator schema must define at least one condition")
    return schema


def design_names(schema: Mapping[str, Any]) -> List[str]:
    return [field["name"] for field in schema["design_variables"]]


def condition_names(schema: Mapping[str, Any]) -> List[str]:
    return [field["name"] for field in schema["conditions"]]


def validate_schema_against_design_bounds(schema: Mapping[str, Any]) -> None:
    bounds = DesignSpaceBounds()
    available = bounds.__dataclass_fields__
    for field in schema["design_variables"]:
        name = field["name"]
        if name not in available:
            raise ValueError(f"Schema design variable is absent from DesignSpaceBounds: {name}")
        expected = tuple(float(value) for value in getattr(bounds, name))
        actual = tuple(float(value) for value in field["bounds"])
        if actual != expected:
            raise ValueError(
                f"Schema bounds for {name} are {actual}, expected {expected}"
            )


def _canonical_float(value: Any) -> str:
    return float(value).hex()


def stable_sample_id(
    dataset_version: str,
    source_file: str,
    source_row: int,
    design_values: Mapping[str, Any],
    ordered_design_names: Sequence[str],
) -> str:
    payload = {
        "dataset_version": dataset_version,
        "source_file": source_file,
        "source_row": int(source_row),
        "design": [
            [name, _canonical_float(design_values[name])]
            for name in ordered_design_names
        ],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _classify_row(
    row: pd.Series,
    x_names: Sequence[str],
    c_names: Sequence[str],
) -> Tuple[bool, str]:
    source_valid = bool(row.get("is_valid", False))
    if not source_valid:
        return False, SOURCE_INVALID
    design = row[list(x_names)].to_numpy(dtype=float)
    if not np.all(np.isfinite(design)):
        return False, NONFINITE_DESIGN
    thrust = float(row[c_names[0]])
    tsfc = float(row[c_names[1]])
    if not math.isfinite(thrust):
        return False, NONFINITE_THRUST
    if thrust <= 0:
        return False, NONPOSITIVE_THRUST
    if not math.isfinite(tsfc):
        return False, NONFINITE_TSFC
    if tsfc <= 0:
        return False, NONPOSITIVE_TSFC
    return True, OK


def build_generator_views(
    source: pd.DataFrame,
    schema: Mapping[str, Any],
    dataset_version: str,
    source_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_names = design_names(schema)
    c_names = condition_names(schema)
    required_columns = set(x_names + c_names + ["is_valid", "n_warnings"])
    missing = required_columns.difference(source.columns)
    if missing:
        raise ValueError(f"Source dataset is missing columns: {sorted(missing)}")

    source_with_rows = source.reset_index(drop=False).rename(
        columns={"index": "source_row"}
    )
    records = []
    invalid_records = []
    for _, row in source_with_rows.iterrows():
        is_valid, failure_code = _classify_row(row, x_names, c_names)
        sample_id = stable_sample_id(
            dataset_version,
            source_id,
            int(row["source_row"]),
            row,
            x_names,
        )
        record: Dict[str, Any] = {
            "sample_id": sample_id,
            "dataset_version": dataset_version,
            "source_file": source_id,
            "source_row": int(row["source_row"]),
            "split": "unassigned" if is_valid else "invalid",
            "source_is_valid": bool(row["is_valid"]),
            "is_valid": is_valid,
            "n_warnings": int(row["n_warnings"]),
            "failure_code": failure_code,
        }
        for name in x_names + c_names:
            record[name] = float(row[name])
        records.append(record)
        if not is_valid:
            invalid_records.append(record.copy())

    ordered_columns = (
        list(schema["identifier_fields"])
        + list(schema["quality_fields"])
        + x_names
        + c_names
    )
    all_view = pd.DataFrame(records)[ordered_columns]
    if all_view["sample_id"].duplicated().any():
        duplicates = all_view.loc[
            all_view["sample_id"].duplicated(keep=False), "sample_id"
        ].tolist()
        raise ValueError(f"Duplicate sample_id values detected: {duplicates[:5]}")
    valid_view = all_view[all_view["is_valid"]].copy().reset_index(drop=True)
    invalid_view = pd.DataFrame(invalid_records)
    if invalid_view.empty:
        invalid_view = pd.DataFrame(columns=ordered_columns)
    else:
        invalid_view = invalid_view[ordered_columns].reset_index(drop=True)
    return all_view, valid_view, invalid_view


def assign_splits(
    all_view: pd.DataFrame,
    valid_view: pd.DataFrame,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fractions = train_fraction + val_fraction + test_fraction
    if not math.isclose(fractions, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Split fractions must sum to one, got {fractions}")
    n_valid = len(valid_view)
    n_val = int(n_valid * val_fraction)
    n_test = int(n_valid * test_fraction)
    n_train = n_valid - n_val - n_test

    rng = np.random.RandomState(seed)
    permutation = rng.permutation(n_valid)
    assignments: Dict[str, str] = {}
    ranges = [
        ("train", permutation[:n_train]),
        ("val", permutation[n_train : n_train + n_val]),
        ("test", permutation[n_train + n_val :]),
    ]
    for split_name, indices in ranges:
        for index in indices:
            assignments[valid_view.iloc[int(index)]["sample_id"]] = split_name

    all_result = all_view.copy()
    valid_result = valid_view.copy()
    all_result["split"] = all_result.apply(
        lambda row: assignments[row["sample_id"]]
        if row["is_valid"]
        else "invalid",
        axis=1,
    )
    valid_result["split"] = valid_result["sample_id"].map(assignments)
    if valid_result["split"].isna().any():
        raise RuntimeError("Some valid samples did not receive a split")

    split_manifest = valid_result[
        ["sample_id", "dataset_version", "source_file", "source_row", "split"]
    ].copy()
    split_manifest["split_seed"] = int(seed)
    return all_result, valid_result, split_manifest


def _sample_id_set_hash(sample_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in sample_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_matrix(
    values: Any,
    names: Sequence[str],
) -> Tuple[np.ndarray, bool]:
    single = False
    if isinstance(values, pd.DataFrame):
        matrix = values[list(names)].to_numpy(dtype=float)
    elif isinstance(values, pd.Series):
        matrix = values[list(names)].to_numpy(dtype=float)[None, :]
        single = True
    elif isinstance(values, Mapping):
        matrix = np.asarray([[values[name] for name in names]], dtype=float)
        single = True
    else:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
            single = True
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError(
            f"Expected shape [n,{len(names)}] for fields {list(names)}, got {matrix.shape}"
        )
    return matrix, single


@dataclass
class GeneratorNormalizer:
    schema_version: str
    design: List[Dict[str, Any]]
    conditions: List[Dict[str, Any]]
    fitted_split: str
    fitted_sample_count: int
    fitted_sample_id_sha256: str

    @classmethod
    def fit(
        cls,
        train_frame: pd.DataFrame,
        schema: Mapping[str, Any],
    ) -> "GeneratorNormalizer":
        if train_frame.empty:
            raise ValueError("Cannot fit a normalizer on an empty train split")
        design_specs = []
        for field in schema["design_variables"]:
            lower, upper = (float(value) for value in field["bounds"])
            if not upper > lower:
                raise ValueError(f"Invalid design bounds for {field['name']}")
            design_specs.append(
                {
                    "name": field["name"],
                    "method": "physical_bounds_to_minus_one_plus_one",
                    "physical_min": lower,
                    "physical_max": upper,
                    "normalized_min": -1.0,
                    "normalized_max": 1.0,
                }
            )

        condition_specs = []
        for field in schema["conditions"]:
            values = train_frame[field["name"]].to_numpy(dtype=float)
            transform = field.get("transform", "identity")
            if transform == "natural_log":
                if np.any(values <= 0):
                    raise ValueError(
                        f"{field['name']} contains non-positive values for log transform"
                    )
                transformed = np.log(values)
            elif transform == "identity":
                transformed = values
            else:
                raise ValueError(f"Unsupported transform: {transform}")
            mean = float(np.mean(transformed))
            std = float(np.std(transformed, ddof=0))
            if not math.isfinite(std) or std <= 0:
                raise ValueError(f"Zero or invalid standard deviation for {field['name']}")
            condition_specs.append(
                {
                    "name": field["name"],
                    "method": "standard",
                    "transform": transform,
                    "mean": mean,
                    "std": std,
                }
            )
        return cls(
            schema_version=str(schema["schema_version"]),
            design=design_specs,
            conditions=condition_specs,
            fitted_split="train",
            fitted_sample_count=len(train_frame),
            fitted_sample_id_sha256=_sample_id_set_hash(
                train_frame["sample_id"].tolist()
            ),
        )

    @property
    def design_names(self) -> List[str]:
        return [field["name"] for field in self.design]

    @property
    def condition_names(self) -> List[str]:
        return [field["name"] for field in self.conditions]

    def transform_design(self, values: Any) -> np.ndarray:
        matrix, single = _as_matrix(values, self.design_names)
        lower = np.asarray([field["physical_min"] for field in self.design])
        upper = np.asarray([field["physical_max"] for field in self.design])
        transformed = 2.0 * (matrix - lower) / (upper - lower) - 1.0
        return transformed[0] if single else transformed

    def inverse_design(self, values: Any) -> np.ndarray:
        matrix, single = _as_matrix(values, self.design_names)
        lower = np.asarray([field["physical_min"] for field in self.design])
        upper = np.asarray([field["physical_max"] for field in self.design])
        restored = lower + (matrix + 1.0) * 0.5 * (upper - lower)
        return restored[0] if single else restored

    def transform_conditions(self, values: Any) -> np.ndarray:
        matrix, single = _as_matrix(values, self.condition_names)
        transformed = matrix.copy()
        for index, field in enumerate(self.conditions):
            if field["transform"] == "natural_log":
                if np.any(transformed[:, index] <= 0):
                    raise ValueError(
                        f"{field['name']} must be positive before log transform"
                    )
                transformed[:, index] = np.log(transformed[:, index])
            transformed[:, index] = (
                transformed[:, index] - field["mean"]
            ) / field["std"]
        return transformed[0] if single else transformed

    def inverse_conditions(self, values: Any) -> np.ndarray:
        matrix, single = _as_matrix(values, self.condition_names)
        restored = matrix.copy()
        for index, field in enumerate(self.conditions):
            restored[:, index] = (
                restored[:, index] * field["std"] + field["mean"]
            )
            if field["transform"] == "natural_log":
                restored[:, index] = np.exp(restored[:, index])
        return restored[0] if single else restored

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "design": self.design,
            "conditions": self.conditions,
            "fitted_split": self.fitted_split,
            "fitted_sample_count": self.fitted_sample_count,
            "fitted_sample_id_sha256": self.fitted_sample_id_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeneratorNormalizer":
        return cls(
            schema_version=payload["schema_version"],
            design=list(payload["design"]),
            conditions=list(payload["conditions"]),
            fitted_split=payload["fitted_split"],
            fitted_sample_count=int(payload["fitted_sample_count"]),
            fitted_sample_id_sha256=payload["fitted_sample_id_sha256"],
        )

    def save(self, path: Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "GeneratorNormalizer":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def engine_inputs_from_cycle_row(
    row: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> EngineInputs:
    kwargs = {
        field["engine_input"]: float(row[field["name"]])
        for field in schema["design_variables"]
        if field.get("engine_input")
    }
    return EngineInputs(**kwargs)


def run_parity_check(
    valid_view: pd.DataFrame,
    schema: Mapping[str, Any],
    parity_config: Mapping[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    for _, row in valid_view.iterrows():
        record: Dict[str, Any] = {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "expected_thrust_N": float(row["thrust_N"]),
            "expected_tsfc_kg_N_s": float(row["tsfc_kg_N_s"]),
        }
        try:
            result = solve_brayton_cycle(
                FlightConditions(),
                engine_inputs_from_cycle_row(row, schema),
            )
            actual_thrust = float(result.thrust_N)
            actual_tsfc = float(result.tsfc_kg_N_s)
            thrust_abs = abs(actual_thrust - record["expected_thrust_N"])
            tsfc_abs = abs(actual_tsfc - record["expected_tsfc_kg_N_s"])
            thrust_rel = thrust_abs / max(abs(record["expected_thrust_N"]), 1e-30)
            tsfc_rel = tsfc_abs / max(
                abs(record["expected_tsfc_kg_N_s"]), 1e-30
            )
            passed = (
                math.isclose(
                    actual_thrust,
                    record["expected_thrust_N"],
                    rel_tol=float(parity_config["thrust_relative_tolerance"]),
                    abs_tol=float(parity_config["thrust_absolute_tolerance"]),
                )
                and math.isclose(
                    actual_tsfc,
                    record["expected_tsfc_kg_N_s"],
                    rel_tol=float(parity_config["tsfc_relative_tolerance"]),
                    abs_tol=float(parity_config["tsfc_absolute_tolerance"]),
                )
            )
            record.update(
                {
                    "actual_thrust_N": actual_thrust,
                    "actual_tsfc_kg_N_s": actual_tsfc,
                    "thrust_absolute_error": thrust_abs,
                    "thrust_relative_error": thrust_rel,
                    "tsfc_absolute_error": tsfc_abs,
                    "tsfc_relative_error": tsfc_rel,
                    "roundtrip_is_valid": bool(result.is_valid),
                    "roundtrip_n_warnings": len(result.warnings),
                    "roundtrip_warnings": " | ".join(result.warnings),
                    "parity_pass": passed,
                    "failure_code": OK if passed else "PARITY_MISMATCH",
                }
            )
        except Exception as exc:
            record.update(
                {
                    "actual_thrust_N": np.nan,
                    "actual_tsfc_kg_N_s": np.nan,
                    "thrust_absolute_error": np.nan,
                    "thrust_relative_error": np.nan,
                    "tsfc_absolute_error": np.nan,
                    "tsfc_relative_error": np.nan,
                    "roundtrip_is_valid": False,
                    "roundtrip_n_warnings": -1,
                    "roundtrip_warnings": "",
                    "parity_pass": False,
                    "failure_code": f"PARITY_EXCEPTION:{type(exc).__name__}",
                }
            )
        rows.append(record)

    report = pd.DataFrame(rows)
    split_summary = {}
    for split_name, group in report.groupby("split"):
        split_summary[split_name] = {
            "count": len(group),
            "pass_count": int(group["parity_pass"].sum()),
            "failure_count": int((~group["parity_pass"]).sum()),
            "max_thrust_absolute_error": float(
                group["thrust_absolute_error"].max()
            ),
            "max_tsfc_absolute_error": float(
                group["tsfc_absolute_error"].max()
            ),
        }
    summary = {
        "count": len(report),
        "pass_count": int(report["parity_pass"].sum()),
        "failure_count": int((~report["parity_pass"]).sum()),
        "pass_rate": float(report["parity_pass"].mean()),
        "by_split": split_summary,
        "tolerances": dict(parity_config),
    }
    return report, summary


def _build_dataset_card(
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
    source_path: Path,
    source_sha256: str,
    all_view: pd.DataFrame,
    valid_view: pd.DataFrame,
    invalid_view: pd.DataFrame,
    split_counts: Mapping[str, int],
    normalizer: GeneratorNormalizer,
    parity_summary: Mapping[str, Any],
    output_hashes: Mapping[str, str],
) -> str:
    tsfc = valid_view["tsfc_kg_N_s"]
    raw_skew = float(tsfc.skew())
    log_skew = float(np.log(tsfc).skew())
    lines = [
        "# NovaTurbo Generator Dataset V1 Card",
        "",
        f"- Dataset version: {config['dataset_version']}",
        f"- Schema version: {schema['schema_version']}",
        f"- Source: {config['source']['source_id']}",
        f"- Source path at build time: {source_path}",
        f"- Source SHA-256: {source_sha256}",
        f"- Total rows: {len(all_view)}",
        f"- Valid rows: {len(valid_view)}",
        f"- Invalid rows: {len(invalid_view)}",
        f"- Split counts: {dict(split_counts)}",
        f"- Parity pass: {parity_summary['pass_count']}/{parity_summary['count']}",
        "",
        "## Inverse problem",
        "",
        "Conditions [thrust_N, tsfc_kg_N_s] map to five cycle variables:",
        "",
    ]
    for field in schema["design_variables"]:
        lines.append(
            f"- {field['name']} [{field['unit']}], bounds={field['bounds']}, "
            f"EngineInputs.{field['engine_input']}"
        )
    lines.extend(
        [
            "",
            "## Normalization",
            "",
            "- Design variables: fixed physical bounds mapped to [-1, 1].",
            "- thrust_N: train-split mean/std standardization.",
            "- tsfc_kg_N_s: natural log followed by train-split mean/std standardization.",
            f"- TSFC raw skewness: {raw_skew:.6g}",
            f"- TSFC log skewness: {log_skew:.6g}",
            f"- Normalizer fitted rows: {normalizer.fitted_sample_count}",
            f"- Train sample-id hash: {normalizer.fitted_sample_id_sha256}",
            "",
            "## Validity",
            "",
            "A generator-valid row must be source-valid, have finite 5D variables, "
            "finite positive thrust, and finite positive TSFC. Invalid rows remain "
            "in generator_dataset_v1_all.csv and invalid_samples.csv.",
            "",
            "## Leakage controls",
            "",
            "- sample_id is content-derived from dataset version, stable source id, "
            "source row, and hexadecimal 5D values.",
            "- train/val/test are disjoint by sample_id.",
            "- condition statistics are fitted from train rows only.",
            "",
            "## Output hashes",
            "",
        ]
    )
    lines.extend(f"- {name}: {value}" for name, value in output_hashes.items())
    lines.append("")
    return "\n".join(lines)


def load_wp1_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"dataset_version", "schema", "source", "output", "split", "parity"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"WP1 config is missing keys: {sorted(missing)}")
    return config


def _prepare_outputs(output_dir: Path, split_path: Path, refuse_overwrite: bool) -> None:
    if refuse_overwrite and output_dir.exists():
        raise FileExistsError(
            f"Generator dataset output exists and will not be overwritten: {output_dir}"
        )
    if refuse_overwrite and split_path.exists():
        raise FileExistsError(
            f"Generator split output exists and will not be overwritten: {split_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=not refuse_overwrite)
    split_path.parent.mkdir(parents=True, exist_ok=True)


def run_wp1(
    config_path: Path,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    config_path = config_path.resolve()
    config = load_wp1_config(config_path)
    schema_path = (repo_root / config["schema"]).resolve()
    source_path = (repo_root / config["source"]["csv"]).resolve()
    output_dir = (
        output_dir_override.resolve()
        if output_dir_override is not None
        else (repo_root / config["output"]["directory"]).resolve()
    )
    split_path = (
        output_dir / "generator_v1_split.csv"
        if output_dir_override is not None
        else (repo_root / config["output"]["split_csv"]).resolve()
    )
    _prepare_outputs(
        output_dir,
        split_path,
        bool(config["output"].get("refuse_overwrite", True)),
    )
    started = dt.datetime.now(dt.timezone.utc)
    try:
        schema = load_schema(schema_path)
        validate_schema_against_design_bounds(schema)
        if not source_path.is_file():
            raise FileNotFoundError(f"WP1 source dataset not found: {source_path}")
        source_sha256 = sha256_file(source_path)
        source = pd.read_csv(source_path)

        all_view, valid_view, invalid_view = build_generator_views(
            source,
            schema,
            config["dataset_version"],
            config["source"]["source_id"],
        )
        all_view, valid_view, split_manifest = assign_splits(
            all_view,
            valid_view,
            seed=int(config["split"]["seed"]),
            train_fraction=float(config["split"]["train_fraction"]),
            val_fraction=float(config["split"]["val_fraction"]),
            test_fraction=float(config["split"]["test_fraction"]),
        )

        all_path = output_dir / "generator_dataset_v1_all.csv"
        valid_path = output_dir / "generator_dataset_v1_valid.csv"
        invalid_path = output_dir / "invalid_samples.csv"
        all_view.to_csv(all_path, index=False)
        valid_view.to_csv(valid_path, index=False)
        invalid_view.to_csv(invalid_path, index=False)
        split_manifest.to_csv(split_path, index=False)
        shutil.copy2(schema_path, output_dir / "generator_v1_schema.yaml")
        shutil.copy2(config_path, output_dir / "build_config.yaml")

        train_frame = valid_view[valid_view["split"] == "train"]
        normalizer = GeneratorNormalizer.fit(train_frame, schema)
        normalizer_path = output_dir / "normalizer_v1.json"
        normalizer.save(normalizer_path)

        parity_report, parity_summary = run_parity_check(
            valid_view, schema, config["parity"]
        )
        parity_path = output_dir / "parity_report.csv"
        parity_failures_path = output_dir / "parity_failures.csv"
        parity_report.to_csv(parity_path, index=False)
        parity_report[~parity_report["parity_pass"]].to_csv(
            parity_failures_path, index=False
        )
        write_json(output_dir / "parity_summary.json", parity_summary)

        split_counts = {
            name: int(value)
            for name, value in valid_view["split"].value_counts().items()
        }
        failure_counts = {
            name: int(value)
            for name, value in all_view["failure_code"].value_counts().items()
        }
        output_hashes = {
            "generator_dataset_v1_all.csv": sha256_file(all_path),
            "generator_dataset_v1_valid.csv": sha256_file(valid_path),
            "invalid_samples.csv": sha256_file(invalid_path),
            "generator_v1_split.csv": sha256_file(split_path),
            "normalizer_v1.json": sha256_file(normalizer_path),
            "parity_report.csv": sha256_file(parity_path),
        }
        card = _build_dataset_card(
            config,
            schema,
            source_path,
            source_sha256,
            all_view,
            valid_view,
            invalid_view,
            split_counts,
            normalizer,
            parity_summary,
            output_hashes,
        )
        (output_dir / "dataset_card.md").write_text(card, encoding="utf-8")

        environment = collect_environment()
        write_json(output_dir / "environment.json", environment)
        git_state = collect_git_state(repo_root, output_dir)
        source_manifest = collect_source_manifest(repo_root)
        write_json(output_dir / "source_manifest.json", source_manifest)

        completed = dt.datetime.now(dt.timezone.utc)
        manifest = {
            "status": "complete",
            "dataset_version": config["dataset_version"],
            "schema_version": schema["schema_version"],
            "started_utc": started,
            "completed_utc": completed,
            "elapsed_seconds": (completed - started).total_seconds(),
            "source": {
                "path": str(source_path),
                "source_id": config["source"]["source_id"],
                "sha256": source_sha256,
                "row_count": len(source),
            },
            "outputs": {
                "directory": str(output_dir),
                "split_csv": str(split_path),
                "hashes": output_hashes,
            },
            "counts": {
                "all": len(all_view),
                "valid": len(valid_view),
                "invalid": len(invalid_view),
                "splits": split_counts,
                "failure_codes": failure_counts,
            },
            "normalizer": normalizer.to_dict(),
            "parity": parity_summary,
            "git": git_state,
            "environment_file": "environment.json",
            "source_manifest_file": "source_manifest.json",
        }
        write_yaml(output_dir / "wp1_manifest.yaml", manifest)
        return output_dir
    except Exception as exc:
        write_json(
            output_dir / "failure.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "timestamp_utc": dt.datetime.now(dt.timezone.utc),
            },
        )
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the NovaTurbo generator_v1 data view."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/generator_wp1.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory; split CSV is then written inside it.",
    )
    args = parser.parse_args(argv)
    output_dir = run_wp1(
        args.config,
        repo_root=Path.cwd(),
        output_dir_override=args.output_dir,
    )
    print(f"WP1 generator dataset complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
