"""Build the active-10D generator view from the frozen WP2 raw dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from .baseline import sha256_file, write_json, write_yaml
from .dataset import evaluate_design
from .generator_dataset import GeneratorNormalizer, condition_names, design_names, load_schema


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _config_hash(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_active10d_view(
    config_path: Path,
    *,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    schema_path = _resolve(repo_root, str(config["schema"]))
    raw_path = _resolve(repo_root, str(config["source"]["raw_csv"]))
    base_path = _resolve(repo_root, str(config["source"]["base_valid_csv"]))
    target_path = _resolve(repo_root, str(config["source"]["base_target_suite"]))
    output_dir = (
        Path(output_dir_override).resolve()
        if output_dir_override is not None
        else _resolve(repo_root, str(config["output"]["directory"]))
    )
    if output_dir.exists() and bool(config["output"].get("refuse_overwrite", True)):
        raise FileExistsError(f"Active-10D output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "targets").mkdir()

    schema = load_schema(schema_path)
    x_names = design_names(schema)
    c_names = condition_names(schema)
    raw = pd.read_csv(raw_path)
    base = pd.read_csv(base_path)
    source_rows = base["source_row"].to_numpy(dtype=int)
    if np.any(source_rows < 0) or np.any(source_rows >= len(raw)):
        raise ValueError("Base source_row is outside the frozen raw dataset")
    aligned = raw.iloc[source_rows].reset_index(drop=True)
    if not np.array_equal(
        aligned["global_sample_index"].to_numpy(dtype=int), source_rows
    ):
        raise ValueError("Raw global_sample_index no longer matches source_row")

    metadata_names = [
        "sample_id",
        "source_file",
        "source_row",
        "split",
        "source_is_valid",
        "is_valid",
        "n_warnings",
        "failure_code",
    ]
    view = base[metadata_names].copy().reset_index(drop=True)
    view.insert(1, "dataset_version", str(config["dataset_version"]))
    for name in x_names + c_names:
        view[name] = aligned[name].to_numpy(dtype=float)
    ordered = list(schema["identifier_fields"]) + list(schema["quality_fields"])
    ordered += x_names + c_names
    view = view[ordered]
    if not view["is_valid"].astype(bool).all():
        raise ValueError("Active-10D training view must contain only valid rows")
    if not np.isfinite(view[x_names + c_names].to_numpy(dtype=float)).all():
        raise ValueError("Active-10D view contains non-finite values")
    for field in schema["design_variables"]:
        values = view[field["name"]].to_numpy(dtype=float)
        lower, upper = (float(value) for value in field["bounds"])
        if np.any(values < lower) or np.any(values > upper):
            raise ValueError(f"{field['name']} is outside its frozen bounds")

    dataset_path = output_dir / "generator_active10d_valid.csv"
    view.to_csv(dataset_path, index=False)
    train = view.loc[view["split"] == "train"]
    normalizer = GeneratorNormalizer.fit(train, schema)
    normalizer_path = output_dir / "normalizer_active10d.json"
    normalizer.save(normalizer_path)

    source_lookup = {
        str(sample_id): raw.iloc[int(source_row)]
        for sample_id, source_row in zip(base["sample_id"], source_rows)
    }
    targets = pd.read_csv(target_path)
    median_mass = float(train["total_mass_kg"].median())
    masses = []
    strategies = []
    for target in targets.to_dict(orient="records"):
        target_type = str(target["target_type"])
        source_ids = str(target.get("source_sample_ids", "")).split("|")
        if target_type == "ID" and source_ids[0] in source_lookup:
            mass = float(source_lookup[source_ids[0]]["total_mass_kg"])
            strategy = "held_out_source_mass"
        elif (
            target_type == "Interpolation"
            and len(source_ids) == 2
            and all(value in source_lookup for value in source_ids)
        ):
            alpha = float(target["interpolation_alpha"])
            design = {
                name: (1.0 - alpha) * float(source_lookup[source_ids[0]][name])
                + alpha * float(source_lookup[source_ids[1]][name])
                for name in x_names
            }
            performance = evaluate_design(pd.Series(design))
            if not bool(performance["is_valid"]):
                raise ValueError("Interpolation target produced invalid active-10D mass")
            mass = float(performance["total_mass_kg"])
            strategy = "interpolated_active10d_forward_mass"
        else:
            mass = median_mass
            strategy = "train_median_mass_for_rejection_target"
        masses.append(mass)
        strategies.append(strategy)
    targets["total_mass_kg"] = masses
    targets["target_thrust_to_weight"] = (
        targets["thrust_N"] / (targets["total_mass_kg"] * 9.81)
    )
    targets["mass_target_strategy"] = strategies
    active_target_path = output_dir / "targets" / "target_suite_active10d_v1.csv"
    targets.to_csv(active_target_path, index=False)

    split_counts = {
        str(key): int(value) for key, value in view["split"].value_counts().items()
    }
    manifest: Dict[str, Any] = {
        "status": "complete",
        "version": str(config["version"]),
        "dataset_version": str(config["dataset_version"]),
        "schema_version": str(schema["schema_version"]),
        "design_names": x_names,
        "condition_names": c_names,
        "row_count": int(len(view)),
        "split_counts": split_counts,
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": _config_hash(config),
        "source_hashes": {
            "raw_csv": sha256_file(raw_path),
            "base_valid_csv": sha256_file(base_path),
            "base_target_suite": sha256_file(target_path),
            "schema": sha256_file(schema_path),
        },
        "output_hashes": {
            "dataset": sha256_file(dataset_path),
            "normalizer": sha256_file(normalizer_path),
            "target_suite": sha256_file(active_target_path),
        },
    }
    write_yaml(output_dir / "manifest.yaml", manifest)
    write_json(
        output_dir / "dimension_audit.json",
        {
            "active_design_count": len(x_names),
            "condition_count": len(c_names),
            "inactive_19d_fields": [
                "compressor_blade_count",
                "combustor_inner_diameter_mm",
                "combustor_liner_thickness_mm",
                "combustor_num_injectors",
                "combustor_air_fuel_ratio",
                "turbine_blade_count",
                "turbine_hub_tip_ratio",
                "rpm",
                "lattice_cell_size_mm",
            ],
        },
    )
    return output_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build active-10D generator view")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/generator_active10d_view.yaml"),
    )
    args = parser.parse_args(argv)
    output = build_active10d_view(args.config)
    print(json.dumps({"output": str(output), "status": "complete"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
