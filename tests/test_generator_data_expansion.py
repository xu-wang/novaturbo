"""Tests for WP2 data expansion, cleaning, and frozen target construction."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from src.ai.generator_data_expansion import (
    apply_cleaning_rules,
    build_coverage_report,
    create_global_lhs_inputs,
    generate_raw_shards,
    load_wp2_config,
    merge_raw_shards,
    run_wp2,
)
from src.ai.generator_dataset import load_schema


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "config" / "generator_v1_schema.yaml"


def _small_config(output_dir: Path) -> dict:
    return {
        "version": "wp2_test_v1",
        "dataset_version": "generator_v1_expanded_test_v1",
        "schema": str(SCHEMA_PATH),
        "sampling": {
            "method": "global_latin_hypercube",
            "n_samples": 300,
            "shard_size": 100,
            "seed": 97,
            "bounds_version": "DesignSpaceBounds_v1",
        },
        "output": {
            "directory": str(output_dir),
            "resume": True,
            "refuse_completed_run": True,
        },
        "cleaning": {
            "require_source_valid": True,
            "require_finite_design": True,
            "require_finite_conditions": True,
            "thrust_min_exclusive": 0.0,
            "thrust_max_inclusive": None,
            "tsfc_min_exclusive": 0.0,
            "tsfc_max_inclusive": None,
            "duplicate_key": "raw_sample_id",
        },
        "split": {
            "seed": 98,
            "train_fraction": 0.8,
            "val_fraction": 0.1,
            "test_fraction": 0.1,
        },
        "coverage": {"bins": 10, "sparse_condition_quantile": 0.9},
        "targets": {
            "version": "target_suite_test_v1",
            "seed": 99,
            "id_count": 12,
            "interpolation_count": 12,
            "ood_feasible_unknown_count": 6,
            "infeasible_count": 6,
            "interpolation_alpha": [0.25, 0.75],
            "infeasible_empirical_margin": [0.05, 0.25],
        },
        "parity": {
            "enabled": True,
            "thrust_absolute_tolerance": 1e-10,
            "thrust_relative_tolerance": 1e-10,
            "tsfc_absolute_tolerance": 1e-12,
            "tsfc_relative_tolerance": 1e-10,
        },
    }


def test_global_lhs_is_reproducible_unique_and_stratified(tmp_path):
    first = create_global_lhs_inputs(200, 11, "test_v1")
    second = create_global_lhs_inputs(200, 11, "test_v1")
    pd.testing.assert_frame_equal(first, second)
    assert first["raw_sample_id"].is_unique
    assert first["global_sample_index"].tolist() == list(range(200))

    schema = load_schema(SCHEMA_PATH)
    valid = first.copy()
    valid["thrust_N"] = np.linspace(30.0, 80.0, len(valid))
    valid["tsfc_kg_N_s"] = np.linspace(2e-5, 5e-5, len(valid))
    report = build_coverage_report(first, valid, bins=10, output_dir=tmp_path)
    assert report["all_dimensions_pass"] is True
    assert all(
        not item["unexpected_empty_bins"]
        for item in report["dimensions"].values()
    )


def test_shards_resume_and_merge_without_duplicates(tmp_path):
    config = _small_config(tmp_path / "output")
    inputs = create_global_lhs_inputs(30, 7, config["dataset_version"])
    raw_dir = tmp_path / "raw"
    paths, reused = generate_raw_shards(
        inputs,
        raw_dir,
        config,
        "config-hash",
        "schema-hash",
        "bounds-hash",
        "commit",
    )
    assert reused == 0
    second_paths, second_reused = generate_raw_shards(
        inputs,
        raw_dir,
        config,
        "config-hash",
        "schema-hash",
        "bounds-hash",
        "commit",
    )
    assert paths == second_paths
    assert second_reused == len(paths)
    merged, report = merge_raw_shards(
        paths, 30, "raw_sample_id", tmp_path / "merged.csv"
    )
    assert len(merged) == 30
    assert merged["raw_sample_id"].is_unique
    assert report["duplicate_rows_removed"] == 0


def test_cleaning_rules_retain_invalid_rows_and_reasons():
    schema = load_schema(SCHEMA_PATH)
    raw = create_global_lhs_inputs(4, 3, "cleaning_v1")
    raw["is_valid"] = True
    raw["n_warnings"] = 0
    raw["thrust_N"] = [40.0, -1.0, 50.0, 60.0]
    raw["tsfc_kg_N_s"] = [3e-5, 3e-5, np.nan, 4e-5]
    raw.loc[3, "is_valid"] = False
    rules = _small_config(Path("unused"))["cleaning"]
    cleaned, report = apply_cleaning_rules(raw, schema, rules)
    assert len(cleaned) == 4
    assert report["valid_rows"] == 1
    assert report["invalid_rows_retained"] is True
    assert cleaned["cleaning_failure_code"].tolist() == [
        "OK",
        "THRUST_BELOW_MIN",
        "NONFINITE_CONDITION",
        "SOURCE_INVALID",
    ]


def test_wp2_small_end_to_end_freezes_leak_free_targets(tmp_path):
    output_dir = tmp_path / "wp2"
    config = _small_config(output_dir)
    config_path = tmp_path / "wp2.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    loaded = load_wp2_config(config_path)
    assert loaded["sampling"]["n_samples"] == 300

    result = run_wp2(
        config_path,
        repo_root=REPO_ROOT,
        output_dir_override=output_dir,
    )
    valid = pd.read_csv(result / "generator_dataset_v1_valid.csv")
    split = pd.read_csv(result / "generator_v1_split.csv")
    suite = pd.read_csv(result / "targets" / "target_suite_v1.csv")
    target_manifest = yaml.safe_load(
        (result / "targets" / "target_suite_manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert len(valid) == int((split["split"].isin(["train", "val", "test"])).sum())
    assert suite["target_id"].is_unique
    assert suite["target_type"].value_counts().to_dict() == {
        "ID": 12,
        "Interpolation": 12,
        "OOD-feasible-unknown": 6,
        "Infeasible": 6,
    }
    assert not suite["independent_condition_mix"].astype(bool).any()
    train_ids = set(split.loc[split["split"] == "train", "sample_id"])
    target_source_ids = {
        item
        for value in suite["source_sample_ids"].astype(str)
        for item in value.split("|")
    }
    assert train_ids.isdisjoint(target_source_ids)
    assert target_manifest["train_source_id_overlap"] == 0
    assert target_manifest["ordinary_exact_train_condition_overlap"] == 0
    assert (result / "coverage_report.json").is_file()
    assert (result / "feasible_domain_report.json").is_file()
    assert (result / "data_quality_report.yaml").is_file()
    assert (result / "wp2_manifest.yaml").is_file()

    with pytest.raises(FileExistsError):
        run_wp2(
            config_path,
            repo_root=REPO_ROOT,
            output_dir_override=output_dir,
        )


def test_wp2_cli_is_directly_executable():
    completed = subprocess.run(
        [sys.executable, "scripts/run_wp2_data_expansion.py", "--help"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr
    assert "data expansion" in completed.stdout
