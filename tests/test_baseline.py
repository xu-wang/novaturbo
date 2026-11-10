"""Tests for the frozen WP0 baseline orchestration."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.ai.baseline import (
    approximate_hypervolume,
    build_split_manifest,
    prepare_run_directory,
    run_wp0_baseline,
    set_global_seed,
)
from src.ai.surrogate import SurrogateConfig


def test_prepare_run_directory_refuses_overwrite(tmp_path):
    run_dir = prepare_run_directory(tmp_path, "baseline")
    assert run_dir.is_dir()
    with pytest.raises(FileExistsError):
        prepare_run_directory(tmp_path, "baseline")


def test_set_global_seed_reproduces_numpy_sequence():
    set_global_seed(17)
    first = np.random.random(8)
    set_global_seed(17)
    second = np.random.random(8)
    np.testing.assert_array_equal(first, second)


def test_build_split_manifest_is_deterministic(tmp_path):
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "is_valid": [True] * 20 + [False] * 4,
            "dummy": np.arange(24),
        }
    ).to_csv(csv_path, index=False)
    config = SurrogateConfig(val_split=0.15, test_split=0.10)
    first = build_split_manifest(
        csv_path, config, 42, tmp_path / "split_first.csv"
    )
    second = build_split_manifest(
        csv_path, config, 42, tmp_path / "split_second.csv"
    )
    pd.testing.assert_frame_equal(first, second)
    assert first["split"].value_counts().to_dict() == {
        "train": 15,
        "val": 3,
        "test": 2,
    }
    assert first["source_row"].max() == 19


def test_approximate_hypervolume_simple_two_dimensional_case():
    result = approximate_hypervolume(
        points=np.array([[0.5, 0.5]]),
        objective_min=np.array([0.0, 0.0]),
        objective_max=np.array([1.0, 1.0]),
        reference_point=np.array([1.0, 1.0]),
        sobol_power=12,
    )
    assert result["value"] == pytest.approx(0.25, abs=0.002)


def test_wp0_small_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    config = {
        "version": "test_baseline",
        "run_name": "test_run",
        "output_root": "unused",
        "dataset": {"n_samples": 40, "seed": 7},
        "surrogate": {
            "seed": 11,
            "split_seed": 42,
            "hidden_layers": [16, 16],
            "activation": "relu",
            "dropout": 0.0,
            "batch_norm": False,
            "batch_size": 16,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "epochs": 2,
            "patience": 2,
            "lr_scheduler": False,
            "val_split": 0.15,
            "test_split": 0.10,
        },
        "optimizer": {
            "seed": 13,
            "population_size": 8,
            "n_generations": 3,
            "crossover_rate": 0.9,
            "mutation_rate": 0.1,
            "mutation_scale": 0.1,
            "tournament_size": 3,
            "objectives": [
                "thrust_N",
                "tsfc_kg_N_s",
                "total_mass_kg",
                "thrust_to_weight",
            ],
            "maximize": [True, False, False, True],
            "min_thrust_N": 30.0,
            "max_tsfc": 0.00005,
            "max_mass_kg": 3.0,
            "use_surrogate": False,
            "surrogate_model_path": None,
        },
        "reproducibility": {
            "repeats": 1,
            "deterministic_torch": True,
            "relative_tolerance": 1e-6,
            "absolute_tolerance": 1e-8,
        },
        "metrics": {
            "hypervolume_sobol_power": 8,
            "normalized_reference_point": [1.1, 1.1, 1.1, 1.1],
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    run_dir = run_wp0_baseline(
        config_path,
        repo_root=repo_root,
        output_root_override=tmp_path / "artifacts",
    )

    assert (run_dir / "baseline_manifest.yaml").is_file()
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "data" / "dataset_40.csv").is_file()
    assert (run_dir / "splits" / "split_indices.csv").is_file()
    assert (
        run_dir / "replicates" / "replicate_01" / "surrogate" / "metrics.json"
    ).is_file()
    assert (
        run_dir / "replicates" / "replicate_01" / "nsga2" / "pareto.csv"
    ).is_file()

