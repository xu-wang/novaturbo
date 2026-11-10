from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.ai.stage2_wp5 import load_stage2_wp5_config
from src.ai.stage2_wp5_matrix import (
    _candidate_protocol_is_valid,
    _read_csv_allow_empty,
    _run_config,
    build_wp5_run_matrix,
    load_stage2_wp5_matrix_config,
)


def test_candidate_protocol_gate_allows_zero_only_when_n_valid_is_zero():
    assert _candidate_protocol_is_valid(
        pd.DataFrame({"N_valid": [0, 3, 7, 76], "N_keep": [0, 2, 4, 64]})
    )
    assert not _candidate_protocol_is_valid(
        pd.DataFrame({"N_valid": [0, 3], "N_keep": [1, 2]})
    )


def test_empty_failure_ledger_csv_is_read_as_empty_frame(tmp_path):
    path = tmp_path / "forward_results.csv"
    path.write_text("", encoding="utf-8")

    assert _read_csv_allow_empty(path).empty


def test_repository_formal_matrix_is_frozen_and_complete():
    repo_root = Path(__file__).resolve().parents[1]
    config = load_stage2_wp5_matrix_config(
        repo_root / "config" / "stage2" / "s2_wp5_matrix.yaml"
    )
    matrix = build_wp5_run_matrix(config, repo_root)

    assert len(matrix) == 288
    assert matrix["run_id"].is_unique
    assert set(matrix["model"]) == {"nflows", "cinn", "cfm"}
    assert set(matrix["training_seed"].astype(int)) == {42, 43}
    assert set(matrix.loc[matrix["arm"] == "auto_delivery", "target_type"]) == {
        "ID",
        "Interpolation",
    }
    assert len(matrix.loc[matrix["arm"] == "auto_delivery"]) == 240
    assert len(matrix.loc[matrix["arm"] == "multiobjective_refinement"]) == 48


def test_generated_configs_preserve_v21_and_route_multiobjective_to_nsga2():
    repo_root = Path(__file__).resolve().parents[1]
    matrix_config = load_stage2_wp5_matrix_config(
        repo_root / "config" / "stage2" / "s2_wp5_matrix.yaml"
    )
    template = load_stage2_wp5_config(
        repo_root / "config" / "stage2" / "s2_wp5.yaml"
    )
    matrix = build_wp5_run_matrix(matrix_config, repo_root)
    row = matrix.loc[matrix["arm"] == "multiobjective_refinement"].iloc[0]
    generated = _run_config(template, matrix_config, row)

    assert generated["generation"]["n_raw"] == 128
    assert generated["generation"]["candidate_protocol"] == "v2_1_fixed_raw"
    assert generated["delivery"]["extra_engineering_objectives"] is True
    assert generated["nsga2"]["objectives"] == [
        "target_weighted_relative_error",
        "total_mass_kg",
    ]
    assert generated["nsga2"]["maximize"] == [False, False]


def test_formal_matrix_rejects_non_multiobjective_secondary_arm(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (repo_root / "config" / "stage2" / "s2_wp5_matrix.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["arms"]["multiobjective_refinement"]["extra_engineering_objectives"] = False
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="extra engineering objectives"):
        load_stage2_wp5_matrix_config(path)
