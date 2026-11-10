"""WP19-23E NSGA-II initialization and Pareto tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import src.ai.geometry19_nsga2 as nsga2
from src.ai.geometry19_nsga2 import (
    build_initial_population,
    crowding_distance,
    hit_aligned_fronts,
    nondominated_fronts,
    power_of_two_population_size,
    run_nsga2_stage,
)
from src.ai.geometry19_inverse_contract import load_target
from src.physics import ModelConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def _fields() -> list[dict]:
    schema = yaml.safe_load(
        (REPO_ROOT / "config/physics/geometry19_schema_v1.yaml").read_text(encoding="utf-8")
    )
    return list(schema["design_variables"])


def test_nondominated_sort_and_crowding_are_deterministic() -> None:
    objectives = np.asarray([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 4.0]])
    fronts = nondominated_fronts(objectives)
    assert fronts == [[0, 1, 2], [3]]
    distance = crowding_distance(objectives, fronts[0])
    assert np.isinf(distance[0])
    assert distance[1] == 2.0
    assert np.isinf(distance[2])


def test_hit_aligned_fronts_put_failed_axes_before_constraint_violation() -> None:
    objectives = np.asarray(
        [
            [0.8, 1.4, 0.7],
            [1.1, 1.1, 0.7],
            [0.9, 0.9, 0.9],
        ]
    )
    feasible = np.asarray([False, True, False])
    violations = np.asarray([0.2, 0.0, 0.5])
    assert hit_aligned_fronts(objectives, violations, feasible) == [[2], [0], [1]]


def test_initial_population_keeps_warm_designs_and_projects_integer_fill() -> None:
    fields = _fields()
    geometry = {item["name"]: item["reference_value"] for item in fields}
    warm = [
        {
            "evaluation_id": "dflow:1",
            "design_hash": "one",
            "geometry19": geometry,
            "physics_acceptable": True,
            "total_true_error": 0.1,
        }
    ]
    genes, provenance = build_initial_population(warm, fields, 40, 42)
    assert genes.shape == (40, 19)
    assert provenance[0]["source"] == "true_physics_warm_start"
    integer_indices = [index for index, item in enumerate(fields) if item["dtype"] == "int64"]
    assert np.array_equal(genes[:, integer_indices], np.rint(genes[:, integer_indices]))
    lower = np.asarray([item["bounds"][0] for item in fields])
    upper = np.asarray([item["bounds"][1] for item in fields])
    assert np.all(genes >= lower)
    assert np.all(genes <= upper)


@pytest.mark.parametrize(
    ("warm_count", "expected"),
    [(1, 16), (16, 16), (17, 32), (32, 32), (33, 64), (64, 64), (80, 64)],
)
def test_population_size_is_bounded_power_of_two(
    warm_count: int, expected: int
) -> None:
    assert power_of_two_population_size(warm_count, 16, 64) == expected


def test_zero_valid_candidate_summary_runs_labeled_cold_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_path = REPO_ROOT / "config/inverse_design/targets/wp19_23_smoke_test0_v1.yaml"
    target = load_target(target_path)
    calibration = ModelConfig.from_yaml(
        REPO_ROOT / "config/physics/geometry19_calibration_bootstrap_v1.yaml"
    )
    candidate_root = tmp_path / "direct"
    candidate_root.mkdir()
    (candidate_root / "summary.json").write_text(
        json.dumps(
            {
                "target_id": target.target_id,
                "sampling_mode": "direct",
                "candidate_protocol": {"n_valid": 0},
                "hashes": {"calibration_hash": calibration.calibration_pack_hash},
            }
        ),
        encoding="utf-8",
    )

    def evaluate(candidate_id, variant, geometry, design_hash, *args, **kwargs):
        error = 0.2 + 1.0e-6 * candidate_id
        return {
            "candidate_id": candidate_id,
            "evaluation_id": f"{variant}:{candidate_id}",
            "variant": variant,
            "design_hash": design_hash,
            "geometry19": geometry,
            "physics_acceptable": True,
            "performance": {
                "thrust_N": target.thrust_N * (1.0 + error),
                "tsfc_kg_N_s": target.tsfc_kg_N_s * (1.0 + error),
                "mass_kg": target.mass_kg * (1.0 + error),
            },
            "relative_errors": {
                "thrust_N": error,
                "tsfc_kg_N_s": error,
                "mass_kg": error,
            },
            "total_true_error": error,
            "development_diagnostic_hit": False,
            "root_classification": "CONVERGED_UNIQUE",
            "flow_root_count": 1,
            "selected_flow_root_index": 0,
        }

    monkeypatch.setattr(nsga2, "evaluate_geometry19_candidate", evaluate)
    summary = run_nsga2_stage(
        REPO_ROOT,
        target_path,
        candidate_root=candidate_root,
        sampling_mode="direct",
        cold_start=True,
        output_root=tmp_path / "wp19_23e",
    )
    assert summary["initialization_mode"] == "cold_start_lhs"
    assert summary["counts"]["population_size"] == 64
    assert summary["counts"]["warm_start_count"] == 0
    assert summary["counts"]["lhs_fill_count"] == 64
    assert summary["counts"]["logical_physics_evaluations"] == 640


def test_cold_start_comparator_rejects_production_cold_start_inputs(
    tmp_path: Path,
) -> None:
    target_path = REPO_ROOT / "config/inverse_design/targets/wp19_23_smoke_test0_v1.yaml"
    with pytest.raises(nsga2.Geometry19NSGA2Error, match="mutually exclusive"):
        run_nsga2_stage(
            REPO_ROOT,
            target_path,
            cold_start=True,
            cold_start_comparison=True,
            output_root=tmp_path / "wp19_23e",
        )
