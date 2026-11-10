"""WP7 B0 experiment-matrix and diagnosis tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.ai.wp7 import (
    _matched_comparison,
    _nondominated,
    _screening_control_comparison,
    _target_error_columns,
    analyze_wp7,
    build_experiment_matrix,
    build_run_matrix,
)


def test_run_matrix_pairs_random_and_all_generators():
    config = {
        "seeds": [4, 5],
        "models": [
            {"model_id": "cvae"},
            {"model_id": "nflows"},
            {"model_id": "cinn"},
            {"model_id": "cfm"},
        ],
    }
    targets = pd.DataFrame(
        [
            {
                "target_id": "target-abcdef",
                "target_type": "ID",
            }
        ]
    )
    runs = build_run_matrix(config, targets)
    assert len(runs) == 10
    assert runs.groupby(["target_id", "seed"]).size().tolist() == [5, 5]
    logical = build_experiment_matrix(runs)
    assert set(logical["logical_arm"]) == {"A", "B", "C", "D"}
    assert len(logical) == 20


def test_run_matrix_adds_one_shared_screened_random_control_per_pair():
    config = {
        "seeds": [4],
        "screened_random_control": True,
        "models": [{"model_id": "cvae"}, {"model_id": "nflows"}],
    }
    targets = pd.DataFrame([{"target_id": "target-abcdef", "target_type": "ID"}])
    runs = build_run_matrix(config, targets)
    assert runs["arm"].tolist() == ["B", "S", "D", "D"]
    logical = build_experiment_matrix(runs)
    assert set(logical["logical_arm"]) == {"A", "B", "C", "D", "E", "F"}


def test_nondominated_and_target_error_are_minimization_consistent():
    points = np.asarray([[1.0, 2.0], [2.0, 1.0], [2.0, 2.0], [1.5, 1.5]])
    front = _nondominated(points)
    assert {tuple(value) for value in front} == {
        (1.0, 2.0),
        (2.0, 1.0),
        (1.5, 1.5),
    }
    ledger = pd.DataFrame(
        {
            "performance__thrust_N": [100.0, 90.0],
            "performance__tsfc_kg_N_s": [4.0e-5, 4.4e-5],
            "is_feasible": [True, False],
        }
    )
    derived = {
        "epsilons": {"thrust_N": 1e-12, "tsfc_kg_N_s": 1e-15},
        "weights": {"thrust": 0.5, "tsfc": 0.5, "constraint_violation": 1.0},
        "success_tolerances": {
            "thrust_relative": 0.01,
            "tsfc_relative": 0.01,
            "weighted_total": 0.01,
        },
    }
    scored = _target_error_columns(
        ledger,
        {"thrust_N": 100.0, "tsfc_kg_N_s": 4.0e-5},
        derived,
    )
    assert scored["within_tolerance"].tolist() == [True, False]
    assert scored.loc[1, "total_error"] > 1.0


def test_end_to_end_comparison_uses_common_budget_checkpoint():
    rows = []
    for scope in ("nsga_only", "end_to_end"):
        for method, arm, model, counts, hypervolumes in (
            ("random_nsga2", "B", "random", [10, 20], [0.2, 0.5]),
            ("cvae_hybrid", "D", "cvae", [15, 25] if scope == "end_to_end" else [10, 20], [0.3, 0.7]),
        ):
            for generation, (count, hv) in enumerate(zip(counts, hypervolumes)):
                rows.append(
                    {
                        "run_id": f"{model}-{scope}",
                        "logical_arm": arm,
                        "method": method,
                        "model_id": model,
                        "target_id": "target",
                        "target_type": "ID",
                        "seed": 42,
                        "budget_scope": scope,
                        "generation": generation,
                        "evaluation_count": count,
                        "hypervolume": hv,
                        "best_total_error": 1.0 - hv,
                    }
                )
    matched = _matched_comparison(pd.DataFrame(rows))
    end_to_end = matched.loc[matched["budget_scope"] == "end_to_end"].iloc[0]
    assert end_to_end["common_budget"] == 20
    assert end_to_end["hybrid_evaluation_count"] == 15
    assert end_to_end["random_evaluation_count"] == 20
    assert end_to_end["hypervolume_delta"] == 0.3 - 0.5


def test_generator_is_compared_directly_with_screened_random_at_common_budget():
    rows = []
    for scope in ("nsga_only", "end_to_end"):
        for arm, model, counts, hypervolumes in (
            ("F", "screened_random", [16, 26], [0.4, 0.6]),
            ("D", "cvae", [14, 24], [0.5, 0.8]),
        ):
            for generation, (count, hv) in enumerate(zip(counts, hypervolumes)):
                rows.append(
                    {
                        "run_id": f"{model}-{scope}",
                        "logical_arm": arm,
                        "model_id": model,
                        "target_id": "target",
                        "target_type": "ID",
                        "seed": 42,
                        "budget_scope": scope,
                        "generation": generation,
                        "evaluation_count": count,
                        "hypervolume": hv,
                        "best_total_error": 1.0 - hv,
                    }
                )
    compared = _screening_control_comparison(pd.DataFrame(rows))
    end_to_end = compared.loc[compared["budget_scope"] == "end_to_end"].iloc[0]
    assert end_to_end["common_budget"] == 24
    assert end_to_end["generator_evaluation_count"] == 24
    assert end_to_end["screened_random_evaluation_count"] == 16
    assert np.isclose(end_to_end["hypervolume_delta"], 0.8 - 0.4)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fake_run(root: Path, run, hybrid: bool):
    run_dir = root / "runs" / run["run_id"]
    for name in ("generator", "round_trip", "optimization", "summary"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.yaml").write_text("status: completed\n", encoding="utf-8")
    target = {"thrust_N": 100.0, "tsfc_kg_N_s": 4.0e-5}
    (run_dir / "target.yaml").write_text(yaml.safe_dump(target), encoding="utf-8")
    derived = {
        "epsilons": {"thrust_N": 1e-12, "tsfc_kg_N_s": 1e-15},
        "weights": {"thrust": 0.5, "tsfc": 0.5, "constraint_violation": 1.0},
        "success_tolerances": {
            "thrust_relative": 0.03,
            "tsfc_relative": 0.03,
            "weighted_total": 0.03,
        },
    }
    _write_json(run_dir / "derived_metrics.json", derived)
    thrust = [80.0, 105.0, 95.0, 101.0] if hybrid else [70.0, 90.0, 92.0, 96.0]
    ledger = pd.DataFrame(
        {
            "evaluation_index": range(4),
            "generation": [0, 0, 1, 1],
            "is_feasible": [True] * 4,
            "objective_internal_0": -np.asarray(thrust),
            "objective_internal_1": [4.7e-5, 4.1e-5, 4.2e-5, 4.0e-5],
            "objective_internal_2": [1.0, 0.9, 0.8, 0.7],
            "objective_internal_3": [-8.0, -10.0, -11.0, -12.0],
            "performance__thrust_N": thrust,
            "performance__tsfc_kg_N_s": [4.7e-5, 4.1e-5, 4.2e-5, 4.0e-5],
        }
    )
    ledger.to_csv(run_dir / "optimization" / "evaluation_ledger.csv", index=False)
    prefilter = 2 if hybrid else 0
    _write_json(
        run_dir / "summary" / "metrics.json",
        {
            "cost": {
                "prefilter_0d_evaluations": prefilter,
                "nsga2_0d_evaluations": 4,
                "total_0d_evaluations": 4 + prefilter,
                "current_execution_wall_seconds": 1.0,
                "generator_sampling_seconds": 0.1 if hybrid else 0.0,
                "round_trip_seconds": 0.1 if hybrid else 0.0,
                "optimizer_seconds": 0.8,
            }
        },
    )
    _write_json(
        run_dir / "generator" / "sampling_metrics.json",
        {"prefilter_0d_evaluations": prefilter, "ode_nfe": 8 if hybrid else 0},
    )
    if hybrid:
        forward = pd.DataFrame(
            {
                "candidate_id": [0, 1],
                "forward_valid": [True, True],
                "total_error": [0.02, 0.08],
                "within_tolerance": [True, False],
            }
        )
        ranked = forward.copy()
        ranked["selected_rank"] = [1.0, np.nan]
        metrics = {
            "duplicate_rate": 0.0,
            "diversity": {
                "mean_nearest_neighbour_distance": 0.5,
                "coverage_fraction_by_dimension": {"x": 0.5, "y": 0.7},
            },
            "top_k": {
                "1": {"effective_k": 1, "success": True, "best_total_error": 0.02}
            },
        }
        failures = pd.DataFrame({"failure_code": ["OUT_OF_BOUNDS"]})
    else:
        forward = pd.DataFrame(columns=["candidate_id", "forward_valid", "total_error", "within_tolerance"])
        ranked = pd.DataFrame(columns=[*forward.columns, "selected_rank"])
        metrics = {"duplicate_rate": 0.0, "diversity": {}, "top_k": {}}
        failures = pd.DataFrame(columns=["failure_code"])
    forward.to_csv(run_dir / "round_trip" / "forward_results.csv", index=False)
    ranked.to_csv(run_dir / "round_trip" / "ranked_candidates.csv", index=False)
    failures.to_csv(run_dir / "round_trip" / "failures.csv", index=False)
    _write_json(run_dir / "round_trip" / "metrics.json", metrics)


def test_analysis_writes_b0_outputs_without_performance_gate(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "b0"
    output.mkdir()
    matrix = pd.DataFrame(
        [
            {
                "run_id": "random-run",
                "arm": "B",
                "mode": "random",
                "model_id": "random",
                "target_id": "target",
                "target_type": "ID",
                "seed": 42,
            },
            {
                "run_id": "cvae-run",
                "arm": "D",
                "mode": "hybrid",
                "model_id": "cvae",
                "target_id": "target",
                "target_type": "ID",
                "seed": 42,
            },
        ]
    )
    for run in matrix.to_dict(orient="records"):
        _fake_run(output, run, run["arm"] == "D")
    reference = tmp_path / "reference.csv"
    pd.DataFrame(
        {
            "is_valid": [True, True, True],
            "thrust_N": [50.0, 100.0, 150.0],
            "tsfc_kg_N_s": [3e-5, 4e-5, 5e-5],
            "total_mass_kg": [0.5, 1.0, 1.5],
            "thrust_to_weight": [5.0, 10.0, 15.0],
        }
    ).to_csv(reference, index=False)
    config = {
        "base_run_config": "config/inverse_design/runs/wp7_b0_smoke_template.yaml",
        "analysis": {
            "objective_reference_dataset": str(reference),
            "normalized_reference_point": [1.1] * 4,
            "hypervolume_sobol_power": 6,
            "confidence_z": 1.96,
        },
    }
    summary = analyze_wp7(output, config, matrix, repo_root)
    assert summary["benchmark_is_gate"] is False
    assert summary["generator_superiority_required"] is False
    assert summary["completed_run_count"] == 2
    benchmark = pd.read_csv(output / "baseline_benchmark.csv")
    assert set(benchmark["budget_scope"]) == {"nsga_only", "end_to_end"}
    assert (output / "stage_attribution.csv").is_file()
    assert (output / "failure_breakdown.csv").is_file()
    assert (output / "baseline_diagnosis.md").is_file()
    assert (output / "stage2_hypotheses.md").is_file()
    assert (output / "comparison_curves" / "hypervolume_end_to_end.png").is_file()
