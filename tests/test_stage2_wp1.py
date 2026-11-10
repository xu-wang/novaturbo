from __future__ import annotations

import pandas as pd

from src.ai.stage2_wp1 import (
    _end_to_end_target_ids,
    aggregate_end_to_end,
    aggregate_generator_only,
    build_target_matrix,
)


def _suite() -> pd.DataFrame:
    rows = []
    for target_type, offset in (("ID", 0.0), ("Interpolation", 0.5)):
        for index in range(8):
            rows.append(
                {
                    "target_id": f"{target_type}-{index:02d}",
                    "target_type": target_type,
                    "thrust_N": 20.0 + 10.0 * index + offset,
                    "tsfc_kg_N_s": 2.5e-5 + 1.0e-6 * (index + offset),
                    "total_mass_kg": 0.3 + 0.05 * ((3 * index) % 8) + offset * 0.01,
                }
            )
    rows.extend(
        [
            {
                "target_id": "OOD-00",
                "target_type": "OOD-feasible-unknown",
                "thrust_N": 200.0,
                "tsfc_kg_N_s": 6.0e-5,
                "total_mass_kg": 1.2,
            },
            {
                "target_id": "BAD-00",
                "target_type": "Infeasible",
                "thrust_N": 300.0,
                "tsfc_kg_N_s": 1.0e-5,
                "total_mass_kg": 0.1,
            },
        ]
    )
    return pd.DataFrame(rows)


SELECTION = {
    "method": "standardized_farthest_point",
    "condition_columns": ["thrust_N", "tsfc_kg_N_s", "total_mass_kg"],
    "end_to_end_counts": {"ID": 4, "Interpolation": 4},
}


def test_target_matrix_uses_full_suite_for_generator_only_and_ordinary_subset():
    matrix = build_target_matrix(_suite(), SELECTION)

    assert matrix["generator_only"].all()
    assert matrix["end_to_end"].sum() == 8
    counts = matrix.loc[matrix["end_to_end"], "target_type"].value_counts()
    assert counts.to_dict() == {"ID": 4, "Interpolation": 4}
    assert not matrix.loc[
        matrix["target_type"].isin(["OOD-feasible-unknown", "Infeasible"]),
        "end_to_end",
    ].any()


def test_target_matrix_selection_is_order_invariant_and_deterministic():
    original = build_target_matrix(_suite(), SELECTION)
    shuffled = build_target_matrix(
        _suite().sample(frac=1.0, random_state=17).reset_index(drop=True), SELECTION
    )

    assert _end_to_end_target_ids(original) == _end_to_end_target_ids(shuffled)


def test_farthest_point_selection_covers_center_and_extremes():
    matrix = build_target_matrix(_suite(), SELECTION)
    chosen = matrix.loc[(matrix["target_type"] == "ID") & matrix["end_to_end"]]

    assert len(chosen) == 4
    assert chosen["condition_boundary_score"].max() > 0.9
    assert chosen["condition_boundary_score"].min() < 0.5


def test_generator_only_aggregate_preserves_seed_and_target_type(tmp_path):
    for seed, error in ((42, 0.1), (43, 0.3)):
        seed_dir = tmp_path / "generator_only" / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        (seed_dir / "functional_gate.json").write_text(
            '{"passed": true}', encoding="utf-8"
        )
        pd.DataFrame([{"model": "nflows", "target_count": 1}]).to_csv(
            seed_dir / "model_summary.csv", index=False
        )
        pd.DataFrame(
            [
                {
                    "model": "nflows",
                    "target_id": f"target-{seed}",
                    "target_type": "ID",
                    "candidate_count": 10,
                    "candidate_valid_count": 8,
                    "forward_valid_count": 8,
                    "best_total_error": error,
                    "best_within_tolerance": error < 0.2,
                    "mean_nearest_neighbour_distance": 0.5,
                }
            ]
        ).to_csv(seed_dir / "target_summary.csv", index=False)
        pd.DataFrame(
            [
                {
                    "model": "nflows",
                    "target_id": f"target-{seed}",
                    "target_type": "ID",
                    "candidate_id": 0,
                    "failure_code": "OUT_OF_BOUNDS",
                    "solver_warnings": "",
                }
            ]
        ).to_csv(seed_dir / "failures.csv", index=False)

    paths = aggregate_generator_only(tmp_path, [42, 43])
    summary = pd.read_csv(paths["target_type_summary"])
    failures = pd.read_csv(paths["failure_summary"])

    assert summary.loc[0, "training_seed_count"] == 2
    assert summary.loc[0, "target_run_count"] == 2
    assert summary.loc[0, "within_tolerance_rate"] == 0.5
    assert summary.loc[0, "best_total_error_mean"] == 0.2
    assert failures.loc[0, "failure_count"] == 2


def test_end_to_end_aggregate_checks_count_and_builds_difficulty(tmp_path):
    benchmark_dir = tmp_path / "end_to_end"
    benchmark_dir.mkdir()
    metrics = pd.DataFrame(
        [
            {
                "target_id": target,
                "target_type": "ID",
                "method": method,
                "initial_best_error": error,
                "initial_top10_mean_error": error + 0.1,
                "initial_has_hit": error < 0.2,
                "search_best_error": error / 2,
                "search_has_hit": True,
                "search_evaluations_to_hit": 10,
                "total_evaluations": 100,
            }
            for target, method, error in (
                ("easy", "screened_random", 0.1),
                ("easy", "nflows", 0.05),
                ("hard", "screened_random", 0.5),
                ("hard", "nflows", 0.2),
            )
        ]
    )
    metrics.to_csv(benchmark_dir / "run_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "model_id": "nflows",
                "baseline": "screened_random",
                "target_type": "ID",
                "initial_best_win": True,
                "initial_top10_win": True,
                "search_best_win": True,
                "initial_best_error_improvement": 0.05,
                "initial_top10_error_improvement": 0.05,
            }
        ]
    ).to_csv(benchmark_dir / "paired_comparison.csv", index=False)

    paths = aggregate_end_to_end(tmp_path, expected_run_count=4)
    difficulty = pd.read_csv(paths["target_difficulty"])
    summary = pd.read_csv(paths["target_type_summary"])

    assert difficulty.loc[0, "target_id"] == "hard"
    assert set(summary["method"]) == {"screened_random", "nflows"}
