"""WP5 Warm-Start initialization and 5D-to-19D adapter tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import src.ai.optimizer as optimizer_module
from src.ai.dataset import DesignSpaceBounds
from src.ai.optimizer import (
    INTEGER_PARAMETER_NAMES,
    TARGET_WEIGHTED_RELATIVE_ERROR,
    InitialPopulation,
    Individual,
    OptimizationConfig,
    optimize,
    optimizer_parameter_space,
)
from src.ai.warm_start import (
    build_background_population,
    build_warm_start_population,
)
from src.ai.wp5 import run_wp5


GENERATOR_FIELDS = [
    "compressor_pressure_ratio",
    "compressor_efficiency",
    "turbine_inlet_temp_K",
    "turbine_efficiency",
    "mass_flow_kg_s",
]


@pytest.fixture
def fake_evaluation(monkeypatch):
    def evaluate(population, _param_names, config):
        for individual in population:
            value = float(np.sum(individual.genes))
            individual.objectives = np.asarray(
                [value + index for index in range(len(config.objectives))]
            )
            individual.performance = {
                name: value for name in config.objectives
            }
            individual.is_feasible = True

    monkeypatch.setattr(optimizer_module, "_evaluate_population", evaluate)


def _config(population_size=8, n_generations=0, **kwargs):
    return OptimizationConfig(
        population_size=population_size,
        n_generations=n_generations,
        **kwargs,
    )


def _unique_population(count, bounds=None):
    bounds = bounds or DesignSpaceBounds()
    names, lower, upper = optimizer_parameter_space(bounds)
    fractions = np.linspace(0.05, 0.95, count)
    genes = np.asarray([lower + fraction * (upper - lower) for fraction in fractions])
    return genes, names


def _generator_candidates(count=120):
    fractions = np.linspace(0.01, 0.99, count)
    return pd.DataFrame(
        {
            "candidate_id": np.arange(count),
            "compressor_pressure_ratio": 2.0 + 3.0 * fractions,
            "compressor_efficiency": 0.70 + 0.15 * fractions,
            "turbine_inlet_temp_K": 800.0 + 400.0 * fractions,
            "turbine_efficiency": 0.75 + 0.13 * fractions,
            "mass_flow_kg_s": 0.08 + 0.22 * fractions,
        }
    )


def test_no_initial_arguments_preserve_exact_wp0_random_population(fake_evaluation):
    bounds = DesignSpaceBounds()
    names, lower, upper = optimizer_parameter_space(bounds)
    seed = 917
    expected_rng = np.random.RandomState(seed)
    expected = np.asarray(
        [
            lower + expected_rng.random(len(names)) * (upper - lower)
            for _ in range(6)
        ]
    )

    np.random.seed(seed)
    _, history = optimize(
        config=_config(population_size=6), bounds=bounds, verbose=False
    )

    np.testing.assert_array_equal(history["initial_population_genes"], expected)
    assert history["initialization"]["mode"] == "legacy_random"
    assert history["initialization"]["wp0_compatible"] is True


def test_valid_19d_population_enters_optimizer_and_projects_integers(fake_evaluation):
    genes, names = _unique_population(8)
    _, history = optimize(
        config=_config(population_size=8), initial_genes=genes, verbose=False
    )
    actual = np.asarray(history["initial_population_genes"])

    assert actual.shape == (8, 19)
    for name in INTEGER_PARAMETER_NAMES:
        column = names.index(name)
        np.testing.assert_array_equal(actual[:, column], np.rint(actual[:, column]))
    assert history["initialization"]["integer_projection_value_count"] > 0


def test_short_population_random_fill_is_seed_reproducible(fake_evaluation):
    genes, _ = _unique_population(3)
    config = _config(population_size=8, warm_start_seed=123)
    _, first = optimize(config=config, initial_genes=genes, verbose=False)
    _, second = optimize(config=config, initial_genes=genes, verbose=False)

    np.testing.assert_array_equal(
        first["initial_population_genes"], second["initial_population_genes"]
    )
    assert first["initialization"]["fill_count"] == 5
    assert [row["source"] for row in first["initial_population_provenance"]][-5:] == [
        "random_fill"
    ] * 5


def test_long_population_is_stably_truncated(fake_evaluation):
    genes, _ = _unique_population(10)
    _, history = optimize(
        config=_config(population_size=6), initial_genes=genes, verbose=False
    )

    assert history["initialization"]["truncated_count"] == 4
    assert history["initialization"]["fill_count"] == 0
    assert len(history["initial_population_genes"]) == 6


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_initial_gene_is_rejected(fake_evaluation, invalid):
    genes, _ = _unique_population(4)
    genes[1, 2] = invalid
    with pytest.raises(ValueError, match="NaN/inf"):
        optimize(
            config=_config(population_size=4), initial_genes=genes, verbose=False
        )


def test_out_of_bounds_policy_rejects_or_clips(fake_evaluation):
    genes, _ = _unique_population(4)
    genes[0, 0] = 1000.0
    with pytest.raises(ValueError, match="outside DesignSpaceBounds"):
        optimize(
            config=_config(population_size=4), initial_genes=genes, verbose=False
        )

    _, history = optimize(
        config=_config(
            population_size=4, warm_start_out_of_bounds_policy="clip"
        ),
        initial_genes=genes,
        verbose=False,
    )
    assert history["initial_population_genes"][0][0] == 5.0
    assert history["initialization"]["out_of_bounds_value_count"] == 1


def test_duplicates_are_removed_before_fill(fake_evaluation):
    genes, _ = _unique_population(3)
    duplicated = np.vstack([genes[0], genes[0], genes[1], genes[2]])
    _, history = optimize(
        config=_config(population_size=4, warm_start_seed=7),
        initial_genes=duplicated,
        verbose=False,
    )

    assert history["initialization"]["duplicate_count"] == 1
    assert history["initialization"]["duplicate_source_rows"] == [1]
    assert history["initialization"]["fill_count"] == 1


def test_initial_genes_and_initializer_are_mutually_exclusive(fake_evaluation):
    genes, _ = _unique_population(4)
    with pytest.raises(ValueError, match="mutually exclusive"):
        optimize(
            config=_config(population_size=4),
            initial_genes=genes,
            initializer=lambda _context: genes,
            verbose=False,
        )


def test_custom_initializer_receives_19d_context(fake_evaluation):
    captured = {}

    def initializer(context):
        captured["names"] = context.param_names
        captured["shape"] = context.bounds_lo.shape
        return np.tile((context.bounds_lo + context.bounds_hi) / 2.0, (5, 1))

    _, history = optimize(
        config=_config(population_size=5), initializer=initializer, verbose=False
    )

    assert len(captured["names"]) == 19
    assert captured["shape"] == (19,)
    # The identical initializer rows are de-duplicated, then safely filled.
    assert history["initialization"]["duplicate_count"] == 4
    assert history["initialization"]["interface"] == "initializer"


def test_external_population_aligns_downstream_rng_with_wp0(fake_evaluation):
    genes, names = _unique_population(5)
    seed = 73
    expected_rng = np.random.RandomState(seed)
    expected_rng.random((5, len(names)))
    expected_next = expected_rng.random()

    np.random.seed(seed)
    _, history = optimize(
        config=_config(population_size=5), initial_genes=genes, verbose=False
    )
    actual_next = np.random.random()

    assert actual_next == expected_next
    assert history["initialization"]["legacy_rng_alignment_draw_count"] == 95


def test_named_5d_overlay_does_not_modify_other_14_dimensions():
    bounds = DesignSpaceBounds()
    background, names = build_background_population(
        bounds, population_size=20, seed=42, method="random"
    )
    candidates = _generator_candidates(20)
    shuffled_fields = list(reversed(GENERATOR_FIELDS))
    result = build_warm_start_population(
        candidates,
        shuffled_fields,
        mode="hybrid",
        population_size=20,
        generator_ratio=0.7,
        background_seed=42,
        bounds=bounds,
    )
    actual = result.population.genes
    generated_indices = [names.index(name) for name in GENERATOR_FIELDS]
    background_indices = [
        index for index, name in enumerate(names) if name not in GENERATOR_FIELDS
    ]

    np.testing.assert_array_equal(
        actual[:, background_indices], background[:, background_indices]
    )
    for row in range(14):
        for field in GENERATOR_FIELDS:
            assert actual[row, names.index(field)] == pytest.approx(
                candidates.iloc[row][field]
            )
    assert result.metadata["selected_generated_count"] == 14
    assert result.provenance["source"].value_counts().to_dict() == {
        "generator": 14,
        "random": 6,
    }


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [(0.0, 0), (0.3, 30), (0.5, 50), (0.7, 70), (1.0, 100)],
)
def test_hybrid_ratio_ablation_builds_exact_population(ratio, expected):
    result = build_warm_start_population(
        _generator_candidates(120),
        GENERATOR_FIELDS,
        mode="hybrid",
        population_size=100,
        generator_ratio=ratio,
        background_seed=9,
    )
    counts = result.provenance["source"].value_counts().to_dict()

    assert result.population.genes.shape == (100, 19)
    assert counts.get("generator", 0) == expected
    assert counts.get("random", 0) == 100 - expected


def test_generator_shortfall_uses_traceable_random_fill():
    result = build_warm_start_population(
        _generator_candidates(3),
        GENERATOR_FIELDS,
        mode="generator",
        population_size=10,
        background_seed=21,
    )

    assert result.metadata["requested_generated_count"] == 10
    assert result.metadata["selected_generated_count"] == 3
    assert result.metadata["random_fill_count"] == 7
    assert result.provenance["source"].value_counts().to_dict() == {
        "random_fill": 7,
        "generator": 3,
    }


@pytest.mark.parametrize("mode", ["random", "generator", "hybrid"])
def test_all_three_modes_complete_nsga2(fake_evaluation, mode):
    np.random.seed(44)
    config = _config(population_size=10, n_generations=2)
    if mode == "random":
        initial = None
    else:
        result = build_warm_start_population(
            _generator_candidates(20),
            GENERATOR_FIELDS,
            mode=mode,
            population_size=10,
            generator_ratio=0.7 if mode == "hybrid" else None,
            background_seed=44,
        )
        initial = result.population
    pareto, history = optimize(
        config=config,
        initial_genes=initial,
        verbose=False,
    )

    assert pareto
    assert len(history["pareto_size"]) == 2
    assert len(history["feasible_count"]) == 2
    assert len(history["initial_population_genes"]) == 10
    assert len(history["evaluation_ledger"]) == 30
    assert history["evaluation_ledger"][0]["evaluation_stage"] == "initial"
    assert history["evaluation_ledger"][-1]["generation"] == 2
    assert all(isinstance(individual, Individual) for individual in pareto)


def test_target_directed_objective_is_weighted_relative_error(monkeypatch):
    monkeypatch.setattr(
        optimizer_module,
        "evaluate_design",
        lambda _row: {
            "is_valid": True,
            "thrust_N": 90.0,
            "tsfc_kg_N_s": 4.4e-5,
            "total_mass_kg": 1.2,
        },
    )
    individual = Individual(genes=np.zeros(19))
    config = _config(
        population_size=1,
        objectives=[TARGET_WEIGHTED_RELATIVE_ERROR],
        maximize=[False],
        target_values={
            "thrust_N": 100.0,
            "tsfc_kg_N_s": 4.0e-5,
            "total_mass_kg": 1.0,
        },
        target_weights={
            "thrust_N": 1.0 / 3.0,
            "tsfc_kg_N_s": 1.0 / 3.0,
            "total_mass_kg": 1.0 / 3.0,
        },
    )

    optimizer_module._evaluate_population(
        [individual], list(DesignSpaceBounds.__dataclass_fields__), config
    )

    assert individual.objectives.tolist() == pytest.approx([(0.1 + 0.1 + 0.2) / 3.0])
    assert individual.performance[TARGET_WEIGHTED_RELATIVE_ERROR] == pytest.approx(
        (0.1 + 0.1 + 0.2) / 3.0
    )
    assert individual.performance["target_relative_error__thrust_N"] == pytest.approx(0.1)


def test_target_directed_objective_rejects_missing_target_values():
    with pytest.raises(ValueError, match="target_values"):
        optimize(
            config=_config(
                population_size=2,
                objectives=[TARGET_WEIGHTED_RELATIVE_ERROR],
                maximize=[False],
            ),
            verbose=False,
        )


def test_target_error_can_be_combined_with_engineering_objective(monkeypatch):
    monkeypatch.setattr(
        optimizer_module,
        "evaluate_design",
        lambda _row: {
            "is_valid": True,
            "thrust_N": 90.0,
            "tsfc_kg_N_s": 4.4e-5,
            "total_mass_kg": 1.2,
        },
    )
    config = _config(
        population_size=2,
        n_generations=0,
        objectives=[TARGET_WEIGHTED_RELATIVE_ERROR, "total_mass_kg"],
        maximize=[False, False],
        target_values={"thrust_N": 100.0, "tsfc_kg_N_s": 4.0e-5},
        target_weights={"thrust_N": 0.5, "tsfc_kg_N_s": 0.5},
    )

    initial, _ = build_background_population(
        DesignSpaceBounds(), population_size=2, seed=17, method="random"
    )
    pareto, history = optimize(
        config=config,
        initial_genes=initial,
        verbose=False,
    )

    assert pareto
    assert all(len(individual.objectives) == 2 for individual in pareto)
    assert len(history["evaluation_ledger"]) == 2


def test_wp5_small_end_to_end_writes_functional_gate(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    candidates = _generator_candidates(12)
    candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))
    candidates.insert(0, "forward_valid", True)
    candidates.insert(0, "target_type", "ID")
    candidates.insert(0, "target_id", "target-test")
    candidates.insert(0, "model", "cvae")
    ranked_path = tmp_path / "ranked.csv"
    candidates.to_csv(ranked_path, index=False)
    config = {
        "version": "wp5_test",
        "input": {
            "ranked_candidates_csv": str(ranked_path),
            "model": "cvae",
            "target_type": "ID",
            "target_id": "target-test",
            "minimum_forward_valid_candidates": 6,
            "generator_schema": "config/generator_v1_schema.yaml",
        },
        "population": {
            "background_seed": 17,
            "background_method": "random",
            "duplicate_tolerance": 1.0e-9,
        },
        "optimizer": {
            "seed": 17,
            "population_size": 6,
            "n_generations": 1,
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
            "warm_start_seed": 17,
            "warm_start_fill_method": "random",
            "warm_start_out_of_bounds_policy": "reject",
            "warm_start_duplicate_tolerance": 1.0e-9,
            "warm_start_align_legacy_rng": True,
        },
        "runs": [
            {"run_id": "random", "mode": "random", "generator_ratio": 0.0},
            {"run_id": "hybrid", "mode": "hybrid", "generator_ratio": 0.5},
            {"run_id": "generator", "mode": "generator", "generator_ratio": 1.0},
        ],
        "output": {"directory": "unused", "refuse_overwrite": True},
    }
    config_path = tmp_path / "wp5.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    output = run_wp5(
        config_path,
        repo_root=repo_root,
        output_dir_override=tmp_path / "output",
    )

    gate = __import__("json").loads(
        (output / "functional_gate.json").read_text(encoding="utf-8")
    )
    summary = pd.read_csv(output / "run_summary.csv")
    assert gate["passed"] is True
    assert set(summary["mode"]) == {"random", "hybrid", "generator"}
    assert (summary["population_size"] == 6).all()
    assert summary["background_14d_preserved"].all()
    assert (output / "runs" / "hybrid" / "population_provenance.csv").is_file()
