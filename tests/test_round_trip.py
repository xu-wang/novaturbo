"""WP4 physics-in-the-loop round-trip tests."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import pytest

from src.ai.generator_dataset import GeneratorNormalizer, load_schema
from src.ai.generator_dataset import engine_inputs_from_cycle_row
from src.ai.dataset import evaluate_design
from src.ai.generators.base import GeneratedSamples
from src.ai.round_trip import (
    BRAYTON_EXCEPTION,
    DUPLICATE,
    INVALID_CYCLE,
    NON_FINITE,
    OUT_OF_BOUNDS,
    SAMPLING_BUDGET_EXHAUSTED,
    TARGET_OUT_OF_DOMAIN,
    DerivedRoundTripMetrics,
    RoundTripEvaluator,
    derive_round_trip_metrics,
    load_metrics_config,
    sampling_budget_exhausted_result,
    target_out_of_domain_result,
)
from src.ai.wp4 import (
    _apply_generator_overrides,
    _resolve_model_spec,
    _sample_normalized_candidates,
    load_wp4_config,
)
from src.physics.brayton import FlightConditions, solve_brayton_cycle


class _DeterministicBatchGenerator:
    design_dim = 2

    def __init__(self):
        self.calls = []
        self._batches = [
            np.asarray(
                [[0.1, 0.2], [1.2, 0.0], [-0.4, 0.5], [0.0, -1.4]],
                dtype=np.float32,
            ),
            np.asarray(
                [[0.6, 0.7], [-0.8, 0.9], [2.0, 0.0], [0.3, -0.2]],
                dtype=np.float32,
            ),
        ]

    def sample(self, condition, n_samples, seed=None, space="normalized"):
        self.calls.append((n_samples, seed, space))
        values = self._batches[len(self.calls) - 1][:n_samples]
        return GeneratedSamples(
            values=torch.as_tensor(values),
            space=space,
            condition=torch.as_tensor(condition),
            seed=seed,
            diagnostics={"fake_batch": len(self.calls) - 1},
        )


class _CFMOverrideTarget:
    generator_name = "cfm"

    def __init__(self):
        self.model_config = {"ode_steps": 32, "unrelated": "frozen"}


class _ReverseScoreReranker:
    def predict_conditions(self, designs):
        return torch.as_tensor(designs)[:, :2]


def test_wp4c_override_changes_only_cfm_ode_steps():
    generator = _CFMOverrideTarget()
    audit = _apply_generator_overrides(
        generator, {"generator_overrides": {"ode_steps": 16}}
    )
    assert audit == {"ode_steps_original": 32, "ode_steps_effective": 16}
    assert generator.model_config == {"ode_steps": 16, "unrelated": "frozen"}


def test_wp4c_config_rejects_non_frozen_ode_steps(tmp_path):
    config = {
        "version": "test",
        "metrics_config": "metrics.yaml",
        "data": {},
        "models": {"cfm": "model.pt"},
        "runtime": {
            "sample_count": 128,
            "sampling": {"policy": "v2_1_fixed_raw"},
            "generator_overrides": {"ode_steps": 12},
        },
        "output": {},
    }
    path = tmp_path / "config.yaml"
    import yaml

    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="8, 16, 32, or 64"):
        load_wp4_config(path)


def test_wp4_model_alias_preserves_legacy_and_resolves_generator_name():
    assert _resolve_model_spec("cfm", "model.pt") == (
        "model.pt",
        "cfm",
        {},
        {},
    )
    assert _resolve_model_spec(
        "diag_cfm_paper",
        {
            "checkpoint": "paper.pt",
            "generator_name": "diag_cfm",
            "generator_overrides": {"ode_method": "euler", "ode_steps": 30},
        },
    ) == (
        "paper.pt",
        "diag_cfm",
        {"ode_method": "euler", "ode_steps": 30},
        {},
    )


def test_diag_rerank_sampling_selects_lowest_reverse_condition_error():
    generator = _DeterministicBatchGenerator()
    values, diagnostics = _sample_normalized_candidates(
        generator,
        np.asarray([0.0, 0.0]),
        2,
        1234,
        {
            "sampling": {"policy": "diag_rerank_fixed_raw", "draw_count": 4},
            "_diag_reranker": _ReverseScoreReranker(),
        },
    )
    np.testing.assert_allclose(values, [[0.1, 0.2], [-0.4, 0.5]])
    assert diagnostics["network_draw_count"] == 4
    assert len(
        diagnostics["generator_diagnostics"]["candidate_diagnostics"][
            "diag_reverse_condition_mse"
        ]
    ) == 2


def test_bounded_rejection_refills_and_preserves_first_draw_seed():
    generator = _DeterministicBatchGenerator()
    runtime = {
        "sampling": {
            "policy": "bounded_rejection",
            "lower_bound": -1.0,
            "upper_bound": 1.0,
            "tolerance": 0.0,
            "batch_size": 4,
            "max_draw_multiplier": 2,
        }
    }

    values, diagnostics = _sample_normalized_candidates(
        generator,
        np.asarray([0.0, 0.0]),
        4,
        1234,
        runtime,
    )

    assert generator.calls[0] == (4, 1234, "normalized")
    np.testing.assert_allclose(
        values,
        np.asarray([[0.1, 0.2], [-0.4, 0.5], [0.6, 0.7], [-0.8, 0.9]]),
    )
    assert diagnostics["returned_count"] == 4
    assert diagnostics["network_draw_count"] == 8
    assert diagnostics["range_rejected_count"] == 3
    assert diagnostics["batch_count"] == 2
    assert diagnostics["budget_exhausted"] is False


def test_sampling_budget_exhaustion_is_a_registered_target_result():
    target = {
        "target_id": "ood-empty",
        "target_type": "OOD-feasible-unknown",
        "expected_status": "feasible_sparse_low_confidence",
    }

    result = sampling_budget_exhausted_result(
        target, "cvae", is_training_domain=False
    )

    assert result.metrics["candidate_count"] == 0
    assert result.metrics["abstention"]["rejected"] is True
    assert result.failures.iloc[0]["failure_code"] == SAMPLING_BUDGET_EXHAUSTED


@pytest.fixture
def schema():
    return load_schema(Path("config/generator_v1_schema.yaml"))


@pytest.fixture
def normalizer(schema):
    names = [field["name"] for field in schema["design_variables"]]
    condition_names = [field["name"] for field in schema["conditions"]]
    rows = []
    for index in range(8):
        fraction = index / 7
        row = {"sample_id": f"sample-{index}"}
        for field in schema["design_variables"]:
            lower, upper = field["bounds"]
            row[field["name"]] = lower + fraction * (upper - lower)
        row[condition_names[0]] = 20.0 + 100.0 * fraction
        row[condition_names[1]] = 2.0e-5 + 3.0e-5 * fraction
        rows.append(row)
    return GeneratorNormalizer.fit(pd.DataFrame(rows), schema)


@pytest.fixture
def metrics_config():
    return load_metrics_config("config/generator_wp4_metrics.yaml")


@pytest.fixture
def derived():
    return DerivedRoundTripMetrics(
        metrics_version="test_v1",
        thrust_relative_tolerance=1.0e-6,
        tsfc_relative_tolerance=1.0e-6,
        total_error_tolerance=1.0e-6,
        thrust_epsilon=1.0e-12,
        tsfc_epsilon=1.0e-15,
        thrust_weight=0.5,
        tsfc_weight=0.5,
        violation_weight=1.0,
        normalized_condition_min=(-3.0, -3.0),
        normalized_condition_max=(3.0, 3.0),
        domain_margin_fraction=0.05,
        derivation={"method": "unit_test"},
    )


def test_known_design_round_trip_has_near_zero_error(
    schema, normalizer, metrics_config, derived
):
    physical = np.asarray([[3.5, 0.80, 1050.0, 0.82, 0.18]])
    cycle = solve_brayton_cycle(
        FlightConditions(), engine_inputs_from_cycle_row(dict(zip(normalizer.design_names, physical[0])), schema)
    )
    assert cycle.is_valid
    target = {
        "target_id": "known",
        "target_type": "ID",
        "expected_status": "known_feasible",
        "thrust_N": cycle.thrust_N,
        "tsfc_kg_N_s": cycle.tsfc_kg_N_s,
    }
    evaluator = RoundTripEvaluator(schema, normalizer, metrics_config, derived)
    normalized = normalizer.transform_design(physical)
    result = evaluator.evaluate(target, normalized, physical)

    assert result.metrics["forward_valid_count"] == 1
    assert result.metrics["best"]["thrust_relative_error"] == pytest.approx(0.0)
    assert result.metrics["best"]["tsfc_relative_error"] == pytest.approx(0.0)
    assert result.metrics["best"]["within_tolerance"] is True


def test_active10d_round_trip_scores_mass_condition():
    schema = load_schema(Path("config/generator_active10d_schema.yaml"))
    row = {
        field["name"]: 0.5 * (field["bounds"][0] + field["bounds"][1])
        for field in schema["design_variables"]
    }
    performance = evaluate_design(pd.Series(row))
    training = pd.DataFrame(
        [
            {
                "sample_id": f"active-{index}",
                **{
                    field["name"]: field["bounds"][0]
                    + (index + 1) / 10 * (field["bounds"][1] - field["bounds"][0])
                    for field in schema["design_variables"]
                },
                "thrust_N": 30.0 + index,
                "tsfc_kg_N_s": 3.0e-5 + index * 1.0e-6,
                "total_mass_kg": 0.4 + index * 0.05,
            }
            for index in range(8)
        ]
    )
    normalizer = GeneratorNormalizer.fit(training, schema)
    metrics = DerivedRoundTripMetrics(
        metrics_version="active10d_test_v1",
        thrust_relative_tolerance=1.0e-8,
        tsfc_relative_tolerance=1.0e-8,
        total_error_tolerance=1.0e-8,
        thrust_epsilon=1.0e-12,
        tsfc_epsilon=1.0e-15,
        thrust_weight=1.0 / 3.0,
        tsfc_weight=1.0 / 3.0,
        violation_weight=1.0,
        normalized_condition_min=(-10.0, -10.0, -10.0),
        normalized_condition_max=(10.0, 10.0, 10.0),
        domain_margin_fraction=0.05,
        derivation={"method": "unit_test"},
        mass_relative_tolerance=1.0e-8,
        mass_epsilon=1.0e-15,
        mass_weight=1.0 / 3.0,
    )
    target = {
        "target_id": "active-known",
        "target_type": "ID",
        "expected_status": "known_feasible",
        "thrust_N": performance["thrust_N"],
        "tsfc_kg_N_s": performance["tsfc_kg_N_s"],
        "total_mass_kg": performance["total_mass_kg"],
    }
    evaluator = RoundTripEvaluator(
        schema,
        normalizer,
        load_metrics_config(Path("config/generator_active10d_metrics.yaml")),
        metrics,
    )
    physical = np.asarray([[row[name] for name in normalizer.design_names]])
    result = evaluator.evaluate(target, normalizer.transform_design(physical), physical)

    assert result.metrics["best"]["mass_relative_error"] == pytest.approx(0.0)
    assert result.metrics["best"]["within_tolerance"] is True


def test_multiple_distinct_inverse_solutions_are_not_scored_by_x_mse(
    schema, normalizer, metrics_config, derived
):
    target = {
        "target_id": "multi",
        "target_type": "ID",
        "expected_status": "known_feasible",
        "thrust_N": 50.0,
        "tsfc_kg_N_s": 4.0e-5,
    }

    def same_performance(_row):
        return SimpleNamespace(
            is_valid=True,
            warnings=[],
            thrust_N=target["thrust_N"],
            tsfc_kg_N_s=target["tsfc_kg_N_s"],
        )

    evaluator = RoundTripEvaluator(
        schema, normalizer, metrics_config, derived, forward_solver=same_performance
    )
    normalized = np.asarray([[-0.8, -0.4, 0.0, 0.4, 0.8], [0.8, 0.4, 0.0, -0.4, -0.8]])
    result = evaluator.evaluate(target, normalized)

    assert result.metrics["forward_valid_count"] == 2
    assert result.forward_results["total_error"].tolist() == pytest.approx([0.0, 0.0])
    assert not any("mse" in column.lower() for column in result.forward_results.columns)
    assert result.metrics["diversity"]["minimum_pairwise_distance"] > 0.0


def test_top_k_uses_effective_candidate_count(
    schema, normalizer, metrics_config, derived
):
    target = {
        "target_id": "short",
        "target_type": "ID",
        "expected_status": "known_feasible",
        "thrust_N": 50.0,
        "tsfc_kg_N_s": 4.0e-5,
    }

    def fake_solver(row):
        valid = row["mass_flow_kg_s"] < 0.2
        return SimpleNamespace(
            is_valid=valid,
            warnings=[] if valid else ["invalid"],
            thrust_N=50.0,
            tsfc_kg_N_s=4.0e-5,
        )

    evaluator = RoundTripEvaluator(
        schema, normalizer, metrics_config, derived, forward_solver=fake_solver
    )
    normalized = np.asarray([[0.0, 0.0, 0.0, 0.0, -0.9], [0.0, 0.2, 0.0, 0.0, 0.9]])
    result = evaluator.evaluate(target, normalized)

    assert result.metrics["top_k"]["5"]["effective_k"] == 1
    assert result.metrics["top_k"]["5"]["success"] is True


def test_physics_error_topk_selects_lowest_target_errors(
    schema, normalizer, metrics_config, derived
):
    target = {
        "target_id": "topk",
        "target_type": "ID",
        "expected_status": "known_feasible",
        "thrust_N": 50.0,
        "tsfc_kg_N_s": 4.0e-5,
    }

    def fake_solver(row):
        offset = row["compressor_pressure_ratio"] - 2.0
        return SimpleNamespace(
            is_valid=True,
            warnings=[],
            thrust_N=50.0 * (1.0 + 0.01 * offset),
            tsfc_kg_N_s=4.0e-5,
        )

    topk_config = copy.deepcopy(metrics_config)
    topk_config["selection"]["strategy"] = "physics_error_topk"
    topk_config["selection"]["selected_count"] = 2
    evaluator = RoundTripEvaluator(
        schema, normalizer, topk_config, derived, forward_solver=fake_solver
    )
    physical = np.asarray(
        [
            [4.0, 0.80, 1050.0, 0.82, 0.18],
            [2.1, 0.80, 1050.0, 0.82, 0.18],
            [3.0, 0.80, 1050.0, 0.82, 0.18],
        ]
    )
    result = evaluator.evaluate(
        target, normalizer.transform_design(physical), physical
    )

    selected = result.ranked_candidates.dropna(subset=["selected_rank"])
    assert selected.sort_values("selected_rank")["candidate_id"].tolist() == [1, 2]
    assert result.metrics["diversity"]["strategy"] == "physics_error_topk"


def test_nonfinite_bounds_duplicates_and_solver_failures_have_codes(
    schema, normalizer, metrics_config, derived
):
    target = {
        "target_id": "failures",
        "target_type": "ID",
        "expected_status": "known_feasible",
        "thrust_N": 50.0,
        "tsfc_kg_N_s": 4.0e-5,
    }

    def fake_solver(row):
        if row["compressor_pressure_ratio"] > 4.5:
            raise RuntimeError("synthetic solver failure")
        if row["mass_flow_kg_s"] > 0.25:
            return SimpleNamespace(
                is_valid=False,
                warnings=["synthetic invalid cycle"],
                thrust_N=10.0,
                tsfc_kg_N_s=5.0e-5,
            )
        return SimpleNamespace(
            is_valid=True, warnings=[], thrust_N=50.0, tsfc_kg_N_s=4.0e-5
        )

    evaluator = RoundTripEvaluator(
        schema, normalizer, metrics_config, derived, forward_solver=fake_solver
    )
    normalized = np.asarray(
        [
            [-1.0, -1.0, -1.0, -1.0, -1.0],  # inclusive boundary is valid
            [-1.0, -1.0, -1.0, -1.0, -1.0],  # duplicate
            [np.nan, 0.0, 0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.9],       # invalid cycle
            [0.9, 0.0, 0.0, 0.0, 0.0],       # exception
        ]
    )
    physical = normalizer.inverse_design(np.nan_to_num(normalized, nan=0.0))
    physical[2, 0] = np.nan
    result = evaluator.evaluate(target, normalized, physical)
    codes = result.forward_results["failure_code"].tolist()

    assert codes[0] == "OK"
    assert codes[1] == DUPLICATE
    assert codes[2] == NON_FINITE
    assert codes[3] == OUT_OF_BOUNDS
    assert codes[4] == INVALID_CYCLE
    assert codes[5] == BRAYTON_EXCEPTION
    selected = result.ranked_candidates.dropna(subset=["selected_rank"])
    assert selected["forward_valid"].all()
    assert (selected["failure_code"] == "OK").all()


def test_ood_target_is_reported_separately(
    schema, normalizer, metrics_config, derived
):
    evaluator = RoundTripEvaluator(schema, normalizer, metrics_config, derived)
    invalid_target = {
        "target_id": "ood",
        "target_type": "Infeasible",
        "expected_status": "reject",
        "thrust_N": -1.0,
        "tsfc_kg_N_s": 4.0e-5,
    }
    with pytest.raises(ValueError, match=TARGET_OUT_OF_DOMAIN):
        evaluator.normalize_target(invalid_target)
    result = target_out_of_domain_result(invalid_target, "cvae")
    assert result.failures.iloc[0]["failure_code"] == TARGET_OUT_OF_DOMAIN
    assert result.metrics["abstention"]["rejected"] is True
    assert result.metrics["abstention"]["false_accept"] is False


def test_feasible_ood_success_is_not_a_false_accept(
    schema, normalizer, metrics_config, derived
):
    target = {
        "target_id": "ood-feasible",
        "target_type": "OOD-feasible-unknown",
        "expected_status": "feasible_sparse_low_confidence",
        "thrust_N": 50.0,
        "tsfc_kg_N_s": 4.0e-5,
    }

    def exact_solver(_row):
        return SimpleNamespace(
            is_valid=True,
            warnings=[],
            thrust_N=target["thrust_N"],
            tsfc_kg_N_s=target["tsfc_kg_N_s"],
        )

    evaluator = RoundTripEvaluator(
        schema, normalizer, metrics_config, derived, forward_solver=exact_solver
    )
    result = evaluator.evaluate(target, np.zeros((1, 5)))

    assert result.metrics["abstention"]["is_ood_target"] is True
    assert result.metrics["abstention"]["is_rejection_target"] is False
    assert result.metrics["abstention"]["accepted"] is True
    assert result.metrics["abstention"]["false_accept"] is False


def test_thresholds_are_derived_from_frozen_split_statistics(
    tmp_path, schema, normalizer, metrics_config
):
    frame = pd.DataFrame(
        [
            {"sample_id": "tr-1", "split": "train", "thrust_N": 50.0, "tsfc_kg_N_s": 3.0e-5},
            {"sample_id": "tr-2", "split": "train", "thrust_N": 100.0, "tsfc_kg_N_s": 5.0e-5},
            {"sample_id": "te-1", "split": "test", "thrust_N": 51.0, "tsfc_kg_N_s": 3.1e-5},
            {"sample_id": "te-2", "split": "test", "thrust_N": 98.0, "tsfc_kg_N_s": 4.9e-5},
        ]
    )
    path = tmp_path / "valid.csv"
    frame.to_csv(path, index=False)
    derived_metrics = derive_round_trip_metrics(path, normalizer, metrics_config)

    assert derived_metrics.thrust_relative_tolerance > 0.0
    assert derived_metrics.tsfc_relative_tolerance > 0.0
    assert derived_metrics.derivation["method"] == "nearest_train_condition_for_each_test_target"
    assert derived_metrics.derivation["dataset_sha256"]
