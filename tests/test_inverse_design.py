"""WP6 complete inverse-design orchestration tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.ai.baseline import sha256_file
from src.ai.dataset import DesignSpaceBounds, evaluate_design
from src.ai.generator_dataset import (
    GeneratorNormalizer,
    engine_inputs_from_cycle_row,
    load_schema,
)
from src.ai.generators import CVAEGenerator
from src.ai.inverse_design import (
    PipelineInterrupted,
    TargetSpec,
    _read_ranked_candidates,
    load_inverse_design_config,
    run_inverse_design,
)
from src.ai.warm_start import build_background_population
from src.physics.brayton import FlightConditions, solve_brayton_cycle


def test_read_ranked_candidates_accepts_a_legitimate_empty_file(tmp_path: Path):
    path = tmp_path / "ranked_candidates.csv"
    path.write_text("", encoding="utf-8")
    frame = _read_ranked_candidates(path, ["x", "y"])
    assert frame.empty
    assert list(frame.columns) == ["x", "y"]


def _pipeline_fixture(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    schema_path = repo_root / "config" / "generator_v1_schema.yaml"
    schema = load_schema(schema_path)
    design_fields = [field["name"] for field in schema["design_variables"]]
    rows = []
    for index, fraction in enumerate(np.linspace(0.15, 0.85, 24)):
        row = {
            "sample_id": f"sample-{index:03d}",
            "dataset_version": "wp6_test_dataset_v1",
            "split": "train" if index < 18 else "test",
        }
        for field in schema["design_variables"]:
            lower, upper = field["bounds"]
            row[field["name"]] = lower + fraction * (upper - lower)
        cycle = solve_brayton_cycle(
            FlightConditions(), engine_inputs_from_cycle_row(row, schema)
        )
        assert cycle.is_valid
        row["thrust_N"] = cycle.thrust_N
        row["tsfc_kg_N_s"] = cycle.tsfc_kg_N_s
        rows.append(row)
    dataset = pd.DataFrame(rows)
    dataset_path = tmp_path / "dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    normalizer = GeneratorNormalizer.fit(
        dataset.loc[dataset["split"] == "train"], schema
    )
    normalizer_path = tmp_path / "normalizer.json"
    normalizer.save(normalizer_path)

    midpoint = {
        field["name"]: 0.5 * (field["bounds"][0] + field["bounds"][1])
        for field in schema["design_variables"]
    }
    target_cycle = solve_brayton_cycle(
        FlightConditions(), engine_inputs_from_cycle_row(midpoint, schema)
    )
    target = {
        "target_version": "wp6_test_target_v1",
        "target_id": "target-midpoint",
        "target_type": "ID",
        "expected_status": "known_feasible",
        "thrust_N": float(target_cycle.thrust_N),
        "tsfc_kg_N_s": float(target_cycle.tsfc_kg_N_s),
        "tolerance_profile": "round_trip_metrics_v1",
    }
    target_path = tmp_path / "target.yaml"
    target_path.write_text(yaml.safe_dump(target, sort_keys=False), encoding="utf-8")

    model = CVAEGenerator(
        {
            "hidden_dims": [8, 8],
            "latent_dim": 2,
            "beta": 0.001,
            "learning_rate": 0.001,
        },
        {"seed": 4, "device": "cpu", "epochs": 1, "patience": 1},
        schema_version=schema["schema_version"],
        normalizer=normalizer,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint_path = tmp_path / "cvae.pt"
    model.save(checkpoint_path)

    base_config = {
        "version": "wp6_test_run_v1",
        "run_id": "test-generator",
        "mode": "generator",
        "model": {
            "model_id": "cvae",
            "checkpoint_path": str(checkpoint_path),
        },
        "data": {
            "dataset_csv": str(dataset_path),
            "dataset_version": "wp6_test_dataset_v1",
            "schema": str(schema_path),
            "normalizer": str(normalizer_path),
            "normalizer_version": (
                f"{normalizer.schema_version}_{normalizer.fitted_split}_"
                f"{normalizer.fitted_sample_count}"
            ),
            "normalizer_sha256": sha256_file(normalizer_path),
            "metrics_config": str(
                repo_root / "config" / "generator_wp4_metrics.yaml"
            ),
        },
        "seeds": {"generator": 8, "optimizer": 13, "background": 13},
        "generation": {"n_generated": 2},
        "selection": {
            "strategy": "physics_error_diverse",
            "selected_count": 6,
        },
        "population": {
            "generator_ratio": 1.0,
            "background_method": "random",
            "duplicate_tolerance": 1.0e-9,
        },
        "optimizer": {
            "seed": 13,
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
            "warm_start_out_of_bounds_policy": "reject",
        },
        "runtime": {"device": "cpu"},
        "output": {"root": str(tmp_path / "unused"), "refuse_overwrite": True},
    }
    return repo_root, target_path, base_config, normalizer


def _write_run_config(path: Path, config):
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _random_config(base, run_id="test-random"):
    config = copy.deepcopy(base)
    config["run_id"] = run_id
    config["mode"] = "random"
    config["model"] = {"model_id": None, "checkpoint_path": None}
    config["population"]["generator_ratio"] = 0.0
    return config


def _screened_random_config(base, run_id="test-screened-random"):
    config = copy.deepcopy(base)
    config["run_id"] = run_id
    config["mode"] = "screened_random"
    config["model"] = {"model_id": None, "checkpoint_path": None}
    config["generation"]["n_generated"] = 12
    config["population"]["generator_ratio"] = 0.5
    return config


def test_target_and_run_config_validation(tmp_path):
    target = TargetSpec.from_mapping(
        {
            "target_version": "v1",
            "target_id": "t",
            "target_type": "ID",
            "expected_status": "known_feasible",
            "thrust_N": 50.0,
            "tsfc_kg_N_s": 4.0e-5,
            "tolerance_profile": "round_trip_metrics_v1",
        }
    )
    assert target.target_id == "t"
    with pytest.raises(ValueError, match="Unsupported target_type"):
        TargetSpec.from_mapping({**target.to_dict(), "target_type": "UNKNOWN"})


def test_v21_inverse_design_requires_exactly_128_raw_candidates(tmp_path):
    _, _, config, _ = _pipeline_fixture(tmp_path)
    config["generation"]["candidate_protocol"] = "v2_1_fixed_raw"
    config["generation"]["n_generated"] = 127
    config_path = _write_run_config(tmp_path / "invalid-v21.yaml", config)
    with pytest.raises(ValueError, match="n_generated=128"):
        load_inverse_design_config(config_path)


def test_v21_inverse_design_trims_before_round_trip(tmp_path):
    repo_root, target_path, config, _ = _pipeline_fixture(tmp_path)
    config["generation"]["candidate_protocol"] = "v2_1_fixed_raw"
    config["generation"]["n_generated"] = 128
    config_path = _write_run_config(tmp_path / "v21.yaml", config)
    output = run_inverse_design(
        target_path,
        config_path,
        repo_root=repo_root,
        output_dir_override=tmp_path / "v21-output",
    )

    sampling = json.loads(
        (output / "generator" / "sampling_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    checked = pd.read_csv(output / "generator" / "checked_candidates.csv")
    forward = pd.read_csv(output / "round_trip" / "forward_results.csv")

    # The zeroed fixture emits 128 identical rows: one is valid and the rest are
    # duplicates.  Only the retained power-of-two subset may reach the 0D solver.
    assert sampling["candidate_protocol"] == "v2_1_fixed_raw"
    assert sampling["N_raw"] == 128
    assert sampling["N_valid"] == 1
    assert sampling["N_keep"] == 1
    assert sampling["N_trimmed"] == 0
    assert sampling["prefilter_0d_evaluations"] == 1
    assert len(checked) == 128
    assert checked["kept_by_v2_1"].sum() == 1
    assert len(forward) == 1
    assert forward["candidate_id"].tolist() == [0]


def test_complete_generator_pipeline_falls_back_and_balances_ledgers(tmp_path):
    repo_root, target_path, config, _ = _pipeline_fixture(tmp_path)
    config_path = _write_run_config(tmp_path / "run.yaml", config)
    output = run_inverse_design(
        target_path,
        config_path,
        repo_root=repo_root,
        output_dir_override=tmp_path / "output",
    )

    manifest = yaml.safe_load((output / "manifest.yaml").read_text(encoding="utf-8"))
    metrics = json.loads(
        (output / "summary" / "metrics.json").read_text(encoding="utf-8")
    )
    ledger = pd.read_csv(output / "optimization" / "evaluation_ledger.csv")
    cost = pd.read_csv(output / "summary" / "evaluation_cost_ledger.csv")
    provenance = pd.read_csv(output / "population" / "population_provenance.csv")

    assert manifest["status"] == "completed"
    assert all(value["status"] == "completed" for value in manifest["stages"].values())
    assert metrics["population"]["generated_count"] == 1
    assert metrics["population"]["random_fill_count"] == 5
    assert len(ledger) == 12
    assert metrics["cost"]["nsga2_0d_evaluations"] == 12
    assert cost["zero_d_evaluation_count"].sum() == metrics["cost"]["total_0d_evaluations"]
    assert metrics["ledger_checks"]["counts_consistent"] is True
    assert provenance["source"].value_counts().to_dict() == {
        "random_fill": 5,
        "generator": 1,
    }
    pareto = pd.read_csv(output / "optimization" / "pareto.csv")
    assert pareto["provenance"].str.len().gt(2).all()


def test_screened_random_uses_same_physics_filter_without_checkpoint(tmp_path):
    repo_root, target_path, base, _ = _pipeline_fixture(tmp_path)
    config = _screened_random_config(base)
    config_path = _write_run_config(tmp_path / "screened-random.yaml", config)
    output = run_inverse_design(
        target_path,
        config_path,
        repo_root=repo_root,
        output_dir_override=tmp_path / "screened-random-output",
    )

    sampling = json.loads(
        (output / "generator" / "sampling_metrics.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (output / "summary" / "metrics.json").read_text(encoding="utf-8")
    )
    provenance = pd.read_csv(output / "population" / "population_provenance.csv")
    assert sampling["requested_count"] == 12
    assert sampling["returned_count"] == 12
    assert sampling["diagnostics"]["candidate_source"] == "screened_random"
    assert sampling["prefilter_0d_evaluations"] == 12
    assert metrics["cost"]["prefilter_0d_evaluations"] == 12
    assert set(provenance["source"]).issubset({"screened_random", "random_fill", "random"})
    assert (provenance["source"] == "screened_random").sum() > 0


def test_preference_initializer_evaluates_complete_19d_pool_and_counts_cost(tmp_path):
    repo_root, target_path, base, _ = _pipeline_fixture(tmp_path)
    config = _screened_random_config(base, run_id="test-preference")
    reference_genes, names = build_background_population(
        DesignSpaceBounds(), population_size=16, seed=71, method="random"
    )
    reference_rows = []
    for genes in reference_genes:
        reference_rows.append(evaluate_design(pd.Series(dict(zip(names, genes)))))
    reference_path = tmp_path / "objective-reference.csv"
    pd.DataFrame(reference_rows).to_csv(reference_path, index=False)
    config["selection"]["selected_count"] = 12
    config["population"]["preference"] = {
        "enabled": True,
        "pool_size": 16,
        "reference_dataset": str(reference_path),
        "background_seed_offset": 100,
        "directions": [
            [0.4, 0.2, 0.2, 0.2],
            [0.2, 0.4, 0.2, 0.2],
            [0.2, 0.2, 0.4, 0.2],
            [0.2, 0.2, 0.2, 0.4],
        ],
    }
    config_path = _write_run_config(tmp_path / "preference.yaml", config)
    output = run_inverse_design(
        target_path,
        config_path,
        repo_root=repo_root,
        output_dir_override=tmp_path / "preference-output",
    )

    evaluations = pd.read_csv(
        output / "population" / "preference_evaluations.csv"
    )
    selection = pd.read_csv(output / "population" / "preference_selection.csv")
    provenance = pd.read_csv(output / "population" / "population_provenance.csv")
    metrics = json.loads(
        (output / "summary" / "metrics.json").read_text(encoding="utf-8")
    )
    cost = pd.read_csv(output / "summary" / "evaluation_cost_ledger.csv")

    assert len(evaluations) == 16
    assert len(selection) == 3
    assert provenance["preference_selected"].astype(bool).sum() == 3
    assert provenance.loc[:2, "source"].eq("screened_random").all()
    assert metrics["cost"]["preference_19d_evaluations"] == 16
    assert metrics["cost"]["initialization_0d_evaluations"] == 28
    assert cost["zero_d_evaluation_count"].sum() == metrics["cost"][
        "total_0d_evaluations"
    ]


def test_interrupted_pipeline_resumes_from_completed_population(tmp_path):
    repo_root, target_path, base, _ = _pipeline_fixture(tmp_path)
    config = _random_config(base)
    config_path = _write_run_config(tmp_path / "random.yaml", config)
    output = tmp_path / "resume-output"
    with pytest.raises(PipelineInterrupted):
        run_inverse_design(
            target_path,
            config_path,
            repo_root=repo_root,
            output_dir_override=output,
            stop_after_stage="population",
        )
    interrupted = yaml.safe_load((output / "manifest.yaml").read_text(encoding="utf-8"))
    assert interrupted["status"] == "interrupted"
    assert interrupted["stages"]["population"]["status"] == "completed"
    initial_hash = sha256_file(output / "population" / "initial_genes.csv")

    resumed = run_inverse_design(
        target_path,
        config_path,
        repo_root=repo_root,
        output_dir_override=output,
        resume=True,
    )
    completed = yaml.safe_load((resumed / "manifest.yaml").read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert sha256_file(output / "population" / "initial_genes.csv") == initial_hash


def test_resume_rejects_changed_run_config(tmp_path):
    repo_root, target_path, base, _ = _pipeline_fixture(tmp_path)
    config = _random_config(base, run_id="mismatch")
    config_path = _write_run_config(tmp_path / "run.yaml", config)
    output = tmp_path / "mismatch-output"
    with pytest.raises(PipelineInterrupted):
        run_inverse_design(
            target_path,
            config_path,
            repo_root=repo_root,
            output_dir_override=output,
            stop_after_stage="validated",
        )
    config["optimizer"]["n_generations"] = 2
    _write_run_config(config_path, config)
    with pytest.raises(ValueError, match="Resume input mismatch"):
        run_inverse_design(
            target_path,
            config_path,
            repo_root=repo_root,
            output_dir_override=output,
            resume=True,
        )


def test_checkpoint_normalizer_mismatch_is_rejected(tmp_path, monkeypatch):
    repo_root, target_path, config, normalizer = _pipeline_fixture(tmp_path)
    config_path = _write_run_config(tmp_path / "run.yaml", config)
    mismatched = GeneratorNormalizer.from_dict(normalizer.to_dict())
    mismatched.conditions[0] = dict(mismatched.conditions[0])
    mismatched.conditions[0]["mean"] += 1.0
    fake = SimpleNamespace(
        generator_name="cvae",
        normalizer=mismatched,
        schema_version="generator_v1",
    )
    monkeypatch.setattr("src.ai.inverse_design.load_generator", lambda *_args, **_kwargs: fake)

    with pytest.raises(ValueError, match="Checkpoint normalizer differs"):
        run_inverse_design(
            target_path,
            config_path,
            repo_root=repo_root,
            output_dir_override=tmp_path / "bad-output",
        )


def test_out_of_domain_target_skips_sampling_and_falls_back_safely(tmp_path):
    repo_root, target_path, config, _ = _pipeline_fixture(tmp_path)
    target = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    target["target_id"] = "negative-thrust"
    target["target_type"] = "Infeasible"
    target["expected_status"] = "reject"
    target["thrust_N"] = -1.0
    target_path.write_text(yaml.safe_dump(target, sort_keys=False), encoding="utf-8")
    config["run_id"] = "ood-fallback"
    config_path = _write_run_config(tmp_path / "run.yaml", config)

    output = run_inverse_design(
        target_path,
        config_path,
        repo_root=repo_root,
        output_dir_override=tmp_path / "output",
    )
    metrics = json.loads(
        (output / "summary" / "metrics.json").read_text(encoding="utf-8")
    )
    failures = pd.read_csv(output / "round_trip" / "failures.csv")

    assert metrics["generation"]["returned_count"] == 0
    assert metrics["population"]["generated_count"] == 0
    assert metrics["population"]["random_fill_count"] == 6
    assert metrics["cost"]["prefilter_0d_evaluations"] == 0
    assert failures.iloc[0]["failure_code"] == "TARGET_OUT_OF_DOMAIN"


def test_same_target_config_and_seeds_produce_identical_core_artifacts(tmp_path):
    repo_root, target_path, config, _ = _pipeline_fixture(tmp_path)
    config_path = _write_run_config(tmp_path / "run.yaml", config)
    first = run_inverse_design(
        target_path,
        config_path,
        repo_root=repo_root,
        output_dir_override=tmp_path / "first",
    )
    second = run_inverse_design(
        target_path,
        config_path,
        repo_root=repo_root,
        output_dir_override=tmp_path / "second",
    )
    artifacts = [
        "generator/raw_candidates.csv",
        "round_trip/ranked_candidates.csv",
        "population/initial_genes.csv",
        "population/population_provenance.csv",
        "optimization/pareto.csv",
        "optimization/history.csv",
        "optimization/evaluation_ledger.csv",
    ]
    assert all(sha256_file(first / path) == sha256_file(second / path) for path in artifacts)
