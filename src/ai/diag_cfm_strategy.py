"""Independent high-precision CFM-4/Diag-CFM delivery strategy.

This module intentionally lives outside :mod:`stage2_wp5`, whose source hash is
part of the frozen CFM-4 production baseline.  It reuses the frozen numerical
helpers but owns the optional prefilter, forced gradient pass, accounting, and
conditional NSGA-II policy.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from .baseline import (
    collect_environment,
    collect_git_state,
    collect_source_manifest,
    sha256_file,
    write_json,
    write_yaml,
)
from .candidate_protocol import V21_RAW_CANDIDATE_COUNT, apply_candidate_protocol_v21
from .dataset import DesignSpaceBounds
from .generator_dataset import GeneratorNormalizer, condition_names, design_names, load_schema
from .generators import load_generator
from .inverse_design import TARGET_TYPES, TargetSpec
from .optimizer import optimize
from .round_trip import (
    RoundTripEvaluator,
    derive_round_trip_metrics,
    load_metrics_config,
    sampling_budget_exhausted_result,
)
from .stage2_wp3 import load_forward_surrogate
from .stage2_wp5 import (
    _accept_updates,
    _hit_count,
    _nsga2_config,
    _pareto_frame,
    _resolve,
    _resolve_device,
    _top_candidate_ids,
    load_stage2_wp5_config,
    surrogate_gradient_refinement,
)
from .warm_start import build_warm_start_population


def load_diag_cfm_strategy_config(path: Path) -> Dict[str, Any]:
    """Validate the frozen WP5 contract plus the independent strategy fields."""

    config = load_stage2_wp5_config(path)
    prefilter = dict(config["generation"].get("diag_prefilter", {}))
    if not bool(prefilter.get("enabled", False)):
        raise ValueError("Diag-CFM full strategy requires diag_prefilter.enabled=true")
    if int(prefilter.get("pool_size", 0)) < V21_RAW_CANDIDATE_COUNT:
        raise ValueError("diag_prefilter.pool_size must be at least 128")
    if not prefilter.get("reranker_checkpoint"):
        raise ValueError("diag_prefilter requires reranker_checkpoint")
    if str(prefilter.get("score", "reverse_condition_mse")) != "reverse_condition_mse":
        raise ValueError("diag_prefilter only supports reverse_condition_mse")
    if str(config["delivery"]["mode"]) != "generator_auto_refinement":
        raise ValueError("Diag-CFM full strategy requires generator_auto_refinement")
    gradient_enabled = bool(config["gradient"].get("enabled", True))
    if gradient_enabled and not config["gradient"].get("surrogate_checkpoint"):
        raise ValueError("enabled gradient refinement requires surrogate_checkpoint")
    return config


def _load_strategy_target(
    repo_root: Path, raw: Mapping[str, Any]
) -> tuple[TargetSpec, str]:
    if raw.get("spec_yaml"):
        mapping = yaml.safe_load(
            _resolve(repo_root, str(raw["spec_yaml"])).read_text(encoding="utf-8")
        )
    else:
        suite = pd.read_csv(_resolve(repo_root, str(raw["suite_csv"])))
        target_id = str(raw["target_id"])
        selected = suite.loc[suite["target_id"].astype(str) == target_id]
        if len(selected) != 1:
            raise ValueError(f"target_id={target_id} must identify exactly one target")
        row = selected.iloc[0]
        mapping = {
            "target_version": str(row.get("target_suite_version", "stage2_target_v1")),
            "target_id": target_id,
            "target_type": str(row["target_type"]),
            "expected_status": str(row["expected_status"]),
            "thrust_N": float(row["thrust_N"]),
            "tsfc_kg_N_s": float(row["tsfc_kg_N_s"]),
            "total_mass_kg": float(row["total_mass_kg"]),
            "tolerance_profile": str(
                raw.get("tolerance_profile", "round_trip_metrics_v1")
            ),
        }
    original_type = str(mapping["target_type"])
    if original_type not in TARGET_TYPES:
        if original_type != "Boundary":
            raise ValueError(f"Unsupported strategy target_type: {original_type}")
        mapping = {**mapping, "target_type": "OOD-feasible-unknown"}
    return TargetSpec.from_mapping(mapping), original_type


def _full_budget_gradient_config(
    raw: Mapping[str, Any], existing_hits: int, candidate_count: int
) -> Dict[str, Any]:
    """Disable the frozen helper's early-hit exit without changing its source."""

    config = dict(raw)
    config["existing_hit_count"] = int(existing_hits)
    config["minimum_hit_count"] = int(existing_hits + candidate_count + 1)
    return config


def run_diag_cfm_strategy_target(
    config_path: Path,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
) -> Path:
    """Run one auditable CFM-4 -> Diag -> gradient -> 0D -> NSGA decision."""

    repo_root = (repo_root or Path.cwd()).resolve()
    config_path = Path(config_path).resolve()
    config = load_diag_cfm_strategy_config(config_path)
    output_dir = (
        output_dir_override.resolve()
        if output_dir_override is not None
        else _resolve(repo_root, str(config["output"]["directory"]))
    )
    if output_dir.exists() and bool(config["output"].get("refuse_overwrite", True)):
        raise FileExistsError(f"Diag-CFM strategy output exists: {output_dir}")

    data = config["data"]
    dataset_path = _resolve(repo_root, str(data["dataset_csv"]))
    schema_path = _resolve(repo_root, str(data["schema"]))
    normalizer_path = _resolve(repo_root, str(data["normalizer"]))
    metrics_path = _resolve(repo_root, str(data["metrics_config"]))
    checkpoint_path = _resolve(repo_root, str(config["model"]["checkpoint_path"]))
    prefilter = dict(config["generation"]["diag_prefilter"])
    reranker_path = _resolve(repo_root, str(prefilter["reranker_checkpoint"]))
    gradient_enabled = bool(config["gradient"].get("enabled", True))
    nsga2_enabled = bool(config["nsga2"].get("enabled", False))
    surrogate_path = (
        _resolve(repo_root, str(config["gradient"]["surrogate_checkpoint"]))
        if gradient_enabled
        else None
    )
    required_paths = [
        dataset_path,
        schema_path,
        normalizer_path,
        metrics_path,
        checkpoint_path,
        reranker_path,
    ]
    if surrogate_path is not None:
        required_paths.append(surrogate_path)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Diag-CFM strategy input is missing: {path}")

    target, original_target_type = _load_strategy_target(repo_root, config["target"])
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("generation", "direct", "gradient", "nsga2", "delivery"):
        (output_dir / name).mkdir()
    shutil.copy2(config_path, output_dir / "build_config.yaml")
    started = dt.datetime.now(dt.timezone.utc)

    device = _resolve_device(str(config["runtime"].get("device", "auto")))
    schema = load_schema(schema_path)
    normalizer = GeneratorNormalizer.load(normalizer_path)
    if (
        design_names(schema) != normalizer.design_names
        or condition_names(schema) != normalizer.condition_names
    ):
        raise ValueError("Diag-CFM strategy schema and normalizer differ")
    metrics_config = load_metrics_config(metrics_path)
    derived = derive_round_trip_metrics(dataset_path, normalizer, metrics_config)
    evaluator = RoundTripEvaluator(schema, normalizer, metrics_config, derived)
    generator = load_generator(checkpoint_path, device=str(device))
    reranker = load_generator(reranker_path, device=str(device))
    if generator.generator_name != "cfm":
        raise ValueError("Diag-CFM strategy generation checkpoint must be CFM")
    if reranker.generator_name != "diag_cfm":
        raise ValueError("Diag-CFM strategy reranker checkpoint must be Diag-CFM")
    for adapter in (generator, reranker):
        if adapter.normalizer is None or adapter.normalizer.to_dict() != normalizer.to_dict():
            raise ValueError("Diag-CFM strategy generator normalizer differs")
    reranker.model_config["ode_method"] = str(prefilter.get("ode_method", "rk4"))
    reranker.model_config["ode_steps"] = int(prefilter.get("ode_steps", 32))

    normalized_target = evaluator.normalize_target(target.to_dict())
    n_raw = int(config["generation"]["n_raw"])
    pool_size = int(prefilter["pool_size"])
    generation_started = time.perf_counter()
    sampled = generator.sample(
        normalized_target,
        pool_size,
        seed=int(config["generation"]["seed"]),
        space="normalized",
    )
    pool_normalized = sampled.values.detach().cpu().numpy().astype(float)
    rerank_started = time.perf_counter()
    predicted_conditions = reranker.predict_conditions(pool_normalized).detach().cpu().numpy()
    rerank_scores = np.mean(
        (predicted_conditions - normalized_target[None, :]) ** 2, axis=1
    )
    rerank_seconds = time.perf_counter() - rerank_started
    selected_positions = np.argsort(rerank_scores, kind="stable")[:n_raw]
    raw_normalized = pool_normalized[selected_positions]
    generation_seconds = time.perf_counter() - generation_started
    selected_rank = np.full(pool_size, -1, dtype=int)
    selected_rank[selected_positions] = np.arange(n_raw, dtype=int)
    pd.DataFrame(
        pool_normalized,
        columns=[f"normalized_{name}" for name in normalizer.design_names],
    ).assign(
        pool_candidate_id=np.arange(pool_size, dtype=int),
        diag_reverse_condition_mse=rerank_scores,
        selected_for_v2_1=selected_rank >= 0,
        selected_rank=selected_rank,
    ).to_csv(output_dir / "generation" / "prefilter_pool.csv", index=False)

    raw_physical = normalizer.inverse_design(raw_normalized)
    raw_ids = np.arange(n_raw, dtype=int)
    validation = evaluator.validate_candidates(
        raw_normalized, raw_physical, candidate_ids=raw_ids
    )
    protocol = apply_candidate_protocol_v21(
        validation["candidate_id"].to_numpy(int),
        validation["candidate_valid"].to_numpy(bool),
    )
    validation["kept_by_v2_1"] = validation["candidate_id"].isin(
        protocol.kept_candidate_ids
    )
    kept_positions = np.asarray(protocol.kept_positions, dtype=int)
    kept_ids = np.asarray(protocol.kept_candidate_ids, dtype=int)
    kept_normalized = raw_normalized[kept_positions]
    kept_physical = raw_physical[kept_positions]
    pd.DataFrame(
        raw_normalized,
        columns=[f"normalized_{name}" for name in normalizer.design_names],
    ).assign(candidate_id=raw_ids).to_csv(
        output_dir / "generation" / "raw_candidates.csv", index=False
    )
    validation.to_csv(output_dir / "generation" / "validation.csv", index=False)
    write_json(
        output_dir / "generation" / "metrics.json",
        {
            "candidate_protocol": "v2_1_fixed_raw",
            "N_raw": protocol.n_raw,
            "N_valid": protocol.n_valid,
            "N_keep": protocol.n_keep,
            "N_trimmed": protocol.n_trimmed,
            "sampling_seconds": generation_seconds,
            "network_draw_count": pool_size,
            "diag_prefilter_enabled": True,
            "diagnostics": {
                "sampling_path": "cfm_pool_diag_reverse_rerank",
                "pool_size": pool_size,
                "selected_count": n_raw,
                "rerank_seconds": rerank_seconds,
                "generator_diagnostics": dict(sampled.diagnostics),
            },
        },
    )

    direct_started = time.perf_counter()
    if len(kept_ids):
        direct = evaluator.evaluate(
            target.to_dict(), kept_normalized, kept_physical, candidate_ids=kept_ids
        )
    else:
        direct = sampling_budget_exhausted_result(
            target.to_dict(),
            str(config["model"]["model_id"]),
            is_training_domain=evaluator.target_in_training_domain(target.to_dict()),
        )
    direct_seconds = time.perf_counter() - direct_started
    direct.forward_results.to_csv(output_dir / "direct" / "forward_results.csv", index=False)
    direct.ranked_candidates.to_csv(output_dir / "direct" / "ranked_candidates.csv", index=False)
    write_json(output_dir / "direct" / "metrics.json", direct.metrics)
    forward_by_id = {
        int(row["candidate_id"]): dict(row)
        for row in direct.forward_results.to_dict(orient="records")
        if int(row["candidate_id"]) >= 0
    }
    normalized_by_id = {
        int(candidate_id): np.asarray(kept_normalized[index], dtype=float)
        for index, candidate_id in enumerate(kept_ids)
    }
    costs = {
        "direct_0d": int(direct.metrics.get("candidate_valid_count", 0)),
        "latent_repair_0d": 0,
        "gradient_0d": 0,
        "nsga2_0d": 0,
    }
    timings = {
        "generation_and_prefilter": generation_seconds,
        "direct_0d": direct_seconds,
        "latent_repair": 0.0,
        "design_gradient": 0.0,
        "nsga2": 0.0,
    }
    actions = []

    ids = (
        _top_candidate_ids(
            forward_by_id, int(config["gradient"]["candidate_count"])
        )
        if gradient_enabled
        else []
    )
    if gradient_enabled and ids:
        assert surrogate_path is not None
        surrogate, _ = load_forward_surrogate(surrogate_path, normalizer, device)
        action_started = time.perf_counter()
        initial_designs = np.asarray([normalized_by_id[value] for value in ids])
        baseline_errors = [float(forward_by_id[value]["total_error"]) for value in ids]

        def validate(values: np.ndarray, callback_ids: Sequence[int], step: int):
            del step
            physical = normalizer.inverse_design(values)
            result = evaluator.evaluate(
                target.to_dict(), values, physical, candidate_ids=callback_ids
            )
            return result.forward_results, int(result.metrics["candidate_valid_count"])

        gradient_result = surrogate_gradient_refinement(
            initial_designs,
            ids,
            baseline_errors,
            normalized_target,
            surrogate,
            _full_budget_gradient_config(
                config["gradient"], _hit_count(forward_by_id), len(ids)
            ),
            validate,
            weights=config["gradient"].get("condition_weights"),
        )
        costs["gradient_0d"] = int(
            gradient_result.diagnostics["actual_0d_evaluations"]
        )
        accepted = (
            _accept_updates(
                forward_by_id,
                normalized_by_id,
                ids,
                gradient_result.normalized_designs,
                gradient_result.accepted_forward,
            )
            if not gradient_result.accepted_forward.empty
            else 0
        )
        gradient_result.validation_ledger.to_csv(
            output_dir / "gradient" / "validation_ledger.csv", index=False
        )
        write_json(
            output_dir / "gradient" / "diagnostics.json",
            {
                **gradient_result.diagnostics,
                "forced_full_budget": True,
                "accepted_actual_improvements": accepted,
            },
        )
        timings["design_gradient"] = time.perf_counter() - action_started
        actions.append("design_gradient")
    elif gradient_enabled:
        actions.append("design_gradient_skipped_no_forward_valid_candidate")

    minimum_hits = int(config["delivery"]["minimum_hit_count"])
    nsga_pareto = pd.DataFrame()
    if nsga2_enabled and _hit_count(forward_by_id) < minimum_hits:
        action_started = time.perf_counter()
        ids = _top_candidate_ids(forward_by_id, int(config["nsga2"]["population_size"]))
        if ids:
            candidates = pd.DataFrame(
                [
                    {
                        "candidate_id": value,
                        **{
                            name: float(
                                normalizer.inverse_design(normalized_by_id[value][None, :])[
                                    0, index
                                ]
                            )
                            for index, name in enumerate(normalizer.design_names)
                        },
                    }
                    for value in ids
                ]
            )
            seed, optimizer_config = _nsga2_config(config["nsga2"], target, derived)
            warm = build_warm_start_population(
                candidates,
                normalizer.design_names,
                mode="generator",
                population_size=optimizer_config.population_size,
                background_seed=optimizer_config.warm_start_seed,
                background_method=optimizer_config.warm_start_fill_method,
                duplicate_tolerance=optimizer_config.warm_start_duplicate_tolerance,
                candidate_source="diag_cfm_full_strategy",
            )
            np.random.seed(seed)
            pareto, history = optimize(
                config=optimizer_config,
                bounds=DesignSpaceBounds(),
                verbose=False,
                initial_genes=warm.population,
            )
            nsga_pareto = _pareto_frame(pareto, history["param_names"], derived)
            nsga_pareto.to_csv(output_dir / "nsga2" / "pareto.csv", index=False)
            pd.DataFrame(history["evaluation_ledger"]).to_csv(
                output_dir / "nsga2" / "evaluation_ledger.csv", index=False
            )
            costs["nsga2_0d"] = len(history["evaluation_ledger"])
            actions.append("nsga2")
        else:
            actions.append("nsga2_skipped_no_forward_valid_candidate")
        timings["nsga2"] = time.perf_counter() - action_started

    if not nsga_pareto.empty:
        hits = nsga_pareto.loc[nsga_pareto["within_tolerance"].astype(bool)]
        delivered = (hits if len(hits) else nsga_pareto).sort_values(
            "total_error"
        ).head(int(config["delivery"]["maximum_delivered_candidates"])).copy()
        delivery_source = "nsga2"
    else:
        final_valid = pd.DataFrame(list(forward_by_id.values()))
        if not final_valid.empty:
            final_valid = final_valid.loc[
                final_valid["forward_valid"].astype(bool)
            ].sort_values(["total_error", "candidate_id"])
        hits = (
            final_valid.loc[final_valid["within_tolerance"].astype(bool)]
            if len(final_valid)
            else final_valid
        )
        delivered = (hits if len(hits) else final_valid).head(
            int(config["delivery"]["maximum_delivered_candidates"])
        ).copy()
        delivery_source = (
            "design_gradient" if "design_gradient" in actions else "direct"
        )

    final_pool = pd.DataFrame(list(forward_by_id.values()))
    if not final_pool.empty:
        final_pool = final_pool.sort_values(["total_error", "candidate_id"])
    final_pool.to_csv(output_dir / "delivery" / "final_candidate_pool.csv", index=False)
    delivered["delivery_source"] = delivery_source
    delivered.to_csv(output_dir / "delivery" / "delivered_candidates.csv", index=False)
    pd.DataFrame(
        [
            {"stage": key, "zero_d_evaluation_count": int(value)}
            for key, value in costs.items()
        ]
    ).to_csv(output_dir / "delivery" / "evaluation_cost_ledger.csv", index=False)
    nsga_hit = bool(
        not nsga_pareto.empty
        and nsga_pareto["within_tolerance"].astype(bool).any()
    )
    stop_reason = (
        "nsga2_completed_with_target_hit"
        if nsga_hit
        else "nsga2_budget_exhausted"
        if "nsga2" in actions
        else "design_gradient_minimum_hit_count_reached"
        if "design_gradient" in actions and _hit_count(forward_by_id) >= minimum_hits
        else "direct_minimum_hit_count_reached"
        if _hit_count(forward_by_id) >= minimum_hits
        else "configured_refinement_exhausted"
    )
    decision = {
        "mode": "diag_cfm_full_strategy",
        "planned_actions": [
            *(["design_gradient"] if gradient_enabled else []),
            *(["nsga2_if_needed"] if nsga2_enabled else []),
        ],
        "executed_actions": actions,
        "minimum_hit_count": minimum_hits,
        "direct_hit_count": int(
            direct.forward_results.get(
                "within_tolerance", pd.Series(dtype=bool)
            ).astype(bool).sum()
        ),
        "final_non_nsga_hit_count": _hit_count(forward_by_id),
        "delivered_count": len(delivered),
        "delivery_source": delivery_source,
        "stop_reason": stop_reason,
        "postprocessing": {
            "surrogate_gradient_enabled": gradient_enabled,
            "conditional_nsga2_enabled": nsga2_enabled,
        },
        "cost": {**costs, "total_0d": int(sum(costs.values()))},
        "timing_seconds": {**timings, "total": float(sum(timings.values()))},
    }
    write_json(output_dir / "delivery" / "decision.json", decision)
    write_json(output_dir / "environment.json", collect_environment())
    write_json(output_dir / "git_state.json", collect_git_state(repo_root, output_dir))
    write_json(output_dir / "source_manifest.json", collect_source_manifest(repo_root))
    write_yaml(
        output_dir / "manifest.yaml",
        {
            "version": config["version"],
            "status": "completed",
            "started_utc": started.isoformat(),
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "target": dataclasses.asdict(target),
            "original_target_type": original_target_type,
            "candidate_protocol": "v2_1_fixed_raw",
            "generator_checkpoint": str(checkpoint_path),
            "generator_checkpoint_sha256": sha256_file(checkpoint_path),
            "diag_prefilter": {
                "pool_size": pool_size,
                "reranker_checkpoint": str(reranker_path),
                "reranker_checkpoint_sha256": sha256_file(reranker_path),
            },
            "surrogate_checkpoint": (
                str(surrogate_path) if surrogate_path is not None else None
            ),
            "surrogate_checkpoint_sha256": (
                sha256_file(surrogate_path) if surrogate_path is not None else None
            ),
            "decision": decision,
        },
    )
    return output_dir
