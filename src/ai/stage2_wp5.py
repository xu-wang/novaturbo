"""Stage-II WP5 direct delivery and budgeted optional refinement."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

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
from .generator_dataset import (
    GeneratorNormalizer,
    condition_names,
    design_names,
    load_schema,
)
from .generators import load_generator
from .inverse_design import TargetSpec
from .optimizer import (
    TARGET_WEIGHTED_RELATIVE_ERROR,
    OptimizationConfig,
    optimize,
)
from .round_trip import (
    RoundTripEvaluator,
    derive_round_trip_metrics,
    load_metrics_config,
    sampling_budget_exhausted_result,
)
from .stage2_wp3 import (
    ForwardConsistencySurrogate,
    consistency_scores,
    load_forward_surrogate,
    repair_latents,
)
from .warm_start import build_warm_start_population


DELIVERY_MODES = (
    "generator_only",
    "generator_screened",
    "generator_latent_repair",
    "generator_gradient_refinement",
    "generator_nsga2_refinement",
    "generator_auto_refinement",
)


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for S2-WP5 but is unavailable")
    return device


def load_stage2_wp5_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "version",
        "data",
        "target",
        "model",
        "generation",
        "delivery",
        "gradient",
        "latent_repair",
        "nsga2",
        "runtime",
        "output",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"S2-WP5 config is missing keys: {sorted(missing)}")
    mode = str(config["delivery"]["mode"])
    if mode not in DELIVERY_MODES:
        raise ValueError(f"delivery.mode must be one of {list(DELIVERY_MODES)}")
    generation = config["generation"]
    if str(generation.get("candidate_protocol")) != "v2_1_fixed_raw":
        raise ValueError("S2-WP5 requires candidate_protocol=v2_1_fixed_raw")
    if int(generation.get("n_raw", 0)) != V21_RAW_CANDIDATE_COUNT:
        raise ValueError("S2-WP5 requires generation.n_raw=128")
    delivery = config["delivery"]
    if int(delivery.get("minimum_hit_count", 0)) <= 0:
        raise ValueError("delivery.minimum_hit_count must be positive")
    if int(delivery.get("maximum_delivered_candidates", 0)) <= 0:
        raise ValueError("delivery.maximum_delivered_candidates must be positive")
    if bool(delivery.get("extra_engineering_objectives", False)) and mode not in {
        "generator_nsga2_refinement",
        "generator_auto_refinement",
    }:
        raise ValueError(
            "extra engineering objectives require an NSGA-II-capable delivery mode"
        )
    gradient = config["gradient"]
    for name in ("candidate_count", "steps", "validation_interval", "maximum_0d_evaluations"):
        if int(gradient.get(name, 0)) <= 0:
            raise ValueError(f"gradient.{name} must be positive")
    if float(gradient.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("gradient.learning_rate must be positive")
    trust = float(gradient.get("trust_region_linf", 0.0))
    if not 0.0 < trust <= 2.0:
        raise ValueError("gradient.trust_region_linf must be in (0,2]")
    latent = config["latent_repair"]
    if int(latent.get("candidate_count", 0)) <= 0:
        raise ValueError("latent_repair.candidate_count must be positive")
    if int(latent.get("maximum_0d_evaluations", 0)) < int(latent["candidate_count"]):
        raise ValueError("latent_repair maximum_0d_evaluations is below candidate_count")
    nsga2 = config["nsga2"]
    if mode == "generator_nsga2_refinement" and not bool(nsga2.get("enabled", False)):
        raise ValueError("generator_nsga2_refinement requires nsga2.enabled=true")
    if bool(nsga2.get("enabled", False)):
        if int(nsga2.get("population_size", 0)) <= 0 or int(nsga2.get("n_generations", -1)) < 0:
            raise ValueError("NSGA-II population_size must be positive and n_generations non-negative")
        objectives = list(nsga2.get("objectives", []))
        maximize = list(nsga2.get("maximize", []))
        if not objectives or len(objectives) != len(maximize):
            raise ValueError("NSGA-II objectives and maximize must be non-empty and aligned")
        if TARGET_WEIGHTED_RELATIVE_ERROR not in objectives:
            raise ValueError("S2-WP5 NSGA-II must retain target_weighted_relative_error")
    return config


def refinement_plan(mode: str, *, supports_latent: bool) -> tuple[str, ...]:
    """Return the ordered post-screening actions for a delivery mode."""

    if mode not in DELIVERY_MODES:
        raise ValueError(f"Unsupported S2-WP5 mode: {mode}")
    if mode in {"generator_only", "generator_screened"}:
        return ()
    if mode == "generator_latent_repair":
        if not supports_latent:
            raise ValueError("generator_latent_repair requires a repairable latent space")
        return ("latent_repair",)
    if mode == "generator_gradient_refinement":
        return ("design_gradient",)
    if mode == "generator_nsga2_refinement":
        return ("nsga2",)
    return (
        ("latent_repair", "design_gradient", "nsga2")
        if supports_latent
        else ("design_gradient", "nsga2")
    )


@dataclass(frozen=True)
class GradientRefinementResult:
    candidate_ids: tuple[int, ...]
    normalized_designs: np.ndarray
    best_actual_errors: np.ndarray
    improved_mask: np.ndarray
    accepted_forward: pd.DataFrame
    validation_ledger: pd.DataFrame
    diagnostics: Dict[str, Any]


ValidationCallback = Callable[[np.ndarray, Sequence[int], int], tuple[pd.DataFrame, int]]


def surrogate_gradient_refinement(
    initial_normalized: np.ndarray,
    candidate_ids: Sequence[int],
    baseline_actual_errors: Sequence[float],
    normalized_target: np.ndarray,
    surrogate: ForwardConsistencySurrogate,
    config: Mapping[str, Any],
    validation_callback: ValidationCallback,
    *,
    weights: Optional[Sequence[float]] = None,
) -> GradientRefinementResult:
    """Refine active-10D candidates with surrogate gradients and true-0D acceptance."""

    initial_np = np.asarray(initial_normalized, dtype=np.float32)
    ids = np.asarray(candidate_ids, dtype=int)
    baseline = np.asarray(baseline_actual_errors, dtype=float)
    if initial_np.ndim != 2 or len(initial_np) != len(ids) or len(ids) != len(baseline):
        raise ValueError("gradient inputs must have aligned candidate rows")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("gradient candidate IDs must be unique")
    if not np.isfinite(initial_np).all() or not np.isfinite(baseline).all():
        raise ValueError("gradient inputs must be finite")
    if not len(ids):
        raise ValueError("gradient refinement requires at least one candidate")

    device = next(surrogate.parameters()).device
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)
    surrogate.eval()
    initial = torch.as_tensor(initial_np, dtype=torch.float32, device=device)
    design = initial.clone().requires_grad_(True)
    target = torch.as_tensor(normalized_target, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([design], lr=float(config["learning_rate"]))
    trust = float(config["trust_region_linf"])
    prior_weight = float(config.get("prior_weight", 0.01))
    gradient_clip = float(config.get("gradient_clip_norm", 1.0))
    interval = int(config["validation_interval"])
    steps = int(config["steps"])
    maximum_evaluations = int(config["maximum_0d_evaluations"])
    minimum_hits = int(config.get("minimum_hit_count", 1))
    existing_hits = int(config.get("existing_hit_count", 0))

    best_design = initial_np.astype(float, copy=True)
    best_errors = baseline.copy()
    improved = np.zeros(len(ids), dtype=bool)
    best_rows: Dict[int, Dict[str, Any]] = {}
    ledger_frames = []
    history = []
    actual_evaluations = 0
    termination_reason = "steps_completed"
    last_validated_step = 0

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = surrogate(design)
        consistency = consistency_scores(prediction, target, weights)
        displacement = (design - initial).square().mean(dim=1)
        objective = consistency + prior_weight * displacement
        loss = objective.mean()
        if not torch.isfinite(loss):
            termination_reason = f"nonfinite_objective_at_step_{step}"
            break
        loss.backward()
        if design.grad is None or not torch.isfinite(design.grad).all():
            termination_reason = f"nonfinite_gradient_at_step_{step}"
            break
        if gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_([design], gradient_clip)
        optimizer.step()
        with torch.no_grad():
            lower = torch.maximum(torch.full_like(initial, -1.0), initial - trust)
            upper = torch.minimum(torch.full_like(initial, 1.0), initial + trust)
            design.copy_(torch.maximum(torch.minimum(design, upper), lower))

        should_validate = step % interval == 0 or step == steps
        if not should_validate:
            continue
        if actual_evaluations + len(ids) > maximum_evaluations:
            termination_reason = "maximum_0d_evaluations_reached"
            break
        current_np = design.detach().cpu().numpy().astype(float)
        frame, evaluated_count = validation_callback(current_np, ids.tolist(), step)
        actual_evaluations += int(evaluated_count)
        last_validated_step = step
        required = {"candidate_id", "forward_valid", "total_error", "within_tolerance"}
        if required.difference(frame.columns):
            raise ValueError("gradient validation callback returned an incomplete frame")
        annotated = frame.copy()
        annotated.insert(0, "gradient_step", step)
        annotated["surrogate_consistency"] = (
            consistency.detach().cpu().numpy().astype(float)
        )
        ledger_frames.append(annotated)
        for row_index, candidate_id in enumerate(ids):
            match = frame.loc[frame["candidate_id"].astype(int) == int(candidate_id)]
            if match.empty or not bool(match.iloc[0]["forward_valid"]):
                continue
            error = float(match.iloc[0]["total_error"])
            if math.isfinite(error) and error < best_errors[row_index]:
                best_errors[row_index] = error
                best_design[row_index] = current_np[row_index]
                improved[row_index] = True
                best_rows[int(candidate_id)] = match.iloc[0].to_dict()
        hit_count = sum(
            bool(row.get("within_tolerance", False)) for row in best_rows.values()
        )
        history.append(
            {
                "step": step,
                "surrogate_objective_mean": float(loss.detach()),
                "best_actual_error": float(np.min(best_errors)),
                "improved_count": int(improved.sum()),
                "accepted_hit_count": int(hit_count),
                "cumulative_0d_evaluations": actual_evaluations,
            }
        )
        if existing_hits + hit_count >= minimum_hits:
            termination_reason = "minimum_hit_count_reached"
            break

    accepted = pd.DataFrame(list(best_rows.values()))
    ledger = pd.concat(ledger_frames, ignore_index=True) if ledger_frames else pd.DataFrame()
    diagnostics = {
        "candidate_count": len(ids),
        "steps_configured": steps,
        "last_validated_step": last_validated_step,
        "actual_0d_evaluations": actual_evaluations,
        "maximum_0d_evaluations": maximum_evaluations,
        "improved_count": int(improved.sum()),
        "improved_fraction": float(improved.mean()),
        "best_error_before": float(np.min(baseline)),
        "best_error_after": float(np.min(best_errors)),
        "termination_reason": termination_reason,
        "history": history,
    }
    return GradientRefinementResult(
        candidate_ids=tuple(int(value) for value in ids),
        normalized_designs=best_design,
        best_actual_errors=best_errors,
        improved_mask=improved,
        accepted_forward=accepted,
        validation_ledger=ledger,
        diagnostics=diagnostics,
    )


def _load_target(repo_root: Path, config: Mapping[str, Any]) -> TargetSpec:
    if config.get("spec_yaml"):
        path = _resolve(repo_root, str(config["spec_yaml"]))
        return TargetSpec.from_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))
    suite_path = _resolve(repo_root, str(config["suite_csv"]))
    target_id = str(config["target_id"])
    suite = pd.read_csv(suite_path)
    selected = suite.loc[suite["target_id"].astype(str) == target_id]
    if len(selected) != 1:
        raise ValueError(f"target_id={target_id} must identify exactly one target")
    row = selected.iloc[0]
    return TargetSpec.from_mapping(
        {
            "target_version": str(row.get("target_suite_version", "stage2_target_v1")),
            "target_id": target_id,
            "target_type": str(row["target_type"]),
            "expected_status": str(row["expected_status"]),
            "thrust_N": float(row["thrust_N"]),
            "tsfc_kg_N_s": float(row["tsfc_kg_N_s"]),
            "total_mass_kg": float(row["total_mass_kg"]),
            "tolerance_profile": str(config.get("tolerance_profile", "round_trip_metrics_v1")),
        }
    )


def _top_candidate_ids(
    forward_by_id: Mapping[int, Mapping[str, Any]], count: int
) -> list[int]:
    valid = [
        value for value in forward_by_id.values()
        if bool(value.get("forward_valid", False)) and math.isfinite(float(value.get("total_error", math.inf)))
    ]
    valid.sort(key=lambda row: (float(row["total_error"]), int(row["candidate_id"])))
    return [int(row["candidate_id"]) for row in valid[:count]]


def _hit_count(forward_by_id: Mapping[int, Mapping[str, Any]]) -> int:
    return sum(bool(value.get("within_tolerance", False)) for value in forward_by_id.values())


def _accept_updates(
    forward_by_id: Dict[int, Dict[str, Any]],
    normalized_by_id: Dict[int, np.ndarray],
    candidate_ids: Sequence[int],
    normalized: np.ndarray,
    frame: pd.DataFrame,
) -> int:
    accepted = 0
    indexed = frame.assign(candidate_id=frame["candidate_id"].astype(int)).set_index("candidate_id")
    for row_index, candidate_id in enumerate(candidate_ids):
        candidate_id = int(candidate_id)
        if candidate_id not in indexed.index:
            continue
        row = indexed.loc[candidate_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if not bool(row["forward_valid"]):
            continue
        prior = float(forward_by_id[candidate_id].get("total_error", math.inf))
        current = float(row["total_error"])
        if math.isfinite(current) and current < prior:
            updated = row.to_dict()
            updated["candidate_id"] = candidate_id
            forward_by_id[candidate_id] = updated
            normalized_by_id[candidate_id] = np.asarray(normalized[row_index], dtype=float)
            accepted += 1
    return accepted


def _nsga2_config(
    raw: Mapping[str, Any], target: TargetSpec, derived: Any
) -> tuple[int, OptimizationConfig]:
    values = dict(raw)
    values.pop("enabled", None)
    seed = int(values.pop("seed"))
    target_values = {
        "thrust_N": target.thrust_N,
        "tsfc_kg_N_s": target.tsfc_kg_N_s,
    }
    target_weights = {
        "thrust_N": derived.thrust_weight,
        "tsfc_kg_N_s": derived.tsfc_weight,
    }
    target_epsilons = {
        "thrust_N": derived.thrust_epsilon,
        "tsfc_kg_N_s": derived.tsfc_epsilon,
    }
    if target.total_mass_kg is not None:
        target_values["total_mass_kg"] = target.total_mass_kg
        target_weights["total_mass_kg"] = derived.mass_weight
        target_epsilons["total_mass_kg"] = derived.mass_epsilon
    values.update(
        {
            "target_values": target_values,
            "target_weights": target_weights,
            "target_epsilons": target_epsilons,
        }
    )
    return seed, OptimizationConfig(**values)


def _pareto_frame(pareto: Sequence[Any], param_names: Sequence[str], derived: Any) -> pd.DataFrame:
    rows = []
    for index, individual in enumerate(pareto):
        row: Dict[str, Any] = {
            "pareto_index": index,
            "individual_id": individual.individual_id,
            "is_feasible": bool(individual.is_feasible),
            "provenance": json.dumps(individual.provenance, sort_keys=True),
        }
        row.update({name: float(individual.genes[i]) for i, name in enumerate(param_names)})
        row.update({f"performance__{name}": value for name, value in individual.performance.items()})
        error_f = float(individual.performance.get("target_relative_error__thrust_N", math.inf))
        error_t = float(individual.performance.get("target_relative_error__tsfc_kg_N_s", math.inf))
        error_m = float(individual.performance.get("target_relative_error__total_mass_kg", 0.0))
        total = float(individual.performance.get(TARGET_WEIGHTED_RELATIVE_ERROR, math.inf))
        row["total_error"] = total
        row["within_tolerance"] = bool(
            individual.is_feasible
            and error_f <= derived.thrust_relative_tolerance
            and error_t <= derived.tsfc_relative_tolerance
            and error_m <= derived.mass_relative_tolerance
            and total <= derived.total_error_tolerance
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_stage2_wp5(
    config_path: Path,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
) -> Path:
    """Run one auditable direct-delivery/refinement decision."""

    repo_root = (repo_root or Path.cwd()).resolve()
    config_path = Path(config_path).resolve()
    config = load_stage2_wp5_config(config_path)
    output_dir = (
        output_dir_override.resolve()
        if output_dir_override is not None
        else _resolve(repo_root, str(config["output"]["directory"]))
    )
    if output_dir.exists() and bool(config["output"].get("refuse_overwrite", True)):
        raise FileExistsError(f"S2-WP5 output exists: {output_dir}")
    data = config["data"]
    dataset_path = _resolve(repo_root, str(data["dataset_csv"]))
    schema_path = _resolve(repo_root, str(data["schema"]))
    normalizer_path = _resolve(repo_root, str(data["normalizer"]))
    metrics_path = _resolve(repo_root, str(data["metrics_config"]))
    checkpoint_path = _resolve(repo_root, str(config["model"]["checkpoint_path"]))
    for path in (dataset_path, schema_path, normalizer_path, metrics_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"S2-WP5 input is missing: {path}")
    target = _load_target(repo_root, config["target"])
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("generation", "direct", "latent", "gradient", "nsga2", "delivery"):
        (output_dir / name).mkdir()
    shutil.copy2(config_path, output_dir / "build_config.yaml")
    started = dt.datetime.now(dt.timezone.utc)
    device = _resolve_device(str(config["runtime"].get("device", "auto")))
    schema = load_schema(schema_path)
    normalizer = GeneratorNormalizer.load(normalizer_path)
    if design_names(schema) != normalizer.design_names or condition_names(schema) != normalizer.condition_names:
        raise ValueError("S2-WP5 schema and normalizer differ")
    metrics_config = load_metrics_config(metrics_path)
    derived = derive_round_trip_metrics(dataset_path, normalizer, metrics_config)
    evaluator = RoundTripEvaluator(schema, normalizer, metrics_config, derived)
    generator = load_generator(checkpoint_path, device=str(device))
    if generator.normalizer is None or generator.normalizer.to_dict() != normalizer.to_dict():
        raise ValueError("S2-WP5 generator normalizer differs")
    mode = str(config["delivery"]["mode"])
    plan = refinement_plan(mode, supports_latent=bool(generator.supports_latent_repair))
    minimum_hits = int(config["delivery"]["minimum_hit_count"])
    maximum_delivered = int(config["delivery"]["maximum_delivered_candidates"])
    extra_objectives = bool(config["delivery"].get("extra_engineering_objectives", False))

    normalized_target_np = evaluator.normalize_target(target.to_dict())
    condition = torch.as_tensor(normalized_target_np, dtype=torch.float32, device=device).unsqueeze(0)
    n_raw = int(config["generation"]["n_raw"])
    generation_seed = int(config["generation"]["seed"])
    latent = None
    generation_started = time.perf_counter()
    if "latent_repair" in plan and generator.supports_latent_repair:
        latent = generator.sample_latent(condition, n_raw, seed=generation_seed)
        with torch.no_grad():
            raw_normalized = generator.decode_latent(latent, condition)[0].cpu().numpy()
        generation_diagnostics = {"sampling_path": "latent_decode"}
    else:
        sampled = generator.sample(normalized_target_np, n_raw, seed=generation_seed, space="normalized")
        raw_normalized = sampled.values.cpu().numpy()
        generation_diagnostics = dict(sampled.diagnostics)
    generation_seconds = time.perf_counter() - generation_started
    raw_physical = normalizer.inverse_design(raw_normalized)
    raw_ids = np.arange(n_raw, dtype=int)
    validation = evaluator.validate_candidates(raw_normalized, raw_physical, candidate_ids=raw_ids)
    protocol = apply_candidate_protocol_v21(
        validation["candidate_id"].to_numpy(int),
        validation["candidate_valid"].to_numpy(bool),
    )
    validation["kept_by_v2_1"] = validation["candidate_id"].isin(protocol.kept_candidate_ids)
    kept_positions = np.asarray(protocol.kept_positions, dtype=int)
    kept_ids = np.asarray(protocol.kept_candidate_ids, dtype=int)
    kept_normalized = raw_normalized[kept_positions]
    kept_physical = raw_physical[kept_positions]
    pd.DataFrame(raw_normalized, columns=[f"normalized_{name}" for name in normalizer.design_names]).assign(candidate_id=raw_ids).to_csv(
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
            "diagnostics": generation_diagnostics,
        },
    )

    if len(kept_ids):
        direct = evaluator.evaluate(target.to_dict(), kept_normalized, kept_physical, candidate_ids=kept_ids)
    else:
        direct = sampling_budget_exhausted_result(
            target.to_dict(), str(config["model"]["model_id"]),
            is_training_domain=evaluator.target_in_training_domain(target.to_dict()),
        )
    direct.forward_results.to_csv(output_dir / "direct" / "forward_results.csv", index=False)
    direct.ranked_candidates.to_csv(output_dir / "direct" / "ranked_candidates.csv", index=False)
    write_json(output_dir / "direct" / "metrics.json", direct.metrics)
    initial_0d = int(direct.metrics.get("candidate_valid_count", 0))
    forward_by_id = {
        int(row["candidate_id"]): dict(row)
        for row in direct.forward_results.to_dict(orient="records")
        if int(row["candidate_id"]) >= 0
    }
    normalized_by_id = {
        int(candidate_id): np.asarray(kept_normalized[index], dtype=float)
        for index, candidate_id in enumerate(kept_ids)
    }
    costs = {"direct_0d": initial_0d, "latent_repair_0d": 0, "gradient_0d": 0, "nsga2_0d": 0}
    actions = []
    stop_reason: Optional[str] = None

    if _hit_count(forward_by_id) >= minimum_hits and not extra_objectives:
        stop_reason = "direct_minimum_hit_count_reached"
    else:
        effective_plan = ("nsga2",) if extra_objectives and "nsga2" in plan else plan
        surrogate = None
        needs_surrogate = any(action in {"latent_repair", "design_gradient"} for action in effective_plan)
        if needs_surrogate:
            surrogate_path = _resolve(repo_root, str(config["gradient"]["surrogate_checkpoint"]))
            surrogate, _ = load_forward_surrogate(surrogate_path, normalizer, device)
        nsga_pareto = pd.DataFrame()
        nsga_history: Dict[str, Any] = {}
        for action in effective_plan:
            if _hit_count(forward_by_id) >= minimum_hits and not extra_objectives:
                stop_reason = f"{actions[-1]}_minimum_hit_count_reached"
                break
            if action == "latent_repair":
                assert latent is not None and surrogate is not None
                ids = _top_candidate_ids(forward_by_id, int(config["latent_repair"]["candidate_count"]))
                if not ids:
                    actions.append("latent_repair_skipped_no_forward_valid_candidate")
                    continue
                repaired_latent, repaired_design, diagnostics = repair_latents(
                    generator,
                    surrogate,
                    latent[:, ids, :],
                    condition,
                    config["latent_repair"],
                    config["gradient"].get("condition_weights"),
                )
                del repaired_latent
                repaired_np = repaired_design[0].detach().cpu().numpy()
                repaired_physical = normalizer.inverse_design(repaired_np)
                repaired_result = evaluator.evaluate(target.to_dict(), repaired_np, repaired_physical, candidate_ids=ids)
                costs["latent_repair_0d"] += int(repaired_result.metrics["candidate_valid_count"])
                accepted = _accept_updates(forward_by_id, normalized_by_id, ids, repaired_np, repaired_result.forward_results)
                repaired_result.forward_results.assign(refinement_action="latent_repair").to_csv(
                    output_dir / "latent" / "validation.csv", index=False
                )
                write_json(output_dir / "latent" / "diagnostics.json", {**diagnostics, "accepted_actual_improvements": accepted})
                actions.append("latent_repair")
            elif action == "design_gradient":
                assert surrogate is not None
                ids = _top_candidate_ids(forward_by_id, int(config["gradient"]["candidate_count"]))
                if not ids:
                    actions.append("design_gradient_skipped_no_forward_valid_candidate")
                    continue
                initial_designs = np.asarray([normalized_by_id[value] for value in ids])
                baseline_errors = [float(forward_by_id[value]["total_error"]) for value in ids]

                def validate(values: np.ndarray, callback_ids: Sequence[int], step: int):
                    physical = normalizer.inverse_design(values)
                    result = evaluator.evaluate(target.to_dict(), values, physical, candidate_ids=callback_ids)
                    return result.forward_results, int(result.metrics["candidate_valid_count"])

                gradient_config = dict(config["gradient"])
                gradient_config["minimum_hit_count"] = minimum_hits
                gradient_config["existing_hit_count"] = _hit_count(forward_by_id)
                gradient_result = surrogate_gradient_refinement(
                    initial_designs,
                    ids,
                    baseline_errors,
                    normalized_target_np,
                    surrogate,
                    gradient_config,
                    validate,
                    weights=config["gradient"].get("condition_weights"),
                )
                costs["gradient_0d"] += int(gradient_result.diagnostics["actual_0d_evaluations"])
                accepted = _accept_updates(
                    forward_by_id,
                    normalized_by_id,
                    ids,
                    gradient_result.normalized_designs,
                    gradient_result.accepted_forward,
                ) if not gradient_result.accepted_forward.empty else 0
                gradient_result.validation_ledger.to_csv(output_dir / "gradient" / "validation_ledger.csv", index=False)
                write_json(output_dir / "gradient" / "diagnostics.json", {**gradient_result.diagnostics, "accepted_actual_improvements": accepted})
                actions.append("design_gradient")
            elif action == "nsga2":
                if not bool(config["nsga2"].get("enabled", False)):
                    actions.append("nsga2_skipped_disabled")
                    continue
                ids = _top_candidate_ids(forward_by_id, int(config["nsga2"]["population_size"]))
                if not ids:
                    actions.append("nsga2_skipped_no_forward_valid_candidate")
                    continue
                candidates = pd.DataFrame(
                    [
                        {"candidate_id": value, **{
                            name: float(normalizer.inverse_design(normalized_by_id[value][None, :])[0, index])
                            for index, name in enumerate(normalizer.design_names)
                        }}
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
                    candidate_source="wp5_refined_generator",
                )
                np.random.seed(seed)
                pareto, nsga_history = optimize(
                    config=optimizer_config,
                    bounds=DesignSpaceBounds(),
                    verbose=False,
                    initial_genes=warm.population,
                )
                param_names = list(nsga_history["param_names"])
                nsga_pareto = _pareto_frame(pareto, param_names, derived)
                nsga_pareto.to_csv(output_dir / "nsga2" / "pareto.csv", index=False)
                pd.DataFrame(nsga_history["evaluation_ledger"]).to_csv(
                    output_dir / "nsga2" / "evaluation_ledger.csv", index=False
                )
                costs["nsga2_0d"] = len(nsga_history["evaluation_ledger"])
                actions.append("nsga2")
                stop_reason = (
                    "nsga2_completed_with_target_hit"
                    if not nsga_pareto.empty and bool(nsga_pareto["within_tolerance"].any())
                    else "nsga2_budget_exhausted"
                )
                break
        if stop_reason is None:
            stop_reason = (
                "refinement_minimum_hit_count_reached"
                if _hit_count(forward_by_id) >= minimum_hits
                else "configured_refinement_exhausted"
            )

    if "nsga_pareto" in locals() and not nsga_pareto.empty:
        hits = nsga_pareto.loc[nsga_pareto["within_tolerance"].astype(bool)]
        delivered = (hits if len(hits) else nsga_pareto).sort_values("total_error").head(maximum_delivered).copy()
        delivery_source = "nsga2"
    else:
        pool = pd.DataFrame(list(forward_by_id.values()))
        if not pool.empty:
            pool = pool.loc[pool["forward_valid"].astype(bool)].sort_values(["total_error", "candidate_id"])
        hits = pool.loc[pool["within_tolerance"].astype(bool)] if len(pool) else pool
        delivered = (hits if len(hits) else pool).head(maximum_delivered).copy()
        delivery_source = actions[-1] if actions else "direct"
    delivered["delivery_source"] = delivery_source
    delivered.to_csv(output_dir / "delivery" / "delivered_candidates.csv", index=False)
    cost_ledger = pd.DataFrame(
        [{"stage": key, "zero_d_evaluation_count": int(value)} for key, value in costs.items()]
    )
    cost_ledger.to_csv(output_dir / "delivery" / "evaluation_cost_ledger.csv", index=False)
    decision = {
        "mode": mode,
        "planned_actions": list(plan),
        "executed_actions": actions,
        "minimum_hit_count": minimum_hits,
        "direct_hit_count": int(direct.forward_results.get("within_tolerance", pd.Series(dtype=bool)).astype(bool).sum()),
        "final_non_nsga_hit_count": _hit_count(forward_by_id),
        "delivered_count": len(delivered),
        "delivery_source": delivery_source,
        "stop_reason": stop_reason,
        "cost": {**costs, "total_0d": int(sum(costs.values()))},
    }
    write_json(output_dir / "delivery" / "decision.json", decision)
    write_json(output_dir / "environment.json", collect_environment())
    write_json(output_dir / "git_state.json", collect_git_state(repo_root, output_dir))
    write_json(output_dir / "source_manifest.json", collect_source_manifest(repo_root))
    surrogate_value = str(config["gradient"].get("surrogate_checkpoint", ""))
    surrogate_path = _resolve(repo_root, surrogate_value) if surrogate_value else None
    manifest = {
        "version": config["version"],
        "status": "completed",
        "started_utc": started.isoformat(),
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target": dataclasses.asdict(target),
        "mode": mode,
        "candidate_protocol": "v2_1_fixed_raw",
        "generator_checkpoint": str(checkpoint_path),
        "generator_checkpoint_sha256": sha256_file(checkpoint_path),
        "surrogate_checkpoint": surrogate_value,
        "surrogate_checkpoint_sha256": (
            sha256_file(surrogate_path)
            if surrogate_path is not None and surrogate_path.is_file()
            else None
        ),
        "decision": decision,
    }
    write_yaml(output_dir / "manifest.yaml", manifest)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = run_stage2_wp5(args.config, output_dir_override=args.output_dir)
    print(json.dumps({"output": str(output), "status": "completed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
