"""Evaluation-only full-flow comparison of shared48 S0 and performance3_log.

This module deliberately does not change the production pipeline contract.  The
performance3 arm is adapted locally because its three log outputs cannot supply
the 45 surrogate constraint residuals expected by the legacy S0 objective.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import yaml

import src.ai.geometry19_candidates as candidate_module
import src.ai.geometry19_refinement as refinement_module
from src.ai.generators import CFMGenerator
from src.ai.geometry19_candidates import run_candidate_protocol
from src.ai.geometry19_cfm import integrate_cfm_sources
from src.ai.geometry19_delivery import run_delivery_stage
from src.ai.geometry19_hit_alignment import ordered_tensor, torch_hit_aligned_terms
from src.ai.geometry19_inverse_contract import OUTPUT_NAMES, Geometry19InverseTarget, load_wp19_23_contract
from src.ai.geometry19_nsga2 import run_nsga2_stage
from src.ai.geometry19_physics import run_true_physics_stage
from src.ai.geometry19_refinement import Geometry19RefinementError, RefinementModels, run_refinement_stage
from src.ai.geometry19_surrogate import load_weight_bundle, sha256_file


PROTOCOL_VERSION = "wp19_23_performance3_full_pipeline_comparison_v1"


class Geometry19Performance3ComparisonError(ValueError):
    """Raised when the locked comparison contract is inconsistent."""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True), encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _decode_log_outputs(normalized: torch.Tensor, stats: Mapping[str, np.ndarray]) -> torch.Tensor:
    mean = torch.as_tensor(stats["output_mean"], dtype=normalized.dtype, device=normalized.device)
    std = torch.as_tensor(stats["output_std"], dtype=normalized.dtype, device=normalized.device)
    if normalized.shape[-1] != 3 or mean.numel() != 3 or std.numel() != 3:
        raise Geometry19Performance3ComparisonError("performance3_log must have exactly three outputs")
    return torch.exp(normalized * std + mean)


def _performance3_dflow(
    generator: CFMGenerator,
    initial_sources: torch.Tensor,
    direct_design: torch.Tensor,
    normalized_condition: torch.Tensor,
    target: Geometry19InverseTarget,
    protocol: Mapping[str, Any],
    forward_model: nn.Module,
    forward_normalization: Mapping[str, np.ndarray],
    feasibility_model: nn.Module,
    feasibility_normalization: Mapping[str, np.ndarray],
    target_loss_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    del feasibility_normalization
    if target_loss_mode != "hit_aligned":
        raise Geometry19Performance3ComparisonError("performance3 comparison requires hit-aligned loss")
    config = protocol["dflow"]
    generator.eval().requires_grad_(False)
    forward_model.eval().requires_grad_(False)
    if feasibility_model is not None:
        feasibility_model.eval().requires_grad_(False)
    initial = initial_sources.detach().clone()
    sources = initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([sources], lr=float(config["learning_rate"]))
    target_values = torch.tensor([target.thrust_N, target.tsfc_kg_N_s, target.mass_kg], dtype=sources.dtype, device=sources.device)
    tolerances = ordered_tensor(protocol["diagnostic_acceptance"]["relative_tolerances"], dtype=sources.dtype, device=sources.device)
    epsilons = ordered_tensor(protocol["diagnostic_acceptance"]["epsilons"], dtype=sources.dtype, device=sources.device)
    checkpoints = {int(value) for value in config["checkpoints"]}
    history: list[dict[str, Any]] = []

    def terms(design: torch.Tensor, current_sources: torch.Tensor) -> dict[str, torch.Tensor]:
        physical = candidate_module._physical_design_from_normalized(design, generator)
        forward_input = candidate_module._normalize_for_bundle(physical, forward_normalization)
        prediction = _decode_log_outputs(forward_model(forward_input), forward_normalization)
        hit = torch_hit_aligned_terms(prediction, target_values, tolerances, epsilons, alpha=float(protocol["hit_alignment"]["smooth_max_alpha"]))
        zeros = torch.zeros_like(hit["smooth_rho"])
        prior = (current_sources - initial).square().mean(dim=1)
        boundary = torch.relu(design.abs() - 1.0).square().mean(dim=1)
        objective = (
            float(config["target_loss_weight"]) * hit["smooth_rho"]
            + float(config["source_prior_weight"]) * prior
            + float(config["design_bound_penalty"]) * boundary
        )
        return {
            "physical_prediction": prediction,
            "axis_ratios": hit["axis_ratios"],
            "rho": hit["rho"],
            "sum_axis_ratios": hit["sum_axis_ratios"],
            "smooth_rho": hit["smooth_rho"],
            "target_loss": hit["smooth_rho"],
            "constraint_loss": zeros,
            "feasibility_loss": zeros,
            "converged_probability": torch.ones_like(zeros),
            "source_prior": prior,
            "boundary_penalty": boundary,
            "objective": objective,
        }

    with torch.no_grad():
        direct_design = direct_design.detach()
        before = terms(direct_design, initial)
    for step in range(1, int(config["steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        design = integrate_cfm_sources(generator, sources, normalized_condition)
        current = terms(design, sources)
        loss = current["objective"].mean()
        if not torch.isfinite(loss):
            raise Geometry19Performance3ComparisonError(f"D-Flow objective became non-finite at step {step}")
        loss.backward()
        if sources.grad is None or not torch.isfinite(sources.grad).all():
            raise Geometry19Performance3ComparisonError(f"D-Flow gradient became non-finite at step {step}")
        torch.nn.utils.clip_grad_norm_([sources], float(config["gradient_clip_norm"]))
        optimizer.step()
        with torch.no_grad():
            limit = float(config["maximum_source_displacement_linf"])
            sources.copy_(initial + (sources - initial).clamp(-limit, limit))
        if step in checkpoints:
            history.append({
                "step": step,
                "target_loss_metric": target_loss_mode,
                "surrogate_responsibility": "performance_only_v21_feasibility_authority",
                "mean_objective": float(current["objective"].detach().mean().cpu()),
                "mean_target_loss": float(current["target_loss"].detach().mean().cpu()),
                "mean_constraint_loss": 0.0,
                "mean_feasibility_loss": 0.0,
                "maximum_source_displacement_linf": float((sources.detach() - initial).abs().max().cpu()),
            })
    with torch.no_grad():
        final_design = integrate_cfm_sources(generator, sources, normalized_condition)
        after = terms(final_design, sources)

    def snapshot(values: Mapping[str, torch.Tensor], index: int) -> dict[str, Any]:
        predicted = values["physical_prediction"][index].detach().cpu().tolist()
        return {
            "predicted_targets": {name: float(value) for name, value in zip(OUTPUT_NAMES, predicted, strict=True)},
            "target_loss": float(values["target_loss"][index].detach().cpu()),
            "target_loss_metric": target_loss_mode,
            "surrogate_responsibility": "performance_only_v21_feasibility_authority",
            "hit_alignment": {
                "axis_ratios": {name: float(value) for name, value in zip(OUTPUT_NAMES, values["axis_ratios"][index].detach().cpu().tolist(), strict=True)},
                "rho": float(values["rho"][index].detach().cpu()),
                "sum_axis_ratios": float(values["sum_axis_ratios"][index].detach().cpu()),
                "smooth_rho": float(values["smooth_rho"][index].detach().cpu()),
            },
            "constraint_loss": 0.0,
            "feasibility_loss": 0.0,
            "converged_probability": 1.0,
            "source_prior": float(values["source_prior"][index].detach().cpu()),
            "boundary_penalty": float(values["boundary_penalty"][index].detach().cpu()),
            "objective": float(values["objective"][index].detach().cpu()),
        }

    metrics = []
    for index in range(initial.shape[0]):
        source_delta = sources[index].detach() - initial[index]
        design_delta = final_design[index].detach() - direct_design[index]
        metrics.append({
            "surrogate_before": snapshot(before, index),
            "surrogate_after": snapshot(after, index),
            "source_displacement": {"linf": float(source_delta.abs().max().cpu()), "l2": float(torch.linalg.vector_norm(source_delta).cpu()), "rms": float(source_delta.square().mean().sqrt().cpu())},
            "normalized_design_displacement": {"linf": float(design_delta.abs().max().cpu()), "l2": float(torch.linalg.vector_norm(design_delta).cpu()), "rms": float(design_delta.square().mean().sqrt().cpu())},
        })
    return final_design.detach(), sources.detach(), tuple(history), tuple(metrics)


def _performance3_rerank(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    target: Geometry19InverseTarget,
    protocol: Mapping[str, Any],
    root: Path,
    *,
    device: str,
    forward_model_tag: str | None = None,
    forward_weights_override: str | Path | None = None,
) -> dict[str, Any]:
    del forward_model_tag
    valid = [row for row in rows if row.get("valid")]
    summary["constraint_eligible_sort"] = "predicted_rho_ascending_then_candidate_id"
    if not valid:
        summary["surrogate_hit_alignment"] = {"version": protocol["hit_alignment"]["version"], "ranked_candidates": 0}
        return {}
    if forward_weights_override is None:
        raise Geometry19Performance3ComparisonError("performance3 model override is required")
    model_dir = Path(forward_weights_override)
    if not model_dir.is_absolute():
        model_dir = root / model_dir
    model_tag, features, targets = candidate_module._bundle_contract(model_dir)
    model, stats, _ = load_weight_bundle(model_dir, expected_model_tag=model_tag, expected_feature_names=features, expected_target_names=targets, device=device)
    model.eval().requires_grad_(False)
    dtype = next(model.parameters()).dtype
    torch_device = next(model.parameters()).device
    physical = torch.tensor([[row["geometry19"][name] for name in features] for row in valid], dtype=dtype, device=torch_device)
    with torch.no_grad():
        prediction = _decode_log_outputs(model(candidate_module._normalize_for_bundle(physical, stats)), stats)
        target_values = torch.tensor([target.thrust_N, target.tsfc_kg_N_s, target.mass_kg], dtype=dtype, device=torch_device)
        hit = torch_hit_aligned_terms(
            prediction,
            target_values,
            ordered_tensor(protocol["diagnostic_acceptance"]["relative_tolerances"], dtype=dtype, device=torch_device),
            ordered_tensor(protocol["diagnostic_acceptance"]["epsilons"], dtype=dtype, device=torch_device),
            alpha=float(protocol["hit_alignment"]["smooth_max_alpha"]),
        )
    for index, row in enumerate(valid):
        row["surrogate_hit_alignment"] = {
            "version": protocol["hit_alignment"]["version"],
            "predicted_outputs": {name: float(value) for name, value in zip(OUTPUT_NAMES, prediction[index].detach().cpu().tolist(), strict=True)},
            "axis_ratios": {name: float(value) for name, value in zip(OUTPUT_NAMES, hit["axis_ratios"][index].detach().cpu().tolist(), strict=True)},
            "rho": float(hit["rho"][index].detach().cpu()),
            "sum_axis_ratios": float(hit["sum_axis_ratios"][index].detach().cpu()),
            "smooth_rho": float(hit["smooth_rho"][index].detach().cpu()),
        }
    ordered = sorted(valid, key=lambda row: (float(row["surrogate_hit_alignment"]["rho"]), int(row["candidate_id"])))
    keep_count = int(summary["n_keep"])
    kept_ids = {int(row["candidate_id"]) for row in ordered[:keep_count]}
    ranks = {int(row["candidate_id"]): rank for rank, row in enumerate(ordered)}
    for row in rows:
        candidate_id = int(row["candidate_id"])
        row["kept"] = candidate_id in kept_ids
        row["selection_rank"] = ranks.get(candidate_id)
        row["selection_status"] = "KEPT_FOR_TRUE_PHYSICS" if row["kept"] else ("HIT_ALIGNED_RANK_TRIMMED" if row["valid"] else "FILTER_REJECTED")
    summary["kept_candidate_ids"] = [int(row["candidate_id"]) for row in ordered[:keep_count]]
    summary["surrogate_hit_alignment"] = {
        "version": protocol["hit_alignment"]["version"],
        "ranked_candidates": len(ordered),
        "minimum_predicted_rho": float(ordered[0]["surrogate_hit_alignment"]["rho"]),
        "maximum_kept_predicted_rho": float(ordered[keep_count - 1]["surrogate_hit_alignment"]["rho"]),
    }
    return {"model_tag": model_tag, **candidate_module._bundle_hashes(model_dir)}


def _performance3_refinement_objective(
    normalized: torch.Tensor,
    origin: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    target_values: torch.Tensor,
    models: RefinementModels,
    weights: Mapping[str, Any],
    hit_alignment: Mapping[str, Any],
    diagnostic_acceptance: Mapping[str, Any],
    target_loss_mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if target_loss_mode != "hit_aligned":
        raise Geometry19Performance3ComparisonError("performance3 refinement requires hit-aligned loss")
    physical = refinement_module._physical_from_normalized(normalized, lower, upper)
    prediction = _decode_log_outputs(models.forward(refinement_module._normalize_input(physical, models.forward_stats)), models.forward_stats)
    hit = torch_hit_aligned_terms(
        prediction,
        target_values,
        ordered_tensor(diagnostic_acceptance["relative_tolerances"], dtype=physical.dtype, device=physical.device),
        ordered_tensor(diagnostic_acceptance["epsilons"], dtype=physical.dtype, device=physical.device),
        alpha=float(hit_alignment["smooth_max_alpha"]),
    )
    zeros = torch.zeros_like(hit["smooth_rho"])
    trust = (normalized - origin).square().mean(dim=1)
    boundary = torch.relu(normalized.abs() - 1.0).square().mean(dim=1)
    total = float(weights["target_loss_weight"]) * hit["smooth_rho"] + float(weights["source_prior_weight"]) * trust + float(weights["design_bound_penalty"]) * boundary
    return total, {"target": hit["smooth_rho"], "axis_ratios": hit["axis_ratios"], "rho": hit["rho"], "sum_axis_ratios": hit["sum_axis_ratios"], "smooth_rho": hit["smooth_rho"], "constraint": zeros, "feasibility": zeros, "trust": trust}


@contextmanager
def _performance3_adapter() -> Iterator[None]:
    originals = (candidate_module._dflow_generate, candidate_module.rerank_candidates_by_predicted_rho, refinement_module._surrogate_objective)
    candidate_module._dflow_generate = _performance3_dflow
    candidate_module.rerank_candidates_by_predicted_rho = _performance3_rerank
    refinement_module._surrogate_objective = _performance3_refinement_objective
    try:
        yield
    finally:
        candidate_module._dflow_generate, candidate_module.rerank_candidates_by_predicted_rho, refinement_module._surrogate_objective = originals


def _validate_performance3_manifest(model_dir: Path) -> None:
    manifest = json.loads((model_dir / "model_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("output_transform") != "natural_log" or manifest.get("target_names") != list(OUTPUT_NAMES):
        raise Geometry19Performance3ComparisonError("performance3_log manifest contract mismatch")


def _choose_targets(rows: Sequence[Mapping[str, Any]], regions: Sequence[str], count: int, seed: int) -> list[dict[str, Any]]:
    selected = []
    for region in regions:
        pool = [dict(row) for row in rows if row["target_region"] == region]
        pool.sort(key=lambda row: sha256(f"{seed}:{row['target']['target_id']}".encode("utf-8")).hexdigest())
        if len(pool) < count:
            raise Geometry19Performance3ComparisonError(f"Not enough targets in region {region}")
        selected.extend(pool[:count])
    return selected


def _best_rho(*summaries: Mapping[str, Any] | None) -> float | None:
    values = []
    for summary in summaries:
        approximation = (summary or {}).get("best_approximation")
        if approximation and approximation.get("hit_alignment", {}).get("rho") is not None:
            values.append(float(approximation["hit_alignment"]["rho"]))
    return min(values) if values else None


def _run_arm(root: Path, target_path: Path, arm_root: Path, arm: str, model_dir: Path, generator_checkpoint: Path, device: str, resume: bool) -> dict[str, Any]:
    started = time.perf_counter()
    candidate_parent = arm_root / "wp19_23b"
    candidate_root = candidate_parent / "dflow"
    physics_root = arm_root / "wp19_23c"
    refinement_root = arm_root / "wp19_23d"
    nsga_root = arm_root / "wp19_23e"
    delivery_root = arm_root / "wp19_23f"
    adapter = _performance3_adapter() if arm == "performance3_log" else nullcontext()
    with adapter:
        b = _read_json(candidate_root / "summary.json") if resume else None
        if b is None:
            b = run_candidate_protocol(root, target_path, mode="dflow", device=device, output_root=candidate_root, generator_checkpoint=generator_checkpoint, dflow_target_loss_mode="hit_aligned", candidate_ranking_mode="predicted_rho", forward_weights_override=model_dir)
        c = _read_json(physics_root / "summary.json") if resume else None
        if c is None:
            c = run_true_physics_stage(root, target_path, candidate_root=candidate_parent, output_root=physics_root, sampling_mode="dflow")
        d = None
        e = None
        terminal_root = physics_root
        terminal_stage = "WP19-23C"
        if not c["downstream_stop"]:
            d = _read_json(refinement_root / "summary.json") if resume else None
            if d is None:
                d = run_refinement_stage(root, target_path, physics_root=physics_root, output_root=refinement_root, device=device, target_loss_mode="hit_aligned", true_acceptance_mode="hit_aligned", forward_weights_override=model_dir)
            terminal_root = refinement_root
            terminal_stage = "WP19-23D"
            if not d["downstream_stop"]:
                e = _read_json(nsga_root / "summary.json") if resume else None
                if e is None:
                    e = run_nsga2_stage(root, target_path, physics_root=physics_root, refinement_root=refinement_root, output_root=nsga_root)
                terminal_root = nsga_root
                terminal_stage = "WP19-23E"
        f = _read_json(delivery_root / "summary.json") if resume else None
        if f is None:
            f = run_delivery_stage(root, target_path, source_root=terminal_root, output_root=delivery_root, storage_root=arm_root / "mesh_storage", network_architecture_modified=True, export_mesh=False)
    c_calls = int(c["counts"]["unique_v2_1_evaluations"])
    d_calls = int(d["counts"]["true_v2_1_evaluations"]) if d else 0
    e_calls = int(e["counts"]["unique_v2_1_evaluations"]) if e else 0
    delivery_calls = int(f["counts"]["v2_1_recalculation_passed"])
    return {
        "arm": arm,
        "completed": True,
        "terminal_search_stage": terminal_stage,
        "hit": bool(f["counts"]["development_diagnostic_hits"]),
        "generation_hit": bool(c["downstream_stop"]),
        "refinement_hit": bool(d and d["downstream_stop"]),
        "nsga_hit": bool(e and e["counts"]["diagnostic_hits"]),
        "n_raw": int(b["candidate_protocol"]["n_raw"]),
        "n_valid": int(b["candidate_protocol"]["n_valid"]),
        "n_keep": int(b["candidate_protocol"]["n_keep"]),
        "physics_acceptable_c": int(c["counts"]["physics_acceptable"]),
        "constraint_feasible_c": int(c["counts"]["constraint_feasible"]),
        "c_v2_1_calls": c_calls,
        "d_v2_1_calls": d_calls,
        "e_v2_1_calls": e_calls,
        "search_v2_1_calls": c_calls + d_calls + e_calls,
        "delivery_v2_1_calls": delivery_calls,
        "total_v2_1_calls": c_calls + d_calls + e_calls + delivery_calls,
        "best_true_rho": _best_rho(c, d, e),
        "elapsed_seconds": time.perf_counter() - started,
        "model_manifest_sha256": sha256_file(model_dir / "model_manifest.json"),
    }


def _bootstrap_interval(differences: np.ndarray, regions: Sequence[str], replicates: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    region_values = {region: differences[np.asarray(regions) == region] for region in sorted(set(regions))}
    samples = np.empty(replicates, dtype=float)
    for index in range(replicates):
        pieces = [rng.choice(values, size=len(values), replace=True) for values in region_values.values()]
        samples[index] = float(np.concatenate(pieces).mean())
    return {"point": float(differences.mean()), "lower": float(np.quantile(samples, 0.025)), "upper": float(np.quantile(samples, 0.975))}


def _paired_exact_p(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(gains, losses) + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def _aggregate(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    arms = ("shared48_s0", "performance3_log")
    def arm_metrics(selected: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
        values = [row[arm] for row in selected]
        hits = [bool(value["hit"]) for value in values]
        return {
            "targets": len(values),
            "hits": sum(hits),
            "hit_rate": float(np.mean(hits)),
            "generation_hits": sum(bool(value["generation_hit"]) for value in values),
            "refinement_hits": sum(bool(value["refinement_hit"]) for value in values),
            "nsga_hits": sum(bool(value["nsga_hit"]) for value in values),
            "search_v2_1_calls": sum(int(value["search_v2_1_calls"]) for value in values),
            "total_v2_1_calls": sum(int(value["total_v2_1_calls"]) for value in values),
            "mean_valid_rate": float(np.mean([value["n_valid"] / value["n_raw"] for value in values])),
            "mean_physics_acceptable_rate_c": float(np.mean([value["physics_acceptable_c"] / max(value["c_v2_1_calls"], 1) for value in values])),
            "median_best_true_rho": float(np.median([value["best_true_rho"] for value in values if value["best_true_rho"] is not None])),
        }

    by_arm = {arm: arm_metrics(rows, arm) for arm in arms}
    by_region = {
        region: {
            arm: arm_metrics(
                [row for row in rows if row["target_region"] == region], arm
            )
            for arm in arms
        }
        for region in config["target_suite"]["regions"]
    }
    left = np.asarray([bool(row["shared48_s0"]["hit"]) for row in rows], dtype=int)
    right = np.asarray([bool(row["performance3_log"]["hit"]) for row in rows], dtype=int)
    differences = right - left
    gains = int(np.sum((right == 1) & (left == 0)))
    losses = int(np.sum((right == 0) & (left == 1)))
    stats = config["statistics"]
    return {
        "arms": by_arm,
        "paired": {
            "performance3_only_hits": gains,
            "s0_only_hits": losses,
            "both_hit": int(np.sum((right == 1) & (left == 1))),
            "neither_hit": int(np.sum((right == 0) & (left == 0))),
            "exact_two_sided_p": _paired_exact_p(gains, losses),
            "performance3_lower_best_true_rho": sum(
                row["performance3_log"]["best_true_rho"]
                < row["shared48_s0"]["best_true_rho"]
                for row in rows
            ),
            "s0_lower_best_true_rho": sum(
                row["shared48_s0"]["best_true_rho"]
                < row["performance3_log"]["best_true_rho"]
                for row in rows
            ),
            "cluster_bootstrap_hit_rate_difference": _bootstrap_interval(differences.astype(float), [str(row["target_region"]) for row in rows], int(stats["cluster_bootstrap_replicates"]), int(stats["bootstrap_seed"])),
        },
        "regions": by_region,
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    left = summary["aggregate"]["arms"]["shared48_s0"]
    right = summary["aggregate"]["arms"]["performance3_log"]
    paired = summary["aggregate"]["paired"]
    interval = paired["cluster_bootstrap_hit_rate_difference"]
    lines = [
        "# S0 与 performance3_log 全流程对比",
        "",
        f"> 生成时间：{summary['generated_at_utc']}",
        "> 本实验不修改生产pipeline；两臂使用相同目标、bounded CFM、256源、V2.1判据、梯度增强和NSGA-II预算。",
        "",
        "## 关键结果",
        "",
        "| 指标 | S0 | performance3_log | 差值 |",
        "|---|---:|---:|---:|",
        f"| 最终命中 | {left['hits']}/{left['targets']} ({left['hit_rate']:.1%}) | {right['hits']}/{right['targets']} ({right['hit_rate']:.1%}) | {right['hit_rate']-left['hit_rate']:+.1%} |",
        f"| C阶段命中 | {left['generation_hits']} | {right['generation_hits']} | {right['generation_hits']-left['generation_hits']:+d} |",
        f"| D阶段救回 | {left['refinement_hits']} | {right['refinement_hits']} | {right['refinement_hits']-left['refinement_hits']:+d} |",
        f"| E阶段救回 | {left['nsga_hits']} | {right['nsga_hits']} | {right['nsga_hits']-left['nsga_hits']:+d} |",
        f"| 搜索V2.1调用 | {left['search_v2_1_calls']} | {right['search_v2_1_calls']} | {right['search_v2_1_calls']-left['search_v2_1_calls']:+d} |",
        f"| 平均生成合法率 | {left['mean_valid_rate']:.1%} | {right['mean_valid_rate']:.1%} | {right['mean_valid_rate']-left['mean_valid_rate']:+.1%} |",
        f"| C阶段物理可接受率 | {left['mean_physics_acceptable_rate_c']:.1%} | {right['mean_physics_acceptable_rate_c']:.1%} | {right['mean_physics_acceptable_rate_c']-left['mean_physics_acceptable_rate_c']:+.1%} |",
        f"| median最终最佳真值rho | {left['median_best_true_rho']:.3f} | {right['median_best_true_rho']:.3f} | {right['median_best_true_rho']-left['median_best_true_rho']:+.3f} |",
        "",
        "## 配对判断",
        "",
        f"performance3独占命中 {paired['performance3_only_hits']}，S0独占命中 {paired['s0_only_hits']}，双方命中 {paired['both_hit']}，双方未命中 {paired['neither_hit']}。",
        f"最终命中率差值的分层bootstrap 95% CI为 [{interval['lower']:+.1%}, {interval['upper']:+.1%}]，双侧精确检验p={paired['exact_two_sided_p']:.4f}。",
        f"逐目标比较最终最佳真值rho：performance3更低 {paired['performance3_lower_best_true_rho']} 个，S0更低 {paired['s0_lower_best_true_rho']} 个。",
        "",
        "## 分区结果",
        "",
        "| 区域 | S0命中 | performance3命中 | S0搜索调用 | performance3搜索调用 | S0 median rho | performance3 median rho |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            (
                f"| {region} | {values['shared48_s0']['hits']}/{values['shared48_s0']['targets']} | "
                f"{values['performance3_log']['hits']}/{values['performance3_log']['targets']} | "
                f"{values['shared48_s0']['search_v2_1_calls']} | {values['performance3_log']['search_v2_1_calls']} | "
                f"{values['shared48_s0']['median_best_true_rho']:.3f} | {values['performance3_log']['median_best_true_rho']:.3f} |"
            )
            for region, values in summary["aggregate"]["regions"].items()
        ],
        "",
        "## 解释边界",
        "",
        "`performance3_log`臂不是把三输出模型误当成48输出模型：D-Flow和梯度增强仅优化三维性能目标；不可用的45维代理约束损失被关闭，最终合法性、约束与命中均由V2.1真值判定。S0臂保持当前完整目标。NSGA-II本身只调用V2.1，两臂算法与预算完全相同。",
        "",
        "无论结果如何，本实验均不自动修改pipeline默认项，等待用户确认。",
        "",
    ]
    return "\n".join(lines)


def run_comparison(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    device: str | None = None,
    output_root: str | Path | None = None,
    resume: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    if config.get("protocol_version") != PROTOCOL_VERSION or config.get("pipeline_mutation_allowed") is not False:
        raise Geometry19Performance3ComparisonError("Comparison protocol or mutation guard mismatch")
    contract = load_wp19_23_contract(root / "config/physics/wp19_23_inverse_closure_v1.yaml")
    shared = config["shared_pipeline"]
    if int(shared["raw_candidate_count"]) != int(contract.payload["generation"]["raw_candidate_count"]):
        raise Geometry19Performance3ComparisonError("Comparison raw count differs from active contract")
    source_path = root / config["target_suite"]["source"]
    target_count = int(
        config["smoke"]["targets_per_region"]
        if smoke
        else config["target_suite"]["targets_per_region"]
    )
    selected = _choose_targets(
        _read_jsonl(source_path),
        config["target_suite"]["regions"],
        target_count,
        int(config["target_suite"]["selection_seed"]),
    )
    run_name = "smoke" if smoke else "formal"
    runtime = Path(output_root).resolve() if output_root else root / config["artifacts"]["runtime_root"] / run_name
    if runtime.exists() and not resume:
        raise Geometry19Performance3ComparisonError(f"Refusing to overwrite comparison: {runtime}")
    runtime.mkdir(parents=True, exist_ok=True)
    selected_path = runtime / config["artifacts"]["selected_target_suite"]
    if selected_path.exists() and _read_jsonl(selected_path) != selected:
        raise Geometry19Performance3ComparisonError("Resume target suite mismatch")
    if not selected_path.exists():
        _write_jsonl(selected_path, selected)
    model_dirs = {arm: (root / spec["model_root"]).resolve() for arm, spec in config["arms"].items()}
    _validate_performance3_manifest(model_dirs["performance3_log"])
    generator_checkpoint = (root / shared["generator_checkpoint"]).resolve()
    runtime_device = device or str(shared["device"])
    rows = []
    for index, selected_row in enumerate(selected):
        target = selected_row["target"]
        target_root = runtime / f"{index:02d}_{selected_row['target_region']}_{target['target_id'][:8]}"
        target_path = target_root / "target.yaml"
        if not target_path.exists():
            _write_yaml(target_path, target)
        arm_results = {}
        for arm in ("shared48_s0", "performance3_log"):
            result_path = target_root / arm / "comparison_result.json"
            result = _read_json(result_path) if resume else None
            if result is None:
                try:
                    result = _run_arm(root, target_path, target_root / arm, arm, model_dirs[arm], generator_checkpoint, runtime_device, resume)
                except Geometry19RefinementError as exc:
                    result = {"arm": arm, "completed": False, "error_type": type(exc).__name__, "message": str(exc), "hit": False}
                _write_json(result_path, result)
            arm_results[arm] = result
        if not all(value.get("completed") for value in arm_results.values()):
            raise Geometry19Performance3ComparisonError(f"Incomplete paired flow for target {target['target_id']}")
        rows.append({"target_id": target["target_id"], "target_region": selected_row["target_region"], **arm_results})
        _write_jsonl(runtime / config["artifacts"]["comparison_ledger"], rows)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "work_package": config["work_package"],
        "status": "EVALUATION_COMPLETE_NO_INTEGRATION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_use_allowed": False,
        "pipeline_mutated": False,
        "smoke": bool(smoke),
        "target_count": len(rows),
        "target_suite_sha256": sha256_file(selected_path),
        "source_target_suite_sha256": sha256_file(source_path),
        "aggregate": _aggregate(rows, config),
        "model_manifests": {arm: sha256_file(path / "model_manifest.json") for arm, path in model_dirs.items()},
        "protected_pipeline_hashes": {path: sha256_file(root / path) for path in config["protected_pipeline_files"]},
    }
    _write_json(runtime / config["artifacts"]["summary"], summary)
    (runtime / config["artifacts"]["report"]).write_text(_render_report(summary), encoding="utf-8")
    return summary
