"""Direct/D-Flow generation and fixed-raw candidate filtering for WP19-23B."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from src.ai.candidate_protocol import apply_candidate_protocol_v21
from src.ai.generators import CFMGenerator, load_generator
from src.ai.geometry19_cfm import integrate_cfm_sources
from src.ai.geometry19_inverse_contract import (
    FROZEN_RAW_CANDIDATE_COUNT,
    Geometry19InverseTarget,
    OUTPUT_NAMES,
    load_target,
)
from src.ai.geometry19_pipeline_context import PipelineContext, resolve_pipeline_context
from src.ai.geometry19_hit_alignment import ordered_tensor, torch_hit_aligned_terms
from src.ai.geometry19_forward_model import (
    attach_forward_spec,
    decode_forward_outputs,
    resolve_forward_model_spec,
    spec_from_normalization,
    transform_physical_targets,
)
from src.ai.geometry19_surrogate import load_weight_bundle, sha256_file
from src.physics import Geometry19, GeometryStateError, ModelConfig, derive_geometry_state


class Geometry19CandidateError(ValueError):
    """Raised when WP19-23B generation or candidate filtering violates contract."""


@dataclass(frozen=True)
class CandidateBatch:
    mode: str
    candidate_ids: np.ndarray
    initial_sources: np.ndarray
    direct_normalized: np.ndarray
    active_normalized: np.ndarray
    refined_sources: np.ndarray | None
    dflow_history: tuple[dict[str, Any], ...]
    dflow_candidate_metrics: tuple[dict[str, Any], ...] = ()
    dflow_model_hashes: Mapping[str, Any] = field(default_factory=dict)
    generator_model_tag: str | None = None


def sample_device_invariant_sources(
    generator: CFMGenerator,
    count: int,
    seed: int,
) -> torch.Tensor:
    """Draw the paired source matrix on CPU, then transfer it to the runtime device."""

    if count <= 0:
        raise Geometry19CandidateError("Source count must be positive")
    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(int(seed))
    sources = torch.randn(
        count,
        generator.design_dim,
        generator=random_generator,
        device="cpu",
        dtype=generator.dtype,
    )
    return sources.to(device=generator.device)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _bundle_contract(directory: Path) -> tuple[str, list[str], list[str]]:
    manifest = json.loads((directory / "model_manifest.json").read_text(encoding="utf-8"))
    return (
        str(manifest["model_tag"]),
        list(manifest["feature_names"]),
        list(manifest["target_names"]),
    )


def _bundle_hashes(directory: Path) -> dict[str, str]:
    paths = {
        "manifest_sha256": directory / "model_manifest.json",
        "weights_sha256": directory / "weights_best_val.pt",
        "normalization_sha256": directory / "normalization.npz",
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    hashes["bundle_sha256"] = sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return hashes


def _resolve_generator_checkpoint(
    root: Path,
    protocol: Mapping[str, Any],
    override: str | Path | None,
) -> Path:
    generator = protocol["models"]["generator"]
    default_variant = generator["default_variant"]
    configured = Path(
        override
        if override is not None
        else generator["variants"][default_variant]["checkpoint"]
    )
    return configured.resolve() if configured.is_absolute() else (root / configured).resolve()


def _physical_design_from_normalized(
    normalized: torch.Tensor,
    generator: CFMGenerator,
) -> torch.Tensor:
    if generator.normalizer is None:
        raise Geometry19CandidateError("CFM checkpoint has no attached normalizer")
    lower = torch.tensor(
        [item["physical_min"] for item in generator.normalizer.design],
        dtype=normalized.dtype,
        device=normalized.device,
    )
    upper = torch.tensor(
        [item["physical_max"] for item in generator.normalizer.design],
        dtype=normalized.dtype,
        device=normalized.device,
    )
    return lower + 0.5 * (normalized + 1.0) * (upper - lower)


def _normalize_for_bundle(
    physical_design: torch.Tensor,
    normalization: Mapping[str, np.ndarray],
) -> torch.Tensor:
    mean = torch.as_tensor(
        normalization["input_mean"],
        dtype=physical_design.dtype,
        device=physical_design.device,
    )
    std = torch.as_tensor(
        normalization["input_std"],
        dtype=physical_design.dtype,
        device=physical_design.device,
    )
    return (physical_design - mean) / std


def _dflow_generate(
    generator: CFMGenerator,
    initial_sources: torch.Tensor,
    direct_design: torch.Tensor,
    normalized_condition: torch.Tensor,
    target: Geometry19InverseTarget,
    protocol: Mapping[str, Any],
    forward_model: nn.Module,
    forward_normalization: Mapping[str, np.ndarray],
    feasibility_model: nn.Module | None,
    feasibility_normalization: Mapping[str, np.ndarray] | None,
    target_loss_mode: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    config = protocol["dflow"]
    if target_loss_mode not in {"legacy_mse", "hit_aligned"}:
        raise Geometry19CandidateError("Unsupported D-Flow target loss mode")
    generator.eval().requires_grad_(False)
    forward_model.eval().requires_grad_(False)
    initial = initial_sources.detach().clone()
    sources = initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([sources], lr=float(config["learning_rate"]))
    target_values = torch.tensor(
        [target.thrust_N, target.tsfc_kg_N_s, target.mass_kg],
        dtype=sources.dtype,
        device=sources.device,
    )
    forward_spec = spec_from_normalization(forward_normalization)
    if forward_spec.use_feasibility_loss:
        if feasibility_model is None or feasibility_normalization is None:
            raise Geometry19CandidateError(
                "Configured feasibility loss requires a feasibility bundle"
            )
        feasibility_model.eval().requires_grad_(False)
    output_mean = torch.as_tensor(
        forward_normalization["output_mean"], dtype=sources.dtype, device=sources.device
    )
    output_std = torch.as_tensor(
        forward_normalization["output_std"], dtype=sources.dtype, device=sources.device
    )
    tolerance_values = ordered_tensor(
        protocol["diagnostic_acceptance"]["relative_tolerances"],
        dtype=sources.dtype,
        device=sources.device,
    )
    epsilon_values = ordered_tensor(
        protocol["diagnostic_acceptance"]["epsilons"],
        dtype=sources.dtype,
        device=sources.device,
    )
    smooth_max_alpha = float(protocol["hit_alignment"]["smooth_max_alpha"])
    transformed_target = transform_physical_targets(
        target_values, forward_spec.output_transform
    )
    normalized_target = (transformed_target - output_mean[:3]) / output_std[:3]
    checkpoints = set(int(value) for value in config["checkpoints"])
    history: list[dict[str, Any]] = []

    def surrogate_terms(
        design: torch.Tensor, current_sources: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        physical = _physical_design_from_normalized(design, generator)
        forward_input = _normalize_for_bundle(physical, forward_normalization)
        normalized_prediction = forward_model(forward_input)
        physical_prediction = decode_forward_outputs(
            normalized_prediction, forward_normalization
        )
        hit_terms = torch_hit_aligned_terms(
            physical_prediction,
            target_values,
            tolerance_values,
            epsilon_values,
            alpha=smooth_max_alpha,
        )
        target_loss = (
            (normalized_prediction[:, :3] - normalized_target).square().mean(dim=1)
            if target_loss_mode == "legacy_mse"
            else hit_terms["smooth_rho"]
        )
        zeros = torch.zeros_like(target_loss)
        if forward_spec.use_constraint_loss:
            constraint_scale = output_std[3:].abs().clamp_min(1.0e-12)
            constraint_loss = (
                torch.relu(physical_prediction[:, 3:]) / constraint_scale
            ).square().mean(dim=1)
        else:
            constraint_loss = zeros
        if forward_spec.use_feasibility_loss:
            feasibility_input = _normalize_for_bundle(physical, feasibility_normalization)
            logits = feasibility_model(feasibility_input)
            converged_log_probability = torch.logsumexp(
                logits[:, :2], dim=1
            ) - torch.logsumexp(logits, dim=1)
            feasibility_loss = -converged_log_probability
            converged_probability = converged_log_probability.exp()
        else:
            feasibility_loss = zeros
            converged_probability = torch.ones_like(zeros)
        prior = (current_sources - initial).square().mean(dim=1)
        boundary = torch.relu(design.abs() - 1.0).square().mean(dim=1)
        objective = (
            float(config["target_loss_weight"]) * target_loss
            + float(config["constraint_loss_weight"]) * constraint_loss
            + float(config["feasibility_loss_weight"]) * feasibility_loss
            + float(config["source_prior_weight"]) * prior
            + float(config["design_bound_penalty"]) * boundary
        )
        return {
            "physical_prediction": physical_prediction,
            "axis_ratios": hit_terms["axis_ratios"],
            "rho": hit_terms["rho"],
            "sum_axis_ratios": hit_terms["sum_axis_ratios"],
            "smooth_rho": hit_terms["smooth_rho"],
            "target_loss": target_loss,
            "constraint_loss": constraint_loss,
            "feasibility_loss": feasibility_loss,
            "converged_probability": converged_probability,
            "source_prior": prior,
            "boundary_penalty": boundary,
            "objective": objective,
        }

    with torch.no_grad():
        direct_design = direct_design.detach()
        before_terms = surrogate_terms(direct_design, initial)
    for step in range(1, int(config["steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        design = integrate_cfm_sources(generator, sources, normalized_condition)
        terms = surrogate_terms(design, sources)
        objective = terms["objective"]
        loss = objective.mean()
        if not torch.isfinite(loss):
            raise Geometry19CandidateError(f"D-Flow objective became non-finite at step {step}")
        loss.backward()
        if sources.grad is None or not torch.isfinite(sources.grad).all():
            raise Geometry19CandidateError(f"D-Flow gradient became non-finite at step {step}")
        torch.nn.utils.clip_grad_norm_([sources], float(config["gradient_clip_norm"]))
        optimizer.step()
        with torch.no_grad():
            displacement = (sources - initial).clamp(
                -float(config["maximum_source_displacement_linf"]),
                float(config["maximum_source_displacement_linf"]),
            )
            sources.copy_(initial + displacement)
        if step in checkpoints:
            history.append(
                {
                    "step": step,
                    "target_loss_metric": target_loss_mode,
                    "mean_objective": float(objective.detach().mean().cpu()),
                    "mean_target_loss": float(terms["target_loss"].detach().mean().cpu()),
                    "mean_constraint_loss": float(
                        terms["constraint_loss"].detach().mean().cpu()
                    ),
                    "mean_feasibility_loss": float(
                        terms["feasibility_loss"].detach().mean().cpu()
                    ),
                    "maximum_source_displacement_linf": float(
                        (sources.detach() - initial).abs().max().cpu()
                    ),
                }
            )
    with torch.no_grad():
        final_design = integrate_cfm_sources(generator, sources, normalized_condition)
        after_terms = surrogate_terms(final_design, sources)

    def snapshot(terms: Mapping[str, torch.Tensor], index: int) -> dict[str, Any]:
        predictions = terms["physical_prediction"][index, :3].detach().cpu().tolist()
        return {
            "predicted_targets": {
                name: float(value) for name, value in zip(OUTPUT_NAMES, predictions, strict=True)
            },
            "target_loss": float(terms["target_loss"][index].detach().cpu()),
            "target_loss_metric": target_loss_mode,
            "hit_alignment": {
                "axis_ratios": {
                    name: float(value)
                    for name, value in zip(
                        OUTPUT_NAMES,
                        terms["axis_ratios"][index].detach().cpu().tolist(),
                        strict=True,
                    )
                },
                "rho": float(terms["rho"][index].detach().cpu()),
                "sum_axis_ratios": float(
                    terms["sum_axis_ratios"][index].detach().cpu()
                ),
                "smooth_rho": float(terms["smooth_rho"][index].detach().cpu()),
            },
            "constraint_loss": float(terms["constraint_loss"][index].detach().cpu()),
            "feasibility_loss": float(terms["feasibility_loss"][index].detach().cpu()),
            "converged_probability": float(
                terms["converged_probability"][index].detach().cpu()
            ),
            "source_prior": float(terms["source_prior"][index].detach().cpu()),
            "boundary_penalty": float(terms["boundary_penalty"][index].detach().cpu()),
            "objective": float(terms["objective"][index].detach().cpu()),
        }

    metrics: list[dict[str, Any]] = []
    for index in range(initial.shape[0]):
        source_delta = sources[index].detach() - initial[index]
        design_delta = final_design[index].detach() - direct_design[index]
        metrics.append(
            {
                "surrogate_before": snapshot(before_terms, index),
                "surrogate_after": snapshot(after_terms, index),
                "source_displacement": {
                    "linf": float(source_delta.abs().max().cpu()),
                    "l2": float(torch.linalg.vector_norm(source_delta).cpu()),
                    "rms": float(source_delta.square().mean().sqrt().cpu()),
                },
                "normalized_design_displacement": {
                    "linf": float(design_delta.abs().max().cpu()),
                    "l2": float(torch.linalg.vector_norm(design_delta).cpu()),
                    "rms": float(design_delta.square().mean().sqrt().cpu()),
                },
            }
        )
    return final_design.detach(), sources.detach(), tuple(history), tuple(metrics)


def generate_candidate_batch(
    repository_root: str | Path,
    target: Geometry19InverseTarget,
    *,
    mode: str,
    device: str,
    generator_checkpoint: str | Path | None = None,
    dflow_target_loss_mode: str = "hit_aligned",
    forward_model_tag: str | None = None,
    forward_weights_override: str | Path | None = None,
    pipeline_context: PipelineContext | None = None,
) -> CandidateBatch:
    root = Path(repository_root).resolve()
    context = resolve_pipeline_context(root, pipeline_context)
    protocol = context.protocol
    generation = protocol["generation"]
    if mode not in generation["supported_sampling_modes"]:
        raise Geometry19CandidateError(f"Unsupported sampling mode: {mode}")
    count = int(generation["raw_candidate_count"])
    if (
        count != FROZEN_RAW_CANDIDATE_COUNT
        or int(generation["generator_call_count_per_target"]) != 1
    ):
        raise Geometry19CandidateError("WP19-23B requires one 256-source generation call")
    checkpoint = _resolve_generator_checkpoint(root, protocol, generator_checkpoint)
    generator = load_generator(checkpoint, device=device)
    if not isinstance(generator, CFMGenerator) or generator.design_dim != 19:
        raise Geometry19CandidateError("Frozen generator is not the Geometry19 CFM")
    generator_model_tag = str(generator.model_config.get("model_tag", ""))
    if not generator_model_tag:
        raise Geometry19CandidateError("Geometry19 CFM checkpoint has no model_tag")
    physical_condition = np.asarray(
        [target.thrust_N, target.tsfc_kg_N_s, target.mass_kg], dtype=np.float64
    )
    normalized_condition = torch.as_tensor(
        generator.normalizer.transform_conditions(physical_condition),
        dtype=generator.dtype,
        device=generator.device,
    )
    source_seed = int(generation["source_seed"])
    if generation["source_generation_device"] != "cpu":
        raise Geometry19CandidateError("Paired WP19-23B sources must be generated on CPU")
    sources = sample_device_invariant_sources(generator, count, source_seed)
    with torch.no_grad():
        direct = integrate_cfm_sources(generator, sources, normalized_condition)
    active = direct
    refined_sources = None
    history: tuple[dict[str, Any], ...] = ()
    candidate_metrics: tuple[dict[str, Any], ...] = ()
    dflow_model_hashes: dict[str, Any] = {}
    if mode == "dflow":
        forward_spec = resolve_forward_model_spec(
            root,
            protocol,
            logical_tag=forward_model_tag,
            weights_override=forward_weights_override,
        )
        forward_dir = forward_spec.directory
        forward_tag, features, forward_targets = _bundle_contract(forward_dir)
        if features != generator.normalizer.design_names:
            raise Geometry19CandidateError("Generator and surrogate feature contracts differ")
        forward_model, forward_stats, _ = load_weight_bundle(
            forward_dir,
            expected_model_tag=forward_tag,
            expected_feature_names=features,
            expected_target_names=forward_targets,
            device=device,
        )
        forward_stats = attach_forward_spec(forward_stats, forward_spec)
        feasibility_model = None
        feasibility_stats = None
        feasibility_hashes = None
        if forward_spec.use_feasibility_loss:
            feasibility_dir = root / protocol["models"]["feasibility"]["weights"]
            feasibility_tag, feasibility_features, classes = _bundle_contract(
                feasibility_dir
            )
            if features != feasibility_features:
                raise Geometry19CandidateError(
                    "Forward and feasibility feature contracts differ"
                )
            feasibility_model, feasibility_stats, _ = load_weight_bundle(
                feasibility_dir,
                expected_model_tag=feasibility_tag,
                expected_feature_names=features,
                expected_target_names=classes,
                device=device,
            )
            feasibility_hashes = {
                "model_tag": feasibility_tag,
                "used_in_objective": True,
                **_bundle_hashes(feasibility_dir),
            }
        active, refined, history, candidate_metrics = _dflow_generate(
            generator,
            sources,
            direct,
            normalized_condition,
            target,
            protocol,
            forward_model,
            forward_stats,
            feasibility_model,
            feasibility_stats,
            dflow_target_loss_mode,
        )
        refined_sources = refined.cpu().numpy()
        dflow_model_hashes = {
            "forward": {
                "logical_tag": forward_spec.logical_tag,
                "model_tag": forward_tag,
                "output_transform": forward_spec.output_transform,
                "output_contract": forward_spec.output_contract,
                "use_constraint_loss": forward_spec.use_constraint_loss,
                "use_feasibility_loss": forward_spec.use_feasibility_loss,
                **_bundle_hashes(forward_dir),
            },
        }
        if feasibility_hashes is not None:
            dflow_model_hashes["feasibility"] = feasibility_hashes
    return CandidateBatch(
        mode=mode,
        candidate_ids=np.arange(count, dtype=int),
        initial_sources=sources.detach().cpu().numpy(),
        direct_normalized=direct.detach().cpu().numpy(),
        active_normalized=active.detach().cpu().numpy(),
        refined_sources=refined_sources,
        dflow_history=history,
        dflow_candidate_metrics=candidate_metrics,
        dflow_model_hashes=dflow_model_hashes,
        generator_model_tag=generator_model_tag,
    )


def filter_candidate_batch(
    batch: CandidateBatch,
    schema: Mapping[str, Any],
    model_config: ModelConfig,
    *,
    discrete_tolerance_count: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tolerance = (
        None if discrete_tolerance_count is None else float(discrete_tolerance_count)
    )
    if tolerance is not None and (
        not np.isfinite(tolerance) or tolerance < 0.0 or tolerance >= 0.5
    ):
        raise Geometry19CandidateError(
            "Discrete tolerance must be finite and in [0, 0.5) count"
        )
    discrete_policy = (
        "unconditional_nearest_integer"
        if tolerance is None
        else "nearest_integer_with_frozen_absolute_tolerance"
    )
    fields = list(schema["design_variables"])
    names = [item["name"] for item in fields]
    lower = np.asarray([item["bounds"][0] for item in fields], dtype=float)
    upper = np.asarray([item["bounds"][1] for item in fields], dtype=float)
    integer_indices = [index for index, item in enumerate(fields) if item["dtype"] == "int64"]
    rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    valid_mask: list[bool] = []
    for position, candidate_id in enumerate(batch.candidate_ids):
        normalized = np.asarray(batch.active_normalized[position], dtype=float)
        failure_code = None
        mapping_error = None
        canonical: dict[str, Any] | None = None
        design_hash = None
        discrete_distances: dict[str, float] | None = None
        if not np.isfinite(normalized).all():
            failure_code = "NON_FINITE_GENERATION"
        elif np.any(normalized < -1.0) or np.any(normalized > 1.0):
            failure_code = "OUT_OF_SCHEMA_BOUNDS"
        else:
            physical = lower + 0.5 * (normalized + 1.0) * (upper - lower)
            rounded = np.rint(physical[integer_indices])
            distances = np.abs(physical[integer_indices] - rounded)
            comparison_slack = 8.0 * np.finfo(float).eps * np.maximum(
                1.0, np.abs(physical[integer_indices])
            )
            discrete_distances = {
                names[index]: float(distance)
                for index, distance in zip(integer_indices, distances, strict=True)
            }
            if tolerance is not None and np.any(
                distances > tolerance + comparison_slack
            ):
                failure_code = "INVALID_DISCRETE_COORDINATE"
            else:
                physical[integer_indices] = rounded
            if failure_code is None and (
                np.any(physical < lower) or np.any(physical > upper)
            ):
                failure_code = "INVALID_DISCRETE_COORDINATE"
            if failure_code is None:
                mapping = {
                    name: int(physical[index]) if index in integer_indices else float(physical[index])
                    for index, name in enumerate(names)
                }
                try:
                    design = Geometry19.from_mapping(mapping)
                    derive_geometry_state(design, model_config)
                    canonical = design.as_mapping()
                    design_hash = sha256(
                        json.dumps(
                            canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
                        ).encode("utf-8")
                    ).hexdigest()
                    if design_hash in seen_hashes:
                        failure_code = "DUPLICATE_CANDIDATE"
                    else:
                        seen_hashes.add(design_hash)
                except (TypeError, ValueError, GeometryStateError) as exc:
                    failure_code = "GEOMETRY_MAPPING_FAILED"
                    mapping_error = f"{type(exc).__name__}: {exc}"
        valid = failure_code is None
        valid_mask.append(valid)
        rows.append(
            {
                "candidate_id": int(candidate_id),
                "sampling_mode": batch.mode,
                "normalized_design": normalized.tolist(),
                "geometry19": canonical,
                "design_hash": design_hash,
                "discrete_mapping": {
                    "policy": discrete_policy,
                    "absolute_tolerance_count": tolerance,
                    "distance_to_nearest_integer_count": discrete_distances,
                },
                "valid": valid,
                "failure_code": failure_code,
                "mapping_error": mapping_error,
                "kept": False,
            }
        )
    result = apply_candidate_protocol_v21(
        batch.candidate_ids,
        valid_mask,
        expected_raw_count=FROZEN_RAW_CANDIDATE_COUNT,
    )
    kept = set(result.kept_candidate_ids)
    valid_rank = {
        int(candidate_id): rank
        for rank, candidate_id in enumerate(
            sorted(
                int(row["candidate_id"])
                for row in rows
                if bool(row["valid"])
            )
        )
    }
    for row in rows:
        row["kept"] = row["candidate_id"] in kept
        row["constraint_satisfied"] = bool(row["valid"])
        row["selection_rank"] = valid_rank.get(int(row["candidate_id"]))
        row["selection_status"] = (
            "KEPT_FOR_TRUE_PHYSICS"
            if row["kept"]
            else ("POWER_OF_TWO_TRIMMED" if row["valid"] else "FILTER_REJECTED")
        )
    counts: dict[str, int] = {}
    for row in rows:
        code = row["failure_code"] or "VALID"
        counts[code] = counts.get(code, 0) + 1
    summary = {
        "n_raw": result.n_raw,
        "n_valid": result.n_valid,
        "n_keep": result.n_keep,
        "n_trimmed": result.n_trimmed,
        "valid_rate": result.valid_rate,
        "kept_candidate_ids": list(result.kept_candidate_ids),
        "constraint_eligible_sort": "candidate_id_ascending",
        "discrete_mapping": {
            "policy": discrete_policy,
            "absolute_tolerance_count": tolerance,
            "boundary_inclusive": True,
            "tie_break": "numpy_rint_nearest_even",
        },
        "failure_counts": counts,
    }
    return rows, summary


def rerank_candidates_by_predicted_rho(
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
    """Rank every valid candidate by the same tolerance-normalized hit metric."""

    valid = [row for row in rows if row.get("valid")]
    if not valid:
        summary["constraint_eligible_sort"] = "predicted_rho_ascending_then_candidate_id"
        summary["surrogate_hit_alignment"] = {
            "version": protocol["hit_alignment"]["version"],
            "ranked_candidates": 0,
        }
        return {}
    forward_spec = resolve_forward_model_spec(
        root,
        protocol,
        logical_tag=forward_model_tag,
        weights_override=forward_weights_override,
    )
    forward_dir = forward_spec.directory
    forward_tag, features, forward_targets = _bundle_contract(forward_dir)
    forward_model, normalization, _ = load_weight_bundle(
        forward_dir,
        expected_model_tag=forward_tag,
        expected_feature_names=features,
        expected_target_names=forward_targets,
        device=device,
    )
    normalization = attach_forward_spec(normalization, forward_spec)
    forward_model.eval().requires_grad_(False)
    dtype = next(forward_model.parameters()).dtype
    torch_device = next(forward_model.parameters()).device
    physical = torch.tensor(
        [[row["geometry19"][name] for name in features] for row in valid],
        dtype=dtype,
        device=torch_device,
    )
    forward_input = _normalize_for_bundle(physical, normalization)
    with torch.no_grad():
        physical_prediction = decode_forward_outputs(
            forward_model(forward_input), normalization
        )
        target_values = torch.tensor(
            [target.thrust_N, target.tsfc_kg_N_s, target.mass_kg],
            dtype=dtype,
            device=torch_device,
        )
        terms = torch_hit_aligned_terms(
            physical_prediction,
            target_values,
            ordered_tensor(
                protocol["diagnostic_acceptance"]["relative_tolerances"],
                dtype=dtype,
                device=torch_device,
            ),
            ordered_tensor(
                protocol["diagnostic_acceptance"]["epsilons"],
                dtype=dtype,
                device=torch_device,
            ),
            alpha=float(protocol["hit_alignment"]["smooth_max_alpha"]),
        )
    for index, row in enumerate(valid):
        axis_values = terms["axis_ratios"][index].detach().cpu().tolist()
        row["surrogate_hit_alignment"] = {
            "version": protocol["hit_alignment"]["version"],
            "predicted_outputs": {
                name: float(value)
                for name, value in zip(
                    OUTPUT_NAMES,
                    physical_prediction[index, :3].detach().cpu().tolist(),
                    strict=True,
                )
            },
            "axis_ratios": {
                name: float(value)
                for name, value in zip(OUTPUT_NAMES, axis_values, strict=True)
            },
            "rho": float(terms["rho"][index].detach().cpu()),
            "sum_axis_ratios": float(
                terms["sum_axis_ratios"][index].detach().cpu()
            ),
            "smooth_rho": float(terms["smooth_rho"][index].detach().cpu()),
        }
    ordered = sorted(
        valid,
        key=lambda row: (
            float(row["surrogate_hit_alignment"]["rho"]),
            int(row["candidate_id"]),
        ),
    )
    kept_ids = {int(row["candidate_id"]) for row in ordered[: int(summary["n_keep"])]}
    ranks = {int(row["candidate_id"]): rank for rank, row in enumerate(ordered)}
    for row in rows:
        candidate_id = int(row["candidate_id"])
        row["kept"] = candidate_id in kept_ids
        row["selection_rank"] = ranks.get(candidate_id)
        row["selection_status"] = (
            "KEPT_FOR_TRUE_PHYSICS"
            if row["kept"]
            else ("HIT_ALIGNED_RANK_TRIMMED" if row["valid"] else "FILTER_REJECTED")
        )
    summary["kept_candidate_ids"] = [
        int(row["candidate_id"]) for row in ordered[: int(summary["n_keep"])]
    ]
    summary["constraint_eligible_sort"] = "predicted_rho_ascending_then_candidate_id"
    summary["surrogate_hit_alignment"] = {
        "version": protocol["hit_alignment"]["version"],
        "ranked_candidates": len(ordered),
        "minimum_predicted_rho": float(ordered[0]["surrogate_hit_alignment"]["rho"]),
        "maximum_kept_predicted_rho": float(
            ordered[int(summary["n_keep"]) - 1]["surrogate_hit_alignment"]["rho"]
        ),
    }
    return {
        "logical_tag": forward_spec.logical_tag,
        "model_tag": forward_tag,
        "output_transform": forward_spec.output_transform,
        "output_contract": forward_spec.output_contract,
        **_bundle_hashes(forward_dir),
    }


def run_candidate_protocol(
    repository_root: str | Path,
    target_path: str | Path,
    *,
    mode: str,
    device: str,
    output_root: str | Path | None = None,
    generator_checkpoint: str | Path | None = None,
    dflow_target_loss_mode: str = "hit_aligned",
    candidate_ranking_mode: str = "predicted_rho",
    forward_model_tag: str | None = None,
    forward_weights_override: str | Path | None = None,
    pipeline_context: PipelineContext | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    target_source = Path(target_path).resolve()
    target = load_target(target_source)
    context = resolve_pipeline_context(root, pipeline_context)
    contract_path = context.contract_path
    protocol = context.protocol
    runtime_root = Path(output_root).resolve() if output_root else (
        root
        / protocol["candidate_generation_artifacts"]["runtime_root"]
        / target.target_id
        / mode
    )
    if runtime_root.exists():
        raise Geometry19CandidateError(f"Refusing to overwrite candidate run: {runtime_root}")
    runtime_root.mkdir(parents=True)
    try:
        checkpoint = _resolve_generator_checkpoint(root, protocol, generator_checkpoint)
        batch = generate_candidate_batch(
            root,
            target,
            mode=mode,
            device=device,
            generator_checkpoint=checkpoint,
            dflow_target_loss_mode=dflow_target_loss_mode,
            forward_model_tag=forward_model_tag,
            forward_weights_override=forward_weights_override,
            pipeline_context=context,
        )
        physics_runtime = context.physics_runtime
        schema_path = physics_runtime.source_paths["design_schema"]
        schema = physics_runtime.design_schema
        model_config = physics_runtime.model_config
        rows, filter_summary = filter_candidate_batch(
            batch,
            schema,
            model_config,
            discrete_tolerance_count=protocol["candidate_filter"][
                "discrete_absolute_tolerance_count"
            ],
        )
        if candidate_ranking_mode == "predicted_rho":
            ranking_forward_hashes = rerank_candidates_by_predicted_rho(
                rows,
                filter_summary,
                target,
                protocol,
                root,
                device=device,
                forward_model_tag=forward_model_tag,
                forward_weights_override=forward_weights_override,
            )
        elif candidate_ranking_mode == "candidate_id":
            ranking_forward_hashes = {}
        else:
            raise Geometry19CandidateError("Unsupported candidate ranking mode")
        source_seed = int(protocol["generation"]["source_seed"])
        model_tag = batch.generator_model_tag
        if not model_tag:
            raise Geometry19CandidateError("Generated batch has no generator model_tag")
        checkpoint_hash = sha256_file(checkpoint)
        model_hashes: dict[str, Any] = {
            "generator": {
                "model_tag": model_tag,
                "checkpoint_sha256": checkpoint_hash,
            },
            **dict(batch.dflow_model_hashes),
            "ranking_forward": ranking_forward_hashes,
        }
        resolved_forward_tag = (
            model_hashes.get("forward", {}).get("logical_tag")
            or model_hashes.get("ranking_forward", {}).get("logical_tag")
        )
        raw_rows = []
        for index, candidate_id in enumerate(batch.candidate_ids):
            dflow_metrics = (
                dict(batch.dflow_candidate_metrics[index])
                if batch.dflow_candidate_metrics
                else None
            )
            raw_rows.append(
                {
                    "target_id": target.target_id,
                    "candidate_id": int(candidate_id),
                    "source_id": int(candidate_id),
                    "source_seed": source_seed,
                    "model_tag": model_tag,
                    "checkpoint_hash": checkpoint_hash,
                    "model_hashes": model_hashes,
                    "source_initial": batch.initial_sources[index].tolist(),
                    "source_refined": (
                        batch.refined_sources[index].tolist()
                        if batch.refined_sources is not None
                        else None
                    ),
                    "direct_normalized_design": batch.direct_normalized[index].tolist(),
                    "active_normalized_design": batch.active_normalized[index].tolist(),
                    "x_19_raw": batch.active_normalized[index].tolist(),
                    "sampling_mode": mode,
                    "surrogate_before": (
                        dflow_metrics["surrogate_before"] if dflow_metrics else None
                    ),
                    "surrogate_after": (
                        dflow_metrics["surrogate_after"] if dflow_metrics else None
                    ),
                    "source_displacement": (
                        dflow_metrics["source_displacement"] if dflow_metrics else None
                    ),
                    "normalized_design_displacement": (
                        dflow_metrics["normalized_design_displacement"]
                        if dflow_metrics
                        else None
                    ),
                    "true_physics_acceptance_status": "PENDING_WP19_23C",
                }
            )
        _write_json(runtime_root / "target.json", target.as_mapping())
        _write_jsonl(runtime_root / "raw_candidates.jsonl", raw_rows)
        _write_jsonl(runtime_root / "candidate_filter_ledger.jsonl", rows)
        summary = {
            "work_package": "WP19-23B",
            "status": "PASS" if filter_summary["n_keep"] > 0 else "NO_VALID_CFM_CANDIDATE",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sampling_mode": mode,
            "target_id": target.target_id,
            "formal_use_allowed": False,
            "true_physics_acceptance_status": "PENDING_WP19_23C",
            "generator_checkpoint": str(checkpoint),
            "generator_override": generator_checkpoint is not None,
            "development_overrides": {
                "dflow_target_loss_mode": dflow_target_loss_mode,
                "candidate_ranking_mode": candidate_ranking_mode,
                "forward_model_tag": resolved_forward_tag,
                "forward_weights": (
                    str(Path(forward_weights_override).resolve())
                    if forward_weights_override is not None
                    else None
                ),
            },
            "candidate_protocol": filter_summary,
            "dflow_history": list(batch.dflow_history),
            "model_hashes": model_hashes,
            "hashes": {
                "protocol_hash": sha256_file(contract_path),
                "target_hash": sha256_file(target_source),
                "schema_hash": sha256_file(schema_path),
                "calibration_hash": model_config.calibration_pack_hash,
                "physics_bundle_hash": physics_runtime.physics_bundle_hash,
                "generator_checkpoint_hash": checkpoint_hash,
                "forward_checkpoint_hash": (
                    batch.dflow_model_hashes.get("forward", {}).get("bundle_sha256")
                    or ranking_forward_hashes.get("bundle_sha256")
                ),
                "feasibility_checkpoint_hash": (
                    batch.dflow_model_hashes.get("feasibility", {}).get("bundle_sha256")
                ),
                "initial_sources_sha256": sha256(
                    np.ascontiguousarray(batch.initial_sources).tobytes()
                ).hexdigest(),
            },
        }
        _write_json(runtime_root / "summary.json", summary)
        _write_json(
            runtime_root / "manifest.json",
            {
                "protocol_version": protocol["protocol_version"],
                "target_id": target.target_id,
                "sampling_mode": mode,
                "required_files": protocol["candidate_generation_artifacts"]["required_files"],
                "summary_sha256": sha256_file(runtime_root / "summary.json"),
            },
        )
        return summary
    except Exception as exc:
        _write_json(
            runtime_root / "failure.json",
            {
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise


def run_paired_candidate_audit(
    repository_root: str | Path,
    target_path: str | Path,
    *,
    device: str,
    output_root: str | Path | None = None,
    generator_checkpoint: str | Path | None = None,
    pipeline_context: PipelineContext | None = None,
) -> dict[str, Any]:
    """Generate an offline direct/D-Flow audit from one frozen source batch.

    This comparison artifact is not consumed by the production WP19-23
    pipeline; production chooses exactly one generation mode before filtering.
    """

    root = Path(repository_root).resolve()
    target_source = Path(target_path).resolve()
    target = load_target(target_source)
    context = resolve_pipeline_context(root, pipeline_context)
    contract_path = context.contract_path
    protocol = context.protocol
    runtime_root = Path(output_root).resolve() if output_root else (
        root
        / protocol["candidate_generation_artifacts"]["runtime_root"]
        / target.target_id
        / "paired"
    )
    if runtime_root.exists():
        raise Geometry19CandidateError(f"Refusing to overwrite paired candidate run: {runtime_root}")
    runtime_root.mkdir(parents=True)
    try:
        checkpoint = _resolve_generator_checkpoint(root, protocol, generator_checkpoint)
        paired_batch = generate_candidate_batch(
            root,
            target,
            mode="dflow",
            device=device,
            generator_checkpoint=checkpoint,
            pipeline_context=context,
        )
        direct_batch = CandidateBatch(
            mode="direct",
            candidate_ids=paired_batch.candidate_ids,
            initial_sources=paired_batch.initial_sources,
            direct_normalized=paired_batch.direct_normalized,
            active_normalized=paired_batch.direct_normalized,
            refined_sources=None,
            dflow_history=(),
            generator_model_tag=paired_batch.generator_model_tag,
        )
        physics_runtime = context.physics_runtime
        schema_path = physics_runtime.source_paths["design_schema"]
        schema = physics_runtime.design_schema
        model_config = physics_runtime.model_config
        discrete_tolerance = protocol["candidate_filter"][
            "discrete_absolute_tolerance_count"
        ]
        direct_rows, direct_filter = filter_candidate_batch(
            direct_batch,
            schema,
            model_config,
            discrete_tolerance_count=discrete_tolerance,
        )
        dflow_rows, dflow_filter = filter_candidate_batch(
            paired_batch,
            schema,
            model_config,
            discrete_tolerance_count=discrete_tolerance,
        )

        paired_valid_mask = [
            bool(direct_row["valid"] or dflow_row["valid"])
            for direct_row, dflow_row in zip(direct_rows, dflow_rows, strict=True)
        ]
        paired_protocol = apply_candidate_protocol_v21(
            paired_batch.candidate_ids,
            paired_valid_mask,
            expected_raw_count=FROZEN_RAW_CANDIDATE_COUNT,
        )
        paired_kept = set(paired_protocol.kept_candidate_ids)
        source_seed = int(protocol["generation"]["source_seed"])
        model_tag = paired_batch.generator_model_tag
        if not model_tag:
            raise Geometry19CandidateError("Generated batch has no generator model_tag")
        checkpoint_hash = sha256_file(checkpoint)
        raw_rows: list[dict[str, Any]] = []
        pair_rows: list[dict[str, Any]] = []
        for index, candidate_id_value in enumerate(paired_batch.candidate_ids):
            candidate_id = int(candidate_id_value)
            direct_row = direct_rows[index]
            dflow_row = dflow_rows[index]
            kept = candidate_id in paired_kept
            dflow_metrics = (
                dict(paired_batch.dflow_candidate_metrics[index])
                if paired_batch.dflow_candidate_metrics
                else None
            )
            direct_row["paired_kept"] = kept
            dflow_row["paired_kept"] = kept
            raw_rows.append(
                {
                    "target_id": target.target_id,
                    "candidate_id": candidate_id,
                    "source_id": candidate_id,
                    "source_seed": source_seed,
                    "model_tag": model_tag,
                    "checkpoint_hash": checkpoint_hash,
                    "model_hashes": {
                        "generator": {
                            "model_tag": model_tag,
                            "checkpoint_sha256": checkpoint_hash,
                        },
                        **dict(paired_batch.dflow_model_hashes),
                    },
                    "sampling_mode": "paired",
                    "source_initial": paired_batch.initial_sources[index].tolist(),
                    "source_refined": paired_batch.refined_sources[index].tolist(),
                    "design_direct": paired_batch.direct_normalized[index].tolist(),
                    "design_dflow": paired_batch.active_normalized[index].tolist(),
                    "x_19_raw": {
                        "direct": paired_batch.direct_normalized[index].tolist(),
                        "dflow": paired_batch.active_normalized[index].tolist(),
                    },
                    "surrogate_before": (
                        dflow_metrics["surrogate_before"] if dflow_metrics else None
                    ),
                    "surrogate_after": (
                        dflow_metrics["surrogate_after"] if dflow_metrics else None
                    ),
                    "source_displacement": (
                        dflow_metrics["source_displacement"] if dflow_metrics else None
                    ),
                    "normalized_design_displacement": (
                        dflow_metrics["normalized_design_displacement"]
                        if dflow_metrics
                        else None
                    ),
                    "true_physics_acceptance_status": "PENDING_WP19_23C",
                }
            )
            pair_rows.append(
                {
                    "candidate_id": candidate_id,
                    "direct_valid": bool(direct_row["valid"]),
                    "direct_failure_code": direct_row["failure_code"],
                    "dflow_valid": bool(dflow_row["valid"]),
                    "dflow_failure_code": dflow_row["failure_code"],
                    "paired_valid": bool(paired_valid_mask[index]),
                    "paired_kept": kept,
                    "selection_status": (
                        "PENDING_WP19_23C_TRUE_PHYSICS" if kept else
                        ("POWER_OF_TWO_TRIMMED" if paired_valid_mask[index] else "NO_LEGAL_VARIANT")
                    ),
                }
            )

        counts: dict[str, int] = {}
        for row in pair_rows:
            status = str(row["selection_status"])
            counts[status] = counts.get(status, 0) + 1
        paired_filter = {
            "n_raw": paired_protocol.n_raw,
            "n_valid": paired_protocol.n_valid,
            "n_keep": paired_protocol.n_keep,
            "n_trimmed": paired_protocol.n_trimmed,
            "valid_rate": paired_protocol.valid_rate,
            "kept_candidate_ids": list(paired_protocol.kept_candidate_ids),
            "selection_counts": counts,
            "validity_rule": "direct_valid_or_dflow_valid",
        }
        summary = {
            "work_package": "WP19-23B",
            "status": "PASS" if paired_protocol.n_keep > 0 else "NO_VALID_CFM_CANDIDATE",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sampling_mode": "paired",
            "target_id": target.target_id,
            "formal_use_allowed": False,
            "generator_call_count": 1,
            "true_physics_acceptance_status": "PENDING_WP19_23C",
            "generator_checkpoint": str(checkpoint),
            "generator_override": generator_checkpoint is not None,
            "candidate_protocol": paired_filter,
            "variant_filter_diagnostics": {
                "direct": direct_filter,
                "dflow": dflow_filter,
            },
            "dflow_history": list(paired_batch.dflow_history),
            "model_hashes": {
                "generator": {
                    "model_tag": model_tag,
                    "checkpoint_sha256": checkpoint_hash,
                },
                **dict(paired_batch.dflow_model_hashes),
            },
            "hashes": {
                "protocol_hash": sha256_file(contract_path),
                "target_hash": sha256_file(target_source),
                "schema_hash": sha256_file(schema_path),
                "calibration_hash": model_config.calibration_pack_hash,
                "physics_bundle_hash": physics_runtime.physics_bundle_hash,
                "generator_checkpoint_hash": checkpoint_hash,
                "initial_sources_sha256": sha256(
                    np.ascontiguousarray(paired_batch.initial_sources).tobytes()
                ).hexdigest(),
            },
        }
        _write_json(runtime_root / "target.json", target.as_mapping())
        _write_jsonl(runtime_root / "raw_candidates.jsonl", raw_rows)
        _write_jsonl(runtime_root / "paired_candidate_ledger.jsonl", pair_rows)
        _write_jsonl(runtime_root / "direct/candidate_filter_ledger.jsonl", direct_rows)
        _write_jsonl(runtime_root / "dflow/candidate_filter_ledger.jsonl", dflow_rows)
        _write_json(runtime_root / "summary.json", summary)
        _write_json(
            runtime_root / "manifest.json",
            {
                "protocol_version": protocol["protocol_version"],
                "target_id": target.target_id,
                "sampling_mode": "paired",
                "generator_call_count": 1,
                "required_files": [
                    "manifest.json",
                    "target.json",
                    "raw_candidates.jsonl",
                    "paired_candidate_ledger.jsonl",
                    "direct/candidate_filter_ledger.jsonl",
                    "dflow/candidate_filter_ledger.jsonl",
                    "summary.json",
                ],
                "summary_sha256": sha256_file(runtime_root / "summary.json"),
            },
        )
        return summary
    except Exception as exc:
        _write_json(
            runtime_root / "failure.json",
            {"status": "FAILED", "error_type": type(exc).__name__, "message": str(exc)},
        )
        raise
