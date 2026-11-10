"""Surrogate-gradient proposals with true V2.1 checkpoint acceptance for WP19-23D."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from src.ai.geometry19_inverse_contract import load_target
from src.ai.geometry19_hit_alignment import (
    evaluation_hit_key,
    ordered_tensor,
    torch_hit_aligned_terms,
)
from src.ai.geometry19_forward_model import (
    ForwardModelSpec,
    attach_forward_spec,
    decode_forward_outputs,
    resolve_forward_model_spec,
    transform_physical_targets,
)
from src.ai.geometry19_physics import evaluate_geometry19_candidate
from src.ai.geometry19_pipeline_context import PipelineContext, resolve_pipeline_context
from src.ai.geometry19_selection import (
    attach_dispositions,
    build_disposition_rows,
    candidate_sort_key,
    constraint_state,
    rank_delivery_candidates,
)
from src.ai.geometry19_surrogate import load_weight_bundle, sha256_file
from src.physics import (
    Environment,
    Geometry19,
    GeometryStateError,
    ModelConfig,
    derive_geometry_state,
)


class Geometry19RefinementError(ValueError):
    """Raised when WP19-23D refinement violates its frozen contract."""


@dataclass(frozen=True)
class RefinementModels:
    forward: nn.Module
    forward_stats: Mapping[str, np.ndarray]
    forward_spec: ForwardModelSpec
    feasibility: nn.Module | None
    feasibility_stats: Mapping[str, np.ndarray] | None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise Geometry19RefinementError(f"Cannot read ledger: {path}") from exc


def _normalize_geometry(
    geometry: Mapping[str, Any], fields: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    lower = np.asarray([item["bounds"][0] for item in fields], dtype=np.float64)
    upper = np.asarray([item["bounds"][1] for item in fields], dtype=np.float64)
    physical = np.asarray([geometry[item["name"]] for item in fields], dtype=np.float64)
    return 2.0 * (physical - lower) / (upper - lower) - 1.0


def select_refinement_seeds(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[Mapping[str, Any]],
    maximum: int,
) -> list[dict[str, Any]]:
    """Select valid unique-root-first seeds with deterministic 19D diversity."""

    if maximum <= 0:
        return []
    unique: dict[str, dict[str, Any]] = {}

    def quality(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            *evaluation_hit_key(row),
            float(row["total_true_error"]),
            str(row["evaluation_id"]),
        )

    for source in rows:
        if not source.get("physics_acceptable") or source.get("total_true_error") is None:
            continue
        design_hash = str(source["design_hash"])
        row = dict(source)
        previous = unique.get(design_hash)
        if previous is None or quality(row) < quality(previous):
            row["normalized_geometry"] = _normalize_geometry(row["geometry19"], fields).tolist()
            unique[design_hash] = row
    remaining = sorted(
        unique.values(),
        key=lambda row: (
            row["root_classification"] != "CONVERGED_UNIQUE",
            *quality(row),
        ),
    )
    if not remaining:
        return []
    selected = [remaining.pop(0)]
    while remaining and len(selected) < maximum:
        selected_matrix = np.asarray(
            [row["normalized_geometry"] for row in selected], dtype=np.float64
        )

        def rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
            vector = np.asarray(row["normalized_geometry"], dtype=np.float64)
            distance = float(np.min(np.linalg.norm(selected_matrix - vector, axis=1)))
            return (
                row["root_classification"] != "CONVERGED_UNIQUE",
                -distance,
                *quality(row),
            )

        chosen = min(remaining, key=rank)
        remaining.remove(chosen)
        selected.append(chosen)
    return selected


def _bundle_contract(directory: Path) -> tuple[str, list[str], list[str], dict[str, Any]]:
    manifest = json.loads((directory / "model_manifest.json").read_text(encoding="utf-8"))
    return (
        str(manifest["model_tag"]),
        list(manifest["feature_names"]),
        list(manifest["target_names"]),
        manifest,
    )


def _load_models(
    root: Path,
    protocol: Mapping[str, Any],
    device: str,
    forward_model_tag: str | None = None,
    forward_weights_override: str | Path | None = None,
) -> RefinementModels:
    forward_spec = resolve_forward_model_spec(
        root,
        protocol,
        logical_tag=forward_model_tag,
        weights_override=forward_weights_override,
    )
    forward_dir = forward_spec.directory
    forward_tag, features, targets, _ = _bundle_contract(forward_dir)
    forward, forward_stats, _ = load_weight_bundle(
        forward_dir,
        expected_model_tag=forward_tag,
        expected_feature_names=features,
        expected_target_names=targets,
        device=device,
    )
    forward_stats = attach_forward_spec(forward_stats, forward_spec)
    forward.eval().requires_grad_(False)
    feasibility = None
    feasibility_stats = None
    if forward_spec.use_feasibility_loss:
        feasibility_dir = root / protocol["models"]["feasibility"]["weights"]
        feasibility_tag, feasibility_features, classes, _ = _bundle_contract(
            feasibility_dir
        )
        if features != feasibility_features:
            raise Geometry19RefinementError(
                "Forward and feasibility feature contracts differ"
            )
        feasibility, feasibility_stats, _ = load_weight_bundle(
            feasibility_dir,
            expected_model_tag=feasibility_tag,
            expected_feature_names=features,
            expected_target_names=classes,
            device=device,
        )
        feasibility.eval().requires_grad_(False)
    return RefinementModels(
        forward,
        forward_stats,
        forward_spec,
        feasibility,
        feasibility_stats,
    )


def _physical_from_normalized(
    normalized: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    return lower + 0.5 * (normalized + 1.0) * (upper - lower)


def _normalize_input(
    physical: torch.Tensor, stats: Mapping[str, np.ndarray]
) -> torch.Tensor:
    mean = torch.as_tensor(stats["input_mean"], dtype=physical.dtype, device=physical.device)
    std = torch.as_tensor(stats["input_std"], dtype=physical.dtype, device=physical.device)
    return (physical - mean) / std


def _surrogate_objective(
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
    physical = _physical_from_normalized(normalized, lower, upper)
    forward_input = _normalize_input(physical, models.forward_stats)
    normalized_prediction = models.forward(forward_input)
    output_mean = torch.as_tensor(
        models.forward_stats["output_mean"], dtype=physical.dtype, device=physical.device
    )
    output_std = torch.as_tensor(
        models.forward_stats["output_std"], dtype=physical.dtype, device=physical.device
    )
    physical_prediction = decode_forward_outputs(
        normalized_prediction, models.forward_stats
    )
    hit_terms = torch_hit_aligned_terms(
        physical_prediction,
        target_values,
        ordered_tensor(
            diagnostic_acceptance["relative_tolerances"],
            dtype=physical.dtype,
            device=physical.device,
        ),
        ordered_tensor(
            diagnostic_acceptance["epsilons"],
            dtype=physical.dtype,
            device=physical.device,
        ),
        alpha=float(hit_alignment["smooth_max_alpha"]),
    )
    transformed_target = transform_physical_targets(
        target_values, models.forward_spec.output_transform
    )
    normalized_target = (transformed_target - output_mean[:3]) / output_std[:3]
    target_loss = (
        (normalized_prediction[:, :3] - normalized_target).square().mean(dim=1)
        if target_loss_mode == "legacy_mse"
        else hit_terms["smooth_rho"]
    )
    zeros = torch.zeros_like(target_loss)
    if models.forward_spec.use_constraint_loss:
        constraint_scale = output_std[3:].abs().clamp_min(1.0e-12)
        constraint_loss = (
            torch.relu(physical_prediction[:, 3:]) / constraint_scale
        ).square().mean(dim=1)
    else:
        constraint_loss = zeros
    if models.forward_spec.use_feasibility_loss:
        if models.feasibility is None or models.feasibility_stats is None:
            raise Geometry19RefinementError(
                "Configured feasibility loss requires a feasibility bundle"
            )
        feasibility_input = _normalize_input(physical, models.feasibility_stats)
        logits = models.feasibility(feasibility_input)
        feasibility_loss = -(
            torch.logsumexp(logits[:, :2], dim=1) - torch.logsumexp(logits, dim=1)
        )
    else:
        feasibility_loss = zeros
    trust = (normalized - origin).square().mean(dim=1)
    boundary = torch.relu(normalized.abs() - 1.0).square().mean(dim=1)
    total = (
        float(weights["target_loss_weight"]) * target_loss
        + float(weights["constraint_loss_weight"]) * constraint_loss
        + float(weights["feasibility_loss_weight"]) * feasibility_loss
        + float(weights["source_prior_weight"]) * trust
        + float(weights["design_bound_penalty"]) * boundary
    )
    return total, {
        "target": target_loss,
        "axis_ratios": hit_terms["axis_ratios"],
        "rho": hit_terms["rho"],
        "sum_axis_ratios": hit_terms["sum_axis_ratios"],
        "smooth_rho": hit_terms["smooth_rho"],
        "constraint": constraint_loss,
        "feasibility": feasibility_loss,
        "trust": trust,
    }


def _discrete_neighbor_proposal(
    normalized: torch.Tensor,
    baseline: torch.Tensor,
    origin: torch.Tensor,
    fields: Sequence[Mapping[str, Any]],
    lower: torch.Tensor,
    upper: torch.Tensor,
    target_values: torch.Tensor,
    models: RefinementModels,
    weights: Mapping[str, Any],
    hit_alignment: Mapping[str, Any],
    diagnostic_acceptance: Mapping[str, Any],
    target_loss_mode: str,
    model_config: ModelConfig,
) -> tuple[dict[str, Any], np.ndarray, float]:
    integer_indices = [index for index, item in enumerate(fields) if item["dtype"] == "int64"]
    candidates = []
    seen = set()
    for fraction in (1.0, 0.5, 0.25, 0.0):
        continuous = baseline + fraction * (normalized - baseline)
        physical = _physical_from_normalized(continuous[None, :], lower, upper)[0]
        choices = []
        for index in integer_indices:
            base = int(round(float(physical[index].detach().cpu())))
            low = int(fields[index]["bounds"][0])
            high = int(fields[index]["bounds"][1])
            choices.append(
                sorted({value for value in (base - 1, base, base + 1) if low <= value <= high})
            )
        for values in product(*choices):
            candidate = continuous.detach().clone()
            admissible = True
            for index, value in zip(integer_indices, values):
                normalized_value = (
                    2.0 * (value - lower[index]) / (upper[index] - lower[index]) - 1.0
                )
                if abs(float(normalized_value - origin[index])) > 0.25 + 1.0e-7:
                    admissible = False
                    break
                candidate[index] = normalized_value
            if not admissible:
                continue
            candidate = torch.maximum(torch.minimum(candidate, origin + 0.25), origin - 0.25)
            candidate = candidate.clamp(-1.0, 1.0)
            key = tuple(float(value) for value in candidate.detach().cpu())
            if key not in seen:
                candidates.append(candidate)
                seen.add(key)
    if not candidates:
        raise Geometry19RefinementError("No discrete neighbor satisfies the trust region")
    matrix = torch.stack(candidates)
    origins = origin[None, :].expand_as(matrix)
    with torch.no_grad():
        objectives, _ = _surrogate_objective(
            matrix,
            origins,
            lower,
            upper,
            target_values,
            models,
            weights,
            hit_alignment,
            diagnostic_acceptance,
            target_loss_mode,
        )
    order = torch.argsort(objectives, stable=True).detach().cpu().tolist()
    for position in order:
        candidate = matrix[position]
        candidate_physical = _physical_from_normalized(candidate[None, :], lower, upper)[0]
        mapping = {}
        for index, item in enumerate(fields):
            value = float(candidate_physical[index].detach().cpu())
            mapping[item["name"]] = int(round(value)) if item["dtype"] == "int64" else value
        try:
            design = Geometry19.from_mapping(mapping)
            derive_geometry_state(design, model_config)
        except (TypeError, ValueError, GeometryStateError):
            continue
        return design.as_mapping(), candidate.detach().cpu().numpy(), float(objectives[position].cpu())
    raise Geometry19RefinementError("No legal discrete-neighbor proposal remained")


def _design_hash(geometry: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(geometry, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def run_refinement_stage(
    repository_root: str | Path,
    target_path: str | Path,
    *,
    physics_root: str | Path | None = None,
    output_root: str | Path | None = None,
    device: str = "cpu",
    target_loss_mode: str = "hit_aligned",
    true_acceptance_mode: str = "hit_aligned",
    forward_model_tag: str | None = None,
    forward_weights_override: str | Path | None = None,
    pipeline_context: PipelineContext | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    if target_loss_mode not in {"legacy_mse", "hit_aligned"}:
        raise Geometry19RefinementError("Unsupported refinement target loss mode")
    if true_acceptance_mode not in {"legacy_mean", "hit_aligned"}:
        raise Geometry19RefinementError("Unsupported true acceptance mode")
    target_source = Path(target_path).resolve()
    target = load_target(target_source)
    context = resolve_pipeline_context(root, pipeline_context)
    contract_path = context.contract_path
    contract = context.contract
    protocol = context.protocol
    physics_runtime = context.physics_runtime
    source_root = Path(physics_root).resolve() if physics_root else (
        root
        / protocol["artifact_protocol"]["runtime_root"]
        / target.target_id
        / "wp19_23c"
    )
    runtime = Path(output_root).resolve() if output_root else (
        root
        / protocol["artifact_protocol"]["runtime_root"]
        / target.target_id
        / "wp19_23d"
    )
    if runtime.exists():
        raise Geometry19RefinementError(f"Refusing to overwrite refinement run: {runtime}")
    runtime.mkdir(parents=True)
    try:
        physics_summary_path = source_root / "summary.json"
        physics_summary = json.loads(physics_summary_path.read_text(encoding="utf-8"))
        if physics_summary["target_id"] != target.target_id:
            raise Geometry19RefinementError("WP19-23C target mismatch")
        if physics_summary["downstream_stop"]:
            raise Geometry19RefinementError("WP19-23C already hit; refinement is forbidden")
        physics_rows = _read_jsonl(source_root / "physics_forward_ledger.jsonl")
        selected_pool = [
            row for row in physics_rows if row["variant"] in {"direct", "dflow"}
        ]
        schema_path = physics_runtime.source_paths["design_schema"]
        schema = physics_runtime.design_schema
        fields = list(schema["design_variables"])
        maximum = int(protocol["refinement"]["maximum_candidates"])
        seeds = select_refinement_seeds(selected_pool, fields, maximum)
        if not seeds:
            raise Geometry19RefinementError("No true-physics-valid refinement seed exists")
        model_config = physics_runtime.model_config
        constraint_profile = physics_runtime.constraint_profile
        dominance_contract = context.constraint_dominance
        if model_config.calibration_pack_hash != physics_summary["hashes"]["calibration_hash"]:
            raise Geometry19RefinementError("Refinement calibration hash mismatch")
        source_physics_bundle_hash = physics_summary["hashes"].get(
            "physics_bundle_hash"
        )
        if (
            source_physics_bundle_hash is not None
            and source_physics_bundle_hash != physics_runtime.physics_bundle_hash
        ):
            raise Geometry19RefinementError("Refinement physics bundle mismatch")
        models = _load_models(
            root,
            protocol,
            device,
            forward_model_tag,
            forward_weights_override,
        )
        forward_dir = models.forward_spec.directory
        torch_device = next(models.forward.parameters()).device
        dtype = next(models.forward.parameters()).dtype
        lower = torch.tensor(
            [item["bounds"][0] for item in fields], dtype=dtype, device=torch_device
        )
        upper = torch.tensor(
            [item["bounds"][1] for item in fields], dtype=dtype, device=torch_device
        )
        current = torch.tensor(
            [row["normalized_geometry"] for row in seeds], dtype=dtype, device=torch_device
        )
        origins = current.detach().clone()
        integer_indices = [index for index, item in enumerate(fields) if item["dtype"] == "int64"]
        continuous_mask = torch.ones_like(current)
        continuous_mask[:, integer_indices] = 0.0
        target_values = torch.tensor(
            [target.thrust_N, target.tsfc_kg_N_s, target.mass_kg],
            dtype=dtype,
            device=torch_device,
        )
        weights = protocol["dflow"]
        checkpoints = [int(value) for value in protocol["refinement"]["checkpoints"]]
        if checkpoints != [0, 8, 16, 24]:
            raise Geometry19RefinementError("WP19-23D requires checkpoints [0,8,16,24]")
        current_results = [dict(row) for row in seeds]
        ledger: list[dict[str, Any]] = []
        true_evaluations = 0
        stopped_at = None
        previous_step = 0
        for checkpoint in checkpoints[1:]:
            variable = current.detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([variable], lr=float(weights["learning_rate"]))
            for _ in range(checkpoint - previous_step):
                optimizer.zero_grad(set_to_none=True)
                objective, _ = _surrogate_objective(
                    variable,
                    origins,
                    lower,
                    upper,
                    target_values,
                    models,
                    weights,
                    protocol["hit_alignment"],
                    protocol["diagnostic_acceptance"],
                    target_loss_mode,
                )
                loss = objective.mean()
                if not torch.isfinite(loss):
                    raise Geometry19RefinementError("Refinement objective became non-finite")
                loss.backward()
                if variable.grad is None or not torch.isfinite(variable.grad).all():
                    raise Geometry19RefinementError("Refinement gradient became non-finite")
                variable.grad.mul_(continuous_mask)
                torch.nn.utils.clip_grad_norm_([variable], float(weights["gradient_clip_norm"]))
                optimizer.step()
                with torch.no_grad():
                    variable.copy_(torch.maximum(torch.minimum(variable, origins + 0.25), origins - 0.25))
                    variable.clamp_(-1.0, 1.0)
                    variable[:, integer_indices] = current[:, integer_indices]
            next_current = current.detach().clone()
            for seed_index, seed in enumerate(seeds):
                geometry, proposed_normalized, surrogate_loss = _discrete_neighbor_proposal(
                    variable[seed_index],
                    current[seed_index],
                    origins[seed_index],
                    fields,
                    lower,
                    upper,
                    target_values,
                    models,
                    weights,
                    protocol["hit_alignment"],
                    protocol["diagnostic_acceptance"],
                    target_loss_mode,
                    model_config,
                )
                proposed = evaluate_geometry19_candidate(
                    int(seed["candidate_id"]),
                    f"refinement_step_{checkpoint}_seed_{seed_index}",
                    geometry,
                    _design_hash(geometry),
                    target,
                    contract.diagnostic_acceptance,
                    model_config,
                    constraint_profile,
                    Environment(),
                    physics_runtime=physics_runtime,
                    constraint_dominance_contract=dominance_contract,
                )
                true_evaluations += 1
                prior = current_results[seed_index]
                prior_error = float(prior["total_true_error"])
                proposed_error = (
                    float(proposed["total_true_error"])
                    if proposed["physics_acceptable"] and proposed["total_true_error"] is not None
                    else math.inf
                )
                prior_feasible, prior_violation = constraint_state(prior)
                proposed_feasible, proposed_violation = constraint_state(proposed)
                prior_key = (
                    *evaluation_hit_key(prior),
                    not prior_feasible,
                    prior_violation,
                )
                proposed_key = (
                    *evaluation_hit_key(proposed),
                    not proposed_feasible,
                    proposed_violation,
                )
                accepted = (
                    proposed_error < prior_error
                    if true_acceptance_mode == "legacy_mean"
                    else proposed_key < prior_key
                )
                if accepted:
                    current_results[seed_index] = proposed
                    next_current[seed_index] = torch.as_tensor(
                        proposed_normalized, dtype=dtype, device=torch_device
                    )
                ledger.append(
                    {
                        "seed_index": seed_index,
                        "seed_evaluation_id": seed["evaluation_id"],
                        "candidate_id": int(seed["candidate_id"]),
                        "checkpoint": checkpoint,
                        "surrogate_objective": surrogate_loss,
                        "prior_evaluation_id": prior["evaluation_id"],
                        "prior_total_true_error": prior_error,
                        "prior_hit_alignment": prior.get("hit_alignment"),
                        "proposed": proposed,
                        "accepted": accepted,
                        "acceptance_reason": (
                            (
                                "LEGACY_MEAN_RELATIVE_ERROR_IMPROVEMENT"
                                if true_acceptance_mode == "legacy_mean"
                                else "HIT_ALIGNED_LEXICOGRAPHIC_IMPROVEMENT"
                            )
                            if accepted
                            else (
                            "TRUE_PHYSICS_INVALID"
                            if not math.isfinite(proposed_error)
                            else (
                                "NOT_LEGACY_MEAN_RELATIVE_ERROR_BETTER"
                                if true_acceptance_mode == "legacy_mean"
                                else "NOT_HIT_ALIGNED_BETTER"
                            )
                            )
                        ),
                    }
                )
            current = next_current
            previous_step = checkpoint
            if any(row["development_diagnostic_hit"] for row in current_results):
                stopped_at = checkpoint
                break
        hits = [row for row in current_results if row["development_diagnostic_hit"]]
        valid = [row for row in current_results if row["physics_acceptable"]]
        valid.sort(key=candidate_sort_key)
        delivered = rank_delivery_candidates(
            hits, int(protocol["delivery"]["maximum_candidates"]), fields
        )
        proposed_rows = [row["proposed"] for row in ledger]
        dispositions = build_disposition_rows(
            proposed_rows,
            delivered,
            stage="WP19-23D",
            next_stage=None if hits else "WP19-23E",
            contract=dominance_contract,
        )
        attach_dispositions(proposed_rows, dispositions)
        attach_dispositions(delivered, dispositions)
        summary = {
            "work_package": "WP19-23D",
            "status": "REFINEMENT_DIAGNOSTIC_HIT" if hits else "NO_REFINEMENT_HIT",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_id": target.target_id,
            "formal_use_allowed": False,
            "formal_hit": None,
            "development_overrides": {
                "target_loss_mode": target_loss_mode,
                "true_acceptance_mode": true_acceptance_mode,
                "forward_model_tag": models.forward_spec.logical_tag,
                "forward_weights": (
                    str(Path(forward_weights_override).resolve())
                    if forward_weights_override is not None
                    else None
                ),
            },
            "downstream_stop": bool(hits),
            "next_stage": "DELIVERY" if hits else "WP19-23E_WARM_START_NSGA2",
            "stopped_at_checkpoint": stopped_at,
            "counts": {
                "seed_count": len(seeds),
                "true_v2_1_evaluations": true_evaluations,
                "true_physics_evaluations": true_evaluations,
                "strictly_accepted_updates": sum(bool(row["accepted"]) for row in ledger),
                "hit_aligned_accepted_updates": sum(
                    bool(row["accepted"]) for row in ledger
                    if true_acceptance_mode == "hit_aligned"
                ),
                "diagnostic_hits": len(hits),
                "delivered_count": len(delivered),
                "constraint_feasible": sum(
                    bool((row.get("constraint_dominance") or {}).get("constraint_feasible"))
                    for row in proposed_rows
                ),
            },
            "seed_evaluation_ids": [row["evaluation_id"] for row in seeds],
            "best_approximation": valid[0] if valid else None,
            "delivered_evaluation_ids": [row["evaluation_id"] for row in delivered],
            "disposition_counts": {
                reason: sum(row["disposition_reason"] == reason for row in dispositions)
                for reason in sorted({row["disposition_reason"] for row in dispositions})
            },
            "hashes": {
                "protocol_hash": sha256_file(contract_path),
                "target_hash": sha256_file(target_source),
                "schema_hash": sha256_file(schema_path),
                "calibration_hash": model_config.calibration_pack_hash,
                "physics_backend_id": physics_runtime.backend_id,
                "physics_bundle_hash": physics_runtime.physics_bundle_hash,
                "constraint_profile_hash": constraint_profile.source_hash,
                "constraint_dominance_hash": dominance_contract.source_hash,
                "wp19_23c_summary_hash": sha256_file(physics_summary_path),
                "forward_manifest_hash": sha256_file(
                    forward_dir / "model_manifest.json"
                ),
                "feasibility_manifest_hash": (
                    sha256_file(
                        root
                        / protocol["models"]["feasibility"]["weights"]
                        / "model_manifest.json"
                    )
                    if models.forward_spec.use_feasibility_loss
                    else None
                ),
                "code_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
            },
        }
        _write_json(runtime / "target.json", target.as_mapping())
        _write_jsonl(runtime / "refinement_ledger.jsonl", ledger)
        _write_jsonl(runtime / "candidate_disposition_ledger.jsonl", dispositions)
        _write_json(runtime / "delivered_candidates.json", delivered)
        _write_json(runtime / "summary.json", summary)
        _write_json(
            runtime / "manifest.json",
            {
                "protocol_version": protocol["protocol_version"],
                "target_id": target.target_id,
                "stage": "WP19-23D",
                "required_files": [
                    "manifest.json",
                    "target.json",
                    "refinement_ledger.jsonl",
                    "candidate_disposition_ledger.jsonl",
                    "delivered_candidates.json",
                    "summary.json",
                ],
                "summary_sha256": sha256_file(runtime / "summary.json"),
            },
        )
        return summary
    except Exception as exc:
        _write_json(
            runtime / "failure.json",
            {"status": "FAILED", "error_type": type(exc).__name__, "message": str(exc)},
        )
        raise
