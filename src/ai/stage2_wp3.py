"""Stage-II WP3 forward-consistency screening and latent-space repair."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from .baseline import (
    collect_environment,
    collect_git_state,
    collect_source_manifest,
    sha256_file,
    write_json,
    write_yaml,
)
from .generator_dataset import GeneratorNormalizer, condition_names, design_names, load_schema
from .generators import load_generator
from .round_trip import RoundTripEvaluator, derive_round_trip_metrics, load_metrics_config
from .wp4 import _target_summary_row


ORDINARY_TARGET_TYPES = ("ID", "Interpolation")
EXPERIMENT_METHODS = ("baseline", "screening", "repair")


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _seed(base_seed: int, *parts: Any) -> int:
    payload = ":".join([str(base_seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for S2-WP3 but is unavailable")
    return device


def load_stage2_wp3_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "version",
        "data",
        "surrogate",
        "model_sets",
        "experiments",
        "runtime",
        "output",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"S2-WP3 config is missing keys: {sorted(missing)}")
    if set(config["experiments"]) != set(EXPERIMENT_METHODS):
        raise ValueError(
            f"experiments must define exactly {list(EXPERIMENT_METHODS)}"
        )
    experiment_ids = [
        str(config["experiments"][name]["experiment_id"])
        for name in EXPERIMENT_METHODS
    ]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("Every S2-WP3 arm needs a unique experiment_id")
    runtime = config["runtime"]
    pool_size = int(runtime["candidate_pool_size"])
    evaluation_count = int(runtime["physical_evaluation_count"])
    if pool_size < evaluation_count or evaluation_count <= 0:
        raise ValueError(
            "candidate_pool_size must be >= positive physical_evaluation_count"
        )
    maximum_targets = runtime.get("maximum_targets")
    if maximum_targets is not None and int(maximum_targets) <= 0:
        raise ValueError("runtime.maximum_targets must be null or positive")
    repair = config["experiments"]["repair"]
    if int(repair["steps"]) <= 0 or float(repair["learning_rate"]) <= 0.0:
        raise ValueError("repair steps and learning_rate must be positive")
    if not config["model_sets"]:
        raise ValueError("model_sets cannot be empty")
    allowed_models = {"nflows", "cinn"}
    for model_set in config["model_sets"]:
        model_ids = [str(item["model_id"]) for item in model_set["models"]]
        unknown = set(model_ids).difference(allowed_models)
        if unknown:
            raise ValueError(f"S2-WP3 only supports nflows/cinn: {sorted(unknown)}")
        if not model_ids or len(model_ids) != len(set(model_ids)):
            raise ValueError("Each model_set needs unique nflows/cinn models")
    return config


class ForwardConsistencySurrogate(nn.Module):
    """Differentiable normalized-design to normalized-condition map."""

    def __init__(
        self,
        design_dim: int,
        condition_dim: int,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = int(design_dim)
        for width_value in hidden_dims:
            width = int(width_value)
            if width <= 0:
                raise ValueError("surrogate hidden dimensions must be positive")
            layers.extend([nn.Linear(previous, width), nn.SiLU()])
            previous = width
        layers.append(nn.Linear(previous, int(condition_dim)))
        self.network = nn.Sequential(*layers)
        self.design_dim = int(design_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dims = [int(value) for value in hidden_dims]

    def forward(self, normalized_design: torch.Tensor) -> torch.Tensor:
        return self.network(normalized_design)


def consistency_scores(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Per-candidate weighted error in normalized condition space."""

    if predicted.ndim != 2:
        raise ValueError("predicted conditions must have shape [n,condition_dim]")
    value = torch.as_tensor(target, dtype=predicted.dtype, device=predicted.device)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.shape[0] == 1 and predicted.shape[0] > 1:
        value = value.expand(predicted.shape[0], -1)
    if value.shape != predicted.shape:
        raise ValueError("predicted and target condition shapes differ")
    if weights is None:
        weight = torch.ones(predicted.shape[1], device=predicted.device)
    else:
        weight = torch.as_tensor(
            weights, dtype=predicted.dtype, device=predicted.device
        )
    if weight.shape != (predicted.shape[1],) or torch.any(weight <= 0.0):
        raise ValueError("consistency weights must be positive and match conditions")
    weight = weight / weight.sum()
    return ((predicted - value).square() * weight).sum(dim=1)


def select_by_consistency(scores: np.ndarray, count: int) -> np.ndarray:
    """Stable ranking; non-finite surrogate scores are explicit worst candidates."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError("consistency scores must be a vector")
    if count <= 0 or count > len(values):
        raise ValueError("selection count is outside the score vector")
    ranked = np.where(np.isfinite(values), values, np.inf)
    return np.lexsort((np.arange(len(ranked)), ranked))[: int(count)]


def _normalized_arrays(
    frame: pd.DataFrame,
    normalizer: GeneratorNormalizer,
) -> tuple[np.ndarray, np.ndarray]:
    x = normalizer.transform_design(frame[normalizer.design_names].to_numpy(float))
    condition = normalizer.transform_conditions(
        frame[normalizer.condition_names].to_numpy(float)
    )
    return np.asarray(x, dtype=np.float32), np.asarray(condition, dtype=np.float32)


def _surrogate_metrics(
    model: ForwardConsistencySurrogate,
    frame: pd.DataFrame,
    normalizer: GeneratorNormalizer,
    device: torch.device,
) -> Dict[str, Any]:
    x, condition = _normalized_arrays(frame, normalizer)
    with torch.no_grad():
        predicted = model(torch.from_numpy(x).to(device)).cpu().numpy()
    residual = predicted - condition
    predicted_physical = normalizer.inverse_conditions(predicted)
    actual_physical = frame[normalizer.condition_names].to_numpy(float)
    relative = np.abs(predicted_physical - actual_physical) / np.maximum(
        np.abs(actual_physical), 1.0e-15
    )
    return {
        "sample_count": int(len(frame)),
        "normalized_rmse": float(np.sqrt(np.mean(residual**2))),
        "normalized_mae": float(np.mean(np.abs(residual))),
        "physical_relative_error_median": {
            name: float(np.median(relative[:, index]))
            for index, name in enumerate(normalizer.condition_names)
        },
        "physical_relative_error_p95": {
            name: float(np.quantile(relative[:, index], 0.95))
            for index, name in enumerate(normalizer.condition_names)
        },
    }


def train_forward_surrogate(
    dataset_path: Path,
    normalizer: GeneratorNormalizer,
    config: Mapping[str, Any],
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[ForwardConsistencySurrogate, Dict[str, Any]]:
    """Train once, select on validation only, then freeze a traceable checkpoint."""

    frame = pd.read_csv(dataset_path)
    required = set(normalizer.design_names + normalizer.condition_names + ["split"])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Surrogate dataset is missing fields: {sorted(missing)}")
    splits = {
        name: frame.loc[frame["split"].astype(str) == name].reset_index(drop=True)
        for name in ("train", "val", "test")
    }
    if splits["train"].empty or splits["val"].empty:
        raise ValueError("Surrogate training requires non-empty train and val splits")
    seed = int(config.get("seed", 7301))
    _set_seed(seed)
    train_x, train_y = _normalized_arrays(splits["train"], normalizer)
    val_x, val_y = _normalized_arrays(splits["val"], normalizer)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=int(config.get("batch_size", 1024)),
        shuffle=True,
        generator=generator,
        num_workers=int(config.get("num_workers", 0)),
    )
    model = ForwardConsistencySurrogate(
        design_dim=len(normalizer.design_names),
        condition_dim=len(normalizer.condition_names),
        hidden_dims=config.get("hidden_dims", [256, 256, 128]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1.0e-3)),
        weight_decay=float(config.get("weight_decay", 1.0e-5)),
    )
    val_x_tensor = torch.from_numpy(val_x).to(device)
    val_y_tensor = torch.from_numpy(val_y).to(device)
    best_loss = math.inf
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stale = 0
    history = []
    epochs = int(config.get("epochs", 200))
    patience = int(config.get("patience", 20))
    min_delta = float(config.get("min_delta", 1.0e-6))
    for epoch in range(1, epochs + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x_batch), y_batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Forward surrogate produced non-finite loss")
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach()) * len(x_batch)
            train_count += len(x_batch)
        model.eval()
        with torch.no_grad():
            val_loss = float(nn.functional.mse_loss(model(val_x_tensor), val_y_tensor))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_sum / train_count,
                "val_loss": val_loss,
            }
        )
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("No forward-surrogate checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    metrics = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "history": history,
        "splits": {
            name: _surrogate_metrics(model, split, normalizer, device)
            for name, split in splits.items()
            if not split.empty
        },
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": "active10d_forward_consistency_v1",
            "model_state_dict": best_state,
            "model_config": {
                "design_dim": model.design_dim,
                "condition_dim": model.condition_dim,
                "hidden_dims": model.hidden_dims,
            },
            "training_config": dict(config),
            "normalizer": normalizer.to_dict(),
            "dataset_sha256": sha256_file(dataset_path),
            "metrics": metrics,
        },
        checkpoint_path,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, metrics


def load_forward_surrogate(
    checkpoint_path: Path,
    normalizer: GeneratorNormalizer,
    device: torch.device,
) -> tuple[ForwardConsistencySurrogate, Dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("normalizer") != normalizer.to_dict():
        raise ValueError("Forward-surrogate normalizer differs from S2-WP3 normalizer")
    model = ForwardConsistencySurrogate(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, dict(payload.get("metrics", {}))


def repair_latents(
    generator: Any,
    surrogate: ForwardConsistencySurrogate,
    initial_latent: torch.Tensor,
    normalized_condition: torch.Tensor,
    config: Mapping[str, Any],
    weights: Optional[Sequence[float]] = None,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Optimize only base variables; generator and forward model remain frozen."""

    if not bool(generator.supports_latent_repair):
        raise ValueError(f"{generator.generator_name} has no repairable latent space")
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    surrogate.eval()
    generator.eval()
    initial = initial_latent.detach()
    if initial.ndim != 3 or initial.shape[0] != 1:
        raise ValueError("S2-WP3 repair currently requires one condition per call")
    with torch.no_grad():
        initial_design = generator.decode_latent(initial, normalized_condition)
        initial_prediction = surrogate(
            initial_design.reshape(-1, generator.design_dim)
        )
        initial_score = consistency_scores(
            initial_prediction, normalized_condition, weights
        )
        finite_mask = (
            torch.isfinite(initial_design).all(dim=-1).reshape(-1)
            & torch.isfinite(initial_prediction).all(dim=-1)
            & torch.isfinite(initial_score)
        )
        maximum_initial_abs_design = float(
            config.get("maximum_initial_abs_design", 10.0)
        )
        finite_mask &= (
            initial_design.abs().amax(dim=-1).reshape(-1)
            <= maximum_initial_abs_design
        )
    if not bool(finite_mask.any()):
        return initial, initial_design, {
            "before_consistency_mean": math.nan,
            "after_consistency_mean": math.nan,
            "improved_fraction": 0.0,
            "optimized_candidate_count": 0,
            "skipped_nonfinite_count": int(initial.shape[1]),
            "latent_displacement_mean": 0.0,
            "latent_displacement_max": 0.0,
            "termination_reason": "no_numerically_safe_candidate",
            "history": [],
        }
    finite_indices = torch.nonzero(finite_mask, as_tuple=False).reshape(-1)
    optimized_initial = initial[:, finite_indices, :]
    latent = optimized_initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=float(config["learning_rate"]))
    condition = torch.as_tensor(
        normalized_condition, dtype=latent.dtype, device=latent.device
    )
    if condition.ndim == 1:
        condition = condition.unsqueeze(0)
    prior_weight = float(config.get("prior_weight", 0.01))
    boundary_weight = float(config.get("boundary_weight", 1.0))
    maximum_displacement = config.get("maximum_latent_displacement")
    gradient_clip_norm = float(config.get("gradient_clip_norm", 1.0))
    history = []
    with torch.no_grad():
        before_design = generator.decode_latent(optimized_initial, condition)
        before_prediction = surrogate(before_design.reshape(-1, generator.design_dim))
        before = consistency_scores(before_prediction, condition, weights)
    termination_reason = "steps_completed"
    last_finite_latent = latent.detach().clone()
    for step in range(1, int(config["steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        design = generator.decode_latent(latent, condition)
        prediction = surrogate(design.reshape(-1, generator.design_dim))
        consistency = consistency_scores(prediction, condition, weights)
        displacement = (latent - optimized_initial).square().mean(dim=-1).reshape(-1)
        boundary = torch.relu(design.abs() - 1.0).square().mean(dim=-1).reshape(-1)
        objective = consistency + prior_weight * displacement + boundary_weight * boundary
        loss = objective.mean()
        if not torch.isfinite(objective).all() or not torch.isfinite(loss):
            termination_reason = f"nonfinite_objective_at_step_{step}"
            latent = last_finite_latent.clone().requires_grad_(True)
            break
        last_finite_latent = latent.detach().clone()
        loss.backward()
        if latent.grad is None or not torch.isfinite(latent.grad).all():
            termination_reason = f"nonfinite_gradient_at_step_{step}"
            latent = last_finite_latent.clone().requires_grad_(True)
            break
        if gradient_clip_norm > 0.0:
            with torch.no_grad():
                gradient_norm = torch.linalg.vector_norm(
                    latent.grad, dim=-1, keepdim=True
                )
                scale = torch.clamp(
                    gradient_clip_norm / gradient_norm.clamp_min(1.0e-12),
                    max=1.0,
                )
                latent.grad.mul_(scale)
        optimizer.step()
        if maximum_displacement is not None:
            with torch.no_grad():
                delta = (latent - optimized_initial).clamp(
                    -float(maximum_displacement), float(maximum_displacement)
                )
                latent.copy_(optimized_initial + delta)
        if step == 1 or step == int(config["steps"]) or step % 8 == 0:
            history.append(
                {
                    "step": step,
                    "objective_mean": float(loss.detach()),
                    "consistency_mean": float(consistency.mean().detach()),
                    "boundary_penalty_mean": float(boundary.mean().detach()),
                }
            )
    with torch.no_grad():
        repaired_design = generator.decode_latent(latent, condition)
        prediction = surrogate(repaired_design.reshape(-1, generator.design_dim))
        after = consistency_scores(prediction, condition, weights)
        if (
            not torch.isfinite(repaired_design).all()
            or not torch.isfinite(prediction).all()
            or not torch.isfinite(after).all()
        ):
            termination_reason = "nonfinite_final_state_reverted"
            latent = last_finite_latent
            repaired_design = generator.decode_latent(latent, condition)
            prediction = surrogate(repaired_design.reshape(-1, generator.design_dim))
            after = consistency_scores(prediction, condition, weights)
        displacement = torch.linalg.vector_norm(latent - optimized_initial, dim=-1)
        repaired_latent = initial.clone()
        repaired_design_full = initial_design.clone()
        repaired_latent[:, finite_indices, :] = latent.detach()
        repaired_design_full[:, finite_indices, :] = repaired_design.detach()
    diagnostics = {
        "before_consistency_mean": float(before.mean()),
        "after_consistency_mean": float(after.mean()),
        "improved_fraction": float((after < before).float().mean()),
        "optimized_candidate_count": int(finite_mask.sum()),
        "skipped_nonfinite_count": int((~finite_mask).sum()),
        "latent_displacement_mean": float(displacement.mean()),
        "latent_displacement_max": float(displacement.max()),
        "termination_reason": termination_reason,
        "history": history,
    }
    return repaired_latent, repaired_design_full, diagnostics


def _load_targets(repo_root: Path, data_config: Mapping[str, Any]) -> pd.DataFrame:
    suite_path = _resolve(repo_root, str(data_config["target_suite"]))
    frozen_path = _resolve(repo_root, str(data_config["frozen_targets"]))
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    target_ids = [str(value) for value in frozen["target_ids"]]
    suite = pd.read_csv(suite_path)
    indexed = suite.assign(target_id=suite["target_id"].astype(str)).set_index(
        "target_id", drop=False
    )
    missing = [target_id for target_id in target_ids if target_id not in indexed.index]
    if missing:
        raise ValueError(f"Frozen S2-WP3 targets are absent from suite: {missing[:3]}")
    targets = indexed.loc[target_ids].reset_index(drop=True)
    if not targets["target_type"].isin(ORDINARY_TARGET_TYPES).all():
        raise ValueError("S2-WP3 frozen targets must be ID or Interpolation")
    return targets


def _candidate_ledger(
    result: Any,
    normalized: np.ndarray,
    predicted_score: np.ndarray,
    context: Mapping[str, Any],
) -> pd.DataFrame:
    frame = result.forward_results.copy()
    score_by_id = {
        int(index): float(score) for index, score in enumerate(predicted_score)
    }
    frame["surrogate_consistency"] = frame["candidate_id"].map(score_by_id)
    for key, value in reversed(list(context.items())):
        frame.insert(0, key, value)
    return frame


def _arm_summary(
    result: Any,
    predicted_score: np.ndarray,
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    row = _target_summary_row(
        str(context["model"]),
        context["target"],
        int(context["sample_seed"]),
        result,
    )
    forward = result.forward_results
    hits = forward.loc[forward["within_tolerance"].astype(bool), "candidate_id"]
    finite_scores = np.asarray(predicted_score, dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    row.update(
        {
            "training_seed": int(context["training_seed"]),
            "method": str(context["method"]),
            "experiment_id": str(context["experiment_id"]),
            "checkpoint_sha256": str(context["checkpoint_sha256"]),
            "evaluations_to_first_hit": int(hits.min()) + 1 if len(hits) else math.nan,
            "surrogate_consistency_mean": (
                float(np.mean(finite_scores)) if len(finite_scores) else math.nan
            ),
            "surrogate_consistency_min": (
                float(np.min(finite_scores)) if len(finite_scores) else math.nan
            ),
            "surrogate_nonfinite_count": int(
                len(predicted_score) - len(finite_scores)
            ),
        }
    )
    return row


def aggregate_wp3_results(target_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (model, method, target_type), frame in target_summary.groupby(
        ["model", "method", "target_type"], sort=True
    ):
        candidate_count = pd.to_numeric(frame["candidate_count"], errors="raise")
        forward_valid = pd.to_numeric(frame["forward_valid_count"], errors="raise")
        errors = pd.to_numeric(frame["best_total_error"], errors="coerce")
        rows.append(
            {
                "model": model,
                "method": method,
                "target_type": target_type,
                "run_count": int(len(frame)),
                "best_total_error_median": float(errors.median()),
                "best_total_error_mean": float(errors.mean()),
                "within_tolerance_rate": float(frame["best_within_tolerance"].mean()),
                "forward_valid_candidate_rate": float((forward_valid / candidate_count).mean()),
                "evaluations_to_first_hit_median": float(
                    pd.to_numeric(frame["evaluations_to_first_hit"], errors="coerce").median()
                ),
                "diversity_mean": float(
                    pd.to_numeric(
                        frame["mean_nearest_neighbour_distance"], errors="coerce"
                    ).mean()
                ),
            }
        )
    summary = pd.DataFrame(rows)

    keys = ["model", "training_seed", "target_id", "target_type"]
    baseline = target_summary.loc[target_summary["method"] == "baseline"].copy()
    baseline = baseline[keys + [
        "best_total_error",
        "best_within_tolerance",
        "forward_valid_count",
        "candidate_count",
        "mean_nearest_neighbour_distance",
    ]].rename(columns={
        "best_total_error": "baseline_best_total_error",
        "best_within_tolerance": "baseline_within_tolerance",
        "forward_valid_count": "baseline_forward_valid_count",
        "candidate_count": "baseline_candidate_count",
        "mean_nearest_neighbour_distance": "baseline_diversity",
    })
    changed = target_summary.loc[target_summary["method"] != "baseline"].merge(
        baseline, on=keys, how="left", validate="many_to_one"
    )
    changed["best_error_improvement"] = (
        changed["baseline_best_total_error"] - changed["best_total_error"]
    )
    changed["best_error_win"] = changed["best_error_improvement"] > 0.0
    changed["valid_rate_improvement"] = (
        changed["forward_valid_count"] / changed["candidate_count"]
        - changed["baseline_forward_valid_count"] / changed["baseline_candidate_count"]
    )
    changed["diversity_ratio"] = (
        changed["mean_nearest_neighbour_distance"] / changed["baseline_diversity"]
    )
    paired_rows = []
    for (model, method, target_type), frame in changed.groupby(
        ["model", "method", "target_type"], sort=True
    ):
        paired_rows.append(
            {
                "model": model,
                "method": method,
                "target_type": target_type,
                "pair_count": int(len(frame)),
                "best_error_win_rate": float(frame["best_error_win"].mean()),
                "best_error_improvement_median": float(
                    frame["best_error_improvement"].median()
                ),
                "within_tolerance_rate_change": float(
                    frame["best_within_tolerance"].mean()
                    - frame["baseline_within_tolerance"].mean()
                ),
                "valid_rate_improvement_mean": float(
                    frame["valid_rate_improvement"].mean()
                ),
                "diversity_ratio_mean": float(frame["diversity_ratio"].mean()),
            }
        )
    return summary, pd.DataFrame(paired_rows)


def _write_report(
    path: Path,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    surrogate_metrics: Mapping[str, Any],
) -> None:
    test_metrics = surrogate_metrics.get("splits", {}).get("test", {})
    lines = [
        "# S2-WP3 nflows/cINN目标一致性与局部修正报告",
        "",
        "## Forward surrogate",
        "",
        f"- Test normalized RMSE: {test_metrics.get('normalized_rmse', math.nan):.6g}",
        f"- Test normalized MAE: {test_metrics.get('normalized_mae', math.nan):.6g}",
        "- Surrogate and generator checkpoints remained frozen during every arm.",
        "",
        "## Ablation summary",
        "",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"- {row['model']} / {row['method']} / {row['target_type']}: "
            f"runs={row['run_count']}, median_J={row['best_total_error_median']:.6g}, "
            f"hit_rate={row['within_tolerance_rate']:.4f}, "
            f"valid_rate={row['forward_valid_candidate_rate']:.4f}"
        )
    lines.extend(["", "## Paired against frozen baseline", ""])
    for row in paired.to_dict(orient="records"):
        lines.append(
            f"- {row['model']} / {row['method']} / {row['target_type']}: "
            f"win_rate={row['best_error_win_rate']:.4f}, "
            f"median_delta_J={row['best_error_improvement_median']:.6g}, "
            f"valid_rate_delta={row['valid_rate_improvement_mean']:.6g}, "
            f"diversity_ratio={row['diversity_ratio_mean']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Attribution",
            "",
            "`screening` only ranks the shared raw candidate pool. `repair` starts "
            "from the same baseline candidates and optimizes base variables without "
            "using the screening selection. The two changes therefore remain "
            "separately attributable.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage2_wp3(
    config_path: Path,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    config_path = Path(config_path).resolve()
    config = load_stage2_wp3_config(config_path)
    output_dir = (
        output_dir_override.resolve()
        if output_dir_override is not None
        else _resolve(repo_root, str(config["output"]["directory"]))
    )
    if output_dir.exists() and bool(config["output"].get("refuse_overwrite", True)):
        raise FileExistsError(f"S2-WP3 output exists and will not be overwritten: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = dt.datetime.now(dt.timezone.utc)
    shutil.copy2(config_path, output_dir / "build_config.yaml")

    data_config = config["data"]
    dataset_path = _resolve(repo_root, str(data_config["dataset_csv"]))
    schema_path = _resolve(repo_root, str(data_config["schema"]))
    normalizer_path = _resolve(repo_root, str(data_config["normalizer"]))
    metrics_path = _resolve(repo_root, str(data_config["metrics_config"]))
    target_suite_path = _resolve(repo_root, str(data_config["target_suite"]))
    frozen_targets_path = _resolve(repo_root, str(data_config["frozen_targets"]))
    for path in (
        dataset_path,
        schema_path,
        normalizer_path,
        metrics_path,
        target_suite_path,
        frozen_targets_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"S2-WP3 input does not exist: {path}")

    device = _resolve_device(str(config["runtime"].get("device", "auto")))
    schema = load_schema(schema_path)
    normalizer = GeneratorNormalizer.load(normalizer_path)
    if design_names(schema) != normalizer.design_names:
        raise ValueError("S2-WP3 schema and normalizer design fields differ")
    if condition_names(schema) != normalizer.condition_names:
        raise ValueError("S2-WP3 schema and normalizer condition fields differ")
    metrics_config = load_metrics_config(metrics_path)
    derived = derive_round_trip_metrics(dataset_path, normalizer, metrics_config)
    evaluator = RoundTripEvaluator(schema, normalizer, metrics_config, derived)
    targets = _load_targets(repo_root, data_config)
    maximum_targets = config["runtime"].get("maximum_targets")
    if maximum_targets is not None:
        targets = targets.head(int(maximum_targets)).reset_index(drop=True)
    targets.to_csv(output_dir / "evaluated_targets.csv", index=False)

    surrogate_checkpoint = output_dir / "forward_surrogate.pt"
    surrogate, surrogate_metrics = train_forward_surrogate(
        dataset_path,
        normalizer,
        config["surrogate"],
        device,
        surrogate_checkpoint,
    )
    write_json(output_dir / "forward_surrogate_metrics.json", surrogate_metrics)
    surrogate_sha256 = sha256_file(surrogate_checkpoint)

    pool_size = int(config["runtime"]["candidate_pool_size"])
    evaluation_count = int(config["runtime"]["physical_evaluation_count"])
    base_seed = int(config["runtime"].get("sample_seed_base", 20260812))
    weights = config["surrogate"].get("condition_weights")
    target_rows = []
    candidate_frames = []
    repair_records = []
    checkpoint_records = []

    for model_set in config["model_sets"]:
        training_seed = int(model_set["training_seed"])
        for model_entry in model_set["models"]:
            model_name = str(model_entry["model_id"])
            checkpoint = _resolve(repo_root, str(model_entry["checkpoint_path"]))
            if not checkpoint.is_file():
                raise FileNotFoundError(f"S2-WP3 checkpoint is missing: {checkpoint}")
            checkpoint_sha256 = sha256_file(checkpoint)
            checkpoint_records.append(
                {
                    "model": model_name,
                    "training_seed": training_seed,
                    "path": str(checkpoint),
                    "sha256": checkpoint_sha256,
                }
            )
            generator = load_generator(checkpoint, device=str(device))
            if generator.generator_name != model_name or not generator.supports_latent_repair:
                raise ValueError(f"Checkpoint does not provide repairable {model_name}")
            if generator.normalizer is None or generator.normalizer.to_dict() != normalizer.to_dict():
                raise ValueError(f"{model_name} checkpoint normalizer differs")
            for parameter in generator.parameters():
                parameter.requires_grad_(False)
            generator.eval()

            for target in targets.to_dict(orient="records"):
                target_id = str(target["target_id"])
                normalized_condition_np = evaluator.normalize_target(target)
                normalized_condition = torch.as_tensor(
                    normalized_condition_np, dtype=torch.float32, device=device
                ).unsqueeze(0)
                sample_seed = _seed(base_seed, model_name, training_seed, target_id)
                latent = generator.sample_latent(
                    normalized_condition, pool_size, seed=sample_seed
                )
                with torch.no_grad():
                    raw_design = generator.decode_latent(latent, normalized_condition)
                    raw_prediction = surrogate(
                        raw_design.reshape(-1, generator.design_dim)
                    )
                    raw_scores = consistency_scores(
                        raw_prediction, normalized_condition, weights
                    ).cpu().numpy()

                baseline_indices = np.arange(evaluation_count)
                screening_indices = select_by_consistency(raw_scores, evaluation_count)
                repaired_latent, repaired_design, repair_diagnostics = repair_latents(
                    generator,
                    surrogate,
                    latent[:, :evaluation_count, :],
                    normalized_condition,
                    config["experiments"]["repair"],
                    weights,
                )
                del repaired_latent
                with torch.no_grad():
                    repaired_prediction = surrogate(
                        repaired_design.reshape(-1, generator.design_dim)
                    )
                    repaired_scores = consistency_scores(
                        repaired_prediction, normalized_condition, weights
                    ).cpu().numpy()
                repair_records.append(
                    {
                        "model": model_name,
                        "training_seed": training_seed,
                        "target_id": target_id,
                        "target_type": target["target_type"],
                        **{key: value for key, value in repair_diagnostics.items() if key != "history"},
                    }
                )

                arms = {
                    "baseline": (
                        raw_design[0, baseline_indices].detach().cpu().numpy(),
                        raw_scores[baseline_indices],
                    ),
                    "screening": (
                        raw_design[0, screening_indices].detach().cpu().numpy(),
                        raw_scores[screening_indices],
                    ),
                    "repair": (
                        repaired_design[0].detach().cpu().numpy(),
                        repaired_scores,
                    ),
                }
                for method, (normalized, scores) in arms.items():
                    result = evaluator.evaluate(
                        target,
                        normalized,
                        normalizer.inverse_design(normalized),
                    )
                    experiment_id = str(
                        config["experiments"][method]["experiment_id"]
                    )
                    context = {
                        "model": model_name,
                        "training_seed": training_seed,
                        "method": method,
                        "experiment_id": experiment_id,
                        "sample_seed": sample_seed,
                        "checkpoint_sha256": checkpoint_sha256,
                        "target": target,
                    }
                    target_rows.append(_arm_summary(result, scores, context))
                    ledger_context = {
                        "model": model_name,
                        "training_seed": training_seed,
                        "method": method,
                        "experiment_id": experiment_id,
                        "target_id": target_id,
                        "target_type": target["target_type"],
                        "sample_seed": sample_seed,
                    }
                    candidate_frames.append(
                        _candidate_ledger(result, normalized, scores, ledger_context)
                    )

    target_summary = pd.DataFrame(target_rows)
    candidate_ledger = pd.concat(candidate_frames, ignore_index=True)
    repair_summary = pd.DataFrame(repair_records)
    summary, paired = aggregate_wp3_results(target_summary)
    expected_runs = (
        sum(len(value["models"]) for value in config["model_sets"])
        * len(targets)
        * len(EXPERIMENT_METHODS)
    )
    if len(target_summary) != expected_runs:
        raise RuntimeError(
            f"S2-WP3 produced {len(target_summary)} arms, expected {expected_runs}"
        )
    if set(target_summary["candidate_count"].astype(int)) != {evaluation_count}:
        raise RuntimeError("S2-WP3 arms used different physical evaluation budgets")
    if any(
        sha256_file(Path(record["path"])) != record["sha256"]
        for record in checkpoint_records
    ):
        raise RuntimeError("A frozen generator checkpoint changed during S2-WP3")

    target_summary.to_csv(output_dir / "target_summary.csv", index=False)
    candidate_ledger.to_csv(output_dir / "candidate_ledger.csv", index=False)
    repair_summary.to_csv(output_dir / "repair_diagnostics.csv", index=False)
    summary.to_csv(output_dir / "ablation_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_vs_baseline.csv", index=False)
    pd.DataFrame(checkpoint_records).to_csv(
        output_dir / "checkpoint_manifest.csv", index=False
    )
    _write_report(output_dir / "report.md", summary, paired, surrogate_metrics)

    completed = dt.datetime.now(dt.timezone.utc)
    gate = {
        "passed": True,
        "expected_arm_count": expected_runs,
        "actual_arm_count": len(target_summary),
        "equal_physical_evaluation_budget": True,
        "generator_checkpoints_unchanged": True,
        "surrogate_checkpoint_sha256": surrogate_sha256,
        "separate_experiment_ids": {
            name: config["experiments"][name]["experiment_id"]
            for name in EXPERIMENT_METHODS
        },
        "performance_superiority_required_for_functional_gate": False,
    }
    write_json(output_dir / "functional_gate.json", gate)
    manifest = {
        "version": config["version"],
        "started_utc": started.isoformat(),
        "completed_utc": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "device": str(device),
        "inputs": {
            str(path.relative_to(repo_root)): sha256_file(path)
            for path in (
                config_path,
                dataset_path,
                schema_path,
                normalizer_path,
                metrics_path,
                target_suite_path,
                frozen_targets_path,
            )
        },
        "checkpoints": checkpoint_records,
        "surrogate_checkpoint_sha256": surrogate_sha256,
        "gate": gate,
        "git": collect_git_state(repo_root, output_dir),
        "environment": collect_environment(),
    }
    write_yaml(output_dir / "manifest.yaml", manifest)
    write_json(
        output_dir / "source_manifest.json",
        collect_source_manifest(repo_root),
    )
    return output_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output = run_stage2_wp3(
        args.config,
        repo_root=args.repo_root,
        output_dir_override=args.output_dir,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
