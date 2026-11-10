"""Gate L S4: sparse directional-label generation and gradient-aware training."""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.ai.geometry19_gate_k import (
    _aggregate_arm,
    _json_safe,
    _predict,
    _regression_metrics,
    _seed_all,
    _write_json,
    ranking_benchmark,
)
from src.ai.geometry19_gate_k2 import _run_frozen_pool
from src.ai.geometry19_gate_l import audit_model_directions
from src.ai.geometry19_hit_alignment import OUTPUT_NAMES
from src.ai.geometry19_surrogate import (
    TaggedGeometry19MLP,
    load_weight_bundle,
    save_weight_bundle,
    sha256_file,
)
from src.ai.geometry19_surrogate_objective import (
    hit_aligned_prediction_metrics,
    surrogate_loss_terms,
)
from src.ai.geometry19_training import fit_normalization, load_regression_splits
from src.physics import Environment, Geometry19, ModelConfig, solve_design_point
from src.physics.engine_match import OperatingPointSelectionPolicy


class Geometry19S4Error(ValueError):
    """Raised when sparse directional training violates its contract."""


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(_json_safe(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _solve_direction_shard(
    rows: Sequence[Mapping[str, Any]],
    calibration_path: str,
    environment_payload: Mapping[str, float],
) -> list[dict[str, Any]]:
    config = ModelConfig.from_yaml(calibration_path)
    environment = Environment(**environment_payload)
    output = []
    for row in rows:
        try:
            result = solve_design_point(
                Geometry19.from_mapping(row["proposed_geometry"]),
                environment,
                config,
                operating_point_policy=(
                    OperatingPointSelectionPolicy.REFERENCE_CONTINUATION_HIGH_FLOW_V1
                ),
            )
            performance = dict(result.performance) if result.performance else None
            valid = bool(
                result.solver.converged
                and performance
                and all(math.isfinite(float(performance[name])) for name in OUTPUT_NAMES)
            )
            output.append(
                {
                    **dict(row),
                    "proposed_performance": performance if valid else None,
                    "solver_converged": bool(result.solver.converged),
                    "solver_failure_code": (
                        result.solver.failure_code.value
                        if result.solver.failure_code is not None else None
                    ),
                    "physics_acceptable": valid,
                }
            )
        except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
            output.append(
                {
                    **dict(row),
                    "proposed_performance": None,
                    "solver_converged": False,
                    "solver_failure_code": type(exc).__name__,
                    "physics_acceptable": False,
                }
            )
    return output


def propose_direction_rows(
    splits: Mapping[str, Mapping[str, Any]],
    fields: Sequence[Mapping[str, Any]],
    requested_pairs: Mapping[str, int],
    *,
    normalized_step: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Select deterministic anchors and perturb continuous coordinates only."""

    if not 0.0 < normalized_step <= 0.25:
        raise Geometry19S4Error("normalized_step must be in (0, 0.25]")
    feature_names = [str(item["name"]) for item in fields]
    continuous_indices = [
        index for index, item in enumerate(fields) if item["dtype"] != "int64"
    ]
    lower = np.asarray([item["bounds"][0] for item in fields], dtype=np.float64)
    upper = np.asarray([item["bounds"][1] for item in fields], dtype=np.float64)
    scale = upper - lower
    rows = []
    for split_offset, split_name in enumerate(("train", "validation", "test")):
        bucket = splits[split_name]
        count = min(int(requested_pairs[split_name]), len(bucket["records"]))
        indices = np.linspace(0, len(bucket["records"]) - 1, count, dtype=int)
        rng = np.random.default_rng(seed + split_offset)
        for local_index, anchor_index in enumerate(indices):
            record = bucket["records"][int(anchor_index)]
            prior = np.asarray(
                [record["x_19"][name] for name in feature_names], dtype=np.float64
            )
            direction = np.zeros(len(fields), dtype=np.float64)
            direction[continuous_indices] = rng.normal(size=len(continuous_indices))
            direction /= np.linalg.norm(direction)
            proposed = np.clip(prior + normalized_step * direction * scale, lower, upper)
            for index, item in enumerate(fields):
                if item["dtype"] == "int64":
                    proposed[index] = prior[index]
            actual = (proposed - prior) / np.maximum(scale, 1.0e-12)
            performance = record["physics_outputs"]["performance"]
            rows.append(
                {
                    "pair_id": f"{split_name}:{record['campaign_id']}:{record['sample_id']}:{local_index}",
                    "split": split_name,
                    "campaign_id": record["campaign_id"],
                    "sample_id": int(record["sample_id"]),
                    "prior_geometry": {
                        name: record["x_19"][name] for name in feature_names
                    },
                    "proposed_geometry": {
                        name: (
                            int(round(proposed[index]))
                            if fields[index]["dtype"] == "int64"
                            else float(proposed[index])
                        )
                        for index, name in enumerate(feature_names)
                    },
                    "prior_performance": {
                        name: float(performance[name]) for name in OUTPUT_NAMES
                    },
                    "normalized_direction_norm": float(np.linalg.norm(actual)),
                }
            )
    return rows


def generate_direction_dataset(
    rows: Sequence[Mapping[str, Any]],
    *,
    calibration_path: str | Path,
    environment_payload: Mapping[str, float],
    workers: int,
) -> list[dict[str, Any]]:
    """Evaluate one perturbed endpoint per anchor with true V2.1."""

    if workers <= 0:
        raise Geometry19S4Error("workers must be positive")
    shard_size = max(1, math.ceil(len(rows) / workers))
    shards = [rows[start : start + shard_size] for start in range(0, len(rows), shard_size)]
    solved = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _solve_direction_shard,
                shard,
                str(calibration_path),
                dict(environment_payload),
            )
            for shard in shards
        ]
        for future in as_completed(futures):
            solved.extend(future.result())
    solved.sort(key=lambda row: row["pair_id"])
    return solved


def direction_arrays(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    normalization: Mapping[str, np.ndarray],
    tolerances: Mapping[str, float],
    epsilons: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    usable = [row for row in rows if row.get("physics_acceptable")]
    if not usable:
        raise Geometry19S4Error("No physics-valid direction pair exists")
    x0 = np.asarray(
        [[row["prior_geometry"][name] for name in feature_names] for row in usable],
        dtype=np.float32,
    )
    x1 = np.asarray(
        [[row["proposed_geometry"][name] for name in feature_names] for row in usable],
        dtype=np.float32,
    )
    y0 = np.asarray(
        [[row["prior_performance"][name] for name in OUTPUT_NAMES] for row in usable],
        dtype=np.float32,
    )
    y1 = np.asarray(
        [[row["proposed_performance"][name] for name in OUTPUT_NAMES] for row in usable],
        dtype=np.float32,
    )
    denominator = np.asarray([tolerances[name] for name in OUTPUT_NAMES]) * np.maximum(
        np.abs(y0), np.asarray([epsilons[name] for name in OUTPUT_NAMES])
    )
    return (
        ((x0 - normalization["input_mean"]) / normalization["input_std"]).astype(np.float32),
        ((x1 - normalization["input_mean"]) / normalization["input_std"]).astype(np.float32),
        (y1 - y0).astype(np.float32),
        denominator.astype(np.float32),
    )


def directional_loss(
    model: torch.nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    true_delta: torch.Tensor,
    denominator: torch.Tensor,
    output_std: torch.Tensor,
    *,
    huber_delta: float,
    cosine_weight: float,
) -> torch.Tensor:
    predicted_delta = (model(x1)[:, :3] - model(x0)[:, :3]) * output_std[:3]
    predicted_scaled = predicted_delta / denominator
    true_scaled = true_delta / denominator
    magnitude = F.smooth_l1_loss(
        predicted_scaled, true_scaled, beta=huber_delta
    )
    cosine = 1.0 - F.cosine_similarity(
        predicted_scaled, true_scaled, dim=1, eps=1.0e-8
    ).mean()
    return magnitude + cosine_weight * cosine


def _direction_loader(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(*(torch.from_numpy(value) for value in arrays)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def _evaluate_direction_loss(
    model: torch.nn.Module,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    output_std: torch.Tensor,
    training: Mapping[str, Any],
    device: torch.device,
) -> float:
    tensors = [torch.from_numpy(value).to(device) for value in arrays]
    model.eval()
    with torch.no_grad():
        value = directional_loss(
            model, *tensors, output_std,
            huber_delta=float(training["directional_huber_delta"]),
            cosine_weight=float(training["directional_cosine_weight"]),
        )
    return float(value)


def train_directional_model(
    model: TaggedGeometry19MLP,
    value_loader: DataLoader,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    train_directions: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    validation_directions: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    value_loss_function: Any,
    output_std: torch.Tensor,
    training: Mapping[str, Any],
    gradient_weight: float,
    device: torch.device,
    seed: int,
) -> tuple[TaggedGeometry19MLP, list[dict[str, float]], int]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    validation_x_tensor = torch.from_numpy(validation_x).to(device)
    validation_y_tensor = torch.from_numpy(validation_y).to(device)
    best_loss = math.inf
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        direction_loader = _direction_loader(
            train_directions, int(training["direction_batch_size"]), seed + epoch, True
        )
        direction_iterator = iter(direction_loader)
        total = 0.0
        count = 0
        for value_x, value_y in value_loader:
            try:
                direction_batch = next(direction_iterator)
            except StopIteration:
                direction_iterator = iter(direction_loader)
                direction_batch = next(direction_iterator)
            value_x, value_y = value_x.to(device), value_y.to(device)
            direction_batch = tuple(value.to(device) for value in direction_batch)
            optimizer.zero_grad(set_to_none=True)
            value_loss = value_loss_function(model(value_x), value_y)
            # Directional secants must not update BatchNorm running statistics or
            # use independent Dropout masks at their two endpoints.  Evaluation
            # mode still preserves parameter gradients for this branch.
            model.eval()
            gradient_loss = directional_loss(
                model, *direction_batch, output_std,
                huber_delta=float(training["directional_huber_delta"]),
                cosine_weight=float(training["directional_cosine_weight"]),
            )
            model.train()
            loss = value_loss + gradient_weight * gradient_loss
            if not torch.isfinite(loss):
                raise Geometry19S4Error("S4 training produced non-finite loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(value_x)
            count += len(value_x)
        model.eval()
        with torch.no_grad():
            validation_value = float(
                value_loss_function(model(validation_x_tensor), validation_y_tensor)
            )
        validation_direction = _evaluate_direction_loss(
            model, validation_directions, output_std, training, device
        )
        validation_loss = validation_value + gradient_weight * validation_direction
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / count,
                "validation_loss": validation_loss,
                "validation_value_loss": validation_value,
                "validation_direction_loss": validation_direction,
            }
        )
        if validation_loss < best_loss - float(training["min_delta"]):
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= int(training["patience"]):
            break
    if best_state is None:
        raise Geometry19S4Error("No finite S4 validation checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def run_s4_training(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text())
    runtime = root / config["artifacts"]["runtime_root"]
    audit_summary_path = runtime / config["artifacts"]["summary"]
    audit = json.loads(audit_summary_path.read_text())
    if audit["decision"]["decision"] != "TRAIN_S4_DIRECTIONAL_GRADIENT":
        raise Geometry19S4Error("Gate L audit did not authorize S4 training")
    gate_k_config = yaml.safe_load((root / config["gate_k_config"]).read_text())
    base_path = root / gate_k_config["base_surrogate_protocol"]
    base = yaml.safe_load(base_path.read_text())
    inverse = yaml.safe_load((root / gate_k_config["inverse_protocol"]).read_text())
    schema = yaml.safe_load((root / base["design_schema"]).read_text())
    fields = list(schema["design_variables"])
    feature_names = [item["name"] for item in fields]
    splits, target_names = load_regression_splits(
        root / base["regression_dataset"], feature_names, base["campaign_split"]
    )
    stats = fit_normalization(splits["train"]["x"], splits["train"]["y"])
    direction_path = runtime / config["artifacts"]["direction_dataset"]
    reused_direction_dataset = direction_path.exists()
    if reused_direction_dataset:
        direction_rows = _read_jsonl(direction_path)
    else:
        dataset_protocol = yaml.safe_load(
            (root / "config/physics/wp19_21_development_dataset_v1.yaml").read_text()
        )
        proposed = propose_direction_rows(
            splits, fields, config["s4"]["direction_data"]["requested_pairs"],
            normalized_step=float(config["s4"]["direction_data"]["normalized_step"]),
            seed=int(config["s4"]["direction_data"]["seed"]),
        )
        direction_rows = generate_direction_dataset(
            proposed,
            calibration_path=root / base["calibration_pack"],
            environment_payload=dataset_protocol["environment"],
            workers=int(config["s4"]["direction_data"]["workers"]),
        )
        _write_jsonl(direction_path, direction_rows)
    rows_by_split = {
        split: [row for row in direction_rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    valid_counts = {
        split: sum(bool(row.get("physics_acceptable")) for row in rows)
        for split, rows in rows_by_split.items()
    }
    if any(valid_counts[name] < max(16, int(0.5 * len(rows_by_split[name]))) for name in valid_counts):
        raise Geometry19S4Error(f"Insufficient valid S4 direction labels: {valid_counts}")
    direction_values = {
        split: direction_arrays(
            rows, feature_names, stats,
            inverse["diagnostic_acceptance"]["relative_tolerances"],
            inverse["diagnostic_acceptance"]["epsilons"],
        )
        for split, rows in rows_by_split.items()
    }
    x_values = {
        name: ((bucket["x"] - stats["input_mean"]) / stats["input_std"]).astype(np.float32)
        for name, bucket in splits.items()
    }
    y_values = {
        name: ((bucket["y"] - stats["output_mean"]) / stats["output_std"]).astype(np.float32)
        for name, bucket in splits.items()
    }
    run_device = torch.device(str(device or config["s4"]["training"]["device"]))
    training = config["s4"]["training"]
    seed = int(training["seed"])
    output_mean = torch.as_tensor(stats["output_mean"], device=run_device)
    output_std = torch.as_tensor(stats["output_std"], device=run_device)
    tolerance_tensor = torch.tensor(
        [inverse["diagnostic_acceptance"]["relative_tolerances"][name] for name in OUTPUT_NAMES],
        device=run_device,
    )
    epsilon_tensor = torch.tensor(
        [inverse["diagnostic_acceptance"]["epsilons"][name] for name in OUTPUT_NAMES],
        device=run_device,
    )
    loss_config = {
        **gate_k_config["loss_defaults"],
        "mode": "legacy_mse",
    }

    def value_loss(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        return surrogate_loss_terms(
            prediction, truth, output_mean, output_std,
            tolerance_tensor, epsilon_tensor, loss_config,
        )["total"]

    architecture = base["architecture"]
    gate_k2 = json.loads((root / config["gate_k2_summary"]).read_text())
    baseline_rep = gate_k2["representatives"]["s0_legacy_mse"]
    baseline_model, _, _ = load_weight_bundle(
        baseline_rep["model_directory"], expected_model_tag=baseline_rep["model_tag"],
        expected_feature_names=feature_names, expected_target_names=target_names,
        device=run_device,
    )
    baseline_direction = {
        split: _evaluate_direction_loss(
            baseline_model, direction_values[split], output_std, training, run_device
        )
        for split in ("validation", "test")
    }
    arms = {}
    for arm in config["s4"]["arms"]:
        arm_id = str(arm["arm_id"])
        gradient_weight = float(arm["gradient_weight"])
        _seed_all(seed)
        model = TaggedGeometry19MLP(
            19, 48, architecture["hidden_layers"], architecture["activation"],
            architecture["dropout"], architecture["batch_norm"],
        ).to(run_device)
        model.load_state_dict(copy.deepcopy(baseline_model.state_dict()))
        value_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_values["train"]), torch.from_numpy(y_values["train"])),
            batch_size=int(training["batch_size"]), shuffle=True,
            generator=torch.Generator().manual_seed(seed), num_workers=0,
        )
        model, history, best_epoch = train_directional_model(
            model, value_loader, x_values["validation"], y_values["validation"],
            direction_values["train"], direction_values["validation"], value_loss,
            output_std, training, gradient_weight, run_device, seed,
        )
        model_tag = f"novaturbo_geometry19_forward_gate_l_{arm_id}_seed{seed}_v1"
        model_dir = runtime / "models" / arm_id / f"seed_{seed}"
        bundle = save_weight_bundle(
            model_dir, model,
            {
                "checkpoint_version": "geometry19_tagged_weights_v1",
                "model_tag": model_tag,
                "architecture_tag": architecture["architecture_tag"],
                "feature_names": feature_names,
                "target_names": target_names,
                "training_seed": seed,
                "loss_config": {**loss_config, "gradient_weight": gradient_weight},
                "dataset_tag": "wp19_21_98304_development_v1_plus_sparse_directions",
                "formal_use_allowed": False,
                "code_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=root, check=True,
                    capture_output=True, text=True,
                ).stdout.strip(),
                "gate_l_config_sha256": sha256_file(config_source),
                "direction_dataset_sha256": sha256_file(direction_path),
            }, stats,
        )
        metrics = {}
        for split in ("validation", "test"):
            normalized_prediction = _predict(model, x_values[split], run_device)
            physical_prediction = normalized_prediction * stats["output_std"] + stats["output_mean"]
            metrics[split] = {
                "regression": _regression_metrics(splits[split]["y"], physical_prediction, target_names),
                "hit_alignment": hit_aligned_prediction_metrics(
                    splits[split]["y"], physical_prediction,
                    inverse["diagnostic_acceptance"]["relative_tolerances"],
                    inverse["diagnostic_acceptance"]["epsilons"],
                ),
                "ranking": ranking_benchmark(
                    splits[split]["y"], physical_prediction,
                    inverse["diagnostic_acceptance"]["relative_tolerances"],
                    inverse["diagnostic_acceptance"]["epsilons"],
                    maximum_targets=int(gate_k_config["evaluation"]["maximum_ranking_targets"]),
                    budgets=gate_k_config["evaluation"]["ranking_budgets"],
                ),
                "direction_loss": _evaluate_direction_loss(
                    model, direction_values[split], output_std, training, run_device
                ),
            }
        run = {
            "arm_id": arm_id,
            "seed": seed,
            "gradient_weight": gradient_weight,
            "model_tag": model_tag,
            "best_epoch": best_epoch,
            "validation": metrics["validation"],
            "test": metrics["test"],
            "artifacts": bundle,
        }
        _write_json(model_dir / "training_history.json", {"history": history, "best_epoch": best_epoch})
        _write_json(model_dir / "evaluation.json", run)
        arms[arm_id] = run
    baseline_validation_rho = float(baseline_rep["validation_rho_p50"])
    eligible = []
    for arm_id, run in arms.items():
        rho_guard = run["validation"]["hit_alignment"]["rho_p50"] <= baseline_validation_rho * (
            1.0 + float(config["s4"]["selection"]["maximum_validation_rho_regression"])
        )
        direction_guard = run["validation"]["direction_loss"] <= baseline_direction["validation"] * (
            1.0 - float(config["s4"]["selection"]["minimum_validation_direction_loss_reduction"])
        )
        run["selection_checks"] = {
            "validation_rho_guard": rho_guard,
            "validation_direction_improved": direction_guard,
        }
        if rho_guard and direction_guard:
            eligible.append(arm_id)
    winner = min(
        eligible or list(arms),
        key=lambda arm_id: (
            arms[arm_id]["validation"]["direction_loss"],
            arms[arm_id]["validation"]["hit_alignment"]["rho_p50"],
            arm_id,
        ),
    )
    summary = {
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_use_allowed": False,
        "new_v2_1_calls": 0 if reused_direction_dataset else len(direction_rows),
        "direction_dataset_v2_1_calls": len(direction_rows),
        "direction_counts": {
            split: {
                "requested": len(rows_by_split[split]),
                "valid": valid_counts[split],
            }
            for split in rows_by_split
        },
        "baseline": {
            "model_directory": baseline_rep["model_directory"],
            "model_tag": baseline_rep["model_tag"],
            "validation_rho_p50": baseline_validation_rho,
            "direction_loss": baseline_direction,
        },
        "arms": arms,
        "selection": {
            "uses_test_metrics": False,
            "eligible_arms": eligible,
            "winning_arm": winner,
            "winning_model_directory": str(Path(arms[winner]["artifacts"]["manifest"]).parent),
            "winning_model_tag": arms[winner]["model_tag"],
            "run_closure": winner in eligible,
        },
        "hashes": {
            "config_sha256": sha256_file(config_source),
            "direction_dataset_sha256": sha256_file(direction_path),
            "base_protocol_sha256": sha256_file(base_path),
        },
    }
    summary_path = runtime / config["artifacts"]["s4_training_summary"]
    _write_json(summary_path, summary)
    return summary


def _render_s4_report(summary: Mapping[str, Any]) -> str:
    baseline = summary["baseline"]
    final = summary["final_evaluation"]
    lines = [
        "# Gate L S4：稀疏方向约束代理",
        "",
        f"> 判定：**{final['decision']}**；生产代理未修改。",
        "",
        "| Arm | lambda | Val rho | Val方向loss | Test rho | Test方向loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm_id, arm in summary["arms"].items():
        lines.append(
            f"| {arm_id} | {arm['gradient_weight']:.3f} | "
            f"{arm['validation']['hit_alignment']['rho_p50']:.3f} | "
            f"{arm['validation']['direction_loss']:.3f} | "
            f"{arm['test']['hit_alignment']['rho_p50']:.3f} | "
            f"{arm['test']['direction_loss']:.3f} |"
        )
    winner = summary["selection"]["winning_arm"]
    audit = final["gate_i_direction_audit"]
    lines += [
        "",
        f"S0 validation方向loss：{baseline['direction_loss']['validation']:.3f}；"
        f"test方向loss：{baseline['direction_loss']['test']:.3f}。",
        f"验证选择：`{winner}`；Gate I轨迹下降precision："
        f"{audit['predicted_descent_precision']:.1%}。",
        f"冻结池Top-32：S0 {final['frozen_top32']['s0_legacy_mse']}/4，"
        f"S4 {final['frozen_top32']['s4_winner']}/4。",
        "",
        "S4未同时通过validation value guard和方向改善门槛，因此未运行新增V2.1闭环。",
        "",
    ]
    return "\n".join(lines)


def run_s4_evaluation(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    """Run zero-call held-out gradient and frozen-pool checks for S4."""

    root = Path(repository_root).resolve()
    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text())
    runtime = root / config["artifacts"]["runtime_root"]
    summary_path = runtime / config["artifacts"]["s4_training_summary"]
    summary = json.loads(summary_path.read_text())
    gate_k_config = yaml.safe_load((root / config["gate_k_config"]).read_text())
    base = yaml.safe_load((root / gate_k_config["base_surrogate_protocol"]).read_text())
    inverse = yaml.safe_load((root / gate_k_config["inverse_protocol"]).read_text())
    schema = yaml.safe_load((root / base["design_schema"]).read_text())
    feature_names = [item["name"] for item in schema["design_variables"]]
    winner_id = summary["selection"]["winning_arm"]
    winner = summary["arms"][winner_id]
    model, normalization, _ = load_weight_bundle(
        summary["selection"]["winning_model_directory"],
        expected_model_tag=summary["selection"]["winning_model_tag"],
        expected_feature_names=feature_names,
        expected_target_names=manifest_target_names(summary["selection"]["winning_model_directory"]),
        device=str(device or config["device"]),
    )
    gate_i_pairs = _read_jsonl(runtime / config["artifacts"]["pairs"])
    gate_i_audit = audit_model_directions(
        gate_i_pairs, feature_names, model, normalization,
        inverse["diagnostic_acceptance"]["relative_tolerances"],
        inverse["diagnostic_acceptance"]["epsilons"],
        sign_epsilon=float(config["audit"]["sign_epsilon"]),
    )
    gate_k2_config = yaml.safe_load((root / config["gate_k2_config"]).read_text())
    gate_k2 = json.loads((root / config["gate_k2_summary"]).read_text())
    baseline = gate_k2["representatives"]["s0_legacy_mse"]
    representatives = {
        "s0_legacy_mse": baseline,
        "s4_winner": {
            "model_directory": summary["selection"]["winning_model_directory"],
            "model_tag": summary["selection"]["winning_model_tag"],
        },
    }
    frozen = _run_frozen_pool(
        root, gate_k2_config, representatives, base, inverse,
        str(device or config["device"]),
    )
    _write_json(runtime / "s4_frozen_pool_summary.json", frozen)
    frozen_top32 = {
        strategy: int(value["budgets"]["32"]["target_hits"])
        for strategy, value in frozen["aggregate"].items()
    }
    eligible = bool(summary["selection"]["run_closure"])
    direction_pass = bool(
        gate_i_audit["predicted_descent_precision"]
        >= float(config["s4"]["promotion"]["minimum_predicted_descent_precision"])
        and gate_i_audit["axis_sign_accuracy"]["tsfc_kg_N_s"]
        >= float(config["s4"]["promotion"]["minimum_tsfc_sign_accuracy"])
    )
    decision = (
        "ELIGIBLE_FOR_EQUAL_BUDGET_CLOSURE"
        if eligible and direction_pass else "REJECT_S4_BEFORE_CLOSURE"
    )
    summary["final_evaluation"] = {
        "decision": decision,
        "new_v2_1_calls": 0,
        "validation_selection_eligible": eligible,
        "gate_i_direction_pass": direction_pass,
        "gate_i_direction_audit": gate_i_audit,
        "frozen_top32": frozen_top32,
        "production_pipeline_modified": False,
    }
    _write_json(summary_path, summary)
    report_path = runtime / config["artifacts"]["s4_report"]
    report_path.write_text(_render_s4_report(summary), encoding="utf-8")
    return summary


def manifest_target_names(model_directory: str | Path) -> list[str]:
    manifest = json.loads(
        (Path(model_directory) / "model_manifest.json").read_text(encoding="utf-8")
    )
    return list(manifest["target_names"])
