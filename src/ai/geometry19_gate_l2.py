"""Gate L2: train Geometry19 proxies on V2.1 central-difference gradients."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.ai.geometry19_gate_k import (
    _best_stage_alignment,
    _predict,
    _refinement_diagnostics,
    _regression_metrics,
    _seed_all,
    _write_json,
    ranking_benchmark,
)
from src.ai.geometry19_candidates import run_candidate_protocol
from src.ai.geometry19_physics import run_true_physics_stage
from src.ai.geometry19_pipeline import build_target_mapping, resolve_generator_checkpoint
from src.ai.geometry19_refinement import run_refinement_stage
from src.ai.geometry19_gate_k2 import _run_frozen_pool
from src.ai.geometry19_gate_l import audit_model_directions
from src.ai.geometry19_gate_l_s4 import (
    _read_jsonl,
    _write_jsonl,
    generate_direction_dataset,
)
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


GATE_L2_VERSION = "wp19_22_gate_l2_v21_gradient_v1"


class Geometry19GateL2Error(ValueError):
    """Raised when the V2.1 central-gradient contract is invalid."""


def build_negative_proposals(
    positive_rows: Sequence[Mapping[str, Any]],
    fields: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Reflect valid positive perturbations around each anchor without clipping."""

    names = [str(item["name"]) for item in fields]
    lower = np.asarray([item["bounds"][0] for item in fields], dtype=np.float64)
    upper = np.asarray([item["bounds"][1] for item in fields], dtype=np.float64)
    integer_indices = [
        index for index, item in enumerate(fields) if item["dtype"] == "int64"
    ]
    proposals = []
    counts = {"positive_rows": len(positive_rows), "positive_invalid": 0, "asymmetric_bounds": 0}
    for row in positive_rows:
        if not row.get("physics_acceptable") or not row.get("proposed_performance"):
            counts["positive_invalid"] += 1
            continue
        center = np.asarray(
            [row["prior_geometry"][name] for name in names], dtype=np.float64
        )
        plus = np.asarray(
            [row["proposed_geometry"][name] for name in names], dtype=np.float64
        )
        minus = center - (plus - center)
        if np.any(minus < lower - 1.0e-12) or np.any(minus > upper + 1.0e-12):
            counts["asymmetric_bounds"] += 1
            continue
        if any(minus[index] != center[index] for index in integer_indices):
            raise Geometry19GateL2Error("Central gradient changed an integer coordinate")
        proposals.append(
            {
                "pair_id": str(row["pair_id"]),
                "split": str(row["split"]),
                "campaign_id": str(row["campaign_id"]),
                "sample_id": int(row["sample_id"]),
                "prior_geometry": dict(row["prior_geometry"]),
                "prior_performance": dict(row["prior_performance"]),
                "plus_geometry": dict(row["proposed_geometry"]),
                "plus_performance": dict(row["proposed_performance"]),
                "proposed_geometry": {
                    name: (
                        int(round(minus[index]))
                        if fields[index]["dtype"] == "int64" else float(minus[index])
                    )
                    for index, name in enumerate(names)
                },
            }
        )
    counts["negative_evaluations_requested"] = len(proposals)
    return proposals, counts


def combine_central_gradient_rows(
    negative_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Retain pairs whose positive and negative V2.1 endpoints are both valid."""

    output = []
    for row in negative_results:
        if not row.get("physics_acceptable") or not row.get("proposed_performance"):
            continue
        output.append(
            {
                "pair_id": row["pair_id"],
                "split": row["split"],
                "campaign_id": row["campaign_id"],
                "sample_id": row["sample_id"],
                "center_geometry": row["prior_geometry"],
                "center_performance": row["prior_performance"],
                "plus_geometry": row["plus_geometry"],
                "plus_performance": row["plus_performance"],
                "minus_geometry": row["proposed_geometry"],
                "minus_performance": row["proposed_performance"],
                "positive_source": "V2.1_REUSED_GATE_L_S4",
                "negative_source": "V2.1_GATE_L2",
                "gradient_scheme": "CENTRAL_DIRECTIONAL_DIFFERENCE",
            }
        )
    return output


def central_gradient_arrays(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    normalization: Mapping[str, np.ndarray],
    tolerances: Mapping[str, float],
    epsilons: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return center, unit normalized direction, true V2.1 JVP, denominator."""

    if not rows:
        raise Geometry19GateL2Error("No central-gradient row exists")
    center = np.asarray(
        [[row["center_geometry"][name] for name in feature_names] for row in rows],
        dtype=np.float32,
    )
    plus = np.asarray(
        [[row["plus_geometry"][name] for name in feature_names] for row in rows],
        dtype=np.float32,
    )
    center_normalized = (center - normalization["input_mean"]) / normalization["input_std"]
    plus_normalized = (plus - normalization["input_mean"]) / normalization["input_std"]
    delta = plus_normalized - center_normalized
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    if np.any(norm <= 1.0e-8):
        raise Geometry19GateL2Error("Central-gradient direction is zero")
    unit = delta / norm
    plus_y = np.asarray(
        [[row["plus_performance"][name] for name in OUTPUT_NAMES] for row in rows],
        dtype=np.float32,
    )
    minus_y = np.asarray(
        [[row["minus_performance"][name] for name in OUTPUT_NAMES] for row in rows],
        dtype=np.float32,
    )
    center_y = np.asarray(
        [[row["center_performance"][name] for name in OUTPUT_NAMES] for row in rows],
        dtype=np.float32,
    )
    true_jvp = (plus_y - minus_y) / (2.0 * norm)
    denominator = np.asarray([tolerances[name] for name in OUTPUT_NAMES]) * np.maximum(
        np.abs(center_y), np.asarray([epsilons[name] for name in OUTPUT_NAMES])
    )
    return (
        center_normalized.astype(np.float32),
        unit.astype(np.float32),
        true_jvp.astype(np.float32),
        denominator.astype(np.float32),
    )


def v21_gradient_loss(
    model: torch.nn.Module,
    center: torch.Tensor,
    unit_direction: torch.Tensor,
    true_jvp: torch.Tensor,
    denominator: torch.Tensor,
    output_std: torch.Tensor,
    *,
    huber_delta: float,
    cosine_weight: float,
    create_graph: bool,
) -> torch.Tensor:
    """Match the proxy JVP to a V2.1 central finite-difference label."""

    _, normalized_jvp = torch.autograd.functional.jvp(
        model, center, unit_direction, create_graph=create_graph, strict=True
    )
    predicted_scaled = normalized_jvp[:, :3] * output_std[:3] / denominator
    true_scaled = true_jvp / denominator
    magnitude = F.smooth_l1_loss(
        predicted_scaled, true_scaled, beta=huber_delta
    )
    cosine = 1.0 - F.cosine_similarity(
        predicted_scaled, true_scaled, dim=1, eps=1.0e-8
    ).mean()
    return magnitude + cosine_weight * cosine


def _gradient_loader(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(*(torch.from_numpy(value) for value in arrays)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


def _evaluate_gradient_loss(
    model: torch.nn.Module,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    output_std: torch.Tensor,
    training: Mapping[str, Any],
    device: torch.device,
) -> float:
    tensors = tuple(torch.from_numpy(value).to(device) for value in arrays)
    model.eval()
    with torch.enable_grad():
        loss = v21_gradient_loss(
            model, *tensors, output_std,
            huber_delta=float(training["gradient_huber_delta"]),
            cosine_weight=float(training["gradient_cosine_weight"]),
            create_graph=False,
        )
    return float(loss.detach())


def train_v21_gradient_model(
    model: TaggedGeometry19MLP,
    value_loader: DataLoader,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    train_gradients: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    validation_gradients: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    value_loss_function: Any,
    output_std: torch.Tensor,
    training: Mapping[str, Any],
    gradient_weight: float,
    device: torch.device,
    seed: int,
) -> tuple[TaggedGeometry19MLP, list[dict[str, float]], int]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
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
        gradient_loader = _gradient_loader(
            train_gradients, int(training["gradient_batch_size"]), seed + epoch, True
        )
        gradient_iterator = iter(gradient_loader)
        total = 0.0
        count = 0
        for value_x, value_y in value_loader:
            try:
                gradient_batch = next(gradient_iterator)
            except StopIteration:
                gradient_iterator = iter(gradient_loader)
                gradient_batch = next(gradient_iterator)
            value_x, value_y = value_x.to(device), value_y.to(device)
            gradient_batch = tuple(value.to(device) for value in gradient_batch)
            optimizer.zero_grad(set_to_none=True)
            value_loss = value_loss_function(model(value_x), value_y)
            model.eval()
            gradient_loss = v21_gradient_loss(
                model, *gradient_batch, output_std,
                huber_delta=float(training["gradient_huber_delta"]),
                cosine_weight=float(training["gradient_cosine_weight"]),
                create_graph=True,
            )
            model.train()
            loss = value_loss + gradient_weight * gradient_loss
            if not torch.isfinite(loss):
                raise Geometry19GateL2Error("V2.1 gradient training produced non-finite loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(value_x)
            count += len(value_x)
        model.eval()
        with torch.no_grad():
            validation_value = float(
                value_loss_function(model(validation_x_tensor), validation_y_tensor)
            )
        validation_gradient = _evaluate_gradient_loss(
            model, validation_gradients, output_std, training, device
        )
        validation_loss = validation_value + gradient_weight * validation_gradient
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / count,
                "validation_loss": validation_loss,
                "validation_value_loss": validation_value,
                "validation_gradient_loss": validation_gradient,
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
        raise Geometry19GateL2Error("No V2.1-gradient checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def _manifest_target_names(directory: str | Path) -> list[str]:
    return list(json.loads((Path(directory) / "model_manifest.json").read_text())["target_names"])


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Gate L2：V2.1中心差分梯度监督",
        "",
        f"> 判定：**{summary['decision']['decision']}**；生产代理未修改。",
        "",
        "| Arm | lambda | Val rho | Val gradient loss | Test rho | Test gradient loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm_id, arm in summary["arms"].items():
        lines.append(
            f"| {arm_id} | {arm['gradient_weight']:.3f} | "
            f"{arm['validation']['hit_alignment']['rho_p50']:.3f} | "
            f"{arm['validation']['gradient_loss']:.3f} | "
            f"{arm['test']['hit_alignment']['rho_p50']:.3f} | "
            f"{arm['test']['gradient_loss']:.3f} |"
        )
    lines += [
        "",
        f"Validation选择：`{summary['selection']['winning_arm']}`；"
        f"eligible={summary['selection']['eligible_arms']}。",
        f"Gate I下降precision：{summary['decision']['gate_i_audit']['predicted_descent_precision']:.1%}；"
        f"TSFC符号：{summary['decision']['gate_i_audit']['axis_sign_accuracy']['tsfc_kg_N_s']:.1%}。",
        f"冻结Top-32：S0 {summary['decision']['frozen_top32']['s0_legacy_mse']}/4，"
        f"L2 {summary['decision']['frozen_top32']['gate_l2_winner']}/4。",
        "",
    ]
    return "\n".join(lines)


def run_gate_l2(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text())
    if config.get("protocol_version") != GATE_L2_VERSION:
        raise Geometry19GateL2Error("Unsupported Gate L2 protocol")
    runtime = root / config["artifacts"]["runtime_root"]
    if runtime.exists():
        raise Geometry19GateL2Error(f"Refusing to overwrite Gate L2 run: {runtime}")
    runtime.mkdir(parents=True)
    base_path = root / config["base_surrogate_protocol"]
    base = yaml.safe_load(base_path.read_text())
    inverse = yaml.safe_load((root / config["inverse_protocol"]).read_text())
    gate_k_config = yaml.safe_load((root / config["gate_k_config"]).read_text())
    schema = yaml.safe_load((root / base["design_schema"]).read_text())
    fields = list(schema["design_variables"])
    feature_names = [item["name"] for item in fields]
    splits, target_names = load_regression_splits(
        root / base["regression_dataset"], feature_names, base["campaign_split"]
    )
    stats = fit_normalization(splits["train"]["x"], splits["train"]["y"])
    positive_path = root / config["positive_direction_dataset"]
    positive_rows = _read_jsonl(positive_path)
    negative_proposals, proposal_counts = build_negative_proposals(positive_rows, fields)
    dataset_protocol = yaml.safe_load(
        (root / "config/physics/wp19_21_development_dataset_v1.yaml").read_text()
    )
    negative_results = generate_direction_dataset(
        negative_proposals,
        calibration_path=root / base["calibration_pack"],
        environment_payload=dataset_protocol["environment"],
        workers=int(config["gradient_data"]["workers"]),
    )
    central_rows = combine_central_gradient_rows(negative_results)
    central_path = runtime / config["artifacts"]["central_gradient_dataset"]
    _write_jsonl(central_path, central_rows)
    rows_by_split = {
        split: [row for row in central_rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    counts = {split: len(rows) for split, rows in rows_by_split.items()}
    for split, minimum in config["gradient_data"]["minimum_valid_pairs"].items():
        if counts[split] < int(minimum):
            raise Geometry19GateL2Error(f"Insufficient {split} central gradients: {counts}")
    gradient_arrays = {
        split: central_gradient_arrays(
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
    run_device = torch.device(str(device or config["device"]))
    training = config["training"]
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
    loss_config = {**gate_k_config["loss_defaults"], "mode": "legacy_mse"}

    def value_loss(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        return surrogate_loss_terms(
            prediction, truth, output_mean, output_std,
            tolerance_tensor, epsilon_tensor, loss_config,
        )["total"]

    gate_k2_path = root / config["gate_k2_summary"]
    gate_k2 = json.loads(gate_k2_path.read_text())
    baseline_rep = gate_k2["representatives"]["s0_legacy_mse"]
    baseline_model, _, _ = load_weight_bundle(
        baseline_rep["model_directory"], expected_model_tag=baseline_rep["model_tag"],
        expected_feature_names=feature_names, expected_target_names=target_names,
        device=run_device,
    )
    baseline_gradient = {
        split: _evaluate_gradient_loss(
            baseline_model, gradient_arrays[split], output_std, training, run_device
        )
        for split in ("validation", "test")
    }
    architecture = base["architecture"]
    arms = {}
    for arm in config["arms"]:
        arm_id = str(arm["arm_id"])
        weight = float(arm["gradient_weight"])
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
        model, history, best_epoch = train_v21_gradient_model(
            model, value_loader, x_values["validation"], y_values["validation"],
            gradient_arrays["train"], gradient_arrays["validation"], value_loss,
            output_std, training, weight, run_device, seed,
        )
        model_tag = f"novaturbo_geometry19_forward_gate_l2_{arm_id}_seed{seed}_v1"
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
                "loss_config": {**loss_config, "v21_gradient_weight": weight},
                "dataset_tag": "wp19_21_plus_v21_central_gradients_v1",
                "formal_use_allowed": False,
                "code_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=root, check=True,
                    capture_output=True, text=True,
                ).stdout.strip(),
                "gate_l2_config_sha256": sha256_file(config_source),
                "central_gradient_dataset_sha256": sha256_file(central_path),
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
                "gradient_loss": _evaluate_gradient_loss(
                    model, gradient_arrays[split], output_std, training, run_device
                ),
            }
        arms[arm_id] = {
            "arm_id": arm_id, "seed": seed, "gradient_weight": weight,
            "model_tag": model_tag, "best_epoch": best_epoch,
            "validation": metrics["validation"], "test": metrics["test"],
            "artifacts": bundle,
        }
        _write_json(model_dir / "training_history.json", {"best_epoch": best_epoch, "history": history})
    baseline_val_rho = float(baseline_rep["validation_rho_p50"])
    eligible = []
    for arm_id, arm in arms.items():
        rho_guard = arm["validation"]["hit_alignment"]["rho_p50"] <= baseline_val_rho * (
            1.0 + float(config["selection"]["maximum_validation_rho_regression"])
        )
        gradient_guard = arm["validation"]["gradient_loss"] <= baseline_gradient["validation"] * (
            1.0 - float(config["selection"]["minimum_validation_gradient_loss_reduction"])
        )
        arm["selection_checks"] = {
            "validation_rho_guard": rho_guard,
            "validation_gradient_improved": gradient_guard,
        }
        if rho_guard and gradient_guard:
            eligible.append(arm_id)
    winner_id = min(
        eligible or list(arms),
        key=lambda name: (
            arms[name]["validation"]["gradient_loss"],
            arms[name]["validation"]["hit_alignment"]["rho_p50"], name,
        ),
    )
    winner = arms[winner_id]
    gate_l_config = yaml.safe_load((root / config["gate_l_config"]).read_text())
    gate_i_pairs = _read_jsonl(root / config["gate_l_runtime"] / gate_l_config["artifacts"]["pairs"])
    winner_model, winner_stats, _ = load_weight_bundle(
        Path(winner["artifacts"]["manifest"]).parent,
        expected_model_tag=winner["model_tag"], expected_feature_names=feature_names,
        expected_target_names=target_names, device=run_device,
    )
    gate_i_audit = audit_model_directions(
        gate_i_pairs, feature_names, winner_model, winner_stats,
        inverse["diagnostic_acceptance"]["relative_tolerances"],
        inverse["diagnostic_acceptance"]["epsilons"],
        sign_epsilon=float(gate_l_config["audit"]["sign_epsilon"]),
    )
    representatives = {
        "s0_legacy_mse": baseline_rep,
        "gate_l2_winner": {
            "model_directory": str(Path(winner["artifacts"]["manifest"]).parent),
            "model_tag": winner["model_tag"],
        },
    }
    gate_k2_config = yaml.safe_load((root / config["gate_k2_config"]).read_text())
    frozen = _run_frozen_pool(root, gate_k2_config, representatives, base, inverse, str(run_device))
    _write_json(runtime / config["artifacts"]["frozen_summary"], frozen)
    frozen_top32 = {
        name: int(value["budgets"]["32"]["target_hits"])
        for name, value in frozen["aggregate"].items()
    }
    direction_pass = (
        gate_i_audit["predicted_descent_precision"]
        >= float(config["promotion"]["minimum_gate_i_descent_precision"])
        and gate_i_audit["axis_sign_accuracy"]["tsfc_kg_N_s"]
        >= float(config["promotion"]["minimum_gate_i_tsfc_sign_accuracy"])
    )
    frozen_pass = frozen_top32["gate_l2_winner"] >= frozen_top32["s0_legacy_mse"]
    closure_eligible = winner_id in eligible and direction_pass and frozen_pass
    decision = "RUN_EQUAL_BUDGET_CLOSURE" if closure_eligible else "REJECT_BEFORE_CLOSURE"
    summary = {
        "protocol_version": GATE_L2_VERSION,
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_use_allowed": False,
        "production_pipeline_modified": False,
        "new_v2_1_calls": len(negative_results),
        "gradient_data": {**proposal_counts, "valid_by_split": counts},
        "baseline": {
            "model_directory": baseline_rep["model_directory"],
            "model_tag": baseline_rep["model_tag"],
            "validation_rho_p50": baseline_val_rho,
            "gradient_loss": baseline_gradient,
        },
        "arms": arms,
        "selection": {
            "uses_test_metrics": False,
            "eligible_arms": eligible,
            "winning_arm": winner_id,
        },
        "decision": {
            "decision": decision,
            "validation_eligible": winner_id in eligible,
            "gate_i_direction_pass": direction_pass,
            "frozen_no_regression": frozen_pass,
            "gate_i_audit": gate_i_audit,
            "frozen_top32": frozen_top32,
        },
        "hashes": {
            "config_sha256": sha256_file(config_source),
            "positive_dataset_sha256": sha256_file(positive_path),
            "central_dataset_sha256": sha256_file(central_path),
            "gate_k2_summary_sha256": sha256_file(gate_k2_path),
        },
    }
    summary_path = runtime / config["artifacts"]["training_summary"]
    _write_json(summary_path, summary)
    (runtime / config["artifacts"]["report"]).write_text(
        _render_report(summary), encoding="utf-8"
    )
    return summary


def closure_comparison_key(result: Mapping[str, Any]) -> tuple[Any, ...]:
    """Rank complete closures with target coverage before approximation quality."""

    rate = result["refinement"].get("rho_improved_rate")
    return (
        -int(result["final_target_hits"]),
        -int(result["c_target_hits"]),
        float(result["median_best_true_rho"]),
        int(result["total_v2_1_calls"]),
        -float(rate if rate is not None else -math.inf),
        str(result["arm_id"]),
    )


def _render_closure_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Gate L2：S5 与 Gate K2 S0 完整链路对比",
        "",
        "| Arm | C命中 | D救援 | 最终命中 | median rho | V2.1 | D rho改善率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm_id in ("s0_legacy_mse", "s5_v21_gradient"):
        arm = summary["arms"][arm_id]
        rate = arm["refinement"].get("rho_improved_rate")
        rate_text = f"{rate:.1%}" if rate is not None else "n/a"
        lines.append(
            f"| {arm_id} | {arm['c_target_hits']}/4 | {arm['d_rescue_hits']}/4 | "
            f"{arm['final_target_hits']}/4 | {arm['median_best_true_rho']:.3f} | "
            f"{arm['total_v2_1_calls']} | {rate_text} |"
        )
    lines += [
        "",
        f"完整链路胜出：`{summary['comparison']['winning_arm']}`。",
        f"生产晋级判定：`{summary['comparison']['decision']}`。",
        "",
    ]
    return "\n".join(lines)


def run_gate_l2_closure(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    device: str | None = None,
) -> dict[str, Any]:
    """Run validation-selected S5 through B/C/D and compare with frozen K2 S0."""

    root = Path(repository_root).resolve()
    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text())
    runtime = root / config["artifacts"]["runtime_root"]
    training_summary_path = runtime / config["artifacts"]["training_summary"]
    training_summary = json.loads(training_summary_path.read_text())
    closure_root = runtime / config["artifacts"]["closure_root"]
    if closure_root.exists():
        raise Geometry19GateL2Error(f"Refusing to overwrite Gate L2 closure: {closure_root}")
    closure_root.mkdir(parents=True)
    winner_id = training_summary["selection"]["winning_arm"]
    winner = training_summary["arms"][winner_id]
    forward_dir = Path(winner["artifacts"]["manifest"]).parent
    gate_i = yaml.safe_load((root / config["closure"]["gate_i_config"]).read_text())
    run_device = str(device or config["device"])
    targets = {}
    for target_item in gate_i["targets"]:
        sample_id = str(target_item["sample_id"])
        target = build_target_mapping(
            thrust_N=target_item["thrust_N"],
            tsfc_kg_N_s=target_item["tsfc_kg_N_s"],
            mass_kg=target_item["mass_kg"],
            target_type=target_item.get("target_type", "ID"),
            expected_status=target_item.get("expected_status", "KNOWN_FEASIBLE"),
        )
        target_path = closure_root / "targets" / f"sample_{sample_id}.yaml"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(yaml.safe_dump(target, sort_keys=False), encoding="utf-8")
        run_root = closure_root / "runs" / f"sample_{sample_id}"
        b_summary = run_candidate_protocol(
            root, target_path, mode="dflow", device=run_device,
            output_root=run_root / "candidates/dflow",
            generator_checkpoint=resolve_generator_checkpoint(root, "bounded-sigmoid"),
            dflow_target_loss_mode="hit_aligned",
            candidate_ranking_mode="predicted_rho",
            forward_weights_override=forward_dir,
        )
        observed_raw = int(b_summary["candidate_protocol"]["n_raw"])
        expected_raw = int(
            yaml.safe_load((root / config["gate_k2_config"]).read_text())["budgets"][
                "raw_candidates_per_target_per_arm"
            ]
        )
        if observed_raw != expected_raw:
            raise Geometry19GateL2Error(
                f"S5 raw-candidate budget mismatch: {observed_raw} != {expected_raw}"
            )
        c_budget = int(config["closure"]["c_true_physics_budget_per_target"])
        if int(b_summary["candidate_protocol"]["n_keep"]) < c_budget:
            raise Geometry19GateL2Error(f"S5 has insufficient N_keep for {sample_id}")
        c_summary = run_true_physics_stage(
            root, target_path, candidate_root=run_root / "candidates",
            output_root=run_root / "wp19_23c", sampling_mode="dflow",
            evaluation_budget=c_budget,
        )
        d_summary = None
        if not c_summary["downstream_stop"]:
            d_summary = run_refinement_stage(
                root, target_path, physics_root=run_root / "wp19_23c",
                output_root=run_root / "wp19_23d", device=run_device,
                target_loss_mode="hit_aligned", true_acceptance_mode="hit_aligned",
                forward_weights_override=forward_dir,
            )
            d_calls = int(d_summary["counts"]["true_v2_1_evaluations"])
            if d_calls > int(config["closure"]["maximum_d_true_physics_per_target"]):
                raise Geometry19GateL2Error(f"S5 D budget exceeded for {sample_id}")
        alignment = _best_stage_alignment(c_summary, d_summary)
        c_hit = bool(c_summary["downstream_stop"])
        d_hit = bool(d_summary and d_summary["downstream_stop"])
        targets[sample_id] = {
            "n_keep": int(b_summary["candidate_protocol"]["n_keep"]),
            "c_hit": c_hit,
            "d_rescue": d_hit,
            "final_hit": c_hit or d_hit,
            "c_v2_1_calls": int(c_summary["counts"]["unique_v2_1_evaluations"]),
            "d_v2_1_calls": (
                int(d_summary["counts"]["true_v2_1_evaluations"]) if d_summary else 0
            ),
            "best_hit_alignment": dict(alignment) if alignment else None,
        }
    refinement = _refinement_diagnostics(
        sorted(closure_root.glob("runs/sample_*/wp19_23d/refinement_ledger.jsonl"))
    )
    rho_values = [
        float(item["best_hit_alignment"]["rho"])
        for item in targets.values() if item["best_hit_alignment"]
    ]
    c_calls = sum(item["c_v2_1_calls"] for item in targets.values())
    d_calls = sum(item["d_v2_1_calls"] for item in targets.values())
    s5_result = {
        "arm_id": "s5_v21_gradient",
        "source_training_arm": winner_id,
        "targets": targets,
        "c_target_hits": sum(item["c_hit"] for item in targets.values()),
        "d_rescue_hits": sum(item["d_rescue"] for item in targets.values()),
        "final_target_hits": sum(item["final_hit"] for item in targets.values()),
        "final_hit_targets": sorted(
            sample_id for sample_id, item in targets.items() if item["final_hit"]
        ),
        "median_best_true_rho": float(np.median(rho_values)),
        "c_v2_1_calls": c_calls,
        "d_v2_1_calls": d_calls,
        "total_v2_1_calls": c_calls + d_calls,
        "refinement": refinement,
    }
    gate_k2_path = root / config["gate_k2_summary"]
    gate_k2 = json.loads(gate_k2_path.read_text())
    source_s0 = gate_k2["arms"]["s0_legacy_mse"]
    s0_result = {
        "arm_id": "s0_legacy_mse",
        "targets": source_s0["targets"],
        "c_target_hits": int(source_s0["c_target_hits"]),
        "d_rescue_hits": int(source_s0["d_rescue_hits"]),
        "final_target_hits": int(source_s0["final_target_hits"]),
        "final_hit_targets": list(source_s0["final_hit_targets"]),
        "median_best_true_rho": float(source_s0["median_best_true_rho"]),
        "c_v2_1_calls": int(source_s0["c_v2_1_calls"]),
        "d_v2_1_calls": int(source_s0["d_v2_1_calls"]),
        "total_v2_1_calls": int(source_s0["total_v2_1_calls"]),
        "refinement": dict(source_s0["refinement"]),
    }
    arms = {"s0_legacy_mse": s0_result, "s5_v21_gradient": s5_result}
    winning_arm = min(arms, key=lambda name: closure_comparison_key(arms[name]))
    decision = (
        "PROMOTE_S5_CANDIDATE"
        if winning_arm == "s5_v21_gradient"
        and s5_result["final_target_hits"] >= s0_result["final_target_hits"]
        else "KEEP_K2_S0"
    )
    summary = {
        "protocol_version": GATE_L2_VERSION,
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_use_allowed": False,
        "production_pipeline_modified": False,
        "equal_budget": {
            "raw_candidates_per_target": expected_raw,
            "c_v2_1_per_target": c_budget,
            "maximum_d_v2_1_per_target": int(
                config["closure"]["maximum_d_true_physics_per_target"]
            ),
        },
        "arms": arms,
        "comparison": {
            "method": "lexicographic_hits_c_hits_median_rho_cost_refinement",
            "winning_arm": winning_arm,
            "decision": decision,
        },
        "new_v2_1_calls": s5_result["total_v2_1_calls"],
        "hashes": {
            "config_sha256": sha256_file(config_source),
            "training_summary_sha256": sha256_file(training_summary_path),
            "gate_k2_summary_sha256": sha256_file(gate_k2_path),
        },
    }
    summary_path = closure_root / config["artifacts"]["closure_summary"]
    _write_json(summary_path, summary)
    (closure_root / config["artifacts"]["closure_report"]).write_text(
        _render_closure_report(summary), encoding="utf-8"
    )
    return summary
