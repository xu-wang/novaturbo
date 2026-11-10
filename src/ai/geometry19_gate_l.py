"""Gate L: audit surrogate directional gradients on true V2.1 trajectories."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from src.ai.geometry19_gate_k import _json_safe, _write_json
from src.ai.geometry19_hit_alignment import OUTPUT_NAMES
from src.ai.geometry19_surrogate import load_weight_bundle, sha256_file


GATE_L_VERSION = "wp19_22_gate_l_gradient_alignment_v1"


class Geometry19GateLError(ValueError):
    """Raised when the Gate L audit contract is invalid."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_json_safe(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def collect_direction_pairs(
    trajectory_root: str | Path,
    feature_fields: Sequence[Mapping[str, Any]],
    *,
    minimum_normalized_direction_norm: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Recover continuous-only, true-valid refinement transitions."""

    root = Path(trajectory_root)
    feature_names = [str(item["name"]) for item in feature_fields]
    integer_names = {
        str(item["name"]) for item in feature_fields if item["dtype"] == "int64"
    }
    lower = np.asarray([item["bounds"][0] for item in feature_fields], dtype=np.float64)
    upper = np.asarray([item["bounds"][1] for item in feature_fields], dtype=np.float64)
    scale = np.maximum(upper - lower, 1.0e-12)
    pairs: list[dict[str, Any]] = []
    counts = {
        "ledger_rows": 0,
        "true_valid_pairs": 0,
        "continuous_only_pairs": 0,
        "duplicate_pairs": 0,
        "near_zero_pairs": 0,
    }
    seen: set[tuple[str, str, str]] = set()
    for ledger_path in sorted(root.glob("runs/sample_*/*/wp19_23d/refinement_ledger.jsonl")):
        arm_id = ledger_path.parts[-3]
        sample_id = ledger_path.parts[-4].removeprefix("sample_")
        target_path = root / "targets" / f"sample_{sample_id}.yaml"
        target = yaml.safe_load(target_path.read_text(encoding="utf-8"))
        physics_path = ledger_path.parent.parent / "wp19_23c" / "physics_forward_ledger.jsonl"
        lookup = {row["evaluation_id"]: row for row in _read_jsonl(physics_path)}
        for ledger_index, row in enumerate(_read_jsonl(ledger_path)):
            counts["ledger_rows"] += 1
            proposed = row["proposed"]
            prior = lookup.get(row["prior_evaluation_id"])
            lookup[proposed["evaluation_id"]] = proposed
            if not (
                prior
                and prior.get("physics_acceptable")
                and proposed.get("physics_acceptable")
                and prior.get("performance")
                and proposed.get("performance")
            ):
                continue
            counts["true_valid_pairs"] += 1
            if any(
                prior["geometry19"][name] != proposed["geometry19"][name]
                for name in integer_names
            ):
                continue
            counts["continuous_only_pairs"] += 1
            prior_x = np.asarray(
                [prior["geometry19"][name] for name in feature_names], dtype=np.float64
            )
            proposed_x = np.asarray(
                [proposed["geometry19"][name] for name in feature_names], dtype=np.float64
            )
            normalized_norm = float(np.linalg.norm((proposed_x - prior_x) / scale))
            if normalized_norm <= minimum_normalized_direction_norm:
                counts["near_zero_pairs"] += 1
                continue
            key = (
                sample_id,
                str(prior.get("design_hash")),
                str(proposed.get("design_hash")),
            )
            if key in seen:
                counts["duplicate_pairs"] += 1
                continue
            seen.add(key)
            pairs.append(
                {
                    "pair_id": f"{sample_id}:{arm_id}:{ledger_index}",
                    "sample_id": sample_id,
                    "source_arm": arm_id,
                    "checkpoint": int(row["checkpoint"]),
                    "prior_evaluation_id": prior["evaluation_id"],
                    "proposed_evaluation_id": proposed["evaluation_id"],
                    "prior_geometry": dict(prior["geometry19"]),
                    "proposed_geometry": dict(proposed["geometry19"]),
                    "prior_performance": dict(prior["performance"]),
                    "proposed_performance": dict(proposed["performance"]),
                    "target_performance": {
                        name: float(target[name]) for name in OUTPUT_NAMES
                    },
                    "normalized_direction_norm": normalized_norm,
                    "accepted": bool(row["accepted"]),
                }
            )
    return pairs, counts


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else None


def _rho(values: np.ndarray, target: np.ndarray, denominator: np.ndarray) -> float:
    return float(np.max(np.abs(values - target) / denominator))


def audit_model_directions(
    pairs: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    model: torch.nn.Module,
    normalization: Mapping[str, np.ndarray],
    tolerances: Mapping[str, float],
    epsilons: Mapping[str, float],
    *,
    sign_epsilon: float,
) -> dict[str, Any]:
    """Compare model JVPs and finite-step rankings with true V2.1 changes."""

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    output_std = np.asarray(normalization["output_std"][:3], dtype=np.float64)
    rows = []
    for pair in pairs:
        prior_x = np.asarray(
            [pair["prior_geometry"][name] for name in feature_names], dtype=np.float64
        )
        proposed_x = np.asarray(
            [pair["proposed_geometry"][name] for name in feature_names], dtype=np.float64
        )
        prior_normalized = (prior_x - normalization["input_mean"]) / normalization["input_std"]
        proposed_normalized = (proposed_x - normalization["input_mean"]) / normalization["input_std"]
        direction = proposed_normalized - prior_normalized
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 0.0:
            continue
        unit_direction = direction / direction_norm
        x_tensor = torch.as_tensor(prior_normalized[None, :], dtype=dtype, device=device)
        v_tensor = torch.as_tensor(unit_direction[None, :], dtype=dtype, device=device)
        with torch.enable_grad():
            prediction_tensor, jvp_tensor = torch.autograd.functional.jvp(
                model, x_tensor, v_tensor, create_graph=False, strict=True
            )
        with torch.no_grad():
            proposed_prediction_tensor = model(
                torch.as_tensor(proposed_normalized[None, :], dtype=dtype, device=device)
            )
        prior_prediction = (
            prediction_tensor.detach().cpu().numpy()[0, :3] * output_std
            + normalization["output_mean"][:3]
        )
        proposed_prediction = (
            proposed_prediction_tensor.detach().cpu().numpy()[0, :3] * output_std
            + normalization["output_mean"][:3]
        )
        predicted_jvp = jvp_tensor.detach().cpu().numpy()[0, :3] * output_std
        prior_truth = np.asarray(
            [pair["prior_performance"][name] for name in OUTPUT_NAMES], dtype=np.float64
        )
        proposed_truth = np.asarray(
            [pair["proposed_performance"][name] for name in OUTPUT_NAMES], dtype=np.float64
        )
        target = np.asarray(
            [pair["target_performance"][name] for name in OUTPUT_NAMES], dtype=np.float64
        )
        denominator = np.asarray([tolerances[name] for name in OUTPUT_NAMES]) * np.maximum(
            np.abs(prior_truth), np.asarray([epsilons[name] for name in OUTPUT_NAMES])
        )
        true_direction = (proposed_truth - prior_truth) / denominator / direction_norm
        predicted_direction = predicted_jvp / denominator
        true_step = (proposed_truth - prior_truth) / denominator
        predicted_step = (proposed_prediction - prior_prediction) / denominator
        true_sign = np.sign(true_direction)
        predicted_sign = np.sign(predicted_direction)
        active = np.abs(true_direction) > sign_epsilon
        target_denominator = np.asarray([tolerances[name] for name in OUTPUT_NAMES]) * np.maximum(
            np.abs(target), np.asarray([epsilons[name] for name in OUTPUT_NAMES])
        )
        true_rho_before = _rho(prior_truth, target, target_denominator)
        true_rho_after = _rho(proposed_truth, target, target_denominator)
        predicted_rho_before = _rho(prior_prediction, target, target_denominator)
        predicted_rho_after = _rho(proposed_prediction, target, target_denominator)
        worst_axis = int(np.argmax(np.abs(true_direction)))
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "cosine": _cosine(true_direction, predicted_direction),
                "finite_step_cosine": _cosine(true_step, predicted_step),
                "axis_sign_correct": [
                    bool(true_sign[index] == predicted_sign[index]) if active[index] else None
                    for index in range(3)
                ],
                "worst_axis": OUTPUT_NAMES[worst_axis],
                "worst_axis_sign_correct": bool(
                    true_sign[worst_axis] == predicted_sign[worst_axis]
                ),
                "true_rho_delta": true_rho_before - true_rho_after,
                "predicted_rho_delta": predicted_rho_before - predicted_rho_after,
                "true_descent": bool(true_rho_after < true_rho_before),
                "predicted_descent": bool(predicted_rho_after < predicted_rho_before),
            }
        )
    cosines = [row["cosine"] for row in rows if row["cosine"] is not None]
    finite_cosines = [
        row["finite_step_cosine"] for row in rows if row["finite_step_cosine"] is not None
    ]
    predicted_descents = [row for row in rows if row["predicted_descent"]]
    true_descents = [row for row in rows if row["true_descent"]]
    axis_accuracy = {}
    for index, name in enumerate(OUTPUT_NAMES):
        values = [
            row["axis_sign_correct"][index]
            for row in rows if row["axis_sign_correct"][index] is not None
        ]
        axis_accuracy[name] = float(np.mean(values)) if values else None
    return {
        "pair_count": len(rows),
        "median_jvp_cosine": float(np.median(cosines)) if cosines else None,
        "positive_jvp_cosine_rate": float(np.mean(np.asarray(cosines) > 0.0)) if cosines else None,
        "median_finite_step_cosine": float(np.median(finite_cosines)) if finite_cosines else None,
        "axis_sign_accuracy": axis_accuracy,
        "worst_axis_sign_accuracy": float(
            np.mean([row["worst_axis_sign_correct"] for row in rows])
        ) if rows else None,
        "predicted_descent_count": len(predicted_descents),
        "true_descent_count": len(true_descents),
        "predicted_descent_precision": (
            float(np.mean([row["true_descent"] for row in predicted_descents]))
            if predicted_descents else None
        ),
        "true_descent_recall": (
            float(np.mean([row["predicted_descent"] for row in true_descents]))
            if true_descents else None
        ),
        "rows": rows,
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Gate L：代理梯度方向审计",
        "",
        f"> 连续真实方向对：{summary['pair_counts']['retained_pairs']}；新增V2.1调用：0。",
        "",
        "| 模型 | JVP cosine P50 | cosine>0 | TSFC符号 | 最差轴符号 | 预测下降precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy_id, result in summary["strategies"].items():
        lines.append(
            f"| {strategy_id} | {result['median_jvp_cosine']:.3f} | "
            f"{result['positive_jvp_cosine_rate']:.1%} | "
            f"{result['axis_sign_accuracy']['tsfc_kg_N_s']:.1%} | "
            f"{result['worst_axis_sign_accuracy']:.1%} | "
            f"{result['predicted_descent_precision']:.1%} |"
        )
    lines += [
        "",
        f"审计对象：`{summary['decision']['reference_strategy']}`。",
        f"判定：`{summary['decision']['decision']}`。",
        "",
    ]
    return "\n".join(lines)


def run_gate_l_audit(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    device: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    if config.get("protocol_version") != GATE_L_VERSION:
        raise Geometry19GateLError("Unsupported Gate L protocol")
    runtime = Path(output_root).resolve() if output_root else root / config["artifacts"]["runtime_root"]
    if runtime.exists():
        raise Geometry19GateLError(f"Refusing to overwrite Gate L run: {runtime}")
    runtime.mkdir(parents=True)
    gate_k_config = yaml.safe_load((root / config["gate_k_config"]).read_text())
    base = yaml.safe_load((root / gate_k_config["base_surrogate_protocol"]).read_text())
    inverse = yaml.safe_load((root / gate_k_config["inverse_protocol"]).read_text())
    schema = yaml.safe_load((root / base["design_schema"]).read_text())
    fields = list(schema["design_variables"])
    feature_names = [item["name"] for item in fields]
    pairs, pair_counts = collect_direction_pairs(
        root / config["trajectory_root"], fields,
        minimum_normalized_direction_norm=float(
            config["audit"]["minimum_normalized_direction_norm"]
        ),
    )
    pair_counts["retained_pairs"] = len(pairs)
    if len(pairs) < int(config["audit"]["minimum_direction_pairs"]):
        raise Geometry19GateLError("Insufficient continuous direction pairs")
    gate_k2_path = root / config["gate_k2_summary"]
    gate_k2 = json.loads(gate_k2_path.read_text())
    representatives = gate_k2["representatives"]
    first_manifest = json.loads(
        (Path(next(iter(representatives.values()))["model_directory"]) / "model_manifest.json").read_text()
    )
    target_names = list(first_manifest["target_names"])
    production_dir = root / base["artifacts"]["weights_root"] / "forward_mlp_48d_v1/seed_42/baseline"
    strategies = {
        "production": {
            "model_directory": str(production_dir),
            "model_tag": base["forward"]["model_tag"],
        },
        **{
            arm_id: {
                "model_directory": item["model_directory"],
                "model_tag": item["model_tag"],
            }
            for arm_id, item in representatives.items()
        },
    }
    run_device = str(device or config["device"])
    results = {}
    for strategy_id, strategy in strategies.items():
        model, normalization, _ = load_weight_bundle(
            strategy["model_directory"],
            expected_model_tag=strategy["model_tag"],
            expected_feature_names=feature_names,
            expected_target_names=target_names,
            device=run_device,
        )
        results[strategy_id] = audit_model_directions(
            pairs, feature_names, model, normalization,
            inverse["diagnostic_acceptance"]["relative_tolerances"],
            inverse["diagnostic_acceptance"]["epsilons"],
            sign_epsilon=float(config["audit"]["sign_epsilon"]),
        )
    reference = "s0_legacy_mse"
    reference_result = results[reference]
    trigger = config["audit"]["trigger"]
    failed_checks = {
        "median_cosine": reference_result["median_jvp_cosine"]
        < float(trigger["minimum_median_cosine"]),
        "tsfc_sign_accuracy": reference_result["axis_sign_accuracy"]["tsfc_kg_N_s"]
        < float(trigger["minimum_tsfc_sign_accuracy"]),
        "predicted_descent_precision": reference_result["predicted_descent_precision"]
        < float(trigger["minimum_predicted_descent_precision"]),
    }
    decision = "TRAIN_S4_DIRECTIONAL_GRADIENT" if any(failed_checks.values()) else "KEEP_VALUE_ONLY_PROXY"
    summary = {
        "protocol_version": GATE_L_VERSION,
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_use_allowed": False,
        "new_v2_1_calls": 0,
        "device": run_device,
        "pair_counts": pair_counts,
        "strategies": results,
        "decision": {
            "reference_strategy": reference,
            "failed_gradient_checks": failed_checks,
            "decision": decision,
        },
        "hashes": {
            "config_sha256": sha256_file(config_source),
            "gate_k2_summary_sha256": sha256_file(gate_k2_path),
        },
    }
    _write_jsonl(runtime / config["artifacts"]["pairs"], pairs)
    _write_json(runtime / config["artifacts"]["summary"], summary)
    report_path = runtime / config["artifacts"]["report"]
    report_path.write_text(_render_report(summary), encoding="utf-8")
    _write_json(
        runtime / config["artifacts"]["manifest"],
        {
            "protocol_version": GATE_L_VERSION,
            "summary_sha256": sha256_file(runtime / config["artifacts"]["summary"]),
            "pairs_sha256": sha256_file(runtime / config["artifacts"]["pairs"]),
            "report_sha256": sha256_file(report_path),
        },
    )
    return summary
