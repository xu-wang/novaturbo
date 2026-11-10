"""Controlled Geometry19 performance-only surrogate ablation.

This experiment deliberately changes one factor at a time relative to Gate K:
first remove the 45 constraint outputs, then change only the representation of
the three positive performance outputs to log space.  Existing Gate K S0
models are reused as the shared-48 baselines.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import yaml

from src.ai.geometry19_surrogate import (
    TaggedGeometry19MLP,
    load_weight_bundle,
    save_weight_bundle,
    sha256_file,
)
from src.ai.geometry19_training import (
    PERFORMANCE_NAMES,
    _loader,
    _normalize_x,
    _train_model,
    fit_normalization,
    load_regression_splits,
)
from src.physics import Geometry19


PROTOCOL_VERSION = "wp19_22_performance_surrogate_v2_ablation"


class Geometry19PerformanceSurrogateV2Error(ValueError):
    """Raised when the controlled performance-surrogate experiment is invalid."""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def transform_outputs(values: np.ndarray, mode: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if mode == "identity":
        return array.copy()
    if mode == "natural_log":
        if np.any(array <= 0.0):
            raise Geometry19PerformanceSurrogateV2Error(
                "natural_log requires strictly positive performance outputs"
            )
        return np.log(array).astype(np.float32)
    raise Geometry19PerformanceSurrogateV2Error(f"Unsupported output transform: {mode}")


def inverse_transform_outputs(values: np.ndarray, mode: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if mode == "identity":
        return array.copy()
    if mode == "natural_log":
        return np.exp(array).astype(np.float32)
    raise Geometry19PerformanceSurrogateV2Error(f"Unsupported output transform: {mode}")


def performance_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    tolerances: Sequence[float],
) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    tolerance = np.asarray(tolerances, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 3:
        raise Geometry19PerformanceSurrogateV2Error("Performance arrays must have shape [n,3]")
    relative = np.abs(prediction - truth) / np.maximum(np.abs(truth), 1.0e-15)
    axis_ratios = relative / tolerance[None, :]
    rho = axis_ratios.max(axis=1)
    outputs = {}
    for index, name in enumerate(PERFORMANCE_NAMES):
        residual = prediction[:, index] - truth[:, index]
        denominator = np.sum((truth[:, index] - truth[:, index].mean()) ** 2)
        outputs[name] = {
            "relative_error_p50": float(np.quantile(relative[:, index], 0.50)),
            "relative_error_p90": float(np.quantile(relative[:, index], 0.90)),
            "relative_error_p95": float(np.quantile(relative[:, index], 0.95)),
            "normalized_error_p50": float(np.quantile(axis_ratios[:, index], 0.50)),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "r2": float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else None,
        }
    return {
        "sample_count": int(len(truth)),
        "outputs": outputs,
        "rho_p50": float(np.quantile(rho, 0.50)),
        "rho_p90": float(np.quantile(rho, 0.90)),
        "rho_p95": float(np.quantile(rho, 0.95)),
        "strict_prediction_rate": float(np.mean(rho <= 1.0)),
    }


def ranking_metrics(
    predicted_rho: np.ndarray,
    true_rho: np.ndarray,
    target_ids: Sequence[str],
    budgets: Sequence[int],
) -> dict[str, Any]:
    predicted = np.asarray(predicted_rho, dtype=np.float64)
    truth = np.asarray(true_rho, dtype=np.float64)
    targets = np.asarray(target_ids, dtype=object)
    rows = []
    for target_id in sorted(set(targets.tolist())):
        mask = targets == target_id
        order = np.argsort(predicted[mask], kind="stable")
        ordered_true = truth[mask][order]
        hit_positions = np.flatnonzero(ordered_true <= 1.0)
        rows.append(
            {
                "target_id": target_id,
                "candidate_count": int(mask.sum()),
                "true_hit_count": int(np.sum(ordered_true <= 1.0)),
                "first_hit_rank": int(hit_positions[0] + 1) if len(hit_positions) else None,
                "coverage": {
                    str(budget): bool(np.any(ordered_true[:budget] <= 1.0))
                    for budget in budgets
                },
            }
        )
    return {
        "target_count": len(rows),
        "targets_with_true_hits": sum(row["true_hit_count"] > 0 for row in rows),
        "target_coverage": {
            str(budget): sum(row["coverage"][str(budget)] for row in rows)
            for budget in budgets
        },
        "first_hit_ranks": [row["first_hit_rank"] for row in rows],
        "targets": rows,
    }


def _predict(
    model: nn.Module,
    x: np.ndarray,
    normalization: Mapping[str, np.ndarray],
    device: torch.device,
    output_transform: str,
    *,
    shared48: bool = False,
) -> np.ndarray:
    normalized_x = _normalize_x(x.astype(np.float32), normalization)
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(normalized_x), 4096):
            values.append(
                model(torch.from_numpy(normalized_x[start : start + 4096]).to(device))
                .cpu()
                .numpy()
            )
    normalized_y = np.concatenate(values)
    transformed = (
        normalized_y * normalization["output_std"] + normalization["output_mean"]
    )
    if shared48:
        transformed = transformed[:, :3]
    return inverse_transform_outputs(transformed, output_transform).astype(np.float64)


def _load_locked_references(path: Path, feature_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = _read_jsonl(path)
    x = np.asarray(
        [[row["reference_x_19"][name] for name in feature_names] for row in rows],
        dtype=np.float32,
    )
    y = np.asarray(
        [[row["target"][name] for name in PERFORMANCE_NAMES] for row in rows],
        dtype=np.float32,
    )
    return x, y


def _load_gate_h_pool(
    candidate_root: Path,
    truth_root: Path,
    feature_names: Sequence[str],
    tolerances: np.ndarray,
) -> dict[str, Any]:
    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    target_values: list[list[float]] = []
    target_ids: list[str] = []
    seen: set[tuple[str, str]] = set()
    for target_dir in sorted(path for path in truth_root.iterdir() if path.is_dir()):
        target_path = candidate_root / target_dir.name / "target.yaml"
        pool_path = target_dir / "full_pool_evaluations.jsonl"
        if not target_path.is_file() or not pool_path.is_file():
            continue
        target = yaml.safe_load(target_path.read_text(encoding="utf-8"))
        target_vector = [float(target[name]) for name in PERFORMANCE_NAMES]
        for row in _read_jsonl(pool_path):
            evaluation = row["evaluation"]
            design = evaluation.get("geometry19")
            performance = evaluation.get("performance")
            if not evaluation.get("physics_acceptable") or not design or not performance:
                continue
            key = (str(target["target_id"]), str(evaluation["design_hash"]))
            if key in seen:
                continue
            seen.add(key)
            x_rows.append([design[name] for name in feature_names])
            y_rows.append([performance[name] for name in PERFORMANCE_NAMES])
            target_values.append(target_vector)
            target_ids.append(str(target["target_id"]))
    if not x_rows:
        raise Geometry19PerformanceSurrogateV2Error("Gate H pool has no physics-acceptable rows")
    truth = np.asarray(y_rows, dtype=np.float64)
    targets = np.asarray(target_values, dtype=np.float64)
    true_relative = np.abs(truth - targets) / np.maximum(np.abs(targets), 1.0e-15)
    return {
        "x": np.asarray(x_rows, dtype=np.float32),
        "y": truth,
        "targets": targets,
        "target_ids": target_ids,
        "true_rho": (true_relative / tolerances[None, :]).max(axis=1),
    }


def _load_model(
    directory: Path,
    feature_names: Sequence[str],
    device: torch.device,
) -> tuple[TaggedGeometry19MLP, dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads((directory / "model_manifest.json").read_text(encoding="utf-8"))
    return load_weight_bundle(
        directory,
        expected_model_tag=manifest["model_tag"],
        expected_feature_names=feature_names,
        expected_target_names=manifest["target_names"],
        device=device,
    )


def _evaluate_model(
    model: nn.Module,
    normalization: Mapping[str, np.ndarray],
    device: torch.device,
    output_transform: str,
    tolerances: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    locked_x: np.ndarray,
    locked_y: np.ndarray,
    gate_h: Mapping[str, Any],
    budgets: Sequence[int],
    *,
    shared48: bool = False,
) -> dict[str, Any]:
    test_prediction = _predict(
        model, test_x, normalization, device, output_transform, shared48=shared48
    )
    locked_prediction = _predict(
        model, locked_x, normalization, device, output_transform, shared48=shared48
    )
    pool_prediction = _predict(
        model, gate_h["x"], normalization, device, output_transform, shared48=shared48
    )
    target_values = np.asarray(gate_h["targets"], dtype=np.float64)
    predicted_relative = np.abs(pool_prediction - target_values) / np.maximum(
        np.abs(target_values), 1.0e-15
    )
    predicted_rho = (predicted_relative / tolerances[None, :]).max(axis=1)
    return {
        "test": performance_metrics(test_y, test_prediction, tolerances),
        "locked_reference": performance_metrics(locked_y, locked_prediction, tolerances),
        "gate_h_pool": {
            "performance": performance_metrics(gate_h["y"], pool_prediction, tolerances),
            "ranking": ranking_metrics(
                predicted_rho,
                gate_h["true_rho"],
                gate_h["target_ids"],
                budgets,
            ),
        },
    }


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _aggregate(runs: Sequence[Mapping[str, Any]], budgets: Sequence[int]) -> dict[str, Any]:
    return {
        "run_count": len(runs),
        "test_rho_p50_median": _median([run["metrics"]["test"]["rho_p50"] for run in runs]),
        "locked_rho_p50_median": _median(
            [run["metrics"]["locked_reference"]["rho_p50"] for run in runs]
        ),
        "test_axis_p50_median": {
            name: _median(
                [run["metrics"]["test"]["outputs"][name]["relative_error_p50"] for run in runs]
            )
            for name in PERFORMANCE_NAMES
        },
        "locked_axis_p50_median": {
            name: _median(
                [
                    run["metrics"]["locked_reference"]["outputs"][name]["relative_error_p50"]
                    for run in runs
                ]
            )
            for name in PERFORMANCE_NAMES
        },
        "gate_h_target_coverage_median": {
            str(budget): _median(
                [
                    run["metrics"]["gate_h_pool"]["ranking"]["target_coverage"][str(budget)]
                    for run in runs
                ]
            )
            for budget in budgets
        },
    }


def _promotion(
    baseline: Mapping[str, Any],
    arm: Mapping[str, Any],
    policy: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    baseline_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def reduction(key: str) -> float:
        return 1.0 - float(arm[key]) / float(baseline[key])

    test_rho_reduction = reduction("test_rho_p50_median")
    locked_rho_reduction = reduction("locked_rho_p50_median")
    tsfc_reduction = 1.0 - (
        float(arm["test_axis_p50_median"]["tsfc_kg_N_s"])
        / float(baseline["test_axis_p50_median"]["tsfc_kg_N_s"])
    )
    other_worsening = {
        name: float(arm["test_axis_p50_median"][name])
        / float(baseline["test_axis_p50_median"][name])
        - 1.0
        for name in ("thrust_N", "mass_kg")
    }
    positive_seeds = sum(
        float(run["metrics"]["test"]["rho_p50"])
        < float(base["metrics"]["test"]["rho_p50"])
        and float(run["metrics"]["locked_reference"]["rho_p50"])
        < float(base["metrics"]["locked_reference"]["rho_p50"])
        for run, base in zip(runs, baseline_runs, strict=True)
    )
    coverage_ok = all(
        float(arm["gate_h_target_coverage_median"][key])
        >= float(baseline["gate_h_target_coverage_median"][key])
        for key in baseline["gate_h_target_coverage_median"]
    )
    checks = {
        "test_rho": test_rho_reduction >= float(policy["minimum_test_rho_p50_reduction"]),
        "locked_rho": locked_rho_reduction >= float(policy["minimum_locked_rho_p50_reduction"]),
        "tsfc": tsfc_reduction >= float(policy["minimum_tsfc_p50_reduction"]),
        "other_axes": max(other_worsening.values())
        <= float(policy["maximum_other_axis_p50_worsening"]),
        "positive_seeds": positive_seeds >= int(policy["minimum_positive_seed_count"]),
        "gate_h_coverage": coverage_ok
        if bool(policy["require_gate_h_recall_non_degradation"])
        else True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "test_rho_p50_reduction": test_rho_reduction,
        "locked_rho_p50_reduction": locked_rho_reduction,
        "tsfc_p50_reduction": tsfc_reduction,
        "other_axis_worsening": other_worsening,
        "positive_seed_count": positive_seeds,
    }


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# WP19-22 独立三输出性能代理消融",
        "",
        f"> 状态：**{summary['status']}**  ",
        "> 只比较共享48输出、独立3输出和独立3输出log参数化；未改变数据、隐藏层或训练预算。",
        "",
        "## 聚合结果",
        "",
        "| Arm | Test rho P50 | Locked-24 rho P50 | 推力P50 | TSFC P50 | 质量P50 | Gate-H Top32/64/128/256 | Gate |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for arm_id, arm in summary["arms"].items():
        aggregate = arm["aggregate"]
        axis = aggregate["test_axis_p50_median"]
        coverage = aggregate["gate_h_target_coverage_median"]
        gate = "BASELINE" if arm_id == summary["baseline_arm"] else (
            "PASS" if arm["promotion"]["passed"] else "FAIL"
        )
        lines.append(
            f"| {arm_id} | {aggregate['test_rho_p50_median']:.3f} | "
            f"{aggregate['locked_rho_p50_median']:.3f} | {axis['thrust_N']:.3%} | "
            f"{axis['tsfc_kg_N_s']:.3%} | {axis['mass_kg']:.3%} | "
            f"{coverage['32']:.0f}/{coverage['64']:.0f}/{coverage['128']:.0f}/{coverage['256']:.0f} | {gate} |"
        )
    lines.extend(
        [
            "",
            "## 判定边界",
            "",
            "Gate H 四目标已经被历史实验重复使用，只作为候选排序回归，不参与checkpoint选择。",
            "24个新锁定参考点来自隔离test campaign，只用于最终盲评。任何单seed改善均不允许替换默认代理。",
            "",
        ]
    )
    return "\n".join(lines)


def run_performance_surrogate_v2(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    smoke: bool = False,
    device: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise Geometry19PerformanceSurrogateV2Error("Protocol version mismatch")
    base_protocol_path = root / config["base_surrogate_protocol"]
    base_protocol = yaml.safe_load(base_protocol_path.read_text(encoding="utf-8"))
    inverse_protocol_path = root / config["inverse_protocol"]
    inverse_protocol = yaml.safe_load(inverse_protocol_path.read_text(encoding="utf-8"))
    schema_path = root / base_protocol["design_schema"]
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    feature_names = [item["name"] for item in schema["design_variables"]]
    if feature_names != list(Geometry19.coordinate_names()):
        raise Geometry19PerformanceSurrogateV2Error("Geometry19 feature order mismatch")
    maximum = int(config["smoke"]["maximum_rows_per_split"]) if smoke else None
    splits, target_names = load_regression_splits(
        root / config["regression_dataset"],
        feature_names,
        base_protocol["campaign_split"],
        maximum,
    )
    if target_names[:3] != PERFORMANCE_NAMES:
        raise Geometry19PerformanceSurrogateV2Error("Performance target order mismatch")
    tolerances = np.asarray(
        [inverse_protocol["diagnostic_acceptance"]["relative_tolerances"][name] for name in PERFORMANCE_NAMES],
        dtype=np.float64,
    )
    locked_x, locked_y = _load_locked_references(
        root / config["locked_reference_targets"], feature_names
    )
    gate_h = _load_gate_h_pool(
        root / config["gate_h_candidate_root"],
        root / config["gate_h_truth_root"],
        feature_names,
        tolerances,
    )
    runtime = Path(output_root).resolve() if output_root else (
        root / config["artifacts"]["runtime_root"] / ("smoke" if smoke else "formal")
    )
    if runtime.exists():
        raise Geometry19PerformanceSurrogateV2Error(f"Refusing to overwrite run: {runtime}")
    runtime.mkdir(parents=True)
    run_device = torch.device(device or config["training"]["device"])
    training = copy.deepcopy(config["training"])
    seeds = list(config["smoke"]["seeds"] if smoke else config["seeds"])
    if smoke:
        training.update(config["smoke"]["training"])
    architecture = base_protocol["architecture"]
    budgets = [int(value) for value in config["ranking_budgets"]]
    arms: dict[str, Any] = {}

    baseline_id = str(config["baseline"]["arm_id"])
    baseline_runs = []
    for seed in seeds:
        model_dir = root / config["baseline"]["model_root"] / f"seed_{seed}"
        model, normalization, manifest = _load_model(model_dir, feature_names, run_device)
        metrics = _evaluate_model(
            model,
            normalization,
            run_device,
            "identity",
            tolerances,
            splits["test"]["x"],
            splits["test"]["y"][:, :3],
            locked_x,
            locked_y,
            gate_h,
            budgets,
            shared48=True,
        )
        baseline_runs.append({"seed": seed, "model_tag": manifest["model_tag"], "metrics": metrics})
    arms[baseline_id] = {"runs": baseline_runs, "aggregate": _aggregate(baseline_runs, budgets)}

    for arm_spec in config["arms"]:
        arm_id = str(arm_spec["arm_id"])
        mode = str(arm_spec["output_transform"])
        runs = []
        for seed in seeds:
            _seed_all(seed)
            train_y_physical = splits["train"]["y"][:, :3]
            transformed_train_y = transform_outputs(train_y_physical, mode)
            normalization = fit_normalization(splits["train"]["x"], transformed_train_y)
            x_values = {
                split: _normalize_x(bucket["x"], normalization)
                for split, bucket in splits.items()
            }
            y_values = {
                split: (
                    transform_outputs(bucket["y"][:, :3], mode) - normalization["output_mean"]
                )
                / normalization["output_std"]
                for split, bucket in splits.items()
            }
            y_values = {key: value.astype(np.float32) for key, value in y_values.items()}
            model = TaggedGeometry19MLP(
                19,
                3,
                architecture["hidden_layers"],
                architecture["activation"],
                architecture["dropout"],
                architecture["batch_norm"],
            ).to(run_device)
            model, history, best_epoch = _train_model(
                model,
                _loader(
                    x_values["train"],
                    y_values["train"],
                    int(training["batch_size"]),
                    True,
                    seed,
                ),
                x_values["validation"],
                y_values["validation"],
                nn.functional.mse_loss,
                training,
                run_device,
            )
            model_tag = f"novaturbo_geometry19_{arm_id}_seed{seed}_v1"
            model_dir = runtime / "models" / arm_id / f"seed_{seed}"
            artifacts = save_weight_bundle(
                model_dir,
                model,
                {
                    "checkpoint_version": "geometry19_performance3_v1",
                    "model_tag": model_tag,
                    "architecture_tag": f"{architecture['architecture_tag']}_performance3",
                    "feature_names": feature_names,
                    "target_names": PERFORMANCE_NAMES,
                    "training_seed": seed,
                    "output_transform": mode,
                    "formal_use_allowed": False,
                    "protocol_hash": sha256_file(config_source),
                },
                normalization,
            )
            metrics = _evaluate_model(
                model,
                normalization,
                run_device,
                mode,
                tolerances,
                splits["test"]["x"],
                splits["test"]["y"][:, :3],
                locked_x,
                locked_y,
                gate_h,
                budgets,
            )
            runs.append(
                {
                    "seed": seed,
                    "model_tag": model_tag,
                    "output_transform": mode,
                    "best_epoch": best_epoch,
                    "completed_epochs": len(history),
                    "best_validation_loss": min(row["validation_loss"] for row in history),
                    "artifacts": artifacts,
                    "metrics": metrics,
                }
            )
            _write_json(model_dir / "training_history.json", history)
        aggregate = _aggregate(runs, budgets)
        arms[arm_id] = {
            "output_transform": mode,
            "runs": runs,
            "aggregate": aggregate,
            "promotion": _promotion(
                arms[baseline_id]["aggregate"],
                aggregate,
                config["promotion"],
                runs,
                baseline_runs,
            ),
        }

    passed = [arm_id for arm_id, value in arms.items() if value.get("promotion", {}).get("passed")]
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "PASS_ARM_AVAILABLE" if passed else "NO_ARM_PASSED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_use_allowed": False,
        "smoke": smoke,
        "baseline_arm": baseline_id,
        "passed_arms": passed,
        "seeds": seeds,
        "feature_names": feature_names,
        "performance_names": PERFORMANCE_NAMES,
        "tolerances": dict(zip(PERFORMANCE_NAMES, tolerances.tolist(), strict=True)),
        "datasets": {
            "train": len(splits["train"]["x"]),
            "validation": len(splits["validation"]["x"]),
            "test": len(splits["test"]["x"]),
            "locked_reference": len(locked_x),
            "gate_h_pool": len(gate_h["x"]),
        },
        "arms": arms,
        "hashes": {
            "config": sha256_file(config_source),
            "base_protocol": sha256_file(base_protocol_path),
            "inverse_protocol": sha256_file(inverse_protocol_path),
            "locked_reference_targets": sha256_file(root / config["locked_reference_targets"]),
        },
        "code_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }
    _write_json(runtime / config["artifacts"]["summary"], summary)
    _write_text(runtime / config["artifacts"]["report"], render_report(summary))
    return summary
