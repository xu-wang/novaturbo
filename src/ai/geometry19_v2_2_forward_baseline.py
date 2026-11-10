"""Leakage-safe 19D-to-performance baseline for the V2.2 development dataset."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import yaml

from src.ai.geometry19_performance_surrogate_v2 import (
    inverse_transform_outputs,
    performance_metrics,
    transform_outputs,
)
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
    _predict,
    _train_model,
    fit_normalization,
)
from src.physics import Geometry19


PROTOCOL_VERSION = "geometry19_v2_2_forward_baseline_v1"
SPLIT_NAMES = ("train", "validation", "test")


class Geometry19V22ForwardBaselineError(ValueError):
    """Raised when the V2.2 baseline contract or artifacts are inconsistent."""


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


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_v2_2_splits(
    root: Path,
    config: Mapping[str, Any],
    feature_names: Sequence[str],
    segment_names: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load and audit the frozen root-grouped split without re-splitting it."""
    manifest_path = root / config["dataset_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "COVERAGE_GATE_PASS"
        or manifest.get("training_ready") is not True
        or manifest.get("formal_use_allowed") is not False
    ):
        raise Geometry19V22ForwardBaselineError(
            "Dataset must pass the software coverage gate and remain non-formal"
        )

    buckets: dict[str, dict[str, Any]] = {}
    seen_designs: set[str] = set()
    group_splits: dict[tuple[str, str], str] = {}
    roots_by_split_segment: dict[str, dict[str, int]] = {}
    for split_name in SPLIT_NAMES:
        split_path = root / config["split_datasets"][split_name]
        expected_hash = manifest["hashes"][f"{split_name}_split_hash"]
        if sha256_file(split_path) != expected_hash:
            raise Geometry19V22ForwardBaselineError(
                f"Frozen {split_name} split hash mismatch"
            )
        rows = _read_jsonl(split_path)
        if len(rows) != int(manifest["counts"]["splits"][split_name]):
            raise Geometry19V22ForwardBaselineError(
                f"Frozen {split_name} split count mismatch"
            )
        x_rows: list[list[float]] = []
        y_rows: list[list[float]] = []
        observed_segments: set[str] = set()
        for row in rows:
            segment = str(row["assigned_segment"])
            if segment not in segment_names:
                raise Geometry19V22ForwardBaselineError(f"Unknown segment: {segment}")
            if not row.get("training_candidate"):
                raise Geometry19V22ForwardBaselineError("Split contains a non-training row")
            if not row["constraint_assessment"]["physics_domain_valid"]:
                raise Geometry19V22ForwardBaselineError("Split contains an invalid physics row")
            design_hash = str(row["design_hash"])
            if design_hash in seen_designs:
                raise Geometry19V22ForwardBaselineError("Design leakage across frozen splits")
            seen_designs.add(design_hash)
            group = (segment, str(row["root_seed_id"]))
            previous = group_splits.setdefault(group, split_name)
            if previous != split_name:
                raise Geometry19V22ForwardBaselineError("Root-neighborhood split leakage")
            x_rows.append([float(row["geometry19"][name]) for name in feature_names])
            y_rows.append([float(row["performance"][name]) for name in PERFORMANCE_NAMES])
            observed_segments.add(segment)
        if observed_segments != set(segment_names):
            raise Geometry19V22ForwardBaselineError(
                f"Split {split_name} does not cover every target-domain segment"
            )
        roots_by_split_segment[split_name] = {
            segment: len(
                {
                    str(row["root_seed_id"])
                    for row in rows
                    if row["assigned_segment"] == segment
                }
            )
            for segment in segment_names
        }
        x = np.asarray(x_rows, dtype=np.float32)
        y = np.asarray(y_rows, dtype=np.float32)
        if x.shape != (len(rows), 19) or y.shape != (len(rows), 3):
            raise Geometry19V22ForwardBaselineError("Unexpected V2.2 tensor shape")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y <= 0.0):
            raise Geometry19V22ForwardBaselineError("V2.2 baseline inputs must be finite and positive")
        buckets[split_name] = {"x": x, "y": y, "records": rows}

    if len(seen_designs) != int(manifest["counts"]["combined_accepted_unique"]):
        raise Geometry19V22ForwardBaselineError("Frozen splits do not exhaust the dataset")
    audit = {
        "dataset_version": manifest["dataset_version"],
        "dataset_manifest_hash": sha256_file(manifest_path),
        "split_hashes_verified": True,
        "design_hashes_unique_across_splits": True,
        "root_neighborhoods_unique_to_one_split": True,
        "all_segments_present_in_each_split": True,
        "group_count": len(group_splits),
        "design_count": len(seen_designs),
        "independent_roots_by_split_segment": roots_by_split_segment,
    }
    return buckets, audit


def evaluate_predictions(
    truth: np.ndarray,
    prediction: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    segment_names: Sequence[str],
    tolerances: Sequence[float],
    output_std: np.ndarray,
    output_transform: str,
) -> dict[str, Any]:
    """Return overall and per-segment physical-unit errors."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(prediction).all() or np.any(prediction <= 0.0):
        raise Geometry19V22ForwardBaselineError("Baseline predictions must be finite and positive")

    def metrics(indices: np.ndarray) -> dict[str, Any]:
        values = performance_metrics(truth[indices], prediction[indices], tolerances)
        transformed_truth = transform_outputs(truth[indices], output_transform)
        transformed_prediction = transform_outputs(prediction[indices], output_transform)
        normalized_residual = (transformed_prediction - transformed_truth) / output_std
        values["normalized_rmse"] = float(np.sqrt(np.mean(normalized_residual**2)))
        values["finite"] = True
        return values

    all_indices = np.arange(len(truth))
    by_segment: dict[str, Any] = {}
    labels = np.asarray([str(row["assigned_segment"]) for row in records], dtype=object)
    for segment in segment_names:
        indices = np.flatnonzero(labels == segment)
        if not len(indices):
            raise Geometry19V22ForwardBaselineError(f"No rows for segment {segment}")
        by_segment[segment] = metrics(indices)
    overall = metrics(all_indices)
    worst_segment = max(
        segment_names,
        key=lambda name: (by_segment[name]["rho_p95"], str(name)),
    )
    return {
        "overall": overall,
        "by_segment": by_segment,
        "worst_segment_by_rho_p95": str(worst_segment),
        "worst_segment_rho_p95": float(by_segment[worst_segment]["rho_p95"]),
    }


def select_seed(runs: Sequence[Mapping[str, Any]]) -> int:
    """Select only from validation metrics; test metrics are deliberately ignored."""
    if not runs:
        raise Geometry19V22ForwardBaselineError("No seed runs were produced")
    selected = min(
        runs,
        key=lambda run: (
            float(run["metrics"]["validation"]["worst_segment_rho_p95"]),
            float(run["metrics"]["validation"]["overall"]["normalized_rmse"]),
            int(run["seed"]),
        ),
    )
    return int(selected["seed"])


def _predict_physical(
    model: nn.Module,
    x: np.ndarray,
    normalization: Mapping[str, np.ndarray],
    device: torch.device,
    output_transform: str,
) -> np.ndarray:
    normalized_x = _normalize_x(x, normalization)
    normalized_prediction = _predict(model, normalized_x, device)
    transformed = (
        normalized_prediction * normalization["output_std"]
        + normalization["output_mean"]
    )
    return inverse_transform_outputs(transformed, output_transform).astype(np.float64)


def _render_report(summary: Mapping[str, Any]) -> str:
    selected = next(
        run for run in summary["runs"] if run["seed"] == summary["selected_seed"]
    )
    test = selected["metrics"]["test"]
    lines = [
        "# Geometry19 V2.2 三性能前向代理基线",
        "",
        f"- 状态：`{summary['status']}`",
        f"- 选择种子：`{summary['selected_seed']}`（只按 validation 选择）",
        f"- 数据：train / validation / test = {summary['datasets']['train']} / {summary['datasets']['validation']} / {summary['datasets']['test']}",
        "- 输出：19D → `thrust_N / tsfc_kg_N_s / mass_kg`，自然对数参数化。",
        "- 精度门：`NOT_PRE_REGISTERED_BASELINE_ONLY`；完成状态不代表精度通过。",
        "- 正式用途：禁止；本结果只适用于简化 V2.2 软件目标域。",
        "",
        "## 种子级结果",
        "",
        "| seed | validation worst-segment rho P95 | validation normalized RMSE | test rho P50 | test rho P95 | reload |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for run in summary["runs"]:
        validation = run["metrics"]["validation"]
        run_test = run["metrics"]["test"]["overall"]
        lines.append(
            f"| {run['seed']} | {validation['worst_segment_rho_p95']:.4f} | "
            f"{validation['overall']['normalized_rmse']:.4f} | {run_test['rho_p50']:.4f} | "
            f"{run_test['rho_p95']:.4f} | {str(run['weight_reload_equal']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## 选定种子的 test 分线段误差",
            "",
            "| 线段 | n | 独立根 | rho P50 | rho P95 | 推力相对误差 P50/P95 | TSFC 相对误差 P50/P95 | 质量相对误差 P50/P95 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in summary["segment_names"]:
        values = test["by_segment"][name]
        outputs = values["outputs"]
        root_count = summary["leakage_audit"]["independent_roots_by_split_segment"][
            "test"
        ][name]
        lines.append(
            f"| `{name}` | {values['sample_count']} | {root_count} | {values['rho_p50']:.4f} | "
            f"{values['rho_p95']:.4f} | "
            f"{outputs['thrust_N']['relative_error_p50']:.3%}/{outputs['thrust_N']['relative_error_p95']:.3%} | "
            f"{outputs['tsfc_kg_N_s']['relative_error_p50']:.3%}/{outputs['tsfc_kg_N_s']['relative_error_p95']:.3%} | "
            f"{outputs['mass_kg']['relative_error_p50']:.3%}/{outputs['mass_kg']['relative_error_p95']:.3%} |"
        )
    lines.extend(
        [
            "",
            f"整体 test normalized RMSE 为 `{test['overall']['normalized_rmse']:.6f}`，"
            f"rho P50/P95 为 `{test['overall']['rho_p50']:.4f}` / `{test['overall']['rho_p95']:.4f}`。",
            "",
            "分线段 test 指标的独立根支持数只有 1–2；行级样本共享根邻域，不能把 n 当作独立泛化样本数，也不足以给出稳定置信区间。",
            "",
            "根种子邻域未跨 split；归一化统计只由 train 拟合；epoch 早停和代表 seed 选择只使用 validation。test 仅在选择规则冻结后报告。代理没有物理接受权，也未接入默认 V2.1/V2.2 pipeline。",
            "",
        ]
    )
    return "\n".join(lines)


def run_v2_2_forward_baseline(
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
        raise Geometry19V22ForwardBaselineError("Protocol version mismatch")
    if config.get("formal_use_allowed") is not False:
        raise Geometry19V22ForwardBaselineError("V2.2 software baseline cannot be formal")

    schema_path = root / config["design_schema"]
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    feature_names = [item["name"] for item in schema["design_variables"]]
    if feature_names != list(Geometry19.coordinate_names()):
        raise Geometry19V22ForwardBaselineError("Geometry19 feature order mismatch")
    domain_path = root / config["target_domain"]
    domain = yaml.safe_load(domain_path.read_text(encoding="utf-8"))
    if domain.get("formal_use_allowed") is not False:
        raise Geometry19V22ForwardBaselineError("Target domain must remain non-formal")
    segment_names = [str(item["id"]) for item in domain["segments"]]
    tolerances = np.asarray(
        [float(domain["axes"][name]["relative_tolerance"]) for name in PERFORMANCE_NAMES],
        dtype=np.float64,
    )
    splits, leakage_audit = load_v2_2_splits(root, config, feature_names, segment_names)

    runtime = Path(output_root).resolve() if output_root else (
        root / config["artifacts"]["runtime_root"] / ("smoke" if smoke else "formal")
    )
    if runtime.exists():
        raise Geometry19V22ForwardBaselineError(f"Refusing to overwrite run: {runtime}")
    runtime.mkdir(parents=True)
    run_device = torch.device(device or config["training"]["device"])
    if run_device.type == "cuda" and not torch.cuda.is_available():
        raise Geometry19V22ForwardBaselineError("CUDA was requested but is unavailable")
    training = copy.deepcopy(config["training"])
    seeds = [int(value) for value in (config["smoke"]["seeds"] if smoke else config["seeds"])]
    if smoke:
        training.update(config["smoke"]["training"])
    architecture = config["architecture"]
    output_transform = str(config["model"]["output_transform"])
    implementation_hash = sha256_file(Path(__file__))

    transformed_train_y = transform_outputs(splits["train"]["y"], output_transform)
    normalization = fit_normalization(splits["train"]["x"], transformed_train_y)
    normalized_x = {
        name: _normalize_x(bucket["x"], normalization) for name, bucket in splits.items()
    }
    normalized_y = {
        name: (
            (transform_outputs(bucket["y"], output_transform) - normalization["output_mean"])
            / normalization["output_std"]
        ).astype(np.float32)
        for name, bucket in splits.items()
    }

    runs: list[dict[str, Any]] = []
    for seed in seeds:
        _seed_all(seed)
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
                normalized_x["train"],
                normalized_y["train"],
                int(training["batch_size"]),
                True,
                seed,
            ),
            normalized_x["validation"],
            normalized_y["validation"],
            nn.functional.mse_loss,
            training,
            run_device,
        )
        predictions = {
            name: _predict_physical(
                model, bucket["x"], normalization, run_device, output_transform
            )
            for name, bucket in splits.items()
        }
        metrics = {
            name: evaluate_predictions(
                bucket["y"],
                predictions[name],
                bucket["records"],
                segment_names,
                tolerances,
                normalization["output_std"],
                output_transform,
            )
            for name, bucket in splits.items()
        }
        model_tag = f"novaturbo_geometry19_v2_2_performance3_log_seed{seed}_baseline_v1"
        model_dir = runtime / "models" / f"seed_{seed}"
        artifacts = save_weight_bundle(
            model_dir,
            model,
            {
                "checkpoint_version": "geometry19_v2_2_performance3_baseline_v1",
                "model_tag": model_tag,
                "architecture_tag": architecture["architecture_tag"],
                "feature_names": feature_names,
                "target_names": PERFORMANCE_NAMES,
                "training_seed": seed,
                "output_transform": output_transform,
                "dataset_version": leakage_audit["dataset_version"],
                "dataset_manifest_hash": leakage_audit["dataset_manifest_hash"],
                "training_protocol_hash": sha256_file(config_source),
                "implementation_hash": implementation_hash,
                "formal_use_allowed": False,
                "deployment_allowed": False,
            },
            normalization,
        )
        _write_json(model_dir / "training_history.json", {"best_epoch": best_epoch, "history": history})
        _write_json(model_dir / "metrics.json", metrics)
        reloaded, reloaded_normalization, _ = load_weight_bundle(
            model_dir,
            expected_model_tag=model_tag,
            expected_feature_names=feature_names,
            expected_target_names=PERFORMANCE_NAMES,
            device=run_device,
        )
        original_output = _predict(model, normalized_x["test"], run_device)
        reloaded_output = _predict(
            reloaded,
            _normalize_x(splits["test"]["x"], reloaded_normalization),
            run_device,
        )
        runs.append(
            {
                "seed": seed,
                "model_tag": model_tag,
                "best_epoch": best_epoch,
                "completed_epochs": len(history),
                "best_validation_loss": min(row["validation_loss"] for row in history),
                "weight_reload_equal": bool(np.array_equal(original_output, reloaded_output)),
                "artifacts": artifacts,
                "metrics": metrics,
            }
        )

    selected_seed = select_seed(runs)
    all_checks_passed = all(
        run["weight_reload_equal"]
        and all(
            run["metrics"][split]["overall"]["finite"] for split in SPLIT_NAMES
        )
        for run in runs
    )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "BASELINE_COMPLETE" if all_checks_passed else "BASELINE_ARTIFACT_CHECK_FAILED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "accuracy_gate_status": "NOT_PRE_REGISTERED_BASELINE_ONLY",
        "smoke": smoke,
        "device": str(run_device),
        "seeds": seeds,
        "selected_seed": selected_seed,
        "selection": {
            "split": "validation",
            "primary_metric": "worst_segment_rho_p95",
            "secondary_metric": "overall.normalized_rmse",
            "test_used_for_selection": False,
        },
        "feature_names": feature_names,
        "performance_names": PERFORMANCE_NAMES,
        "segment_names": segment_names,
        "tolerances": dict(zip(PERFORMANCE_NAMES, tolerances.tolist(), strict=True)),
        "datasets": {name: len(bucket["x"]) for name, bucket in splits.items()},
        "leakage_audit": leakage_audit,
        "normalization_fit_split": "train",
        "runs": runs,
        "hashes": {
            "config": sha256_file(config_source),
            "schema": sha256_file(schema_path),
            "target_domain": sha256_file(domain_path),
            "dataset_manifest": leakage_audit["dataset_manifest_hash"],
            "implementation": implementation_hash,
        },
        "code_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "code_worktree_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
    }
    _write_json(runtime / config["artifacts"]["summary"], summary)
    _write_text(runtime / config["artifacts"]["report"], _render_report(summary))
    return summary
