"""Low-mass corrective retraining with fresh globally root-disjoint confirmation."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
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
    _predict,
    _train_model,
    fit_normalization,
)
from src.ai.geometry19_v2_2_forward_baseline import (
    _predict_physical,
    _seed_all,
    _write_json,
    _write_text,
    evaluate_predictions,
)
from src.ai.geometry19_performance_surrogate_v2 import transform_outputs
from src.physics import Geometry19


PROTOCOL_VERSION = "geometry19_v2_2_low_mass_corrective_retraining_v1"
SPLIT_NAMES = ("train", "validation", "confirmation")


class Geometry19V22CorrectiveRetrainingError(ValueError):
    """Raised when corrective retraining violates its frozen contract."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_fresh_splits(
    root: Path,
    config: Mapping[str, Any],
    feature_names: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_path = root / config["split_manifest"]
    if sha256_file(manifest_path) != config["split_manifest_sha256"]:
        raise Geometry19V22CorrectiveRetrainingError("Fresh split manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "FRESH_ROOT_SPLIT_PASS"
        or manifest.get("training_ready") is not True
        or manifest.get("formal_use_allowed") is not False
        or manifest.get("grouping_key") != "root_seed_id"
    ):
        raise Geometry19V22CorrectiveRetrainingError("Fresh split contract is not training-ready")
    target_segments = tuple(str(value) for value in config["target_segments"])
    buckets: dict[str, dict[str, Any]] = {}
    roots_by_role: dict[str, set[str]] = {}
    designs: set[str] = set()
    for role in SPLIT_NAMES:
        path = root / config["split_datasets"][role]
        if sha256_file(path) != manifest["hashes"][f"{role}_split_hash"]:
            raise Geometry19V22CorrectiveRetrainingError(f"{role} split hash mismatch")
        rows = _read_jsonl(path)
        if len(rows) != int(manifest["counts"][role]["rows"]):
            raise Geometry19V22CorrectiveRetrainingError(f"{role} split count mismatch")
        role_designs = {str(row["design_hash"]) for row in rows}
        if designs & role_designs or len(role_designs) != len(rows):
            raise Geometry19V22CorrectiveRetrainingError("Design leakage across fresh roles")
        designs.update(role_designs)
        roots_by_role[role] = {str(row["root_seed_id"]) for row in rows}
        x = np.asarray(
            [[float(row["geometry19"][name]) for name in feature_names] for row in rows],
            dtype=np.float32,
        )
        y = np.asarray(
            [[float(row["performance"][name]) for name in PERFORMANCE_NAMES] for row in rows],
            dtype=np.float32,
        )
        if x.shape != (len(rows), 19) or y.shape != (len(rows), 3):
            raise Geometry19V22CorrectiveRetrainingError("Unexpected fresh split tensor shape")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y <= 0.0):
            raise Geometry19V22CorrectiveRetrainingError("Fresh split contains invalid values")
        buckets[role] = {"x": x, "y": y, "records": rows}
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            if roots_by_role[left] & roots_by_role[right]:
                raise Geometry19V22CorrectiveRetrainingError("Global root leakage across roles")
    for role in ("validation", "confirmation"):
        labels = {str(row["assigned_segment"]) for row in buckets[role]["records"]}
        if not set(target_segments) <= labels:
            raise Geometry19V22CorrectiveRetrainingError(
                f"{role} does not cover both corrective target segments"
            )
    return buckets, {
        "split_manifest_hash": sha256_file(manifest_path),
        "global_root_disjoint": True,
        "design_disjoint": True,
        "roots_by_role": {role: len(values) for role, values in roots_by_role.items()},
        "confirmation_roots_globally_fresh": True,
    }


def _confirmation_gate(
    baseline: Mapping[str, Any],
    challenger: Mapping[str, Any],
    policy: Mapping[str, Any],
    target_segments: Sequence[str],
) -> dict[str, Any]:
    baseline_overall = float(baseline["overall"]["rho_p50"])
    challenger_overall = float(challenger["overall"]["rho_p50"])
    overall_reduction = 1.0 - challenger_overall / baseline_overall
    segment_reductions = {
        segment: 1.0
        - float(challenger["by_segment"][segment]["rho_p50"])
        / float(baseline["by_segment"][segment]["rho_p50"])
        for segment in target_segments
    }
    checks = {
        "overall_rho_p50_reduction": overall_reduction
        >= float(policy["minimum_overall_rho_p50_reduction"]),
        "every_segment_rho_p50_reduction": min(segment_reductions.values())
        >= float(policy["minimum_each_segment_rho_p50_reduction"]),
        "worst_segment_rho_p95": float(challenger["worst_segment_rho_p95"])
        <= float(policy["maximum_worst_segment_rho_p95"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "overall_rho_p50_reduction": overall_reduction,
        "segment_rho_p50_reductions": segment_reductions,
        "challenger_worst_segment_rho_p95": float(challenger["worst_segment_rho_p95"]),
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    baseline = summary["confirmation"]["baseline"]
    challenger = summary["confirmation"]["challenger"]
    lines = [
        "# Geometry19 V2.2 低质量线段纠偏重训",
        "",
        f"- 状态：`{summary['status']}`",
        f"- 代表 seed：`{summary['selected_seed']}`（仅由 validation 选择）",
        f"- train / validation / confirmation：{summary['datasets']['train']} / {summary['datasets']['validation']} / {summary['datasets']['confirmation']}",
        "- confirmation：两个在旧 1298 条数据中全局未出现的 root_seed_id。",
        "- 正式用途与部署：禁止；不修改默认 pipeline。",
        "",
        "## Validation seed 选择",
        "",
        "| seed | worst-segment rho P95 | normalized RMSE | reload |",
        "|---:|---:|---:|---|",
    ]
    for run in summary["runs"]:
        validation = run["metrics"]["validation"]
        lines.append(
            f"| {run['seed']} | {validation['worst_segment_rho_p95']:.4f} | "
            f"{validation['overall']['normalized_rmse']:.4f} | "
            f"{str(run['weight_reload_equal']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Fresh confirmation 配对结果",
            "",
            "| 线段 | n/根 | 原 seed-43 rho P50/P95 | 纠偏模型 rho P50/P95 | P50 降幅 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for segment in summary["target_segments"]:
        before = baseline["by_segment"][segment]
        after = challenger["by_segment"][segment]
        reduction = summary["confirmation_gate"]["segment_rho_p50_reductions"][segment]
        lines.append(
            f"| `{segment}` | {after['sample_count']}/1 | "
            f"{before['rho_p50']:.4f}/{before['rho_p95']:.4f} | "
            f"{after['rho_p50']:.4f}/{after['rho_p95']:.4f} | {reduction:.1%} |"
        )
    lines.extend(
        [
            "",
            f"整体 confirmation rho P50 从 `{baseline['overall']['rho_p50']:.4f}` 降至 "
            f"`{challenger['overall']['rho_p50']:.4f}`，降幅 "
            f"`{summary['confirmation_gate']['overall_rho_p50_reduction']:.1%}`。",
            "",
            "confirmation 只在 validation 完成 seed 选择后评价一次，且从不参与 checkpoint、seed 或超参数选择。即使纠偏门通过，也仅说明两条低质量线段的开发性改善，不构成五线段或正式用途准入。",
            "",
        ]
    )
    return "\n".join(lines)


def run_corrective_retraining(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    device: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise Geometry19V22CorrectiveRetrainingError("Protocol version mismatch")
    if config.get("formal_use_allowed") is not False:
        raise Geometry19V22CorrectiveRetrainingError("Corrective retraining cannot be formal")
    schema_path = root / config["design_schema"]
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    feature_names = [item["name"] for item in schema["design_variables"]]
    if feature_names != list(Geometry19.coordinate_names()):
        raise Geometry19V22CorrectiveRetrainingError("Geometry19 feature order mismatch")
    domain_path = root / config["target_domain"]
    domain = yaml.safe_load(domain_path.read_text(encoding="utf-8"))
    target_segments = tuple(str(value) for value in config["target_segments"])
    tolerances = np.asarray(
        [float(domain["axes"][name]["relative_tolerance"]) for name in PERFORMANCE_NAMES],
        dtype=np.float64,
    )
    splits, split_audit = load_fresh_splits(root, config, feature_names)
    runtime = Path(output_root).resolve() if output_root else (
        root / config["artifacts"]["runtime_root"] / "formal"
    )
    if runtime.exists():
        raise Geometry19V22CorrectiveRetrainingError(f"Refusing to overwrite run: {runtime}")
    runtime.mkdir(parents=True)
    run_device = torch.device(device or config["training"]["device"])
    if run_device.type == "cuda" and not torch.cuda.is_available():
        raise Geometry19V22CorrectiveRetrainingError("CUDA was requested but is unavailable")
    training = copy.deepcopy(config["training"])
    architecture = config["architecture"]
    output_transform = str(config["model"]["output_transform"])
    transformed_train_y = transform_outputs(splits["train"]["y"], output_transform)
    normalization = fit_normalization(splits["train"]["x"], transformed_train_y)
    normalized_x = {
        role: _normalize_x(bucket["x"], normalization) for role, bucket in splits.items()
    }
    normalized_y = {
        role: (
            (transform_outputs(bucket["y"], output_transform) - normalization["output_mean"])
            / normalization["output_std"]
        ).astype(np.float32)
        for role, bucket in splits.items()
    }
    validation_indices = np.asarray(
        [
            index
            for index, row in enumerate(splits["validation"]["records"])
            if row["assigned_segment"] in target_segments
        ],
        dtype=int,
    )

    runs: list[dict[str, Any]] = []
    for seed_value in config["seeds"]:
        seed = int(seed_value)
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
                normalized_x["train"], normalized_y["train"],
                int(training["batch_size"]), True, seed,
            ),
            normalized_x["validation"][validation_indices],
            normalized_y["validation"][validation_indices],
            nn.functional.mse_loss,
            training,
            run_device,
        )
        train_prediction = _predict_physical(
            model, splits["train"]["x"], normalization, run_device, output_transform
        )
        validation_prediction = _predict_physical(
            model,
            splits["validation"]["x"][validation_indices],
            normalization,
            run_device,
            output_transform,
        )
        train_segments = tuple(
            segment["id"] for segment in domain["segments"]
        )
        metrics = {
            "train": evaluate_predictions(
                splits["train"]["y"], train_prediction, splits["train"]["records"],
                train_segments, tolerances, normalization["output_std"], output_transform,
            ),
            "validation": evaluate_predictions(
                splits["validation"]["y"][validation_indices], validation_prediction,
                [splits["validation"]["records"][index] for index in validation_indices],
                target_segments, tolerances,
                normalization["output_std"], output_transform,
            ),
        }
        model_tag = f"novaturbo_geometry19_v2_2_low_mass_corrective_seed{seed}_v1"
        model_dir = runtime / "models" / f"seed_{seed}"
        artifacts = save_weight_bundle(
            model_dir,
            model,
            {
                "checkpoint_version": "geometry19_v2_2_low_mass_corrective_v1",
                "model_tag": model_tag,
                "architecture_tag": architecture["architecture_tag"],
                "feature_names": feature_names,
                "target_names": PERFORMANCE_NAMES,
                "training_seed": seed,
                "output_transform": output_transform,
                "split_manifest_hash": split_audit["split_manifest_hash"],
                "training_protocol_hash": sha256_file(config_source),
                "formal_use_allowed": False,
                "deployment_allowed": False,
            },
            normalization,
        )
        _write_json(model_dir / "training_history.json", {"best_epoch": best_epoch, "history": history})
        _write_json(model_dir / "development_metrics.json", metrics)
        reloaded, reloaded_stats, _ = load_weight_bundle(
            model_dir,
            expected_model_tag=model_tag,
            expected_feature_names=feature_names,
            expected_target_names=PERFORMANCE_NAMES,
            device=run_device,
        )
        weight_reload_equal = np.array_equal(
            _predict(model, normalized_x["validation"][validation_indices], run_device),
            _predict(
                reloaded,
                _normalize_x(
                    splits["validation"]["x"][validation_indices], reloaded_stats
                ),
                run_device,
            ),
        )
        runs.append(
            {
                "seed": seed,
                "model_tag": model_tag,
                "best_epoch": best_epoch,
                "completed_epochs": len(history),
                "best_validation_loss": min(row["validation_loss"] for row in history),
                "weight_reload_equal": bool(weight_reload_equal),
                "artifacts": artifacts,
                "metrics": metrics,
            }
        )

    selected = min(
        runs,
        key=lambda run: (
            float(run["metrics"]["validation"]["worst_segment_rho_p95"]),
            float(run["metrics"]["validation"]["overall"]["normalized_rmse"]),
            int(run["seed"]),
        ),
    )
    selected_dir = runtime / "models" / f"seed_{selected['seed']}"
    selected_model, selected_stats, _ = load_weight_bundle(
        selected_dir,
        expected_model_tag=selected["model_tag"],
        expected_feature_names=feature_names,
        expected_target_names=PERFORMANCE_NAMES,
        device=run_device,
    )
    challenger_prediction = _predict_physical(
        selected_model,
        splits["confirmation"]["x"],
        selected_stats,
        run_device,
        output_transform,
    )
    challenger_metrics = evaluate_predictions(
        splits["confirmation"]["y"], challenger_prediction,
        splits["confirmation"]["records"], target_segments, tolerances,
        selected_stats["output_std"], output_transform,
    )

    baseline_dir = root / config["baseline_model_root"]
    baseline_manifest = json.loads(
        (baseline_dir / "model_manifest.json").read_text(encoding="utf-8")
    )
    baseline_model, baseline_stats, _ = load_weight_bundle(
        baseline_dir,
        expected_model_tag=baseline_manifest["model_tag"],
        expected_feature_names=feature_names,
        expected_target_names=PERFORMANCE_NAMES,
        device=run_device,
    )
    baseline_prediction = _predict_physical(
        baseline_model,
        splits["confirmation"]["x"],
        baseline_stats,
        run_device,
        str(baseline_manifest["output_transform"]),
    )
    baseline_metrics = evaluate_predictions(
        splits["confirmation"]["y"], baseline_prediction,
        splits["confirmation"]["records"], target_segments, tolerances,
        baseline_stats["output_std"], str(baseline_manifest["output_transform"]),
    )
    gate = _confirmation_gate(
        baseline_metrics, challenger_metrics, config["confirmation_gate"], target_segments
    )
    artifact_checks = all(run["weight_reload_equal"] for run in runs)
    status = (
        "CORRECTIVE_CONFIRMATION_PASS"
        if artifact_checks and gate["passed"]
        else "CORRECTIVE_CONFIRMATION_FAIL"
    )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "full_domain_confirmation_ready": False,
        "device": str(run_device),
        "datasets": {role: len(bucket["x"]) for role, bucket in splits.items()},
        "validation_target_rows": len(validation_indices),
        "target_segments": list(target_segments),
        "selected_seed": int(selected["seed"]),
        "selection": {
            "split": "validation",
            "primary_metric": "worst_segment_rho_p95",
            "secondary_metric": "overall.normalized_rmse",
            "confirmation_used_for_selection": False,
        },
        "confirmation_evaluated_once_after_selection": True,
        "split_audit": split_audit,
        "runs": runs,
        "confirmation": {
            "baseline_model_tag": baseline_manifest["model_tag"],
            "baseline": baseline_metrics,
            "challenger_model_tag": selected["model_tag"],
            "challenger": challenger_metrics,
        },
        "confirmation_gate": gate,
        "hashes": {
            "config": sha256_file(config_source),
            "schema": sha256_file(schema_path),
            "target_domain": sha256_file(domain_path),
            "split_manifest": split_audit["split_manifest_hash"],
            "baseline_manifest": sha256_file(baseline_dir / "model_manifest.json"),
            "implementation": sha256_file(Path(__file__)),
        },
        "code_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "code_worktree_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=root, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
        ),
    }
    _write_json(runtime / config["artifacts"]["summary"], summary)
    _write_text(runtime / config["artifacts"]["report"], _render_report(summary))
    return summary
