"""Post-hoc audit of the failed V2.2 low-mass corrective confirmation."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from src.ai.geometry19_surrogate import load_weight_bundle, sha256_file
from src.ai.geometry19_training import PERFORMANCE_NAMES
from src.ai.geometry19_v2_2_forward_baseline import (
    _predict_physical,
    _write_json,
    _write_text,
    evaluate_predictions,
)


PROTOCOL_VERSION = "geometry19_v2_2_corrective_failure_audit_v1"


class Geometry19V22CorrectiveAuditError(ValueError):
    """Raised when frozen corrective-audit evidence is inconsistent."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _nearest(query: np.ndarray, reference: np.ndarray) -> dict[str, dict[str, float]]:
    delta = query[:, None, :] - reference[None, :, :]
    return {
        "l2": _quantiles(np.min(np.linalg.norm(delta, axis=2), axis=1)),
        "linf": _quantiles(np.min(np.max(np.abs(delta), axis=2), axis=1)),
    }


def _root_concentration(rows: Sequence[dict[str, Any]], segment: str) -> dict[str, Any]:
    counts = Counter(
        str(row["root_seed_id"])
        for row in rows
        if row["assigned_segment"] == segment
    )
    values = np.asarray(list(counts.values()), dtype=float)
    shares = values / values.sum()
    return {
        "rows": int(values.sum()),
        "roots": len(values),
        "minimum_rows_per_root": int(values.min()),
        "maximum_rows_per_root": int(values.max()),
        "maximum_root_row_share": float(shares.max()),
        "effective_root_count_by_row_weight": float(1.0 / np.sum(shares**2)),
        "rows_by_root": dict(sorted(counts.items())),
    }


def _bias_metrics(
    truth: np.ndarray, prediction: np.ndarray
) -> dict[str, dict[str, float]]:
    signed = (prediction - truth) / truth
    absolute = np.abs(signed)
    return {
        name: {
            "signed_relative_error_p50": float(np.quantile(signed[:, index], 0.50)),
            "absolute_relative_error_p50": float(np.quantile(absolute[:, index], 0.50)),
            "absolute_relative_error_p95": float(np.quantile(absolute[:, index], 0.95)),
        }
        for index, name in enumerate(PERFORMANCE_NAMES)
    }


def _load_model(directory: Path, feature_names: Sequence[str], device: torch.device):
    manifest = json.loads((directory / "model_manifest.json").read_text(encoding="utf-8"))
    model, normalization, _ = load_weight_bundle(
        directory,
        expected_model_tag=manifest["model_tag"],
        expected_feature_names=feature_names,
        expected_target_names=PERFORMANCE_NAMES,
        device=device,
    )
    return model, normalization, manifest


def _render_report(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Geometry19 V2.2 纠偏失败后验审计",
        "",
        f"- 审计状态：`{audit['status']}`",
        f"- 决策：`{audit['decision']}`",
        "- confirmation：已揭盲，只用于解释失败，禁止继续选模或调参。",
        "- 默认 pipeline：不变。",
        "",
        "## 根级数据结构",
        "",
        "| 线段 | train 行/根/有效根 | validation 行/根 | confirmation 行/根 |",
        "|---|---:|---:|---:|",
    ]
    for segment in audit["target_segments"]:
        train = audit["root_concentration"]["train"][segment]
        validation = audit["root_concentration"]["validation"][segment]
        confirmation = audit["root_concentration"]["confirmation"][segment]
        lines.append(
            f"| `{segment}` | {train['rows']}/{train['roots']}/{train['effective_root_count_by_row_weight']:.2f} | "
            f"{validation['rows']}/{validation['roots']} | {confirmation['rows']}/{confirmation['roots']} |"
        )
    lines.extend(
        [
            "",
            f"网络参数量为 `{audit['model_capacity']['parameter_count']}`，train 全局根数为 "
            f"`{audit['model_capacity']['train_global_root_count']}`，约 "
            f"`{audit['model_capacity']['parameters_per_train_root']:.0f}` 参数/根。",
            "",
            "## 根外支持距离",
            "",
            "| 角色/线段 | 19D 最近 train L2 P50/P95 | 性能空间最近 train L∞ P50/P95 |",
            "|---|---:|---:|",
        ]
    )
    for role in ("validation", "confirmation"):
        for segment in audit["target_segments"]:
            support = audit["support_distance"][role][segment]
            design = support["design_row_to_train"]["l2"]
            performance = support["performance_row_to_train"]["linf"]
            lines.append(
                f"| {role} / `{segment}` | {design['p50']:.3f}/{design['p95']:.3f} | "
                f"{performance['p50']:.3f}/{performance['p95']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## 模型对照",
            "",
            "| 数据角色/线段 | 旧基线 rho P50/P95 | 纠偏模型 rho P50/P95 |",
            "|---|---:|---:|",
        ]
    )
    for role in ("validation", "confirmation"):
        for segment in audit["target_segments"]:
            baseline = audit["model_comparison"][role]["baseline"]["by_segment"][segment]
            challenger = audit["model_comparison"][role]["corrective"]["by_segment"][segment]
            lines.append(
                f"| {role} / `{segment}` | {baseline['rho_p50']:.3f}/{baseline['rho_p95']:.3f} | "
                f"{challenger['rho_p50']:.3f}/{challenger['rho_p95']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## 审计结论",
            "",
            "1. 训练已收敛但根外泛化间隙大：选定模型的 validation normalized RMSE 约为 train 的数倍，失败不是权重未保存或数值异常。",
            "2. validation 与 confirmation 对两模型的相对排序不一致。`low_mass_mid_to_high` 的 validation→train 19D 最近邻 L2 P50 为 0.911，而 confirmation 仅 0.130，两个 validation 根不能代表不同根外距离层级；当前 confirmation 每段也只有一个根，无法估计置信区间。",
            "3. 当前 loss 按行均匀加权，而验收对象是根外泛化；不同根拥有不同记录数，训练目标与审计单位不一致。",
            "4. 现有高容量 BatchNorm MLP 相对于独立根数存在容量风险；在孤立局部邻域数据上可能学习根簇特征，而不是稳定跨根响应。",
            "5. `low_mass_mid_to_high` confirmation 的退化主要来自推力中位有符号误差 -2.784%，刚超过 2.745% 接受容差；质量与 TSFC 中位绝对误差仍约 0.560%/0.355%。这要求训练 loss 与 rho 验收轴对齐，不能只看归一化 MSE。",
            "",
            "## 改进方向（按优先级）",
            "",
            "1. **先重做 development 选择器**：隔离已揭盲 confirmation，用现有 development 根做全局 root_seed_id GroupKFold，并按 19D 根中心距离分层；以多折 worst-root rho P95 选择方案，保留旧模型 no-change arm。",
            "2. **训练单位和目标同时对齐**：采用 root-balanced batch/loss，使每根等权；把 hit-aligned smooth-max rho 加入训练目标，避免某一轴刚越过容差而归一化 MSE 仍看似良好。",
            "3. **只在 development 做容量消融**：比较小型 MLP、去 BatchNorm/改 LayerNorm、岭回归或树模型；模型参数量应按有效根数约束，不能按行数解释容量。",
            "4. **补桥接而非邻域加密**：按归一化 19D 根中心距离和局部性能 Jacobian 选择新根/桥接点，优先填补 development 根簇之间的空隙。",
            "5. **最后才准备新确认集**：development 方案冻结后，每条低质量线段至少获取并冻结 3 个全新 confirmation 根，只允许一次最终评价；此前 `deployment_allowed=false`。",
            "",
        ]
    )
    return "\n".join(lines)


def run_corrective_failure_audit(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    device: str = "cpu",
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise Geometry19V22CorrectiveAuditError("Audit protocol version mismatch")
    if config.get("confirmation_reuse_allowed") is not False:
        raise Geometry19V22CorrectiveAuditError("Exposed confirmation cannot be reused")
    for spec_name in ("split_manifest", "baseline_summary", "corrective_summary"):
        spec = config[spec_name]
        if sha256_file(root / spec["path"]) != spec["sha256"]:
            raise Geometry19V22CorrectiveAuditError(f"Frozen {spec_name} hash mismatch")
    corrective_summary = json.loads(
        (root / config["corrective_summary"]["path"]).read_text(encoding="utf-8")
    )
    if corrective_summary["status"] != "CORRECTIVE_CONFIRMATION_FAIL":
        raise Geometry19V22CorrectiveAuditError("Audit requires a failed confirmation")

    schema = yaml.safe_load((root / config["design_schema"]).read_text(encoding="utf-8"))
    feature_names = [item["name"] for item in schema["design_variables"]]
    bounds = np.asarray([item["bounds"] for item in schema["design_variables"]], dtype=float)
    domain = yaml.safe_load((root / config["target_domain"]).read_text(encoding="utf-8"))
    tolerances = np.asarray(
        [float(domain["axes"][name]["relative_tolerance"]) for name in PERFORMANCE_NAMES]
    )
    target_segments = tuple(str(value) for value in config["target_segments"])
    rows_by_role = {
        role: _read_jsonl(root / path)
        for role, path in config["split_datasets"].items()
    }
    root_concentration = {
        role: {
            segment: _root_concentration(rows, segment)
            for segment in target_segments
        }
        for role, rows in rows_by_role.items()
    }

    def matrices(rows: Sequence[dict[str, Any]], segment: str):
        selected = [row for row in rows if row["assigned_segment"] == segment]
        x = np.asarray(
            [[float(row["geometry19"][name]) for name in feature_names] for row in selected]
        )
        y = np.asarray(
            [[float(row["performance"][name]) for name in PERFORMANCE_NAMES] for row in selected]
        )
        normalized_x = (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        scaled_y = np.log(y) / np.log1p(tolerances)[None, :]
        return selected, normalized_x, scaled_y

    support_distance: dict[str, Any] = {"validation": {}, "confirmation": {}}
    for segment in target_segments:
        train_rows, train_x, train_y = matrices(rows_by_role["train"], segment)
        train_centroids: dict[str, list[np.ndarray]] = defaultdict(list)
        for row, value in zip(train_rows, train_x, strict=True):
            train_centroids[str(row["root_seed_id"])].append(value)
        train_root_x = np.stack(
            [np.mean(values, axis=0) for values in train_centroids.values()]
        )
        for role in ("validation", "confirmation"):
            held_rows, held_x, held_y = matrices(rows_by_role[role], segment)
            held_centroids: dict[str, list[np.ndarray]] = defaultdict(list)
            for row, value in zip(held_rows, held_x, strict=True):
                held_centroids[str(row["root_seed_id"])].append(value)
            held_root_x = np.stack(
                [np.mean(values, axis=0) for values in held_centroids.values()]
            )
            support_distance[role][segment] = {
                "design_row_to_train": _nearest(held_x, train_x),
                "performance_row_to_train": _nearest(held_y, train_y),
                "design_root_centroid_to_train_root_centroid": _nearest(
                    held_root_x, train_root_x
                ),
            }

    run_device = torch.device(device)
    baseline_model, baseline_stats, baseline_manifest = _load_model(
        root / config["baseline_model_root"], feature_names, run_device
    )
    corrective_model, corrective_stats, corrective_manifest = _load_model(
        root / config["corrective_model_root"], feature_names, run_device
    )
    model_comparison: dict[str, Any] = {}
    prediction_bias: dict[str, Any] = {}
    for role in ("validation", "confirmation"):
        selected_rows = [
            row
            for row in rows_by_role[role]
            if row["assigned_segment"] in target_segments
        ]
        x = np.asarray(
            [[float(row["geometry19"][name]) for name in feature_names] for row in selected_rows],
            dtype=np.float32,
        )
        y = np.asarray(
            [[float(row["performance"][name]) for name in PERFORMANCE_NAMES] for row in selected_rows],
            dtype=np.float32,
        )
        predictions = {
            "baseline": _predict_physical(
                baseline_model, x, baseline_stats, run_device,
                str(baseline_manifest["output_transform"]),
            ),
            "corrective": _predict_physical(
                corrective_model, x, corrective_stats, run_device,
                str(corrective_manifest["output_transform"]),
            ),
        }
        model_comparison[role] = {
            name: evaluate_predictions(
                y, prediction, selected_rows, target_segments, tolerances,
                (baseline_stats if name == "baseline" else corrective_stats)["output_std"],
                str((baseline_manifest if name == "baseline" else corrective_manifest)["output_transform"]),
            )
            for name, prediction in predictions.items()
        }
        prediction_bias[role] = {}
        labels = np.asarray([row["assigned_segment"] for row in selected_rows], dtype=object)
        for segment in target_segments:
            indices = np.flatnonzero(labels == segment)
            prediction_bias[role][segment] = {
                name: _bias_metrics(y[indices], prediction[indices])
                for name, prediction in predictions.items()
            }

    parameter_count = sum(parameter.numel() for parameter in corrective_model.parameters())
    train_global_roots = len({row["root_seed_id"] for row in rows_by_role["train"]})
    selected_run = next(
        run
        for run in corrective_summary["runs"]
        if run["seed"] == corrective_summary["selected_seed"]
    )
    generalization_gap = {
        "train_normalized_rmse": selected_run["metrics"]["train"]["overall"]["normalized_rmse"],
        "validation_normalized_rmse": selected_run["metrics"]["validation"]["overall"]["normalized_rmse"],
        "validation_to_train_rmse_ratio": (
            selected_run["metrics"]["validation"]["overall"]["normalized_rmse"]
            / selected_run["metrics"]["train"]["overall"]["normalized_rmse"]
        ),
    }
    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "AUDIT_COMPLETE",
        "decision": "NO_MODEL_CHANGE_ROOT_GENERALIZATION_REDESIGN_REQUIRED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "confirmation_reuse_allowed": False,
        "target_segments": list(target_segments),
        "root_concentration": root_concentration,
        "support_distance": support_distance,
        "model_comparison": model_comparison,
        "prediction_bias": prediction_bias,
        "generalization_gap": generalization_gap,
        "model_capacity": {
            "parameter_count": parameter_count,
            "train_global_root_count": train_global_roots,
            "parameters_per_train_root": parameter_count / train_global_roots,
            "architecture": corrective_model.architecture_mapping(),
        },
        "improvement_priorities": [
            "distance_stratified_global_root_group_kfold_on_development_only",
            "root_balanced_hit_aligned_loss_and_worst_root_checkpoint_selection",
            "development_only_capacity_and_normalization_ablation",
            "design_space_bridge_sampling_between_root_clusters",
            "at_least_3_new_confirmation_roots_per_segment_only_after_selection_freeze",
        ],
        "hashes": {
            "config": sha256_file(config_source),
            "split_manifest": sha256_file(root / config["split_manifest"]["path"]),
            "baseline_summary": sha256_file(root / config["baseline_summary"]["path"]),
            "corrective_summary": sha256_file(root / config["corrective_summary"]["path"]),
            "baseline_model_manifest": sha256_file(
                root / config["baseline_model_root"] / "model_manifest.json"
            ),
            "corrective_model_manifest": sha256_file(
                root / config["corrective_model_root"] / "model_manifest.json"
            ),
            "implementation": sha256_file(Path(__file__)),
        },
    }
    output = Path(output_root).resolve() if output_root else root / config["artifacts"]["output_root"]
    if output.exists():
        raise Geometry19V22CorrectiveAuditError(f"Refusing to overwrite audit: {output}")
    output.mkdir(parents=True)
    _write_json(output / config["artifacts"]["summary"], audit)
    _write_text(output / config["artifacts"]["report"], _render_report(audit))
    return audit
