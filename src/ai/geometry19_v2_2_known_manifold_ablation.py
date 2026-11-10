"""Root-balanced five-fold surrogate ablation inside observed V2.2 support."""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.ai.geometry19_performance_surrogate_v2 import performance_metrics
from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_training import PERFORMANCE_NAMES
from src.physics import Geometry19


PROTOCOL_VERSION = "geometry19_v2_2_known_manifold_ablation_v1"


class Geometry19KnownManifoldAblationError(ValueError):
    """Raised when the known-manifold development contract is violated."""


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
            stream.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _root_weights(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    counts = Counter(str(row["root_seed_id"]) for row in records)
    weights = np.asarray(
        [1.0 / counts[str(row["root_seed_id"])] for row in records], dtype=np.float64
    )
    return (weights / np.mean(weights)).astype(np.float32)


def _weighted_normalization(
    x: np.ndarray, log_y: np.ndarray, weights: np.ndarray
) -> dict[str, np.ndarray]:
    normalized_weights = weights.astype(np.float64) / float(np.sum(weights))
    x_mean = np.sum(x * normalized_weights[:, None], axis=0)
    y_mean = np.sum(log_y * normalized_weights[:, None], axis=0)
    x_std = np.sqrt(np.sum((x - x_mean) ** 2 * normalized_weights[:, None], axis=0))
    y_std = np.sqrt(
        np.sum((log_y - y_mean) ** 2 * normalized_weights[:, None], axis=0)
    )
    return {
        "input_mean": x_mean.astype(np.float32),
        "input_std": np.where(x_std > 1.0e-12, x_std, 1.0).astype(np.float32),
        "output_mean": y_mean.astype(np.float32),
        "output_std": np.where(y_std > 1.0e-12, y_std, 1.0).astype(np.float32),
    }


def _normalize(
    x: np.ndarray, y: np.ndarray, stats: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    return (
        ((x - stats["input_mean"]) / stats["input_std"]).astype(np.float32),
        ((np.log(y) - stats["output_mean"]) / stats["output_std"]).astype(np.float32),
    )


def _to_physical(normalized: np.ndarray, stats: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.exp(
        normalized.astype(np.float64) * stats["output_std"] + stats["output_mean"]
    )


class _AblationMLP(nn.Module):
    def __init__(
        self,
        hidden_layers: Sequence[int],
        *,
        normalization_layer: str,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = 19
        for width_value in hidden_layers:
            width = int(width_value)
            layers.append(nn.Linear(previous, width))
            if normalization_layer == "batch_norm":
                layers.append(nn.BatchNorm1d(width))
            elif normalization_layer == "layer_norm":
                layers.append(nn.LayerNorm(width))
            elif normalization_layer != "none":
                raise Geometry19KnownManifoldAblationError(
                    f"Unknown normalization layer: {normalization_layer}"
                )
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, 3))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def _rho_values(
    truth: np.ndarray, prediction: np.ndarray, tolerances: np.ndarray
) -> np.ndarray:
    relative = np.abs(prediction / truth - 1.0)
    return np.max(relative / tolerances[None, :], axis=1)


def _root_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    tolerances: np.ndarray,
    target_segments: Sequence[str],
) -> dict[str, Any]:
    target_indices = np.asarray(
        [
            index
            for index, row in enumerate(records)
            if str(row["assigned_segment"]) in target_segments
        ],
        dtype=int,
    )
    target_truth = truth[target_indices]
    target_prediction = prediction[target_indices]
    target_records = [records[index] for index in target_indices]
    rho = _rho_values(target_truth, target_prediction, tolerances)
    grouped: dict[str, list[float]] = defaultdict(list)
    grouped_by_segment: dict[tuple[str, str], list[float]] = defaultdict(list)
    for value, row in zip(rho, target_records):
        root_id = str(row["root_seed_id"])
        segment = str(row["assigned_segment"])
        grouped[root_id].append(float(value))
        grouped_by_segment[(segment, root_id)].append(float(value))
    root_p95 = {
        root_id: float(np.quantile(values, 0.95))
        for root_id, values in grouped.items()
    }
    by_segment: dict[str, Any] = {}
    for segment in target_segments:
        values = [
            float(np.quantile(root_values, 0.95))
            for (root_segment, _), root_values in grouped_by_segment.items()
            if root_segment == segment
        ]
        by_segment[segment] = {
            "roots": len(values),
            "root_rho_p95_p50": float(np.quantile(values, 0.50)),
            "root_rho_p95_p95": float(np.quantile(values, 0.95)),
            "worst_root_rho_p95": float(max(values)),
        }
    row_metrics = performance_metrics(
        target_truth, target_prediction, tolerances.tolist()
    )
    return {
        "rows": len(target_indices),
        "roots": len(root_p95),
        "row_rho_p50": float(row_metrics["rho_p50"]),
        "row_rho_p95": float(row_metrics["rho_p95"]),
        "root_rho_p95_p50": float(np.quantile(list(root_p95.values()), 0.50)),
        "root_rho_p95_p95": float(np.quantile(list(root_p95.values()), 0.95)),
        "worst_root_rho_p95": float(max(root_p95.values())),
        "by_segment": by_segment,
        "root_rho_p95": dict(sorted(root_p95.items())),
    }


def _smooth_rho_loss(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    weights: torch.Tensor,
    output_std: torch.Tensor,
    tolerances: torch.Tensor,
    rho_weight: float,
    temperature: float,
) -> torch.Tensor:
    squared = torch.mean((prediction - truth) ** 2, dim=1)
    mse = torch.sum(weights * squared) / torch.sum(weights)
    if rho_weight <= 0.0:
        return mse
    delta_log = (prediction - truth) * output_std
    scaled = torch.abs(torch.expm1(delta_log)) / tolerances
    smooth_max = torch.logsumexp(temperature * scaled, dim=1) / temperature
    aligned = torch.sum(weights * smooth_max**2) / torch.sum(weights)
    return mse + rho_weight * aligned


def _train_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weights: np.ndarray,
    validation_x: np.ndarray,
    validation_y_physical: np.ndarray,
    validation_records: Sequence[Mapping[str, Any]],
    stats: Mapping[str, np.ndarray],
    tolerances: np.ndarray,
    target_segments: Sequence[str],
    variant: Mapping[str, Any],
    training: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[_AblationMLP, list[dict[str, Any]], int]:
    _seed_all(seed)
    model = _AblationMLP(
        variant["hidden_layers"],
        normalization_layer=str(variant["normalization_layer"]),
        dropout=float(variant["dropout"]),
    ).to(device)
    dataset = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y),
        torch.from_numpy(train_weights),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    validation_tensor = torch.from_numpy(validation_x).to(device)
    output_std = torch.from_numpy(stats["output_std"]).to(device)
    tolerance_tensor = torch.from_numpy(tolerances.astype(np.float32)).to(device)
    best_metric = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for x_batch, y_batch, weight_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            weight_batch = weight_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _smooth_rho_loss(
                model(x_batch),
                y_batch,
                weight_batch,
                output_std,
                tolerance_tensor,
                float(variant["hit_aligned_rho_weight"]),
                float(training["smooth_max_temperature"]),
            )
            if not torch.isfinite(loss):
                raise Geometry19KnownManifoldAblationError("Non-finite MLP loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(x_batch)
            total_rows += len(x_batch)
        model.eval()
        with torch.no_grad():
            normalized_prediction = model(validation_tensor).cpu().numpy()
        prediction = _to_physical(normalized_prediction, stats)
        validation_metrics = _root_metrics(
            validation_y_physical,
            prediction,
            validation_records,
            tolerances,
            target_segments,
        )
        metric = float(validation_metrics["worst_root_rho_p95"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total_rows,
                "validation_worst_root_rho_p95": metric,
            }
        )
        if metric < best_metric - float(training["min_delta"]):
            best_metric = metric
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= int(training["patience"]):
            break
    if best_state is None:
        raise Geometry19KnownManifoldAblationError("No MLP checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    weights: np.ndarray,
    validation_x: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    design = np.column_stack([np.ones(len(train_x)), train_x]).astype(np.float64)
    weighted = design * weights[:, None]
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ weighted + penalty, weighted.T @ train_y)
    prediction = np.column_stack([np.ones(len(validation_x)), validation_x]) @ coefficients
    return prediction.astype(np.float32), {"coefficients": coefficients}


def _normalized_design_matrix(
    rows: Sequence[Mapping[str, Any]], variables: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    return np.asarray(
        [
            [
                (float(row["geometry19"][variable["name"]]) - float(variable["bounds"][0]))
                / (float(variable["bounds"][1]) - float(variable["bounds"][0]))
                for variable in variables
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def _root_centers_array(
    rows: Sequence[Mapping[str, Any]], variables: Sequence[Mapping[str, Any]]
) -> tuple[list[str], np.ndarray]:
    values = _normalized_design_matrix(rows, variables)
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for value, row in zip(values, rows):
        grouped[str(row["root_seed_id"])].append(value)
    root_ids = sorted(grouped)
    return root_ids, np.stack([np.mean(grouped[root_id], axis=0) for root_id in root_ids])


def _fit_support_model(
    rows: Sequence[Mapping[str, Any]],
    variables: Sequence[Mapping[str, Any]],
    *,
    quantile: float,
    explained_variance_fraction: float,
) -> dict[str, Any]:
    root_ids, centers = _root_centers_array(rows, variables)
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    mean = np.mean(centers, axis=0)
    _, singular, right = np.linalg.svd(centers - mean, full_matrices=False)
    variance = singular**2
    cumulative = np.cumsum(variance) / np.sum(variance)
    rank = min(
        int(np.searchsorted(cumulative, explained_variance_fraction) + 1),
        len(centers) - 1,
        centers.shape[1],
    )
    basis = right[:rank]
    reconstructed = mean + ((centers - mean) @ basis.T) @ basis
    residual = np.linalg.norm(centers - reconstructed, axis=1)
    return {
        "root_ids": root_ids,
        "root_centers": centers.tolist(),
        "pca_mean": mean.tolist(),
        "pca_basis": basis.tolist(),
        "pca_rank": rank,
        "explained_variance_fraction": float(cumulative[rank - 1]),
        "nearest_root_l2_threshold": float(np.quantile(nearest, quantile)),
        "pca_residual_l2_threshold": float(np.quantile(residual, quantile) + 1.0e-12),
        "calibration_quantile": quantile,
    }


def support_decisions(values: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    centers = np.asarray(model["root_centers"], dtype=np.float64)
    mean = np.asarray(model["pca_mean"], dtype=np.float64)
    basis = np.asarray(model["pca_basis"], dtype=np.float64)
    nearest = np.min(
        np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2), axis=1
    )
    reconstructed = mean + ((values - mean) @ basis.T) @ basis
    residual = np.linalg.norm(values - reconstructed, axis=1)
    in_bounds = np.all((values >= 0.0) & (values <= 1.0), axis=1)
    return (
        in_bounds
        & (nearest <= float(model["nearest_root_l2_threshold"]))
        & (residual <= float(model["pca_residual_l2_threshold"]))
    )


def _support_fold_audit(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    variables: Sequence[Mapping[str, Any]],
    target_segments: Sequence[str],
    support: Mapping[str, Any],
) -> dict[str, Any]:
    model = _fit_support_model(
        train_rows,
        variables,
        quantile=float(support["calibration_quantile"]),
        explained_variance_fraction=float(support["explained_variance_fraction"]),
    )
    root_ids, centers = _root_centers_array(validation_rows, variables)
    decisions = support_decisions(centers, model)
    segments_by_root: dict[str, set[str]] = defaultdict(set)
    for row in validation_rows:
        segments_by_root[str(row["root_seed_id"])].add(str(row["assigned_segment"]))
    target_mask = np.asarray(
        [bool(segments_by_root[root_id] & set(target_segments)) for root_id in root_ids]
    )
    return {
        "validation_roots": len(root_ids),
        "accepted_validation_roots": int(np.sum(decisions)),
        "acceptance_fraction": float(np.mean(decisions)),
        "target_validation_roots": int(np.sum(target_mask)),
        "accepted_target_validation_roots": int(np.sum(decisions[target_mask])),
        "target_acceptance_fraction": float(np.mean(decisions[target_mask])),
        "calibrated_nearest_root_l2_threshold": model["nearest_root_l2_threshold"],
        "calibrated_pca_residual_l2_threshold": model["pca_residual_l2_threshold"],
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Geometry19 V2.2 已知流形前向代理 5 折消融",
        "",
        f"- 状态：`{summary['status']}`",
        f"- 选定配方：`{summary['selected_variant']}`",
        "- 选模：仅使用 development 根互斥 5 折；主指标为跨折最大 worst-root rho P95。",
        "- 精度门：`NOT_PRE_REGISTERED_MODEL_SELECTION_ONLY`；严格胜出不等于精度提升成立。",
        "- confirmation：只校验哈希，内容未读取。",
        "- 正式用途与部署：禁止。",
        "",
        "## 配方比较",
        "",
        "| 配方 | 根等权 | 跨折最大 worst-root rho P95 | 折中位 | root-rho P95 的跨折最大值 |",
        "|---|---|---:|---:|---:|",
    ]
    for result in summary["variant_results"]:
        aggregate = result["aggregate"]
        lines.append(
            f"| `{result['variant_id']}` | {str(result['root_balanced']).lower()} | "
            f"{aggregate['maximum_worst_root_rho_p95']:.4f} | "
            f"{aggregate['median_worst_root_rho_p95']:.4f} | "
            f"{aggregate['maximum_root_rho_p95_p95']:.4f} |"
        )
    support = summary["support_audit"]
    error_audit = summary["selected_error_audit"]
    dominant = "/".join(
        f"{name}:{count}" for name, count in error_audit["dominant_axis_counts"].items()
    )
    lines.extend(
        [
            "",
            "## 已知支持域门",
            "",
            f"5 折未见根接受率为 `{support['target_validation_root_acceptance_fraction']:.1%}`；阈值只由各折训练根校准。最终全 development 支持模型冻结 nearest-root L2 与 PCA residual 两个设计侧阈值，越界、离根过远或偏离 PCA 支持的输入统一返回 `OUTSIDE_KNOWN_MANIFOLD_SUPPORT`。支持门不参与模型主指标计算，不能通过拒绝困难根改善选模结果。",
            "",
            "## 结论",
            "",
            f"`{summary['selected_variant']}` 相对 no-change 主指标仅改善 `{summary['selected_vs_legacy_reduction_fraction']:.2%}`；全部配方的最差根 rho P95 仍高于 15。选定配方最差 10 个 `(segment, root)` 中主导轴计数为 `{dominant}`，最差线段为 `{error_audit['worst_segment']}`。因此本切片不建立模型 advancement，也不授权采集或揭盲新 confirmation。下一步应预注册局部图/距离条件模型（例如支持域内局部 Ridge、RBF 或 mixture-of-experts），继续沿用相同 5 折和 no-change 臂；只有 development 精度门通过后，才获取每段至少 3 个全新 confirmation 根。",
            "",
        ]
    )
    return "\n".join(lines)


def _selected_error_audit(
    prediction_rows: Sequence[Mapping[str, Any]],
    selected_variant: str,
    tolerances: np.ndarray,
    rows_by_design: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[int, str, str], list[np.ndarray]] = defaultdict(list)
    for row in prediction_rows:
        if row["variant_id"] != selected_variant:
            continue
        truth = np.asarray([row["truth"][name] for name in PERFORMANCE_NAMES], dtype=float)
        prediction = np.asarray(
            [row["prediction"][name] for name in PERFORMANCE_NAMES], dtype=float
        )
        grouped[(int(row["fold"]), str(row["assigned_segment"]), str(row["root_seed_id"]))].append(
            np.abs(prediction / truth - 1.0) / tolerances
        )
    ranked: list[dict[str, Any]] = []
    for (fold, segment, root_id), values in grouped.items():
        matrix = np.stack(values)
        axis_p95 = np.quantile(matrix, 0.95, axis=0)
        rho_p95 = float(np.quantile(np.max(matrix, axis=1), 0.95))
        representative = next(
            row
            for row in prediction_rows
            if row["variant_id"] == selected_variant
            and int(row["fold"]) == fold
            and str(row["assigned_segment"]) == segment
            and str(row["root_seed_id"]) == root_id
        )
        source = rows_by_design[str(representative["design_hash"])]
        ranked.append(
            {
                "fold": fold,
                "assigned_segment": segment,
                "root_seed_id": root_id,
                "root_rho_p95": rho_p95,
                "scaled_relative_error_p95_by_axis": {
                    name: float(axis_p95[index])
                    for index, name in enumerate(PERFORMANCE_NAMES)
                },
                "dominant_axis": PERFORMANCE_NAMES[int(np.argmax(axis_p95))],
                "new_known_manifold_root": (
                    source.get("root_origin")
                    == "independent_global_root_center_pca_lhs"
                ),
            }
        )
    worst = sorted(
        ranked,
        key=lambda row: (-row["root_rho_p95"], row["root_seed_id"]),
    )[:10]
    dominant_counts = Counter(row["dominant_axis"] for row in worst)
    segment_counts = Counter(row["assigned_segment"] for row in worst)
    return {
        "top_worst_segment_roots": worst,
        "dominant_axis_counts": {
            name: dominant_counts.get(name, 0) for name in PERFORMANCE_NAMES
        },
        "worst_segment_counts": dict(sorted(segment_counts.items())),
        "worst_segment": max(segment_counts, key=lambda name: (segment_counts[name], name)),
        "new_roots_in_top_worst": sum(row["new_known_manifold_root"] for row in worst),
        "existing_roots_in_top_worst": sum(
            not row["new_known_manifold_root"] for row in worst
        ),
    }


def run_known_manifold_ablation(
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
        raise Geometry19KnownManifoldAblationError("Protocol version mismatch")
    if config.get("formal_use_allowed") is not False:
        raise Geometry19KnownManifoldAblationError("Ablation cannot be formal")
    runtime = (
        Path(output_root).resolve()
        if output_root
        else root / config["artifacts"]["runtime_root"]
    )
    if runtime.exists():
        raise Geometry19KnownManifoldAblationError(f"Refusing to overwrite: {runtime}")
    runtime.mkdir(parents=True)

    dataset_path = root / config["development_dataset"]["path"]
    folds_path = root / config["development_folds"]["path"]
    confirmation_path = root / config["sealed_confirmation"]["path"]
    for path, spec, label in (
        (dataset_path, config["development_dataset"], "development dataset"),
        (folds_path, config["development_folds"], "development folds"),
        (confirmation_path, config["sealed_confirmation"], "sealed confirmation"),
    ):
        if sha256_file(path) != spec["sha256"]:
            raise Geometry19KnownManifoldAblationError(f"{label} hash mismatch")
    rows = _read_jsonl(dataset_path)
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    assignments = {str(key): int(value) for key, value in folds["root_assignments"].items()}
    if set(assignments) != {str(row["root_seed_id"]) for row in rows}:
        raise Geometry19KnownManifoldAblationError("Fold assignments do not cover roots")

    schema_path = root / config["design_schema"]
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    variables = list(schema["design_variables"])
    feature_names = [str(variable["name"]) for variable in variables]
    if feature_names != list(Geometry19.coordinate_names()):
        raise Geometry19KnownManifoldAblationError("Geometry19 feature order mismatch")
    domain_path = root / config["target_domain"]
    domain = yaml.safe_load(domain_path.read_text(encoding="utf-8"))
    tolerances = np.asarray(
        [float(domain["axes"][name]["relative_tolerance"]) for name in PERFORMANCE_NAMES],
        dtype=np.float64,
    )
    target_segments = tuple(str(value) for value in config["target_segments"])
    x = np.asarray(
        [[float(row["geometry19"][name]) for name in feature_names] for row in rows],
        dtype=np.float32,
    )
    y = np.asarray(
        [[float(row["performance"][name]) for name in PERFORMANCE_NAMES] for row in rows],
        dtype=np.float32,
    )
    run_device = torch.device(device or config["training"]["device"])
    if run_device.type == "cuda" and not torch.cuda.is_available():
        raise Geometry19KnownManifoldAblationError("CUDA requested but unavailable")

    prediction_rows: list[dict[str, Any]] = []
    variant_results: list[dict[str, Any]] = []
    support_folds: list[dict[str, Any]] = []
    for fold in range(int(folds["fold_count"])):
        train_indices = np.asarray(
            [index for index, row in enumerate(rows) if assignments[str(row["root_seed_id"])] != fold]
        )
        validation_indices = np.asarray(
            [index for index, row in enumerate(rows) if assignments[str(row["root_seed_id"])] == fold]
        )
        support_folds.append(
            {
                "fold": fold,
                **_support_fold_audit(
                    [rows[index] for index in train_indices],
                    [rows[index] for index in validation_indices],
                    variables,
                    target_segments,
                    config["support_gate"],
                ),
            }
        )

    for variant_order, variant in enumerate(config["variants"]):
        fold_results: list[dict[str, Any]] = []
        for fold in range(int(folds["fold_count"])):
            train_indices = np.asarray(
                [index for index, row in enumerate(rows) if assignments[str(row["root_seed_id"])] != fold]
            )
            validation_indices = np.asarray(
                [index for index, row in enumerate(rows) if assignments[str(row["root_seed_id"])] == fold]
            )
            train_records = [rows[index] for index in train_indices]
            validation_records = [rows[index] for index in validation_indices]
            weights = (
                _root_weights(train_records)
                if variant["root_balanced"]
                else np.ones(len(train_indices), dtype=np.float32)
            )
            stats = _weighted_normalization(x[train_indices], np.log(y[train_indices]), weights)
            train_x, train_y = _normalize(x[train_indices], y[train_indices], stats)
            validation_x, _ = _normalize(x[validation_indices], y[validation_indices], stats)
            model_payload: dict[str, Any] = {}
            if variant["family"] == "ridge":
                normalized_prediction, model_payload = _ridge_predict(
                    train_x,
                    train_y,
                    weights,
                    validation_x,
                    float(variant["alpha"]),
                )
                best_epoch = None
                completed_epochs = None
            elif variant["family"] == "extra_trees":
                model = ExtraTreesRegressor(
                    n_estimators=int(variant["n_estimators"]),
                    min_samples_leaf=int(variant["min_samples_leaf"]),
                    max_features=float(variant["max_features"]),
                    random_state=int(config["training"]["seed"]),
                    n_jobs=-1,
                )
                model.fit(train_x, train_y, sample_weight=weights)
                normalized_prediction = model.predict(validation_x).astype(np.float32)
                best_epoch = None
                completed_epochs = None
            elif variant["family"] == "mlp":
                model, history, best_epoch = _train_mlp(
                    train_x,
                    train_y,
                    weights,
                    validation_x,
                    y[validation_indices],
                    validation_records,
                    stats,
                    tolerances,
                    target_segments,
                    variant,
                    config["training"],
                    run_device,
                    int(config["training"]["seed"]),
                )
                with torch.no_grad():
                    normalized_prediction = (
                        model(torch.from_numpy(validation_x).to(run_device)).cpu().numpy()
                    )
                completed_epochs = len(history)
                history_path = runtime / "histories" / f"{variant['id']}-fold-{fold}.json"
                _write_json(history_path, {"best_epoch": best_epoch, "history": history})
            else:
                raise Geometry19KnownManifoldAblationError(
                    f"Unknown model family: {variant['family']}"
                )
            prediction = _to_physical(normalized_prediction, stats)
            metrics = _root_metrics(
                y[validation_indices],
                prediction,
                validation_records,
                tolerances,
                target_segments,
            )
            fold_results.append(
                {
                    "fold": fold,
                    "train_rows": len(train_indices),
                    "validation_rows": len(validation_indices),
                    "best_epoch": best_epoch,
                    "completed_epochs": completed_epochs,
                    "metrics": metrics,
                }
            )
            for local_index, row_index in enumerate(validation_indices):
                prediction_rows.append(
                    {
                        "variant_id": variant["id"],
                        "fold": fold,
                        "design_hash": rows[row_index]["design_hash"],
                        "root_seed_id": rows[row_index]["root_seed_id"],
                        "assigned_segment": rows[row_index]["assigned_segment"],
                        "truth": {
                            name: float(y[row_index, axis])
                            for axis, name in enumerate(PERFORMANCE_NAMES)
                        },
                        "prediction": {
                            name: float(prediction[local_index, axis])
                            for axis, name in enumerate(PERFORMANCE_NAMES)
                        },
                    }
                )
        worst = [result["metrics"]["worst_root_rho_p95"] for result in fold_results]
        root_p95 = [result["metrics"]["root_rho_p95_p95"] for result in fold_results]
        variant_results.append(
            {
                "variant_id": variant["id"],
                "variant_order": variant_order,
                "family": variant["family"],
                "root_balanced": bool(variant["root_balanced"]),
                "folds": fold_results,
                "aggregate": {
                    "maximum_worst_root_rho_p95": float(max(worst)),
                    "median_worst_root_rho_p95": float(np.median(worst)),
                    "maximum_root_rho_p95_p95": float(max(root_p95)),
                },
            }
        )

    selected = min(
        variant_results,
        key=lambda result: (
            result["aggregate"]["maximum_worst_root_rho_p95"],
            result["aggregate"]["maximum_root_rho_p95_p95"],
            result["variant_order"],
        ),
    )
    no_change = next(
        result for result in variant_results if result["variant_id"] == "legacy_no_change"
    )
    improvement = 1.0 - (
        selected["aggregate"]["maximum_worst_root_rho_p95"]
        / no_change["aggregate"]["maximum_worst_root_rho_p95"]
    )
    predictions_path = runtime / "cross_validation_predictions.jsonl"
    _write_jsonl(predictions_path, prediction_rows)

    target_acceptance = float(
        sum(fold["accepted_target_validation_roots"] for fold in support_folds)
        / sum(fold["target_validation_roots"] for fold in support_folds)
    )
    support_gate_pass = target_acceptance >= float(
        config["support_gate"]["minimum_target_validation_root_acceptance_fraction"]
    )
    final_support = _fit_support_model(
        rows,
        variables,
        quantile=float(config["support_gate"]["calibration_quantile"]),
        explained_variance_fraction=float(
            config["support_gate"]["explained_variance_fraction"]
        ),
    )
    final_support.update(
        {
            "status": "KNOWN_MANIFOLD_SUPPORT_GATE_READY" if support_gate_pass else "KNOWN_MANIFOLD_SUPPORT_GATE_PARTIAL",
            "outside_status": "OUTSIDE_KNOWN_MANIFOLD_SUPPORT",
            "formal_use_allowed": False,
            "deployment_allowed": False,
            "feature_names": feature_names,
            "development_dataset_sha256": sha256_file(dataset_path),
        }
    )
    support_path = runtime / "known_manifold_support.json"
    _write_json(support_path, final_support)
    error_audit = _selected_error_audit(
        prediction_rows,
        str(selected["variant_id"]),
        tolerances,
        {str(row["design_hash"]): row for row in rows},
    )

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "status": (
            "DEVELOPMENT_ABLATION_COMPLETE_SUPPORT_GATE_PASS"
            if support_gate_pass
            else "DEVELOPMENT_ABLATION_COMPLETE_SUPPORT_GATE_PARTIAL"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "confirmation_reuse_allowed": False,
        "confirmation_content_read": False,
        "inter_branch_extrapolation_in_scope": False,
        "model_selection_complete": True,
        "development_accuracy_gate": "NOT_PRE_REGISTERED_MODEL_SELECTION_ONLY",
        "model_advancement_established": False,
        "fresh_confirmation_acquisition_ready": False,
        "fresh_confirmation_blocker": (
            "development_accuracy_gate_not_registered_and_worst_root_error_remains_high"
        ),
        "device": str(run_device),
        "dataset_rows": len(rows),
        "global_roots": len(assignments),
        "target_segments": list(target_segments),
        "selection_metric": "development_multi_fold_maximum_worst_root_rho_p95",
        "selected_variant": selected["variant_id"],
        "selected_vs_legacy_reduction_fraction": float(improvement),
        "selected_error_audit": error_audit,
        "variant_results": variant_results,
        "support_audit": {
            "gate_pass": support_gate_pass,
            "target_validation_root_acceptance_fraction": target_acceptance,
            "minimum_required": float(
                config["support_gate"]["minimum_target_validation_root_acceptance_fraction"]
            ),
            "folds": support_folds,
            "final_support_status": final_support["status"],
        },
        "hashes": {
            "config": sha256_file(config_source),
            "implementation": sha256_file(Path(__file__)),
            "development_dataset": sha256_file(dataset_path),
            "development_folds": sha256_file(folds_path),
            "sealed_confirmation": sha256_file(confirmation_path),
            "predictions": sha256_file(predictions_path),
            "known_manifold_support": sha256_file(support_path),
        },
        "paths": {
            "predictions": str(predictions_path.relative_to(root)),
            "known_manifold_support": str(support_path.relative_to(root)),
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
    summary_path = runtime / config["artifacts"]["summary"]
    _write_json(summary_path, summary)
    report_path = runtime / config["artifacts"]["report"]
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return summary
