"""Campaign-aware training and closed-loop audit for tagged Geometry19 MLPs."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import subprocess
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.ai.geometry19_surrogate import (
    TaggedGeometry19MLP,
    load_weight_bundle,
    save_weight_bundle,
    sha256_file,
)
from src.physics import ConstraintProfile, Geometry19, ModelConfig
from src.visualization import Geometry19DesignStore


PERFORMANCE_NAMES = ["thrust_N", "tsfc_kg_N_s", "mass_kg"]


class Geometry19TrainingError(ValueError):
    """Raised when the Geometry19 training contract is inconsistent."""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _campaign_lookup(split_config: Mapping[str, Sequence[str]]) -> dict[str, str]:
    lookup = {}
    for split, campaigns in split_config.items():
        for campaign in campaigns:
            if campaign in lookup:
                raise Geometry19TrainingError(f"Campaign appears in multiple splits: {campaign}")
            lookup[str(campaign)] = str(split)
    if set(split_config) != {"train", "validation", "test"}:
        raise Geometry19TrainingError("Campaign split must define train/validation/test")
    return lookup


def load_regression_splits(
    path: str | Path,
    feature_names: Sequence[str],
    campaign_split: Mapping[str, Sequence[str]],
    maximum_rows_per_split: int | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    lookup = _campaign_lookup(campaign_split)
    buckets = {name: {"x": [], "y": [], "records": []} for name in campaign_split}
    constraint_names: list[str] | None = None
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            split = lookup.get(str(record["campaign_id"]))
            if split is None:
                raise Geometry19TrainingError("Regression record has an unknown campaign")
            if maximum_rows_per_split is not None and len(buckets[split]["x"]) >= maximum_rows_per_split:
                continue
            current_constraints = sorted(record["constraint_values"]["all"])
            if constraint_names is None:
                constraint_names = current_constraints
            elif current_constraints != constraint_names:
                raise Geometry19TrainingError("Constraint output contract changed within dataset")
            buckets[split]["x"].append([record["x_19"][name] for name in feature_names])
            performance = record["physics_outputs"]["performance"]
            buckets[split]["y"].append(
                [
                    *(performance[name] for name in PERFORMANCE_NAMES),
                    *(record["constraint_values"]["all"][name] for name in constraint_names),
                ]
            )
            buckets[split]["records"].append(record)
    if constraint_names is None or len(constraint_names) != 45:
        raise Geometry19TrainingError("Geometry19 forward contract requires 45 constraints")
    for split, bucket in buckets.items():
        bucket["x"] = np.asarray(bucket["x"], dtype=np.float32)
        bucket["y"] = np.asarray(bucket["y"], dtype=np.float32)
        if not len(bucket["x"]) or not np.isfinite(bucket["x"]).all() or not np.isfinite(bucket["y"]).all():
            raise Geometry19TrainingError(f"Invalid or empty regression split: {split}")
    return buckets, [*PERFORMANCE_NAMES, *(f"constraint.{name}" for name in constraint_names)]


def load_classification_splits(
    path: str | Path,
    feature_names: Sequence[str],
    class_names: Sequence[str],
    campaign_split: Mapping[str, Sequence[str]],
    maximum_rows_per_split: int | None = None,
) -> dict[str, dict[str, Any]]:
    lookup = _campaign_lookup(campaign_split)
    class_index = {name: index for index, name in enumerate(class_names)}
    buckets = {name: {"x": [], "y": [], "records": []} for name in campaign_split}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            split = lookup.get(str(record["campaign_id"]))
            if split is None or record["classification"] not in class_index:
                raise Geometry19TrainingError("Classification campaign/class contract mismatch")
            if maximum_rows_per_split is not None and len(buckets[split]["x"]) >= maximum_rows_per_split:
                continue
            buckets[split]["x"].append([record["x_19"][name] for name in feature_names])
            buckets[split]["y"].append(class_index[record["classification"]])
            buckets[split]["records"].append(record)
    for split, bucket in buckets.items():
        bucket["x"] = np.asarray(bucket["x"], dtype=np.float32)
        bucket["y"] = np.asarray(bucket["y"], dtype=np.int64)
        if not len(bucket["x"]) or not np.isfinite(bucket["x"]).all():
            raise Geometry19TrainingError(f"Invalid or empty classification split: {split}")
    return buckets


def fit_normalization(
    train_x: np.ndarray, train_y: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    x_mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    x_std = np.where(x_std > 1.0e-12, x_std, 1.0).astype(np.float32)
    result = {"input_mean": x_mean, "input_std": x_std}
    if train_y is not None:
        y_mean = train_y.mean(axis=0, dtype=np.float64).astype(np.float32)
        y_std = train_y.std(axis=0, dtype=np.float64).astype(np.float32)
        result.update(
            output_mean=y_mean,
            output_std=np.where(y_std > 1.0e-12, y_std, 1.0).astype(np.float32),
        )
    return result


def _normalize_x(x: np.ndarray, stats: Mapping[str, np.ndarray]) -> np.ndarray:
    return ((x - stats["input_mean"]) / stats["input_std"]).astype(np.float32)


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def _train_model(
    model: TaggedGeometry19MLP,
    train_loader: DataLoader,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    training: Mapping[str, Any],
    device: torch.device,
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
        total_loss = 0.0
        total_count = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(x_batch), y_batch)
            if not torch.isfinite(loss):
                raise Geometry19TrainingError("Training produced non-finite loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(x_batch)
            total_count += len(x_batch)
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(validation_x_tensor), validation_y_tensor))
        train_loss = total_loss / total_count
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
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
        raise Geometry19TrainingError("No finite validation checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def _predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            values.append(model(torch.from_numpy(x[start : start + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(values)


def _regression_metrics(truth: np.ndarray, prediction: np.ndarray, target_names: Sequence[str]) -> dict[str, Any]:
    outputs = {}
    for index, name in enumerate(target_names):
        denominator = np.maximum(np.abs(truth[:, index]), 1.0e-12)
        relative = np.abs(prediction[:, index] - truth[:, index]) / denominator
        outputs[name] = {
            "mae": float(mean_absolute_error(truth[:, index], prediction[:, index])),
            "rmse": float(mean_squared_error(truth[:, index], prediction[:, index]) ** 0.5),
            "r2": float(r2_score(truth[:, index], prediction[:, index])) if np.ptp(truth[:, index]) > 0 else None,
            "relative_error_p50": float(np.quantile(relative, 0.5)),
            "relative_error_p95": float(np.quantile(relative, 0.95)),
        }
    sign_accuracy = float(np.mean((truth[:, 3:] <= 0.0) == (prediction[:, 3:] <= 0.0)))
    return {
        "sample_count": len(truth),
        "finite": bool(np.isfinite(prediction).all()),
        "constraint_sign_accuracy": sign_accuracy,
        "outputs": outputs,
        "performance_mean_r2": float(np.mean([outputs[name]["r2"] for name in PERFORMANCE_NAMES])),
    }


def _classification_metrics(truth: np.ndarray, logits: np.ndarray, class_names: Sequence[str]) -> dict[str, Any]:
    prediction = logits.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=np.arange(len(class_names)), zero_division=0
    )
    converged_truth = np.isin(truth, [0, 1])
    converged_prediction = np.isin(prediction, [0, 1])
    return {
        "sample_count": len(truth),
        "finite": bool(np.isfinite(logits).all()),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "binary_convergence_f1": float(f1_score(converged_truth, converged_prediction)),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=np.arange(len(class_names))).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
    }


def _artifact_manifest(
    model_tag: str,
    architecture_tag: str,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    provenance: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "checkpoint_version": "geometry19_tagged_weights_v1",
        "model_tag": model_tag,
        "architecture_tag": architecture_tag,
        "feature_names": list(feature_names),
        "target_names": list(target_names),
        "training_seed": seed,
        **dict(provenance),
    }


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _select_cases(
    records: Sequence[Mapping[str, Any]],
    x: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    bounds: np.ndarray,
    seed: int,
) -> dict[str, int]:
    scale = np.std(truth, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    error = np.sqrt(np.mean(((prediction - truth) / scale) ** 2, axis=1))
    normalized = (x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    boundary = np.min(np.minimum(normalized, 1.0 - normalized), axis=1)
    orders = {
        "best_prediction": np.argsort(error),
        "worst_prediction": np.argsort(-error),
        "boundary": np.argsort(boundary),
        "random": np.random.default_rng(seed).permutation(len(records)),
    }
    selected = {}
    used = set()
    for kind, order in orders.items():
        for index in order:
            value = int(index)
            if value not in used and records[value]["geometry_mapping"]["valid"]:
                selected[kind] = value
                used.add(value)
                break
    return selected


def _render_report(report: Mapping[str, Any]) -> str:
    forward = report["forward"]["test_metrics"]
    classifier = report["feasibility"]["test_metrics"]
    lines = [
        "# WP19-22 Geometry19 Tagged MLP 训练与闭环审计",
        "",
        f"> 状态：**{report['status']}**  ",
        "> 数据级别：`development_bootstrap`；正式用途允许：`false`",
        "",
        "## 契约",
        "",
        f"- Forward：19D → 48D，tag `{report['forward']['model_tag']}`。",
        f"- Feasibility：19D → 5-class，tag `{report['feasibility']['model_tag']}`。",
        "- 隐藏层保持旧 surrogate 的 `[128,256,256,128,64]`，旧实现和旧权重未覆盖。",
        "",
        "## Test 指标",
        "",
        f"- Forward performance mean R²：`{forward['performance_mean_r2']:.6f}`。",
        f"- Constraint sign accuracy：`{forward['constraint_sign_accuracy']:.6f}`。",
        f"- Feasibility accuracy / macro-F1：`{classifier['accuracy']:.6f}` / `{classifier['macro_f1']:.6f}`。",
        f"- Binary convergence F1：`{classifier['binary_convergence_f1']:.6f}`。",
        "- `FLOW_RESIDUAL_DISCONTINUITY` 为稀有类，必须结合 per-class support/F1 审计，不以总体 accuracy 代替。",
        "",
        "## 闭环",
        "",
        f"- 权重独立保存与重载一致：`{str(report['weight_reload_equal']).lower()}`。",
        f"- V2.1 真实回算：`{report['closed_loop']['physics_passed_count']}/4`。",
        f"- Preview/UI-ready：`{report['closed_loop']['visualization_passed_count']}/4`。",
        "",
        "代理预测仅用于开发筛选，所有候选仍以 V2.1 真实物理回算为准。",
        "",
    ]
    return "\n".join(lines)


def run_geometry19_training(
    protocol_path: str | Path,
    repository_root: str | Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    protocol_source = Path(protocol_path).resolve()
    protocol_bytes = protocol_source.read_bytes()
    protocol = yaml.safe_load(protocol_bytes)
    manifest = json.loads((root / protocol["dataset_manifest"]).read_text(encoding="utf-8"))
    schema_bytes = (root / protocol["design_schema"]).read_bytes()
    schema = yaml.safe_load(schema_bytes)
    if manifest["status"] != "COMPLETE" or manifest["summary"]["sample_count"] != 98304:
        raise Geometry19TrainingError("Frozen 98,304-point dataset is incomplete")
    if manifest["dataset_tier"] != protocol["dataset_tier"] or protocol["formal_use_allowed"] is not False:
        raise Geometry19TrainingError("Dataset tier/formal-use contract mismatch")
    current_hashes = {
        "schema_hash": sha256(schema_bytes).hexdigest(),
        "calibration_hash": ModelConfig.from_yaml(
            root / protocol["calibration_pack"]
        ).calibration_pack_hash,
        "constraint_profile_hash": ConstraintProfile.from_yaml(
            root / protocol["constraint_profile"]
        ).source_hash,
    }
    for name, current_value in current_hashes.items():
        if current_value != manifest["provenance"][name]:
            raise Geometry19TrainingError(
                f"Current {name} does not match the frozen dataset provenance"
            )
    feature_names = [item["name"] for item in schema["design_variables"]]
    bounds = np.asarray([item["bounds"] for item in schema["design_variables"]], dtype=np.float32)
    maximum = int(protocol["smoke"]["maximum_rows_per_split"]) if smoke else None
    regression, target_names = load_regression_splits(
        root / protocol["regression_dataset"], feature_names, protocol["campaign_split"], maximum
    )
    classes = list(protocol["feasibility"]["classes"])
    classification = load_classification_splits(
        root / protocol["feasibility_dataset"], feature_names, classes, protocol["campaign_split"], maximum
    )
    training = dict(protocol["training"])
    if smoke:
        training.update(epochs=protocol["smoke"]["epochs"], patience=protocol["smoke"]["patience"])
    seed = int(training["seed"])
    _seed_all(seed)
    device = torch.device(training["device"])
    architecture = protocol["architecture"]
    run_suffix = "smoke" if smoke else "baseline"
    runtime_root = root / protocol["artifacts"]["runtime_root"] / run_suffix
    weights_root = root / protocol["artifacts"]["weights_root"]
    if runtime_root.exists():
        raise Geometry19TrainingError(f"Output already exists: {runtime_root}")
    runtime_root.mkdir(parents=True)
    provenance = {
        "dataset_tag": "wp19_21_98304_development_v1",
        "physics_tag": "geometry19_closed_v2_1",
        "schema_hash": current_hashes["schema_hash"],
        "calibration_hash": manifest["provenance"]["calibration_hash"],
        "constraint_profile_hash": manifest["provenance"]["constraint_profile_hash"],
        "dataset_manifest_hash": sha256_file(root / protocol["dataset_manifest"]),
        "training_protocol_hash": sha256(protocol_bytes).hexdigest(),
        "code_commit": _git_commit(root),
        "formal_use_allowed": False,
    }

    forward_stats = fit_normalization(regression["train"]["x"], regression["train"]["y"])
    forward_x = {name: _normalize_x(bucket["x"], forward_stats) for name, bucket in regression.items()}
    forward_y = {
        name: ((bucket["y"] - forward_stats["output_mean"]) / forward_stats["output_std"]).astype(np.float32)
        for name, bucket in regression.items()
    }
    forward_model = TaggedGeometry19MLP(
        19, 48, architecture["hidden_layers"], architecture["activation"],
        architecture["dropout"], architecture["batch_norm"],
    ).to(device)
    performance_weight = float(protocol["forward"]["performance_loss_weight"])
    constraint_weight = float(protocol["forward"]["constraint_loss_weight"])

    def forward_loss(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        return performance_weight * nn.functional.mse_loss(prediction[:, :3], truth[:, :3]) + constraint_weight * nn.functional.mse_loss(prediction[:, 3:], truth[:, 3:])

    forward_model, forward_history, forward_best = _train_model(
        forward_model,
        _loader(forward_x["train"], forward_y["train"], int(training["batch_size"]), True, seed),
        forward_x["validation"], forward_y["validation"], forward_loss, training, device,
    )
    forward_predictions = {}
    forward_metrics = {}
    for split in regression:
        normalized_prediction = _predict(forward_model, forward_x[split], device)
        physical_prediction = normalized_prediction * forward_stats["output_std"] + forward_stats["output_mean"]
        forward_predictions[split] = physical_prediction
        forward_metrics[split] = _regression_metrics(regression[split]["y"], physical_prediction, target_names)

    classification_stats = fit_normalization(classification["train"]["x"])
    classification_x = {name: _normalize_x(bucket["x"], classification_stats) for name, bucket in classification.items()}
    counts = np.bincount(classification["train"]["y"], minlength=len(classes))
    class_weights = np.sqrt(len(classification["train"]["y"]) / (len(classes) * np.maximum(counts, 1))).astype(np.float32)
    classification_model = TaggedGeometry19MLP(
        19, 5, architecture["hidden_layers"], architecture["activation"],
        architecture["dropout"], architecture["batch_norm"],
    ).to(device)
    cross_entropy = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
    classification_model, class_history, class_best = _train_model(
        classification_model,
        _loader(classification_x["train"], classification["train"]["y"], int(training["batch_size"]), True, seed),
        classification_x["validation"], classification["validation"]["y"], cross_entropy, training, device,
    )
    classification_metrics = {
        split: _classification_metrics(
            bucket["y"], _predict(classification_model, classification_x[split], device), classes
        )
        for split, bucket in classification.items()
    }

    forward_dir = weights_root / "forward_mlp_48d_v1" / f"seed_{seed}" / run_suffix
    feasibility_dir = weights_root / "feasibility_mlp_5class_v1" / f"seed_{seed}" / run_suffix
    forward_bundle = save_weight_bundle(
        forward_dir,
        forward_model,
        _artifact_manifest(
            protocol["forward"]["model_tag"], architecture["architecture_tag"],
            feature_names, target_names, provenance, seed,
        ),
        forward_stats,
    )
    class_bundle = save_weight_bundle(
        feasibility_dir,
        classification_model,
        _artifact_manifest(
            protocol["feasibility"]["model_tag"], architecture["architecture_tag"],
            feature_names, classes, provenance, seed,
        ),
        classification_stats,
    )
    feature_contract = {
        "contract_version": "geometry19_feature_contract_v1",
        "dimension": len(feature_names),
        "ordered_names": feature_names,
        "schema_hash": current_hashes["schema_hash"],
    }
    _write_json(forward_dir / "feature_contract.json", feature_contract)
    _write_json(feasibility_dir / "feature_contract.json", feature_contract)
    _write_json(
        forward_dir / "target_contract.json",
        {
            "contract_version": "geometry19_forward_target_contract_v1",
            "dimension": len(target_names),
            "ordered_names": target_names,
            "performance_dimension": len(PERFORMANCE_NAMES),
            "constraint_dimension": len(target_names) - len(PERFORMANCE_NAMES),
        },
    )
    _write_json(
        feasibility_dir / "class_contract.json",
        {
            "contract_version": "geometry19_feasibility_class_contract_v1",
            "dimension": len(classes),
            "ordered_names": classes,
            "converged_class_indices": [0, 1],
        },
    )
    split_manifest = {
        name: {
            "campaigns": protocol["campaign_split"][name],
            "regression_count": len(regression[name]["x"]),
            "classification_count": len(classification[name]["x"]),
        }
        for name in protocol["campaign_split"]
    }
    for directory, metrics, history, best_epoch in (
        (forward_dir, forward_metrics, forward_history, forward_best),
        (feasibility_dir, classification_metrics, class_history, class_best),
    ):
        _write_json(directory / "split_manifest.json", split_manifest)
        _write_json(directory / "metrics.json", metrics)
        _write_json(directory / "training_history.json", {"best_epoch": best_epoch, "history": history})
    reloaded_forward, reloaded_forward_stats, _ = load_weight_bundle(
        forward_dir,
        expected_model_tag=protocol["forward"]["model_tag"],
        expected_feature_names=feature_names,
        expected_target_names=target_names,
        device=device,
    )
    reloaded_class, reloaded_class_stats, _ = load_weight_bundle(
        feasibility_dir,
        expected_model_tag=protocol["feasibility"]["model_tag"],
        expected_feature_names=feature_names,
        expected_target_names=classes,
        device=device,
    )
    reload_equal = (
        np.array_equal(
            _predict(reloaded_forward, _normalize_x(regression["test"]["x"][:256], reloaded_forward_stats), device),
            _predict(forward_model, forward_x["test"][:256], device),
        )
        and np.array_equal(
            _predict(reloaded_class, _normalize_x(classification["test"]["x"][:256], reloaded_class_stats), device),
            _predict(classification_model, classification_x["test"][:256], device),
        )
    )

    cases = []
    if not smoke:
        selected = _select_cases(
            regression["test"]["records"], regression["test"]["x"],
            regression["test"]["y"], forward_predictions["test"], bounds, seed,
        )
        store = Geometry19DesignStore(root, root / protocol["artifacts"]["mesh_storage_root"])
        for kind, index in selected.items():
            record = regression["test"]["records"][index]
            design = Geometry19.from_mapping(record["x_19"])
            solved = store.solve(design)
            physics_match = all(
                math.isclose(
                    float(solved["physics"]["performance"][name]),
                    float(record["physics_outputs"]["performance"][name]),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
                for name in PERFORMANCE_NAMES
            )
            visual = store.regenerate(design, quality=protocol["validation"]["mesh_quality"])
            ui_path = runtime_root / "ui_cases" / f"{kind}.json"
            _write_json(
                ui_path,
                {
                    "model_tag": protocol["forward"]["model_tag"],
                    "selection_kind": kind,
                    "geometry19": record["x_19"],
                    "best": {"geometry19": record["x_19"]},
                    "sample_id": record["sample_id"],
                    "surrogate_prediction": {
                        name: float(forward_predictions["test"][index, target_index])
                        for target_index, name in enumerate(target_names)
                    },
                    "true_physics": solved["physics"],
                    "design_id": visual["design_id"],
                },
            )
            cases.append(
                {
                    "selection_kind": kind,
                    "sample_id": record["sample_id"],
                    "physics_recalculation_match": physics_match,
                    "mesh_diagnostics_passed": bool(visual["mesh_diagnostics"]["passed"]),
                    "ui_json": str(ui_path.relative_to(root)),
                    "design_id": visual["design_id"],
                }
            )
    physics_passed = sum(case["physics_recalculation_match"] for case in cases)
    visual_passed = sum(case["mesh_diagnostics_passed"] for case in cases)
    expected_cases = 0 if smoke else 4
    status = "COMPLETE" if (
        reload_equal
        and all(metric["finite"] for metric in forward_metrics.values())
        and all(metric["finite"] for metric in classification_metrics.values())
        and physics_passed == expected_cases
        and visual_passed == expected_cases
    ) else "AUDIT_FAILED"
    report = {
        "work_package": "WP19-22",
        "protocol_version": protocol["protocol_version"],
        "run_kind": run_suffix,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "dataset_tier": protocol["dataset_tier"],
        "formal_use_allowed": False,
        "architecture_tag": architecture["architecture_tag"],
        "feature_names": feature_names,
        "forward": {
            "model_tag": protocol["forward"]["model_tag"],
            "target_names": target_names,
            "split_counts": {name: len(bucket["x"]) for name, bucket in regression.items()},
            "best_epoch": forward_best,
            "test_metrics": forward_metrics["test"],
            "artifacts": {key: str(Path(value).relative_to(root)) for key, value in forward_bundle.items()},
        },
        "feasibility": {
            "model_tag": protocol["feasibility"]["model_tag"],
            "classes": classes,
            "split_counts": {name: len(bucket["x"]) for name, bucket in classification.items()},
            "best_epoch": class_best,
            "test_metrics": classification_metrics["test"],
            "artifacts": {key: str(Path(value).relative_to(root)) for key, value in class_bundle.items()},
        },
        "weight_reload_equal": reload_equal,
        "closed_loop": {
            "physics_passed_count": physics_passed,
            "visualization_passed_count": visual_passed,
            "cases": cases,
        },
        "provenance": provenance,
    }
    _write_json(runtime_root / "summary.json", report)
    if not smoke:
        _write_json(root / protocol["artifacts"]["summary_json"], report)
        report_path = root / protocol["artifacts"]["report_markdown"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_render_report(report), encoding="utf-8")
    return report
