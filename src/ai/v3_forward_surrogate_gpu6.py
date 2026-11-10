"""V3 22D software-only forward surrogate experiment pinned to physical GPU 6."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
import yaml


CONFIG = "config/experiments/geometry22_v3_forward_surrogate_gpu6_v1.yaml"
CODE = "src/ai/v3_forward_surrogate_gpu6.py"
NAMES = ("thrust_N", "tsfc_kg_N_s", "mass_kg")
SPLITS = ("development", "validation", "holdout")


class V3ForwardSurrogateError(ValueError):
    """The V3 surrogate source, training order, or artifact identity is invalid."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_new(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)


def _config(root: Path) -> dict[str, Any]:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    mlp = config.get("models", {}).get("mlp", {})
    if (
        config.get("version") != "geometry22_v3_forward_surrogate_gpu6_v1"
        or config.get("scope") != "isolated_software_forward_surrogate_experiment"
        or config.get("input_dimension") != 22
        or config.get("output_names") != list(NAMES)
        or config.get("target_transform") != "natural_log"
        or config.get("normalization") != "center_balanced_mean_std_fit_development_only"
        or config.get("sample_weight") != "inverse_development_center_count_normalized_to_mean_one"
        or config.get("models", {}).get("constant") != "center_balanced_development_log_mean"
        or config["models"].get("ridge_alphas") != [1.0, 100.0]
        or mlp.get("hidden_layers") != [128, 128]
        or mlp.get("activation") != "silu"
        or mlp.get("seeds") != [42, 43]
        or mlp.get("batch_size") != 512
        or mlp.get("learning_rate") != 0.001
        or mlp.get("weight_decay") != 0.0001
        or mlp.get("maximum_epochs") != 120
        or mlp.get("minimum_epochs") != 20
        or mlp.get("patience") != 20
        or mlp.get("min_delta") != 0.0001
        or mlp.get("device") != "cuda:0"
        or mlp.get("physical_gpu_index") != 6
        or config.get("metric", {}).get("row") != "max_absolute_natural_log_prediction_over_truth_across_three_outputs"
        or config["metric"].get("primary") != "validation_worst_center_p95_row_metric"
        or config["metric"].get("secondary") != "validation_overall_p95_row_metric"
        or config["metric"].get("tertiary") != "model_id_lexicographic"
        or config.get("holdout") != "selected_model_only_after_validation_selection_and_checkpoint_seal"
        or config.get("accuracy_gate") != "diagnostic_only_no_threshold_promotion"
        or config.get("source_records_training_allowed") is not False
        or config.get("isolated_software_training_authorized") is not True
        or config.get("formal_use_allowed") is not False
        or config.get("deployment_allowed") is not False
    ):
        raise V3ForwardSurrogateError("V3 surrogate protocol changed scope")
    return config


def _source(root: Path, config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    spec = config["source_dataset"]
    source = root / spec["path"]
    if _sha(source / "manifest.json") != spec["manifest_sha256"]:
        raise V3ForwardSurrogateError("Candidate dataset manifest changed")
    for split in SPLITS:
        if _sha(source / f"{split}_candidates.jsonl") != spec[f"{split}_sha256"]:
            raise V3ForwardSurrogateError(f"Candidate {split} split changed")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["combined_data_coverage_gate_pass"] is not True
        or manifest["candidate_count"] != 16168
        or manifest["split_counts"] != spec["expected_counts"]
        or manifest["center_counts"] != spec["expected_centers"]
        or manifest["training_allowed"] is not False
        or manifest["formal_use_allowed"] is not False
        or manifest["deployment_allowed"] is not False
    ):
        raise V3ForwardSurrogateError("Candidate source coverage or scope changed")
    return source, manifest


def _load_split(source: Path, split: str, expected_count: int, expected_centers: int,
                schema_hash: str, physics_hash: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = _rows(source / f"{split}_candidates.jsonl")
    if len(rows) != expected_count:
        raise V3ForwardSurrogateError(f"{split} candidate count changed")
    centers = [row["center_design_id"] for row in rows]
    designs = [row["design_id"] for row in rows]
    if len(set(centers)) != expected_centers or len(set(designs)) != len(designs):
        raise V3ForwardSurrogateError(f"{split} center/design population changed")
    x = np.asarray([row["normalized_design"] for row in rows], dtype=np.float64)
    y = np.asarray([[row["performance"][name] for name in NAMES] for row in rows], dtype=np.float64)
    if (
        x.shape != (len(rows), 22) or y.shape != (len(rows), 3)
        or not np.isfinite(x).all() or not np.isfinite(y).all()
        or np.any(x < -1.000000001) or np.any(x > 1.000000001)
        or np.any(y <= 0)
        or any(row["center_split"] != split or row["design_schema_hash"] != schema_hash
               or row["physics_bundle_hash"] != physics_hash
               or row["software_regression_candidate"] is not True
               or row["training_allowed"] is not False
               or row["formal_use_allowed"] is not False
               or row["deployment_allowed"] is not False for row in rows)
    ):
        raise V3ForwardSurrogateError(f"{split} data identity or values changed")
    return x, y, centers


def _weights(centers: list[str]) -> np.ndarray:
    counts = Counter(centers)
    w = np.asarray([1.0 / counts[center] for center in centers], dtype=np.float64)
    return w / w.mean()


def _stats(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> dict[str, list[float]]:
    w = weights / weights.sum()
    log_y = np.log(y)
    x_mean = np.sum(x*w[:, None], axis=0)
    y_mean = np.sum(log_y*w[:, None], axis=0)
    x_std = np.sqrt(np.sum((x-x_mean)**2*w[:, None], axis=0))
    y_std = np.sqrt(np.sum((log_y-y_mean)**2*w[:, None], axis=0))
    return {
        "input_mean": x_mean.tolist(),
        "input_std": np.where(x_std > 1e-12, x_std, 1.0).tolist(),
        "output_log_mean": y_mean.tolist(),
        "output_log_std": np.where(y_std > 1e-12, y_std, 1.0).tolist(),
        "fit_split": "development",
        "target_transform": "natural_log",
    }


def _normalize(x: np.ndarray, y: np.ndarray, stats: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (
        (x-np.asarray(stats["input_mean"]))/np.asarray(stats["input_std"]),
        (np.log(y)-np.asarray(stats["output_log_mean"]))/np.asarray(stats["output_log_std"]),
    )


def _decode(normalized: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    return np.exp(normalized*np.asarray(stats["output_log_std"])+np.asarray(stats["output_log_mean"]))


def _metrics(truth: np.ndarray, prediction: np.ndarray, centers: list[str]) -> dict[str, Any]:
    if prediction.shape != truth.shape or not np.isfinite(prediction).all() or np.any(prediction <= 0):
        raise V3ForwardSurrogateError("Surrogate prediction is non-finite or non-positive")
    row = np.max(np.abs(np.log(prediction/truth)), axis=1)
    relative = np.abs(prediction/truth-1.0)
    by_center = defaultdict(list)
    for value, center in zip(row, centers, strict=True):
        by_center[center].append(float(value))
    center_p95 = {center: float(np.quantile(values, 0.95)) for center, values in sorted(by_center.items())}
    return {
        "rows": len(row),
        "centers": len(center_p95),
        "overall_p50_max_abs_log_error": float(np.quantile(row, 0.5)),
        "overall_p95_max_abs_log_error": float(np.quantile(row, 0.95)),
        "worst_center_p95_max_abs_log_error": float(max(center_p95.values())),
        "center_p95_max_abs_log_error": center_p95,
        "absolute_relative_error": {
            name: {"p50": float(np.quantile(relative[:, i], 0.5)),
                   "p95": float(np.quantile(relative[:, i], 0.95))}
            for i, name in enumerate(NAMES)
        },
    }


def _network() -> nn.Module:
    return nn.Sequential(nn.Linear(22, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 3))


def _gpu_device() -> torch.device:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6":
        raise V3ForwardSurrogateError("Physical GPU 6 requires CUDA_VISIBLE_DEVICES=6")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise V3ForwardSurrogateError("Physical GPU 6 is unavailable to PyTorch")
    return torch.device("cuda:0")


def _network_predictions(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        return model(tensor).detach().cpu().numpy().astype(np.float64)


def _fit_network(x: np.ndarray, y: np.ndarray, weights: np.ndarray,
                 val_x: np.ndarray, val_y: np.ndarray, seed: int,
                 config: dict[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
    mlp = config["models"]["mlp"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _gpu_device()
    model = _network().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=mlp["learning_rate"], weight_decay=mlp["weight_decay"])
    train_x = torch.as_tensor(x, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(y, dtype=torch.float32, device=device)
    train_w = torch.as_tensor(weights, dtype=torch.float32, device=device)
    validation_x = torch.as_tensor(val_x, dtype=torch.float32, device=device)
    validation_y = torch.as_tensor(val_y, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best = math.inf
    best_state = None
    best_epoch = 0
    history = []
    for epoch in range(1, mlp["maximum_epochs"] + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for indices in order.split(mlp["batch_size"]):
            indices = indices.to(device)
            prediction = model(train_x[indices])
            loss = ((prediction-train_y[indices])**2*train_w[indices, None]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(torch.mean((model(validation_x)-validation_y)**2).item())
        if not math.isfinite(validation_loss):
            raise V3ForwardSurrogateError("V3 MLP validation loss is non-finite")
        history.append({"epoch": epoch, "validation_normalized_mse": validation_loss})
        if validation_loss < best-mlp["min_delta"]:
            best = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch >= mlp["minimum_epochs"] and epoch-best_epoch >= mlp["patience"]:
            break
    if best_state is None:
        raise V3ForwardSurrogateError("No V3 MLP checkpoint selected")
    model.load_state_dict(best_state)
    return model, {"seed": seed, "best_epoch": best_epoch, "completed_epochs": len(history),
                   "best_validation_normalized_mse": best, "history": history}


def freeze_v3_forward_protocol(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / config["output_root"]
    if output.exists():
        raise V3ForwardSurrogateError(f"Refusing to overwrite V3 surrogate protocol: {output}")
    source, manifest = _source(root, config)
    schema_hash = manifest["identity"]["design_schema_hash"]
    physics_hash = manifest["identity"]["physics_bundle_hash"]
    development = _load_split(source, "development", 11289, 23, schema_hash, physics_hash)
    validation = _load_split(source, "validation", 2394, 5, schema_hash, physics_hash)
    if set(development[2]) & set(validation[2]):
        raise V3ForwardSurrogateError("Development/validation centers overlap")
    lock = {
        "status": "v3_forward_software_training_protocol_frozen",
        "config_sha256": _sha(root / CONFIG),
        "code_sha256": _sha(root / CODE),
        "dataset_manifest_sha256": config["source_dataset"]["manifest_sha256"],
        "split_sha256": {split: config["source_dataset"][f"{split}_sha256"] for split in SPLITS},
        "design_schema_hash": schema_hash,
        "physics_bundle_hash": physics_hash,
        "development_center_ids": sorted(set(development[2])),
        "validation_center_ids": sorted(set(validation[2])),
        "holdout_content_parsed": False,
        "training_executed": False,
        "isolated_software_training_authorized": True,
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }
    output.mkdir(parents=True)
    _write_new(output / "protocol_lock.json", _bytes(lock))
    return verify_v3_forward_protocol(root, output)


def verify_v3_forward_protocol(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    config = _config(root)
    source, manifest = _source(root, config)
    lock = json.loads((output / "protocol_lock.json").read_text(encoding="utf-8"))
    if (
        lock["status"] != "v3_forward_software_training_protocol_frozen"
        or lock["config_sha256"] != _sha(root / CONFIG)
        or lock["code_sha256"] != _sha(root / CODE)
        or lock["dataset_manifest_sha256"] != config["source_dataset"]["manifest_sha256"]
        or lock["split_sha256"] != {split: config["source_dataset"][f"{split}_sha256"] for split in SPLITS}
        or lock["design_schema_hash"] != manifest["identity"]["design_schema_hash"]
        or lock["physics_bundle_hash"] != manifest["identity"]["physics_bundle_hash"]
        or lock["holdout_content_parsed"] is not False
        or lock["training_executed"] is not False
        or lock["isolated_software_training_authorized"] is not True
        or lock["formal_use_allowed"] is not False
        or lock["deployment_allowed"] is not False
    ):
        raise V3ForwardSurrogateError("V3 surrogate protocol lock changed")
    development = _load_split(source, "development", 11289, 23, lock["design_schema_hash"], lock["physics_bundle_hash"])
    validation = _load_split(source, "validation", 2394, 5, lock["design_schema_hash"], lock["physics_bundle_hash"])
    if (lock["development_center_ids"] != sorted(set(development[2]))
            or lock["validation_center_ids"] != sorted(set(validation[2]))
            or set(development[2]) & set(validation[2])):
        raise V3ForwardSurrogateError("V3 surrogate center split changed")
    return lock


def _asset_prediction(asset: dict[str, Any], x_normalized: np.ndarray,
                      stats: dict[str, Any], output: Path) -> np.ndarray:
    path = output / asset["path"]
    if _sha(path) != asset["sha256"]:
        raise V3ForwardSurrogateError("V3 model asset hash changed")
    if asset["kind"] == "constant":
        value = json.loads(path.read_text(encoding="utf-8"))
        normalized = np.zeros((len(x_normalized), 3), dtype=np.float64)
        if value != {"kind": "constant", "normalized_prediction": [0.0, 0.0, 0.0]}:
            raise V3ForwardSurrogateError("Constant baseline asset changed")
    elif asset["kind"] == "ridge":
        value = json.loads(path.read_text(encoding="utf-8"))
        coef = np.asarray(value["coefficients"], dtype=np.float64)
        if value["kind"] != "ridge" or coef.shape != (22, 3) or value["alpha"] != asset["alpha"]:
            raise V3ForwardSurrogateError("Ridge baseline asset changed")
        normalized = x_normalized @ coef
    elif asset["kind"] == "mlp":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["model_id"] != asset["model_id"] or payload["seed"] != asset["seed"]:
            raise V3ForwardSurrogateError("V3 MLP checkpoint identity changed")
        device = _gpu_device()
        model = _network().to(device)
        model.load_state_dict(payload["state_dict"])
        normalized = _network_predictions(model, x_normalized, device)
    else:
        raise V3ForwardSurrogateError("Unknown V3 model kind")
    return _decode(normalized, stats)


def fit_v3_forward_surrogate(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / config["output_root"]
    lock = verify_v3_forward_protocol(root, output)
    if any((output / name).exists() for name in ("fit_manifest.json", "selection.json", "normalization.json")):
        raise V3ForwardSurrogateError("V3 surrogate fit already exists; refusing to overwrite")
    if (output / "models").exists():
        raise V3ForwardSurrogateError("Orphan V3 model directory exists")
    source, _ = _source(root, config)
    x, y, centers = _load_split(source, "development", 11289, 23, lock["design_schema_hash"], lock["physics_bundle_hash"])
    val_x, val_y, val_centers = _load_split(source, "validation", 2394, 5,
                                             lock["design_schema_hash"], lock["physics_bundle_hash"])
    weights = _weights(centers)
    stats = _stats(x, y, weights)
    nx, ny = _normalize(x, y, stats)
    nvx, nvy = _normalize(val_x, val_y, stats)
    model_dir = output / "models"
    model_dir.mkdir()
    assets = []

    def add_asset(model_id: str, kind: str, value: Any, prediction: np.ndarray,
                  *, alpha: float | None = None, seed: int | None = None,
                  training: dict[str, Any] | None = None) -> None:
        suffix = ".pt" if kind == "mlp" else ".json"
        path = model_dir / f"{model_id}{suffix}"
        if kind == "mlp":
            torch.save(value, path)
        else:
            _write_new(path, _bytes(value))
        asset = {
            "model_id": model_id,
            "kind": kind,
            "path": str(path.relative_to(output)),
            "sha256": _sha(path),
            "validation": _metrics(val_y, prediction, val_centers),
            "checkpoint_reload_equal": False,
        }
        if alpha is not None:
            asset["alpha"] = alpha
        if seed is not None:
            asset["seed"] = seed
        if training is not None:
            asset["training"] = training
        reloaded = _asset_prediction(asset, nvx, stats, output)
        asset["checkpoint_reload_equal"] = bool(np.array_equal(prediction, reloaded))
        if not asset["checkpoint_reload_equal"]:
            raise V3ForwardSurrogateError(f"V3 {model_id} reload prediction changed")
        assets.append(asset)

    constant = _decode(np.zeros_like(nvy), stats)
    add_asset("constant", "constant", {"kind": "constant", "normalized_prediction": [0.0, 0.0, 0.0]}, constant)
    for alpha in config["models"]["ridge_alphas"]:
        matrix = nx.T @ (weights[:, None]*nx) + float(alpha)*np.eye(22)
        rhs = nx.T @ (weights[:, None]*ny)
        coef = np.linalg.solve(matrix, rhs)
        prediction = _decode(nvx @ coef, stats)
        model_id = f"ridge_{int(alpha)}"
        add_asset(model_id, "ridge", {"kind": "ridge", "alpha": float(alpha),
                                      "coefficients": coef.tolist()}, prediction, alpha=float(alpha))
    for seed in config["models"]["mlp"]["seeds"]:
        model, training = _fit_network(nx, ny, weights, nvx, nvy, int(seed), config)
        prediction = _decode(_network_predictions(model, nvx, _gpu_device()), stats)
        model_id = f"mlp_seed_{seed}"
        payload = {"model_id": model_id, "seed": seed,
                   "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}}
        add_asset(model_id, "mlp", payload, prediction, seed=int(seed), training=training)
        print(f"V3 forward model fitted: {model_id}, best_epoch={training['best_epoch']}", flush=True)
    selected = min(assets, key=lambda asset: (
        asset["validation"]["worst_center_p95_max_abs_log_error"],
        asset["validation"]["overall_p95_max_abs_log_error"],
        asset["model_id"],
    ))
    _write_new(output / "normalization.json", _bytes(stats))
    selection = {
        "selected_model_id": selected["model_id"],
        "selection_split": "validation",
        "primary_metric": "worst_center_p95_max_abs_log_error",
        "secondary_metric": "overall_p95_max_abs_log_error",
        "holdout_content_parsed": False,
        "selected_validation": selected["validation"],
    }
    _write_new(output / "selection.json", _bytes(selection))
    manifest = {
        "status": "v3_forward_software_fit_selected_before_holdout",
        "training_device": "cuda:0",
        "physical_gpu_index": 6,
        "visible_gpu_name": torch.cuda.get_device_name(0),
        "protocol_lock_sha256": _sha(output / "protocol_lock.json"),
        "normalization_sha256": _sha(output / "normalization.json"),
        "selection_sha256": _sha(output / "selection.json"),
        "source_dataset_manifest_sha256": config["source_dataset"]["manifest_sha256"],
        "normalization_fit_split": "development",
        "development_rows": len(x),
        "validation_rows": len(val_x),
        "development_centers": len(set(centers)),
        "validation_centers": len(set(val_centers)),
        "models": assets,
        "selected_model_id": selected["model_id"],
        "holdout_content_parsed": False,
        "holdout_predictions_executed": False,
        "accuracy_gate_status": "diagnostic_only_no_threshold_promotion",
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }
    _write_new(output / "fit_manifest.json", _bytes(manifest))
    return verify_v3_forward_fit(root, output)


def verify_v3_forward_fit(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    lock = verify_v3_forward_protocol(root, output)
    config = _config(root)
    manifest = json.loads((output / "fit_manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    stats = json.loads((output / "normalization.json").read_text(encoding="utf-8"))
    if (
        manifest["status"] != "v3_forward_software_fit_selected_before_holdout"
        or manifest["training_device"] != "cuda:0"
        or manifest["physical_gpu_index"] != 6
        or manifest["protocol_lock_sha256"] != _sha(output / "protocol_lock.json")
        or manifest["normalization_sha256"] != _sha(output / "normalization.json")
        or manifest["selection_sha256"] != _sha(output / "selection.json")
        or manifest["source_dataset_manifest_sha256"] != config["source_dataset"]["manifest_sha256"]
        or manifest["normalization_fit_split"] != "development"
        or manifest["development_rows"] != 11289 or manifest["validation_rows"] != 2394
        or manifest["development_centers"] != 23 or manifest["validation_centers"] != 5
        or manifest["holdout_content_parsed"] is not False
        or manifest["holdout_predictions_executed"] is not False
        or manifest["formal_use_allowed"] is not False
        or manifest["deployment_allowed"] is not False
        or selection["selection_split"] != "validation"
        or selection["holdout_content_parsed"] is not False
        or stats["fit_split"] != "development"
        or stats["target_transform"] != "natural_log"
        or len(stats["input_mean"]) != 22 or len(stats["input_std"]) != 22
        or len(stats["output_log_mean"]) != 3 or len(stats["output_log_std"]) != 3
        or len(manifest["models"]) != 5
    ):
        raise V3ForwardSurrogateError("V3 fit manifest or preholdout selection changed")
    assets = manifest["models"]
    chosen = min(assets, key=lambda asset: (
        asset["validation"]["worst_center_p95_max_abs_log_error"],
        asset["validation"]["overall_p95_max_abs_log_error"], asset["model_id"]
    ))
    if (
        manifest["selected_model_id"] != chosen["model_id"]
        or selection["selected_model_id"] != chosen["model_id"]
        or selection["selected_validation"] != chosen["validation"]
        or {asset["model_id"] for asset in assets} != {
            "constant", "ridge_1", "ridge_100", "mlp_seed_42", "mlp_seed_43"
        }
        or any(_sha(output / asset["path"]) != asset["sha256"]
               or asset["checkpoint_reload_equal"] is not True for asset in assets)
        or lock["holdout_content_parsed"] is not False
    ):
        raise V3ForwardSurrogateError("V3 model or validation selection changed")
    return manifest


def evaluate_v3_forward_holdout(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / config["output_root"]
    fit = verify_v3_forward_fit(root, output)
    if (output / "holdout_report.json").exists() or (output / "complete_manifest.json").exists():
        raise V3ForwardSurrogateError("V3 holdout evaluation already exists; refusing to repeat")
    source, _ = _source(root, config)
    # Holdout content is parsed only after the validation selection and assets are sealed.
    x, y, centers = _load_split(
        source, "holdout", 2485, 5,
        json.loads((output / "protocol_lock.json").read_text(encoding="utf-8"))["design_schema_hash"],
        json.loads((output / "protocol_lock.json").read_text(encoding="utf-8"))["physics_bundle_hash"],
    )
    lock = json.loads((output / "protocol_lock.json").read_text(encoding="utf-8"))
    if set(centers) & (set(lock["development_center_ids"]) | set(lock["validation_center_ids"])):
        raise V3ForwardSurrogateError("V3 holdout centers leaked into model selection")
    stats = json.loads((output / "normalization.json").read_text(encoding="utf-8"))
    normalized_x, _ = _normalize(x, y, stats)
    selected = next(asset for asset in fit["models"] if asset["model_id"] == fit["selected_model_id"])
    prediction = _asset_prediction(selected, normalized_x, stats, output)
    metrics = _metrics(y, prediction, centers)
    report = {
        "status": "selected_v3_software_model_holdout_evaluated_once",
        "selected_model_id": selected["model_id"],
        "selected_asset_sha256": selected["sha256"],
        "fit_manifest_sha256": _sha(output / "fit_manifest.json"),
        "holdout_dataset_sha256": config["source_dataset"]["holdout_sha256"],
        "holdout_rows": len(x),
        "holdout_centers": len(set(centers)),
        "metrics": metrics,
        "model_selection_used_holdout": False,
        "accuracy_gate_status": "diagnostic_only_no_threshold_promotion",
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }
    _write_new(output / "holdout_report.json", _bytes(report))
    complete = {
        "status": "v3_forward_software_baseline_complete_development_only",
        "protocol_lock_sha256": _sha(output / "protocol_lock.json"),
        "fit_manifest_sha256": _sha(output / "fit_manifest.json"),
        "selection_sha256": _sha(output / "selection.json"),
        "holdout_report_sha256": _sha(output / "holdout_report.json"),
        "selected_model_id": selected["model_id"],
        "holdout_used_for_selection": False,
        "formal_use_allowed": False,
        "deployment_allowed": False,
    }
    _write_new(output / "complete_manifest.json", _bytes(complete))
    return verify_v3_forward_complete(root, output)


def verify_v3_forward_complete(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(output_root).resolve()
    fit = verify_v3_forward_fit(root, output)
    config = _config(root)
    complete = json.loads((output / "complete_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((output / "holdout_report.json").read_text(encoding="utf-8"))
    selected = next(asset for asset in fit["models"] if asset["model_id"] == fit["selected_model_id"])
    if (
        complete["status"] != "v3_forward_software_baseline_complete_development_only"
        or complete["protocol_lock_sha256"] != _sha(output / "protocol_lock.json")
        or complete["fit_manifest_sha256"] != _sha(output / "fit_manifest.json")
        or complete["selection_sha256"] != _sha(output / "selection.json")
        or complete["holdout_report_sha256"] != _sha(output / "holdout_report.json")
        or complete["selected_model_id"] != selected["model_id"]
        or complete["holdout_used_for_selection"] is not False
        or complete["formal_use_allowed"] is not False
        or complete["deployment_allowed"] is not False
        or report["status"] != "selected_v3_software_model_holdout_evaluated_once"
        or report["selected_model_id"] != selected["model_id"]
        or report["selected_asset_sha256"] != selected["sha256"]
        or report["fit_manifest_sha256"] != _sha(output / "fit_manifest.json")
        or report["holdout_dataset_sha256"] != config["source_dataset"]["holdout_sha256"]
        or report["holdout_rows"] != 2485 or report["holdout_centers"] != 5
        or report["metrics"]["rows"] != 2485 or report["metrics"]["centers"] != 5
        or report["model_selection_used_holdout"] is not False
        or report["formal_use_allowed"] is not False
        or report["deployment_allowed"] is not False
    ):
        raise V3ForwardSurrogateError("V3 holdout report or complete manifest changed")
    return complete
