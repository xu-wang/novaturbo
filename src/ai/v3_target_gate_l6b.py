"""Train an isolated V3 target-conditioned local-center gate on physical GPU 6."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from src.ai.v3_target_route_protocol import verify_v3_target_route_freeze


CONFIG = "config/experiments/geometry22_v3_target_gate_l6b_gpu6_v1.yaml"
CODE = "src/ai/v3_target_gate_l6b.py"
NAMES = ("thrust_N", "tsfc_kg_N_s", "mass_kg")


class V3TargetGateError(ValueError):
    """The V3 L6b route data, device, or sealed model asset is invalid."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_new(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(value if isinstance(value, bytes) else _bytes(value))


def _config(root: Path) -> dict[str, Any]:
    cfg = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    gate, local = cfg.get("gate", {}), cfg.get("local_forward", {})
    if (
        cfg.get("version") != "geometry22_v3_target_gate_l6b_gpu6_v1"
        or cfg.get("scope") != "isolated_software_target_gate_and_local_forward_experiment"
        or cfg.get("runtime_condition") != list(NAMES)
        or cfg.get("route_classes") != 23
        or cfg.get("input_transform") != "frozen_l6a_natural_log_and_equal_center_fit_normalization"
        or cfg.get("fit_split") != "development_internal_fit_only"
        or cfg.get("model_selection_split") != "development_internal_tuning_only"
        or cfg.get("old_validation_and_holdout_content") != "not_parsed"
        or gate.get("architecture") != [3, 64, 64, 23]
        or gate.get("activation") != "silu"
        or gate.get("seeds") != [42, 43, 44]
        or gate.get("training_loss") != "center_balanced_cross_entropy"
        or gate.get("checkpoint_selection") != "minimum_center_balanced_tuning_nll_then_earliest_epoch"
        or gate.get("neural_model_selection") != "minimum_center_balanced_tuning_nll_then_model_id"
        or gate.get("runtime_support_policy") != "mask_to_frozen_eligible_centers_then_top1_or_top2_else_abstain"
        or gate.get("device") != "cuda:0"
        or gate.get("physical_gpu_index") != 6
        or gate.get("cublas_workspace_config") != ":4096:8"
        or local.get("input") != "19_continuous_normalized_design_residual_divided_by_0_08"
        or local.get("output") != "three_natural_log_performances"
        or local.get("ridge_alphas") != [0.01, 0.1, 1.0, 10.0]
        or local.get("selection") != "minimum_tuning_worst_center_p95_max_abs_log_error_then_overall_p95_then_model_id"
        or cfg.get("source_records_training_allowed") is not False
        or cfg.get("isolated_software_training_authorized") is not True
        or cfg.get("formal_use_allowed") is not False
        or cfg.get("deployment_allowed") is not False
    ):
        raise V3TargetGateError("L6b protocol changed scope")
    return cfg


def _source(root: Path, cfg: dict[str, Any]) -> tuple[Path, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    base = root / cfg["l6a"]["path"]
    for filename, key in (
        ("complete_manifest.json", "complete_manifest_sha256"),
        ("target_registry.jsonl", "target_registry_sha256"),
        ("normalization.json", "normalization_sha256"),
        ("center_support.json", "center_support_sha256"),
    ):
        if _sha(base / filename) != cfg["l6a"][key]:
            raise V3TargetGateError(f"L6a source changed: {filename}")
    lock = verify_v3_target_route_freeze(root, base)
    if lock["holdout_content_parsed"] is not False or lock["validation_content_parsed"] is not False:
        raise V3TargetGateError("L6a source is not pre-audit frozen")
    source = root / cfg["source_development"]["path"]
    if _sha(source) != cfg["source_development"]["sha256"]:
        raise V3TargetGateError("V3 development labels changed")
    targets = _rows(base / "target_registry.jsonl")
    normalization = json.loads((base / "normalization.json").read_text(encoding="utf-8"))
    support = json.loads((base / "center_support.json").read_text(encoding="utf-8"))["centers"]
    development = _rows(source)
    if (len(targets), len(development), len(support)) != (11289, 11289, 23):
        raise V3TargetGateError("L6b source population changed")
    by_design = {r["design_id"]: r for r in development}
    if len(by_design) != len(development) or [s["center_id"] for s in support] != lock["route_center_ids"]:
        raise V3TargetGateError("L6b design or center identity changed")
    for target in targets:
        row = by_design.get(target["target_id"])
        if (
            row is None
            or row["center_design_id"] != target["source_center_id"]
            or row["performance"] != target["target_performance"]
            or target["internal_split"] not in ("fit", "tuning")
            or row["training_allowed"] is not False
        ):
            raise V3TargetGateError("L6b target supervision changed")
    return base, targets, normalization, support, development


def _device() -> tuple[torch.device, dict[str, Any]]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6" or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise V3TargetGateError("Set CUDA_VISIBLE_DEVICES=6 and CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise V3TargetGateError("Physical GPU 6 is unavailable")
    physical = subprocess.check_output(
        ["nvidia-smi", "-i", "6", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    if physical.split(",")[0].strip() != "6":
        raise V3TargetGateError("nvidia-smi physical GPU mapping changed")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda:0"), {"visible_devices": "6", "logical_device": "cuda:0",
                                    "physical_device": physical, "torch": torch.__version__,
                                    "cuda": torch.version.cuda, "name": torch.cuda.get_device_name(0)}


def _network() -> nn.Module:
    return nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 23))


def _inputs(targets: list[dict[str, Any]], normalization: dict[str, Any],
            support: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(normalization["log_mean"], dtype=np.float64)
    std = np.asarray(normalization["log_std"], dtype=np.float64)
    if mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0):
        raise V3TargetGateError("L6a condition normalization changed")
    performance = np.asarray([[target["target_performance"][name] for name in NAMES] for target in targets], dtype=np.float64)
    if not np.isfinite(performance).all() or np.any(performance <= 0):
        raise V3TargetGateError("L6b target condition changed")
    x = (np.log(performance) - mean) / std
    classes = {item["center_id"]: item["route_class_index"] for item in support}
    y = np.asarray([classes[target["source_center_id"]] for target in targets], dtype=np.int64)
    split = np.asarray([target["internal_split"] for target in targets])
    if len(set(classes.values())) != 23 or np.sum(split == "fit") != 8992 or np.sum(split == "tuning") != 2297:
        raise V3TargetGateError("L6b class or split count changed")
    return x, y, split


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _support_eligible(x: np.ndarray, support: list[dict[str, Any]]) -> np.ndarray:
    centroid = np.asarray([s["target_centroid_normalized_log"] for s in support])
    radius = np.asarray([s["target_radius_p95_normalized_log"] for s in support])
    return np.linalg.norm(x[:, None, :] - centroid[None, :, :], axis=2) <= radius[None, :]


def _gate_metrics(logits: np.ndarray, labels: np.ndarray, x: np.ndarray,
                  support: list[dict[str, Any]]) -> dict[str, Any]:
    if logits.shape != (len(labels), 23) or not np.isfinite(logits).all():
        raise V3TargetGateError("L6b gate logits are invalid")
    probabilities = _softmax(logits)
    order = np.argsort(-logits, axis=1, kind="stable")
    source = np.arange(len(labels))
    nll = -np.log(np.maximum(probabilities[source, labels], 1e-15))
    top1 = order[:, 0] == labels
    top2 = np.any(order[:, :2] == labels[:, None], axis=1)
    center_ids = sorted(set(labels.tolist()))
    confidence = probabilities.max(axis=1)
    correct = top1.astype(float)
    ece = 0.0
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        mask = (confidence >= low) & (confidence < high if index < 9 else confidence <= high)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    eligible = _support_eligible(x, support)
    masked = np.where(eligible, logits, -np.inf)
    active = eligible.any(axis=1)
    masked_order = np.argsort(-masked, axis=1, kind="stable")
    runtime_top1 = (masked_order[:, 0] == labels) & active
    runtime_top2 = np.any(masked_order[:, :2] == labels[:, None], axis=1) & active
    metrics = {
        "rows": len(labels), "centers": len(center_ids),
        "macro_top1": float(np.mean([top1[labels == center].mean() for center in center_ids])),
        "macro_top2": float(np.mean([top2[labels == center].mean() for center in center_ids])),
        "macro_nll": float(np.mean([nll[labels == center].mean() for center in center_ids])),
        "micro_top1": float(top1.mean()), "micro_top2": float(top2.mean()),
        "ece_10_bins": ece,
        "runtime_support_coverage": float(active.mean()),
        "source_center_support_eligible_fraction": float(eligible[source, labels].mean()),
        "runtime_source_top1_all_rows": float(runtime_top1.mean()),
        "runtime_source_top2_all_rows": float(runtime_top2.mean()),
        "runtime_source_top1_covered_rows": float(runtime_top1[active].mean()) if active.any() else None,
        "runtime_source_top2_covered_rows": float(runtime_top2[active].mean()) if active.any() else None,
        "per_center_top1": {support[center]["center_id"]: float(top1[labels == center].mean()) for center in center_ids},
        "per_center_top2": {support[center]["center_id"]: float(top2[labels == center].mean()) for center in center_ids},
    }
    return metrics


def _baseline_logits(x: np.ndarray, support: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    centroid = np.asarray([s["target_centroid_normalized_log"] for s in support])
    radius = np.asarray([s["target_radius_p95_normalized_log"] for s in support])
    distance = np.linalg.norm(x[:, None, :] - centroid[None, :, :], axis=2) / radius[None, :]
    return {"uniform_prior": np.zeros_like(distance), "nearest_frozen_center_target_support": -distance}


def _predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float64)


def _train_gate(x_fit: np.ndarray, y_fit: np.ndarray, x_tune: np.ndarray, y_tune: np.ndarray,
                seed: int, cfg: dict[str, Any], device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gate = cfg["gate"]
    model = _network().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=gate["learning_rate"], weight_decay=gate["weight_decay"])
    fit_x = torch.as_tensor(x_fit, dtype=torch.float32, device=device)
    fit_y = torch.as_tensor(y_fit, dtype=torch.long, device=device)
    tune_x = torch.as_tensor(x_tune, dtype=torch.float32, device=device)
    tune_y = torch.as_tensor(y_tune, dtype=torch.long, device=device)
    fit_counts = torch.bincount(fit_y, minlength=23).double()
    tune_counts = torch.bincount(tune_y, minlength=23).double()
    if torch.any(fit_counts == 0) or torch.any(tune_counts == 0):
        raise V3TargetGateError("L6b balanced gate requires every center in both splits")
    fit_weights = (len(fit_y) / (23 * fit_counts[fit_y])).float()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best, best_epoch, best_state, stale, anchor = math.inf, 0, None, 0, math.inf
    history = []
    for epoch in range(1, gate["maximum_epochs"] + 1):
        model.train()
        order = torch.randperm(len(fit_y), generator=generator)
        for batch in order.split(gate["batch_size"]):
            indices = batch.to(device)
            loss = (F.cross_entropy(model(fit_x[indices]), fit_y[indices], reduction="none") * fit_weights[indices]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            losses = F.cross_entropy(model(tune_x), tune_y, reduction="none")
            value = float(torch.stack([losses[tune_y == center].mean() for center in range(23)]).mean().item())
        if not math.isfinite(value):
            raise V3TargetGateError("L6b gate tuning NLL is non-finite")
        history.append({"epoch": epoch, "center_balanced_tuning_nll": value})
        if value < best:
            best, best_epoch = value, epoch
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        if value < anchor - gate["min_delta"]:
            anchor, stale = value, 0
        else:
            stale += 1
        if epoch >= gate["minimum_epochs"] and stale >= gate["patience"]:
            break
    if best_state is None:
        raise V3TargetGateError("L6b gate checkpoint was not selected")
    model.load_state_dict(best_state)
    return model, {"seed": seed, "best_epoch": best_epoch, "completed_epochs": len(history),
                   "best_center_balanced_tuning_nll": best, "history": history}


def _local_metrics(truth: np.ndarray, prediction: np.ndarray, centers: list[str]) -> dict[str, Any]:
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise V3TargetGateError("L6b local forward prediction is invalid")
    row = np.max(np.abs(prediction - truth), axis=1)
    by_center: dict[str, list[float]] = defaultdict(list)
    for value, center in zip(row, centers, strict=True):
        by_center[center].append(float(value))
    p95 = {center: float(np.quantile(values, 0.95)) for center, values in sorted(by_center.items())}
    return {"rows": len(row), "centers": len(p95), "overall_p50_max_abs_log_error": float(np.quantile(row, 0.5)),
            "overall_p95_max_abs_log_error": float(np.quantile(row, 0.95)),
            "worst_center_p95_max_abs_log_error": max(p95.values()), "center_p95_max_abs_log_error": p95}


def _fit_local(targets: list[dict[str, Any]], development: list[dict[str, Any]],
               support: list[dict[str, Any]], cfg: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    geometry = {row["design_id"]: row["normalized_design"] for row in development}
    schema = yaml.safe_load((Path(cfg["repository_root"]) / "config/physics/geometry22_schema_v3_development_v1.yaml").read_text())
    variables = sorted(schema["design_variables"], key=lambda item: item["index"])
    continuous = [index for index, variable in enumerate(variables) if variable["dtype"] != "int64"]
    if len(continuous) != 19:
        raise V3TargetGateError("L6b expected 19 continuous design coordinates")
    center_lookup = {item["center_id"]: item for item in support}
    models: dict[str, dict[str, Any]] = {"center_constant": {}}
    for alpha in cfg["local_forward"]["ridge_alphas"]:
        models[f"ridge_{str(alpha).replace('.', '_')}"] = {}
    truth: list[np.ndarray] = []
    labels: list[str] = []
    predictions: dict[str, list[np.ndarray]] = {model_id: [] for model_id in models}
    for center in sorted(center_lookup):
        group = [r for r in targets if r["source_center_id"] == center]
        fit = [r for r in group if r["internal_split"] == "fit"]
        tune = [r for r in group if r["internal_split"] == "tuning"]
        if len(fit) < 100 or len(tune) < 20:
            raise V3TargetGateError("L6b local center has too few fitting/tuning rows")
        anchor = np.asarray(center_lookup[center]["center_normalized_design"])[continuous]

        def input_tensor(rows: list[dict[str, Any]]) -> torch.Tensor:
            values = np.asarray([np.asarray(geometry[r["target_id"]])[continuous] for r in rows])
            return torch.as_tensor((values - anchor) / 0.08, dtype=torch.float64, device=device)

        def target_tensor(rows: list[dict[str, Any]]) -> torch.Tensor:
            values = [[r["target_performance"][name] for name in NAMES] for r in rows]
            return torch.log(torch.as_tensor(values, dtype=torch.float64, device=device))

        fit_x, fit_y = input_tensor(fit), target_tensor(fit)
        tune_x, tune_y = input_tensor(tune), target_tensor(tune)
        x_mean, y_mean = fit_x.mean(dim=0), fit_y.mean(dim=0)
        centered_x, centered_y = fit_x - x_mean, fit_y - y_mean
        predictions["center_constant"].append(y_mean.expand(len(tune), 3).cpu().numpy())
        models["center_constant"][center] = {"fit_rows": len(fit), "intercept_log": y_mean.cpu().tolist()}
        for alpha in cfg["local_forward"]["ridge_alphas"]:
            model_id = f"ridge_{str(alpha).replace('.', '_')}"
            coef = torch.linalg.solve(centered_x.T @ centered_x + float(alpha) * torch.eye(19, dtype=torch.float64, device=device),
                                      centered_x.T @ centered_y)
            intercept = y_mean - x_mean @ coef
            predicted = tune_x @ coef + intercept
            predictions[model_id].append(predicted.cpu().numpy())
            models[model_id][center] = {"fit_rows": len(fit), "intercept_log": intercept.cpu().tolist(),
                                        "coefficients": coef.cpu().tolist(), "alpha": alpha}
        truth.append(tune_y.cpu().numpy())
        labels.extend([center] * len(tune))
    joined_truth = np.concatenate(truth)
    metrics = {model_id: _local_metrics(joined_truth, np.concatenate(parts), labels)
               for model_id, parts in predictions.items()}
    selected = min(models, key=lambda model_id: (metrics[model_id]["worst_center_p95_max_abs_log_error"],
                                                  metrics[model_id]["overall_p95_max_abs_log_error"], model_id))
    return {"models": models, "input_continuous_indices_zero_based": continuous, "output_names": list(NAMES),
            "selected_model_id": selected, "fit_split": "development_internal_fit", "tuning_split": "development_internal_tuning"}, {
                "selected_model_id": selected, "metrics": metrics, "selection_metric": "worst_center_p95_max_abs_log_error",
                "formal_use_allowed": False, "deployment_allowed": False,
            }


def freeze_v3_target_gate(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    cfg = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / cfg["output_root"]
    if output.exists():
        raise V3TargetGateError("L6b output exists; refusing to overwrite")
    _source(root, cfg)
    lock = {"status": "v3_l6b_gpu6_training_protocol_frozen", "config_sha256": _sha(root / CONFIG),
            "code_sha256": _sha(root / CODE), "l6a_complete_manifest_sha256": cfg["l6a"]["complete_manifest_sha256"],
            "target_registry_sha256": cfg["l6a"]["target_registry_sha256"],
            "development_sha256": cfg["source_development"]["sha256"],
            "fit_split": cfg["fit_split"], "model_selection_split": cfg["model_selection_split"],
            "old_validation_and_holdout_content": "not_parsed", "physical_gpu_index": 6,
            "isolated_software_training_authorized": True, "training_executed": False,
            "formal_use_allowed": False, "deployment_allowed": False}
    output.mkdir(parents=True)
    _write_new(output / "protocol_lock.json", lock)
    return verify_v3_target_gate_freeze(root, output)


def verify_v3_target_gate_freeze(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root, output = Path(repository_root).resolve(), Path(output_root).resolve()
    cfg = _config(root)
    _source(root, cfg)
    lock = json.loads((output / "protocol_lock.json").read_text())
    if (
        lock["status"] != "v3_l6b_gpu6_training_protocol_frozen"
        or lock["config_sha256"] != _sha(root / CONFIG)
        or lock["code_sha256"] != _sha(root / CODE)
        or lock["l6a_complete_manifest_sha256"] != cfg["l6a"]["complete_manifest_sha256"]
        or lock["target_registry_sha256"] != cfg["l6a"]["target_registry_sha256"]
        or lock["development_sha256"] != cfg["source_development"]["sha256"]
        or lock["fit_split"] != cfg["fit_split"] or lock["model_selection_split"] != cfg["model_selection_split"]
        or lock["old_validation_and_holdout_content"] != "not_parsed"
        or lock["physical_gpu_index"] != 6
        or lock["isolated_software_training_authorized"] is not True
        or lock["training_executed"] is not False
        or lock["formal_use_allowed"] is not False
        or lock["deployment_allowed"] is not False
    ):
        raise V3TargetGateError("L6b protocol lock changed")
    return lock


def fit_v3_target_gate(repository_root: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    cfg = _config(root)
    output = Path(output_root).resolve() if output_root is not None else root / cfg["output_root"]
    verify_v3_target_gate_freeze(root, output)
    if any((output / name).exists() for name in ("fit_manifest.json", "gate_selection.json", "local_forward_models.json")):
        raise V3TargetGateError("L6b fit exists; refusing to overwrite")
    base, targets, normalization, support, development = _source(root, cfg)
    x, y, split = _inputs(targets, normalization, support)
    fit_mask, tune_mask = split == "fit", split == "tuning"
    fit_x, fit_y = x[fit_mask], y[fit_mask]
    tune_x, tune_y = x[tune_mask], y[tune_mask]
    device, gpu = _device()
    model_dir = output / "models"
    model_dir.mkdir()
    baselines = {name: _gate_metrics(logits, tune_y, tune_x, support)
                 for name, logits in _baseline_logits(tune_x, support).items()}
    neural_models = []
    for seed in cfg["gate"]["seeds"]:
        model, training = _train_gate(fit_x, fit_y, tune_x, tune_y, seed, cfg, device)
        prediction = _predict(model, tune_x, device)
        model_id = f"gate_seed_{seed}"
        path = model_dir / f"{model_id}.pt"
        torch.save({"model_id": model_id, "seed": seed,
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}, path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        reloaded = _network().to(device)
        reloaded.load_state_dict(payload["state_dict"])
        reload_equal = bool(np.array_equal(prediction, _predict(reloaded, tune_x, device)))
        if not reload_equal:
            raise V3TargetGateError("L6b GPU gate checkpoint reload changed predictions")
        neural_models.append({"model_id": model_id, "seed": seed, "path": str(path.relative_to(output)),
                              "sha256": _sha(path), "checkpoint_reload_equal": reload_equal,
                              "training": training, "tuning": _gate_metrics(prediction, tune_y, tune_x, support)})
        print(f"V3 L6b GPU gate trained: {model_id}, best_epoch={training['best_epoch']}", flush=True)
    chosen = min(neural_models, key=lambda item: (item["tuning"]["macro_nll"], item["model_id"]))
    selection = {"selected_neural_model_id": chosen["model_id"], "selected_checkpoint_sha256": chosen["sha256"],
                 "selection_split": "development_internal_tuning", "selection_metric": "macro_nll",
                 "old_validation_and_holdout_content_parsed": False,
                 "baseline_comparison": baselines, "selected_tuning": chosen["tuning"]}
    _write_new(output / "gate_selection.json", selection)
    local_cfg = dict(cfg, repository_root=str(root))
    local_models, local_report = _fit_local(targets, development, support, local_cfg, device)
    _write_new(output / "local_forward_models.json", local_models)
    _write_new(output / "local_forward_report.json", local_report)
    manifest = {"status": "v3_l6b_gpu6_target_gate_and_local_forward_fit_complete_development_only",
                "protocol_lock_sha256": _sha(output / "protocol_lock.json"),
                "gate_selection_sha256": _sha(output / "gate_selection.json"),
                "local_forward_models_sha256": _sha(output / "local_forward_models.json"),
                "local_forward_report_sha256": _sha(output / "local_forward_report.json"),
                "source_l6a_complete_manifest_sha256": _sha(base / "complete_manifest.json"),
                "gpu": gpu, "fit_rows": len(fit_x), "tuning_rows": len(tune_x),
                "neural_models": neural_models,
                "selected_neural_model_id": chosen["model_id"],
                "selected_local_forward_model_id": local_report["selected_model_id"],
                "old_validation_and_holdout_content_parsed": False,
                "candidate_generation_executed": False, "physics_evaluations_executed": 0,
                "formal_use_allowed": False, "deployment_allowed": False}
    _write_new(output / "fit_manifest.json", manifest)
    return verify_v3_target_gate_fit(root, output)


def verify_v3_target_gate_fit(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root, output = Path(repository_root).resolve(), Path(output_root).resolve()
    verify_v3_target_gate_freeze(root, output)
    manifest = json.loads((output / "fit_manifest.json").read_text())
    selection = json.loads((output / "gate_selection.json").read_text())
    local_models = json.loads((output / "local_forward_models.json").read_text())
    local_report = json.loads((output / "local_forward_report.json").read_text())
    chosen = min(manifest["neural_models"], key=lambda item: (item["tuning"]["macro_nll"], item["model_id"]))
    local_chosen = min(local_report["metrics"], key=lambda model_id: (
        local_report["metrics"][model_id]["worst_center_p95_max_abs_log_error"],
        local_report["metrics"][model_id]["overall_p95_max_abs_log_error"], model_id))
    if (
        manifest["status"] != "v3_l6b_gpu6_target_gate_and_local_forward_fit_complete_development_only"
        or manifest["protocol_lock_sha256"] != _sha(output / "protocol_lock.json")
        or manifest["gate_selection_sha256"] != _sha(output / "gate_selection.json")
        or manifest["local_forward_models_sha256"] != _sha(output / "local_forward_models.json")
        or manifest["local_forward_report_sha256"] != _sha(output / "local_forward_report.json")
        or manifest["fit_rows"] != 8992 or manifest["tuning_rows"] != 2297
        or manifest["gpu"]["visible_devices"] != "6" or manifest["gpu"]["logical_device"] != "cuda:0"
        or len(manifest["neural_models"]) != 3
        or {item["model_id"] for item in manifest["neural_models"]} != {"gate_seed_42", "gate_seed_43", "gate_seed_44"}
        or any(_sha(output / item["path"]) != item["sha256"] or item["checkpoint_reload_equal"] is not True
               for item in manifest["neural_models"])
        or manifest["selected_neural_model_id"] != chosen["model_id"]
        or selection["selected_neural_model_id"] != chosen["model_id"]
        or selection["selected_checkpoint_sha256"] != chosen["sha256"]
        or selection["selected_tuning"] != chosen["tuning"]
        or selection["selection_split"] != "development_internal_tuning"
        or selection["old_validation_and_holdout_content_parsed"] is not False
        or manifest["selected_local_forward_model_id"] != local_chosen
        or local_models["selected_model_id"] != local_chosen
        or local_report["selected_model_id"] != local_chosen
        or set(local_models["models"]) != {"center_constant", "ridge_0_01", "ridge_0_1", "ridge_1_0", "ridge_10_0"}
        or manifest["old_validation_and_holdout_content_parsed"] is not False
        or manifest["candidate_generation_executed"] is not False
        or manifest["physics_evaluations_executed"] != 0
        or manifest["formal_use_allowed"] is not False or manifest["deployment_allowed"] is not False
    ):
        raise V3TargetGateError("L6b fit or model selection changed")
    return manifest
