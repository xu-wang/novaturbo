"""Freeze and train a 55-center V3 target gate using development data only."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from src.ai.v3_target_gate_l6b import _device, _fit_local, _softmax


CONFIG = "config/experiments/geometry22_v3_extended_gate_l6f_gpu6_v1.yaml"
CODE = "src/ai/v3_extended_gate_l6f.py"
NAMES = ("thrust_N", "tsfc_kg_N_s", "mass_kg")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write_new(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(value if isinstance(value, bytes) else dump(value))


def config_at(root: Path) -> dict[str, Any]:
    cfg = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if (cfg["version"] != "geometry22_v3_extended_gate_l6f_gpu6_v1"
            or cfg["route_classes"] != 55
            or cfg["runtime_input"] != list(NAMES)
            or cfg["new_internal_split"] != "source_batch_center_band_pair_modulo_5_zero_for_tuning"
            or cfg["input_transform"] != "natural_log_equal_center_fit_mean_std"
            or cfg["target_support"] != "fit_rows_p95_euclidean_normalized_log_radius"
            or cfg["gate"]["architecture"] != [3, 64, 64, 55]
            or cfg["gate"]["seeds"] != [42, 43, 44]
            or cfg["gate"]["physical_gpu_index"] != 6
            or cfg["gate"]["device"] != "cuda:0"
            or cfg["gate"]["cublas_workspace_config"] != ":4096:8"
            or cfg["isolated_software_training_authorized"] is not True
            or cfg["source_records_training_allowed"] is not False
            or cfg["formal_use_allowed"] is not False
            or cfg["deployment_allowed"] is not False):
        raise ValueError("L6f experiment scope changed")
    for name in ("old_targets", "old_center_support", "old_development", "new_registry"):
        spec = cfg[name]
        if sha(root / spec["path"]) != spec["sha256"]:
            raise ValueError(f"L6f source changed: {name}")
    for name in ("new_base", "new_supplement"):
        spec = cfg[name]
        for filename, key in (("complete_manifest.json", "complete_manifest_sha256"),
                              ("coordinates.jsonl", "coordinates_sha256"),
                              ("labels.jsonl", "labels_sha256")):
            if sha(root / spec["root"] / filename) != spec[key]:
                raise ValueError(f"L6f source changed: {name}/{filename}")
    base_manifest = json.loads((root / cfg["new_base"]["root"] / "complete_manifest.json").read_text())
    supplement_manifest = json.loads((root / cfg["new_supplement"]["root"] / "complete_manifest.json").read_text())
    if (base_manifest["data_coverage_gate_pass"] is not False
            or supplement_manifest["data_coverage_gate_pass"] is not True
            or supplement_manifest["combined_candidate_count"] != 15889
            or supplement_manifest["validation_or_holdout_sampled_or_labeled"] is not False):
        raise ValueError("L6f development coverage source changed")
    return cfg


def source_data(root: Path, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    old_targets = rows(root / cfg["old_targets"]["path"])
    old_designs = rows(root / cfg["old_development"]["path"])
    old_support = json.loads((root / cfg["old_center_support"]["path"]).read_text())["centers"]
    new_registry = rows(root / cfg["new_registry"]["path"])
    centers = {item["center_id"]: item for item in old_support}
    centers.update({item["design_id"]: {"center_id": item["design_id"], "center_normalized_design": item["normalized_design"],
                                        "integer_state": {key: item["design"][key] for key in ("compressor_blade_count", "combustor_injector_count", "turbine_blade_count")},
                                        "design_support_radius": cfg["design_support_radius"]}
                    for item in new_registry if item["split"] == "development"})
    if len(old_targets) != 11289 or len(old_designs) != 11289 or len(centers) != 55:
        raise ValueError("L6f 23+32 development center count changed")
    old_by_id = {item["design_id"]: item for item in old_designs}
    if len(old_by_id) != len(old_designs):
        raise ValueError("L6f old design IDs duplicated")
    for record in old_targets:
        source = old_by_id.get(record["target_id"])
        if (source is None or source["center_design_id"] != record["source_center_id"]
                or source["performance"] != record["target_performance"]
                or record["internal_split"] not in ("fit", "tuning")):
            raise ValueError("L6f old development target identity changed")
    designs = list(old_designs)
    targets = list(old_targets)
    for batch in ("new_base", "new_supplement"):
        location = root / cfg[batch]["root"]
        coordinates = rows(location / "coordinates.jsonl")
        labels = rows(location / "labels.jsonl")
        if len(coordinates) != len(labels):
            raise ValueError("L6f new local coordinate/label count changed")
        for coordinate, label in zip(coordinates, labels, strict=True):
            if (coordinate["design_id"] != label["design_id"]
                    or coordinate["center_design_id"] != label["center_design_id"]
                    or coordinate["center_split"] != "development"
                    or label["center_split"] != "development"
                    or coordinate["training_allowed"] is not False
                    or label["training_allowed"] is not False):
                raise ValueError("L6f new development row identity changed")
            if not label["software_regression_candidate"]:
                continue
            performance = label["performance"]
            if performance is None or any(not math.isfinite(performance[name]) or performance[name] <= 0 for name in NAMES):
                raise ValueError("L6f candidate performance invalid")
            targets.append({
                "target_id": coordinate["design_id"],
                "source_center_id": coordinate["center_design_id"],
                "band_index": coordinate["band_index"],
                "source_batch": batch,
                "pair_index": coordinate["pair_index"],
                "internal_split": "tuning" if coordinate["pair_index"] % 5 == 0 else "fit",
                "target_performance": {name: performance[name] for name in NAMES},
            })
            designs.append({"design_id": coordinate["design_id"], "normalized_design": coordinate["normalized_design"]})
    if (len(targets) != 11289 + 15889 or len({item["target_id"] for item in targets}) != len(targets)
            or len({item["design_id"] for item in designs}) != len(designs)):
        raise ValueError("L6f target count or uniqueness changed")
    return targets, designs, centers


def assets(targets: list[dict[str, Any]], centers: dict[str, dict[str, Any]], cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    class_ids = sorted(centers)
    fit = [record for record in targets if record["internal_split"] == "fit"]
    tune = [record for record in targets if record["internal_split"] == "tuning"]
    fit_counts = Counter(record["source_center_id"] for record in fit)
    tune_counts = Counter(record["source_center_id"] for record in tune)
    if set(fit_counts) != set(class_ids) or set(tune_counts) != set(class_ids) or min(fit_counts.values()) < 100 or min(tune_counts.values()) < 20:
        raise ValueError("L6f every class needs sufficient internal fit/tuning rows")
    log_by_center = {center: np.log(np.asarray([[record["target_performance"][name] for name in NAMES]
                                              for record in fit if record["source_center_id"] == center], dtype=np.float64))
                     for center in class_ids}
    center_means = np.asarray([log_by_center[center].mean(axis=0) for center in class_ids])
    mean = center_means.mean(axis=0)
    std = np.sqrt(np.mean([np.mean((log_by_center[center] - mean)**2, axis=0) for center in class_ids], axis=0))
    if np.any(std <= 0):
        raise ValueError("L6f target scale collapsed")
    support = []
    for index, center in enumerate(class_ids):
        cloud = log_by_center[center]
        z = (cloud - mean) / std
        centroid = z.mean(axis=0)
        radius = float(np.quantile(np.linalg.norm(z - centroid, axis=1), .95))
        source = centers[center]
        support.append({
            "center_id": center, "route_class_index": index,
            "fit_rows": len(cloud), "tuning_rows": tune_counts[center],
            "center_normalized_design": source["center_normalized_design"],
            "integer_state": source["integer_state"],
            "design_support_radius": cfg["design_support_radius"],
            "target_centroid_normalized_log": centroid.tolist(),
            "target_radius_p95_normalized_log": radius,
            "target_log_min": cloud.min(axis=0).tolist(),
            "target_log_max": cloud.max(axis=0).tolist(),
        })
    normalization = {"condition_order": list(NAMES), "log_mean": mean.tolist(), "log_std": std.tolist(),
                     "fit_rows": len(fit), "tuning_rows": len(tune), "equal_center_weight": True}
    return normalization, support


def prepare(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    output = root / cfg["output_root"]
    if output.exists():
        raise FileExistsError(output)
    targets, _, centers = source_data(root, cfg)
    normalization, support = assets(targets, centers, cfg)
    output.mkdir(parents=True)
    write_new(output / "target_registry.jsonl", b"".join(dump(item) for item in targets))
    write_new(output / "normalization.json", normalization)
    write_new(output / "center_support.json", {"centers": support})
    lock = {
        "status": "l6f_55_center_development_protocol_frozen_before_training",
        "config_sha256": sha(root / CONFIG), "code_sha256": sha(root / CODE),
        "target_registry_sha256": sha(output / "target_registry.jsonl"),
        "normalization_sha256": sha(output / "normalization.json"),
        "center_support_sha256": sha(output / "center_support.json"),
        "class_count": 55, "fit_rows": normalization["fit_rows"], "tuning_rows": normalization["tuning_rows"],
        "new_validation_or_holdout_labels_parsed": False,
        "isolated_software_training_authorized": True,
        "training_executed": False, "physical_gpu_index": 6,
        "formal_use_allowed": False, "deployment_allowed": False,
    }
    write_new(output / "protocol_lock.json", lock)
    return lock


def network() -> nn.Module:
    return nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 55))


def inputs(targets: list[dict[str, Any]], normalization: dict[str, Any], support: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray([[record["target_performance"][name] for name in NAMES] for record in targets])
    x = (np.log(values) - np.asarray(normalization["log_mean"])) / np.asarray(normalization["log_std"])
    classes = {record["center_id"]: record["route_class_index"] for record in support}
    y = np.asarray([classes[record["source_center_id"]] for record in targets], dtype=np.int64)
    split = np.asarray([record["internal_split"] for record in targets])
    return x, y, split


def metrics(logits: np.ndarray, y: np.ndarray, x: np.ndarray, support: list[dict[str, Any]]) -> dict[str, Any]:
    if logits.shape != (len(y), 55) or not np.isfinite(logits).all():
        raise ValueError("L6f logits invalid")
    probs = _softmax(logits)
    order = np.argsort(-logits, axis=1, kind="stable")
    nll = -np.log(np.maximum(probs[np.arange(len(y)), y], 1e-15))
    centroid = np.asarray([record["target_centroid_normalized_log"] for record in support])
    radius = np.asarray([record["target_radius_p95_normalized_log"] for record in support])
    eligible = np.linalg.norm(x[:, None, :] - centroid[None, :, :], axis=2) <= radius[None, :]
    masked = np.where(eligible, logits, -np.inf)
    masked_order = np.argsort(-masked, axis=1, kind="stable")
    active = eligible.any(axis=1)
    top1 = order[:, 0] == y
    top2 = np.any(order[:, :2] == y[:, None], axis=1)
    return {
        "rows": len(y), "centers": 55,
        "macro_top1": float(np.mean([top1[y == i].mean() for i in range(55)])),
        "macro_top2": float(np.mean([top2[y == i].mean() for i in range(55)])),
        "macro_nll": float(np.mean([nll[y == i].mean() for i in range(55)])),
        "runtime_support_coverage": float(active.mean()),
        "source_center_support_eligible_fraction": float(eligible[np.arange(len(y)), y].mean()),
        "runtime_source_top1_all_rows": float(((masked_order[:, 0] == y) & active).mean()),
        "runtime_source_top2_all_rows": float((np.any(masked_order[:, :2] == y[:, None], axis=1) & active).mean()),
        "minimum_center_top2": float(min(top2[y == i].mean() for i in range(55))),
    }


def predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float64)


def train_one(xfit: np.ndarray, yfit: np.ndarray, xtune: np.ndarray, ytune: np.ndarray, seed: int,
              cfg: dict[str, Any], device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gate = cfg["gate"]
    model = network().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=gate["learning_rate"], weight_decay=gate["weight_decay"])
    fx = torch.as_tensor(xfit, dtype=torch.float32, device=device)
    fy = torch.as_tensor(yfit, dtype=torch.long, device=device)
    tx = torch.as_tensor(xtune, dtype=torch.float32, device=device)
    ty = torch.as_tensor(ytune, dtype=torch.long, device=device)
    fit_counts = torch.bincount(fy, minlength=55).double()
    weights = (len(fy) / (55 * fit_counts[fy])).float()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best, best_epoch, best_state, stale, anchor = math.inf, 0, None, 0, math.inf
    for epoch in range(1, gate["maximum_epochs"] + 1):
        model.train()
        order = torch.randperm(len(fy), generator=generator)
        for batch in order.split(gate["batch_size"]):
            ix = batch.to(device)
            loss = (F.cross_entropy(model(fx[ix]), fy[ix], reduction="none") * weights[ix]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            losses = F.cross_entropy(model(tx), ty, reduction="none")
            value = float(torch.stack([losses[ty == center].mean() for center in range(55)]).mean().item())
        if not math.isfinite(value):
            raise ValueError("L6f tuning NLL nonfinite")
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
        raise ValueError("L6f checkpoint missing")
    model.load_state_dict(best_state)
    return model, {"seed": seed, "best_epoch": best_epoch, "completed_epochs": epoch,
                   "best_center_balanced_tuning_nll": best}


def train(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    output = root / cfg["output_root"]
    lock = json.loads((output / "protocol_lock.json").read_text())
    if (lock["status"] != "l6f_55_center_development_protocol_frozen_before_training"
            or lock["config_sha256"] != sha(root / CONFIG)
            or lock["code_sha256"] != sha(root / CODE)
            or any(lock[key] != sha(output / filename) for key, filename in (
                ("target_registry_sha256", "target_registry.jsonl"),
                ("normalization_sha256", "normalization.json"),
                ("center_support_sha256", "center_support.json")))):
        raise ValueError("L6f pretraining protocol changed")
    if (output / "fit_manifest.json").exists():
        raise FileExistsError(output / "fit_manifest.json")
    targets = rows(output / "target_registry.jsonl")
    normalization = json.loads((output / "normalization.json").read_text())
    support = json.loads((output / "center_support.json").read_text())["centers"]
    expected, designs, centers = source_data(root, cfg)
    if targets != expected or (normalization, support) != assets(expected, centers, cfg):
        raise ValueError("L6f frozen target assets changed")
    x, y, split = inputs(targets, normalization, support)
    fit, tune = split == "fit", split == "tuning"
    if int(fit.sum()) != lock["fit_rows"] or int(tune.sum()) != lock["tuning_rows"]:
        raise ValueError("L6f split changed")
    device, gpu = _device()
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    centroid = np.asarray([record["target_centroid_normalized_log"] for record in support])
    radius = np.asarray([record["target_radius_p95_normalized_log"] for record in support])
    nearest = -np.linalg.norm(x[tune, None, :] - centroid[None, :, :], axis=2) / radius[None, :]
    baselines = {"uniform_prior": metrics(np.zeros_like(nearest), y[tune], x[tune], support),
                 "nearest_frozen_center_target_support": metrics(nearest, y[tune], x[tune], support)}
    trained = []
    for seed in cfg["gate"]["seeds"]:
        model, history = train_one(x[fit], y[fit], x[tune], y[tune], seed, cfg, device)
        logits = predict(model, x[tune], device)
        model_id = f"gate_seed_{seed}"
        path = model_dir / f"{model_id}.pt"
        torch.save({"model_id": model_id, "seed": seed, "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}, path)
        reloaded = network().to(device)
        reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
        if not np.array_equal(logits, predict(reloaded, x[tune], device)):
            raise ValueError("L6f checkpoint reload changed predictions")
        trained.append({"model_id": model_id, "path": str(path.relative_to(output)), "sha256": sha(path),
                        "training": history, "tuning": metrics(logits, y[tune], x[tune], support)})
        print(f"L6f GPU6 trained {model_id}: epoch {history['best_epoch']}", flush=True)
    chosen = min(trained, key=lambda item: (item["tuning"]["macro_nll"], item["model_id"]))
    selection = {"selected_neural_model_id": chosen["model_id"], "selected_checkpoint_sha256": chosen["sha256"],
                 "selection_metric": "development_internal_macro_nll", "baseline_comparison": baselines,
                 "selected_tuning": chosen["tuning"], "new_validation_or_holdout_labels_parsed": False}
    write_new(output / "gate_selection.json", selection)
    local_cfg = dict(cfg, repository_root=str(root))
    local_models, local_report = _fit_local(targets, designs, support, local_cfg, device)
    write_new(output / "local_forward_models.json", local_models)
    write_new(output / "local_forward_report.json", local_report)
    manifest = {
        "status": "l6f_55_center_development_gate_and_local_forward_trained_gpu6",
        "protocol_lock_sha256": sha(output / "protocol_lock.json"),
        "gate_selection_sha256": sha(output / "gate_selection.json"),
        "local_forward_models_sha256": sha(output / "local_forward_models.json"),
        "local_forward_report_sha256": sha(output / "local_forward_report.json"),
        "gpu": gpu, "fit_rows": int(fit.sum()), "tuning_rows": int(tune.sum()),
        "neural_models": trained, "selected_neural_model_id": chosen["model_id"],
        "selected_local_forward_model_id": local_report["selected_model_id"],
        "new_validation_or_holdout_labels_parsed": False,
        "candidate_generation_executed": False, "physics_evaluations_executed": 0,
        "formal_use_allowed": False, "deployment_allowed": False,
    }
    write_new(output / "fit_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "train"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cfg = config_at(root)
    result = prepare(root, cfg) if args.command == "prepare" else train(root, cfg)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
