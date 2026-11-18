#!/usr/bin/env python3
"""Train the V2.1 ``performance3_log`` forward surrogate from the WP19-21 dataset.

This reproduces the *contract* of
``data/inverse_design/wp19_22_performance_surrogate_v2_ablation/formal/models/performance3_log/seed_43``:

* architecture      ``legacy_surrogate_mlp_geometry19_v1`` (128,256,256,128,64 / relu / dropout 0.1 / BN)
* inputs            19 Geometry19 coordinates, order taken from ``geometry19_schema_v1.yaml``
* outputs           thrust_N, tsfc_kg_N_s, mass_kg in natural-log space
* artefacts         weights_best_val.pt + normalization.npz + model_manifest.json

The repository's own ablation runner additionally needs the historical S0 baseline
weights and the locked/gate-H evaluation pools, none of which are part of this
delivery. This script therefore trains the same arm with the same data split,
transform, optimizer and early-stopping policy, and reports honest test metrics.
It does not overwrite anything: the output directory must not exist.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys

# Required by torch.use_deterministic_algorithms(True) on CUDA >= 10.2.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_performance_surrogate_v2 import (  # noqa: E402
    transform_outputs,
)
from src.ai.geometry19_surrogate import (  # noqa: E402
    TaggedGeometry19MLP,
    save_weight_bundle,
    sha256_file,
)
from src.ai.geometry19_training import (  # noqa: E402
    PERFORMANCE_NAMES,
    _loader,
    _normalize_x,
    _predict,
    _seed_all,
    _train_model,
    fit_normalization,
    load_regression_splits,
)


def _relative_errors(truth: np.ndarray, prediction: np.ndarray) -> dict[str, dict[str, float]]:
    output = {}
    for index, name in enumerate(PERFORMANCE_NAMES):
        relative = np.abs(prediction[:, index] - truth[:, index]) / np.abs(truth[:, index])
        output[name] = {
            "relative_error_p50": float(np.median(relative)),
            "relative_error_p90": float(np.percentile(relative, 90)),
            "relative_error_max": float(relative.max()),
            "relative_error_mean": float(relative.mean()),
        }
    return output


def _r2(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    output = {}
    for index, name in enumerate(PERFORMANCE_NAMES):
        residual = float(((truth[:, index] - prediction[:, index]) ** 2).sum())
        total = float(((truth[:, index] - truth[:, index].mean()) ** 2).sum())
        output[name] = float(1.0 - residual / total) if total > 0.0 else float("nan")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "config/physics/wp19_22_geometry19_surrogate_v1.yaml",
    )
    parser.add_argument(
        "--ablation",
        type=Path,
        default=REPO_ROOT / "config/experiments/wp19_22_performance_surrogate_v2_ablation.yaml",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "data/inverse_design/wp19_22_performance_surrogate_v2_ablation/formal/models/performance3_log/seed_43",
    )
    parser.add_argument("--smoke", action="store_true", help="Small subset, 2 epochs, CPU")
    parser.add_argument("--epochs", type=int, default=None, help="Override the protocol epoch budget")
    parser.add_argument("--patience", type=int, default=None, help="Override the early-stopping patience")
    parser.add_argument(
        "--regression",
        type=Path,
        default=None,
        help="Override the regression JSONL path (used for plumbing smoke tests)",
    )
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    ablation = yaml.safe_load(args.ablation.read_text(encoding="utf-8"))
    schema = yaml.safe_load((REPO_ROOT / protocol["design_schema"]).read_text(encoding="utf-8"))
    feature_names = [item["name"] for item in schema["design_variables"]]
    if len(feature_names) != 19:
        raise SystemExit("Geometry19 schema must expose exactly 19 design variables")

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing surrogate bundle: {output}")

    maximum = 512 if args.smoke else None
    regression = (args.regression or (REPO_ROOT / protocol["regression_dataset"])).resolve()
    splits, target_names = load_regression_splits(
        regression,
        feature_names,
        protocol["campaign_split"],
        maximum,
    )
    if target_names[:3] != PERFORMANCE_NAMES:
        raise SystemExit("Performance target order mismatch")

    training = copy.deepcopy(ablation["training"])
    if args.smoke:
        training.update(ablation["smoke"]["training"])
    if args.epochs is not None:
        training["epochs"] = int(args.epochs)
    if args.patience is not None:
        training["patience"] = int(args.patience)
    device = torch.device("cpu" if args.smoke else args.device)
    architecture = protocol["architecture"]
    mode = "natural_log"
    seed = int(args.seed)

    _seed_all(seed)
    train_y = splits["train"]["y"][:, :3]
    transformed = transform_outputs(train_y, mode)
    normalization = fit_normalization(splits["train"]["x"], transformed)
    x_values = {name: _normalize_x(bucket["x"], normalization) for name, bucket in splits.items()}
    y_values = {
        name: ((transform_outputs(bucket["y"][:, :3], mode) - normalization["output_mean"])
               / normalization["output_std"]).astype(np.float32)
        for name, bucket in splits.items()
    }

    model = TaggedGeometry19MLP(
        len(feature_names),
        len(PERFORMANCE_NAMES),
        architecture["hidden_layers"],
        architecture["activation"],
        float(architecture["dropout"]),
        bool(architecture["batch_norm"]),
    ).to(device)
    model, history, best_epoch = _train_model(
        model,
        _loader(x_values["train"], y_values["train"], int(training["batch_size"]), True, seed),
        x_values["validation"],
        y_values["validation"],
        nn.functional.mse_loss,
        training,
        device,
    )

    model_tag = f"novaturbo_geometry19_performance3_log_seed{seed}_v1"
    artifacts = save_weight_bundle(
        output,
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
            "protocol_hash": sha256_file(args.protocol),
            "ablation_protocol_hash": sha256_file(args.ablation),
            "regression_dataset": (
                str(regression.relative_to(REPO_ROOT))
                if regression.is_relative_to(REPO_ROOT)
                else str(regression)
            ),
            "regression_dataset_hash": sha256_file(regression),
            "training_note": "retrained from delivery-local data; not the published WP19-22 checkpoint",
        },
        normalization,
    )

    metrics: dict[str, object] = {
        "model_tag": model_tag,
        "output_transform": mode,
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "completed_epochs": len(history),
        "best_validation_loss": min(row["validation_loss"] for row in history),
        "split_counts": {name: len(bucket["x"]) for name, bucket in splits.items()},
        "history": history,
        "artifacts": artifacts,
    }
    for split_name in ("validation", "test"):
        truth = splits[split_name]["y"][:, :3]
        normalized_prediction = _predict(model, x_values[split_name], device)
        prediction = np.exp(
            normalized_prediction * normalization["output_std"] + normalization["output_mean"]
        )
        metrics[split_name] = {
            "r2": _r2(truth, prediction),
            "relative": _relative_errors(truth, prediction),
        }
    (output / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "COMPLETE",
        "model_tag": model_tag,
        "output": str(output),
        "best_epoch": best_epoch,
        "epochs": len(history),
        "test": metrics["test"]["relative"],
        "validation": metrics["validation"]["relative"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
