"""WP3 orchestration: freeze, smoke-test, train, and checkpoint generators."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from .baseline import (
    collect_environment,
    collect_git_state,
    collect_source_manifest,
    sha256_file,
    write_json,
    write_yaml,
)
from .generators import (
    CFMGenerator,
    CINNGenerator,
    CVAEGenerator,
    DiagCFMGenerator,
    NFlowsGenerator,
    load_generator,
)
from .generators.base import GENERATOR_REGISTRY
from .generators.data import build_generator_dataloaders


MODEL_LOCK_KEYS = {
    "cvae": "cvae_reference",
    "nflows": "nflows",
    "cinn": "freia",
    "cfm": "torchcfm",
    "diag_cfm": None,
}


def load_wp3_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"version", "lockfile", "data", "runtime", "models", "output"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"WP3 config is missing keys: {sorted(missing)}")
    enabled = [name for name, value in config["models"].items() if value["enabled"]]
    unknown = set(enabled).difference(MODEL_LOCK_KEYS)
    if unknown:
        raise ValueError(f"Unknown WP3 models: {sorted(unknown)}")
    if not enabled:
        raise ValueError("WP3 must enable at least one generator")
    if not config["runtime"]["seeds"]:
        raise ValueError("WP3 runtime.seeds cannot be empty")
    return config


def load_and_validate_lock(lock_path: Path, repo_root: Path) -> Dict[str, Any]:
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    if lock.get("lock_version") != "third_party_models_v1":
        raise ValueError("Unsupported third-party model lock format")
    for name, entry in lock["models"].items():
        commit = entry.get("commit")
        if not isinstance(commit, str) or len(commit) != 40:
            raise ValueError(f"{name} does not freeze a full commit SHA")
        license_path = (repo_root / entry["license_file"]).resolve()
        if not license_path.is_file():
            raise FileNotFoundError(f"Missing frozen license for {name}: {license_path}")
        if sha256_file(license_path) != entry["license_sha256"]:
            raise ValueError(f"Frozen license hash mismatch for {name}")
        package = entry.get("package")
        if package:
            package_name, expected = package.split("==", maxsplit=1)
            installed = importlib.metadata.version(package_name)
            if installed != expected:
                raise RuntimeError(
                    f"{package_name} version mismatch: installed={installed}, expected={expected}"
                )
    for name, entry in lock.get("runtime_dependencies", {}).items():
        license_path = (repo_root / entry["license_file"]).resolve()
        if sha256_file(license_path) != entry["license_sha256"]:
            raise ValueError(f"Frozen license hash mismatch for {name}")
        package_name, expected = entry["package"].split("==", maxsplit=1)
        if importlib.metadata.version(package_name) != expected:
            raise RuntimeError(f"Runtime dependency mismatch for {package_name}")
    return lock


def _toy_batch(
    size: int = 16,
    design_dim: int = 5,
    condition_dim: int = 2,
    bounded: bool = False,
) -> Dict[str, Any]:
    generator = torch.Generator().manual_seed(123)
    return {
        "x": (
            2.0 * torch.rand(size, design_dim, generator=generator) - 1.0
            if bounded
            else torch.randn(size, design_dim, generator=generator)
        ),
        "condition": torch.randn(size, condition_dim, generator=generator),
        "sample_id": [f"toy-{index}" for index in range(size)],
    }


def run_upstream_smoke(config: Mapping[str, Any]) -> pd.DataFrame:
    """Exercise pinned upstream core APIs before project-data adaptation."""
    runtime = {
        "seed": 123,
        "device": "cpu",
        "epochs": 1,
        "patience": 1,
        "gradient_clip_norm": 0.0,
    }
    records = []
    smoke_names = [
        name
        for name in ("cvae", "nflows", "cinn", "cfm")
        if name in config["models"]
    ]
    if "diag_cfm" in config["models"]:
        smoke_names.append("diag_cfm")
    for name in smoke_names:
        model_config = dict(config["models"][name])
        model_config["ode_steps"] = min(int(model_config.get("ode_steps", 2)), 2)
        bounded_config = model_config.get("bounded_output")
        bounded_enabled = bool(
            bounded_config is True
            or (
                isinstance(bounded_config, Mapping)
                and bounded_config.get("enabled", True)
            )
        )
        batch = _toy_batch(
            design_dim=int(model_config.get("design_dim", 5)),
            condition_dim=int(model_config.get("condition_dim", 2)),
            bounded=bounded_enabled,
        )
        model = GENERATOR_REGISTRY[name](model_config, runtime)
        model.to(torch.device("cpu"))
        record = {
            "model": name,
            "status": "failed",
            "loss_finite": False,
            "gradient_finite": False,
            "sample_shape": "",
            "sample_finite": False,
            "inverse_max_error": math.nan,
            "upstream_versions": json.dumps(model._upstream_versions(), sort_keys=True),
            "error": "",
        }
        try:
            loss, _ = model._batch_loss(batch, training=True)
            record["loss_finite"] = bool(torch.isfinite(loss))
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            record["gradient_finite"] = bool(
                gradients and all(torch.isfinite(value).all() for value in gradients)
            )
            sample = model.sample(batch["condition"][0], 8, seed=9)
            record["sample_shape"] = str(tuple(sample.values.shape))
            record["sample_finite"] = bool(torch.isfinite(sample.values).all())
            if name == "cinn":
                x = batch["x"][:8]
                condition = batch["condition"][:8]
                z, _ = model.forward_transform(x, condition)
                restored, _ = model.inverse_transform(z, condition)
                record["inverse_max_error"] = float(
                    (restored - x).abs().max().detach()
                )
            record["status"] = "passed" if all(
                [record["loss_finite"], record["gradient_finite"], record["sample_finite"]]
            ) else "failed"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    frame = pd.DataFrame(records)
    failures = frame[frame["status"] != "passed"]
    if not failures.empty:
        raise RuntimeError(
            "Pinned upstream smoke tests failed: "
            + failures[["model", "error"]].to_dict(orient="records").__repr__()
        )
    return frame


def _validate_training_summary(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    expected_runs = sum(
        bool(value["enabled"]) for value in config["models"].values()
    ) * len(config["runtime"]["seeds"])
    if len(summary) != expected_runs:
        raise RuntimeError("WP3 did not complete every model/seed run")
    if not summary["checkpoint_reproducible"].astype(bool).all():
        raise RuntimeError("WP3 checkpoint reproducibility failed")
    if not summary["sample_finite"].astype(bool).all():
        raise RuntimeError("WP3 produced non-finite samples")
    training_contract = dict(config.get("training_contract", {}))
    if bool(training_contract.get("require_full_train_split", False)) and not (
        summary["train_used"] == summary["train_full"]
    ).all():
        raise RuntimeError("WP3 full-training contract did not use the full train split")
    if bool(training_contract.get("require_full_val_split", False)) and not (
        summary["val_used"] == summary["val_full"]
    ).all():
        raise RuntimeError("WP3 full-training contract did not use the full validation split")
    minimum_completed = int(training_contract.get("minimum_epochs_completed", 1))
    if not (summary["epochs_completed"] >= minimum_completed).all():
        raise RuntimeError(
            "WP3 full-training contract stopped before minimum_epochs_completed"
        )
    if bool(training_contract.get("require_early_stopping", False)) and not summary[
        "early_stopped"
    ].astype(bool).all():
        raise RuntimeError(
            "WP3 full-training contract requires every run to reach early stopping"
        )
    accepted = set(
        str(value)
        for value in training_contract.get(
            "accepted_termination_reasons", ["early_stopping", "max_epochs"]
        )
    )
    actual = set(summary["termination_reason"].astype(str))
    if not actual.issubset(accepted):
        raise RuntimeError(
            f"WP3 has unaccepted termination reasons: {sorted(actual - accepted)}"
        )


def _wp3_manifest(
    config: Mapping[str, Any],
    output_dir: Path,
    target_hash: str,
    summary: pd.DataFrame,
    git_state: Mapping[str, Any],
    *,
    started: Optional[dt.datetime] = None,
    completed: Optional[dt.datetime] = None,
    finalized_existing: bool = False,
) -> Dict[str, Any]:
    completed = completed or dt.datetime.now(dt.timezone.utc)
    started = started or completed
    checkpoint_paths = sorted(output_dir.glob("*/seed_*/*.pt"))
    return {
        "status": "complete",
        "wp_version": config["version"],
        "started_utc": started,
        "completed_utc": completed,
        "elapsed_seconds": (completed - started).total_seconds(),
        "finalized_existing_artifacts": finalized_existing,
        "models": [
            name for name, value in config["models"].items() if value["enabled"]
        ],
        "seeds": [int(value) for value in config["runtime"]["seeds"]],
        "run_count": len(summary),
        "functional_gate_pass": True,
        "performance_gate_applied": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "training_contract": dict(config.get("training_contract", {})),
        "data": {
            "dataset_csv": config["data"]["dataset_csv"],
            "normalizer": config["data"]["normalizer"],
            "target_suite": config["data"]["target_suite"],
            "target_suite_sha256": target_hash,
        },
        "upstream_smoke_sha256": sha256_file(output_dir / "upstream_smoke.csv"),
        "seed_summary_sha256": sha256_file(output_dir / "seed_summary.csv"),
        "checkpoints": {
            path.relative_to(output_dir).as_posix(): sha256_file(path)
            for path in checkpoint_paths
        },
        "git": dict(git_state),
    }


def finalize_existing_wp3(
    config_path: Path,
    output_dir: Path,
    *,
    repo_root: Optional[Path] = None,
) -> Path:
    """Audit and finalize a fully trained directory after a policy-only failure."""

    repo_root = (repo_root or Path.cwd()).resolve()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    config = load_wp3_config(config_path)
    lock = load_and_validate_lock((repo_root / config["lockfile"]).resolve(), repo_root)
    summary_path = output_dir / "seed_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError("Cannot finalize WP3 before seed_summary.csv exists")
    summary = pd.read_csv(summary_path)
    _validate_training_summary(summary, config)
    target_path = (repo_root / config["data"]["target_suite"]).resolve()
    smoke = pd.read_csv(output_dir / "upstream_smoke.csv")
    _write_compatibility_report(
        output_dir / "compatibility_report.md",
        smoke,
        lock,
        str(config["runtime"].get("device", "auto")),
    )
    shutil.copy2(config_path, output_dir / "completion_config.yaml")
    environment = collect_environment()
    environment["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    write_json(output_dir / "environment.json", environment)
    git_state = collect_git_state(repo_root, output_dir)
    write_json(output_dir / "source_manifest.json", collect_source_manifest(repo_root))
    manifest = _wp3_manifest(
        config,
        output_dir,
        sha256_file(target_path),
        summary,
        git_state,
        finalized_existing=True,
    )
    write_yaml(output_dir / "wp3_manifest.yaml", manifest)
    failure_path = output_dir / "failure.json"
    if failure_path.is_file():
        failure_path.replace(output_dir / "completion_policy_failure.json")
    return output_dir


def _sample_diagnostics(values: torch.Tensor) -> Dict[str, Any]:
    matrix = values.detach().cpu().numpy()
    rounded_unique = np.unique(np.round(matrix, decimals=8), axis=0)
    out_of_bounds = np.any((matrix < -1.0) | (matrix > 1.0), axis=1)
    return {
        "count": len(matrix),
        "finite": bool(np.isfinite(matrix).all()),
        "unique_count_at_1e-8": len(rounded_unique),
        "duplicate_rate": 1.0 - len(rounded_unique) / len(matrix),
        "out_of_bounds_count": int(out_of_bounds.sum()),
        "out_of_bounds_rate": float(out_of_bounds.mean()),
        "mean_pairwise_dimension_std": float(np.std(matrix, axis=0).mean()),
    }


def _write_compatibility_report(
    output_path: Path,
    smoke: pd.DataFrame,
    lock: Mapping[str, Any],
    device: str,
) -> None:
    lines = [
        "# WP3 Compatibility Report",
        "",
        f"- Formal training device: {device}",
        "- Pinned upstream API smoke device: cpu",
        f"- PyTorch: {torch.__version__}",
        f"- CUDA available: {torch.cuda.is_available()}",
        "",
        "## Pinned packages",
        "",
    ]
    for name, entry in lock["models"].items():
        lines.append(
            f"- {name}: package={entry.get('package')}, release={entry.get('release')}, "
            f"commit={entry['commit']}, license={entry['license']}"
        )
    lines.extend(["", "## Smoke tests", ""])
    for row in smoke.itertuples(index=False):
        lines.append(
            f"- {row.model}: {row.status}; loss_finite={row.loss_finite}; "
            f"gradient_finite={row.gradient_finite}; sample_shape={row.sample_shape}; "
            f"inverse_max_error={row.inverse_max_error}"
        )
    lines.extend(
        [
            "",
            "The pinned upstream API smoke is intentionally device-independent and runs "
            "on CPU. GPU execution is verified separately by the formal training history "
            "and environment manifest.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_wp3(
    config_path: Path,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    config_path = config_path.resolve()
    config = load_wp3_config(config_path)
    lock_path = (repo_root / config["lockfile"]).resolve()
    lock = load_and_validate_lock(lock_path, repo_root)
    output_dir = (
        output_dir_override.resolve()
        if output_dir_override is not None
        else (repo_root / config["output"]["directory"]).resolve()
    )
    if output_dir.exists() and bool(config["output"].get("refuse_overwrite", True)):
        raise FileExistsError(f"WP3 output exists and will not be overwritten: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = dt.datetime.now(dt.timezone.utc)
    try:
        shutil.copy2(config_path, output_dir / "build_config.yaml")
        shutil.copy2(lock_path, output_dir / "third_party_models.lock.yaml")
        smoke = run_upstream_smoke(config)
        smoke.to_csv(output_dir / "upstream_smoke.csv", index=False)
        runtime_device = str(config["runtime"].get("device", "auto"))
        _write_compatibility_report(
            output_dir / "compatibility_report.md", smoke, lock, runtime_device
        )

        target_path = (repo_root / config["data"]["target_suite"]).resolve()
        if not target_path.is_file():
            raise FileNotFoundError(f"WP3 target suite not found: {target_path}")
        target_hash = sha256_file(target_path)
        rows: List[Dict[str, Any]] = []
        for seed_value in config["runtime"]["seeds"]:
            seed = int(seed_value)
            for name, model_config_value in config["models"].items():
                if not bool(model_config_value["enabled"]):
                    continue
                train_loader, val_loader, normalizer, schema_version, data_counts = (
                    build_generator_dataloaders(config["data"], repo_root, seed)
                )
                condition = next(iter(val_loader))["condition"][0]
                runtime = dict(config["runtime"])
                runtime["seed"] = seed
                model_config = dict(model_config_value)
                lock_key = MODEL_LOCK_KEYS[name]
                if lock_key is not None:
                    model_config["upstream_lock"] = lock["models"][lock_key]
                model = GENERATOR_REGISTRY[name](
                    model_config,
                    runtime,
                    schema_version=schema_version,
                    normalizer=normalizer,
                )
                training = model.fit(train_loader, val_loader)
                run_dir = output_dir / name / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                best_path = run_dir / "best.pt"
                last_path = run_dir / "last.pt"
                model.save(best_path, state="best")
                model.save(last_path, state="last")
                write_json(run_dir / "training_history.json", training)

                sample_seed = seed + 100000
                count = int(config["runtime"].get("checkpoint_sample_count", 32))
                before = model.sample(condition, count, seed=sample_seed)
                loaded = load_generator(best_path, device=str(model.device))
                after = loaded.sample(condition, count, seed=sample_seed)
                checkpoint_reproducible = torch.equal(before.values, after.values)
                if not checkpoint_reproducible:
                    raise RuntimeError(
                        f"{name} seed={seed} changed samples after checkpoint load"
                    )
                diagnostics = _sample_diagnostics(before.values)
                diagnostics.update(before.diagnostics)
                write_json(run_dir / "sample_diagnostics.json", diagnostics)

                held_out_log_prob = math.nan
                if name in {"nflows", "cinn"}:
                    held_out = next(iter(val_loader))
                    values = model.log_prob(
                        held_out["x"][:64], held_out["condition"][:64]
                    )
                    held_out_log_prob = float(values.mean())
                row = {
                    "model": name,
                    "seed": seed,
                    "best_epoch": training["best_epoch"],
                    "best_validation_loss": training["best_validation_loss"],
                    "epochs_completed": training["epochs_completed"],
                    "early_stopped": training["early_stopped"],
                    "termination_reason": training["termination_reason"],
                    "final_learning_rate": training["final_learning_rate"],
                    "training_elapsed_seconds": training["elapsed_seconds"],
                    "average_epoch_seconds": training["average_epoch_seconds"],
                    "peak_gpu_memory_bytes": training["peak_gpu_memory_bytes"],
                    "total_parameters": training["total_parameters"],
                    "trainable_parameters": training["trainable_parameters"],
                    "best_checkpoint_bytes": best_path.stat().st_size,
                    "checkpoint_reproducible": checkpoint_reproducible,
                    "sample_finite": diagnostics["finite"],
                    "sample_unique_count": diagnostics["unique_count_at_1e-8"],
                    "sample_out_of_bounds_rate": diagnostics["out_of_bounds_rate"],
                    "sample_diversity_std": diagnostics["mean_pairwise_dimension_std"],
                    "held_out_mean_log_prob": held_out_log_prob,
                    "best_checkpoint": best_path.relative_to(output_dir).as_posix(),
                    "last_checkpoint": last_path.relative_to(output_dir).as_posix(),
                    **data_counts,
                }
                if name in {"cfm", "diag_cfm"}:
                    row["ode_nfe"] = diagnostics.get("nfe")
                    row["ode_wall_seconds"] = diagnostics.get("wall_seconds")
                rows.append(row)

        summary = pd.DataFrame(rows)
        summary.to_csv(output_dir / "seed_summary.csv", index=False)
        _validate_training_summary(summary, config)

        environment = collect_environment()
        environment["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        write_json(output_dir / "environment.json", environment)
        git_state = collect_git_state(repo_root, output_dir)
        write_json(
            output_dir / "source_manifest.json", collect_source_manifest(repo_root)
        )
        completed = dt.datetime.now(dt.timezone.utc)
        manifest = _wp3_manifest(
            config,
            output_dir,
            target_hash,
            summary,
            git_state,
            started=started,
            completed=completed,
        )
        write_yaml(output_dir / "wp3_manifest.yaml", manifest)
        return output_dir
    except Exception as exc:
        write_json(
            output_dir / "failure.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "timestamp_utc": dt.datetime.now(dt.timezone.utc),
            },
        )
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run WP3 open-source conditional generator adaptation."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config/generator_wp3.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.finalize_existing:
        if args.output_dir is None:
            config = load_wp3_config(args.config)
            output_dir = Path(config["output"]["directory"])
        else:
            output_dir = args.output_dir
        output = finalize_existing_wp3(
            args.config, output_dir, repo_root=Path.cwd()
        )
    else:
        output = run_wp3(
            args.config,
            repo_root=Path.cwd(),
            output_dir_override=args.output_dir,
        )
    print(f"WP3 generator adaptation complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
