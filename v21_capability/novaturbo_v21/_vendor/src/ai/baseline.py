"""
WP0 baseline orchestration for NovaTurbo.

This module freezes and reproduces the existing dataset, surrogate, and
Random-NSGA-II behavior without changing their algorithms.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/novaturbo-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import qmc

from .dataset import DesignSpaceBounds, generate_dataset
from .optimizer import OptimizationConfig, optimize
from .surrogate import (
    INPUT_FEATURES,
    OUTPUT_FEATURES,
    SurrogateConfig,
    load_and_prepare_data,
    load_surrogate,
    train_surrogate,
)


BASELINE_KNOWN_BEHAVIORS = [
    "surrogate normalization statistics are computed from all valid rows before splitting",
    "surrogate train/validation/test split uses a hard-coded torch seed of 42",
    "optimizer engineering constraint fields are recorded but feasibility currently uses is_valid only",
    "optimizer initializes all 19 variables as continuous values, including nominal integer fields",
    "optimizer uses NumPy global random state",
]


class Tee:
    """Write text to multiple streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(_jsonable(payload), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_command(args: Sequence[str], cwd: Path) -> Dict[str, Any]:
    completed = subprocess.run(
        list(args),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "args": list(args),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def load_baseline_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "version",
        "run_name",
        "output_root",
        "dataset",
        "surrogate",
        "optimizer",
        "reproducibility",
        "metrics",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Baseline config is missing keys: {sorted(missing)}")
    if int(config["dataset"]["n_samples"]) <= 0:
        raise ValueError("dataset.n_samples must be positive")
    if int(config["reproducibility"]["repeats"]) <= 0:
        raise ValueError("reproducibility.repeats must be positive")
    return config


def prepare_run_directory(output_root: Path, run_name: str) -> Path:
    run_dir = output_root / run_name
    if run_dir.exists():
        raise FileExistsError(
            f"Baseline artifact already exists and will not be overwritten: {run_dir}"
        )
    run_dir.mkdir(parents=True)
    return run_dir


def collect_environment() -> Dict[str, Any]:
    package_names = [
        "numpy",
        "pandas",
        "torch",
        "scipy",
        "matplotlib",
        "PyYAML",
        "pytest",
    ]
    packages: Dict[str, Optional[str]] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "torch": {
            # torch.__version__ is a TorchVersion (a str subclass) in newer
            # releases; PyYAML's safe dumper does not serialize that subclass.
            "version": str(torch.__version__),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
    }


def collect_git_state(repo_root: Path, run_dir: Path) -> Dict[str, Any]:
    commands = {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        "remotes": ["git", "remote", "-v"],
        "tracked_diff": ["git", "diff", "--binary", "HEAD"],
    }
    results = {name: _run_command(args, repo_root) for name, args in commands.items()}
    (run_dir / "git_diff.patch").write_text(
        results["tracked_diff"]["stdout"] + "\n", encoding="utf-8"
    )
    status = results["status"]["stdout"]
    return {
        "commit": results["commit"]["stdout"],
        "branch": results["branch"]["stdout"],
        "dirty": bool(status),
        "status_porcelain": status.splitlines() if status else [],
        "remotes": results["remotes"]["stdout"].splitlines(),
        "tracked_diff_sha256": sha256_file(run_dir / "git_diff.patch"),
    }


def collect_source_manifest(repo_root: Path) -> List[Dict[str, Any]]:
    excluded_parts = {
        ".git",
        ".pytest_cache",
        "__pycache__",
        "baseline_results",
        "data",
    }
    suffixes = {".py", ".yaml", ".yml", ".toml", ".txt", ".md"}
    records = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(repo_root)
        if any(part in excluded_parts for part in relative.parts):
            continue
        if path.suffix not in suffixes and path.name not in {".gitignore"}:
            continue
        records.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def write_pip_freeze(path: Path, repo_root: Path) -> None:
    result = _run_command([sys.executable, "-m", "pip", "freeze"], repo_root)
    content = result["stdout"]
    if result["stderr"]:
        content += "\n# stderr\n" + result["stderr"]
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def summarize_dataset(dataset: pd.DataFrame) -> Dict[str, Any]:
    valid_mask = dataset["is_valid"].astype(bool)
    valid = dataset.loc[valid_mask]
    summary: Dict[str, Any] = {
        "n_rows": len(dataset),
        "n_columns": len(dataset.columns),
        "columns": list(dataset.columns),
        "valid_count": int(valid_mask.sum()),
        "invalid_count": int((~valid_mask).sum()),
        "valid_rate": float(valid_mask.mean()),
        "warning_counts": {
            str(key): int(value)
            for key, value in dataset["n_warnings"].value_counts(dropna=False).items()
        },
    }
    numeric_columns = [
        "thrust_N",
        "tsfc_kg_N_s",
        "total_mass_kg",
        "thrust_to_weight",
    ]
    summary["valid_performance"] = {}
    for column in numeric_columns:
        series = valid[column].replace([np.inf, -np.inf], np.nan).dropna()
        summary["valid_performance"][column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
        }
    return summary


def build_split_manifest(
    csv_path: Path,
    config: SurrogateConfig,
    split_seed: int,
    output_path: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame = frame.reset_index(names="source_row")
    if "is_valid" in frame.columns:
        frame = frame[frame["is_valid"].astype(bool)].reset_index(drop=True)

    n_rows = len(frame)
    n_test = int(n_rows * config.test_split)
    n_val = int(n_rows * config.val_split)
    n_train = n_rows - n_val - n_test
    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(n_rows, generator=generator).tolist()

    split_by_filtered_index: Dict[int, str] = {}
    boundaries = [
        ("train", permutation[:n_train]),
        ("val", permutation[n_train : n_train + n_val]),
        ("test", permutation[n_train + n_val :]),
    ]
    for split_name, indices in boundaries:
        for filtered_index in indices:
            split_by_filtered_index[filtered_index] = split_name

    manifest = pd.DataFrame(
        {
            "filtered_index": np.arange(n_rows, dtype=int),
            "source_row": frame["source_row"].astype(int),
            "split": [
                split_by_filtered_index[index] for index in range(n_rows)
            ],
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest


def _evaluate_surrogate_checkpoint(
    checkpoint_path: Path,
    csv_path: Path,
    config: SurrogateConfig,
) -> Dict[str, Any]:
    _, _, test_loader, stats = load_and_prepare_data(str(csv_path), config)
    model = load_surrogate(str(checkpoint_path), device="cpu")
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for inputs, expected in test_loader:
            predictions.append(model(inputs).cpu().numpy())
            targets.append(expected.cpu().numpy())
    predicted = np.vstack(predictions)
    expected = np.vstack(targets)

    per_output = {}
    for index, name in enumerate(OUTPUT_FEATURES):
        residual = np.sum((expected[:, index] - predicted[:, index]) ** 2)
        total = np.sum(
            (expected[:, index] - expected[:, index].mean()) ** 2
        )
        rmse = np.sqrt(np.mean((expected[:, index] - predicted[:, index]) ** 2))
        per_output[name] = {
            "r2": float(1.0 - residual / (total + 1e-10)),
            "rmse": float(rmse),
        }
    return {
        "test_count": int(stats["n_test"]),
        "per_output": per_output,
    }


def _plot_surrogate_history(history: Mapping[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    axis.plot(epochs, history["train_loss"], label="train")
    axis.plot(epochs, history["val_loss"], label="validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss")
    axis.set_yscale("log")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_surrogate_replicate(
    csv_path: Path,
    config_data: Mapping[str, Any],
    output_dir: Path,
    deterministic_torch: bool,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True)
    config_kwargs = {
        key: value
        for key, value in config_data.items()
        if key not in {"seed", "split_seed"}
    }
    config = SurrogateConfig(**config_kwargs)
    seed = int(config_data["seed"])
    set_global_seed(seed, deterministic_torch=deterministic_torch)

    started = time.perf_counter()
    _, history = train_surrogate(
        str(csv_path),
        config=config,
        save_dir=str(output_dir),
        verbose=True,
    )
    elapsed = time.perf_counter() - started

    history_frame = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history["train_loss"]) + 1),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "learning_rate": history["lr"],
        }
    )
    history_frame.to_csv(output_dir / "history.csv", index=False)
    _plot_surrogate_history(history, output_dir / "training_curve.png")

    checkpoint_path = output_dir / "best_surrogate.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    evaluation = _evaluate_surrogate_checkpoint(checkpoint_path, csv_path, config)
    metrics = {
        "seed": seed,
        "split_seed": int(config_data["split_seed"]),
        "epochs_completed": len(history["train_loss"]),
        "best_epoch": int(checkpoint["epoch"]) + 1,
        "best_validation_loss": float(checkpoint["val_loss"]),
        "last_validation_loss": float(history["val_loss"][-1]),
        "test_loss_from_last_epoch_model": float(history["test_loss"]),
        "elapsed_seconds": elapsed,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        **evaluation,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_yaml(output_dir / "config.yaml", dataclasses.asdict(config))
    return metrics


def _optimizer_initial_population(
    bounds: DesignSpaceBounds,
    population_size: int,
    seed: int,
) -> pd.DataFrame:
    param_names = list(bounds.__dataclass_fields__.keys())
    lower = np.array([getattr(bounds, name)[0] for name in param_names])
    upper = np.array([getattr(bounds, name)[1] for name in param_names])
    rng = np.random.RandomState(seed)
    genes = lower + rng.random((population_size, len(param_names))) * (upper - lower)
    return pd.DataFrame(genes, columns=param_names)


def _pareto_frame(pareto: Iterable[Any], param_names: Sequence[str]) -> pd.DataFrame:
    rows = []
    for index, individual in enumerate(pareto):
        row: Dict[str, Any] = {"pareto_index": index}
        for gene_index, name in enumerate(param_names):
            row[f"gene__{name}"] = float(individual.genes[gene_index])
        for objective_index, value in enumerate(individual.objectives):
            row[f"objective_internal_{objective_index}"] = float(value)
        for name, value in individual.performance.items():
            row[f"performance__{name}"] = value
        row["is_feasible"] = bool(individual.is_feasible)
        row["rank"] = int(individual.rank)
        row["crowding_distance"] = float(individual.crowding_distance)
        rows.append(row)
    return pd.DataFrame(rows)


def _history_frame(
    history: Mapping[str, Any],
    config: OptimizationConfig,
) -> pd.DataFrame:
    rows = []
    best = history.get("best_objectives", [])
    for index in range(len(history["pareto_size"])):
        generation = index + 1
        row: Dict[str, Any] = {
            "generation": generation,
            "evaluation_count": (generation + 1) * config.population_size,
            "pareto_size": history["pareto_size"][index],
            "feasible_count": history["feasible_count"][index],
        }
        if index < len(best):
            for objective_index, name in enumerate(config.objectives):
                internal_value = float(best[index][objective_index])
                row[f"mean_internal__{name}"] = internal_value
                row[f"mean_performance__{name}"] = (
                    -internal_value
                    if config.maximize[objective_index]
                    else internal_value
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _objective_scaling_from_dataset(
    dataset: pd.DataFrame,
    config: OptimizationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    valid = dataset[dataset["is_valid"].astype(bool)]
    values = []
    for index, name in enumerate(config.objectives):
        column = valid[name].to_numpy(dtype=float)
        if config.maximize[index]:
            column = -column
        values.append(column)
    matrix = np.column_stack(values)
    return np.nanmin(matrix, axis=0), np.nanmax(matrix, axis=0)


def approximate_hypervolume(
    points: np.ndarray,
    objective_min: np.ndarray,
    objective_max: np.ndarray,
    reference_point: np.ndarray,
    sobol_power: int,
) -> Dict[str, Any]:
    scale = np.maximum(objective_max - objective_min, 1e-12)
    normalized = (points - objective_min) / scale
    keep = np.all(np.isfinite(normalized), axis=1) & np.all(
        normalized <= reference_point, axis=1
    )
    normalized = normalized[keep]
    if len(normalized) == 0:
        return {
            "value": 0.0,
            "standard_error": 0.0,
            "sample_count": 2**sobol_power,
            "retained_points": 0,
        }

    sampler = qmc.Sobol(d=normalized.shape[1], scramble=False)
    unit_samples = sampler.random_base2(m=sobol_power)
    samples = unit_samples * reference_point
    dominated_count = 0
    chunk_size = 8192
    for start in range(0, len(samples), chunk_size):
        chunk = samples[start : start + chunk_size]
        dominated = np.any(
            np.all(normalized[:, None, :] <= chunk[None, :, :], axis=2),
            axis=0,
        )
        dominated_count += int(dominated.sum())
    probability = dominated_count / len(samples)
    box_volume = float(np.prod(reference_point))
    value = probability * box_volume
    standard_error = (
        np.sqrt(probability * (1.0 - probability) / len(samples)) * box_volume
    )
    return {
        "value": float(value),
        "standard_error": float(standard_error),
        "sample_count": len(samples),
        "retained_points": len(normalized),
        "reference_point": reference_point.tolist(),
        "normalization_min": objective_min.tolist(),
        "normalization_max": objective_max.tolist(),
        "method": "deterministic Sobol estimate in normalized objective space",
    }


def _plot_optimizer_history(history: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(history["evaluation_count"], history["pareto_size"])
    axes[0].set_xlabel("Physics evaluations")
    axes[0].set_ylabel("Pareto size")
    axes[0].grid(alpha=0.3)
    axes[1].plot(history["evaluation_count"], history["feasible_count"])
    axes[1].set_xlabel("Physics evaluations")
    axes[1].set_ylabel("Feasible population")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_pareto(pareto: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(
        pareto["performance__tsfc_kg_N_s"],
        pareto["performance__thrust_N"],
        s=28,
        alpha=0.8,
    )
    axis.set_xlabel("TSFC [kg/(N·s)]")
    axis.set_ylabel("Thrust [N]")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_optimizer_replicate(
    dataset: pd.DataFrame,
    config_data: Mapping[str, Any],
    metrics_config: Mapping[str, Any],
    output_dir: Path,
    deterministic_torch: bool,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True)
    config_kwargs = {
        key: value for key, value in config_data.items() if key != "seed"
    }
    config = OptimizationConfig(**config_kwargs)
    seed = int(config_data["seed"])
    set_global_seed(seed, deterministic_torch=deterministic_torch)
    bounds = DesignSpaceBounds()
    param_names = list(bounds.__dataclass_fields__.keys())

    initial = _optimizer_initial_population(bounds, config.population_size, seed)
    initial.to_csv(output_dir / "expected_initial_population.csv", index=False)
    initial_sha256 = sha256_file(output_dir / "expected_initial_population.csv")

    np.random.seed(seed)
    started = time.perf_counter()
    pareto, history = optimize(config=config, bounds=bounds, verbose=True)
    elapsed = time.perf_counter() - started

    pareto_table = _pareto_frame(pareto, param_names)
    history_table = _history_frame(history, config)
    pareto_table.to_csv(output_dir / "pareto.csv", index=False)
    history_table.to_csv(output_dir / "history.csv", index=False)
    _plot_optimizer_history(history_table, output_dir / "convergence.png")
    _plot_pareto(pareto_table, output_dir / "pareto_thrust_tsfc.png")

    objective_points = np.array(
        [individual.objectives for individual in pareto], dtype=float
    )
    objective_min, objective_max = _objective_scaling_from_dataset(dataset, config)
    hypervolume = approximate_hypervolume(
        objective_points,
        objective_min,
        objective_max,
        np.asarray(metrics_config["normalized_reference_point"], dtype=float),
        int(metrics_config["hypervolume_sobol_power"]),
    )

    performance_summary = {}
    for name in config.objectives:
        values = pareto_table[f"performance__{name}"].astype(float)
        performance_summary[name] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }

    metrics = {
        "seed": seed,
        "population_size": config.population_size,
        "n_generations": config.n_generations,
        "total_physics_evaluations": config.population_size
        * (config.n_generations + 1),
        "elapsed_seconds": elapsed,
        "pareto_size": len(pareto),
        "feasible_final_population": int(history["feasible_count"][-1]),
        "expected_initial_population_sha256": initial_sha256,
        "pareto_csv_sha256": sha256_file(output_dir / "pareto.csv"),
        "history_csv_sha256": sha256_file(output_dir / "history.csv"),
        "hypervolume_mc_normalized_4d": hypervolume,
        "igd": None,
        "igd_reason": "No independent reference Pareto front exists in WP0.",
        "performance_summary": performance_summary,
        "feasibility_definition": "evaluate_design.is_valid only",
    }
    write_json(output_dir / "metrics.json", metrics)
    write_yaml(output_dir / "config.yaml", dataclasses.asdict(config))
    return metrics


def compare_replicates(
    replicate_dirs: Sequence[Path],
    relative_tolerance: float,
    absolute_tolerance: float,
) -> Dict[str, Any]:
    if len(replicate_dirs) < 2:
        return {
            "repeat_count": len(replicate_dirs),
            "status": "not_checked",
            "reason": "At least two repeats are required.",
        }

    reference_surrogate = pd.read_csv(
        replicate_dirs[0] / "surrogate" / "history.csv"
    )
    reference_optimizer = pd.read_csv(
        replicate_dirs[0] / "nsga2" / "history.csv"
    )
    reference_pareto = pd.read_csv(
        replicate_dirs[0] / "nsga2" / "pareto.csv"
    )
    checks = []
    for replicate_dir in replicate_dirs[1:]:
        surrogate = pd.read_csv(replicate_dir / "surrogate" / "history.csv")
        optimizer = pd.read_csv(replicate_dir / "nsga2" / "history.csv")
        pareto = pd.read_csv(replicate_dir / "nsga2" / "pareto.csv")
        surrogate_equal = (
            surrogate.shape == reference_surrogate.shape
            and np.allclose(
                surrogate.select_dtypes(include=[np.number]),
                reference_surrogate.select_dtypes(include=[np.number]),
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
        )
        optimizer_equal = (
            optimizer.shape == reference_optimizer.shape
            and np.allclose(
                optimizer.select_dtypes(include=[np.number]),
                reference_optimizer.select_dtypes(include=[np.number]),
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
        )
        pareto_equal = (
            pareto.shape == reference_pareto.shape
            and list(pareto.columns) == list(reference_pareto.columns)
            and np.allclose(
                pareto.select_dtypes(include=[np.number]),
                reference_pareto.select_dtypes(include=[np.number]),
                rtol=relative_tolerance,
                atol=absolute_tolerance,
                equal_nan=True,
            )
        )
        checks.append(
            {
                "replicate": replicate_dir.name,
                "surrogate_history_equal": surrogate_equal,
                "optimizer_history_equal": optimizer_equal,
                "pareto_equal": pareto_equal,
            }
        )
    passed = all(all(value for key, value in check.items() if key != "replicate") for check in checks)
    return {
        "repeat_count": len(replicate_dirs),
        "status": "passed" if passed else "failed",
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "checks": checks,
    }


def _write_report(
    run_dir: Path,
    config: Mapping[str, Any],
    dataset_summary: Mapping[str, Any],
    replicate_metrics: Sequence[Mapping[str, Any]],
    reproducibility: Mapping[str, Any],
    git_state: Mapping[str, Any],
) -> None:
    first = replicate_metrics[0]
    surrogate = first["surrogate"]
    optimizer = first["optimizer"]
    lines = [
        "# NovaTurbo WP0 Baseline Report",
        "",
        f"- Baseline version: {config['version']}",
        f"- Git commit: {git_state['commit']}",
        f"- Git dirty at execution: {git_state['dirty']}",
        f"- Dataset rows: {dataset_summary['n_rows']}",
        f"- Valid rows: {dataset_summary['valid_count']} ({dataset_summary['valid_rate']:.2%})",
        f"- Surrogate best epoch: {surrogate['best_epoch']}",
        f"- Surrogate best validation loss: {surrogate['best_validation_loss']:.8g}",
        f"- NSGA-II Pareto size: {optimizer['pareto_size']}",
        f"- NSGA-II physics evaluations: {optimizer['total_physics_evaluations']}",
        f"- Reproducibility status: {reproducibility['status']}",
        "",
        "## Frozen baseline behaviors",
        "",
    ]
    lines.extend(f"- {item}" for item in BASELINE_KNOWN_BEHAVIORS)
    lines.extend(
        [
            "",
            "## Artifact entry points",
            "",
            "- data/dataset_10000.csv (name depends on configured sample count)",
            "- splits/split_indices.csv",
            "- replicates/replicate_*/surrogate/",
            "- replicates/replicate_*/nsga2/",
            "- reproducibility_report.json",
            "- baseline_manifest.yaml",
            "",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_wp0_baseline(
    config_path: Path,
    repo_root: Optional[Path] = None,
    output_root_override: Optional[Path] = None,
) -> Path:
    config_path = config_path.resolve()
    repo_root = (repo_root or Path.cwd()).resolve()
    config = load_baseline_config(config_path)
    output_root = (
        output_root_override.resolve()
        if output_root_override is not None
        else (repo_root / config["output_root"]).resolve()
    )
    run_dir = prepare_run_directory(output_root, config["run_name"])
    shutil.copy2(config_path, run_dir / "config.yaml")

    log_path = run_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        tee = Tee(sys.stdout, log_handle)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            started_utc = dt.datetime.now(dt.timezone.utc)
            print(f"NovaTurbo WP0 baseline: {config['run_name']}")
            print(f"Artifacts: {run_dir}")
            try:
                environment = collect_environment()
                write_json(run_dir / "environment.json", environment)
                write_pip_freeze(run_dir / "pip_freeze.txt", repo_root)
                git_state = collect_git_state(repo_root, run_dir)
                source_manifest = collect_source_manifest(repo_root)
                write_json(run_dir / "source_manifest.json", source_manifest)

                data_dir = run_dir / "data"
                dataset = generate_dataset(
                    n_samples=int(config["dataset"]["n_samples"]),
                    output_dir=str(data_dir),
                    seed=int(config["dataset"]["seed"]),
                    verbose=True,
                )
                dataset_path = data_dir / (
                    f"dataset_{int(config['dataset']['n_samples'])}.csv"
                )
                dataset_summary = summarize_dataset(dataset)
                dataset_summary["sha256"] = sha256_file(dataset_path)
                write_json(data_dir / "dataset_summary.json", dataset_summary)

                surrogate_config = SurrogateConfig(
                    **{
                        key: value
                        for key, value in config["surrogate"].items()
                        if key not in {"seed", "split_seed"}
                    }
                )
                split_manifest = build_split_manifest(
                    dataset_path,
                    surrogate_config,
                    int(config["surrogate"]["split_seed"]),
                    run_dir / "splits" / "split_indices.csv",
                )
                split_counts = {
                    key: int(value)
                    for key, value in split_manifest["split"].value_counts().items()
                }
                write_json(
                    run_dir / "splits" / "split_summary.json",
                    {
                        "seed": int(config["surrogate"]["split_seed"]),
                        "counts": split_counts,
                        "sha256": sha256_file(
                            run_dir / "splits" / "split_indices.csv"
                        ),
                    },
                )

                replicate_dirs = []
                replicate_metrics = []
                repeats = int(config["reproducibility"]["repeats"])
                deterministic_torch = bool(
                    config["reproducibility"]["deterministic_torch"]
                )
                for repeat_index in range(1, repeats + 1):
                    replicate_dir = (
                        run_dir / "replicates" / f"replicate_{repeat_index:02d}"
                    )
                    print(f"\n=== Replicate {repeat_index}/{repeats}: surrogate ===")
                    surrogate_metrics = run_surrogate_replicate(
                        dataset_path,
                        config["surrogate"],
                        replicate_dir / "surrogate",
                        deterministic_torch,
                    )
                    print(f"\n=== Replicate {repeat_index}/{repeats}: NSGA-II ===")
                    optimizer_metrics = run_optimizer_replicate(
                        dataset,
                        config["optimizer"],
                        config["metrics"],
                        replicate_dir / "nsga2",
                        deterministic_torch,
                    )
                    replicate_dirs.append(replicate_dir)
                    replicate_metrics.append(
                        {
                            "replicate": repeat_index,
                            "surrogate": surrogate_metrics,
                            "optimizer": optimizer_metrics,
                        }
                    )

                reproducibility = compare_replicates(
                    replicate_dirs,
                    float(config["reproducibility"]["relative_tolerance"]),
                    float(config["reproducibility"]["absolute_tolerance"]),
                )
                write_json(
                    run_dir / "reproducibility_report.json", reproducibility
                )

                completed_utc = dt.datetime.now(dt.timezone.utc)
                manifest = {
                    "baseline_version": config["version"],
                    "run_name": config["run_name"],
                    "status": "complete",
                    "started_utc": started_utc,
                    "completed_utc": completed_utc,
                    "elapsed_seconds": (
                        completed_utc - started_utc
                    ).total_seconds(),
                    "git": git_state,
                    "environment_file": "environment.json",
                    "pip_freeze_file": "pip_freeze.txt",
                    "source_manifest_file": "source_manifest.json",
                    "config_file": "config.yaml",
                    "dataset": {
                        "path": str(dataset_path.relative_to(run_dir)),
                        **dataset_summary,
                    },
                    "split_counts": split_counts,
                    "known_baseline_behaviors": BASELINE_KNOWN_BEHAVIORS,
                    "replicates": replicate_metrics,
                    "reproducibility": reproducibility,
                }
                write_yaml(run_dir / "baseline_manifest.yaml", manifest)
                _write_report(
                    run_dir,
                    config,
                    dataset_summary,
                    replicate_metrics,
                    reproducibility,
                    git_state,
                )
                print(f"\nWP0 baseline complete: {run_dir}")
                return run_dir
            except Exception as exc:
                write_json(
                    run_dir / "failure.json",
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "timestamp_utc": dt.datetime.now(
                            dt.timezone.utc
                        ).isoformat(),
                    },
                )
                print(f"WP0 baseline failed: {type(exc).__name__}: {exc}")
                raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen NovaTurbo WP0 baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/baseline_wp0.yaml"),
        help="WP0 YAML configuration.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the artifact root from the YAML config.",
    )
    args = parser.parse_args(argv)
    run_wp0_baseline(
        args.config,
        repo_root=Path.cwd(),
        output_root_override=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
