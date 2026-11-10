"""Freeze and verify the canonical active-10D inverse-design baseline."""

from __future__ import annotations

import copy
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd
import torch
import yaml

from .baseline import (
    collect_environment,
    collect_source_manifest,
    sha256_file,
    write_json,
    write_yaml,
)
from .generator_dataset import GeneratorNormalizer
from .inverse_design_delivery_cfm4_legacy import run_inverse_design_delivery
from .round_trip import derive_round_trip_metrics, load_metrics_config


DEFAULT_BASELINE_CONFIG = Path(
    "config/inverse_design/baselines/active10d_cfm4_v1.yaml"
)


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return dict(value)


def load_inverse_design_baseline_config(path: Path) -> Dict[str, Any]:
    config = _load_yaml(path)
    required = {
        "baseline_version",
        "baseline_id",
        "output_directory",
        "refuse_overwrite",
        "delivery_config",
        "runtime",
        "critical_artifacts",
        "reference_directories",
        "cases",
        "expected_contract",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Baseline config is missing keys: {sorted(missing)}")
    if set(config["cases"]) != {"ordinary", "extreme"}:
        raise ValueError("Baseline cases must be exactly ordinary and extreme")
    if config["refuse_overwrite"] is not True:
        raise ValueError("Formal baseline must set refuse_overwrite=true")
    for name, case in config["cases"].items():
        case_required = {
            "target_config",
            "generation_seed",
            "legacy_reference",
            "compare_artifacts",
        }
        missing_case = case_required.difference(case)
        if missing_case:
            raise ValueError(
                f"Baseline case {name} is missing keys: {sorted(missing_case)}"
            )
        if not case["compare_artifacts"]:
            raise ValueError(f"Baseline case {name} has no comparison artifacts")
    return config


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
        "stdout": completed.stdout.rstrip(),
        "stderr": completed.stderr.rstrip(),
    }


def collect_clean_git_state(repo_root: Path, *, require_clean: bool) -> Dict[str, Any]:
    commit = _run_command(["git", "rev-parse", "HEAD"], repo_root)
    tree = _run_command(["git", "rev-parse", "HEAD^{tree}"], repo_root)
    branch = _run_command(["git", "branch", "--show-current"], repo_root)
    status = _run_command(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], repo_root
    )
    for name, result in {
        "commit": commit,
        "tree": tree,
        "branch": branch,
        "status": status,
    }.items():
        if result["returncode"] != 0:
            raise RuntimeError(f"Unable to collect Git {name}: {result['stderr']}")
    dirty = bool(status["stdout"])
    if require_clean and dirty:
        raise RuntimeError(
            "Formal inverse-design baseline requires a clean Git worktree"
        )
    return {
        "commit": commit["stdout"],
        "tree": tree["stdout"],
        "branch": branch["stdout"],
        "dirty": dirty,
        "status_porcelain": status["stdout"].splitlines(),
    }


def verify_critical_artifacts(
    config: Mapping[str, Any], repo_root: Path
) -> list[Dict[str, Any]]:
    records = []
    for item in config["critical_artifacts"]:
        path = _resolve(repo_root, str(item["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"Critical baseline artifact is missing: {path}")
        actual = sha256_file(path)
        expected = str(item["sha256"])
        if actual != expected:
            raise ValueError(
                f"Critical artifact hash mismatch for {item['path']}: "
                f"expected {expected}, got {actual}"
            )
        records.append(
            {
                "category": str(item["category"]),
                "source_path": str(item["path"]),
                "size_bytes": path.stat().st_size,
                "sha256": actual,
            }
        )
    return records


def verify_reference_directories(
    config: Mapping[str, Any], repo_root: Path
) -> None:
    for value in config["reference_directories"].values():
        path = _resolve(repo_root, str(value))
        if not path.is_dir():
            raise FileNotFoundError(f"Baseline reference directory is missing: {path}")


def _validate_runtime(config: Mapping[str, Any]) -> Dict[str, Any]:
    runtime = config["runtime"]
    require_gpu = bool(runtime.get("require_gpu", True))
    physical_gpu = int(runtime.get("physical_gpu", 6))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    result = {
        "require_gpu": require_gpu,
        "physical_gpu": physical_gpu,
        "logical_device": str(runtime.get("logical_device", "cuda:0")),
        "CUDA_VISIBLE_DEVICES": visible,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "torch_cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
    if not require_gpu:
        return result
    if visible != str(physical_gpu):
        raise RuntimeError(
            f"Formal baseline requires CUDA_VISIBLE_DEVICES={physical_gpu}, got {visible!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Formal baseline requires exactly one CUDA device visible to PyTorch"
        )
    probe = _run_command(["nvidia-smi", "-i", str(physical_gpu)], Path.cwd())
    if probe["returncode"] != 0:
        raise RuntimeError(f"Physical GPU {physical_gpu} is unavailable: {probe['stderr']}")
    return result


def _snapshot_inputs(
    records: list[Dict[str, Any]], output_dir: Path, repo_root: Path
) -> None:
    for record in records:
        source = _resolve(repo_root, record["source_path"])
        destination = output_dir / "inputs" / record["source_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_hash = sha256_file(destination)
        if copied_hash != record["sha256"]:
            raise RuntimeError(f"Snapshot copy hash mismatch: {destination}")
        record["snapshot_path"] = str(destination.relative_to(output_dir))


def _snapshot_reference_directories(
    config: Mapping[str, Any], output_dir: Path, repo_root: Path
) -> None:
    for name, value in config["reference_directories"].items():
        source = _resolve(repo_root, str(value))
        if not source.is_dir():
            raise FileNotFoundError(f"Baseline reference directory is missing: {source}")
        shutil.copytree(source, output_dir / "references" / str(name))


def _checkpoint_contract(config: Mapping[str, Any], repo_root: Path) -> Dict[str, Any]:
    by_category = {
        str(item["category"]): _resolve(repo_root, str(item["path"]))
        for item in config["critical_artifacts"]
    }
    normalizer = GeneratorNormalizer.load(by_category["normalizer"])
    metrics_config = load_metrics_config(by_category["metrics_config"])
    derived = derive_round_trip_metrics(
        by_category["dataset"], normalizer, metrics_config
    )
    cfm = torch.load(
        by_category["cfm_checkpoint"], map_location="cpu", weights_only=False
    )
    surrogate = torch.load(
        by_category["gradient_surrogate"], map_location="cpu", weights_only=False
    )
    normalizer_dict = normalizer.to_dict()
    if cfm.get("normalizer") != normalizer_dict:
        raise ValueError("CFM checkpoint normalizer differs from frozen normalizer")
    if surrogate.get("normalizer") != normalizer_dict:
        raise ValueError("Gradient surrogate normalizer differs from frozen normalizer")
    if surrogate.get("dataset_sha256") != sha256_file(by_category["dataset"]):
        raise ValueError("Gradient surrogate dataset hash differs from frozen dataset")

    delivery = _load_yaml(_resolve(repo_root, config["delivery_config"]))
    policy = _load_yaml(_resolve(repo_root, delivery["policy_config"]))
    generation = policy["generation"]
    gradient = policy["gradient"]
    nsga2 = policy["nsga2"]
    maximum_nsga = int(nsga2["population_size"]) * (
        int(nsga2["n_generations"]) + 1
    )
    expected = config["expected_contract"]
    actual_expected = {
        "candidate_protocol": str(generation["candidate_protocol"]),
        "n_raw": int(generation["n_raw"]),
        "keep_rule": "power_of_two_floor",
        "refill": False,
        "minimum_hit_count": int(policy["delivery"]["minimum_hit_count"]),
        "maximum_direct_0d_evaluations": int(generation["n_raw"]),
        "gradient_maximum_0d_evaluations": int(
            gradient["maximum_0d_evaluations"]
        ),
        "nsga2_maximum_0d_evaluations": maximum_nsga,
        "maximum_total_0d_evaluations": (
            int(generation["n_raw"])
            + int(gradient["maximum_0d_evaluations"])
            + maximum_nsga
        ),
        "frozen_extreme_total_0d_evaluations": 728,
    }
    if actual_expected != dict(expected):
        raise ValueError(
            f"Executable contract differs from baseline expectation: {actual_expected}"
        )
    cfm_training = dict(cfm["training_metadata"])
    cfm_training.pop("history", None)
    surrogate_metrics = dict(surrogate["metrics"])
    surrogate_metrics.pop("history", None)
    return {
        "contract_version": "active10d_inverse_design_contract_v1",
        "normalizer": normalizer_dict,
        "candidate": {
            **actual_expected,
            "reject_before_0d": ["non_finite", "out_of_bounds", "duplicate"],
            "candidate_validation": metrics_config["candidate_validation"],
        },
        "acceptance": {
            "thrust_relative_tolerance": derived.thrust_relative_tolerance,
            "tsfc_relative_tolerance": derived.tsfc_relative_tolerance,
            "mass_relative_tolerance": derived.mass_relative_tolerance,
            "total_error_tolerance": derived.total_error_tolerance,
            "strict_actual_0d_improvement": True,
            "derivation": derived.derivation,
        },
        "gradient": copy.deepcopy(gradient),
        "nsga2": {
            **copy.deepcopy(nsga2),
            "maximum_0d_evaluations": maximum_nsga,
        },
        "cfm_checkpoint": {
            "format_version": cfm["format_version"],
            "generator_name": cfm["generator_name"],
            "checkpoint_state": cfm["checkpoint_state"],
            "model_config": cfm["model_config"],
            "runtime_config": cfm["runtime_config"],
            "training_metadata": cfm_training,
        },
        "gradient_surrogate": {
            "checkpoint_version": surrogate["checkpoint_version"],
            "model_config": surrogate["model_config"],
            "training_config": surrogate["training_config"],
            "metrics": surrogate_metrics,
            "dataset_sha256": surrogate["dataset_sha256"],
        },
    }


def _sanitize_environment_text(value: str) -> str:
    value = re.sub(r"://[^/@\s]+@", "://<redacted>@", value)
    value = re.sub(r"/t/[^/\s]+/", "/t/<redacted>/", value)
    return re.sub(
        r"(?i)(token|access_token|auth|api_key)=([^&\s]+)",
        r"\1=<redacted>",
        value,
    )


def _capture_environment(
    output_dir: Path,
    repo_root: Path,
    runtime: Mapping[str, Any],
) -> None:
    environment_dir = output_dir / "environment"
    environment_dir.mkdir(parents=True)
    write_json(environment_dir / "python_torch.json", collect_environment())
    write_json(
        environment_dir / "runtime.json",
        {
            **runtime,
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_version": torch.backends.cudnn.version(),
            "environment_variables": {
                name: os.environ.get(name)
                for name in (
                    "CUDA_VISIBLE_DEVICES",
                    "CUDA_DEVICE_ORDER",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "CUBLAS_WORKSPACE_CONFIG",
                    "PYTHONHASHSEED",
                )
            },
        },
    )
    commands = {
        "pip_freeze.txt": [sys.executable, "-m", "pip", "freeze"],
        "os_release.txt": ["cat", "/etc/os-release"],
        "uname.txt": ["uname", "-a"],
        "lscpu.txt": ["lscpu"],
        "nvidia_smi.txt": ["nvidia-smi", "-q"],
        "nvidia_smi_gpu6.txt": [
            "nvidia-smi",
            "-i",
            str(runtime["physical_gpu"]),
            "-q",
        ],
    }
    conda_executable = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda_executable:
        commands["conda_explicit.txt"] = [conda_executable, "list", "--explicit"]
        commands["conda_environment.yaml"] = [conda_executable, "env", "export"]
    command_results = {}
    for filename, args in commands.items():
        result = _run_command(args, repo_root)
        if result["returncode"] != 0:
            raise RuntimeError(
                f"Environment capture failed for {filename}: {result['stderr']}"
            )
        content = _sanitize_environment_text(result["stdout"])
        if result["stderr"]:
            content += "\n# stderr\n" + _sanitize_environment_text(result["stderr"])
        (environment_dir / filename).write_text(content.rstrip() + "\n", encoding="utf-8")
        command_results[filename] = {
            "args": result["args"],
            "returncode": result["returncode"],
        }
    write_json(environment_dir / "commands.json", command_results)


def _case_metrics(case_dir: Path) -> Dict[str, Any]:
    generation = json.loads(
        (case_dir / "generation" / "metrics.json").read_text(encoding="utf-8")
    )
    direct = json.loads(
        (case_dir / "direct" / "metrics.json").read_text(encoding="utf-8")
    )
    decision = json.loads(
        (case_dir / "delivery" / "decision.json").read_text(encoding="utf-8")
    )
    delivered = pd.read_csv(case_dir / "delivery" / "delivered_candidates.csv")
    result = {
        "generation": {
            key: generation[key]
            for key in ("N_raw", "N_valid", "N_keep", "N_trimmed")
        },
        "direct_best": direct["best"],
        "direct_hit_count": decision["direct_hit_count"],
        "decision": decision,
        "delivered_best_total_error": (
            float(delivered["total_error"].min()) if len(delivered) else None
        ),
    }
    gradient_path = case_dir / "gradient" / "diagnostics.json"
    if gradient_path.is_file():
        gradient = json.loads(gradient_path.read_text(encoding="utf-8"))
        gradient.pop("history", None)
        result["gradient"] = gradient
    return result


def _compare_case(
    case_dir: Path,
    reference_dir: Path,
    relative_paths: Sequence[str],
) -> list[Dict[str, Any]]:
    records = []
    for relative in relative_paths:
        actual = case_dir / relative
        expected = reference_dir / relative
        if not actual.is_file() or not expected.is_file():
            raise FileNotFoundError(f"Missing baseline comparison artifact: {relative}")
        actual_hash = sha256_file(actual)
        expected_hash = sha256_file(expected)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Baseline parity failed for {relative}: "
                f"new={actual_hash}, legacy={expected_hash}"
            )
        records.append(
            {
                "path": relative,
                "sha256": actual_hash,
                "byte_identical": True,
            }
        )
    return records


def _generalization_metrics(repo_root: Path, config: Mapping[str, Any]) -> Dict[str, Any]:
    panel = _resolve(
        repo_root, config["reference_directories"]["generalization_panel"]
    )
    frame = pd.read_csv(panel / "target_summary.csv")
    groups = {}
    for target_type, group in frame.groupby("target_type", sort=True):
        groups[str(target_type)] = {
            "target_count": len(group),
            "mean_valid_rate": float(group["valid_rate"].mean()),
            "mean_hit_rate": float(group["hit_rate"].mean()),
            "median_best_total_error": float(group["best_total_error"].median()),
        }
    return {
        "target_count": len(frame),
        "target_type_metrics": groups,
        "all_mean_valid_rate": float(frame["valid_rate"].mean()),
        "all_mean_hit_rate": float(frame["hit_rate"].mean()),
        "all_median_best_total_error": float(frame["best_total_error"].median()),
    }


def _write_artifact_manifest(output_dir: Path) -> Path:
    excluded = {"artifact_manifest.csv", "manifest.yaml"}
    records = []
    for path in sorted(output_dir.rglob("*")):
        relative = str(path.relative_to(output_dir))
        if not path.is_file() or relative in excluded:
            continue
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest_path = output_dir / "artifact_manifest.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    return manifest_path


def verify_inverse_design_baseline(output_dir: Path) -> Dict[str, Any]:
    output = Path(output_dir).resolve()
    manifest = _load_yaml(output / "manifest.yaml")
    if manifest.get("status") != "complete":
        raise ValueError(f"Baseline is not complete: {output}")
    artifact_manifest = output / "artifact_manifest.csv"
    expected_manifest_hash = str(manifest["artifact_manifest_sha256"])
    if sha256_file(artifact_manifest) != expected_manifest_hash:
        raise ValueError("Baseline artifact manifest hash mismatch")
    frame = pd.read_csv(artifact_manifest)
    for row in frame.itertuples(index=False):
        path = output / str(row.path)
        if not path.is_file():
            raise FileNotFoundError(f"Frozen baseline artifact is missing: {path}")
        if path.stat().st_size != int(row.size_bytes) or sha256_file(path) != str(
            row.sha256
        ):
            raise ValueError(f"Frozen baseline artifact changed: {path}")
    return {
        "baseline_id": manifest["baseline_id"],
        "status": "verified",
        "artifact_count": len(frame),
        "source_commit": manifest["source_commit"],
    }


def run_inverse_design_baseline(
    config_path: Path,
    *,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
    require_clean_git: bool = True,
) -> Path:
    root = (repo_root or Path.cwd()).resolve()
    config_path = _resolve(root, config_path)
    config = load_inverse_design_baseline_config(config_path)
    output_dir = (
        _resolve(root, output_dir_override)
        if output_dir_override is not None
        else _resolve(root, config["output_directory"])
    )
    if output_dir.exists():
        raise FileExistsError(f"Inverse-design baseline already exists: {output_dir}")
    git_state = collect_clean_git_state(root, require_clean=require_clean_git)
    input_records = verify_critical_artifacts(config, root)
    verify_reference_directories(config, root)
    runtime = _validate_runtime(config)

    output_dir.mkdir(parents=True, exist_ok=False)
    started = dt.datetime.now(dt.timezone.utc)
    write_yaml(
        output_dir / "manifest.yaml",
        {
            "baseline_version": config["baseline_version"],
            "baseline_id": config["baseline_id"],
            "status": "running",
            "started_utc": started.isoformat(),
            "source_commit": git_state["commit"],
        },
    )
    try:
        configuration_dir = output_dir / "configuration"
        configuration_dir.mkdir()
        shutil.copy2(config_path, configuration_dir / "baseline.yaml")
        _snapshot_inputs(input_records, output_dir, root)
        pd.DataFrame(input_records).to_csv(
            output_dir / "input_manifest.csv", index=False
        )
        _snapshot_reference_directories(config, output_dir, root)
        write_yaml(output_dir / "contract.yaml", _checkpoint_contract(config, root))
        write_json(output_dir / "git_state.json", git_state)
        write_json(
            output_dir / "source_manifest.json", collect_source_manifest(root)
        )
        _capture_environment(output_dir, root, runtime)

        delivery_template = _load_yaml(_resolve(root, config["delivery_config"]))
        case_metrics = {}
        parity = {}
        for name, case in config["cases"].items():
            delivery = copy.deepcopy(delivery_template)
            delivery["generation"]["seed"] = int(case["generation_seed"])
            delivery["runtime"]["device"] = (
                "cuda" if bool(config["runtime"].get("require_gpu", True)) else "cpu"
            )
            with tempfile.TemporaryDirectory(
                prefix=f"novaturbo_baseline_{name}_"
            ) as temporary:
                delivery_path = Path(temporary) / "delivery.yaml"
                delivery_path.write_text(
                    yaml.safe_dump(delivery, sort_keys=False), encoding="utf-8"
                )
                case_dir = run_inverse_design_delivery(
                    delivery_path,
                    _resolve(root, case["target_config"]),
                    repo_root=root,
                    output_dir_override=output_dir / "cases" / name,
                )
            case_metrics[name] = _case_metrics(case_dir)
            parity[name] = _compare_case(
                case_dir,
                _resolve(root, case["legacy_reference"]),
                [str(value) for value in case["compare_artifacts"]],
            )

        legacy_freeze = _load_yaml(
            _resolve(root, "config/stage2/s2_wp5_cfm_default_freeze_v2.yaml")
        )
        baseline_metrics = {
            "metrics_version": "active10d_cfm4_baseline_metrics_v1",
            "cases": case_metrics,
            "generalization_panel_30_targets": _generalization_metrics(root, config),
            "model_selection_39_targets": legacy_freeze["model_selection"],
        }
        write_yaml(output_dir / "baseline_metrics.yaml", baseline_metrics)
        write_json(output_dir / "parity.json", parity)
        artifact_manifest = _write_artifact_manifest(output_dir)
        final_manifest = {
            "baseline_version": config["baseline_version"],
            "baseline_id": config["baseline_id"],
            "status": "complete",
            "started_utc": started.isoformat(),
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_commit": git_state["commit"],
            "source_tree": git_state["tree"],
            "source_branch": git_state["branch"],
            "source_dirty": git_state["dirty"],
            "physical_gpu": runtime["physical_gpu"],
            "logical_device": runtime["logical_device"],
            "critical_artifact_count": len(input_records),
            "artifact_manifest": "artifact_manifest.csv",
            "artifact_manifest_sha256": sha256_file(artifact_manifest),
            "artifact_count": len(pd.read_csv(artifact_manifest)),
            "parity_gate": "passed",
            "cases": {
                name: {
                    "target_config": str(case["target_config"]),
                    "generation_seed": int(case["generation_seed"]),
                    "output": f"cases/{name}",
                    "parity_artifact_count": len(parity[name]),
                }
                for name, case in config["cases"].items()
            },
        }
        write_yaml(output_dir / "manifest.yaml", final_manifest)
        verify_inverse_design_baseline(output_dir)
        return output_dir
    except Exception as error:
        write_yaml(
            output_dir / "manifest.yaml",
            {
                "baseline_version": config["baseline_version"],
                "baseline_id": config["baseline_id"],
                "status": "failed",
                "started_utc": started.isoformat(),
                "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "source_commit": git_state["commit"],
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
