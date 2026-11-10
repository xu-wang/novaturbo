"""Complete reproducible inverse-design orchestration for WP6.

This module connects the frozen WP3 generators, WP4 physics evaluator, WP5
named Warm-Start adapter, and the original NSGA-II optimizer behind one CLI.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
from .candidate_protocol import (
    V21_RAW_CANDIDATE_COUNT,
    apply_candidate_protocol_v21,
)
from .dataset import DesignSpaceBounds
from .generator_dataset import GeneratorNormalizer, design_names, load_schema
from .generators import load_generator
from .optimizer import (
    TARGET_WEIGHTED_RELATIVE_ERROR,
    InitialPopulation,
    OptimizationConfig,
    optimize,
    optimizer_parameter_space,
)
from .preference import (
    evaluate_preference_pool,
    objective_scaling,
    select_reference_directions,
)
from .round_trip import (
    TARGET_OUT_OF_DOMAIN,
    RoundTripEvaluator,
    derive_round_trip_metrics,
    load_metrics_config,
    sampling_budget_exhausted_result,
    target_out_of_domain_result,
)
from .warm_start import build_background_population, build_warm_start_population


PIPELINE_STAGES = (
    "validated",
    "generation_round_trip",
    "population",
    "optimization",
    "summary",
)
TARGET_TYPES = {"ID", "Interpolation", "OOD-feasible-unknown", "Infeasible"}
INITIALIZATION_MODES = {"random", "screened_random", "generator", "hybrid"}
MODEL_INITIALIZATION_MODES = {"generator", "hybrid"}


class PipelineInterrupted(RuntimeError):
    """Raised after an explicitly requested stage boundary interruption."""


@dataclass(frozen=True)
class TargetSpec:
    target_version: str
    target_id: str
    thrust_N: float
    tsfc_kg_N_s: float
    target_type: str
    expected_status: str
    tolerance_profile: str
    total_mass_kg: Optional[float] = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TargetSpec":
        required = {
            "target_version",
            "target_id",
            "thrust_N",
            "tsfc_kg_N_s",
            "target_type",
            "expected_status",
            "tolerance_profile",
        }
        missing = required.difference(value)
        if missing:
            raise ValueError(f"TargetSpec is missing keys: {sorted(missing)}")
        target = cls(
            target_version=str(value["target_version"]),
            target_id=str(value["target_id"]),
            thrust_N=float(value["thrust_N"]),
            tsfc_kg_N_s=float(value["tsfc_kg_N_s"]),
            target_type=str(value["target_type"]),
            expected_status=str(value["expected_status"]),
            tolerance_profile=str(value["tolerance_profile"]),
            total_mass_kg=(
                float(value["total_mass_kg"])
                if value.get("total_mass_kg") is not None
                else None
            ),
        )
        if not target.target_id:
            raise ValueError("TargetSpec.target_id cannot be empty")
        if target.target_type not in TARGET_TYPES:
            raise ValueError(f"Unsupported target_type: {target.target_type}")
        if not math.isfinite(target.thrust_N) or not math.isfinite(target.tsfc_kg_N_s):
            raise ValueError("TargetSpec conditions must be finite")
        if target.total_mass_kg is not None and (
            not math.isfinite(target.total_mass_kg) or target.total_mass_kg <= 0.0
        ):
            raise ValueError("TargetSpec.total_mass_kg must be positive and finite")
        return target

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def load_target_spec(path: Path) -> TargetSpec:
    return TargetSpec.from_mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def load_inverse_design_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "version",
        "run_id",
        "mode",
        "model",
        "data",
        "seeds",
        "generation",
        "selection",
        "population",
        "optimizer",
        "runtime",
        "output",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Inverse-design RunConfig is missing keys: {sorted(missing)}")
    if not str(config["run_id"]):
        raise ValueError("RunConfig.run_id cannot be empty")
    mode = str(config["mode"])
    if mode not in INITIALIZATION_MODES:
        raise ValueError(f"RunConfig.mode must be one of {sorted(INITIALIZATION_MODES)}")
    model_id = config["model"].get("model_id")
    checkpoint = config["model"].get("checkpoint_path")
    if mode in {"random", "screened_random"}:
        if model_id not in {None, "none", "random"} or checkpoint not in {None, ""}:
            raise ValueError(f"{mode} mode must not configure a generator checkpoint")
    elif not model_id or not checkpoint:
        raise ValueError(f"{mode} mode requires model_id and checkpoint_path")
    population_size = int(config["optimizer"]["population_size"])
    if population_size <= 0:
        raise ValueError("optimizer.population_size must be positive")
    ratio = float(config["population"]["generator_ratio"])
    expected_ratio = 0.0 if mode == "random" else 1.0 if mode == "generator" else ratio
    if not 0.0 <= expected_ratio <= 1.0:
        raise ValueError("population.generator_ratio must be between zero and one")
    if int(config["generation"]["n_generated"]) <= 0 and mode != "random":
        raise ValueError("generation.n_generated must be positive for generative modes")
    candidate_protocol = str(
        config["generation"].get("candidate_protocol", "legacy_direct")
    )
    if candidate_protocol not in {"legacy_direct", "v2_1_fixed_raw"}:
        raise ValueError(
            "generation.candidate_protocol must be legacy_direct or v2_1_fixed_raw"
        )
    if candidate_protocol == "v2_1_fixed_raw" and mode != "random":
        if int(config["generation"]["n_generated"]) != V21_RAW_CANDIDATE_COUNT:
            raise ValueError("V2.1 inverse design requires generation.n_generated=128")
    if str(config["selection"]["strategy"]) not in {
        "physics_error_diverse",
        "physics_error_topk",
    }:
        raise ValueError(
            "selection.strategy must be physics_error_diverse or physics_error_topk"
        )
    preference = config["population"].get("preference", {})
    if bool(preference.get("enabled", False)):
        required_preference = {"pool_size", "reference_dataset", "directions"}
        missing_preference = required_preference.difference(preference)
        if missing_preference:
            raise ValueError(
                "population.preference is missing keys: "
                f"{sorted(missing_preference)}"
            )
        if int(preference["pool_size"]) <= 0:
            raise ValueError("population.preference.pool_size must be positive")
        directions = np.asarray(preference["directions"], dtype=float)
        objective_count = len(config["optimizer"]["objectives"])
        if directions.ndim != 2 or directions.shape[1] != objective_count:
            raise ValueError(
                "population.preference.directions must match optimizer objectives"
            )
        if not np.isfinite(directions).all() or np.any(directions <= 0.0):
            raise ValueError(
                "population.preference.directions must be finite and positive"
            )
    return config


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for inverse design but is unavailable")
    return requested


def _stage_template() -> Dict[str, Dict[str, Any]]:
    return {
        stage: {"status": "pending", "completed_utc": None, "artifacts": []}
        for stage in PIPELINE_STAGES
    }


def _new_manifest(
    target: TargetSpec,
    config: Mapping[str, Any],
    target_path: Path,
    run_config_path: Path,
) -> Dict[str, Any]:
    return {
        "format_version": "inverse_design_manifest_v1",
        "wp": "WP6",
        "run_id": str(config["run_id"]),
        "target_id": target.target_id,
        "status": "running",
        "current_stage": "validated",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "finished_utc": None,
        "inputs": {
            "target_config": str(target_path),
            "target_config_sha256": sha256_file(target_path),
            "run_config": str(run_config_path),
            "run_config_sha256": sha256_file(run_config_path),
        },
        "stages": _stage_template(),
        "error": None,
    }


def _validate_resume_manifest(
    manifest: Mapping[str, Any],
    target: TargetSpec,
    config: Mapping[str, Any],
    target_path: Path,
    run_config_path: Path,
) -> None:
    if manifest.get("format_version") != "inverse_design_manifest_v1":
        raise ValueError("Cannot resume an unsupported inverse-design manifest")
    expected = {
        "run_id": str(config["run_id"]),
        "target_id": target.target_id,
        "target_config_sha256": sha256_file(target_path),
        "run_config_sha256": sha256_file(run_config_path),
    }
    actual = {
        "run_id": manifest.get("run_id"),
        "target_id": manifest.get("target_id"),
        "target_config_sha256": manifest.get("inputs", {}).get("target_config_sha256"),
        "run_config_sha256": manifest.get("inputs", {}).get("run_config_sha256"),
    }
    if actual != expected:
        raise ValueError(
            f"Resume input mismatch; expected={expected}, existing_manifest={actual}"
        )


def _write_manifest(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    write_yaml(run_dir / "manifest.yaml", manifest)


def _complete_stage(
    run_dir: Path,
    manifest: Dict[str, Any],
    stage: str,
    artifacts: Sequence[str],
    elapsed_seconds: float,
) -> None:
    for relative in artifacts:
        if not (run_dir / relative).is_file():
            raise RuntimeError(f"Stage {stage} did not create required artifact: {relative}")
    manifest["stages"][stage] = {
        "status": "completed",
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": float(elapsed_seconds),
        "artifacts": list(artifacts),
    }
    current_index = PIPELINE_STAGES.index(stage)
    manifest["current_stage"] = (
        PIPELINE_STAGES[current_index + 1]
        if current_index + 1 < len(PIPELINE_STAGES)
        else None
    )
    manifest["status"] = "running"
    manifest["error"] = None
    _write_manifest(run_dir, manifest)


def _stage_completed(run_dir: Path, manifest: Mapping[str, Any], stage: str) -> bool:
    value = manifest["stages"][stage]
    if value.get("status") != "completed":
        return False
    missing = [
        relative for relative in value.get("artifacts", []) if not (run_dir / relative).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Manifest marks stage {stage} complete but artifacts are missing: {missing}"
        )
    return True


def _interrupt_if_requested(
    run_dir: Path,
    manifest: Dict[str, Any],
    completed_stage: str,
    stop_after_stage: Optional[str],
) -> None:
    if stop_after_stage != completed_stage:
        return
    manifest["status"] = "interrupted"
    manifest["error"] = {"type": "PipelineInterrupted", "stage": completed_stage}
    _write_manifest(run_dir, manifest)
    raise PipelineInterrupted(f"Pipeline stopped after stage {completed_stage}")


def _raw_candidates_frame(
    normalized: np.ndarray,
    physical: np.ndarray,
    field_names: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for candidate_id, (normalized_row, physical_row) in enumerate(
        zip(normalized, physical)
    ):
        row: Dict[str, Any] = {"candidate_id": candidate_id}
        row.update(
            {
                f"normalized_{name}": normalized_row[index]
                for index, name in enumerate(field_names)
            }
        )
        row.update(
            {name: physical_row[index] for index, name in enumerate(field_names)}
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _read_ranked_candidates(path: Path, field_names: Sequence[str]) -> pd.DataFrame:
    """Read ranked candidates while preserving a legitimate zero-candidate run."""

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=list(field_names))


def _pareto_frame(pareto, param_names: Sequence[str]) -> pd.DataFrame:
    rows = []
    for pareto_index, individual in enumerate(pareto):
        row: Dict[str, Any] = {
            "pareto_index": pareto_index,
            "individual_id": individual.individual_id,
            "provenance": json.dumps(individual.provenance, sort_keys=True),
            "is_feasible": bool(individual.is_feasible),
            "rank": int(individual.rank),
            "crowding_distance": float(individual.crowding_distance),
        }
        row.update(
            {
                f"gene__{name}": float(individual.genes[index])
                for index, name in enumerate(param_names)
            }
        )
        row.update(
            {
                f"objective_internal_{index}": float(value)
                for index, value in enumerate(individual.objectives)
            }
        )
        row.update(
            {
                f"performance__{name}": value
                for name, value in individual.performance.items()
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _optimizer_history_frame(
    history: Mapping[str, Any], config: OptimizationConfig
) -> pd.DataFrame:
    rows = []
    for index, (pareto_size, feasible_count) in enumerate(
        zip(history["pareto_size"], history["feasible_count"])
    ):
        row: Dict[str, Any] = {
            "generation": index + 1,
            "evaluation_count": (index + 2) * config.population_size,
            "pareto_size": pareto_size,
            "feasible_count": feasible_count,
        }
        if index < len(history["best_objectives"]):
            row.update(
                {
                    f"mean_internal_objective_{objective_index}": value
                    for objective_index, value in enumerate(
                        history["best_objectives"][index]
                    )
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _evaluation_ledger_frame(history: Mapping[str, Any]) -> pd.DataFrame:
    records = []
    for value in history["evaluation_ledger"]:
        record = dict(value)
        record["parents"] = json.dumps(record.get("parents", []))
        records.append(record)
    return pd.DataFrame(records)


def _optimizer_config(
    config: Mapping[str, Any],
    target: Optional[TargetSpec] = None,
    derived_metrics: Optional[Any] = None,
) -> OptimizationConfig:
    values = dict(config["optimizer"])
    values.pop("seed", None)
    if TARGET_WEIGHTED_RELATIVE_ERROR in values.get("objectives", []):
        if target is None or derived_metrics is None:
            raise ValueError(
                "Target-directed optimization requires TargetSpec and derived metrics"
            )
        target_values = {
            "thrust_N": float(target.thrust_N),
            "tsfc_kg_N_s": float(target.tsfc_kg_N_s),
        }
        target_weights = {
            "thrust_N": float(derived_metrics.thrust_weight),
            "tsfc_kg_N_s": float(derived_metrics.tsfc_weight),
        }
        target_epsilons = {
            "thrust_N": float(derived_metrics.thrust_epsilon),
            "tsfc_kg_N_s": float(derived_metrics.tsfc_epsilon),
        }
        if target.total_mass_kg is not None:
            target_values["total_mass_kg"] = float(target.total_mass_kg)
            target_weights["total_mass_kg"] = float(derived_metrics.mass_weight)
            target_epsilons["total_mass_kg"] = float(derived_metrics.mass_epsilon)
        values["target_values"] = target_values
        values["target_weights"] = target_weights
        values["target_epsilons"] = target_epsilons
    values["warm_start_seed"] = int(config["seeds"]["background"])
    values["warm_start_fill_method"] = str(config["population"]["background_method"])
    values["warm_start_duplicate_tolerance"] = float(
        config["population"]["duplicate_tolerance"]
    )
    values["warm_start_align_legacy_rng"] = True
    return OptimizationConfig(**values)


def _load_initial_population(run_dir: Path) -> InitialPopulation:
    genes_frame = pd.read_csv(run_dir / "population" / "initial_genes.csv")
    genes = genes_frame.drop(columns=["population_index"]).to_numpy(dtype=float)
    provenance = pd.read_csv(
        run_dir / "population" / "population_provenance.csv"
    ).to_dict(orient="records")
    metadata = json.loads(
        (run_dir / "population" / "adapter_metrics.json").read_text(encoding="utf-8")
    )
    return InitialPopulation(
        genes=genes,
        provenance=provenance,
        metadata=metadata,
        apply_integer_projection=False,
    )


def _build_preference_population(
    selected: pd.DataFrame,
    generated_field_names: Sequence[str],
    *,
    mode: str,
    config: Mapping[str, Any],
    optimizer_config: OptimizationConfig,
    background: np.ndarray,
    background_seed: int,
    reference: pd.DataFrame,
) -> Tuple[InitialPopulation, pd.DataFrame, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Select injected rows from an equally evaluated complete-19D pool."""

    preference = config["population"]["preference"]
    pool_size = int(preference["pool_size"])
    ratio = (
        1.0
        if mode == "generator"
        else float(config["population"]["generator_ratio"])
    )
    requested_count = int(round(optimizer_config.population_size * ratio))
    if pool_size < requested_count:
        raise ValueError(
            "population.preference.pool_size must be at least the requested "
            "injected population count"
        )
    source = "screened_random" if mode == "screened_random" else "generator"
    pool_seed = background_seed + int(preference.get("background_seed_offset", 1000003))
    pool_genes, param_names = build_background_population(
        DesignSpaceBounds(),
        pool_size,
        pool_seed,
        str(config["population"]["background_method"]),
    )
    unknown = set(generated_field_names).difference(param_names)
    if unknown:
        raise ValueError(f"Preference fields are absent from 19D schema: {sorted(unknown)}")
    candidate_values = (
        selected[list(generated_field_names)].to_numpy(dtype=float)
        if len(selected)
        else np.empty((0, len(generated_field_names)), dtype=float)
    )
    if not np.isfinite(candidate_values).all():
        raise ValueError("Preference candidates contain NaN/inf")
    generated_indices = [param_names.index(name) for name in generated_field_names]
    _, lower, upper = optimizer_parameter_space(DesignSpaceBounds())
    if len(candidate_values) and (
        np.any(candidate_values < lower[generated_indices])
        or np.any(candidate_values > upper[generated_indices])
    ):
        raise ValueError("Preference candidates are outside DesignSpaceBounds")
    pool_provenance_rows: List[Dict[str, Any]] = []
    for pool_index in range(pool_size):
        if len(candidate_values):
            source_row = pool_index % len(candidate_values)
            pool_genes[pool_index, generated_indices] = candidate_values[source_row]
            row_source = source
            generated_fields = list(generated_field_names)
        else:
            source_row = None
            row_source = "random_fill"
            generated_fields = []
        pool_provenance_rows.append(
            {
                "population_index": pool_index,
                "source": row_source,
                "mode": mode,
                "background_row": pool_index,
                "background_seed": pool_seed,
                "background_method": str(config["population"]["background_method"]),
                "generated_fields": json.dumps(generated_fields),
                "generated_source_row": source_row,
            }
        )
    pool_provenance = pd.DataFrame(pool_provenance_rows).set_index(
        "population_index"
    )
    evaluation, objectives, feasible = evaluate_preference_pool(
        pool_genes,
        param_names,
        optimizer_config.objectives,
        optimizer_config.maximize,
    )
    objective_min, objective_max = objective_scaling(
        reference,
        optimizer_config.objectives,
        optimizer_config.maximize,
    )
    selected_indices, selection = select_reference_directions(
        objectives,
        feasible,
        requested_count,
        preference["directions"],
        objective_min,
        objective_max,
    )
    output = np.asarray(background, dtype=float).copy()
    selection_lookup = selection.set_index("pool_index") if len(selection) else None
    provenance_rows: List[Dict[str, Any]] = []
    for population_index in range(optimizer_config.population_size):
        if population_index < len(selected_indices):
            pool_index = int(selected_indices[population_index])
            output[population_index] = pool_genes[pool_index]
            record = dict(pool_provenance.loc[pool_index])
            record.update(
                {
                    "population_index": population_index,
                    "mode": mode,
                    "preference_selected": True,
                    "preference_pool_index": pool_index,
                    "preference_selection_rank": population_index + 1,
                    "preference_direction_index": int(
                        selection_lookup.loc[pool_index, "direction_index"]
                    ),
                    "preference_asf_score": float(
                        selection_lookup.loc[pool_index, "asf_score"]
                    ),
                }
            )
        else:
            record = {
                "population_index": population_index,
                "source": "random_fill"
                if population_index < requested_count
                else "random",
                "mode": mode,
                "background_row": population_index,
                "background_seed": background_seed,
                "background_method": str(config["population"]["background_method"]),
                "generated_fields": "[]",
                "preference_selected": False,
                "preference_pool_index": np.nan,
                "preference_selection_rank": np.nan,
                "preference_direction_index": np.nan,
                "preference_asf_score": np.nan,
            }
        provenance_rows.append(record)
    provenance = pd.DataFrame(provenance_rows)
    evaluation["selected_for_population"] = evaluation["pool_index"].isin(
        selected_indices
    )
    selected_source_count = int(
        provenance.loc[: max(len(selected_indices) - 1, -1), "source"]
        .eq(source)
        .sum()
    ) if selected_indices else 0
    digest = hashlib.sha256(output.astype("<f8", copy=False).tobytes()).hexdigest()
    metadata = {
        "mode": mode,
        "population_size": optimizer_config.population_size,
        "generator_ratio": ratio,
        "requested_generated_count": requested_count,
        "selected_generated_count": selected_source_count,
        "random_fill_count": requested_count - selected_source_count,
        "planned_random_count": optimizer_config.population_size - requested_count,
        "candidate_source": source,
        "background_seed": background_seed,
        "background_method": str(config["population"]["background_method"]),
        "population_sha256_float64": digest,
        "preference": {
            "enabled": True,
            "pool_size": pool_size,
            "pool_background_seed": pool_seed,
            "source_candidate_count": int(len(candidate_values)),
            "source_candidates_reused": bool(
                len(candidate_values) > 0 and len(candidate_values) < pool_size
            ),
            "evaluation_count": int(len(evaluation)),
            "feasible_count": int(feasible.sum()),
            "requested_selection_count": requested_count,
            "selected_count": int(len(selected_indices)),
            "directions": preference["directions"],
            "objective_min_internal": objective_min.tolist(),
            "objective_max_internal": objective_max.tolist(),
        },
    }
    population = InitialPopulation(
        genes=output,
        provenance=provenance.to_dict(orient="records"),
        metadata=metadata,
        apply_integer_projection=False,
    )
    return population, provenance, metadata, evaluation, selection


def _report_text(target: TargetSpec, config: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# WP6 完整逆设计运行报告",
            "",
            f"- Run ID: {config['run_id']}",
            f"- Target ID: {target.target_id}",
            f"- Mode: {config['mode']}",
            f"- Model: {config['model'].get('model_id')}",
            f"- Status: {metrics['status']}",
            f"- Generator candidates: {metrics['generation']['requested_count']}",
            f"- Selected generator candidates: {metrics['population']['generated_count']}",
            f"- Random fill: {metrics['population']['random_fill_count']}",
            f"- Prefilter 0D evaluations: {metrics['cost']['prefilter_0d_evaluations']}",
            f"- Preference 19D evaluations: {metrics['cost']['preference_19d_evaluations']}",
            f"- NSGA-II 0D evaluations: {metrics['cost']['nsga2_0d_evaluations']}",
            f"- End-to-end 0D evaluations: {metrics['cost']['total_0d_evaluations']}",
            f"- Pareto size: {metrics['optimization']['pareto_size']}",
            "",
            "This is a pipeline integration result. Generator superiority over Random "
            "is not required by WP6/G2.",
            "",
        ]
    )


def run_inverse_design(
    target_config_path: Path,
    run_config_path: Path,
    *,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
    resume: bool = False,
    stop_after_stage: Optional[str] = None,
) -> Path:
    """Execute or safely resume one complete inverse-design run."""

    repo_root = (repo_root or Path.cwd()).resolve()
    target_config_path = Path(target_config_path).resolve()
    run_config_path = Path(run_config_path).resolve()
    target = load_target_spec(target_config_path)
    config = load_inverse_design_config(run_config_path)
    if stop_after_stage is not None and stop_after_stage not in PIPELINE_STAGES:
        raise ValueError(f"stop_after_stage must be one of {PIPELINE_STAGES}")
    output_root = _resolve_path(repo_root, str(config["output"]["root"]))
    run_dir = (
        output_dir_override.resolve()
        if output_dir_override is not None
        else output_root / str(config["run_id"])
    )
    manifest_path = run_dir / "manifest.yaml"
    if run_dir.exists():
        if not resume:
            raise FileExistsError(f"Inverse-design run exists: {run_dir}")
        if not manifest_path.is_file():
            raise ValueError("Cannot resume a directory without manifest.yaml")
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        _validate_resume_manifest(
            manifest, target, config, target_config_path, run_config_path
        )
        if manifest.get("status") == "completed":
            for stage in PIPELINE_STAGES:
                if not _stage_completed(run_dir, manifest, stage):
                    raise RuntimeError(
                        f"Completed manifest contains an incomplete stage: {stage}"
                    )
            return run_dir
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        for name in ("generator", "round_trip", "population", "optimization", "summary"):
            (run_dir / name).mkdir()
        shutil.copy2(target_config_path, run_dir / "target.yaml")
        shutil.copy2(run_config_path, run_dir / "run_config.yaml")
        write_json(run_dir / "target.json", target.to_dict())
        manifest = _new_manifest(
            target, config, target_config_path, run_config_path
        )
        _write_manifest(run_dir, manifest)

    pipeline_started = time.perf_counter()
    try:
        validation_started = time.perf_counter()
        data_config = config["data"]
        dataset_path = _resolve_path(repo_root, data_config["dataset_csv"])
        schema_path = _resolve_path(repo_root, data_config["schema"])
        normalizer_path = _resolve_path(repo_root, data_config["normalizer"])
        metrics_path = _resolve_path(repo_root, data_config["metrics_config"])
        preference_config = config["population"].get("preference", {})
        preference_enabled = bool(preference_config.get("enabled", False)) and str(
            config["mode"]
        ) != "random"
        preference_reference_path = (
            _resolve_path(repo_root, str(preference_config["reference_dataset"]))
            if preference_enabled
            else None
        )
        required_paths = [dataset_path, schema_path, normalizer_path, metrics_path]
        if preference_reference_path is not None:
            required_paths.append(preference_reference_path)
        for path in required_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Required inverse-design input is missing: {path}")
        schema = load_schema(schema_path)
        normalizer = GeneratorNormalizer.load(normalizer_path)
        metrics_config = load_metrics_config(metrics_path)
        dataset_versions = set(
            pd.read_csv(dataset_path, usecols=["dataset_version"])["dataset_version"]
            .dropna()
            .astype(str)
        )
        if dataset_versions != {str(data_config["dataset_version"])}:
            raise ValueError(
                f"Dataset version mismatch: actual={sorted(dataset_versions)}, "
                f"expected={data_config['dataset_version']}"
            )
        if sha256_file(normalizer_path) != str(data_config["normalizer_sha256"]):
            raise ValueError("Normalizer SHA256 differs from frozen RunConfig")
        if normalizer.schema_version != str(schema["schema_version"]):
            raise ValueError("Schema and normalizer versions are incompatible")
        actual_normalizer_version = (
            f"{normalizer.schema_version}_{normalizer.fitted_split}_"
            f"{normalizer.fitted_sample_count}"
        )
        if actual_normalizer_version != str(data_config["normalizer_version"]):
            raise ValueError(
                "Normalizer version mismatch: "
                f"actual={actual_normalizer_version}, "
                f"expected={data_config['normalizer_version']}"
            )
        if target.tolerance_profile != str(metrics_config["metrics_version"]):
            raise ValueError("Target tolerance_profile differs from metrics config")
        derived_metrics = derive_round_trip_metrics(
            dataset_path, normalizer, metrics_config
        )
        optimizer_config = _optimizer_config(config, target, derived_metrics)
        optimizer_seed = int(config["seeds"]["optimizer"])
        background_seed = int(config["seeds"]["background"])
        if background_seed != optimizer_seed:
            raise ValueError(
                "WP6 fairness requires seeds.background == seeds.optimizer"
            )
        device = _resolve_device(str(config["runtime"]["device"]))
        generator = None
        checkpoint_path = None
        if config["mode"] in MODEL_INITIALIZATION_MODES:
            checkpoint_path = _resolve_path(
                repo_root, str(config["model"]["checkpoint_path"])
            )
            generator = load_generator(checkpoint_path, device=device)
            if generator.generator_name != str(config["model"]["model_id"]):
                raise ValueError("Checkpoint generator_name differs from RunConfig")
            if generator.normalizer is None:
                raise ValueError("Generator checkpoint does not contain a normalizer")
            if generator.normalizer.to_dict() != normalizer.to_dict():
                raise ValueError("Checkpoint normalizer differs from RunConfig normalizer")
            if generator.schema_version != schema["schema_version"]:
                raise ValueError("Checkpoint schema version differs from RunConfig schema")
        validation_payload = {
            "dataset": str(dataset_path),
            "dataset_version": str(data_config["dataset_version"]),
            "dataset_sha256": sha256_file(dataset_path),
            "schema": str(schema_path),
            "schema_version": schema["schema_version"],
            "schema_sha256": sha256_file(schema_path),
            "normalizer": str(normalizer_path),
            "normalizer_version": actual_normalizer_version,
            "normalizer_sha256": sha256_file(normalizer_path),
            "normalizer_fitted_sample_id_sha256": normalizer.fitted_sample_id_sha256,
            "metrics_config": str(metrics_path),
            "metrics_version": metrics_config["metrics_version"],
            "metrics_sha256": sha256_file(metrics_path),
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path else None,
            "preference_reference_dataset": (
                str(preference_reference_path)
                if preference_reference_path is not None
                else None
            ),
            "preference_reference_sha256": (
                sha256_file(preference_reference_path)
                if preference_reference_path is not None
                else None
            ),
            "device": device,
        }
        write_json(run_dir / "validated_context.json", validation_payload)
        write_json(run_dir / "derived_metrics.json", derived_metrics.to_dict())
        if not _stage_completed(run_dir, manifest, "validated"):
            _complete_stage(
                run_dir,
                manifest,
                "validated",
                ["validated_context.json", "derived_metrics.json"],
                time.perf_counter() - validation_started,
            )
        _interrupt_if_requested(
            run_dir, manifest, "validated", stop_after_stage
        )

        selection_metrics_config = copy.deepcopy(metrics_config)
        selection_metrics_config["selection"]["selected_count"] = int(
            config["selection"]["selected_count"]
        )
        selection_metrics_config["selection"]["strategy"] = str(
            config["selection"]["strategy"]
        )
        evaluator = RoundTripEvaluator(
            schema, normalizer, selection_metrics_config, derived_metrics
        )
        generation_artifacts = [
            "generator/raw_candidates.csv",
            "generator/checked_candidates.csv",
            "generator/sampling_metrics.json",
            "round_trip/forward_results.csv",
            "round_trip/ranked_candidates.csv",
            "round_trip/failures.csv",
            "round_trip/metrics.json",
        ]
        if not _stage_completed(run_dir, manifest, "generation_round_trip"):
            generation_started = time.perf_counter()
            mode = str(config["mode"])
            if mode == "random":
                raw = pd.DataFrame(columns=["candidate_id", *design_names(schema)])
                checked = pd.DataFrame()
                forward = pd.DataFrame()
                ranked = pd.DataFrame()
                failures = pd.DataFrame(
                    columns=["candidate_id", "failure_code", "solver_warnings"]
                )
                round_trip_metrics = {
                    "skipped": True,
                    "reason": "Random mode has no generator prefilter",
                    "candidate_count": 0,
                    "candidate_valid_count": 0,
                    "forward_valid_count": 0,
                }
                sampling_metrics = {
                    "skipped": True,
                    "requested_count": 0,
                    "returned_count": 0,
                    "sampling_seconds": 0.0,
                    "round_trip_seconds": 0.0,
                    "ode_nfe": 0,
                    "prefilter_0d_evaluations": 0,
                    "diagnostics": {},
                }
            else:
                requested_count = int(config["generation"]["n_generated"])
                candidate_protocol = str(
                    config["generation"].get(
                        "candidate_protocol", "legacy_direct"
                    )
                )
                try:
                    normalized_target = evaluator.normalize_target(target.to_dict())
                except ValueError as exc:
                    if str(exc) != TARGET_OUT_OF_DOMAIN:
                        raise
                    raw = pd.DataFrame(columns=["candidate_id", *design_names(schema)])
                    rejected = target_out_of_domain_result(
                        target.to_dict(),
                        "screened_random"
                        if mode == "screened_random"
                        else str(config["model"]["model_id"]),
                    )
                    checked = rejected.checked_candidates
                    forward = rejected.forward_results
                    ranked = rejected.ranked_candidates
                    failures = rejected.failures
                    if checked.empty:
                        checked = pd.DataFrame(
                            columns=["candidate_id", "candidate_valid", "failure_code"]
                        )
                    if forward.empty:
                        forward = pd.DataFrame(
                            columns=["candidate_id", "forward_valid", "failure_code"]
                        )
                    if ranked.empty:
                        ranked = pd.DataFrame(
                            columns=[
                                "candidate_id",
                                *design_names(schema),
                                "forward_valid",
                                "selected_rank",
                            ]
                        )
                    round_trip_metrics = rejected.metrics
                    sampling_metrics = {
                        "skipped": True,
                        "reason": TARGET_OUT_OF_DOMAIN,
                        "requested_count": requested_count,
                        "returned_count": 0,
                        "sampling_seconds": 0.0,
                        "round_trip_seconds": 0.0,
                        "ode_nfe": 0,
                        "prefilter_0d_evaluations": 0,
                        "diagnostics": {},
                    }
                else:
                    sampling_started = time.perf_counter()
                    if mode == "screened_random":
                        random_generator = np.random.RandomState(
                            int(config["seeds"]["generator"])
                        )
                        normalized = random_generator.uniform(
                            -1.0,
                            1.0,
                            size=(requested_count, len(design_names(schema))),
                        )
                        sample_diagnostics = {
                            "sampler": "uniform_normalized_design_bounds",
                            "candidate_source": "screened_random",
                        }
                        ode_nfe = 0
                    else:
                        assert generator is not None
                        generated = generator.sample(
                            normalized_target,
                            requested_count,
                            seed=int(config["seeds"]["generator"]),
                            space="normalized",
                        )
                        normalized = (
                            generated.values.detach().cpu().numpy().astype(float)
                        )
                        sample_diagnostics = dict(generated.diagnostics)
                        ode_nfe = int(generated.diagnostics.get("nfe", 0))
                    sampling_seconds = time.perf_counter() - sampling_started
                    physical = normalizer.inverse_design(normalized)
                    raw = _raw_candidates_frame(
                        normalized, physical, design_names(schema)
                    )
                    protocol_validation = None
                    candidate_ids = np.arange(len(normalized), dtype=int)
                    if candidate_protocol == "v2_1_fixed_raw":
                        protocol_validation = evaluator.validate_candidates(
                            normalized,
                            physical,
                            candidate_ids=candidate_ids,
                        )
                        protocol = apply_candidate_protocol_v21(
                            protocol_validation["candidate_id"].to_numpy(dtype=int),
                            protocol_validation["candidate_valid"].to_numpy(dtype=bool),
                        )
                        protocol_validation["kept_by_v2_1"] = protocol_validation[
                            "candidate_id"
                        ].isin(protocol.kept_candidate_ids)
                        kept_positions = np.asarray(
                            protocol.kept_positions, dtype=int
                        )
                        normalized_for_0d = normalized[kept_positions]
                        physical_for_0d = physical[kept_positions]
                        candidate_ids_for_0d = np.asarray(
                            protocol.kept_candidate_ids, dtype=int
                        )
                    else:
                        protocol = None
                        normalized_for_0d = normalized
                        physical_for_0d = physical
                        candidate_ids_for_0d = candidate_ids
                    round_trip_started = time.perf_counter()
                    if len(normalized_for_0d):
                        result = evaluator.evaluate(
                            target.to_dict(),
                            normalized_for_0d,
                            physical_for_0d,
                            candidate_ids=candidate_ids_for_0d,
                        )
                    else:
                        result = sampling_budget_exhausted_result(
                            target.to_dict(),
                            "screened_random"
                            if mode == "screened_random"
                            else str(config["model"]["model_id"]),
                            is_training_domain=evaluator.target_in_training_domain(
                                target.to_dict()
                            ),
                        )
                    round_trip_seconds = time.perf_counter() - round_trip_started
                    checked = (
                        protocol_validation
                        if protocol_validation is not None
                        else result.checked_candidates
                    )
                    forward = result.forward_results
                    ranked = result.ranked_candidates
                    failures = result.failures
                    round_trip_metrics = result.metrics
                    sampling_metrics = {
                        "skipped": False,
                        "requested_count": requested_count,
                        "returned_count": len(normalized),
                        "sampling_seconds": sampling_seconds,
                        "round_trip_seconds": round_trip_seconds,
                        "ode_nfe": ode_nfe,
                        "prefilter_0d_evaluations": int(
                            result.metrics["candidate_valid_count"]
                        ),
                        "candidate_protocol": candidate_protocol,
                        "N_raw": int(len(normalized)),
                        "N_valid": int(
                            protocol.n_valid
                            if protocol is not None
                            else result.metrics["candidate_valid_count"]
                        ),
                        "N_keep": int(
                            protocol.n_keep
                            if protocol is not None
                            else result.metrics["candidate_count"]
                        ),
                        "N_trimmed": int(
                            protocol.n_trimmed if protocol is not None else 0
                        ),
                        "valid_rate": float(
                            protocol.valid_rate
                            if protocol is not None
                            else result.metrics["candidate_valid_count"]
                            / max(len(normalized), 1)
                        ),
                        "diagnostics": sample_diagnostics,
                    }
            raw.to_csv(run_dir / "generator" / "raw_candidates.csv", index=False)
            checked.to_csv(
                run_dir / "generator" / "checked_candidates.csv", index=False
            )
            write_json(
                run_dir / "generator" / "sampling_metrics.json", sampling_metrics
            )
            forward.to_csv(
                run_dir / "round_trip" / "forward_results.csv", index=False
            )
            ranked.to_csv(
                run_dir / "round_trip" / "ranked_candidates.csv", index=False
            )
            failures.to_csv(run_dir / "round_trip" / "failures.csv", index=False)
            write_json(
                run_dir / "round_trip" / "metrics.json", round_trip_metrics
            )
            _complete_stage(
                run_dir,
                manifest,
                "generation_round_trip",
                generation_artifacts,
                time.perf_counter() - generation_started,
            )
        _interrupt_if_requested(
            run_dir, manifest, "generation_round_trip", stop_after_stage
        )

        population_artifacts = [
            "population/initial_genes.csv",
            "population/population_provenance.csv",
            "population/adapter_metrics.json",
        ]
        if preference_enabled:
            population_artifacts.extend(
                [
                    "population/preference_evaluations.csv",
                    "population/preference_selection.csv",
                ]
            )
        mode = str(config["mode"])
        if not _stage_completed(run_dir, manifest, "population"):
            population_started = time.perf_counter()
            background, param_names = build_background_population(
                DesignSpaceBounds(),
                optimizer_config.population_size,
                background_seed,
                str(config["population"]["background_method"]),
            )
            if mode == "random":
                initial_population = None
                proposed_genes = background
                provenance = pd.DataFrame(
                    {
                        "population_index": np.arange(optimizer_config.population_size),
                        "source": ["legacy_random"] * optimizer_config.population_size,
                        "mode": ["random"] * optimizer_config.population_size,
                        "background_row": np.arange(optimizer_config.population_size),
                        "background_seed": [background_seed]
                        * optimizer_config.population_size,
                        "background_method": [
                            str(config["population"]["background_method"])
                        ]
                        * optimizer_config.population_size,
                        "generated_fields": ["[]"] * optimizer_config.population_size,
                    }
                )
                adapter_metrics = {
                    "mode": "random",
                    "population_size": optimizer_config.population_size,
                    "generator_ratio": 0.0,
                    "selected_generated_count": 0,
                    "random_fill_count": 0,
                    "planned_random_count": optimizer_config.population_size,
                    "background_seed": background_seed,
                    "background_method": str(
                        config["population"]["background_method"]
                    ),
                }
            else:
                ranked = _read_ranked_candidates(
                    run_dir / "round_trip" / "ranked_candidates.csv",
                    design_names(schema),
                )
                if preference_enabled and not ranked.empty and "forward_valid" in ranked:
                    selected = ranked.loc[
                        ranked["forward_valid"].astype(bool)
                    ].sort_values(["rank", "candidate_id"], kind="stable")
                elif ranked.empty or "selected_rank" not in ranked:
                    selected = pd.DataFrame(columns=design_names(schema))
                else:
                    selected = ranked.dropna(subset=["selected_rank"]).sort_values(
                        ["selected_rank", "candidate_id"], kind="stable"
                    )
                if preference_enabled:
                    assert preference_reference_path is not None
                    preference_reference = pd.read_csv(preference_reference_path)
                    (
                        initial_population,
                        provenance,
                        adapter_metrics,
                        preference_evaluations,
                        preference_selection,
                    ) = _build_preference_population(
                        selected,
                        design_names(schema),
                        mode=mode,
                        config=config,
                        optimizer_config=optimizer_config,
                        background=background,
                        background_seed=background_seed,
                        reference=preference_reference,
                    )
                    proposed_genes = initial_population.genes
                    preference_evaluations.to_csv(
                        run_dir / "population" / "preference_evaluations.csv",
                        index=False,
                    )
                    preference_selection.to_csv(
                        run_dir / "population" / "preference_selection.csv",
                        index=False,
                    )
                else:
                    warm_start_mode = (
                        "generator" if mode == "generator" else "hybrid"
                    )
                    build_result = build_warm_start_population(
                        selected,
                        design_names(schema),
                        mode=warm_start_mode,
                        population_size=optimizer_config.population_size,
                        generator_ratio=(
                            float(config["population"]["generator_ratio"])
                            if mode in {"hybrid", "screened_random"}
                            else None
                        ),
                        background_seed=background_seed,
                        background_method=str(
                            config["population"]["background_method"]
                        ),
                        bounds=DesignSpaceBounds(),
                        duplicate_tolerance=float(
                            config["population"]["duplicate_tolerance"]
                        ),
                        candidate_source=(
                            "screened_random"
                            if mode == "screened_random"
                            else "generator"
                        ),
                    )
                    initial_population = build_result.population
                    proposed_genes = initial_population.genes
                    provenance = build_result.provenance
                    adapter_metrics = build_result.metadata
            initial_frame = pd.DataFrame(proposed_genes, columns=param_names)
            initial_frame.insert(0, "population_index", np.arange(len(initial_frame)))
            initial_frame.to_csv(
                run_dir / "population" / "initial_genes.csv", index=False
            )
            provenance.to_csv(
                run_dir / "population" / "population_provenance.csv", index=False
            )
            write_json(
                run_dir / "population" / "adapter_metrics.json", adapter_metrics
            )
            _complete_stage(
                run_dir,
                manifest,
                "population",
                population_artifacts,
                time.perf_counter() - population_started,
            )
        _interrupt_if_requested(run_dir, manifest, "population", stop_after_stage)

        optimization_artifacts = [
            "optimization/pareto.csv",
            "optimization/history.csv",
            "optimization/evaluation_ledger.csv",
            "optimization/metrics.json",
        ]
        if not _stage_completed(run_dir, manifest, "optimization"):
            optimization_started = time.perf_counter()
            proposed = pd.read_csv(
                run_dir / "population" / "initial_genes.csv"
            ).drop(columns=["population_index"])
            initial_population = (
                None if mode == "random" else _load_initial_population(run_dir)
            )
            np.random.seed(optimizer_seed)
            pareto, history = optimize(
                config=optimizer_config,
                bounds=DesignSpaceBounds(),
                verbose=False,
                initial_genes=initial_population,
            )
            actual_initial = np.asarray(history["initial_population_genes"], dtype=float)
            if not np.allclose(
                actual_initial,
                proposed.to_numpy(dtype=float),
                rtol=0.0,
                atol=1.0e-10,
            ):
                raise RuntimeError(
                    "Optimizer generation-0 population differs from adapter artifact"
                )
            param_names = list(history["param_names"])
            pareto_frame = _pareto_frame(pareto, param_names)
            history_frame = _optimizer_history_frame(history, optimizer_config)
            evaluation_ledger = _evaluation_ledger_frame(history)
            pareto_frame.to_csv(
                run_dir / "optimization" / "pareto.csv", index=False
            )
            history_frame.to_csv(
                run_dir / "optimization" / "history.csv", index=False
            )
            evaluation_ledger.to_csv(
                run_dir / "optimization" / "evaluation_ledger.csv", index=False
            )
            expected_evaluations = optimizer_config.population_size * (
                optimizer_config.n_generations + 1
            )
            if len(evaluation_ledger) != expected_evaluations:
                raise RuntimeError(
                    "NSGA-II evaluation ledger count differs from configured budget"
                )
            optimization_metrics = {
                "optimizer_seed": optimizer_seed,
                "population_size": optimizer_config.population_size,
                "n_generations": optimizer_config.n_generations,
                "nsga2_0d_evaluations": len(evaluation_ledger),
                "pareto_size": len(pareto),
                "final_feasible_count": int(history["feasible_count"][-1]),
                "elapsed_seconds": time.perf_counter() - optimization_started,
                "initialization": history["initialization"],
            }
            write_json(
                run_dir / "optimization" / "metrics.json", optimization_metrics
            )
            _complete_stage(
                run_dir,
                manifest,
                "optimization",
                optimization_artifacts,
                time.perf_counter() - optimization_started,
            )
        _interrupt_if_requested(
            run_dir, manifest, "optimization", stop_after_stage
        )

        summary_artifacts = [
            "summary/metrics.json",
            "summary/evaluation_cost_ledger.csv",
            "summary/report.md",
            "environment.json",
            "git_state.json",
            "source_manifest.json",
        ]
        if not _stage_completed(run_dir, manifest, "summary"):
            summary_started = time.perf_counter()
            sampling_metrics = json.loads(
                (run_dir / "generator" / "sampling_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            adapter_metrics = json.loads(
                (run_dir / "population" / "adapter_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            optimization_metrics = json.loads(
                (run_dir / "optimization" / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            prefilter_count = int(
                sampling_metrics.get("prefilter_0d_evaluations", 0)
            )
            preference_count = int(
                adapter_metrics.get("preference", {}).get("evaluation_count", 0)
            )
            initialization_count = prefilter_count + preference_count
            nsga_count = int(optimization_metrics["nsga2_0d_evaluations"])
            cost_ledger = pd.DataFrame(
                [
                    {
                        "stage": "generator_prefilter",
                        "zero_d_evaluation_count": prefilter_count,
                    },
                    {
                        "stage": "preference_full_19d",
                        "zero_d_evaluation_count": preference_count,
                    },
                    {
                        "stage": "nsga2",
                        "zero_d_evaluation_count": nsga_count,
                    },
                ]
            )
            cost_ledger.to_csv(
                run_dir / "summary" / "evaluation_cost_ledger.csv", index=False
            )
            metrics = {
                "status": "completed",
                "run_id": config["run_id"],
                "target_id": target.target_id,
                "mode": mode,
                "model_id": config["model"].get("model_id"),
                "generation": {
                    "requested_count": int(
                        config["generation"]["n_generated"]
                    ) if mode != "random" else 0,
                    "returned_count": int(sampling_metrics.get("returned_count", 0)),
                    "ode_nfe": int(sampling_metrics.get("ode_nfe", 0)),
                },
                "population": {
                    "population_size": optimizer_config.population_size,
                    "generator_ratio": float(adapter_metrics["generator_ratio"]),
                    "generated_count": int(
                        adapter_metrics.get("selected_generated_count", 0)
                    ),
                    "random_fill_count": int(
                        adapter_metrics.get("random_fill_count", 0)
                    ),
                    "planned_random_count": int(
                        adapter_metrics.get("planned_random_count", 0)
                    ),
                },
                "optimization": optimization_metrics,
                "cost": {
                    "generator_sampling_seconds": float(
                        sampling_metrics.get("sampling_seconds", 0.0)
                    ),
                    "round_trip_seconds": float(
                        sampling_metrics.get("round_trip_seconds", 0.0)
                    ),
                    "adapter_seconds": float(
                        manifest["stages"]["population"].get("elapsed_seconds", 0.0)
                    ),
                    "optimizer_seconds": float(
                        optimization_metrics["elapsed_seconds"]
                    ),
                    "prefilter_0d_evaluations": prefilter_count,
                    "preference_19d_evaluations": preference_count,
                    "initialization_0d_evaluations": initialization_count,
                    "nsga2_0d_evaluations": nsga_count,
                    "total_0d_evaluations": initialization_count + nsga_count,
                    "current_execution_wall_seconds": time.perf_counter()
                    - pipeline_started,
                    "accumulated_stage_wall_seconds": sum(
                        float(value.get("elapsed_seconds", 0.0))
                        for stage, value in manifest["stages"].items()
                        if stage != "summary" and value.get("status") == "completed"
                    )
                    + (time.perf_counter() - summary_started),
                },
                "ledger_checks": {
                    "optimization_ledger_count": len(
                        pd.read_csv(
                            run_dir / "optimization" / "evaluation_ledger.csv"
                        )
                    ),
                    "cost_ledger_sum": int(
                        cost_ledger["zero_d_evaluation_count"].sum()
                    ),
                    "counts_consistent": bool(
                        len(
                            pd.read_csv(
                                run_dir / "optimization" / "evaluation_ledger.csv"
                            )
                        )
                        == nsga_count
                        and int(cost_ledger["zero_d_evaluation_count"].sum())
                        == initialization_count + nsga_count
                    ),
                },
                "generator_superiority_required": False,
            }
            write_json(run_dir / "summary" / "metrics.json", metrics)
            (run_dir / "summary" / "report.md").write_text(
                _report_text(target, config, metrics), encoding="utf-8"
            )
            environment = collect_environment()
            environment["cuda_visible_devices"] = os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            )
            write_json(run_dir / "environment.json", environment)
            write_json(
                run_dir / "git_state.json", collect_git_state(repo_root, run_dir)
            )
            write_json(
                run_dir / "source_manifest.json", collect_source_manifest(repo_root)
            )
            _complete_stage(
                run_dir,
                manifest,
                "summary",
                summary_artifacts,
                time.perf_counter() - summary_started,
            )
        _interrupt_if_requested(run_dir, manifest, "summary", stop_after_stage)

        manifest["status"] = "completed"
        manifest["current_stage"] = None
        manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest["error"] = None
        manifest["provenance"] = validation_payload
        _write_manifest(run_dir, manifest)
        return run_dir
    except PipelineInterrupted:
        raise
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        _write_manifest(run_dir, manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one complete inverse-design job")
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = run_inverse_design(
        args.target_config,
        args.run_config,
        output_dir_override=args.output_dir,
        resume=args.resume,
    )
    manifest = yaml.safe_load((output / "manifest.yaml").read_text(encoding="utf-8"))
    print(json.dumps({"output": str(output), "status": manifest["status"]}))
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
