"""WP5 orchestration for NSGA-II Warm-Start functional validation."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from .baseline import (
    collect_environment,
    collect_git_state,
    collect_source_manifest,
    sha256_file,
    write_json,
    write_yaml,
)
from .dataset import DesignSpaceBounds
from .generator_dataset import design_names, load_schema
from .optimizer import OptimizationConfig, optimize, optimizer_parameter_space
from .warm_start import (
    WARM_START_MODES,
    build_background_population,
    build_warm_start_population,
)


def load_wp5_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"version", "input", "population", "optimizer", "runs", "output"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"WP5 config is missing keys: {sorted(missing)}")
    if not config["runs"]:
        raise ValueError("WP5 config must contain at least one run")
    run_ids = [str(value["run_id"]) for value in config["runs"]]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("WP5 run_id values must be unique")
    for run in config["runs"]:
        if run["mode"] not in WARM_START_MODES:
            raise ValueError(f"Unsupported WP5 mode: {run['mode']}")
    return config


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _select_ranked_candidates(
    frame: pd.DataFrame,
    input_config: Mapping[str, Any],
    generated_fields: Sequence[str],
) -> Tuple[pd.DataFrame, str]:
    required = {
        "model",
        "target_id",
        "target_type",
        "candidate_id",
        "forward_valid",
        "rank",
        *generated_fields,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"WP4 ranked candidates are missing fields: {sorted(missing)}")
    filtered = frame.loc[
        (frame["model"].astype(str) == str(input_config["model"]))
        & (frame["target_type"].astype(str) == str(input_config["target_type"]))
        & _as_bool(frame["forward_valid"])
    ].copy()
    configured_target = input_config.get("target_id")
    if configured_target is not None:
        target_id = str(configured_target)
        filtered = filtered.loc[filtered["target_id"].astype(str) == target_id]
        if filtered.empty:
            raise ValueError(f"No forward-valid WP4 rows for target_id={target_id}")
    else:
        minimum = int(input_config["minimum_forward_valid_candidates"])
        counts = filtered.groupby("target_id").size()
        eligible = sorted(str(value) for value in counts[counts >= minimum].index)
        if not eligible:
            raise ValueError(
                "No target supplies the configured minimum number of forward-valid candidates"
            )
        target_id = eligible[0]
        filtered = filtered.loc[filtered["target_id"].astype(str) == target_id]
    filtered = filtered.sort_values(["rank", "candidate_id"], kind="stable")
    return filtered.reset_index(drop=True), target_id


def _pareto_frame(pareto, param_names: Sequence[str]) -> pd.DataFrame:
    rows = []
    for index, individual in enumerate(pareto):
        row: Dict[str, Any] = {
            "pareto_index": index,
            "individual_id": individual.individual_id,
            "provenance": json.dumps(individual.provenance, sort_keys=True),
            "is_feasible": bool(individual.is_feasible),
            "rank": int(individual.rank),
            "crowding_distance": float(individual.crowding_distance),
        }
        row.update(
            {
                f"gene__{name}": float(individual.genes[field_index])
                for field_index, name in enumerate(param_names)
            }
        )
        row.update(
            {
                f"objective_internal_{objective_index}": float(value)
                for objective_index, value in enumerate(individual.objectives)
            }
        )
        row.update(
            {f"performance__{name}": value for name, value in individual.performance.items()}
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _history_frame(history: Mapping[str, Any], population_size: int) -> pd.DataFrame:
    rows = []
    for index, (pareto_size, feasible_count) in enumerate(
        zip(history["pareto_size"], history["feasible_count"])
    ):
        row: Dict[str, Any] = {
            "generation": index + 1,
            "evaluation_count": (index + 2) * population_size,
            "pareto_size": pareto_size,
            "feasible_count": feasible_count,
        }
        if index < len(history["best_objectives"]):
            for objective_index, value in enumerate(history["best_objectives"][index]):
                row[f"mean_internal_objective_{objective_index}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _validate_background_preservation(
    initial: np.ndarray,
    background: np.ndarray,
    param_names: Sequence[str],
    generated_fields: Sequence[str],
) -> bool:
    untouched = [
        index for index, name in enumerate(param_names) if name not in generated_fields
    ]
    return bool(np.array_equal(initial[:, untouched], background[:, untouched]))


def _write_report(
    path: Path,
    gate: Mapping[str, Any],
    summary: pd.DataFrame,
    target_id: str,
) -> None:
    lines = [
        "# WP5 Warm-Start 功能验收报告",
        "",
        f"- Functional gate: {'PASS' if gate['passed'] else 'FAIL'}",
        f"- Frozen WP4 target: {target_id}",
        "- Generator superiority required: no",
        "- Legacy Random path changed: no",
        "",
        "## Runs",
        "",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"- {row['run_id']}: mode={row['mode']}, ratio={row['generator_ratio']}, "
            f"population={row['population_size']}, generated={row['generated_count']}, "
            f"fill={row['random_fill_count']}, pareto={row['pareto_size']}, "
            f"background_14d_preserved={row['background_14d_preserved']}"
        )
    lines.extend(
        [
            "",
            "WP5 only changes generation-0 population construction. Evaluation, "
            "selection, SBX crossover, polynomial mutation, non-dominated sorting, "
            "and crowding-distance selection continue to use the original NSGA-II path.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_wp5(
    config_path: Path,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    config_path = Path(config_path).resolve()
    config = load_wp5_config(config_path)
    ranked_path = (repo_root / config["input"]["ranked_candidates_csv"]).resolve()
    schema_path = (repo_root / config["input"]["generator_schema"]).resolve()
    if not ranked_path.is_file():
        raise FileNotFoundError(
            f"WP5 requires the completed WP4 ranked ledger: {ranked_path}"
        )
    if not schema_path.is_file():
        raise FileNotFoundError(f"WP5 generator schema is missing: {schema_path}")
    output_dir = (
        output_dir_override.resolve()
        if output_dir_override is not None
        else (repo_root / config["output"]["directory"]).resolve()
    )
    if output_dir.exists() and bool(config["output"].get("refuse_overwrite", True)):
        raise FileExistsError(f"WP5 output exists and will not be overwritten: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = dt.datetime.now(dt.timezone.utc)
    shutil.copy2(config_path, output_dir / "build_config.yaml")

    schema = load_schema(schema_path)
    generated_fields = design_names(schema)
    candidates, target_id = _select_ranked_candidates(
        pd.read_csv(ranked_path), config["input"], generated_fields
    )
    candidates.to_csv(output_dir / "input_ranked_candidates.csv", index=False)
    bounds = DesignSpaceBounds()
    param_names, _, _ = optimizer_parameter_space(bounds)
    optimizer_values = dict(config["optimizer"])
    optimizer_seed = int(optimizer_values.pop("seed"))
    optimizer_config = OptimizationConfig(**optimizer_values)
    background_seed = int(config["population"]["background_seed"])
    background_method = str(config["population"]["background_method"])
    background, background_names = build_background_population(
        bounds,
        optimizer_config.population_size,
        background_seed,
        background_method,
    )
    if background_names != param_names:
        raise RuntimeError("Background and optimizer parameter schemas differ")
    pd.DataFrame(background, columns=param_names).to_csv(
        output_dir / "frozen_background_population.csv", index=False
    )

    run_records: List[Dict[str, Any]] = []
    gate_checks: Dict[str, Any] = {}
    for run_config in config["runs"]:
        run_id = str(run_config["run_id"])
        mode = str(run_config["mode"])
        ratio = float(run_config["generator_ratio"])
        run_dir = output_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        build_result = None
        if mode == "random":
            initial_population = None
        else:
            build_result = build_warm_start_population(
                candidates,
                generated_fields,
                mode=mode,
                population_size=optimizer_config.population_size,
                generator_ratio=ratio if mode == "hybrid" else None,
                background_seed=background_seed,
                background_method=background_method,
                bounds=bounds,
                duplicate_tolerance=float(
                    config["population"]["duplicate_tolerance"]
                ),
            )
            initial_population = build_result.population

        np.random.seed(optimizer_seed)
        run_started = time.perf_counter()
        pareto, history = optimize(
            config=optimizer_config,
            bounds=bounds,
            verbose=False,
            initial_genes=initial_population,
        )
        elapsed = time.perf_counter() - run_started
        initial_matrix = np.asarray(history["initial_population_genes"], dtype=float)
        initial_frame = pd.DataFrame(initial_matrix, columns=param_names)
        initial_frame.insert(0, "population_index", np.arange(len(initial_frame)))
        initial_frame.to_csv(run_dir / "initial_genes.csv", index=False)
        provenance = pd.DataFrame(history["initial_population_provenance"])
        provenance.to_csv(run_dir / "population_provenance.csv", index=False)
        write_json(run_dir / "initialization.json", history["initialization"])
        pareto_frame = _pareto_frame(pareto, param_names)
        pareto_frame.to_csv(run_dir / "pareto.csv", index=False)
        _history_frame(history, optimizer_config.population_size).to_csv(
            run_dir / "history.csv", index=False
        )

        sources = provenance["source"].value_counts().to_dict()
        background_preserved = _validate_background_preservation(
            initial_matrix, background, param_names, generated_fields
        )
        random_matches_wp0 = (
            bool(np.array_equal(initial_matrix, background))
            if mode == "random"
            else None
        )
        record = {
            "run_id": run_id,
            "mode": mode,
            "generator_ratio": ratio,
            "population_size": len(initial_matrix),
            "generated_count": int(sources.get("generator", 0)),
            "random_count": int(
                sources.get("random", 0) + sources.get("legacy_random", 0)
            ),
            "random_fill_count": int(sources.get("random_fill", 0)),
            "pareto_size": len(pareto),
            "final_feasible_count": int(history["feasible_count"][-1]),
            "physics_evaluation_count": optimizer_config.population_size
            * (optimizer_config.n_generations + 1),
            "elapsed_seconds": elapsed,
            "background_14d_preserved": background_preserved,
            "random_matches_wp0_background": random_matches_wp0,
            "initial_genes_sha256": sha256_file(run_dir / "initial_genes.csv"),
            "provenance_sha256": sha256_file(run_dir / "population_provenance.csv"),
            "pareto_sha256": sha256_file(run_dir / "pareto.csv"),
        }
        run_records.append(record)
        expected_generated = int(round(optimizer_config.population_size * ratio))
        gate_checks[run_id] = {
            "completed": True,
            "population_size_correct": len(initial_matrix)
            == optimizer_config.population_size,
            "source_count_complete": sum(sources.values())
            == optimizer_config.population_size,
            "requested_ratio_realized": (
                mode == "random"
                or record["generated_count"] + record["random_fill_count"]
                == expected_generated
            ),
            "background_14d_preserved": background_preserved,
            "random_compatibility_satisfied": bool(
                mode != "random" or random_matches_wp0
            ),
            "pareto_nonempty": bool(len(pareto)),
        }
        write_json(run_dir / "metrics.json", record)

    summary = pd.DataFrame(run_records)
    summary.to_csv(output_dir / "run_summary.csv", index=False)
    required_modes = {"random", "generator", "hybrid"}
    observed_modes = {str(value["mode"]) for value in config["runs"]}
    gate = {
        "gate": "WP5 Warm-Start Functional Gate",
        "passed": bool(
            required_modes.issubset(observed_modes)
            and all(all(check.values()) for check in gate_checks.values())
        ),
        "required_modes_present": required_modes.issubset(observed_modes),
        "generator_superiority_required": False,
        "checks": gate_checks,
    }
    write_json(output_dir / "functional_gate.json", gate)
    _write_report(output_dir / "report.md", gate, summary, target_id)
    write_json(output_dir / "environment.json", collect_environment())
    write_json(output_dir / "git_state.json", collect_git_state(repo_root, output_dir))
    write_json(output_dir / "source_manifest.json", collect_source_manifest(repo_root))
    finished = dt.datetime.now(dt.timezone.utc)
    manifest = {
        "wp": "WP5",
        "version": config["version"],
        "status": "completed" if gate["passed"] else "functional_gate_failed",
        "functional_gate_passed": gate["passed"],
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "inputs": {
            "ranked_candidates_csv": str(ranked_path),
            "ranked_candidates_sha256": sha256_file(ranked_path),
            "generator_schema": str(schema_path),
            "generator_schema_sha256": sha256_file(schema_path),
            "selected_model": config["input"]["model"],
            "selected_target_id": target_id,
        },
        "optimizer_config": dataclasses.asdict(optimizer_config),
        "optimizer_seed": optimizer_seed,
        "runs": run_records,
    }
    write_yaml(output_dir / "wp5_manifest.yaml", manifest)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run WP5 NSGA-II Warm-Start functional validation"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config/optimizer_wp5.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = run_wp5(args.config, output_dir_override=args.output_dir)
    manifest = yaml.safe_load((output / "wp5_manifest.yaml").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output": str(output),
                "functional_gate_passed": manifest["functional_gate_passed"],
            }
        )
    )
    return 0 if manifest["functional_gate_passed"] else 1
