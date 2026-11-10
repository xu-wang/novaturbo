"""Formal Stage-II WP5 experiment-matrix orchestration and aggregation."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from .baseline import sha256_file, write_json, write_yaml
from .stage2_wp5 import load_stage2_wp5_config, run_stage2_wp5


FORMAL_ARMS = ("auto_delivery", "multiobjective_refinement")


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _stable_seed(base_seed: int, *parts: Any) -> int:
    import hashlib

    value = "|".join(str(part) for part in parts).encode("utf-8")
    return int(base_seed) + int.from_bytes(hashlib.sha256(value).digest()[:4], "big") % 1_000_000


def load_stage2_wp5_matrix_config(path: Path) -> Dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"version", "template", "targets", "model_sets", "arms", "runtime", "output"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"S2-WP5 matrix config is missing keys: {sorted(missing)}")
    arms = config["arms"]
    if set(arms) != set(FORMAL_ARMS):
        raise ValueError(f"S2-WP5 formal arms must be {list(FORMAL_ARMS)}")
    if str(arms["auto_delivery"].get("mode")) != "generator_auto_refinement":
        raise ValueError("auto_delivery must use generator_auto_refinement")
    multi = arms["multiobjective_refinement"]
    if str(multi.get("mode")) != "generator_auto_refinement":
        raise ValueError("multiobjective_refinement must use generator_auto_refinement")
    if not bool(multi.get("extra_engineering_objectives", False)):
        raise ValueError("multiobjective_refinement must enable extra engineering objectives")
    target_count = int(multi.get("targets_per_type", 0))
    if target_count <= 0:
        raise ValueError("multiobjective_refinement.targets_per_type must be positive")
    if int(config["runtime"].get("sample_seed_base", 0)) <= 0:
        raise ValueError("runtime.sample_seed_base must be positive")
    if not config["model_sets"]:
        raise ValueError("model_sets must not be empty")
    return config


def _load_targets(repo_root: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    suite_path = _resolve(repo_root, str(config["suite_csv"]))
    frozen_path = _resolve(repo_root, str(config["frozen_ids"]))
    if not suite_path.is_file() or not frozen_path.is_file():
        raise FileNotFoundError("S2-WP5 target suite or frozen ID list is missing")
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    if sha256_file(suite_path) != str(frozen["source_target_suite_sha256"]):
        raise ValueError("S2-WP5 target suite differs from frozen S2-WP1 selection")
    ids = [str(value) for value in frozen["target_ids"]]
    suite = pd.read_csv(suite_path)
    selected = suite.loc[suite["target_id"].astype(str).isin(ids)].copy()
    order = {target_id: index for index, target_id in enumerate(ids)}
    selected["frozen_order"] = selected["target_id"].astype(str).map(order)
    selected = selected.sort_values("frozen_order").reset_index(drop=True)
    if selected["target_id"].astype(str).tolist() != ids:
        raise ValueError("S2-WP5 could not resolve every frozen target exactly once")
    expected = {str(key): int(value) for key, value in frozen["counts"].items()}
    actual = {str(key): int(value) for key, value in selected["target_type"].value_counts().items()}
    if actual != expected:
        raise ValueError(f"S2-WP5 target counts differ: expected={expected}, actual={actual}")
    return selected


def _multiobjective_ids(targets: pd.DataFrame, count_per_type: int) -> set[str]:
    selected: set[str] = set()
    for _, frame in targets.groupby("target_type", sort=False):
        indices = np.linspace(0, len(frame) - 1, num=count_per_type, dtype=int)
        selected.update(frame.iloc[np.unique(indices)]["target_id"].astype(str))
    return selected


def build_wp5_run_matrix(config: Mapping[str, Any], repo_root: Path) -> pd.DataFrame:
    targets = _load_targets(repo_root, config["targets"])
    multi_ids = _multiobjective_ids(
        targets, int(config["arms"]["multiobjective_refinement"]["targets_per_type"])
    )
    rows = []
    base_seed = int(config["runtime"]["sample_seed_base"])
    for model_set in config["model_sets"]:
        training_seed = int(model_set["training_seed"])
        for model in model_set["models"]:
            model_id = str(model["model_id"])
            checkpoint = str(model["checkpoint_path"])
            for target in targets.to_dict(orient="records"):
                target_id = str(target["target_id"])
                for arm in FORMAL_ARMS:
                    if arm == "multiobjective_refinement" and target_id not in multi_ids:
                        continue
                    sample_seed = _stable_seed(base_seed, arm, model_id, training_seed, target_id)
                    run_id = f"{arm}__{model_id}__seed{training_seed}__{target_id[:12]}"
                    rows.append(
                        {
                            "run_id": run_id,
                            "arm": arm,
                            "model": model_id,
                            "training_seed": training_seed,
                            "target_id": target_id,
                            "target_type": str(target["target_type"]),
                            "checkpoint_path": checkpoint,
                            "sample_seed": sample_seed,
                        }
                    )
    matrix = pd.DataFrame(rows)
    if matrix["run_id"].duplicated().any():
        raise RuntimeError("S2-WP5 run IDs are not unique")
    return matrix


def _run_config(
    template: Mapping[str, Any], matrix_config: Mapping[str, Any], row: Mapping[str, Any]
) -> Dict[str, Any]:
    value = copy.deepcopy(template)
    arm = matrix_config["arms"][str(row["arm"])]
    value["version"] = f"{matrix_config['version']}__{row['run_id']}"
    value["target"] = {
        "suite_csv": matrix_config["targets"]["suite_csv"],
        "target_id": str(row["target_id"]),
        "tolerance_profile": str(matrix_config["targets"].get("tolerance_profile", "round_trip_metrics_v1")),
    }
    value["model"] = {
        "model_id": str(row["model"]),
        "checkpoint_path": str(row["checkpoint_path"]),
    }
    value["generation"]["seed"] = int(row["sample_seed"])
    value["delivery"]["mode"] = str(arm["mode"])
    value["delivery"]["minimum_hit_count"] = int(arm["minimum_hit_count"])
    value["delivery"]["maximum_delivered_candidates"] = int(
        arm["maximum_delivered_candidates"]
    )
    value["delivery"]["extra_engineering_objectives"] = bool(
        arm.get("extra_engineering_objectives", False)
    )
    if "nsga2_objectives" in arm:
        value["nsga2"]["objectives"] = list(arm["nsga2_objectives"])
        value["nsga2"]["maximize"] = list(arm["nsga2_maximize"])
    value["runtime"]["device"] = str(matrix_config["runtime"].get("device", "auto"))
    value["output"] = {"directory": "unused", "refuse_overwrite": True}
    return value


def _read_csv_allow_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _summary_row(row: Mapping[str, Any], run_dir: Path) -> Dict[str, Any]:
    generation = json.loads((run_dir / "generation" / "metrics.json").read_text(encoding="utf-8"))
    direct = json.loads((run_dir / "direct" / "metrics.json").read_text(encoding="utf-8"))
    decision = json.loads((run_dir / "delivery" / "decision.json").read_text(encoding="utf-8"))
    direct_frame = _read_csv_allow_empty(run_dir / "direct" / "forward_results.csv")
    delivered = pd.read_csv(run_dir / "delivery" / "delivered_candidates.csv")
    best_direct = direct.get("best") or {}
    best_delivered = float(delivered["total_error"].min()) if len(delivered) else math.inf
    if {"forward_valid", "within_tolerance"}.issubset(direct_frame.columns):
        valid_direct = direct_frame.loc[direct_frame["forward_valid"].astype(bool)]
        direct_hits = valid_direct.loc[valid_direct["within_tolerance"].astype(bool)]
    else:
        valid_direct = direct_frame
        direct_hits = direct_frame
    direct_mass_pool = direct_hits if len(direct_hits) else valid_direct
    direct_hit_positions = (
        np.flatnonzero(direct_frame["within_tolerance"].astype(bool).to_numpy())
        if "within_tolerance" in direct_frame
        else np.asarray([], dtype=int)
    )
    first_hit_evaluation = (
        int(direct_hit_positions[0]) + 1 if len(direct_hit_positions) else math.inf
    )
    direct_min_mass = (
        float(direct_mass_pool["actual_total_mass_kg"].min())
        if len(direct_mass_pool) and "actual_total_mass_kg" in direct_mass_pool
        else math.inf
    )
    delivered_mass_column = (
        "performance__total_mass_kg"
        if "performance__total_mass_kg" in delivered
        else "actual_total_mass_kg"
    )
    final_min_mass = (
        float(delivered[delivered_mass_column].min())
        if len(delivered) and delivered_mass_column in delivered
        else math.inf
    )
    actions = decision.get("executed_actions", [])
    return {
        **dict(row),
        "N_raw": int(generation["N_raw"]),
        "N_valid": int(generation["N_valid"]),
        "N_keep": int(generation["N_keep"]),
        "N_trimmed": int(generation["N_trimmed"]),
        "direct_forward_valid_count": int(direct["forward_valid_count"]),
        "direct_hit_count": int(decision["direct_hit_count"]),
        "direct_first_hit_evaluation": first_hit_evaluation,
        "direct_best_error": float(best_direct.get("total_error", math.inf)),
        "direct_min_deliverable_mass_kg": direct_min_mass,
        "final_best_error": best_delivered,
        "final_hit_count": int(
            delivered["within_tolerance"].astype(bool).sum()
            if "within_tolerance" in delivered
            else 0
        ),
        "final_min_delivered_mass_kg": final_min_mass,
        "delivered_count": int(decision["delivered_count"]),
        "delivery_source": str(decision["delivery_source"]),
        "stop_reason": str(decision["stop_reason"]),
        "executed_actions": json.dumps(actions),
        "used_latent_repair": "latent_repair" in actions,
        "used_design_gradient": "design_gradient" in actions,
        "used_nsga2": "nsga2" in actions,
        "direct_0d": int(decision["cost"]["direct_0d"]),
        "latent_repair_0d": int(decision["cost"]["latent_repair_0d"]),
        "gradient_0d": int(decision["cost"]["gradient_0d"]),
        "nsga2_0d": int(decision["cost"]["nsga2_0d"]),
        "total_0d": int(decision["cost"]["total_0d"]),
        "run_directory": str(run_dir),
    }


def aggregate_wp5_matrix(output_dir: Path, matrix: pd.DataFrame) -> Dict[str, Path]:
    rows = []
    missing = []
    for row in matrix.to_dict(orient="records"):
        run_dir = output_dir / "runs" / str(row["run_id"])
        if not (run_dir / "manifest.yaml").is_file():
            missing.append(str(row["run_id"]))
            continue
        rows.append(_summary_row(row, run_dir))
    if missing:
        raise RuntimeError(f"S2-WP5 matrix is incomplete; missing {len(missing)} runs")
    summary = pd.DataFrame(rows)
    summary_path = output_dir / "target_summary.csv"
    summary.to_csv(summary_path, index=False)
    grouped_rows = []
    for keys, frame in summary.groupby(["arm", "model", "target_type"], sort=True):
        grouped_rows.append(
            {
                "arm": keys[0],
                "model": keys[1],
                "target_type": keys[2],
                "run_count": len(frame),
                "direct_target_success_rate": float((frame["direct_hit_count"] > 0).mean()),
                "direct_hit_count_median": float(frame["direct_hit_count"].median()),
                "direct_first_hit_evaluation_median": float(
                    frame["direct_first_hit_evaluation"].median()
                ),
                "direct_best_error_median": float(frame["direct_best_error"].median()),
                "final_best_error_median": float(frame["final_best_error"].median()),
                "direct_min_deliverable_mass_median_kg": float(
                    frame["direct_min_deliverable_mass_kg"].median()
                ),
                "final_min_delivered_mass_median_kg": float(
                    frame["final_min_delivered_mass_kg"].median()
                ),
                "refinement_activation_rate": float(
                    (frame["used_latent_repair"] | frame["used_design_gradient"] | frame["used_nsga2"]).mean()
                ),
                "nsga2_activation_rate": float(frame["used_nsga2"].mean()),
                "total_0d_median": float(frame["total_0d"].median()),
                "total_0d_mean": float(frame["total_0d"].mean()),
                "N_valid_median": float(frame["N_valid"].median()),
                "N_keep_median": float(frame["N_keep"].median()),
            }
        )
    aggregate = pd.DataFrame(grouped_rows)
    aggregate_path = output_dir / "aggregate_summary.csv"
    aggregate.to_csv(aggregate_path, index=False)
    failures = summary.loc[
        (~np.isfinite(summary["final_best_error"])) | (summary["delivered_count"] <= 0)
    ].copy()
    failures_path = output_dir / "failure_ledger.csv"
    failures.to_csv(failures_path, index=False)
    return {"target_summary": summary_path, "aggregate_summary": aggregate_path, "failure_ledger": failures_path}


def _candidate_protocol_is_valid(summary: pd.DataFrame) -> bool:
    expected = summary["N_valid"].astype(int).map(
        lambda count: 0 if count == 0 else 2 ** int(math.floor(math.log2(count)))
    )
    return bool((summary["N_keep"].astype(int) == expected).all())


def run_stage2_wp5_matrix(
    config_path: Path,
    *,
    repo_root: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
    resume: bool = False,
    prepare_only: bool = False,
) -> Path:
    repo_root = (repo_root or Path.cwd()).resolve()
    config_path = Path(config_path).resolve()
    config = load_stage2_wp5_matrix_config(config_path)
    template_path = _resolve(repo_root, str(config["template"]))
    template = load_stage2_wp5_config(template_path)
    output_dir = (
        output_dir_override.resolve()
        if output_dir_override is not None
        else _resolve(repo_root, str(config["output"]["directory"]))
    )
    matrix = build_wp5_run_matrix(config, repo_root)
    expected_runs = len(matrix)
    if output_dir.exists() and not resume:
        raise FileExistsError(f"S2-WP5 matrix output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = output_dir / "generated_configs"
    runs_dir = output_dir / "runs"
    generated_dir.mkdir(exist_ok=True)
    runs_dir.mkdir(exist_ok=True)
    matrix_path = output_dir / "run_matrix.csv"
    if matrix_path.is_file() and resume:
        existing = pd.read_csv(matrix_path)
        if existing.to_dict(orient="records") != matrix.to_dict(orient="records"):
            raise ValueError("Existing S2-WP5 run matrix differs from current config")
    else:
        matrix.to_csv(matrix_path, index=False)
        shutil.copy2(config_path, output_dir / "build_config.yaml")
    manifest_path = output_dir / "manifest.yaml"
    started = dt.datetime.now(dt.timezone.utc)
    manifest = {
        "version": config["version"],
        "status": "prepared",
        "started_utc": started.isoformat(),
        "config_sha256": sha256_file(config_path),
        "template_sha256": sha256_file(template_path),
        "run_matrix_sha256": sha256_file(matrix_path),
        "expected_run_count": expected_runs,
        "completed_run_count": 0,
    }
    if manifest_path.is_file() and resume:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != config["version"]:
            raise ValueError("Existing S2-WP5 matrix uses a different version")
    write_yaml(manifest_path, manifest)
    if prepare_only:
        return output_dir

    completed_count = 0
    for row in matrix.to_dict(orient="records"):
        run_id = str(row["run_id"])
        run_dir = runs_dir / run_id
        if (run_dir / "manifest.yaml").is_file():
            completed_count += 1
            continue
        if run_dir.exists():
            raise RuntimeError(f"Incomplete S2-WP5 run requires inspection: {run_dir}")
        run_config_path = generated_dir / f"{run_id}.yaml"
        write_yaml(run_config_path, _run_config(template, config, row))
        run_stage2_wp5(run_config_path, repo_root=repo_root, output_dir_override=run_dir)
        completed_count += 1
        manifest["status"] = "running"
        manifest["completed_run_count"] = completed_count
        write_yaml(manifest_path, manifest)

    paths = aggregate_wp5_matrix(output_dir, matrix)
    summary = pd.read_csv(paths["target_summary"])
    failure_count = int((summary["delivered_count"] <= 0).sum())
    candidate_protocol_valid = _candidate_protocol_is_valid(summary)
    failure_ledger_count = len(pd.read_csv(paths["failure_ledger"]))
    failures_retained = failure_ledger_count == failure_count
    gate = {
        "passed": bool(
            len(summary) == expected_runs
            and set(summary["N_raw"].astype(int)) == {128}
            and candidate_protocol_valid
            and failures_retained
        ),
        "expected_run_count": expected_runs,
        "actual_run_count": len(summary),
        "fixed_raw_candidate_count": set(summary["N_raw"].astype(int)) == {128},
        "candidate_keep_rule_exact": candidate_protocol_valid,
        "successful_delivery_run_count": int((summary["delivered_count"] > 0).sum()),
        "failed_delivery_run_count": failure_count,
        "failed_runs_retained_in_ledger": failures_retained,
        "performance_superiority_required_for_functional_gate": False,
    }
    write_json(output_dir / "functional_gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError(f"S2-WP5 formal matrix functional gate failed: {gate}")
    manifest.update(
        {
            "status": "completed",
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "completed_run_count": completed_count,
            "functional_gate": gate,
            "artifacts": {name: str(path) for name, path in paths.items()},
        }
    )
    write_yaml(manifest_path, manifest)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/stage2/s2_wp5_matrix.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = run_stage2_wp5_matrix(
        args.config,
        output_dir_override=args.output_dir,
        resume=args.resume,
        prepare_only=args.prepare_only,
    )
    manifest = yaml.safe_load((output / "manifest.yaml").read_text(encoding="utf-8"))
    print(json.dumps({"output": str(output), "status": manifest["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
