#!/usr/bin/env python3
"""Run the full CFM-4 -> Diag -> gradient -> 0D -> NSGA-II strategy panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.baseline import sha256_file, write_json
from src.ai.diag_cfm_strategy import (
    load_diag_cfm_strategy_config,
    run_diag_cfm_strategy_target,
)


def _target_seed(base_seed: int, namespace: str, target_id: str) -> int:
    payload = f"{base_seed}:{namespace}:{target_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _diversity(frame: pd.DataFrame, normalized_columns: list[str]) -> float:
    values = frame[normalized_columns].to_numpy(dtype=float)
    if len(values) < 2:
        return 0.0
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances, axis=1).mean())


def _summarize_run(
    run_dir: Path,
    seed: int,
    target: dict,
    normalized_columns: list[str],
) -> dict:
    final_pool = pd.read_csv(run_dir / "delivery" / "final_candidate_pool.csv")
    valid = final_pool[final_pool["forward_valid"].astype(bool)].copy()
    errors = valid["total_error"].replace([np.inf, -np.inf], np.nan).dropna()
    decision = json.loads(
        (run_dir / "delivery" / "decision.json").read_text(encoding="utf-8")
    )
    generation = json.loads(
        (run_dir / "generation" / "metrics.json").read_text(encoding="utf-8")
    )
    best_candidates = errors.tolist()
    nsga_path = run_dir / "nsga2" / "pareto.csv"
    nsga_hit = False
    if nsga_path.is_file() and nsga_path.stat().st_size:
        nsga = pd.read_csv(nsga_path)
        if "total_error" in nsga:
            best_candidates.extend(
                pd.to_numeric(nsga["total_error"], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .tolist()
            )
        nsga_hit = bool(nsga.get("within_tolerance", pd.Series(dtype=bool)).astype(bool).any())
    best_total_error = float(min(best_candidates)) if best_candidates else np.nan
    top10 = errors.nsmallest(min(10, len(errors)))
    return {
        "seed": seed,
        "target_id": str(target["target_id"]),
        "target_type": str(target["target_type"]),
        "candidate_count": len(final_pool),
        "forward_valid_count": len(valid),
        "valid_rate": float(generation["N_valid"] / generation["N_raw"]),
        "hit_rate": float(valid["within_tolerance"].astype(bool).mean()) if len(valid) else 0.0,
        "target_success": bool(
            valid.get("within_tolerance", pd.Series(dtype=bool)).astype(bool).any()
            or nsga_hit
        ),
        "best_total_error": best_total_error,
        "top10_error": float(top10.mean()) if len(top10) else np.nan,
        "median_error": float(errors.median()) if len(errors) else np.nan,
        "diversity": _diversity(valid, normalized_columns) if len(valid) else 0.0,
        "network_draw_count": int(generation["network_draw_count"]),
        "generation_seconds": float(decision["timing_seconds"]["generation_and_prefilter"]),
        "direct_0d_seconds": float(decision["timing_seconds"]["direct_0d"]),
        "gradient_seconds": float(decision["timing_seconds"]["design_gradient"]),
        "nsga2_seconds": float(decision["timing_seconds"]["nsga2"]),
        "direct_0d_evaluations": int(decision["cost"]["direct_0d"]),
        "gradient_0d_evaluations": int(decision["cost"]["gradient_0d"]),
        "nsga2_0d_evaluations": int(decision["cost"]["nsga2_0d"]),
        "total_0d_evaluations": int(decision["cost"]["total_0d"]),
        "executed_actions": json.dumps(decision["executed_actions"]),
        "stop_reason": str(decision["stop_reason"]),
        "run_directory": str(run_dir),
    }


def run(config_path: Path, output_override: Path | None, maximum_per_type: int | None) -> Path:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = (
        output_override.resolve()
        if output_override is not None
        else (REPO_ROOT / config["output"]["directory"]).resolve()
    )
    if output_dir.exists():
        raise FileExistsError(f"full-strategy output exists: {output_dir}")
    target_suite_path = (REPO_ROOT / config["target_suite"]).resolve()
    targets = pd.read_csv(target_suite_path)
    if maximum_per_type is not None:
        targets = (
            targets.groupby("target_type", sort=False, group_keys=False)
            .head(maximum_per_type)
            .reset_index(drop=True)
        )
    output_dir.mkdir(parents=True)
    (output_dir / "run_configs").mkdir()
    (output_dir / "runs").mkdir()
    shutil.copy2(config_path, output_dir / "build_config.yaml")
    targets.to_csv(output_dir / "evaluated_targets.csv", index=False)

    normalized_columns = [
        "normalized_compressor_pressure_ratio",
        "normalized_compressor_efficiency",
        "normalized_turbine_inlet_temp_K",
        "normalized_turbine_efficiency",
        "normalized_mass_flow_kg_s",
        "normalized_compressor_diameter_mm",
        "normalized_combustor_length_mm",
        "normalized_combustor_outer_diameter_mm",
        "normalized_nozzle_exit_diameter_mm",
        "normalized_lattice_density",
    ]
    rows = []
    base_seed = int(config["sample_seed"])
    namespace = str(config["sample_seed_namespace"])
    for seed_value, seed_spec in config["seeds"].items():
        seed = int(seed_value)
        for target in targets.to_dict(orient="records"):
            target_id = str(target["target_id"])
            sample_seed = _target_seed(base_seed, namespace, target_id)
            optimizer_seed = _target_seed(base_seed, f"full_strategy_nsga_seed_{seed}", target_id)
            generation = dict(config["generation"])
            generation["seed"] = sample_seed
            generation["diag_prefilter"] = {
                **dict(generation["diag_prefilter"]),
                "reranker_checkpoint": seed_spec["reranker_checkpoint"],
            }
            nsga2 = dict(config["nsga2"])
            nsga2["seed"] = optimizer_seed
            nsga2["warm_start_seed"] = optimizer_seed
            run_dir = output_dir / "runs" / f"seed_{seed}" / target_id[:16]
            run_config = {
                "version": f"{config['version']}__seed_{seed}__{target_id[:16]}",
                "data": dict(config["data"]),
                "target": {
                    "suite_csv": config["target_suite"],
                    "target_id": target_id,
                },
                "model": {
                    "model_id": "cfm_diag_full_strategy",
                    "checkpoint_path": seed_spec["generator_checkpoint"],
                },
                "generation": generation,
                "delivery": dict(config["delivery"]),
                "gradient": dict(config["gradient"]),
                "latent_repair": dict(config["latent_repair"]),
                "nsga2": nsga2,
                "runtime": dict(config["runtime"]),
                "output": {"directory": str(run_dir), "refuse_overwrite": True},
            }
            run_config_path = (
                output_dir / "run_configs" / f"seed_{seed}__{target_id[:16]}.yaml"
            )
            run_config_path.write_text(
                yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8"
            )
            load_diag_cfm_strategy_config(run_config_path)
            run_diag_cfm_strategy_target(
                run_config_path,
                repo_root=REPO_ROOT,
                output_dir_override=run_dir,
            )
            rows.append(_summarize_run(run_dir, seed, target, normalized_columns))

    ledger = pd.DataFrame(rows)
    ledger.to_csv(output_dir / "target_summary.csv", index=False)
    summaries = []
    for seed, frame in ledger.groupby("seed", sort=False):
        summaries.append(_summary_record(f"seed_{seed}", frame))
    summaries.append(_summary_record("pooled_all_seeds", ledger))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "model_summary.csv", index=False)
    write_json(
        output_dir / "provenance.json",
        {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "target_suite": str(target_suite_path),
            "target_suite_sha256": sha256_file(target_suite_path),
            "run_count": len(ledger),
        },
    )
    return output_dir


def _summary_record(label: str, frame: pd.DataFrame) -> dict:
    return {
        "evaluation": label,
        "model": "cfm_diag_gradient_nsga",
        "target_count": len(frame),
        "median_best_total_error": float(frame["best_total_error"].median()),
        "median_top10_error": float(frame["top10_error"].median()),
        "mean_physical_valid_rate": float(frame["valid_rate"].mean()),
        "mean_hit_rate": float(frame["hit_rate"].mean()),
        "target_success_rate": float(frame["target_success"].mean()),
        "mean_diversity": float(frame["diversity"].mean()),
        "mean_generation_seconds": float(frame["generation_seconds"].mean()),
        "mean_direct_0d_seconds": float(frame["direct_0d_seconds"].mean()),
        "mean_gradient_seconds": float(frame["gradient_seconds"].mean()),
        "mean_nsga2_seconds": float(frame["nsga2_seconds"].mean()),
        "total_0d_evaluations": int(frame["total_0d_evaluations"].sum()),
        "mean_0d_evaluations": float(frame["total_0d_evaluations"].mean()),
        "nsga2_trigger_count": int((frame["nsga2_0d_evaluations"] > 0).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--maximum-targets-per-type", type=int, default=None)
    args = parser.parse_args()
    output = run(args.config, args.output_dir, args.maximum_targets_per_type)
    print(json.dumps({"output": str(output), "status": "completed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
