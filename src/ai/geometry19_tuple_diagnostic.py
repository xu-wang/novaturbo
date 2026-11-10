"""Fixed-budget Top-M tuple diagnostic for the trained Geometry19 B arm."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.ai import geometry19_mixed_cfm as _mixed_registration  # noqa: F401
from src.ai.generators import load_generator
from src.ai.geometry19_mixed_cfm import Geometry19TupleCFM
from src.ai.geometry19_mixed_cfm_eval import _top_tuple_labels, _tuple_retrieval_data
from src.ai.geometry19_generator_oracle import _filter_samples, _physics_worker, _read_jsonl, _worker_init, _write_json, _write_jsonl
from src.ai.geometry19_inverse_contract import Geometry19InverseTarget, load_wp19_23_contract
from src.physics import ModelConfig


VERSION = "wp19_23_b_tuple_top_m_diagnostic_v1"


def _aggregate(runs: list[dict[str, Any]], top_m: int) -> dict[str, Any]:
    rows = [row for row in runs if row["top_m"] == top_m]
    finite = [row["best_true_rho"] for row in rows if row["best_true_rho"] is not None]
    targets = sorted({row["target_id"] for row in rows})
    counts = {target: sum(row["oracle_hit"] for row in rows if row["target_id"] == target) for target in targets}
    return {
        "hit_rate": float(np.mean([row["oracle_hit"] for row in rows])),
        "median_best_rho": float(np.median(finite)), "p90_best_rho": float(np.percentile(finite, 90)),
        "coverage": float(np.mean([value >= 1 for value in counts.values()])),
        "stable": float(np.mean([value >= 2 for value in counts.values()])),
        "mean_physics_acceptable": float(np.mean([row["physics_acceptable"] for row in rows])),
    }


def run_tuple_diagnostic(root: str | Path, config_path: str | Path, *, device: str | None = None) -> dict[str, Any]:
    repo = Path(root).resolve()
    config = yaml.safe_load(Path(config_path).resolve().read_text(encoding="utf-8"))
    section = config["tuple_diagnostic"]
    runtime = repo / section["runtime_root"]
    if runtime.exists():
        raise ValueError(f"Refusing to overwrite tuple diagnostic: {runtime}")
    runtime.mkdir(parents=True)
    confirmation_root = repo / config["checkpoint_selection"]["runtime_root"]
    targets = _read_jsonl(confirmation_root / "new_locked_confirmation_targets.jsonl")
    arm = next(row for row in config["arms"] if row["arm_id"] == "b_tuple")
    checkpoint = repo / arm["artifacts"]["weights_root"] / "full400" / "best.pt"
    model = load_generator(checkpoint, device=device or section["device"])
    if not isinstance(model, Geometry19TupleCFM):
        raise TypeError("B checkpoint is not Geometry19TupleCFM")
    train_conditions, train_labels = _tuple_retrieval_data(repo, config, model.normalizer)
    contract = load_wp19_23_contract(repo / "config/physics/wp19_23_inverse_closure_v1.yaml")
    schema = yaml.safe_load((repo / contract.payload["contracts"]["design_schema"]).read_text(encoding="utf-8"))
    physics_model = ModelConfig.from_yaml(repo / contract.payload["contracts"]["calibration_pack"])
    top_values = [int(value) for value in section["top_m"]]
    seeds = [int(value) for value in section["source_seeds"]]
    count = int(section["fixed_total_candidates"])
    runs = []
    with ProcessPoolExecutor(max_workers=int(section["physics_workers"]), initializer=_worker_init, initargs=(str(repo),)) as executor:
        for top_m in top_values:
            for target_row in targets:
                target = Geometry19InverseTarget.from_mapping(target_row["target"])
                physical = np.asarray([target.thrust_N, target.tsfc_kg_N_s, target.mass_kg])
                condition = torch.as_tensor(model.normalizer.transform_conditions(physical), dtype=model.dtype, device=model.device)
                labels = _top_tuple_labels(np.asarray(condition.cpu()), train_conditions, train_labels, top_m, count)
                for seed in seeds:
                    samples = model.sample_with_labels(condition, torch.as_tensor(labels, dtype=torch.long), seed).numpy()
                    candidates, tasks = _filter_samples(samples, f"b_top{top_m}", target.target_id, seed, schema, physics_model)
                    for task in tasks:
                        task["target"] = target.as_mapping()
                    physics = list(executor.map(_physics_worker, tasks, chunksize=32))
                    rhos = [float(row["hit_alignment"]["rho"]) for row in physics if row.get("hit_alignment")]
                    runs.append({
                        "top_m": top_m, "target_id": target.target_id, "target_region": target_row["target_region"],
                        "source_seed": seed, "oracle_hit": bool(rhos and min(rhos) <= 1.0),
                        "best_true_rho": min(rhos) if rhos else None, "physics_acceptable": len(rhos),
                    })
    aggregate = {str(value): _aggregate(runs, value) for value in top_values}
    reference = aggregate[str(section["reference_top_m"])]
    widest = aggregate[str(max(top_values))]
    passed = (
        widest["hit_rate"] >= reference["hit_rate"] + float(section["minimum_hit_rate_gain"])
        or widest["median_best_rho"] <= reference["median_best_rho"] * (1.0 - float(section["minimum_median_rho_reduction_fraction"]))
    )
    summary = {
        "protocol_version": VERSION, "status": "COMPLETE", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixed_total_candidates": count, "top_m": top_values, "aggregate": aggregate,
        "fixed_per_tuple_gate": "PASS" if passed else "FAIL",
    }
    _write_json(runtime / "summary.json", summary)
    _write_jsonl(runtime / "run_ledger.jsonl", runs)
    lines = ["# B方案Top-M Tuple诊断", "", "| Top-M | Hit@512 | median rho | P90 rho | coverage | stable | acceptable |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for value in top_values:
        row = aggregate[str(value)]
        lines.append(f"| {value} | {row['hit_rate']:.1%} | {row['median_best_rho']:.3f} | {row['p90_best_rho']:.3f} | {row['coverage']:.1%} | {row['stable']:.1%} | {row['mean_physics_acceptable']:.1f} |")
    lines += ["", f"- 固定每tuple预算扩展门槛：**{summary['fixed_per_tuple_gate']}**。", ""]
    (runtime / "Tuple_TopM_Report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary

