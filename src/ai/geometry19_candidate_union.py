"""Equal-budget Oracle test for baseline-projection plus selected C candidates."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from src.ai import geometry19_mixed_cfm as _mixed_registration  # noqa: F401
from src.ai.generators import load_generator
from src.ai.geometry19_candidates import sample_device_invariant_sources
from src.ai.geometry19_cfm import integrate_cfm_sources
from src.ai.geometry19_generator_failure import _map_candidates
from src.ai.geometry19_generator_oracle import _filter_samples, _physics_worker, _read_jsonl, _worker_init, _write_json, _write_jsonl
from src.ai.geometry19_inverse_contract import Geometry19InverseTarget, load_wp19_23_contract
from src.physics import ModelConfig


VERSION = "wp19_23_baseline_c_equal_budget_union_v1"


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    finite = [row["best_true_rho"] for row in runs if row["best_true_rho"] is not None]
    targets = sorted({row["target_id"] for row in runs})
    counts = {target: sum(row["oracle_hit"] for row in runs if row["target_id"] == target) for target in targets}
    return {
        "runs": len(runs), "hit_rate": float(np.mean([row["oracle_hit"] for row in runs])),
        "median_best_rho": float(np.median(finite)), "p90_best_rho": float(np.percentile(finite, 90)),
        "any_seed_target_coverage": float(np.mean([value >= 1 for value in counts.values()])),
        "stable_target_fraction": float(np.mean([value >= 2 for value in counts.values()])),
        "mean_physics_acceptable": float(np.mean([row["physics_acceptable"] for row in runs])),
    }


def run_equal_budget_union(root: str | Path, config_path: str | Path, *, device: str | None = None) -> dict[str, Any]:
    repo = Path(root).resolve()
    config = yaml.safe_load(Path(config_path).resolve().read_text(encoding="utf-8"))
    section = config["checkpoint_selection"]
    selection_root = repo / section["runtime_root"]
    runtime = repo / config["candidate_union"]["runtime_root"]
    if runtime.exists():
        raise ValueError(f"Refusing to overwrite union runtime: {runtime}")
    runtime.mkdir(parents=True)
    selection = json.loads((selection_root / "summary.json").read_text(encoding="utf-8"))
    confirmation = json.loads((selection_root / "confirmation_summary.json").read_text(encoding="utf-8"))
    targets = _read_jsonl(selection_root / "new_locked_confirmation_targets.jsonl")
    baseline = load_generator(repo / section["baseline_checkpoint"], device=device or section["device"])
    c_model = load_generator(repo / selection["selected"]["checkpoint"], device=device or section["device"])
    contract = load_wp19_23_contract(repo / "config/physics/wp19_23_inverse_closure_v1.yaml")
    schema = yaml.safe_load((repo / contract.payload["contracts"]["design_schema"]).read_text(encoding="utf-8"))
    physics_model = ModelConfig.from_yaml(repo / contract.payload["contracts"]["calibration_pack"])
    per_arm = int(config["candidate_union"]["candidates_per_arm"])
    seeds = [int(value) for value in section["confirmation_source_seeds"]]
    runs = []
    with ProcessPoolExecutor(max_workers=int(section["physics_workers"]), initializer=_worker_init, initargs=(str(repo),)) as executor:
        for target_row in targets:
            target = Geometry19InverseTarget.from_mapping(target_row["target"])
            physical = np.asarray([target.thrust_N, target.tsfc_kg_N_s, target.mass_kg])
            base_condition = torch.as_tensor(baseline.normalizer.transform_conditions(physical), dtype=baseline.dtype, device=baseline.device)
            c_condition = torch.as_tensor(c_model.normalizer.transform_conditions(physical), dtype=c_model.dtype, device=c_model.device)
            for seed in seeds:
                sources = sample_device_invariant_sources(baseline, per_arm, seed)
                with torch.no_grad():
                    base_samples = integrate_cfm_sources(baseline, sources, base_condition).cpu().numpy()
                _, base_tasks, _ = _map_candidates(base_samples, schema, physics_model, "nearest_integer", target_row["reference_x_19"], per_arm)
                c_samples = c_model.sample(c_condition, per_arm, seed=seed, space="normalized").values.numpy()
                _, c_tasks = _filter_samples(c_samples, "c_selected", target.target_id, seed, schema, physics_model)
                unique = {}
                origin = {}
                for label, tasks in (("baseline", base_tasks), ("c", c_tasks)):
                    for task in tasks:
                        design_hash = task["design_hash"]
                        if design_hash not in unique:
                            unique[design_hash] = task["geometry19"]
                            origin[design_hash] = label
                run_id = f"union:{target.target_id}:{seed}"
                tasks = [
                    {"run_id": run_id, "candidate_id": index, "geometry19": geometry,
                     "design_hash": design_hash, "target": target.as_mapping()}
                    for index, (design_hash, geometry) in enumerate(unique.items())
                ]
                physics = list(executor.map(_physics_worker, tasks, chunksize=32))
                rhos = [float(row["hit_alignment"]["rho"]) for row in physics if row.get("hit_alignment")]
                best = min(physics, key=lambda row: row["hit_alignment"]["rho"] if row.get("hit_alignment") else float("inf"), default=None)
                runs.append({
                    "target_id": target.target_id, "target_region": target_row["target_region"], "source_seed": seed,
                    "raw_budget": 2 * per_arm, "unique_candidates": len(tasks),
                    "physics_acceptable": len(rhos), "oracle_hit": bool(rhos and min(rhos) <= 1.0),
                    "best_true_rho": min(rhos) if rhos else None,
                    "best_origin": origin.get(best["design_hash"]) if best and best.get("design_hash") else None,
                })
    union = _aggregate(runs)
    base = confirmation["aggregate"]["baseline_projection"]
    passed = (
        union["hit_rate"] > base["hit_rate"]
        or (union["hit_rate"] == base["hit_rate"] and union["p90_best_rho"] < base["p90_best_rho"] and union["any_seed_target_coverage"] >= base["any_seed_target_coverage"])
    )
    summary = {
        "protocol_version": VERSION, "status": "COMPLETE", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_c_epoch": selection["selected"]["epoch"], "target_count": len(targets), "source_seeds": seeds,
        "arms": {"baseline_projection_512": base, "baseline256_plus_c256": union},
        "best_origin_counts": {label: sum(row["best_origin"] == label for row in runs) for label in ("baseline", "c")},
        "gate_for_reranker": "PASS" if passed else "FAIL",
    }
    _write_json(runtime / "summary.json", summary)
    _write_jsonl(runtime / "run_ledger.jsonl", runs)
    report = [
        "# Baseline+C等预算联合Oracle", "", f"> C checkpoint: epoch {selection['selected']['epoch']}", "",
        "| Arm | Hit@512 | median rho | P90 rho | coverage | stable | acceptable |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in summary["arms"].items():
        report.append(f"| {label} | {row['hit_rate']:.1%} | {row['median_best_rho']:.3f} | {row['p90_best_rho']:.3f} | {row['any_seed_target_coverage']:.1%} | {row['stable_target_fraction']:.1%} | {row['mean_physics_acceptable']:.1f} |")
    report += ["", f"- 联合池最佳候选来源：{summary['best_origin_counts']}。", f"- reranker门槛：**{summary['gate_for_reranker']}**。", ""]
    (runtime / "Equal_Budget_Union_Report.md").write_text("\n".join(report), encoding="utf-8")
    return summary

