"""Counterfactual condition-use diagnostic for the frozen 19D CFM."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from src.ai.generators import load_generator
from src.ai.geometry19_generator_oracle import (
    _filter_samples,
    _physics_worker,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    _worker_init,
)
from src.ai.geometry19_inverse_contract import Geometry19InverseTarget, load_wp19_23_contract
from src.physics import ModelConfig


VERSION = "wp19_23_condition_counterfactual_v1"


def _balanced_targets(
    rows: Sequence[Mapping[str, Any]], split: str, regions: Sequence[str], count: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    per_region, remainder = divmod(count, len(regions))
    for index, region in enumerate(regions):
        take = per_region + int(index < remainder)
        available = [
            dict(row)
            for row in rows
            if row["split"] == split and row["target_region"] == region
        ]
        if len(available) < take:
            raise ValueError(f"Incomplete target stratum: {region}")
        selected.extend(available[:take])
    if len(selected) != count:
        raise ValueError("Balanced target selection is incomplete")
    return selected


def _aggregate(runs: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    selected = [row for row in runs if row["condition_mode"] == mode]
    finite = [float(row["best_true_rho"]) for row in selected if row["best_true_rho"] is not None]
    if not selected or not finite:
        raise ValueError(f"No finite counterfactual runs for mode {mode}")
    return {
        "runs": len(selected),
        "hit_rate": float(np.mean([row["oracle_hit"] for row in selected])),
        "median_best_rho": float(np.median(finite)),
        "p90_best_rho": float(np.percentile(finite, 90)),
        "mean_physics_acceptable": float(
            np.mean([row["physics_acceptable"] for row in selected])
        ),
        "mean_raw_l2_from_own": float(
            np.mean([row["raw_l2_from_own"] for row in selected])
        ),
        "mean_raw_linf_from_own": float(
            np.mean([row["raw_linf_from_own"] for row in selected])
        ),
    }


def _counterfactual_decision(
    runs: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Mapping[str, Any]], gate: Mapping[str, Any]
) -> dict[str, Any]:
    own = aggregate["own"]
    shuffled = aggregate["shuffled"]
    own_by_key = {
        (row["target_id"], int(row["source_seed"])): row
        for row in runs
        if row["condition_mode"] == "own"
    }
    pairs = [
        (own_by_key[(row["target_id"], int(row["source_seed"]))], row)
        for row in runs
        if row["condition_mode"] == "shuffled"
    ]
    wins = sum(left["oracle_hit"] and not right["oracle_hit"] for left, right in pairs)
    losses = sum(not left["oracle_hit"] and right["oracle_hit"] for left, right in pairs)
    hit_gain = float(own["hit_rate"] - shuffled["hit_rate"])
    checks = {
        "minimum_hit_gain": hit_gain
        >= float(gate["minimum_own_vs_shuffled_hit_gain"]),
        "net_positive_paired_hits": (wins > losses)
        if bool(gate["require_net_positive_paired_hits"])
        else True,
    }
    sufficient = all(checks.values())
    return {
        "condition_utilization_sufficient": sufficient,
        "hit_rate_gain": hit_gain,
        "paired_wins": wins,
        "paired_losses": losses,
        "checks": checks,
        "selected_branch": gate[
            "sufficient_branch" if sufficient else "insufficient_branch"
        ],
    }


def _render(summary: Mapping[str, Any]) -> str:
    lines = [
        "# WP19-23 19D CFM条件反事实诊断",
        "",
        f"> 日期：{summary['generated_at_utc']}",
        "> 范围：冻结19D bounded-sigmoid CFM；direct-only、相同source、无条件最近整数投影、V2.1 Oracle。",
        "> 未使用10D结果、D-Flow、reranker、gradient refinement或NSGA-II。",
        "",
        "| 条件模式 | Hit@512 | median rho | P90 rho | acceptable | raw L2 vs own |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in summary["modes"]:
        row = summary["aggregate"][mode]
        lines.append(
            f"| {mode} | {row['hit_rate']:.1%} | {row['median_best_rho']:.3f} | "
            f"{row['p90_best_rho']:.3f} | {row['mean_physics_acceptable']:.1f} | "
            f"{row['mean_raw_l2_from_own']:.3f} |"
        )
    decision = summary["decision"]
    lines += [
        "",
        "## 冻结判定",
        "",
        f"- own相对shuffled Hit变化：{decision['hit_rate_gain']:+.1%}；",
        f"- 配对胜/负：{decision['paired_wins']}/{decision['paired_losses']}；",
        f"- 条件利用：**{'SUFFICIENT' if decision['condition_utilization_sufficient'] else 'INSUFFICIENT'}**；",
        f"- 下一分支：**{decision['selected_branch']}**。",
        "",
    ]
    return "\n".join(lines)


def run_condition_counterfactual(
    root: str | Path, config_path: str | Path, *, device: str | None = None
) -> dict[str, Any]:
    repo = Path(root).resolve()
    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    if config.get("protocol_version") != VERSION:
        raise ValueError("Condition counterfactual protocol mismatch")
    runtime = repo / config["artifacts"]["runtime_root"]
    if runtime.exists() and bool(config["artifacts"].get("refuse_overwrite", True)):
        raise ValueError(f"Refusing to overwrite condition diagnostic: {runtime}")
    runtime.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    diagnostic = config["diagnostic"]
    modes = [str(value) for value in diagnostic["modes"]]
    if modes != ["own", "shuffled", "train_mean"]:
        raise ValueError("Counterfactual modes must be frozen as own/shuffled/train_mean")
    targets = _balanced_targets(
        _read_jsonl(repo / config["data"]["target_suite"]),
        str(diagnostic["split"]),
        [str(value) for value in diagnostic["balance_regions"]],
        int(diagnostic["target_count"]),
    )
    offset = int(diagnostic["shuffle_offset"])
    if offset <= 0 or offset >= len(targets):
        raise ValueError("shuffle_offset must produce a non-trivial cyclic assignment")

    contract = load_wp19_23_contract(
        repo / "config/physics/wp19_23_inverse_closure_v1.yaml"
    )
    schema = yaml.safe_load(
        (repo / contract.payload["contracts"]["design_schema"]).read_text(encoding="utf-8")
    )
    physics_model = ModelConfig.from_yaml(
        repo / contract.payload["contracts"]["calibration_pack"]
    )
    execution_device = device or str(config["execution"]["device"])
    generator = load_generator(repo / config["model"]["checkpoint"], device=execution_device)
    mean_condition = torch.zeros(
        generator.condition_dim, dtype=generator.dtype, device=generator.device
    )
    seeds = [int(value) for value in diagnostic["source_seeds"]]
    count = int(diagnostic["candidates_per_target"])
    runs: list[dict[str, Any]] = []
    candidates_out: list[dict[str, Any]] = []
    workers = int(config["execution"]["physics_workers"])
    chunksize = int(config["execution"]["physics_chunksize"])
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=(str(repo),)
    ) as executor:
        for target_index, target_row in enumerate(targets):
            target = Geometry19InverseTarget.from_mapping(target_row["target"])
            shuffled_row = targets[(target_index + offset) % len(targets)]
            for seed in seeds:
                own_raw: np.ndarray | None = None
                for mode in modes:
                    condition_row = target_row if mode == "own" else shuffled_row
                    if mode == "train_mean":
                        normalized_condition = mean_condition
                        conditioning_target_id = "train_mean"
                    else:
                        condition_target = Geometry19InverseTarget.from_mapping(
                            condition_row["target"]
                        )
                        physical_condition = np.asarray(
                            [
                                condition_target.thrust_N,
                                condition_target.tsfc_kg_N_s,
                                condition_target.mass_kg,
                            ]
                        )
                        normalized_condition = torch.as_tensor(
                            generator.normalizer.transform_conditions(physical_condition),
                            dtype=generator.dtype,
                            device=generator.device,
                        )
                        conditioning_target_id = condition_target.target_id
                    samples = generator.sample(
                        normalized_condition, count, seed=seed, space="normalized"
                    ).values.numpy()
                    if mode == "own":
                        own_raw = samples.copy()
                    if own_raw is None:
                        raise RuntimeError("own mode must be evaluated before counterfactual modes")
                    delta = samples - own_raw
                    raw_l2 = float(np.mean(np.linalg.norm(delta, axis=1)))
                    raw_linf = float(np.max(np.abs(delta)))
                    filtered, tasks = _filter_samples(
                        samples,
                        mode,
                        target.target_id,
                        seed,
                        schema,
                        physics_model,
                        discrete_tolerance_count=None,
                    )
                    for task in tasks:
                        task["target"] = target.as_mapping()
                    physics = list(
                        executor.map(_physics_worker, tasks, chunksize=chunksize)
                    )
                    by_id = {int(row["candidate_id"]): row for row in physics}
                    rhos: list[float] = []
                    for row in filtered:
                        result = by_id.get(int(row["candidate_id"]))
                        rho = (
                            (result.get("hit_alignment") or {}).get("rho")
                            if result
                            else None
                        )
                        if rho is not None:
                            rhos.append(float(rho))
                        candidates_out.append(
                            {
                                "condition_mode": mode,
                                "target_id": target.target_id,
                                "conditioning_target_id": conditioning_target_id,
                                "source_seed": seed,
                                "candidate_id": int(row["candidate_id"]),
                                "failure_code": row["failure_code"],
                                "rho": rho,
                            }
                        )
                    runs.append(
                        {
                            "condition_mode": mode,
                            "target_id": target.target_id,
                            "target_region": target_row["target_region"],
                            "conditioning_target_id": conditioning_target_id,
                            "source_seed": seed,
                            "oracle_hit": bool(rhos and min(rhos) <= 1.0),
                            "best_true_rho": min(rhos) if rhos else None,
                            "physics_acceptable": len(rhos),
                            "strict_hits": sum(value <= 1.0 for value in rhos),
                            "raw_l2_from_own": raw_l2,
                            "raw_linf_from_own": raw_linf,
                        }
                    )

    aggregate = {mode: _aggregate(runs, mode) for mode in modes}
    decision = _counterfactual_decision(runs, aggregate, config["gate"])
    summary = {
        "protocol_version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": perf_counter() - started,
        "checkpoint": config["model"]["checkpoint"],
        "target_count": len(targets),
        "source_seeds": seeds,
        "candidates_per_target": count,
        "modes": modes,
        "aggregate": aggregate,
        "decision": decision,
    }
    _write_jsonl(runtime / "targets.jsonl", targets)
    _write_jsonl(runtime / "run_ledger.jsonl", runs)
    _write_jsonl(runtime / "candidate_ledger.jsonl", candidates_out)
    _write_json(runtime / "summary.json", summary)
    report = repo / config["artifacts"]["report"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_render(summary), encoding="utf-8")
    return summary
