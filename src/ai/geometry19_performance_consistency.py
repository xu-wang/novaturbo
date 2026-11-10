"""19D bounded-CFM performance-consistency checkpoint and Oracle ablation."""

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
    _condition_features,
    _filter_samples,
    _performance,
    _physics_worker,
    _read_jsonl,
    _select_split,
    _worker_init,
    _write_json,
    _write_jsonl,
)
from src.ai.geometry19_inverse_contract import Geometry19InverseTarget, load_wp19_23_contract
from src.ai.geometry19_pipeline import build_target_mapping
from src.ai.geometry19_surrogate import sha256_file
from src.physics import ModelConfig


VERSION = "wp19_23_performance_consistency_ablation_v1"


def _freeze_confirmation_targets(
    repo: Path, config: Mapping[str, Any], output: Path
) -> list[dict[str, Any]]:
    if output.is_file():
        return _read_jsonl(output)
    used = {
        int(row["sample_id"])
        for path_key in ("existing_target_suite", "prior_confirmation_targets")
        for row in _read_jsonl(repo / config["data"][path_key])
    }
    records = _read_jsonl(repo / config["data"]["regression_dataset"])
    by_campaign: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_campaign.setdefault(str(row["campaign_id"]), []).append(row)
    train_rows = [
        row
        for campaign in config["data"]["train_campaigns"]
        for row in by_campaign[str(campaign)]
    ]
    train_points = _condition_features(
        np.asarray([_performance(row) for row in train_rows])
    )
    available = [
        row
        for row in by_campaign[str(config["data"]["held_out_campaign"])]
        if int(row["sample_id"]) not in used
    ]
    selected = _select_split(
        available,
        train_points,
        int(config["confirmation"]["target_count"]),
        "locked_held_out",
    )
    result = []
    for row in selected:
        performance = row["physics_outputs"]["performance"]
        result.append(
            {
                "split": "performance_consistency_locked_confirmation",
                "target_region": row["target_region"],
                "source_campaign": row["campaign_id"],
                "sample_id": int(row["sample_id"]),
                "train_neighbor_distance": float(row["train_neighbor_distance"]),
                "target": build_target_mapping(
                    thrust_N=performance["thrust_N"],
                    tsfc_kg_N_s=performance["tsfc_kg_N_s"],
                    mass_kg=performance["mass_kg"],
                    target_type="ID",
                    expected_status="KNOWN_FEASIBLE",
                ),
                "reference_x_19": row["x_19"],
            }
        )
    _write_jsonl(output, result)
    return result


def _evaluate(
    repo: Path,
    checkpoint: Path,
    arm_id: str,
    targets: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    count: int,
    device: str,
    workers: int,
    chunksize: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contract = load_wp19_23_contract(
        repo / "config/physics/wp19_23_inverse_closure_v1.yaml"
    )
    schema = yaml.safe_load(
        (repo / contract.payload["contracts"]["design_schema"]).read_text(
            encoding="utf-8"
        )
    )
    physics_model = ModelConfig.from_yaml(
        repo / contract.payload["contracts"]["calibration_pack"]
    )
    generator = load_generator(checkpoint, device=device)
    runs: list[dict[str, Any]] = []
    compact_candidates: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=(str(repo),)
    ) as executor:
        for target_row in targets:
            target = Geometry19InverseTarget.from_mapping(target_row["target"])
            physical_condition = np.asarray(
                [target.thrust_N, target.tsfc_kg_N_s, target.mass_kg]
            )
            normalized_condition = torch.as_tensor(
                generator.normalizer.transform_conditions(physical_condition),
                dtype=generator.dtype,
                device=generator.device,
            )
            for seed in seeds:
                samples = generator.sample(
                    normalized_condition,
                    count,
                    seed=int(seed),
                    space="normalized",
                ).values.numpy()
                candidates, tasks = _filter_samples(
                    samples,
                    arm_id,
                    target.target_id,
                    int(seed),
                    schema,
                    physics_model,
                    discrete_tolerance_count=None,
                )
                for task in tasks:
                    task["target"] = target.as_mapping()
                physics = list(executor.map(_physics_worker, tasks, chunksize=chunksize))
                by_id = {int(row["candidate_id"]): row for row in physics}
                rhos = []
                for row in candidates:
                    result = by_id.get(int(row["candidate_id"]))
                    rho = (result.get("hit_alignment") or {}).get("rho") if result else None
                    if rho is not None:
                        rhos.append(float(rho))
                    compact_candidates.append(
                        {
                            "arm_id": arm_id,
                            "target_id": target.target_id,
                            "target_region": target_row["target_region"],
                            "source_seed": int(seed),
                            "candidate_id": int(row["candidate_id"]),
                            "failure_code": row["failure_code"],
                            "rho": rho,
                        }
                    )
                runs.append(
                    {
                        "arm_id": arm_id,
                        "target_id": target.target_id,
                        "target_region": target_row["target_region"],
                        "source_seed": int(seed),
                        "oracle_hit": bool(rhos and min(rhos) <= 1.0),
                        "best_true_rho": min(rhos) if rhos else None,
                        "physics_acceptable": len(rhos),
                        "strict_hits": sum(value <= 1.0 for value in rhos),
                    }
                )
    return runs, compact_candidates


def _aggregate(runs: Sequence[Mapping[str, Any]], arm_id: str) -> dict[str, Any]:
    selected = [row for row in runs if row["arm_id"] == arm_id]
    finite = [float(row["best_true_rho"]) for row in selected if row["best_true_rho"] is not None]
    targets = sorted({str(row["target_id"]) for row in selected})
    hit_counts = {
        target: sum(bool(row["oracle_hit"]) for row in selected if row["target_id"] == target)
        for target in targets
    }
    return {
        "runs": len(selected),
        "targets": len(targets),
        "hit_rate": float(np.mean([row["oracle_hit"] for row in selected])),
        "median_best_rho": float(np.median(finite)),
        "p90_best_rho": float(np.percentile(finite, 90)),
        "mean_physics_acceptable": float(np.mean([row["physics_acceptable"] for row in selected])),
        "any_seed_target_coverage": float(np.mean([value >= 1 for value in hit_counts.values()])),
        "stable_target_fraction": float(np.mean([value >= 2 for value in hit_counts.values()])),
    }


def _selection_key(row: Mapping[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(row["hit_rate"]),
        -float(row["p90_best_rho"]),
        -float(row["median_best_rho"]),
        -int(row["epoch"]),
    )


def _render(summary: Mapping[str, Any]) -> str:
    lines = [
        "# WP19-23 19D CFM性能一致性损失消融",
        "",
        f"> 日期：{summary['generated_at_utc']}",
        "> 范围：仅19D bounded-sigmoid CFM；direct-only、无条件最近整数投影、V2.1 Oracle评价。",
        "> 未使用10D结果、D-Flow、reranker、gradient refinement或NSGA-II。",
        "",
        "## 锁定确认结果",
        "",
        "| Arm | lambda | checkpoint | Hit@512 | median rho | P90 rho | coverage | stable | acceptable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in summary["arm_ids"]:
        row = summary["confirmation"]["aggregate"][arm]
        spec = summary["arms"][arm]
        lines.append(
            f"| {arm} | {spec['weight']:.2f} | {spec.get('selected_epoch', 'frozen')} | "
            f"{row['hit_rate']:.1%} | {row['median_best_rho']:.3f} | {row['p90_best_rho']:.3f} | "
            f"{row['any_seed_target_coverage']:.1%} | {row['stable_target_fraction']:.1%} | "
            f"{row['mean_physics_acceptable']:.1f} |"
        )
    lines += ["", "## Gate", ""]
    for arm, decision in summary["decision"].items():
        lines.append(
            f"- {arm}: **{'PASS' if decision['passed'] else 'FAIL'}**；"
            f"Hit变化 {decision['hit_rate_delta_pp']:+.1f}pp，"
            f"median降低 {decision['median_reduction_fraction']:.1%}，"
            f"新增/丢失命中 {decision['paired_wins']}/{decision['paired_losses']}。"
        )
    lines += ["", f"最终判定：**{summary['status']}**。", ""]
    return "\n".join(lines)


def run_performance_consistency_ablation(
    root: str | Path, config_path: str | Path, *, device: str | None = None
) -> dict[str, Any]:
    repo = Path(root).resolve()
    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    if config.get("protocol_version") != VERSION:
        raise ValueError("Performance-consistency ablation protocol mismatch")
    runtime = repo / config["artifacts"]["runtime_root"]
    if runtime.exists():
        raise ValueError(f"Refusing to overwrite performance-consistency run: {runtime}")
    runtime.mkdir(parents=True)
    started = perf_counter()
    execution_device = device or config["execution"]["device"]
    workers = int(config["execution"]["physics_workers"])
    chunksize = int(config["execution"]["physics_chunksize"])
    suite = _read_jsonl(repo / config["data"]["existing_target_suite"])
    development_pool = [row for row in suite if row["split"] == "development"]
    requested_development = int(
        config["checkpoint_selection"]["development_target_count"]
    )
    per_region, remainder = divmod(requested_development, 3)
    development = []
    for region_index, region in enumerate(("dense", "mid", "sparse")):
        take = per_region + int(region_index < remainder)
        region_rows = [
            row for row in development_pool if row["target_region"] == region
        ]
        if len(region_rows) < take:
            raise ValueError(f"Development target stratum is incomplete: {region}")
        development.extend(region_rows[:take])
    if len(development) != requested_development:
        raise ValueError("Development target strata are incomplete")
    development_seeds = [int(value) for value in config["checkpoint_selection"]["source_seeds"]]
    development_count = int(config["checkpoint_selection"]["candidates"])

    arms: dict[str, dict[str, Any]] = {}
    development_rows: list[dict[str, Any]] = []
    for spec_value in config["arms"]:
        spec = dict(spec_value)
        arm_id = str(spec["arm_id"])
        if "checkpoint" in spec:
            checkpoint = repo / spec["checkpoint"]
            arms[arm_id] = {**spec, "checkpoint": str(checkpoint.relative_to(repo))}
            continue
        weights_root = repo / spec["weights_root"]
        history = json.loads((weights_root / "training_history.json").read_text(encoding="utf-8"))
        validation_loss = {
            int(row["epoch"]): float(row["val"]["loss"])
            for row in history["history"]
        }
        snapshots = []
        for epoch in config["checkpoint_selection"]["snapshot_epochs"]:
            checkpoint = weights_root / "snapshots" / f"epoch_{int(epoch):04d}.pt"
            if not checkpoint.is_file():
                raise ValueError(f"Missing checkpoint snapshot: {checkpoint}")
            runs, _ = _evaluate(
                repo,
                checkpoint,
                arm_id,
                development,
                development_seeds,
                development_count,
                execution_device,
                workers,
                chunksize,
            )
            aggregate = _aggregate(runs, arm_id)
            row = {
                "arm_id": arm_id,
                "epoch": int(epoch),
                "checkpoint": str(checkpoint.relative_to(repo)),
                "validation_loss": validation_loss[int(epoch)],
                **aggregate,
            }
            snapshots.append(row)
            development_rows.extend({"epoch": int(epoch), **run} for run in runs)
        selected = max(snapshots, key=_selection_key)
        arms[arm_id] = {
            **spec,
            "selected_epoch": selected["epoch"],
            "checkpoint": selected["checkpoint"],
            "development": snapshots,
        }

    confirmation_path = runtime / "locked_confirmation_targets.jsonl"
    confirmation = _freeze_confirmation_targets(repo, config, confirmation_path)
    all_runs: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    confirmation_seeds = [int(value) for value in config["confirmation"]["source_seeds"]]
    confirmation_count = int(config["confirmation"]["candidates"])
    for arm_id, spec in arms.items():
        runs, candidates = _evaluate(
            repo,
            repo / spec["checkpoint"],
            arm_id,
            confirmation,
            confirmation_seeds,
            confirmation_count,
            execution_device,
            workers,
            chunksize,
        )
        all_runs.extend(runs)
        all_candidates.extend(candidates)
    aggregate = {arm: _aggregate(all_runs, arm) for arm in arms}
    baseline = aggregate["g0_frozen_fm"]
    baseline_runs = {
        (row["target_id"], row["source_seed"]): row
        for row in all_runs
        if row["arm_id"] == "g0_frozen_fm"
    }
    decisions = {}
    gates = config["gates"]
    for arm_id in arms:
        if arm_id == "g0_frozen_fm":
            continue
        current = aggregate[arm_id]
        paired = [
            (baseline_runs[(row["target_id"], row["source_seed"])], row)
            for row in all_runs
            if row["arm_id"] == arm_id
        ]
        wins = sum(not left["oracle_hit"] and right["oracle_hit"] for left, right in paired)
        losses = sum(left["oracle_hit"] and not right["oracle_hit"] for left, right in paired)
        hit_delta = float(current["hit_rate"] - baseline["hit_rate"])
        median_reduction = float(
            (baseline["median_best_rho"] - current["median_best_rho"])
            / baseline["median_best_rho"]
        )
        checks = {
            "hit_rate_gain": hit_delta >= float(gates["minimum_hit_rate_gain"]),
            "median_reduction": median_reduction
            >= float(gates["minimum_median_rho_reduction_fraction"]),
            "p90_no_regression": current["p90_best_rho"] <= baseline["p90_best_rho"],
            "net_positive_paired_hits": wins > losses,
        }
        decisions[arm_id] = {
            "passed": all(checks.values()),
            "checks": checks,
            "hit_rate_delta_pp": 100.0 * hit_delta,
            "median_reduction_fraction": median_reduction,
            "paired_wins": wins,
            "paired_losses": losses,
        }
    summary = {
        "protocol_version": VERSION,
        "status": "PASS" if any(row["passed"] for row in decisions.values()) else "NO_ARM_PASSED",
        "formal_use_allowed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "geometry19_19d_only",
        "arm_ids": list(arms),
        "arms": arms,
        "development_target_count": len(development),
        "confirmation_target_count": len(confirmation),
        "confirmation": {"aggregate": aggregate, "source_seeds": confirmation_seeds, "candidates": confirmation_count},
        "decision": decisions,
        "excluded": ["10d", "dflow", "reranker", "gradient_refinement", "nsga2"],
        "elapsed_seconds": perf_counter() - started,
        "hashes": {
            "config": sha256_file(config_source),
            "confirmation_targets": sha256_file(confirmation_path),
        },
    }
    _write_jsonl(runtime / "development_run_ledger.jsonl", development_rows)
    _write_jsonl(runtime / "confirmation_run_ledger.jsonl", all_runs)
    _write_jsonl(runtime / "confirmation_candidate_ledger.jsonl", all_candidates)
    _write_json(runtime / "summary.json", summary)
    report = repo / config["artifacts"]["report"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_render(summary), encoding="utf-8")
    return summary
