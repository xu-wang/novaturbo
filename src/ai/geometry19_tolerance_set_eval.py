"""Checkpoint selection and blind Oracle evaluation for 19D tolerance-set CFM."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import yaml

from src.ai.geometry19_generator_oracle import (
    _condition_features,
    _performance,
    _read_jsonl,
    _select_split,
    _write_json,
    _write_jsonl,
)
from src.ai.geometry19_performance_consistency import (
    _aggregate,
    _evaluate,
    _selection_key,
)
from src.ai.geometry19_pipeline import build_target_mapping


VERSION = "wp19_23_tolerance_set_ablation_v1"
REPLICATION_VERSION = "wp19_23_tolerance_set_seed_replication_v1"


def _freeze_fresh_targets(
    repo: Path, config: Mapping[str, Any], output: Path
) -> list[dict[str, Any]]:
    if output.is_file():
        return _read_jsonl(output)
    used = {
        int(row["sample_id"])
        for relative in config["data"]["excluded_target_files"]
        for row in _read_jsonl(repo / relative)
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
                "split": "tolerance_set_locked_confirmation",
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


def _decision(
    runs: list[dict[str, Any]], aggregate: Mapping[str, Mapping[str, Any]], gates: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    baseline_id = "t0_frozen_baseline"
    baseline = aggregate[baseline_id]
    baseline_runs = {
        (row["target_id"], int(row["source_seed"])): row
        for row in runs
        if row["arm_id"] == baseline_id
    }
    result: dict[str, dict[str, Any]] = {}
    for arm_id, current in aggregate.items():
        if arm_id == baseline_id:
            continue
        paired = [
            (baseline_runs[(row["target_id"], int(row["source_seed"]))], row)
            for row in runs
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
            "p90_no_regression": current["p90_best_rho"] <= baseline["p90_best_rho"]
            if bool(gates["require_p90_no_regression"])
            else True,
            "net_positive_paired_hits": wins > losses
            if bool(gates["require_net_positive_paired_hits"])
            else True,
        }
        result[arm_id] = {
            "passed": all(checks.values()),
            "hit_rate_delta_pp": 100.0 * hit_delta,
            "median_reduction_fraction": median_reduction,
            "paired_wins": wins,
            "paired_losses": losses,
            "checks": checks,
        }
    return result


def _render(summary: Mapping[str, Any]) -> str:
    lines = [
        "# WP19-23 19D容差集合感知CFM消融",
        "",
        f"> 日期：{summary['generated_at_utc']}",
        "> 范围：仅19D bounded-sigmoid CFM；direct-only、无条件最近整数投影、V2.1 Oracle。",
        "> 未使用10D结果、性能代理损失、D-Flow、reranker、gradient refinement或NSGA-II。",
        "",
        "| Arm | 容差幅度 | checkpoint | Hit@512 | median rho | P90 rho | coverage | stable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm_id in summary["arm_ids"]:
        spec = summary["arms"][arm_id]
        row = summary["confirmation"]["aggregate"][arm_id]
        lines.append(
            f"| {arm_id} | {spec['maximum_fraction']:.1f} | "
            f"{spec.get('selected_epoch', 'frozen')} | {row['hit_rate']:.1%} | "
            f"{row['median_best_rho']:.3f} | {row['p90_best_rho']:.3f} | "
            f"{row['any_seed_target_coverage']:.1%} | {row['stable_target_fraction']:.1%} |"
        )
    lines += ["", "## Gate", ""]
    for arm_id, row in summary["decision"].items():
        lines.append(
            f"- {arm_id}: **{'PASS' if row['passed'] else 'FAIL'}**；"
            f"Hit变化 {row['hit_rate_delta_pp']:+.1f}pp，median降低 "
            f"{row['median_reduction_fraction']:.1%}，新增/丢失命中 "
            f"{row['paired_wins']}/{row['paired_losses']}。"
        )
    lines += ["", f"最终判定：**{summary['status']}**。", ""]
    if summary.get("replication") is not None:
        replication = summary["replication"]
        lines += [
            "## 训练seed复现判定",
            "",
            f"- median Hit增益：{replication['median_hit_rate_delta_pp']:+.1f}pp；",
            f"- 正增益seed数：{replication['positive_hit_gain_count']}/{replication['seed_arm_count']}；",
            f"- 复现门槛：**{'PASS' if replication['reproduced'] else 'FAIL'}**；",
            f"- 默认替换门槛：**{'PASS' if replication['promotion_passed'] else 'FAIL'}**。",
            "",
        ]
        if not replication["reproduced"]:
            lines += [
                "复现失败表示当前全容差训练对训练随机seed不稳定；不得用单一成功seed替换默认模型。",
                "",
            ]
    return "\n".join(lines)


def _replication_decision(
    decisions: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    arm_ids = [str(value) for value in config["arm_ids"]]
    selected = [decisions[arm_id] for arm_id in arm_ids]
    hit_deltas = [float(row["hit_rate_delta_pp"]) / 100.0 for row in selected]
    positive = sum(value > 0.0 for value in hit_deltas)
    lower_median = sum(
        float(row["median_reduction_fraction"]) > 0.0 for row in selected
    )
    p90_ok = sum(bool(row["checks"]["p90_no_regression"]) for row in selected)
    net_positive = sum(
        bool(row["checks"]["net_positive_paired_hits"]) for row in selected
    )
    strict_passes = sum(bool(row["passed"]) for row in selected)
    checks = {
        "all_seed_hit_gains_positive": positive == len(selected),
        "median_hit_gain": float(np.median(hit_deltas))
        >= float(config["minimum_median_hit_rate_gain"]),
        "majority_lower_median": lower_median
        >= int(config["minimum_seed_arms_with_lower_median"]),
        "majority_p90_no_regression": p90_ok
        >= int(config["minimum_seed_arms_with_p90_no_regression"]),
        "majority_net_positive_hits": net_positive
        >= int(config["minimum_seed_arms_with_net_positive_hits"]),
    }
    return {
        "seed_arm_count": len(selected),
        "positive_hit_gain_count": positive,
        "lower_median_count": lower_median,
        "p90_no_regression_count": p90_ok,
        "net_positive_hit_count": net_positive,
        "strict_pass_count": strict_passes,
        "median_hit_rate_delta_pp": 100.0 * float(np.median(hit_deltas)),
        "checks": checks,
        "reproduced": all(checks.values()),
        "promotion_passed": strict_passes
        >= int(config["minimum_strict_passes_for_promotion"]),
    }


def run_tolerance_set_ablation(
    root: str | Path, config_path: str | Path, *, device: str | None = None
) -> dict[str, Any]:
    repo = Path(root).resolve()
    source = Path(config_path).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    protocol_version = str(config.get("protocol_version"))
    if protocol_version not in {VERSION, REPLICATION_VERSION}:
        raise ValueError("Tolerance-set ablation protocol mismatch")
    runtime = repo / config["artifacts"]["runtime_root"]
    if runtime.exists():
        raise ValueError(f"Refusing to overwrite tolerance-set run: {runtime}")
    runtime.mkdir(parents=True)
    started = perf_counter()
    execution_device = device or str(config["execution"]["device"])
    workers = int(config["execution"]["physics_workers"])
    chunksize = int(config["execution"]["physics_chunksize"])

    suite = _read_jsonl(repo / config["data"]["development_target_suite"])
    pool = [row for row in suite if row["split"] == "development"]
    requested = int(config["checkpoint_selection"]["development_target_count"])
    per_region, remainder = divmod(requested, 3)
    development = []
    for index, region in enumerate(("dense", "mid", "sparse")):
        take = per_region + int(index < remainder)
        rows = [row for row in pool if row["target_region"] == region]
        if len(rows) < take:
            raise ValueError(f"Incomplete development stratum: {region}")
        development.extend(rows[:take])

    arms: dict[str, dict[str, Any]] = {}
    development_runs: list[dict[str, Any]] = []
    selection = config["checkpoint_selection"]
    for raw_spec in config["arms"]:
        spec = dict(raw_spec)
        arm_id = str(spec["arm_id"])
        if "checkpoint" in spec:
            arms[arm_id] = spec
            continue
        weights_root = repo / spec["weights_root"]
        history = json.loads((weights_root / "training_history.json").read_text(encoding="utf-8"))
        val_loss = {
            int(row["epoch"]): float(row["val"]["loss"])
            for row in history["history"]
        }
        snapshots = []
        for epoch in selection["snapshot_epochs"]:
            checkpoint = weights_root / "snapshots" / f"epoch_{int(epoch):04d}.pt"
            if not checkpoint.is_file():
                raise ValueError(f"Missing checkpoint snapshot: {checkpoint}")
            runs, _ = _evaluate(
                repo,
                checkpoint,
                arm_id,
                development,
                [int(value) for value in selection["source_seeds"]],
                int(selection["candidates"]),
                execution_device,
                workers,
                chunksize,
            )
            aggregate = _aggregate(runs, arm_id)
            row = {
                "arm_id": arm_id,
                "epoch": int(epoch),
                "checkpoint": str(checkpoint.relative_to(repo)),
                "validation_loss": val_loss[int(epoch)],
                **aggregate,
            }
            snapshots.append(row)
            development_runs.extend({"epoch": int(epoch), **run} for run in runs)
        selected = max(snapshots, key=_selection_key)
        arms[arm_id] = {
            **spec,
            "selected_epoch": selected["epoch"],
            "checkpoint": selected["checkpoint"],
            "development": snapshots,
        }

    confirmation_path = runtime / "locked_confirmation_targets.jsonl"
    confirmation = _freeze_fresh_targets(repo, config, confirmation_path)
    all_runs: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    confirmation_config = config["confirmation"]
    for arm_id, spec in arms.items():
        runs, candidates = _evaluate(
            repo,
            repo / spec["checkpoint"],
            arm_id,
            confirmation,
            [int(value) for value in confirmation_config["source_seeds"]],
            int(confirmation_config["candidates"]),
            execution_device,
            workers,
            chunksize,
        )
        all_runs.extend(runs)
        all_candidates.extend(candidates)
    aggregate = {arm_id: _aggregate(all_runs, arm_id) for arm_id in arms}
    decision = _decision(all_runs, aggregate, config["gates"])
    passed = [arm_id for arm_id, row in decision.items() if row["passed"]]
    replication = (
        _replication_decision(decision, config["replication"])
        if "replication" in config
        else None
    )
    if replication is None:
        status = "PASS" if passed else "NO_ARM_PASSED"
    elif replication["promotion_passed"]:
        status = "PROMOTION_PASS"
    elif replication["reproduced"]:
        status = "REPLICATION_PASS_PROMOTION_FAIL"
    else:
        status = "REPLICATION_FAIL"
    summary = {
        "protocol_version": protocol_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": perf_counter() - started,
        "status": status,
        "passing_arms": passed,
        "arm_ids": list(arms),
        "arms": arms,
        "confirmation_target_count": len(confirmation),
        "confirmation": {"aggregate": aggregate},
        "decision": decision,
        "replication": replication,
    }
    _write_jsonl(runtime / "development_run_ledger.jsonl", development_runs)
    _write_jsonl(runtime / "confirmation_run_ledger.jsonl", all_runs)
    _write_jsonl(runtime / "confirmation_candidate_ledger.jsonl", all_candidates)
    _write_json(runtime / "summary.json", summary)
    report = repo / config["artifacts"]["report"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_render(summary), encoding="utf-8")
    return summary
