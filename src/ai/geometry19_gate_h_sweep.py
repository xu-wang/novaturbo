"""Budget sweep and exhaustive truth audit for the frozen Gate H pools."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from src.ai.geometry19_gate_h import (
    _arm_metrics,
    _write_json,
    _write_jsonl,
    select_tolerance_ranked_diverse,
)
from src.ai.geometry19_inverse_contract import load_target, load_wp19_23_contract
from src.ai.geometry19_physics import evaluate_geometry19_candidate
from src.ai.geometry19_selection import ConstraintDominanceContract
from src.ai.geometry19_surrogate import sha256_file
from src.physics import ConstraintProfile, Environment, ModelConfig


SWEEP_VERSION = "wp19_23_gate_h_budget_sweep_v1"
STRATEGIES = ("predicted_rho", "ranked_diverse")


class Geometry19GateHSweepError(ValueError):
    """Raised when the Gate H budget-sweep contract is violated."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def select_predicted_rho(
    rows: Sequence[Mapping[str, Any]], budget: int
) -> list[dict[str, Any]]:
    """Return a nested prefix ordered by tolerance-normalized surrogate error."""

    if budget <= 0 or budget > len(rows):
        raise Geometry19GateHSweepError("Invalid predicted-rho budget")
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (float(row["predicted_rho"]), row["candidate_key"]),
    )[:budget]


def _load_reusable_evaluations(target_root: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for path in sorted(target_root.glob("h*/physics_evaluations.jsonl")):
        for row in _read_jsonl(path):
            design_hash = str(row["design_hash"])
            previous = cache.get(design_hash)
            if previous is None:
                cache[design_hash] = row
                continue
            if previous.get("performance") != row.get("performance"):
                raise Geometry19GateHSweepError(
                    f"Conflicting cached V2.1 evaluation for {design_hash}"
                )
    return cache


def _metrics_for_selection(
    selected: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Mapping[str, Any]],
    tolerances: Mapping[str, float],
) -> dict[str, Any]:
    evaluations = [cache[str(row["design_hash"])] for row in selected]
    metrics = _arm_metrics(selected, evaluations, tolerances)
    metrics.pop("selected_design_hashes")
    return metrics


def _aggregate(
    target_runs: Sequence[Mapping[str, Any]], strategy: str, budget: int
) -> dict[str, Any]:
    rows = [run["strategies"][strategy][str(budget)] for run in target_runs]
    rho = [row["best_true_rho"] for row in rows if row["best_true_rho"] is not None]
    return {
        "targets": len(rows),
        "target_hits": sum(row["target_hit"] for row in rows),
        "target_hit_rate": sum(row["target_hit"] for row in rows) / len(rows),
        "logical_v2_1_evaluations": sum(row["selected"] for row in rows),
        "physics_acceptable": sum(row["physics_acceptable"] for row in rows),
        "median_best_true_rho": float(np.median(rho)) if rho else None,
        "minimum_best_true_rho": min(rho) if rho else None,
    }


def _aggregate_full_pool(target_runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [run["full_pool"] for run in target_runs]
    rho = [row["best_true_rho"] for row in rows if row["best_true_rho"] is not None]
    return {
        "targets": len(rows),
        "target_hits": sum(row["target_hit"] for row in rows),
        "target_hit_rate": sum(row["target_hit"] for row in rows) / len(rows),
        "logical_v2_1_evaluations": sum(row["selected"] for row in rows),
        "physics_acceptable": sum(row["physics_acceptable"] for row in rows),
        "candidate_hits": sum(row["diagnostic_hits"] for row in rows),
        "median_best_true_rho": float(np.median(rho)) if rho else None,
    }


def render_budget_sweep_report(summary: Mapping[str, Any]) -> str:
    budgets = summary["budgets"]
    lines = [
        "# WP19-23 Gate H 第二实验：V2.1 预算曲线与全池审计",
        "",
        f"> 生成时间：{summary['generated_at_utc']}",
        "> 固定使用首轮 Gate H 已生成候选；不重训、不重新生成、不运行 D/E 阶段。",
        "",
        "## 预算曲线",
        "",
        "| 选择策略 | 每目标预算 | 目标命中 | 物理有效 | median最佳rho |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "predicted_rho": "纯代理rho升序（嵌套前缀）",
        "ranked_diverse": "H3：一半rho、一半多样性",
    }
    for strategy in STRATEGIES:
        for budget in budgets:
            row = summary["aggregate"][strategy][str(budget)]
            rho = (
                "N/A"
                if row["median_best_true_rho"] is None
                else f"{row['median_best_true_rho']:.4g}"
            )
            lines.append(
                f"| {labels[strategy]} | {budget} | {row['target_hits']}/{row['targets']} | "
                f"{row['physics_acceptable']}/{row['logical_v2_1_evaluations']} | {rho} |"
            )
    full = summary["aggregate"]["full_pool"]
    rho = "N/A" if full["median_best_true_rho"] is None else f"{full['median_best_true_rho']:.4g}"
    lines.extend(
        [
            "",
            "## 全候选池真值",
            "",
            f"- 全池目标命中：**{full['target_hits']}/{full['targets']}**；",
            f"- 全池候选命中：**{full['candidate_hits']}**；",
            f"- 全池 V2.1 逻辑评价：**{full['logical_v2_1_evaluations']}**；",
            f"- 全池 median 最佳 `rho`：**{rho}**；",
            f"- 本次新增实际 V2.1 调用：**{summary['new_v2_1_evaluations']}**；",
            f"- 从首轮 Gate H 复用：**{summary['reused_v2_1_evaluations']}** 条唯一设计结果。",
            "",
            "`rho <= 1` 等价于推力、TSFC、质量三轴同时命中。",
            "",
            "## 判定",
            "",
            f"- 实验状态：**{summary['gate_h_sweep']['development_status']}**；",
            f"- 候选池覆盖判定：**{summary['gate_h_sweep']['pool_coverage_status']}**；",
            f"- 排序/预算判定：**{summary['gate_h_sweep']['ranking_budget_status']}**；",
            "- Formal Gate：**DEFERRED**。",
            "",
        ]
    )
    return "\n".join(lines)


def run_gate_h_budget_sweep(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    gate_h_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Exhaustively evaluate frozen Gate H pools and derive budget curves."""

    root = Path(repository_root).resolve()
    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    if config.get("protocol_version") != SWEEP_VERSION:
        raise Geometry19GateHSweepError("Gate H sweep protocol version mismatch")
    budgets = [int(value) for value in config["budgets"]]
    if budgets != sorted(set(budgets)) or any(value <= 0 for value in budgets):
        raise Geometry19GateHSweepError("Budgets must be unique positive ascending integers")
    source_runtime = (
        Path(gate_h_root).resolve()
        if gate_h_root is not None
        else (root / config["source_gate_h_runtime"]).resolve()
    )
    runtime = (
        Path(output_root).resolve()
        if output_root is not None
        else (root / config["artifacts"]["runtime_root"]).resolve()
    )
    if runtime.exists():
        raise Geometry19GateHSweepError(f"Refusing to overwrite sweep run: {runtime}")
    source_summary_path = source_runtime / "gate_h_summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if source_summary.get("status") != "PASS":
        raise Geometry19GateHSweepError("Source Gate H run is not PASS")
    runtime.mkdir(parents=True)
    contract_path = root / "config/physics/wp19_23_inverse_closure_v1.yaml"
    contract = load_wp19_23_contract(contract_path)
    protocol = contract.payload
    if source_summary["hashes"]["contract_sha256"] != sha256_file(contract_path):
        raise Geometry19GateHSweepError("Source Gate H contract hash is stale")
    model_config = ModelConfig.from_yaml(root / protocol["contracts"]["calibration_pack"])
    constraint_profile = ConstraintProfile.from_yaml(
        root / protocol["contracts"]["constraint_profile"]
    )
    dominance = ConstraintDominanceContract.from_yaml(
        root / protocol["contracts"]["constraint_dominance"]
    )
    tolerances = {
        name: float(value)
        for name, value in protocol["diagnostic_acceptance"]["relative_tolerances"].items()
    }
    target_runs = []
    reused_total = 0
    new_total = 0
    source_targets = sorted(source_runtime.glob("target_*/target_summary.json"))
    if len(source_targets) != int(source_summary["target_count"]):
        raise Geometry19GateHSweepError("Source Gate H target count mismatch")
    for source_target_summary_path in source_targets:
        source_target_root = source_target_summary_path.parent
        source_target = json.loads(source_target_summary_path.read_text(encoding="utf-8"))
        target_path = source_target_root / "target.yaml"
        target = load_target(target_path)
        pool = _read_jsonl(source_target_root / "projected_union.jsonl")
        if max(budgets) > len(pool):
            raise Geometry19GateHSweepError("Requested budget exceeds candidate pool")
        cache = _load_reusable_evaluations(source_target_root)
        reused_before = len(cache)
        environment = Environment()
        evaluation_ledger = []
        for position, row in enumerate(pool):
            design_hash = str(row["design_hash"])
            reused = design_hash in cache
            if not reused:
                cache[design_hash] = evaluate_geometry19_candidate(
                    position,
                    "gate_h_full_pool",
                    row["geometry19"],
                    design_hash,
                    target,
                    contract.diagnostic_acceptance,
                    model_config,
                    constraint_profile,
                    environment,
                    constraint_dominance_contract=dominance,
                )
            evaluation_ledger.append(
                {
                    "pool_position": position,
                    "candidate_key": row["candidate_key"],
                    "predicted_rho": row["predicted_rho"],
                    "reused_from_gate_h": reused,
                    "evaluation": cache[design_hash],
                }
            )
        reused_total += reused_before
        new_total += len(cache) - reused_before
        strategies: dict[str, dict[str, Any]] = {strategy: {} for strategy in STRATEGIES}
        for budget in budgets:
            selections = {
                "predicted_rho": select_predicted_rho(pool, budget),
                "ranked_diverse": select_tolerance_ranked_diverse(pool, budget),
            }
            for strategy, selected in selections.items():
                strategies[strategy][str(budget)] = _metrics_for_selection(
                    selected, cache, tolerances
                )
        full_pool = _metrics_for_selection(pool, cache, tolerances)
        target_result = {
            "target_index": source_target["target_index"],
            "dataset_sample_id": source_target["dataset_sample_id"],
            "target": source_target["target"],
            "pool_size": len(pool),
            "reused_unique_evaluations": reused_before,
            "new_unique_evaluations": len(cache) - reused_before,
            "strategies": strategies,
            "full_pool": full_pool,
        }
        target_runtime = runtime / source_target_root.name
        _write_json(target_runtime / "budget_sweep_summary.json", target_result)
        _write_jsonl(target_runtime / "full_pool_evaluations.jsonl", evaluation_ledger)
        target_runs.append(target_result)
    aggregate = {
        strategy: {
            str(budget): _aggregate(target_runs, strategy, budget) for budget in budgets
        }
        for strategy in STRATEGIES
    }
    aggregate["full_pool"] = _aggregate_full_pool(target_runs)
    full_hits = int(aggregate["full_pool"]["target_hits"])
    max_budget_hits = max(
        int(aggregate[strategy][str(max(budgets))]["target_hits"])
        for strategy in STRATEGIES
    )
    if full_hits == 0:
        pool_status = "NO_STRICT_HIT_IN_EXHAUSTIVE_PROJECTED_UNION"
    elif full_hits == len(target_runs):
        pool_status = "STRICT_HITS_EXIST_FOR_ALL_TARGETS"
    else:
        pool_status = "STRICT_HITS_EXIST_FOR_SUBSET_OF_TARGETS"
    if full_hits > max_budget_hits:
        ranking_status = "HITS_EXIST_BEYOND_256_SELECTION"
    elif max_budget_hits > 0:
        ranking_status = "BUDGET_SWEEP_RECOVERS_AVAILABLE_HITS"
    else:
        ranking_status = "NO_HIT_RECOVERED_BY_256"
    summary = {
        "protocol_version": SWEEP_VERSION,
        "work_package": "WP19-23-Gate-H-budget-sweep",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_use_allowed": False,
        "source_gate_h_runtime": str(source_runtime),
        "source_gate_h_summary_sha256": sha256_file(source_summary_path),
        "budgets": budgets,
        "target_count": len(target_runs),
        "target_runs": target_runs,
        "aggregate": aggregate,
        "reused_v2_1_evaluations": reused_total,
        "new_v2_1_evaluations": new_total,
        "gate_h_sweep": {
            "development_status": "PASS",
            "pool_coverage_status": pool_status,
            "ranking_budget_status": ranking_status,
            "formal_gate_status": "DEFERRED",
        },
        "hashes": {
            "config_sha256": sha256_file(config_source),
            "contract_sha256": sha256_file(contract_path),
            "code_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        },
    }
    summary_path = runtime / config["artifacts"]["summary"]
    report_path = runtime / config["artifacts"]["report"]
    _write_json(summary_path, summary)
    report_path.write_text(render_budget_sweep_report(summary), encoding="utf-8")
    manifest_path = runtime / config["artifacts"]["manifest"]
    artifacts = [
        {
            "path": str(path.relative_to(runtime)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(path for path in runtime.rglob("*") if path.is_file())
        if path != manifest_path
    ]
    _write_json(
        manifest_path,
        {
            "manifest_version": "wp19_23_gate_h_budget_sweep_manifest_v1",
            "summary_sha256": sha256_file(summary_path),
            "report_sha256": sha256_file(report_path),
            "artifacts": artifacts,
        },
    )
    return summary
