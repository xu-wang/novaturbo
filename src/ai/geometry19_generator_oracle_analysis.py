"""Finalize controls and target-level diagnostics for the generator oracle run."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from src.ai.geometry19_generator_oracle import (
    _physics_worker,
    _prefix_metrics,
    _read_jsonl,
    _worker_init,
    _write_json,
    _write_jsonl,
)
from src.ai.geometry19_inverse_contract import Geometry19InverseTarget, load_wp19_23_contract


OUTPUTS = ("thrust_N", "tsfc_kg_N_s", "mass_kg")


def _rho(performance: Mapping[str, Any], target: Mapping[str, Any], tolerances: Mapping[str, float]) -> float:
    return max(abs(float(performance[name]) - float(target[name])) / abs(float(target[name])) / float(tolerances[name]) for name in OUTPUTS)


def _stored_acceptable(row: Mapping[str, Any]) -> bool:
    return bool(row.get("solver_status", {}).get("converged") and row.get("solver_status", {}).get("conservation_passed") and row.get("constraint_values", {}).get("assessment", {}).get("accepted") and row.get("physics_outputs", {}).get("performance"))


def _control_rows(pool: Sequence[Mapping[str, Any]], target_row: Mapping[str, Any], tolerances: Mapping[str, float], *, sort_by_rho: bool) -> list[dict[str, Any]]:
    target = target_row["target"]
    rows = []
    for index, source in enumerate(pool):
        value = _rho(source["physics_outputs"]["performance"], target, tolerances) if _stored_acceptable(source) else None
        rows.append({"candidate_id": index, "rho": value, "physics_acceptable": value is not None})
    if sort_by_rho:
        rows.sort(key=lambda row: (math.inf if row["rho"] is None else row["rho"], row["candidate_id"]))
    return rows


def _aggregate_control(runs: Sequence[Mapping[str, Any]], prefixes: Sequence[int]) -> dict[str, Any]:
    result: dict[str, Any] = {"runs": len(runs), "targets": len(runs)}
    for prefix in prefixes:
        metrics = [run["prefix_metrics"][str(prefix)] for run in runs]
        best = [row["best_true_rho"] for row in metrics if row["best_true_rho"] is not None]
        result[str(prefix)] = {
            "oracle_hit_rate": float(np.mean([row["oracle_hit"] for row in metrics])),
            "median_best_true_rho": float(np.median(best)) if best else None,
            "p90_best_true_rho": float(np.percentile(best, 90)) if best else None,
            "median_hit_density": float(np.median([row["hit_density"] for row in metrics])),
        }
    return result


def _target_level(candidate_rows: Sequence[Mapping[str, Any]], suite: Sequence[Mapping[str, Any]], arms: Sequence[str], prefixes: Sequence[int]) -> dict[str, Any]:
    split_by_target = {row["target"]["target_id"]: row["split"] for row in suite}
    output: dict[str, Any] = {}
    for arm in arms:
        output[arm] = {}
        for split in ("train_diagnostic", "development", "locked_held_out", "historical_regression"):
            targets = [target_id for target_id, value in split_by_target.items() if value == split]
            split_result: dict[str, Any] = {"targets": len(targets)}
            for prefix in prefixes:
                hit_counts = []
                best_values = []
                for target_id in targets:
                    groups: dict[int, list[Mapping[str, Any]]] = {}
                    for row in candidate_rows:
                        if row["arm_id"] == arm and row["target_id"] == target_id:
                            groups.setdefault(int(row["source_seed"]), []).append(row)
                    per_seed = [_prefix_metrics(sorted(rows, key=lambda item: item["candidate_id"]), [prefix])[str(prefix)] for rows in groups.values()]
                    hit_counts.append(sum(value["oracle_hit"] for value in per_seed))
                    finite = [value["best_true_rho"] for value in per_seed if value["best_true_rho"] is not None]
                    if finite:
                        best_values.append(min(finite))
                split_result[str(prefix)] = {
                    "any_seed_hit_rate": sum(value >= 1 for value in hit_counts) / len(targets),
                    "two_of_three_hit_rate": sum(value >= 2 for value in hit_counts) / len(targets),
                    "all_three_hit_rate": sum(value >= 3 for value in hit_counts) / len(targets),
                    "median_best_across_seeds": float(np.median(best_values)) if best_values else None,
                    "p90_best_across_seeds": float(np.percentile(best_values, 90)) if best_values else None,
                }
            output[arm][split] = split_result
    return output


def _render(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Geometry19 生成器本体 Oracle 实验报告",
        "",
        f"> 生成时间：{summary['generated_at_utc']}",
        "> 范围：direct-only；未使用 D-Flow、reranker、gradient refinement、NSGA-II 或新模型训练。",
        "",
        "## 1. Generator-only 单次采样结果",
        "",
        "以下 Hit 是 `model × target × source seed` 运行级比例。",
        "",
        "| Arm | Split | Hit@256 | Hit@512 | median best rho@256 | P90 best rho@256 | median hit density@256 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm, splits in summary["aggregate"].items():
        for split, row in splits.items():
            a, b = row["256"], row["512"]
            lines.append(f"| {arm} | {split} | {a['oracle_hit_rate']:.1%} | {b['oracle_hit_rate']:.1%} | {a['median_best_true_rho']:.3f} | {a['p90_best_true_rho']:.3f} | {a['median_hit_density']:.2%} |")
    lines += ["", "## 2. Locked held-out 跨三个 source seed 的目标覆盖", "", "| Arm | N | 任一seed命中 | 至少2/3命中 | 3/3命中 | median跨seed最佳rho | P90跨seed最佳rho |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, splits in summary["target_level"].items():
        row = splits["locked_held_out"]
        for prefix in (256, 512):
            value = row[str(prefix)]
            lines.append(f"| {arm} | {prefix} | {value['any_seed_hit_rate']:.1%} | {value['two_of_three_hit_rate']:.1%} | {value['all_three_hit_rate']:.1%} | {value['median_best_across_seeds']:.3f} | {value['p90_best_across_seeds']:.3f} |")
    lines += ["", "## 3. 严格候选漏斗（全部目标、全部seed、N=512）", "", "| Arm | raw | bounds fail | discrete fail | geometry fail | prefilter valid | V2.1 acceptable | strict hits |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, row in summary["funnel"].items():
        lines.append(f"| {arm} | {row['raw']} | {row.get('OUT_OF_SCHEMA_BOUNDS',0)} | {row.get('INVALID_DISCRETE_COORDINATE',0)} | {row.get('GEOMETRY_MAPPING_FAILED',0)} | {row['prefilter_valid']} | {row['physics_acceptable']} | {row['strict_hits']} |")
    lines += ["", "## 4. 非神经对照（locked held-out）", "", "| Control | Hit@256 | Hit@512 | median best rho@256 | P90 best rho@256 |", "|---|---:|---:|---:|---:|"]
    for name, row in summary["controls"].items():
        a, b = row["256"], row["512"]
        lines.append(f"| {name} | {a['oracle_hit_rate']:.1%} | {b['oracle_hit_rate']:.1%} | {a['median_best_true_rho']:.3f} | {a['p90_best_true_rho']:.3f} |")
    lines += ["", "## 5. 契约回放与 Gate", "", f"- 原始可达设计回放：{summary['reference_replay']['strict_hits']}/{summary['reference_replay']['targets']}；", f"- 回放失败：{summary['reference_replay']['failures']}；"]
    for key, value in summary["gate"].items():
        lines.append(f"- {key}: **{value}**")
    lines += ["", "## 6. 分析结论", ""]
    for line in summary["analysis_conclusions"]:
        lines.append(f"- {line}")
    lines += ["", "本实验到此停止。未开展生成器结构、loss、mixed head、reranker、D-Flow 或优化器改进实验。", ""]
    return "\n".join(lines)


def finalize(root: str | Path, config_path: str | Path) -> dict[str, Any]:
    repo = Path(root).resolve()
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    runtime = repo / config["artifacts"]["runtime_root"]
    summary_path = runtime / config["artifacts"]["summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    suite = _read_jsonl(runtime / config["artifacts"]["target_suite"])
    candidates = _read_jsonl(runtime / config["artifacts"]["candidate_ledger"])
    prefixes = [int(value) for value in config["sampling"]["nested_prefixes"]]
    arms = [row["arm_id"] for row in config["models"]]
    summary["target_level"] = _target_level(candidates, suite, arms, prefixes)

    contract = load_wp19_23_contract(repo / "config/physics/wp19_23_inverse_closure_v1.yaml")
    tolerances = contract.diagnostic_acceptance.relative_tolerances
    regression = _read_jsonl(repo / config["data"]["regression_dataset"])
    train = [row for row in regression if row["campaign_id"] in config["data"]["train_campaigns"]]
    doe = []
    with (repo / config["data"]["full_dataset"]).open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["campaign_id"] == config["controls"]["doe_direct_campaign"]:
                doe.append(row)
                if len(doe) == 512:
                    break
    held = [row for row in suite if row["split"] == "locked_held_out"]
    controls = {}
    knn_runs, doe_runs = [], []
    for target_row in held:
        knn = _control_rows(train, target_row, tolerances, sort_by_rho=True)[:512]
        doe_rows = _control_rows(doe, target_row, tolerances, sort_by_rho=False)
        knn_runs.append({"prefix_metrics": _prefix_metrics(knn, prefixes)})
        doe_runs.append({"prefix_metrics": _prefix_metrics(doe_rows, prefixes)})
    controls["condition_kNN"] = _aggregate_control(knn_runs, prefixes)
    controls["DOE_direct_fixed_pool"] = _aggregate_control(doe_runs, prefixes)
    summary["controls"] = controls

    replay_tasks = []
    for index, target_row in enumerate(suite):
        geometry = target_row["reference_x_19"]
        design_hash = sha256(json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        replay_tasks.append({"run_id": "reference_replay", "candidate_id": index, "geometry19": geometry, "design_hash": design_hash, "target": target_row["target"]})
    with ProcessPoolExecutor(max_workers=int(config["execution"]["physics_workers"]), initializer=_worker_init, initargs=(str(repo),)) as executor:
        replay = list(executor.map(_physics_worker, replay_tasks, chunksize=8))
    _write_jsonl(runtime / "reference_replay_ledger.jsonl", replay)
    strict = sum(bool(row["development_diagnostic_hit"]) for row in replay)
    summary["reference_replay"] = {"targets": len(replay), "strict_hits": strict, "failures": len(replay) - strict}

    bounded = summary["aggregate"]["g1_bounded_sigmoid"]["locked_held_out"]
    target_bounded = summary["target_level"]["g1_bounded_sigmoid"]["locked_held_out"]
    funnel = summary["funnel"]["g1_bounded_sigmoid"]
    raw = funnel["raw"]
    summary["analysis_conclusions"] = [
        f"有界参数化消除了越界，但 {funnel.get('INVALID_DISCRETE_COORDINATE',0) / raw:.1%} raw 候选仍在冻结离散容差处失败，主要合法性瓶颈转移到三个整数维。",
        f"有界模型 held-out 运行级 OracleHit@256={bounded['256']['oracle_hit_rate']:.1%}、@512={bounded['512']['oracle_hit_rate']:.1%}，median BestTrueRho@256={bounded['256']['median_best_true_rho']:.3f}，未达到预注册门槛。",
        f"即使合并三个 source seed，held-out 目标任一 seed 的覆盖在 N=256 也只有 {target_bounded['256']['any_seed_hit_rate']:.1%}，至少 2/3 seed 稳定命中的比例为 {target_bounded['256']['two_of_three_hit_rate']:.1%}。",
        f"严格命中总数仅 {funnel['strict_hits']} / {funnel['raw']} raw，median hit density@256 为 {bounded['256']['median_hit_density']:.2%}，说明命中不是高密度条件生成，而是少量目标上的稀疏事件。",
        "结果支持先改进生成器本体；但当前实验只完成诊断，不授权任何具体结构改进方案。",
    ]
    _write_json(summary_path, summary)
    (runtime / config["artifacts"]["report"]).write_text(_render(summary), encoding="utf-8")
    return summary
