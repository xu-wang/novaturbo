"""Fair direct-only Oracle evaluation for the Geometry19 mixed CFM arms."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch
import yaml

# Import registers both checkpoint types before load_generator is called.
from src.ai import geometry19_mixed_cfm as _mixed_registration  # noqa: F401
from src.ai.generators import load_generator
from src.ai.geometry19_mixed_cfm import Geometry19TupleCFM, INTEGER_NAMES
from src.ai.geometry19_generator_failure import _select_targets
from src.ai.geometry19_generator_oracle import (
    _aggregate,
    _filter_samples,
    _physics_worker,
    _prefix_metrics,
    _read_jsonl,
    _worker_init,
    _write_json,
    _write_jsonl,
)
from src.ai.geometry19_inverse_contract import Geometry19InverseTarget, load_wp19_23_contract
from src.physics import ModelConfig


VERSION = "wp19_23_mixed_variable_cfm_eval_v1"


def _render(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Geometry19 混合变量CFM直接生成实验",
        "",
        f"> 生成时间：{summary['generated_at_utc']}",
        "> direct-only、原始候选前缀、无补采样；未使用reranker、D-Flow、梯度精修或NSGA-II。",
        "",
        f"## N={summary['evaluation_key']}主结果",
        "",
        "| Arm | Split | 运行命中率 | median best rho | P90 best rho | mean physics acceptable |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm in summary["arm_ids"]:
        for split in ("train_diagnostic", "locked_held_out"):
            row = summary["aggregate"][arm][split][summary["evaluation_key"]]
            lines.append(
                f"| {arm} | {split} | {row['oracle_hit_rate']:.1%} | "
                f"{row['median_best_true_rho']:.3f} | {row['p90_best_true_rho']:.3f} | "
                f"{row['mean_physics_acceptable']:.1f} |"
            )
    baseline = summary["historical_projection_baseline"]
    lines += [
        "",
        "## 已完成的最近整数投影基线",
        "",
        "| Split | 运行命中率 | median best rho | mean physics acceptable |",
        "|---|---:|---:|---:|",
    ]
    for split in ("train_diagnostic", "locked_held_out"):
        row = baseline[split]
        lines.append(f"| {split} | {row['run_hit_rate']:.1%} | {row['median_best_rho']:.3f} | {row['mean_physics_acceptable']:.1f} |")
    lines += ["", "## 相对投影基线", ""]
    for arm in summary["arm_ids"]:
        delta = summary["comparison_to_projection"][arm]
        lines.append(
            f"- {arm}: held-out命中率变化 {delta['heldout_hit_rate_delta_pp']:+.1f}个百分点，"
            f"median best rho变化 {delta['heldout_median_best_rho_delta']:+.3f}。"
        )
    lines += ["", "## 候选漏斗", "", "| Arm | raw | discrete fail | geometry fail | valid | physics acceptable | strict hits |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, row in summary["funnel"].items():
        lines.append(
            f"| {arm} | {row['raw']} | {row.get('INVALID_DISCRETE_COORDINATE',0)} | "
            f"{row.get('GEOMETRY_MAPPING_FAILED',0)} | {row['prefilter_valid']} | "
            f"{row['physics_acceptable']} | {row['strict_hits']} |"
        )
    lines += ["", "本实验只比较表中生成器本体；未启动任何下游改进。", ""]
    return "\n".join(lines)


def _tuple_retrieval_data(repo: Path, config: Mapping[str, Any], normalizer) -> tuple[np.ndarray, np.ndarray]:
    rows = _read_jsonl(repo / config["data"]["regression_dataset"])
    campaigns = set(config["data"]["campaign_split"]["train"])
    selected = [row for row in rows if row["campaign_id"] in campaigns]
    conditions = np.asarray([
        [row["physics_outputs"]["performance"][name] for name in ("thrust_N", "tsfc_kg_N_s", "mass_kg")]
        for row in selected
    ], dtype=float)
    tuples = np.asarray([[int(row["x_19"][name]) for name in INTEGER_NAMES] for row in selected], dtype=int)
    lower = np.asarray([8, 4, 11], dtype=int)
    return normalizer.transform_conditions(conditions), tuples - lower


def _top_tuple_labels(condition: np.ndarray, train_conditions: np.ndarray, train_labels: np.ndarray, top_m: int, count: int) -> np.ndarray:
    distances = np.sum((train_conditions - condition[None, :]) ** 2, axis=1)
    order = np.argsort(distances, kind="stable")
    chosen = []
    seen = set()
    for index in order:
        value = tuple(int(item) for item in train_labels[index])
        if value not in seen:
            seen.add(value)
            chosen.append(value)
            if len(chosen) == top_m:
                break
    if len(chosen) < top_m:
        raise ValueError("Training data contain fewer unique tuples than tuple_top_m")
    tiled = np.tile(np.asarray(chosen, dtype=int), (int(np.ceil(count / top_m)), 1))
    return tiled[:count]


def run_mixed_cfm_evaluation(
    root: str | Path, config_path: str | Path, *, device: str | None = None,
    smoke: bool = False, selected_arms: list[str] | None = None,
    output_subdir: str | None = None,
    checkpoint_run_kind: str | None = None,
    checkpoint_state: str = "best",
) -> dict[str, Any]:
    repo = Path(root).resolve()
    config = yaml.safe_load(Path(config_path).resolve().read_text(encoding="utf-8"))
    evaluation = config["evaluation"]
    if evaluation["protocol_version"] != VERSION:
        raise ValueError("Mixed CFM evaluation protocol mismatch")
    if checkpoint_state not in {"best", "last"}:
        raise ValueError("checkpoint_state must be best or last")
    runtime = repo / evaluation["runtime_root"] / (output_subdir or ("smoke" if smoke else "baseline"))
    if runtime.exists():
        raise ValueError(f"Refusing to overwrite mixed CFM evaluation: {runtime}")
    runtime.mkdir(parents=True)
    suite = _read_jsonl(repo / evaluation["target_suite"])
    targets = _select_targets(suite, 1 if smoke else int(evaluation["held_out_per_region"]))
    if smoke:
        targets = targets[:1] + [row for row in targets if row["split"] == "locked_held_out"][:1]
    contract = load_wp19_23_contract(repo / "config/physics/wp19_23_inverse_closure_v1.yaml")
    schema = yaml.safe_load((repo / contract.payload["contracts"]["design_schema"]).read_text(encoding="utf-8"))
    physics_model = ModelConfig.from_yaml(repo / contract.payload["contracts"]["calibration_pack"])
    maximum = int(evaluation["smoke_candidates"] if smoke else evaluation["maximum_raw_candidates"])
    prefixes = [maximum] if smoke else [int(value) for value in evaluation["nested_prefixes"]]
    seeds = [int(evaluation["source_seeds"][0])] if smoke else [int(value) for value in evaluation["source_seeds"]]
    arm_specs = [arm for arm in config["arms"] if selected_arms is None or arm["arm_id"] in selected_arms]
    if not arm_specs:
        raise ValueError("No mixed CFM evaluation arms selected")
    arm_ids = [str(value["arm_id"]) for value in arm_specs]
    run_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    physics_rows: list[dict[str, Any]] = []
    started = perf_counter()
    workers = 2 if smoke else int(evaluation["physics_workers"])
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(str(repo),)) as executor:
        for arm in arm_specs:
            arm_id = str(arm["arm_id"])
            run_kind = checkpoint_run_kind or ("smoke" if smoke else "baseline")
            checkpoint = repo / arm["artifacts"]["weights_root"] / run_kind / f"{checkpoint_state}.pt"
            generator = load_generator(checkpoint, device=device or evaluation["device"])
            retrieval = _tuple_retrieval_data(repo, config, generator.normalizer) if isinstance(generator, Geometry19TupleCFM) else None
            for target_row in targets:
                target = Geometry19InverseTarget.from_mapping(target_row["target"])
                physical_condition = np.asarray([target.thrust_N, target.tsfc_kg_N_s, target.mass_kg])
                normalized_condition = torch.as_tensor(
                    generator.normalizer.transform_conditions(physical_condition),
                    dtype=generator.dtype, device=generator.device,
                )
                for seed in seeds:
                    if isinstance(generator, Geometry19TupleCFM):
                        train_conditions, train_labels = retrieval
                        labels = _top_tuple_labels(
                            np.asarray(normalized_condition.cpu()), train_conditions, train_labels,
                            int(evaluation["tuple_top_m"]), maximum,
                        )
                        samples = generator.sample_with_labels(
                            normalized_condition,
                            torch.as_tensor(labels, dtype=torch.long), seed,
                        ).numpy()
                    else:
                        samples = generator.sample(normalized_condition, maximum, seed=seed, space="normalized").values.numpy()
                    candidates, tasks = _filter_samples(samples, arm_id, target.target_id, seed, schema, physics_model)
                    for task in tasks:
                        task["target"] = target.as_mapping()
                    physics = list(executor.map(_physics_worker, tasks, chunksize=int(evaluation["physics_chunksize"])))
                    by_id = {int(row["candidate_id"]): row for row in physics}
                    for row in candidates:
                        result = by_id.get(int(row["candidate_id"]))
                        row["physics_acceptable"] = bool(result and result["physics_acceptable"])
                        row["rho"] = (result.get("hit_alignment") or {}).get("rho") if result else None
                        row["split"] = target_row["split"]
                        row["target_region"] = target_row["target_region"]
                    candidate_rows.extend(candidates)
                    physics_rows.extend(physics)
                    run_rows.append({
                        "run_id": f"{arm_id}:{target.target_id}:{seed}", "arm_id": arm_id,
                        "target_id": target.target_id, "split": target_row["split"],
                        "target_region": target_row["target_region"], "source_seed": seed,
                        "prefix_metrics": _prefix_metrics(candidates, prefixes),
                    })
    splits = ("train_diagnostic", "locked_held_out")
    aggregate = {arm: {split: _aggregate(run_rows, split, arm, prefixes) for split in splits} for arm in arm_ids}
    funnel = {}
    for arm in arm_ids:
        rows = [row for row in candidate_rows if row["arm_id"] == arm]
        funnel[arm] = {
            "raw": len(rows),
            "INVALID_DISCRETE_COORDINATE": sum(row["failure_code"] == "INVALID_DISCRETE_COORDINATE" for row in rows),
            "GEOMETRY_MAPPING_FAILED": sum(row["failure_code"] == "GEOMETRY_MAPPING_FAILED" for row in rows),
            "prefilter_valid": sum(row["failure_code"] is None for row in rows),
            "physics_acceptable": sum(row.get("rho") is not None for row in rows),
            "strict_hits": sum(row.get("rho") is not None and row["rho"] <= 1.0 for row in rows),
        }
    historical = json.loads((repo / evaluation["historical_failure_summary"]).read_text(encoding="utf-8"))
    historical_key = str(evaluation["maximum_raw_candidates"])
    key = str(maximum)
    projection = {
        split: historical["aggregate"]["nearest_integer"][split][historical_key]
        for split in splits
    }
    comparison = {}
    for arm in arm_ids:
        current = aggregate[arm]["locked_held_out"][key]
        base = projection["locked_held_out"]
        comparison[arm] = {
            "heldout_hit_rate_delta_pp": 100.0 * (current["oracle_hit_rate"] - base["run_hit_rate"]),
            "heldout_median_best_rho_delta": current["median_best_true_rho"] - base["median_best_rho"],
        }
    summary = {
        "protocol_version": VERSION, "status": "SMOKE_PASS" if smoke else "COMPLETE",
        "formal_use_allowed": False, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm_ids": arm_ids, "target_count": len(targets), "source_seeds": seeds,
        "maximum_raw_candidates": maximum, "prefixes": prefixes,
        "evaluation_key": key,
        "checkpoint_run_kind": checkpoint_run_kind or ("smoke" if smoke else "baseline"),
        "checkpoint_state": checkpoint_state,
        "physics_calls": len(physics_rows), "elapsed_seconds": perf_counter() - started,
        "aggregate": aggregate, "funnel": funnel,
        "historical_projection_baseline": projection,
        "comparison_to_projection": comparison,
        "excluded": ["reranker", "dflow", "gradient_refinement", "nsga2"],
    }
    _write_jsonl(runtime / "run_ledger.jsonl", run_rows)
    _write_jsonl(runtime / "candidate_ledger.jsonl", candidate_rows)
    _write_jsonl(runtime / "physics_ledger.jsonl", physics_rows)
    _write_json(runtime / "summary.json", summary)
    (runtime / "Mixed_Variable_CFM_Report.md").write_text(_render(summary), encoding="utf-8")
    return summary
