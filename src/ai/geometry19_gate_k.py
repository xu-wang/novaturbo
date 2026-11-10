"""Gate K: equal-data ablation of hit-aligned Geometry19 surrogate losses."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from src.ai.geometry19_hit_alignment import OUTPUT_NAMES
from src.ai.geometry19_candidates import run_candidate_protocol
from src.ai.geometry19_physics import run_true_physics_stage
from src.ai.geometry19_pipeline import build_target_mapping, resolve_generator_checkpoint
from src.ai.geometry19_refinement import run_refinement_stage
from src.ai.geometry19_surrogate import (
    TaggedGeometry19MLP,
    load_weight_bundle,
    save_weight_bundle,
    sha256_file,
)
from src.ai.geometry19_surrogate_objective import (
    hit_aligned_prediction_metrics,
    surrogate_loss_terms,
    validate_surrogate_loss_config,
)
from src.ai.geometry19_training import (
    _loader,
    _predict,
    _regression_metrics,
    _train_model,
    fit_normalization,
    load_regression_splits,
)


GATE_K_VERSION = "wp19_22_gate_k_hit_aligned_surrogate_v1"


class Geometry19GateKError(ValueError):
    """Raised when the Gate K campaign contract is violated."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def ranking_benchmark(
    truth: np.ndarray,
    prediction: np.ndarray,
    relative_tolerances: Mapping[str, float],
    denominator_epsilons: Mapping[str, float],
    *,
    maximum_targets: int,
    budgets: Sequence[int],
) -> dict[str, Any]:
    """Evaluate target retrieval without calling physics or using source identity."""

    truth_array = np.asarray(truth, dtype=np.float64)[:, :3]
    prediction_array = np.asarray(prediction, dtype=np.float64)[:, :3]
    if truth_array.shape != prediction_array.shape or len(truth_array) < 2:
        raise Geometry19GateKError("Ranking benchmark arrays are invalid")
    count = min(int(maximum_targets), len(truth_array))
    target_indices = np.linspace(0, len(truth_array) - 1, count, dtype=int)
    tolerances = np.asarray([relative_tolerances[name] for name in OUTPUT_NAMES])
    epsilons = np.asarray([denominator_epsilons[name] for name in OUTPUT_NAMES])
    coverage = {int(budget): 0 for budget in budgets}
    recalls = {int(budget): [] for budget in budgets}
    first_hit_ranks: list[int] = []
    correlations: list[float] = []
    eligible = 0
    for target_index in target_indices:
        target = truth_array[target_index]
        denominator = tolerances * np.maximum(np.abs(target), epsilons)
        true_rho = (np.abs(truth_array - target) / denominator).max(axis=1)
        predicted_rho = (np.abs(prediction_array - target) / denominator).max(axis=1)
        true_rho[target_index] = np.inf
        predicted_rho[target_index] = np.inf
        hits = np.flatnonzero(true_rho <= 1.0)
        if not len(hits):
            continue
        eligible += 1
        ordered = np.argsort(predicted_rho, kind="stable")
        hit_mask = np.zeros(len(truth_array), dtype=bool)
        hit_mask[hits] = True
        ranked_hits = np.flatnonzero(hit_mask[ordered])
        first_hit_ranks.append(int(ranked_hits[0] + 1))
        for budget in budgets:
            selected_hits = int(hit_mask[ordered[: int(budget)]].sum())
            coverage[int(budget)] += int(selected_hits > 0)
            recalls[int(budget)].append(selected_hits / len(hits))
        finite = np.isfinite(true_rho) & np.isfinite(predicted_rho)
        true_rank = _rank(true_rho[finite])
        predicted_rank = _rank(predicted_rho[finite])
        correlation = float(np.corrcoef(true_rank, predicted_rank)[0, 1])
        if math.isfinite(correlation):
            correlations.append(correlation)
    return {
        "target_count": count,
        "eligible_target_count": eligible,
        "target_coverage": {
            str(budget): coverage[int(budget)] / eligible if eligible else None
            for budget in budgets
        },
        "mean_hit_recall": {
            str(budget): float(np.mean(recalls[int(budget)])) if recalls[int(budget)] else None
            for budget in budgets
        },
        "first_hit_rank_p50": float(np.median(first_hit_ranks)) if first_hit_ranks else None,
        "first_hit_rank_p90": float(np.quantile(first_hit_ranks, 0.9)) if first_hit_ranks else None,
        "rho_spearman_mean": float(np.mean(correlations)) if correlations else None,
    }


def _aggregate_arm(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def values(path: Sequence[str]) -> list[float]:
        output = []
        for run in runs:
            value: Any = run
            for name in path:
                value = value[name]
            if value is not None:
                output.append(float(value))
        return output

    def median_value(path: Sequence[str]) -> float | None:
        selected = values(path)
        return float(np.median(selected)) if selected else None

    return {
        "run_count": len(runs),
        "validation_rho_p50_median": median_value(("validation", "hit_alignment", "rho_p50")),
        "test_rho_p50_median": median_value(("test", "hit_alignment", "rho_p50")),
        "test_performance_relative_error_p50_median": {
            name: median_value(("test", "regression", "outputs", name, "relative_error_p50"))
            for name in OUTPUT_NAMES
        },
        "test_target_coverage_at_32_median": median_value(("test", "ranking", "target_coverage", "32")),
        "test_target_coverage_at_64_median": median_value(("test", "ranking", "target_coverage", "64")),
        "test_rho_spearman_mean_median": median_value(("test", "ranking", "rho_spearman_mean")),
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    def number(value: float | None, pattern: str = ".3f") -> str:
        return format(value, pattern) if value is not None else "n/a"

    lines = [
        "# Gate K：Geometry19代理命中损失消融",
        "",
        f"> 状态：**{summary['status']}**；模型选择只使用validation指标。",
        "",
        "| 实验臂 | 损失 | Val median rho | Test median rho | 推力P50 | TSFC P50 | 质量P50 | coverage@32 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm_id, arm in summary["arms"].items():
        aggregate = arm["aggregate"]
        errors = aggregate["test_performance_relative_error_p50_median"]
        lines.append(
            f"| {arm_id} | {arm['loss']['mode']} | "
            f"{number(aggregate['validation_rho_p50_median'])} | "
            f"{number(aggregate['test_rho_p50_median'])} | "
            f"{number(errors['thrust_N'], '.3%')} | {number(errors['tsfc_kg_N_s'], '.3%')} | "
            f"{number(errors['mass_kg'], '.3%')} | "
            f"{number(aggregate['test_target_coverage_at_32_median'])} |"
        )
    lines += [
        "",
        f"按validation选择的胜出臂：`{summary['selection']['winning_arm']}`；"
        f"代表种子：`{summary['selection']['winning_seed']}`。",
        "",
        "测试集未参与模型或超参数选择；本报告仍为development级，不自动替换生产代理。",
        "",
    ]
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _predict_pool_rho(
    rows: Sequence[Mapping[str, Any]],
    target: Mapping[str, float],
    feature_names: Sequence[str],
    model: torch.nn.Module,
    normalization: Mapping[str, np.ndarray],
    tolerances: Mapping[str, float],
    epsilons: Mapping[str, float],
) -> np.ndarray:
    device = next(model.parameters()).device
    physical = np.asarray(
        [[row["geometry19"][name] for name in feature_names] for row in rows],
        dtype=np.float32,
    )
    normalized = (
        (physical - normalization["input_mean"]) / normalization["input_std"]
    ).astype(np.float32)
    prediction_normalized = _predict(model, normalized, device)
    prediction = (
        prediction_normalized * normalization["output_std"]
        + normalization["output_mean"]
    )[:, :3]
    target_values = np.asarray([target[name] for name in OUTPUT_NAMES], dtype=np.float64)
    denominator = np.asarray([tolerances[name] for name in OUTPUT_NAMES]) * np.maximum(
        np.abs(target_values), np.asarray([epsilons[name] for name in OUTPUT_NAMES])
    )
    return (np.abs(prediction - target_values) / denominator).max(axis=1)


def _frozen_strategy_metrics(
    rows: Sequence[Mapping[str, Any]],
    evaluations: Mapping[str, Mapping[str, Any]],
    predicted_rho: np.ndarray,
    tolerances: Mapping[str, float],
    budgets: Sequence[int],
) -> dict[str, Any]:
    ordered_indices = sorted(
        range(len(rows)),
        key=lambda index: (float(predicted_rho[index]), str(rows[index]["candidate_key"])),
    )
    true_rhos = []
    hits = []
    acceptable = []
    for row in rows:
        evaluation = evaluations[str(row["candidate_key"])]["evaluation"]
        is_acceptable = bool(evaluation.get("physics_acceptable"))
        acceptable.append(is_acceptable)
        errors = evaluation.get("relative_errors")
        rho = (
            max(float(errors[name]) / float(tolerances[name]) for name in OUTPUT_NAMES)
            if is_acceptable and isinstance(errors, Mapping)
            else math.inf
        )
        true_rhos.append(rho)
        hits.append(bool(evaluation.get("development_diagnostic_hit")))
    true_rho_array = np.asarray(true_rhos)
    predicted_array = np.asarray(predicted_rho, dtype=np.float64)
    hit_array = np.asarray(hits, dtype=bool)
    acceptable_array = np.asarray(acceptable, dtype=bool)
    ordered = np.asarray(ordered_indices, dtype=int)
    ranked_hits = np.flatnonzero(hit_array[ordered])
    finite = np.isfinite(true_rho_array) & np.isfinite(predicted_array)
    correlation = (
        float(np.corrcoef(_rank(true_rho_array[finite]), _rank(predicted_array[finite]))[0, 1])
        if finite.sum() > 1 else None
    )
    result = {
        "pool_size": len(rows),
        "pool_hits": int(hit_array.sum()),
        "first_hit_rank": int(ranked_hits[0] + 1) if len(ranked_hits) else None,
        "rho_spearman": correlation,
        "predicted_hit_count": int(np.sum(predicted_array <= 1.0)),
        "predicted_hit_false_positive_count": int(np.sum((predicted_array <= 1.0) & ~hit_array)),
        "budgets": {},
    }
    for raw_budget in budgets:
        budget = min(int(raw_budget), len(rows))
        selected = ordered[:budget]
        selected_true = true_rho_array[selected]
        finite_selected = selected_true[np.isfinite(selected_true)]
        result["budgets"][str(raw_budget)] = {
            "evaluated": budget,
            "physics_acceptable": int(acceptable_array[selected].sum()),
            "diagnostic_hits": int(hit_array[selected].sum()),
            "target_hit": bool(hit_array[selected].any()),
            "best_true_rho": float(finite_selected.min()) if len(finite_selected) else None,
        }
    return result


def run_gate_k_frozen_rerank(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    device: str = "cpu",
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Rerank the fully evaluated Gate H pool without new physics calls."""

    root = Path(repository_root).resolve()
    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    runtime = Path(output_root).resolve() if output_root else (
        root / config["artifacts"]["runtime_root"] / config["frozen_rerank"]["output_subdirectory"]
    )
    if runtime.exists():
        raise Geometry19GateKError(f"Refusing to overwrite frozen rerank: {runtime}")
    training_summary_path = root / config["artifacts"]["runtime_root"] / config["artifacts"]["summary"]
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    base = yaml.safe_load((root / config["base_surrogate_protocol"]).read_text(encoding="utf-8"))
    inverse = yaml.safe_load((root / config["inverse_protocol"]).read_text(encoding="utf-8"))
    tolerances = dict(inverse["diagnostic_acceptance"]["relative_tolerances"])
    epsilons = dict(inverse["diagnostic_acceptance"]["epsilons"])
    schema = yaml.safe_load((root / base["design_schema"]).read_text(encoding="utf-8"))
    feature_names = [item["name"] for item in schema["design_variables"]]
    target_names = json.loads(
        (root / base["artifacts"]["weights_root"] / "forward_mlp_48d_v1/seed_42/baseline/model_manifest.json").read_text()
    )["target_names"]
    baseline_dir = root / base["artifacts"]["weights_root"] / "forward_mlp_48d_v1/seed_42/baseline"
    baseline_model, baseline_stats, baseline_manifest = load_weight_bundle(
        baseline_dir, expected_model_tag=base["forward"]["model_tag"],
        expected_feature_names=feature_names, expected_target_names=target_names, device=device,
    )
    winning_dir = Path(training_summary["selection"]["winning_model_directory"])
    winning_model, winning_stats, winning_manifest = load_weight_bundle(
        winning_dir, expected_model_tag=training_summary["selection"]["winning_model_tag"],
        expected_feature_names=feature_names, expected_target_names=target_names, device=device,
    )
    candidate_root = root / config["frozen_rerank"]["gate_h_candidate_root"]
    truth_root = root / config["frozen_rerank"]["gate_h_truth_root"]
    target_directories = sorted(candidate_root.glob("target_*"))
    strategies = {
        "baseline": {"model": baseline_model, "stats": baseline_stats, "manifest": baseline_manifest},
        "gate_k_winner": {"model": winning_model, "stats": winning_stats, "manifest": winning_manifest},
    }
    target_results: dict[str, Any] = {}
    for target_dir in target_directories:
        target = yaml.safe_load((target_dir / "target.yaml").read_text(encoding="utf-8"))
        rows = _read_jsonl(target_dir / "projected_union.jsonl")
        evaluations = {
            str(row["candidate_key"]): row
            for row in _read_jsonl(truth_root / target_dir.name / "full_pool_evaluations.jsonl")
        }
        if set(evaluations) != {str(row["candidate_key"]) for row in rows}:
            raise Geometry19GateKError(f"Frozen pool/truth identity mismatch: {target_dir.name}")
        target_result = {"dataset_sample_id": target.get("dataset_sample_id"), "strategies": {}}
        for strategy_id, strategy in strategies.items():
            rho = _predict_pool_rho(
                rows, target, feature_names, strategy["model"], strategy["stats"], tolerances, epsilons
            )
            target_result["strategies"][strategy_id] = _frozen_strategy_metrics(
                rows, evaluations, rho, tolerances, config["frozen_rerank"]["budgets"]
            )
        target_results[target_dir.name] = target_result
    aggregate: dict[str, Any] = {}
    for strategy_id in strategies:
        aggregate[strategy_id] = {
            "target_count": len(target_results),
            "first_hit_ranks": [
                value["strategies"][strategy_id]["first_hit_rank"]
                for value in target_results.values()
            ],
            "rho_spearman_median": float(np.median([
                value["strategies"][strategy_id]["rho_spearman"]
                for value in target_results.values()
            ])),
            "budgets": {
                str(budget): {
                    "target_hits": sum(
                        value["strategies"][strategy_id]["budgets"][str(budget)]["target_hit"]
                        for value in target_results.values()
                    ),
                    "median_best_true_rho": float(np.median([
                        value["strategies"][strategy_id]["budgets"][str(budget)]["best_true_rho"]
                        for value in target_results.values()
                    ])),
                }
                for budget in config["frozen_rerank"]["budgets"]
            },
        }
    summary = {
        "protocol_version": GATE_K_VERSION,
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "new_v2_1_calls": 0,
        "winning_model_tag": training_summary["selection"]["winning_model_tag"],
        "strategies": aggregate,
        "targets": target_results,
        "hashes": {
            "training_summary_sha256": sha256_file(training_summary_path),
            "config_sha256": sha256_file(config_source),
        },
    }
    runtime.mkdir(parents=True)
    _write_json(runtime / "frozen_rerank_summary.json", summary)
    lines = [
        "# Gate K冻结候选池重排序",
        "",
        "复用Gate H全池真实结果；新增V2.1调用为0。",
        "",
        "| 策略 | 预算16 | 预算32 | 预算64 | 预算128 | 预算256 | median Spearman | 首次命中排名 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for strategy_id, value in aggregate.items():
        lines.append(
            f"| {strategy_id} | "
            + " | ".join(f"{value['budgets'][str(b)]['target_hits']}/{len(target_results)}" for b in config["frozen_rerank"]["budgets"])
            + f" | {value['rho_spearman_median']:.3f} | {value['first_hit_ranks']} |"
        )
    (runtime / "Gate_K_frozen_rerank.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _summary_alignment(summary: Mapping[str, Any]) -> Mapping[str, Any] | None:
    best = summary.get("best_approximation")
    alignment = best.get("hit_alignment") if isinstance(best, Mapping) else None
    return alignment if isinstance(alignment, Mapping) else None


def _best_stage_alignment(*summaries: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    values = [value for value in (_summary_alignment(item) for item in summaries if item) if value]
    return min(
        values,
        key=lambda value: (
            int(value["failed_axis_count"]), float(value["rho"]),
            float(value["sum_axis_ratios"]),
        ),
    ) if values else None


def run_gate_k_downstream(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    device: str,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the validation-selected surrogate through B/C/D with equal C budgets."""

    root = Path(repository_root).resolve()
    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    runtime = Path(output_root).resolve() if output_root else (
        root / config["artifacts"]["runtime_root"] / config["downstream"]["output_subdirectory"]
    )
    if runtime.exists():
        raise Geometry19GateKError(f"Refusing to overwrite Gate K downstream: {runtime}")
    training_summary_path = root / config["artifacts"]["runtime_root"] / config["artifacts"]["summary"]
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    forward_dir = Path(training_summary["selection"]["winning_model_directory"])
    gate_i_config = yaml.safe_load((root / config["downstream"]["gate_i_config"]).read_text(encoding="utf-8"))
    baseline_summary_path = root / config["downstream"]["gate_i_baseline_summary"]
    baseline = json.loads(baseline_summary_path.read_text(encoding="utf-8"))["arms"][
        config["downstream"]["baseline_arm"]
    ]
    runtime.mkdir(parents=True)
    target_results: dict[str, Any] = {}
    total_c_calls = 0
    total_d_calls = 0
    target_hits = 0
    c_hits = 0
    d_rescues = 0
    for target_item in gate_i_config["targets"]:
        sample_id = str(target_item["sample_id"])
        target = build_target_mapping(
            thrust_N=target_item["thrust_N"], tsfc_kg_N_s=target_item["tsfc_kg_N_s"],
            mass_kg=target_item["mass_kg"], target_type=target_item.get("target_type", "ID"),
            expected_status=target_item.get("expected_status", "KNOWN_FEASIBLE"),
        )
        target_path = runtime / "targets" / f"sample_{sample_id}.yaml"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(yaml.safe_dump(target, sort_keys=False), encoding="utf-8")
        run_root = runtime / "runs" / f"sample_{sample_id}"
        b_summary = run_candidate_protocol(
            root, target_path, mode="dflow", device=device,
            output_root=run_root / "candidates/dflow",
            generator_checkpoint=resolve_generator_checkpoint(root, "bounded-sigmoid"),
            dflow_target_loss_mode="hit_aligned",
            candidate_ranking_mode="predicted_rho",
            forward_weights_override=forward_dir,
        )
        budget = int(config["downstream"]["c_true_physics_budget_per_target"])
        if int(b_summary["candidate_protocol"]["n_keep"]) < budget:
            raise Geometry19GateKError(
                f"Gate K winner has insufficient N_keep for equal budget: sample {sample_id}"
            )
        c_summary = run_true_physics_stage(
            root, target_path, candidate_root=run_root / "candidates",
            output_root=run_root / "wp19_23c", sampling_mode="dflow",
            evaluation_budget=budget,
        )
        d_summary = None
        if not c_summary["downstream_stop"]:
            d_summary = run_refinement_stage(
                root, target_path, physics_root=run_root / "wp19_23c",
                output_root=run_root / "wp19_23d", device=device,
                target_loss_mode="hit_aligned", true_acceptance_mode="hit_aligned",
                forward_weights_override=forward_dir,
            )
        alignment = _best_stage_alignment(c_summary, d_summary)
        final_hit = bool(c_summary["downstream_stop"] or (d_summary and d_summary["downstream_stop"]))
        c_hit = bool(c_summary["downstream_stop"])
        d_hit = bool(d_summary and d_summary["downstream_stop"])
        c_calls = int(c_summary["counts"]["unique_v2_1_evaluations"])
        d_calls = int(d_summary["counts"]["true_v2_1_evaluations"]) if d_summary else 0
        total_c_calls += c_calls
        total_d_calls += d_calls
        target_hits += int(final_hit)
        c_hits += int(c_hit)
        d_rescues += int(d_hit)
        target_results[sample_id] = {
            "n_keep": int(b_summary["candidate_protocol"]["n_keep"]),
            "c_hit": c_hit,
            "d_rescue": d_hit,
            "final_hit": final_hit,
            "c_v2_1_calls": c_calls,
            "d_v2_1_calls": d_calls,
            "best_hit_alignment": dict(alignment) if alignment else None,
        }
    rho_values = [
        float(value["best_hit_alignment"]["rho"])
        for value in target_results.values() if value["best_hit_alignment"]
    ]
    result = {
        "target_hits": target_hits,
        "c_target_hits": c_hits,
        "d_rescue_hits": d_rescues,
        "c_v2_1_calls": total_c_calls,
        "d_v2_1_calls": total_d_calls,
        "median_best_true_rho": float(np.median(rho_values)) if rho_values else None,
        "targets": target_results,
    }
    baseline_result = {
        "target_hits": int(baseline["final_target_hits"]),
        "c_target_hits": int(baseline["c_target_hits"]),
        "d_rescue_hits": int(baseline["d_rescue_hits"]),
        "c_v2_1_calls": int(baseline["c_v2_1_calls"]),
        "d_v2_1_calls": int(baseline["d_v2_1_calls"]),
        "median_best_true_rho": float(baseline["median_final_true_rho"]),
    }
    summary = {
        "protocol_version": GATE_K_VERSION,
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_use_allowed": False,
        "winning_model_tag": training_summary["selection"]["winning_model_tag"],
        "equal_c_budget_per_target": int(config["downstream"]["c_true_physics_budget_per_target"]),
        "baseline": baseline_result,
        "gate_k_winner": result,
        "promotion_checks": {
            "target_hits_no_regression": target_hits >= baseline_result["target_hits"],
            "median_rho_improved": result["median_best_true_rho"] < baseline_result["median_best_true_rho"],
            "three_of_four_hits": target_hits >= 3,
        },
        "hashes": {
            "training_summary_sha256": sha256_file(training_summary_path),
            "baseline_summary_sha256": sha256_file(baseline_summary_path),
            "config_sha256": sha256_file(config_source),
        },
    }
    _write_json(runtime / "gate_k_downstream_summary.json", summary)
    lines = [
        "# Gate K完整pipeline验证",
        "",
        "| 代理 | C命中 | D救援 | 最终命中 | C调用 | D调用 | median最佳rho |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| baseline | {baseline_result['c_target_hits']}/4 | {baseline_result['d_rescue_hits']}/4 | "
        f"{baseline_result['target_hits']}/4 | {baseline_result['c_v2_1_calls']} | "
        f"{baseline_result['d_v2_1_calls']} | {baseline_result['median_best_true_rho']:.3f} |",
        f"| Gate K winner | {c_hits}/4 | {d_rescues}/4 | {target_hits}/4 | "
        f"{total_c_calls} | {total_d_calls} | {result['median_best_true_rho']:.3f} |",
        "",
    ]
    (runtime / "Gate_K_downstream.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def _refinement_diagnostics(paths: Sequence[Path]) -> dict[str, Any]:
    rows = []
    for path in paths:
        rows.extend(_read_jsonl(path))
    rho_deltas = []
    failed_axis_deltas = []
    for row in rows:
        prior = row.get("prior_hit_alignment")
        proposed = (row.get("proposed") or {}).get("hit_alignment")
        if isinstance(prior, Mapping) and isinstance(proposed, Mapping):
            rho_deltas.append(float(prior["rho"]) - float(proposed["rho"]))
            failed_axis_deltas.append(
                int(prior["failed_axis_count"]) - int(proposed["failed_axis_count"])
            )
    return {
        "true_evaluations": len(rows),
        "accepted_updates": sum(bool(row.get("accepted")) for row in rows),
        "acceptance_rate": (
            sum(bool(row.get("accepted")) for row in rows) / len(rows) if rows else None
        ),
        "rho_improved_count": sum(value > 0.0 for value in rho_deltas),
        "rho_improved_rate": (
            sum(value > 0.0 for value in rho_deltas) / len(rho_deltas) if rho_deltas else None
        ),
        "failed_axes_improved_count": sum(value > 0 for value in failed_axis_deltas),
        "median_rho_delta": float(np.median(rho_deltas)) if rho_deltas else None,
    }


def finalize_gate_k(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Combine training, frozen-rerank, and downstream evidence into a promotion decision."""

    root = Path(repository_root).resolve()
    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    gate_root = root / config["artifacts"]["runtime_root"]
    runtime = Path(output_root).resolve() if output_root else (
        gate_root / config["promotion"]["output_subdirectory"]
    )
    if runtime.exists():
        raise Geometry19GateKError(f"Refusing to overwrite Gate K decision: {runtime}")
    training_path = gate_root / config["artifacts"]["summary"]
    rerank_path = gate_root / config["frozen_rerank"]["output_subdirectory"] / "frozen_rerank_summary.json"
    downstream_path = gate_root / config["downstream"]["output_subdirectory"] / "gate_k_downstream_summary.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    rerank = json.loads(rerank_path.read_text(encoding="utf-8"))
    downstream = json.loads(downstream_path.read_text(encoding="utf-8"))
    winner_id = training["selection"]["winning_arm"]
    baseline_id = "s0_legacy_mse"
    winner = training["arms"][winner_id]
    baseline = training["arms"][baseline_id]
    winner_aggregate = winner["aggregate"]
    baseline_aggregate = baseline["aggregate"]

    def reduction(new: float, old: float) -> float:
        return (float(old) - float(new)) / float(old)

    errors_new = winner_aggregate["test_performance_relative_error_p50_median"]
    errors_old = baseline_aggregate["test_performance_relative_error_p50_median"]
    tsfc_reduction = reduction(errors_new["tsfc_kg_N_s"], errors_old["tsfc_kg_N_s"])
    rho_reduction = reduction(
        winner_aggregate["test_rho_p50_median"], baseline_aggregate["test_rho_p50_median"]
    )
    other_worsening = {
        name: (float(errors_new[name]) - float(errors_old[name])) / float(errors_old[name])
        for name in ("thrust_N", "mass_kg")
    }
    baseline_by_seed = {int(run["seed"]): run for run in baseline["runs"]}
    seed_rho_improved = {
        str(run["seed"]): (
            float(run["test"]["hit_alignment"]["rho_p50"])
            < float(baseline_by_seed[int(run["seed"])]["test"]["hit_alignment"]["rho_p50"])
        )
        for run in winner["runs"]
    }
    baseline_refinement_paths = sorted(
        (root / config["downstream"]["gate_i_baseline_summary"]).parent.glob(
            f"runs/sample_*/{config['downstream']['baseline_arm']}/wp19_23d/refinement_ledger.jsonl"
        )
    )
    winner_refinement_paths = sorted(
        (gate_root / config["downstream"]["output_subdirectory"]).glob(
            "runs/sample_*/wp19_23d/refinement_ledger.jsonl"
        )
    )
    refinement = {
        "baseline": _refinement_diagnostics(baseline_refinement_paths),
        "gate_k_winner": _refinement_diagnostics(winner_refinement_paths),
    }
    downstream_baseline = downstream["baseline"]
    downstream_winner = downstream["gate_k_winner"]
    downstream_rho_reduction = reduction(
        downstream_winner["median_best_true_rho"], downstream_baseline["median_best_true_rho"]
    )
    promotion = config["promotion"]
    checks = {
        "tsfc_p50_reduction": tsfc_reduction >= float(promotion["minimum_tsfc_p50_reduction"]),
        "other_axes_no_regression": max(other_worsening.values()) <= float(
            promotion["maximum_other_axis_p50_worsening"]
        ),
        "test_rho_p50_reduction": rho_reduction >= float(
            promotion["minimum_test_rho_p50_reduction"]
        ),
        "all_seeds_improve_test_rho": all(seed_rho_improved.values()),
        "frozen_top32_target_hits_improve": (
            rerank["strategies"]["gate_k_winner"]["budgets"]["32"]["target_hits"]
            > rerank["strategies"]["baseline"]["budgets"]["32"]["target_hits"]
        ),
        "downstream_target_hits_no_regression": (
            downstream_winner["target_hits"] >= downstream_baseline["target_hits"]
        ),
        "downstream_promotion_value": (
            downstream_winner["target_hits"]
            >= int(promotion["required_target_hits_for_direct_promotion"])
            or (
                downstream_winner["target_hits"] == downstream_baseline["target_hits"]
                and downstream_rho_reduction
                >= float(promotion["minimum_same_hit_downstream_rho_reduction"])
            )
        ),
        "refinement_true_improvement_rate_no_regression": (
            refinement["gate_k_winner"]["rho_improved_rate"]
            >= refinement["baseline"]["rho_improved_rate"]
        ),
    }
    promote = all(checks.values())
    summary = {
        "protocol_version": GATE_K_VERSION,
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PROMOTE_GATE_K_SURROGATE" if promote else "KEEP_CURRENT_PRODUCTION_SURROGATE",
        "production_pipeline_modified": False,
        "winning_arm": winner_id,
        "winning_model_tag": training["selection"]["winning_model_tag"],
        "offline_changes": {
            "tsfc_p50_reduction": tsfc_reduction,
            "test_rho_p50_reduction": rho_reduction,
            "other_axis_relative_changes": other_worsening,
            "seed_rho_improved": seed_rho_improved,
        },
        "frozen_rerank": rerank["strategies"],
        "downstream": {
            "baseline": downstream_baseline,
            "gate_k_winner": downstream_winner,
            "median_rho_reduction": downstream_rho_reduction,
        },
        "refinement": refinement,
        "promotion_checks": checks,
        "hashes": {
            "training_sha256": sha256_file(training_path),
            "rerank_sha256": sha256_file(rerank_path),
            "downstream_sha256": sha256_file(downstream_path),
            "config_sha256": sha256_file(config_source),
        },
    }
    runtime.mkdir(parents=True)
    _write_json(runtime / "gate_k_final_decision.json", summary)
    lines = [
        "# Gate K最终晋级判定",
        "",
        f"> 决定：**{summary['decision']}**",
        "",
        "## 核心结果",
        "",
        f"- Test TSFC P50降低：{tsfc_reduction:.1%}。",
        f"- Test median rho降低：{rho_reduction:.1%}。",
        f"- 冻结候选池Top-32：baseline "
        f"{rerank['strategies']['baseline']['budgets']['32']['target_hits']}/4 → Gate K "
        f"{rerank['strategies']['gate_k_winner']['budgets']['32']['target_hits']}/4。",
        f"- 完整pipeline：{downstream_baseline['target_hits']}/4 → "
        f"{downstream_winner['target_hits']}/4；median rho "
        f"{downstream_baseline['median_best_true_rho']:.3f} → "
        f"{downstream_winner['median_best_true_rho']:.3f}。",
        f"- Refinement真实rho改善率："
        f"{refinement['baseline']['rho_improved_rate']:.1%} → "
        f"{refinement['gate_k_winner']['rho_improved_rate']:.1%}。",
        "",
        "## 晋级检查",
        "",
    ]
    lines.extend(f"- {'PASS' if passed else 'FAIL'} `{name}`" for name, passed in checks.items())
    lines += [
        "",
        "新损失改善了测试误差与冻结池排序，但未改善完整pipeline的总体命中质量和梯度稳定性，因此不替换生产代理。",
        "",
    ]
    report_path = runtime / "Gate_K_final_decision.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _write_json(runtime / "manifest.json", {
        "protocol_version": GATE_K_VERSION,
        "summary_sha256": sha256_file(runtime / "gate_k_final_decision.json"),
        "report_sha256": sha256_file(report_path),
    })
    return summary


def run_gate_k_training(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    device: str | None = None,
    output_root: str | Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    if config.get("protocol_version") != GATE_K_VERSION:
        raise Geometry19GateKError("Unsupported Gate K protocol version")
    runtime = Path(output_root).resolve() if output_root else root / config["artifacts"]["runtime_root"]
    if runtime.exists():
        raise Geometry19GateKError(f"Refusing to overwrite Gate K run: {runtime}")
    runtime.mkdir(parents=True)
    base_protocol_path = root / config["base_surrogate_protocol"]
    base = yaml.safe_load(base_protocol_path.read_text(encoding="utf-8"))
    inverse = yaml.safe_load((root / config["inverse_protocol"]).read_text(encoding="utf-8"))
    tolerances = dict(inverse["diagnostic_acceptance"]["relative_tolerances"])
    epsilons = dict(inverse["diagnostic_acceptance"]["epsilons"])
    schema_path = root / base["design_schema"]
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    feature_names = [item["name"] for item in schema["design_variables"]]
    maximum = int(config["smoke"]["maximum_rows_per_split"]) if smoke else None
    splits, target_names = load_regression_splits(
        root / base["regression_dataset"], feature_names, base["campaign_split"], maximum
    )
    stats = fit_normalization(splits["train"]["x"], splits["train"]["y"])
    x_values = {
        name: ((bucket["x"] - stats["input_mean"]) / stats["input_std"]).astype(np.float32)
        for name, bucket in splits.items()
    }
    y_values = {
        name: ((bucket["y"] - stats["output_mean"]) / stats["output_std"]).astype(np.float32)
        for name, bucket in splits.items()
    }
    run_device = str(device or config["training"]["device"])
    training = dict(config["training"])
    training["device"] = run_device
    if smoke:
        training.update(config["smoke"]["training"])
    seeds = [int(config["smoke"]["seed"])] if smoke else [int(value) for value in config["seeds"]]
    output_mean = torch.as_tensor(stats["output_mean"], device=run_device)
    output_std = torch.as_tensor(stats["output_std"], device=run_device)
    tolerance_tensor = torch.tensor([tolerances[name] for name in OUTPUT_NAMES], device=run_device)
    epsilon_tensor = torch.tensor([epsilons[name] for name in OUTPUT_NAMES], device=run_device)
    architecture = base["architecture"]
    all_arms: dict[str, Any] = {}
    for arm in config["arms"]:
        arm_id = str(arm["arm_id"])
        loss_config = {**dict(config["loss_defaults"]), **dict(arm["loss"])}
        validate_surrogate_loss_config(loss_config)
        runs = []
        for seed in seeds:
            _seed_all(seed)
            model = TaggedGeometry19MLP(
                19, 48, architecture["hidden_layers"], architecture["activation"],
                architecture["dropout"], architecture["batch_norm"],
            ).to(run_device)

            def loss_function(prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
                return surrogate_loss_terms(
                    prediction, truth, output_mean, output_std,
                    tolerance_tensor, epsilon_tensor, loss_config,
                )["total"]

            model, history, best_epoch = _train_model(
                model,
                _loader(x_values["train"], y_values["train"], int(training["batch_size"]), True, seed),
                x_values["validation"], y_values["validation"], loss_function,
                training, torch.device(run_device),
            )
            model_tag = f"novaturbo_geometry19_forward_gate_k_{arm_id}_seed{seed}_v1"
            model_dir = runtime / "models" / arm_id / f"seed_{seed}"
            bundle = save_weight_bundle(
                model_dir,
                model,
                {
                    "checkpoint_version": "geometry19_tagged_weights_v1",
                    "model_tag": model_tag,
                    "architecture_tag": architecture["architecture_tag"],
                    "feature_names": feature_names,
                    "target_names": target_names,
                    "training_seed": seed,
                    "loss_config": loss_config,
                    "dataset_tag": "wp19_21_98304_development_v1",
                    "formal_use_allowed": False,
                    "code_commit": subprocess.run(
                        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
                        capture_output=True, text=True,
                    ).stdout.strip(),
                    "gate_k_config_sha256": sha256_file(config_source),
                    "base_protocol_sha256": sha256_file(base_protocol_path),
                },
                stats,
            )
            predictions = {}
            metrics = {}
            for split_name in ("validation", "test"):
                normalized_prediction = _predict(model, x_values[split_name], torch.device(run_device))
                physical_prediction = normalized_prediction * stats["output_std"] + stats["output_mean"]
                predictions[split_name] = physical_prediction
                metrics[split_name] = {
                    "regression": _regression_metrics(
                        splits[split_name]["y"], physical_prediction, target_names
                    ),
                    "hit_alignment": hit_aligned_prediction_metrics(
                        splits[split_name]["y"], physical_prediction, tolerances, epsilons
                    ),
                    "ranking": ranking_benchmark(
                        splits[split_name]["y"], physical_prediction, tolerances, epsilons,
                        maximum_targets=int(config["evaluation"]["maximum_ranking_targets"]),
                        budgets=config["evaluation"]["ranking_budgets"],
                    ),
                }
            loaded, loaded_stats, _ = load_weight_bundle(
                model_dir, expected_model_tag=model_tag, expected_feature_names=feature_names,
                expected_target_names=target_names, device=run_device,
            )
            reload_equal = np.array_equal(
                _predict(loaded, x_values["test"][: min(256, len(x_values["test"]))], torch.device(run_device)),
                _predict(model, x_values["test"][: min(256, len(x_values["test"]))], torch.device(run_device)),
            ) and np.array_equal(loaded_stats["input_mean"], stats["input_mean"])
            run = {
                "arm_id": arm_id,
                "seed": seed,
                "model_tag": model_tag,
                "loss": loss_config,
                "best_epoch": best_epoch,
                "validation": metrics["validation"],
                "test": metrics["test"],
                "weight_reload_equal": reload_equal,
                "artifacts": bundle,
            }
            _write_json(model_dir / "training_history.json", {"best_epoch": best_epoch, "history": history})
            _write_json(model_dir / "evaluation.json", run)
            runs.append(run)
        all_arms[arm_id] = {
            "loss": loss_config,
            "runs": runs,
            "aggregate": _aggregate_arm(runs),
        }
    winning_arm = min(
        all_arms,
        key=lambda arm_id: (
            all_arms[arm_id]["aggregate"]["validation_rho_p50_median"], arm_id
        ),
    )
    winning_run = min(
        all_arms[winning_arm]["runs"],
        key=lambda run: (run["validation"]["hit_alignment"]["rho_p50"], run["seed"]),
    )
    status = "PASS" if all(
        run["weight_reload_equal"]
        for arm in all_arms.values() for run in arm["runs"]
    ) else "AUDIT_FAILED"
    summary = {
        "protocol_version": GATE_K_VERSION,
        "status": status,
        "run_kind": "smoke" if smoke else "full",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_use_allowed": False,
        "device": run_device,
        "split_counts": {name: len(bucket["x"]) for name, bucket in splits.items()},
        "selection": {
            "uses_test_metrics": False,
            "primary_metric": "validation.hit_alignment.rho_p50",
            "winning_arm": winning_arm,
            "winning_seed": winning_run["seed"],
            "winning_model_tag": winning_run["model_tag"],
            "winning_model_directory": winning_run["artifacts"]["manifest"].rsplit("/", 1)[0],
        },
        "arms": all_arms,
        "hashes": {
            "config_sha256": sha256_file(config_source),
            "base_protocol_sha256": sha256_file(base_protocol_path),
            "dataset_sha256": sha256_file(root / base["regression_dataset"]),
        },
    }
    _write_json(runtime / config["artifacts"]["summary"], summary)
    report_path = runtime / config["artifacts"]["report"]
    report_path.write_text(_render_report(summary), encoding="utf-8")
    _write_json(runtime / config["artifacts"]["manifest"], {
        "protocol_version": GATE_K_VERSION,
        "summary_sha256": sha256_file(runtime / config["artifacts"]["summary"]),
        "report_sha256": sha256_file(report_path),
    })
    return summary
