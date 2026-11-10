"""Frozen, anchor-controlled confirmation campaign for Geometry19 reranking."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest
import torch
import yaml

from src.ai.geometry19_pipeline import build_target_mapping
from src.ai.geometry19_reranker import (
    PROTOCOL_VERSION as RERANKER_PROTOCOL_VERSION,
    PairwiseReranker,
    _ensemble_prediction,
    _feature_matrix,
    _scores,
    ranking_metrics,
)
from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_training import PERFORMANCE_NAMES
from src.physics import Geometry19


PROTOCOL_VERSION = "wp19_23_reranker_confirmation_v1"


class RerankerConfirmationError(ValueError):
    """Raised when the frozen confirmation contract is violated."""


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


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _stored_acceptable(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("solver_status", {}).get("converged")
        and row.get("solver_status", {}).get("conservation_passed")
        and row.get("constraint_values", {}).get("assessment", {}).get("accepted")
        and row.get("physics_outputs", {}).get("performance")
    )


def _performance(row: Mapping[str, Any]) -> np.ndarray:
    values = row["physics_outputs"]["performance"]
    return np.asarray([values[name] for name in PERFORMANCE_NAMES], dtype=np.float64)


def select_spread_targets(rows: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    """Deterministically cover the held-out performance domain without scorer access."""
    if count <= 0 or count > len(rows):
        raise RerankerConfirmationError("Invalid target count")
    ordered = sorted(rows, key=lambda row: int(row["sample_id"]))
    points = np.asarray([_performance(row) for row in ordered], dtype=np.float64)
    points[:, 1:] = np.log(points[:, 1:])
    mean = points.mean(axis=0)
    std = points.std(axis=0)
    std[std < 1.0e-12] = 1.0
    normalized = (points - mean) / std
    first = int(np.argmin(np.linalg.norm(normalized, axis=1)))
    selected = [first]
    minimum = np.linalg.norm(normalized - normalized[first], axis=1)
    while len(selected) < count:
        minimum[selected] = -1.0
        index = int(np.argmax(minimum))
        selected.append(index)
        minimum = np.minimum(minimum, np.linalg.norm(normalized - normalized[index], axis=1))
    return [dict(ordered[index]) for index in selected]


def _rho(performance: Sequence[float], target: Sequence[float], tolerances: np.ndarray) -> float:
    performance_array = np.asarray(performance, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    return float(
        np.max(
            np.abs(performance_array - target_array)
            / np.maximum(np.abs(target_array), 1.0e-15)
            / tolerances
        )
    )


def freeze_confirmation_campaign(
    root: str | Path,
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze targets and pools before any scorer is evaluated."""
    repo = Path(root).resolve()
    source = Path(config_path).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise RerankerConfirmationError("Protocol mismatch")
    if config.get("pipeline_mutation_allowed") is not False:
        raise RerankerConfirmationError("Confirmation campaign must forbid pipeline mutation")
    runtime = Path(output_root).resolve() if output_root else repo / config["artifacts"]["runtime_root"]
    if runtime.exists():
        raise RerankerConfirmationError(f"Refusing to overwrite {runtime}")
    runtime.mkdir(parents=True)

    prior_suite_path = repo / config["source"]["prior_target_suite"]
    prior_sample_ids = {int(row["sample_id"]) for row in _read_jsonl(prior_suite_path)}
    dataset_path = repo / config["source"]["regression_dataset"]
    campaign = str(config["source"]["source_campaign"])
    candidates: list[dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (
                row["campaign_id"] == campaign
                and int(row["sample_id"]) not in prior_sample_ids
                and _stored_acceptable(row)
            ):
                candidates.append(row)
    target_count = int(config["source"]["target_count"])
    pool_size = int(config["source"]["pool_size"])
    if len(candidates) < max(target_count, pool_size + 1):
        raise RerankerConfirmationError("Insufficient unused acceptable held-out rows")
    selected = select_spread_targets(candidates, target_count)

    inverse = yaml.safe_load((repo / config["inverse_protocol"]).read_text(encoding="utf-8"))
    tolerances = np.asarray(
        [inverse["diagnostic_acceptance"]["relative_tolerances"][name] for name in PERFORMANCE_NAMES],
        dtype=np.float64,
    )
    suite: list[dict[str, Any]] = []
    for source_row in selected:
        performance = source_row["physics_outputs"]["performance"]
        target = build_target_mapping(
            thrust_N=performance["thrust_N"],
            tsfc_kg_N_s=performance["tsfc_kg_N_s"],
            mass_kg=performance["mass_kg"],
            target_type="ID",
            expected_status="KNOWN_FEASIBLE",
        )
        suite.append(
            {
                "target_id": target["target_id"],
                "source_campaign": campaign,
                "source_sample_id": int(source_row["sample_id"]),
                "target": target,
                "reference_x_19": source_row["x_19"],
            }
        )

    library = [
        {
            "sample_id": int(row["sample_id"]),
            "source_campaign": campaign,
            "x_19": row["x_19"],
            "performance": {
                name: float(row["physics_outputs"]["performance"][name])
                for name in PERFORMANCE_NAMES
            },
            "physics_acceptable": True,
        }
        for row in sorted(candidates, key=lambda row: int(row["sample_id"]))
    ]
    by_sample = {row["sample_id"]: row for row in library}
    all_sample_ids = np.asarray(sorted(by_sample), dtype=np.int64)
    ledger: list[dict[str, Any]] = []
    for target_row in suite:
        anchor = int(target_row["source_sample_id"])
        eligible = all_sample_ids[all_sample_ids != anchor]
        target_vector = [target_row["target"][name] for name in PERFORMANCE_NAMES]
        for pool_seed in config["source"]["pool_seeds"]:
            derived_seed = int(pool_seed) ^ (anchor * 2654435761 % (2**32))
            rng = np.random.default_rng(derived_seed)
            decoys = rng.choice(eligible, size=pool_size - 1, replace=False).tolist()
            members = [anchor, *[int(value) for value in decoys]]
            rng.shuffle(members)
            run_id = f"{target_row['target_id']}:{int(pool_seed)}"
            for candidate_id, sample_id in enumerate(members):
                candidate = by_sample[sample_id]
                performance = [candidate["performance"][name] for name in PERFORMANCE_NAMES]
                ledger.append(
                    {
                        "run_id": run_id,
                        "target_id": target_row["target_id"],
                        "pool_seed": int(pool_seed),
                        "candidate_id": candidate_id,
                        "candidate_sample_id": sample_id,
                        "is_reference_anchor": sample_id == anchor,
                        "physics_acceptable": True,
                        "rho": _rho(performance, target_vector, tolerances),
                    }
                )

    target_path = runtime / config["artifacts"]["target_suite"]
    library_path = runtime / config["artifacts"]["candidate_library"]
    ledger_path = runtime / config["artifacts"]["candidate_ledger"]
    _write_jsonl(target_path, suite)
    _write_jsonl(library_path, library)
    _write_jsonl(ledger_path, ledger)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN_NO_SCORES_COMPUTED",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_mutation_allowed": False,
        "source_campaign": campaign,
        "excluded_prior_target_sample_count": len(prior_sample_ids),
        "target_count": len(suite),
        "pool_seed_count": len(config["source"]["pool_seeds"]),
        "run_count": len(suite) * len(config["source"]["pool_seeds"]),
        "pool_size": pool_size,
        "ledger_row_count": len(ledger),
        "reference_anchor_count": sum(row["is_reference_anchor"] for row in ledger),
        "selection_method": "deterministic_farthest_performance_coverage_without_scorer_access",
        "pool_method": "one_randomized_reference_anchor_plus_uniform_heldout_decoys",
        "hashes": {
            "config": sha256_file(source),
            "source_dataset": sha256_file(dataset_path),
            "prior_target_suite": sha256_file(prior_suite_path),
            "target_suite": sha256_file(target_path),
            "candidate_library": sha256_file(library_path),
            "candidate_ledger": sha256_file(ledger_path),
        },
    }
    _write_json(runtime / config["artifacts"]["freeze_manifest"], manifest)
    return manifest


def cluster_bootstrap_difference(
    baseline_success: Mapping[str, Mapping[str, bool]],
    challenger_success: Mapping[str, Mapping[str, bool]],
    run_target: Mapping[str, str],
    budget: int,
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float]:
    targets = sorted(set(run_target.values()))
    per_target = []
    for target in targets:
        run_ids = [run_id for run_id, value in run_target.items() if value == target]
        per_target.append(
            np.mean(
                [
                    float(challenger_success[run_id][str(budget)])
                    - float(baseline_success[run_id][str(budget)])
                    for run_id in run_ids
                ]
            )
        )
    values = np.asarray(per_target, dtype=np.float64)
    rng = np.random.default_rng(seed + budget)
    samples = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    alpha = 1.0 - confidence_level
    return {
        "point": float(values.mean()),
        "lower": float(np.quantile(samples, alpha / 2.0)),
        "upper": float(np.quantile(samples, 1.0 - alpha / 2.0)),
    }


def _run_success(metrics: Mapping[str, Any]) -> dict[str, dict[str, bool]]:
    return {row["run_id"]: dict(row["success"]) for row in metrics["runs"]}


def _paired_result(
    baseline: Mapping[str, Any], challenger: Mapping[str, Any], budget: int
) -> dict[str, Any]:
    baseline_runs = _run_success(baseline)
    challenger_runs = _run_success(challenger)
    gains = sum(
        not baseline_runs[run_id][str(budget)] and challenger_runs[run_id][str(budget)]
        for run_id in baseline_runs
    )
    losses = sum(
        baseline_runs[run_id][str(budget)] and not challenger_runs[run_id][str(budget)]
        for run_id in baseline_runs
    )
    p_value = float(binomtest(min(gains, losses), gains + losses, 0.5).pvalue) if gains + losses else 1.0
    return {"gains": gains, "losses": losses, "two_sided_exact_p": p_value}


def _render(summary: Mapping[str, Any]) -> str:
    lines = [
        "# WP19-23 独立 Reranker 冻结确认实验",
        "",
        f"> 结论：**{summary['decision']['evidence_status']}**  ",
        "> 锚点控制排序压力测试；不是生成器端到端命中率。无论结果如何均不接入 pipeline。",
        "",
        "## 等预算结果",
        "",
        "| Scorer | Recall@8 | @16 | @32 | @64 | @128 | 首命中排名P50 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scorer in ("shared48_s0", "performance3_log", "independent_reranker"):
        metrics = summary["evaluation"][scorer]
        recall = metrics["hit_bearing_recall"]
        lines.append(
            f"| {scorer} | {recall['8']:.1%} | {recall['16']:.1%} | {recall['32']:.1%} | "
            f"{recall['64']:.1%} | {recall['128']:.1%} | {metrics['median_first_hit_rank']:.1f} |"
        )
    comparison = summary["comparison"]["reranker_vs_performance3_log"]
    interval = comparison["cluster_bootstrap"]["32"]
    paired = comparison["paired"]["32"]
    lines += [
        "",
        "## 预注册主比较",
        "",
        f"- reranker 相对 performance3_log 的 Recall@32 差值：{interval['point']:+.1%}；",
        f"- 目标聚类 95% bootstrap CI：[{interval['lower']:+.1%}, {interval['upper']:+.1%}]；",
        f"- 配对新增/损失命中运行：{paired['gains']}/{paired['losses']}，双侧精确 p={paired['two_sided_exact_p']:.4g}；",
        "- pipeline action：**NONE，等待用户确认**。",
        "",
    ]
    return "\n".join(lines)


def evaluate_confirmation_campaign(
    root: str | Path,
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Evaluate three frozen scorers after verifying the campaign hashes."""
    repo = Path(root).resolve()
    source = Path(config_path).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise RerankerConfirmationError("Protocol mismatch")
    if config.get("pipeline_mutation_allowed") is not False:
        raise RerankerConfirmationError("Pipeline mutation must remain forbidden")
    runtime = Path(output_root).resolve() if output_root else repo / config["artifacts"]["runtime_root"]
    summary_path = runtime / config["artifacts"]["summary"]
    report_path = runtime / config["artifacts"]["report"]
    if summary_path.exists() or report_path.exists():
        raise RerankerConfirmationError("Refusing to overwrite a completed evaluation")
    manifest = json.loads((runtime / config["artifacts"]["freeze_manifest"]).read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_NO_SCORES_COMPUTED":
        raise RerankerConfirmationError("Campaign was not frozen before evaluation")
    if manifest["hashes"]["config"] != sha256_file(source):
        raise RerankerConfirmationError("Config changed after campaign freeze")
    for name in ("target_suite", "candidate_library", "candidate_ledger"):
        path = runtime / config["artifacts"][name]
        if manifest["hashes"][name] != sha256_file(path):
            raise RerankerConfirmationError(f"Frozen artifact changed: {name}")

    protected_before = {
        value: sha256_file(repo / value) for value in config["protected_pipeline_files"]
    }
    suite = _read_jsonl(runtime / config["artifacts"]["target_suite"])
    library = _read_jsonl(runtime / config["artifacts"]["candidate_library"])
    ledger = _read_jsonl(runtime / config["artifacts"]["candidate_ledger"])
    target_by_id = {row["target_id"]: row["target"] for row in suite}
    library_by_id = {int(row["sample_id"]): row for row in library}
    feature_names = list(Geometry19.coordinate_names())
    used_sample_ids = sorted({int(row["candidate_sample_id"]) for row in ledger})
    sample_index = {sample_id: index for index, sample_id in enumerate(used_sample_ids)}
    unique_x = np.asarray(
        [
            [library_by_id[sample_id]["x_19"][name] for name in feature_names]
            for sample_id in used_sample_ids
        ],
        dtype=np.float32,
    )
    torch_device = torch.device(device or "cuda:0")
    seeds = [int(value) for value in config["forward_model_seeds"]]
    s0_unique, _ = _ensemble_prediction(
        repo / config["baseline_model_root"], seeds, feature_names, unique_x, torch_device, shared48=True
    )
    performance_unique, performance_std_unique = _ensemble_prediction(
        repo / config["performance_model_root"], seeds, feature_names, unique_x, torch_device, shared48=False
    )
    member_indices = np.asarray(
        [sample_index[int(row["candidate_sample_id"])] for row in ledger], dtype=np.int64
    )
    row_x = unique_x[member_indices]
    targets = np.asarray(
        [
            [target_by_id[str(row["target_id"])][name] for name in PERFORMANCE_NAMES]
            for row in ledger
        ],
        dtype=np.float64,
    )
    inverse = yaml.safe_load((repo / config["inverse_protocol"]).read_text(encoding="utf-8"))
    tolerances = np.asarray(
        [inverse["diagnostic_acceptance"]["relative_tolerances"][name] for name in PERFORMANCE_NAMES],
        dtype=np.float64,
    )
    raw_features, s0_rho, performance_rho = _feature_matrix(
        row_x,
        targets,
        s0_unique[member_indices],
        performance_unique[member_indices],
        performance_std_unique[member_indices],
        tolerances,
    )

    reranker_config = yaml.safe_load((repo / config["reranker_config"]).read_text(encoding="utf-8"))
    if reranker_config.get("protocol_version") != RERANKER_PROTOCOL_VERSION:
        raise RerankerConfirmationError("Reranker protocol mismatch")
    reranker_runtime = repo / config["reranker_runtime"]
    reranker_summary = json.loads((reranker_runtime / "summary.json").read_text(encoding="utf-8"))
    selected_seed = int(reranker_summary["selected_seed"])
    model_dir = reranker_runtime / "models" / f"seed_{selected_seed}"
    normalization = np.load(model_dir / "feature_normalization.npz")
    mean = normalization["mean"]
    std = normalization["std"]
    if raw_features.shape[1] != len(mean):
        raise RerankerConfirmationError("Reranker feature contract changed")
    model = PairwiseReranker(raw_features.shape[1], reranker_config["training"]["hidden_layers"]).to(torch_device)
    model.load_state_dict(
        torch.load(model_dir / "reranker_state.pt", map_location=torch_device, weights_only=True)
    )
    normalized = ((raw_features - mean) / std).astype(np.float32)
    reranker_scores = _scores(model, normalized, torch_device)
    budgets = [int(value) for value in config["ranking_budgets"]]
    evaluation = {
        "shared48_s0": ranking_metrics(ledger, -s0_rho, budgets),
        "performance3_log": ranking_metrics(ledger, -performance_rho, budgets),
        "independent_reranker": ranking_metrics(ledger, reranker_scores, budgets),
    }
    if any(metrics["hit_bearing_run_count"] != manifest["run_count"] for metrics in evaluation.values()):
        raise RerankerConfirmationError("Anchor-controlled campaign contains a non-hit-bearing run")
    run_target = {
        row["run_id"]: row["target_id"] for row in evaluation["shared48_s0"]["runs"]
    }
    stats = config["statistics"]
    comparisons: dict[str, Any] = {}
    for name, baseline_name in (
        ("reranker_vs_shared48_s0", "shared48_s0"),
        ("reranker_vs_performance3_log", "performance3_log"),
    ):
        baseline = evaluation[baseline_name]
        challenger = evaluation["independent_reranker"]
        baseline_success = _run_success(baseline)
        challenger_success = _run_success(challenger)
        comparisons[name] = {
            "cluster_bootstrap": {
                str(budget): cluster_bootstrap_difference(
                    baseline_success,
                    challenger_success,
                    run_target,
                    budget,
                    replicates=int(stats["cluster_bootstrap_replicates"]),
                    seed=int(stats["bootstrap_seed"]),
                    confidence_level=float(stats["confidence_level"]),
                )
                for budget in budgets
            },
            "paired": {
                str(budget): _paired_result(baseline, challenger, budget) for budget in budgets
            },
        }
    primary = comparisons["reranker_vs_performance3_log"]["cluster_bootstrap"]
    decision = config["decision"]
    evidence = bool(
        primary["32"]["point"] >= float(decision["minimum_recall_at_32_absolute_gain"])
        and (
            not decision["require_recall_at_64_non_degradation"]
            or primary["64"]["point"] >= 0.0
        )
        and (
            not decision["require_cluster_ci_lower_bound_nonnegative"]
            or primary["32"]["lower"] >= 0.0
        )
    )
    protected_after = {
        value: sha256_file(repo / value) for value in config["protected_pipeline_files"]
    }
    if protected_after != protected_before:
        raise RerankerConfirmationError("A protected pipeline file changed during evaluation")
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "COMPLETE_EVALUATION_ONLY",
        "formal_use_allowed": False,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": manifest,
        "selected_reranker_seed": selected_seed,
        "evaluation": evaluation,
        "comparison": comparisons,
        "decision": {
            "evidence_status": "RERANKER_CONFIRMED_OVER_PERFORMANCE3_LOG" if evidence else "RERANKER_NOT_CONFIRMED_OVER_PERFORMANCE3_LOG",
            "pre_registered_evidence_gate_passed": evidence,
            "pipeline_action": "NONE",
            "pipeline_mutation_allowed": False,
            "awaiting_explicit_user_confirmation": True,
        },
        "protected_pipeline_hashes_before": protected_before,
        "protected_pipeline_hashes_after": protected_after,
    }
    _write_json(summary_path, summary)
    report_path.write_text(_render(summary), encoding="utf-8")
    return summary
