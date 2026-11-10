"""E1: frozen-CFM comparison of S0 and performance3_log rho ranking."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest
import torch
import yaml

from src.ai.generators import CFMGenerator, load_generator
from src.ai.geometry19_candidates import sample_device_invariant_sources
from src.ai.geometry19_cfm import integrate_cfm_sources
from src.ai.geometry19_generator_oracle import (
    _condition_features,
    _filter_samples,
    _physics_worker,
    _select_split,
    _worker_init,
)
from src.ai.geometry19_inverse_contract import Geometry19InverseTarget, load_wp19_23_contract
from src.ai.geometry19_pipeline import build_target_mapping
from src.ai.geometry19_reranker import _ensemble_prediction, ranking_metrics
from src.ai.geometry19_reranker_confirmation import cluster_bootstrap_difference
from src.ai.geometry19_surrogate import sha256_file
from src.ai.geometry19_training import PERFORMANCE_NAMES
from src.physics import Geometry19, ModelConfig
from src.physics.wp19_21_dataset import run_wp19_21


PROTOCOL_VERSION = "wp19_23_e1_performance3_pipeline_ranking_v1"


class E1PerformanceRankingError(ValueError):
    """Raised when an E1 freeze or evaluation contract is violated."""


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


def _target_id(row: Mapping[str, Any]) -> str:
    if "target_id" in row:
        return str(row["target_id"])
    return str(row["target"]["target_id"])


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise E1PerformanceRankingError("Protocol mismatch")
    if config.get("pipeline_mutation_allowed") is not False:
        raise E1PerformanceRankingError("E1 must forbid pipeline mutation")
    if config.get("integration_policy") != "forbidden_until_explicit_user_confirmation":
        raise E1PerformanceRankingError("E1 integration policy is not locked")
    return config


def _runtime(repo: Path, config: Mapping[str, Any], output_root: str | Path | None) -> Path:
    return Path(output_root).resolve() if output_root else repo / config["artifacts"]["runtime_root"]


def _performance(row: Mapping[str, Any]) -> np.ndarray:
    values = row["physics_outputs"]["performance"]
    return np.asarray([values[name] for name in PERFORMANCE_NAMES], dtype=np.float64)


def freeze_e1_campaign(
    root: str | Path,
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    device: str | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Generate fresh V2.1 targets and freeze bounded-CFM candidate geometry."""
    repo = Path(root).resolve()
    source = Path(config_path).resolve()
    config = _load_config(source)
    runtime = _runtime(repo, config, output_root)
    if runtime.exists():
        raise E1PerformanceRankingError(f"Refusing to overwrite {runtime}")
    runtime.mkdir(parents=True)

    target_spec = config["target_bank"]
    smoke_spec = config["smoke"]
    bank_count = int(smoke_spec["target_bank_sample_count"] if smoke else target_spec["sample_count"])
    target_count = int(smoke_spec["target_count"] if smoke else target_spec["target_count"])
    bank_root = runtime / config["artifacts"]["target_bank_root"]
    run_wp19_21(
        repo / target_spec["protocol"],
        repo,
        output_root_override=bank_root,
        sample_count_override=bank_count,
        preview_budget_override=0,
        workers_override=int(smoke_spec["physics_workers"] if smoke else target_spec["workers"]),
        shard_size_override=min(bank_count, int(target_spec["shard_size"])),
        seed_override=int(target_spec["seed"]),
        campaign_id=str(target_spec["campaign_id"]),
    )
    fresh = [row for row in _read_jsonl(bank_root / "dataset.jsonl") if _stored_acceptable(row)]
    if len(fresh) < target_count:
        raise E1PerformanceRankingError(
            f"Fresh target bank has only {len(fresh)} acceptable rows for {target_count} targets"
        )
    reference = _read_jsonl(repo / target_spec["reference_training_dataset"])
    train_campaigns = set(target_spec["reference_train_campaigns"])
    train = [row for row in reference if row["campaign_id"] in train_campaigns]
    train_points = _condition_features(np.asarray([_performance(row) for row in train]))
    selected = _select_split(fresh, train_points, target_count, "locked_held_out")

    excluded_ids: set[str] = set()
    excluded_hashes: dict[str, str] = {}
    for relative in target_spec["excluded_target_suites"]:
        path = repo / relative
        excluded_hashes[str(relative)] = sha256_file(path)
        excluded_ids.update(_target_id(row) for row in _read_jsonl(path))
    suite: list[dict[str, Any]] = []
    for row in selected:
        performance = row["physics_outputs"]["performance"]
        target = build_target_mapping(
            thrust_N=performance["thrust_N"],
            tsfc_kg_N_s=performance["tsfc_kg_N_s"],
            mass_kg=performance["mass_kg"],
            target_type="ID",
            expected_status="KNOWN_FEASIBLE",
        )
        if target["target_id"] in excluded_ids:
            raise E1PerformanceRankingError("Fresh target overlaps an excluded target suite")
        suite.append(
            {
                "split": "locked_e1",
                "target_region": row["target_region"],
                "source_campaign": target_spec["campaign_id"],
                "sample_id": int(row["sample_id"]),
                "train_neighbor_distance": float(row["train_neighbor_distance"]),
                "target": target,
                "reference_x_19": row["x_19"],
            }
        )
    suite_path = runtime / config["artifacts"]["target_suite"]
    _write_jsonl(suite_path, suite)

    contract = load_wp19_23_contract(repo / config["inverse_protocol"])
    schema_path = repo / contract.payload["contracts"]["design_schema"]
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    model_config = ModelConfig.from_yaml(repo / contract.payload["contracts"]["calibration_pack"])
    generator_spec = config["generator"]
    generator_path = repo / generator_spec["checkpoint"]
    generator = load_generator(generator_path, device=device or config["execution"]["device"])
    if not isinstance(generator, CFMGenerator):
        raise E1PerformanceRankingError("Frozen checkpoint is not a CFM generator")
    source_seeds = [int(value) for value in (smoke_spec["source_seeds"] if smoke else generator_spec["source_seeds"])]
    count = int(smoke_spec["raw_candidates_per_run"] if smoke else generator_spec["raw_candidates_per_run"])
    candidates: list[dict[str, Any]] = []
    for target_row in suite:
        target = Geometry19InverseTarget.from_mapping(target_row["target"])
        condition = np.asarray([target.thrust_N, target.tsfc_kg_N_s, target.mass_kg], dtype=float)
        normalized_condition = torch.as_tensor(
            generator.normalizer.transform_conditions(condition),
            dtype=generator.dtype,
            device=generator.device,
        )
        for source_seed in source_seeds:
            noise = sample_device_invariant_sources(generator, count, source_seed)
            with torch.no_grad():
                samples = integrate_cfm_sources(generator, noise, normalized_condition).cpu().numpy()
            mapped, _ = _filter_samples(
                samples,
                generator_spec["arm_id"],
                target.target_id,
                source_seed,
                schema,
                model_config,
                discrete_tolerance_count=None,
            )
            for row in mapped:
                row["split"] = "locked_e1"
                row["target_region"] = target_row["target_region"]
            candidates.extend(mapped)
    candidate_path = runtime / config["artifacts"]["candidate_ledger"]
    _write_jsonl(candidate_path, candidates)
    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "CANDIDATES_FROZEN_NO_TRUTH_OR_SCORES",
        "smoke": smoke,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_mutation_allowed": False,
        "target_bank_sample_count": bank_count,
        "target_bank_acceptable_count": len(fresh),
        "target_count": len(suite),
        "target_region_counts": {
            region: sum(row["target_region"] == region for row in suite)
            for region in ("dense", "mid", "sparse")
        },
        "source_seeds": source_seeds,
        "raw_candidates_per_run": count,
        "run_count": len(suite) * len(source_seeds),
        "raw_candidate_count": len(candidates),
        "mapped_candidate_count": sum(row["geometry19"] is not None for row in candidates),
        "failure_counts": {
            code: sum(row["failure_code"] == code for row in candidates)
            for code in sorted({row["failure_code"] for row in candidates if row["failure_code"]})
        },
        "hashes": {
            "config": sha256_file(source),
            "target_bank_dataset": sha256_file(bank_root / "dataset.jsonl"),
            "target_suite": sha256_file(suite_path),
            "candidate_ledger": sha256_file(candidate_path),
            "generator_checkpoint": sha256_file(generator_path),
            "design_schema": sha256_file(schema_path),
            "excluded_target_suites": excluded_hashes,
        },
    }
    _write_json(runtime / config["artifacts"]["freeze_manifest"], manifest)
    return manifest


def evaluate_e1_truth(
    root: str | Path,
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate every frozen mapped candidate with V2.1, without proxy access."""
    repo = Path(root).resolve()
    source = Path(config_path).resolve()
    config = _load_config(source)
    runtime = _runtime(repo, config, output_root)
    truth_path = runtime / config["artifacts"]["truth_ledger"]
    truth_manifest_path = runtime / config["artifacts"]["truth_manifest"]
    if truth_path.exists() or truth_manifest_path.exists():
        raise E1PerformanceRankingError("Refusing to overwrite frozen truth")
    freeze = json.loads((runtime / config["artifacts"]["freeze_manifest"]).read_text(encoding="utf-8"))
    if freeze["status"] != "CANDIDATES_FROZEN_NO_TRUTH_OR_SCORES":
        raise E1PerformanceRankingError("Candidates were not frozen before truth evaluation")
    if freeze["hashes"]["config"] != sha256_file(source):
        raise E1PerformanceRankingError("Config changed after candidate freeze")
    candidate_path = runtime / config["artifacts"]["candidate_ledger"]
    suite_path = runtime / config["artifacts"]["target_suite"]
    if freeze["hashes"]["candidate_ledger"] != sha256_file(candidate_path):
        raise E1PerformanceRankingError("Candidate ledger changed after freeze")
    if freeze["hashes"]["target_suite"] != sha256_file(suite_path):
        raise E1PerformanceRankingError("Target suite changed after freeze")
    candidates = _read_jsonl(candidate_path)
    targets = {_target_id(row): row["target"] for row in _read_jsonl(suite_path)}
    tasks = []
    for row in candidates:
        if row["geometry19"] is None:
            continue
        tasks.append(
            {
                "run_id": row["run_id"],
                "candidate_id": row["candidate_id"],
                "geometry19": row["geometry19"],
                "design_hash": row["design_hash"],
                "target": targets[row["target_id"]],
            }
        )
    counts = {"evaluated": 0, "physics_acceptable": 0, "strict_hits": 0}

    def compact(results: Iterable[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
        for row in results:
            acceptable = bool(row["physics_acceptable"])
            alignment = row.get("hit_alignment") or {}
            rho = alignment.get("rho") if acceptable else None
            counts["evaluated"] += 1
            counts["physics_acceptable"] += int(acceptable)
            counts["strict_hits"] += int(rho is not None and float(rho) <= 1.0)
            yield {
                "run_id": row["run_id"],
                "target_id": row["target_id"],
                "candidate_id": row["candidate_id"],
                "design_hash": row["design_hash"],
                "physics_acceptable": acceptable,
                "rho": rho,
                "root_classification": row["root_classification"],
                "solver_failure_code": row["solver_failure_code"],
            }

    workers = int(config["smoke"]["physics_workers"] if freeze["smoke"] else config["execution"]["physics_workers"])
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=(str(repo),)
    ) as executor:
        results = executor.map(
            _physics_worker,
            tasks,
            chunksize=int(config["execution"]["physics_chunksize"]),
        )
        _write_jsonl(truth_path, compact(results))
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "V21_TRUTH_FROZEN_NO_PROXY_SCORES",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_mutation_allowed": False,
        **counts,
        "run_count": freeze["run_count"],
        "hashes": {
            "config": sha256_file(source),
            "candidate_ledger": sha256_file(candidate_path),
            "truth_ledger": sha256_file(truth_path),
        },
    }
    _write_json(truth_manifest_path, manifest)
    return manifest


def _predicted_rho(
    prediction: np.ndarray, targets: np.ndarray, tolerances: np.ndarray
) -> np.ndarray:
    return np.max(
        np.abs(prediction - targets)
        / np.maximum(np.abs(targets), 1.0e-15)
        / tolerances[None, :],
        axis=1,
    )


def _success_map(metrics: Mapping[str, Any], *, hit_only: bool) -> dict[str, dict[str, bool]]:
    return {
        row["run_id"]: dict(row["success"])
        for row in metrics["runs"]
        if not hit_only or row["has_true_hit"]
    }


def _target_coverage(metrics: Mapping[str, Any], targets: Sequence[str], budget: int) -> float:
    return float(
        np.mean(
            [
                any(
                    row["target_id"] == target and row["success"][str(budget)]
                    for row in metrics["runs"]
                )
                for target in targets
            ]
        )
    )


def _paired(baseline: Mapping[str, Any], challenger: Mapping[str, Any], budget: int) -> dict[str, Any]:
    left = _success_map(baseline, hit_only=True)
    right = _success_map(challenger, hit_only=True)
    gains = sum(not left[key][str(budget)] and right[key][str(budget)] for key in left)
    losses = sum(left[key][str(budget)] and not right[key][str(budget)] for key in left)
    p_value = float(binomtest(min(gains, losses), gains + losses, 0.5).pvalue) if gains + losses else 1.0
    return {"gains": gains, "losses": losses, "two_sided_exact_p": p_value}


def _render(summary: Mapping[str, Any]) -> str:
    lines = [
        "# WP19-23 E1：三输出性能代理真实生成排序确认",
        "",
        f"> 结论：**{summary['decision']['status']}**  ",
        "> 同一冻结 bounded-CFM 候选池，仅比较 S0 与 performance3_log；未启用 reranker、梯度增强或 NSGA-II。",
        "> 无论结果如何均不接入 pipeline。",
        "",
        "## 等预算结果",
        "",
        "| Scorer | Recall@8 | @16 | @32 | @64 | @128 | @256 | 首命中P50 | 首命中P90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scorer in ("shared48_s0", "performance3_log"):
        metrics = summary["evaluation"][scorer]
        recall = metrics["hit_bearing_recall"]
        ranks = np.asarray(
            [row["first_hit_rank"] for row in metrics["runs"] if row["has_true_hit"]],
            dtype=float,
        )
        rank_p50 = "N/A" if not len(ranks) else f"{np.quantile(ranks, 0.5):.1f}"
        rank_p90 = "N/A" if not len(ranks) else f"{np.quantile(ranks, 0.9):.1f}"
        lines.append(
            f"| {scorer} | {recall['8']:.1%} | {recall['16']:.1%} | {recall['32']:.1%} | "
            f"{recall['64']:.1%} | {recall['128']:.1%} | {recall['256']:.1%} | "
            f"{rank_p50} | {rank_p90} |"
        )
    primary = summary["comparison"]["cluster_bootstrap"]["32"]
    paired = summary["comparison"]["paired"]["32"]
    lines += [
        "",
        "## 主比较",
        "",
        f"- hit-bearing runs：{summary['hit_bearing_run_count']}/{summary['run_count']}；",
        f"- performance3_log 相对 S0 的 Recall@32 差值：{primary['point']:+.1%}；",
        f"- 目标聚类95% CI：[{primary['lower']:+.1%}, {primary['upper']:+.1%}]；",
        f"- 新增/损失命中运行：{paired['gains']}/{paired['losses']}，双侧精确 p={paired['two_sided_exact_p']:.4g}；",
        "- pipeline action：**NONE，等待用户确认**。",
        "",
    ]
    return "\n".join(lines)


def score_e1_campaign(
    root: str | Path,
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Score the immutable candidate/truth ledgers and make no pipeline changes."""
    repo = Path(root).resolve()
    source = Path(config_path).resolve()
    config = _load_config(source)
    runtime = _runtime(repo, config, output_root)
    summary_path = runtime / config["artifacts"]["summary"]
    report_path = runtime / config["artifacts"]["report"]
    if summary_path.exists() or report_path.exists():
        raise E1PerformanceRankingError("Refusing to overwrite completed E1 scoring")
    freeze = json.loads((runtime / config["artifacts"]["freeze_manifest"]).read_text(encoding="utf-8"))
    truth_manifest = json.loads((runtime / config["artifacts"]["truth_manifest"]).read_text(encoding="utf-8"))
    candidate_path = runtime / config["artifacts"]["candidate_ledger"]
    truth_path = runtime / config["artifacts"]["truth_ledger"]
    suite_path = runtime / config["artifacts"]["target_suite"]
    if freeze["hashes"]["config"] != sha256_file(source) or truth_manifest["hashes"]["config"] != sha256_file(source):
        raise E1PerformanceRankingError("Config changed after freeze")
    if freeze["hashes"]["candidate_ledger"] != sha256_file(candidate_path):
        raise E1PerformanceRankingError("Candidate ledger changed after freeze")
    if truth_manifest["hashes"]["truth_ledger"] != sha256_file(truth_path):
        raise E1PerformanceRankingError("Truth ledger changed after freeze")

    protected_before = {path: sha256_file(repo / path) for path in config["protected_pipeline_files"]}
    suite = _read_jsonl(suite_path)
    target_rows = {_target_id(row): row for row in suite}
    candidates = [row for row in _read_jsonl(candidate_path) if row["geometry19"] is not None]
    truth = {
        (row["run_id"], int(row["candidate_id"])): row for row in _read_jsonl(truth_path)
    }
    rows: list[dict[str, Any]] = []
    for row in candidates:
        result = truth[(row["run_id"], int(row["candidate_id"]))]
        rows.append(
            {
                **row,
                "physics_acceptable": bool(result["physics_acceptable"]),
                "rho": result["rho"] if result["physics_acceptable"] else None,
            }
        )
    feature_names = list(Geometry19.coordinate_names())
    x = np.asarray(
        [[row["geometry19"][name] for name in feature_names] for row in rows], dtype=np.float32
    )
    targets = np.asarray(
        [
            [target_rows[row["target_id"]]["target"][name] for name in PERFORMANCE_NAMES]
            for row in rows
        ],
        dtype=np.float64,
    )
    inverse = yaml.safe_load((repo / config["inverse_protocol"]).read_text(encoding="utf-8"))
    tolerances = np.asarray(
        [inverse["diagnostic_acceptance"]["relative_tolerances"][name] for name in PERFORMANCE_NAMES],
        dtype=np.float64,
    )
    torch_device = torch.device(device or config["execution"]["device"])
    baseline_spec = config["baseline"]
    challenger_spec = config["challenger"]
    baseline_prediction, _ = _ensemble_prediction(
        repo / baseline_spec["model_root"],
        [int(value) for value in baseline_spec["seeds"]],
        feature_names,
        x,
        torch_device,
        shared48=True,
    )
    challenger_prediction, _ = _ensemble_prediction(
        repo / challenger_spec["model_root"],
        [int(value) for value in challenger_spec["seeds"]],
        feature_names,
        x,
        torch_device,
        shared48=False,
    )
    baseline_rho = _predicted_rho(baseline_prediction, targets, tolerances)
    challenger_rho = _predicted_rho(challenger_prediction, targets, tolerances)
    budgets = [int(value) for value in config["ranking_budgets"]]
    evaluation = {
        "shared48_s0": ranking_metrics(rows, -baseline_rho, budgets),
        "performance3_log": ranking_metrics(rows, -challenger_rho, budgets),
    }
    hit_runs = {
        row["run_id"]
        for row in evaluation["shared48_s0"]["runs"]
        if row["has_true_hit"]
    }
    run_target = {
        row["run_id"]: row["target_id"]
        for row in evaluation["shared48_s0"]["runs"]
        if row["run_id"] in hit_runs
    }
    left = _success_map(evaluation["shared48_s0"], hit_only=True)
    right = _success_map(evaluation["performance3_log"], hit_only=True)
    statistics = config["statistics"]
    empty_interval = {"point": 0.0, "lower": 0.0, "upper": 0.0}
    comparison = {
        "cluster_bootstrap": {
            str(budget): (
                cluster_bootstrap_difference(
                    left,
                    right,
                    run_target,
                    budget,
                    replicates=int(statistics["cluster_bootstrap_replicates"]),
                    seed=int(statistics["bootstrap_seed"]),
                    confidence_level=float(statistics["confidence_level"]),
                )
                if hit_runs
                else dict(empty_interval)
            )
            for budget in budgets
        },
        "paired": {
            str(budget): _paired(
                evaluation["shared48_s0"], evaluation["performance3_log"], budget
            )
            for budget in budgets
        },
    }
    all_targets = sorted(target_rows)
    target_coverage = {
        scorer: {
            str(budget): _target_coverage(metrics, all_targets, budget) for budget in budgets
        }
        for scorer, metrics in evaluation.items()
    }
    region_evaluation: dict[str, Any] = {}
    for region in ("dense", "mid", "sparse"):
        region_rows = [row for row in rows if row["target_region"] == region]
        indices = np.asarray(
            [index for index, row in enumerate(rows) if row["target_region"] == region], dtype=np.int64
        )
        region_evaluation[region] = {
            "shared48_s0": ranking_metrics(region_rows, -baseline_rho[indices], budgets),
            "performance3_log": ranking_metrics(region_rows, -challenger_rho[indices], budgets),
        }
    primary = comparison["cluster_bootstrap"]
    decision = config["decision"]
    sufficient = len(hit_runs) >= int(statistics["minimum_hit_bearing_runs"])
    region_gains = {
        region: (
            metrics["performance3_log"]["hit_bearing_recall"]["32"]
            - metrics["shared48_s0"]["hit_bearing_recall"]["32"]
        )
        for region, metrics in region_evaluation.items()
    }
    passed = bool(
        sufficient
        and primary["32"]["point"] >= float(decision["minimum_recall_at_32_absolute_gain"])
        and (
            not decision["require_cluster_ci_lower_bound_nonnegative"]
            or primary["32"]["lower"] >= 0.0
        )
        and (
            not decision["require_recall_at_64_non_degradation"]
            or primary["64"]["point"] >= 0.0
        )
        and (
            not decision["require_target_coverage_at_32_non_degradation"]
            or target_coverage["performance3_log"]["32"]
            >= target_coverage["shared48_s0"]["32"]
        )
        and min(region_gains.values())
        >= -float(decision["maximum_region_recall_at_32_regression"])
    )
    status = (
        "INCONCLUSIVE_NO_INTEGRATION"
        if not sufficient
        else "EVALUATION_PASS_NO_INTEGRATION"
        if passed
        else "EVALUATION_FAIL_NO_INTEGRATION"
    )
    protected_after = {path: sha256_file(repo / path) for path in config["protected_pipeline_files"]}
    if protected_after != protected_before:
        raise E1PerformanceRankingError("A protected pipeline file changed during E1 scoring")
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "COMPLETE_EVALUATION_ONLY",
        "formal_use_allowed": False,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": freeze["run_count"],
        "raw_candidate_count": freeze["raw_candidate_count"],
        "mapped_candidate_count": freeze["mapped_candidate_count"],
        "physics_acceptable_count": truth_manifest["physics_acceptable"],
        "hit_bearing_run_count": len(hit_runs),
        "evaluation": evaluation,
        "target_coverage": target_coverage,
        "region_evaluation": region_evaluation,
        "region_recall_at_32_gain": region_gains,
        "comparison": comparison,
        "decision": {
            "status": status,
            "pre_registered_gate_passed": passed,
            "sufficient_hit_bearing_runs": sufficient,
            "pipeline_action": "NONE",
            "pipeline_mutation_allowed": False,
            "awaiting_explicit_user_confirmation": True,
        },
        "hashes": {
            "config": sha256_file(source),
            "candidate_ledger": sha256_file(candidate_path),
            "truth_ledger": sha256_file(truth_path),
            "baseline_weights": {
                str(seed): sha256_file(repo / baseline_spec["model_root"] / f"seed_{seed}" / "weights_best_val.pt")
                for seed in baseline_spec["seeds"]
            },
            "challenger_weights": {
                str(seed): sha256_file(repo / challenger_spec["model_root"] / f"seed_{seed}" / "weights_best_val.pt")
                for seed in challenger_spec["seeds"]
            },
            "code_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
            ).stdout.strip(),
        },
        "protected_pipeline_hashes_before": protected_before,
        "protected_pipeline_hashes_after": protected_after,
    }
    _write_json(summary_path, summary)
    report_path.write_text(_render(summary), encoding="utf-8")
    return summary
