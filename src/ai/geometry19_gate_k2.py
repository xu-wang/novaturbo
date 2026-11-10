"""Gate K2: four-loss full-closure Geometry19 surrogate comparison."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from src.ai.geometry19_candidates import run_candidate_protocol
from src.ai.geometry19_gate_k import (
    _best_stage_alignment,
    _frozen_strategy_metrics,
    _predict_pool_rho,
    _read_jsonl,
    _refinement_diagnostics,
    _write_json,
)
from src.ai.geometry19_hit_alignment import OUTPUT_NAMES
from src.ai.geometry19_physics import run_true_physics_stage
from src.ai.geometry19_pipeline import build_target_mapping, resolve_generator_checkpoint
from src.ai.geometry19_refinement import run_refinement_stage
from src.ai.geometry19_surrogate import load_weight_bundle, sha256_file


GATE_K2_VERSION = "wp19_22_gate_k2_full_closure_v1"


class Geometry19GateK2Error(ValueError):
    """Raised when the Gate K2 comparison contract is invalid."""


def select_representative_runs(
    training_summary: Mapping[str, Any], arm_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Select one seed per loss using validation rho only."""

    selected = {}
    for arm_id in arm_ids:
        arm = training_summary["arms"][arm_id]
        run = min(
            arm["runs"],
            key=lambda item: (
                float(item["validation"]["hit_alignment"]["rho_p50"]),
                int(item["seed"]),
            ),
        )
        selected[arm_id] = {
            "arm_id": arm_id,
            "loss": dict(arm["loss"]),
            "seed": int(run["seed"]),
            "model_tag": str(run["model_tag"]),
            "model_directory": str(Path(run["artifacts"]["manifest"]).parent),
            "validation_rho_p50": float(
                run["validation"]["hit_alignment"]["rho_p50"]
            ),
            "test_rho_p50": float(run["test"]["hit_alignment"]["rho_p50"]),
        }
    return selected


def _power_of_two_floor(value: int) -> int:
    return 0 if value <= 0 else 1 << (int(value).bit_length() - 1)


def _closure_sort_key(arm: Mapping[str, Any]) -> tuple[Any, ...]:
    refinement_rate = arm["refinement"].get("rho_improved_rate")
    return (
        -int(arm["final_target_hits"]),
        -int(arm["c_target_hits"]),
        float(arm["median_best_true_rho"]),
        -int(arm["retained_production_hit_count"]),
        -float(refinement_rate if refinement_rate is not None else -math.inf),
        int(arm["total_v2_1_calls"]),
        -int(arm["frozen_top32_target_hits"]),
        str(arm["arm_id"]),
    )


def _load_representative_models(
    representatives: Mapping[str, Mapping[str, Any]],
    feature_names: Sequence[str],
    target_names: Sequence[str],
    device: str,
) -> dict[str, dict[str, Any]]:
    output = {}
    for arm_id, representative in representatives.items():
        model, normalization, manifest = load_weight_bundle(
            representative["model_directory"],
            expected_model_tag=representative["model_tag"],
            expected_feature_names=feature_names,
            expected_target_names=target_names,
            device=device,
        )
        output[arm_id] = {
            "model": model,
            "normalization": normalization,
            "manifest": manifest,
        }
    return output


def _run_frozen_pool(
    root: Path,
    config: Mapping[str, Any],
    representatives: Mapping[str, Mapping[str, Any]],
    base_protocol: Mapping[str, Any],
    inverse_protocol: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    schema = yaml.safe_load((root / base_protocol["design_schema"]).read_text())
    feature_names = [item["name"] for item in schema["design_variables"]]
    first_manifest = json.loads(
        (Path(next(iter(representatives.values()))["model_directory"]) / "model_manifest.json").read_text()
    )
    target_names = list(first_manifest["target_names"])
    models = _load_representative_models(
        representatives, feature_names, target_names, device
    )
    tolerances = dict(inverse_protocol["diagnostic_acceptance"]["relative_tolerances"])
    epsilons = dict(inverse_protocol["diagnostic_acceptance"]["epsilons"])
    candidate_root = root / config["frozen_pool"]["candidates"]
    truth_root = root / config["frozen_pool"]["truth"]
    budgets = [int(value) for value in config["budgets"]["frozen_pool"]]
    targets: dict[str, Any] = {}
    for target_dir in sorted(candidate_root.glob("target_*")):
        target = yaml.safe_load((target_dir / "target.yaml").read_text())
        rows = _read_jsonl(target_dir / "projected_union.jsonl")
        evaluations = {
            str(row["candidate_key"]): row
            for row in _read_jsonl(
                truth_root / target_dir.name / "full_pool_evaluations.jsonl"
            )
        }
        if set(evaluations) != {str(row["candidate_key"]) for row in rows}:
            raise Geometry19GateK2Error("Frozen candidate/truth identity mismatch")
        targets[target_dir.name] = {"strategies": {}}
        for arm_id, bundle in models.items():
            predicted_rho = _predict_pool_rho(
                rows,
                target,
                feature_names,
                bundle["model"],
                bundle["normalization"],
                tolerances,
                epsilons,
            )
            targets[target_dir.name]["strategies"][arm_id] = _frozen_strategy_metrics(
                rows, evaluations, predicted_rho, tolerances, budgets
            )
    aggregate = {}
    for arm_id in representatives:
        aggregate[arm_id] = {
            "target_count": len(targets),
            "new_v2_1_calls": 0,
            "first_hit_ranks": [
                item["strategies"][arm_id]["first_hit_rank"]
                for item in targets.values()
            ],
            "rho_spearman_median": float(
                np.median(
                    [item["strategies"][arm_id]["rho_spearman"] for item in targets.values()]
                )
            ),
            "budgets": {
                str(budget): {
                    "target_hits": sum(
                        item["strategies"][arm_id]["budgets"][str(budget)]["target_hit"]
                        for item in targets.values()
                    ),
                    "median_best_true_rho": float(
                        np.median(
                            [
                                item["strategies"][arm_id]["budgets"][str(budget)][
                                    "best_true_rho"
                                ]
                                for item in targets.values()
                            ]
                        )
                    ),
                }
                for budget in budgets
            },
        }
    return {"aggregate": aggregate, "targets": targets}


def _write_target(path: Path, item: Mapping[str, Any]) -> None:
    target = build_target_mapping(
        thrust_N=item["thrust_N"],
        tsfc_kg_N_s=item["tsfc_kg_N_s"],
        mass_kg=item["mass_kg"],
        target_type=item.get("target_type", "ID"),
        expected_status=item.get("expected_status", "KNOWN_FEASIBLE"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(target, sort_keys=False), encoding="utf-8")


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Gate K2：四种代理损失完整闭环比较",
        "",
        "> 代表seed只按validation rho选择；生产代理未修改。",
        "",
        "| Arm | Loss | Seed | Frozen@32 | C命中 | D救援 | 最终命中 | median rho | V2.1 | D rho改善率 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm_id, arm in summary["arms"].items():
        rate = arm["refinement"]["rho_improved_rate"]
        rate_text = f"{rate:.1%}" if rate is not None else "n/a"
        lines.append(
            f"| {arm_id} | {arm['loss']['mode']} | {arm['representative_seed']} | "
            f"{arm['frozen_top32_target_hits']}/4 | {arm['c_target_hits']}/4 | "
            f"{arm['d_rescue_hits']}/4 | {arm['final_target_hits']}/4 | "
            f"{arm['median_best_true_rho']:.3f} | {arm['total_v2_1_calls']} | "
            f"{rate_text} |"
        )
    lines += [
        "",
        f"闭环字典序胜出：`{summary['selection']['winning_arm']}`。",
        f"阶段判定：`{summary['selection']['phase_a_decision']}`。",
        "",
    ]
    return "\n".join(lines)


def run_gate_k2_phase_a(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    device: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    if config.get("protocol_version") != GATE_K2_VERSION:
        raise Geometry19GateK2Error("Unsupported Gate K2 protocol")
    runtime = Path(output_root).resolve() if output_root else root / config["artifacts"]["runtime_root"]
    if runtime.exists():
        raise Geometry19GateK2Error(f"Refusing to overwrite Gate K2 run: {runtime}")
    runtime.mkdir(parents=True)
    run_device = str(device or config["device"])
    training_path = root / config["gate_k_training_summary"]
    training = json.loads(training_path.read_text())
    representatives = select_representative_runs(training, config["arms"])
    gate_k_config = yaml.safe_load((root / config["gate_k_config"]).read_text())
    base_protocol = yaml.safe_load((root / gate_k_config["base_surrogate_protocol"]).read_text())
    inverse_protocol = yaml.safe_load((root / gate_k_config["inverse_protocol"]).read_text())
    frozen = _run_frozen_pool(
        root, config, representatives, base_protocol, inverse_protocol, run_device
    )
    _write_json(runtime / "representative_models.json", representatives)
    _write_json(runtime / "frozen_pool_summary.json", frozen)

    gate_i_config = yaml.safe_load((root / config["gate_i_config"]).read_text())
    production_summary = json.loads((root / config["gate_i_baseline_summary"]).read_text())[
        "arms"
    ][config["gate_i_baseline_arm"]]
    production_hits = {
        sample_id
        for sample_id, item in production_summary["targets"].items()
        if item["final_hit"]
    }
    closure_targets: dict[str, dict[str, Any]] = {
        arm_id: {} for arm_id in representatives
    }
    for target_item in gate_i_config["targets"]:
        sample_id = str(target_item["sample_id"])
        target_path = runtime / "targets" / f"sample_{sample_id}.yaml"
        _write_target(target_path, target_item)
        b_summaries = {}
        for arm_id, representative in representatives.items():
            arm_root = runtime / "runs" / f"sample_{sample_id}" / arm_id
            b_summaries[arm_id] = run_candidate_protocol(
                root,
                target_path,
                mode="dflow",
                device=run_device,
                output_root=arm_root / "candidates/dflow",
                generator_checkpoint=resolve_generator_checkpoint(root, "bounded-sigmoid"),
                dflow_target_loss_mode="hit_aligned",
                candidate_ranking_mode="predicted_rho",
                forward_weights_override=representative["model_directory"],
            )
            observed_raw = int(
                b_summaries[arm_id]["candidate_protocol"]["n_raw"]
            )
            expected_raw = int(config["budgets"]["raw_candidates_per_target_per_arm"])
            if observed_raw != expected_raw:
                raise Geometry19GateK2Error(
                    f"Raw-candidate budget mismatch for {sample_id}/{arm_id}: "
                    f"{observed_raw} != {expected_raw}"
                )
        common_available = min(
            int(item["candidate_protocol"]["n_keep"])
            for item in b_summaries.values()
        )
        common_budget = _power_of_two_floor(
            min(
                common_available,
                int(config["budgets"]["maximum_c_v2_1_per_target_per_arm"]),
            )
        )
        if common_budget <= 0:
            raise Geometry19GateK2Error(f"No common C budget for target {sample_id}")
        for arm_id, representative in representatives.items():
            arm_root = runtime / "runs" / f"sample_{sample_id}" / arm_id
            c_summary = run_true_physics_stage(
                root,
                target_path,
                candidate_root=arm_root / "candidates",
                output_root=arm_root / "wp19_23c",
                sampling_mode="dflow",
                evaluation_budget=common_budget,
            )
            d_summary = None
            if not c_summary["downstream_stop"]:
                d_summary = run_refinement_stage(
                    root,
                    target_path,
                    physics_root=arm_root / "wp19_23c",
                    output_root=arm_root / "wp19_23d",
                    device=run_device,
                    target_loss_mode="hit_aligned",
                    true_acceptance_mode="hit_aligned",
                    forward_weights_override=representative["model_directory"],
                )
                d_calls = int(d_summary["counts"]["true_v2_1_evaluations"])
                maximum_d_calls = int(
                    config["budgets"]["maximum_d_v2_1_per_target_per_arm"]
                )
                if d_calls > maximum_d_calls:
                    raise Geometry19GateK2Error(
                        f"D-stage budget exceeded for {sample_id}/{arm_id}: "
                        f"{d_calls} > {maximum_d_calls}"
                    )
            alignment = _best_stage_alignment(c_summary, d_summary)
            c_hit = bool(c_summary["downstream_stop"])
            d_hit = bool(d_summary and d_summary["downstream_stop"])
            closure_targets[arm_id][sample_id] = {
                "common_c_budget": common_budget,
                "n_keep": int(b_summaries[arm_id]["candidate_protocol"]["n_keep"]),
                "c_hit": c_hit,
                "d_rescue": d_hit,
                "final_hit": c_hit or d_hit,
                "c_v2_1_calls": int(c_summary["counts"]["unique_v2_1_evaluations"]),
                "d_v2_1_calls": (
                    int(d_summary["counts"]["true_v2_1_evaluations"])
                    if d_summary else 0
                ),
                "best_hit_alignment": dict(alignment) if alignment else None,
            }
    arm_results = {}
    for arm_id, representative in representatives.items():
        targets = closure_targets[arm_id]
        final_hit_targets = {
            sample_id for sample_id, item in targets.items() if item["final_hit"]
        }
        rho_values = [
            float(item["best_hit_alignment"]["rho"])
            for item in targets.values() if item["best_hit_alignment"]
        ]
        refinement_paths = sorted(
            runtime.glob(f"runs/sample_*/{arm_id}/wp19_23d/refinement_ledger.jsonl")
        )
        refinement = _refinement_diagnostics(refinement_paths)
        c_calls = sum(int(item["c_v2_1_calls"]) for item in targets.values())
        d_calls = sum(int(item["d_v2_1_calls"]) for item in targets.values())
        arm_results[arm_id] = {
            "arm_id": arm_id,
            "loss": representative["loss"],
            "representative_seed": representative["seed"],
            "model_tag": representative["model_tag"],
            "validation_rho_p50": representative["validation_rho_p50"],
            "test_rho_p50": representative["test_rho_p50"],
            "frozen": frozen["aggregate"][arm_id],
            "frozen_top32_target_hits": frozen["aggregate"][arm_id]["budgets"]["32"]["target_hits"],
            "targets": targets,
            "c_target_hits": sum(item["c_hit"] for item in targets.values()),
            "d_rescue_hits": sum(item["d_rescue"] for item in targets.values()),
            "final_target_hits": len(final_hit_targets),
            "final_hit_targets": sorted(final_hit_targets),
            "retained_production_hit_count": len(final_hit_targets & production_hits),
            "lost_production_hit_targets": sorted(production_hits - final_hit_targets),
            "gained_target_hits": sorted(final_hit_targets - production_hits),
            "median_best_true_rho": float(np.median(rho_values)),
            "c_v2_1_calls": c_calls,
            "d_v2_1_calls": d_calls,
            "total_v2_1_calls": c_calls + d_calls,
            "refinement": refinement,
        }
    winning_arm = min(arm_results, key=lambda name: _closure_sort_key(arm_results[name]))
    winner = arm_results[winning_arm]
    production_rho = float(production_summary["median_final_true_rho"])
    rho_reduction = (production_rho - winner["median_best_true_rho"]) / production_rho
    production_refinement = _refinement_diagnostics(
        sorted(
            (root / config["gate_i_baseline_summary"]).parent.glob(
                f"runs/sample_*/{config['gate_i_baseline_arm']}/wp19_23d/refinement_ledger.jsonl"
            )
        )
    )
    winner_refinement_rate = winner["refinement"]["rho_improved_rate"]
    production_refinement_rate = production_refinement["rho_improved_rate"]
    refinement_rate_guard = (
        winner_refinement_rate is not None
        and production_refinement_rate is not None
        and winner_refinement_rate
        >= production_refinement_rate
        - float(config["promotion"]["maximum_refinement_improvement_rate_drop"])
    )
    checks = {
        "minimum_target_hits": winner["final_target_hits"] >= int(
            config["promotion"]["minimum_target_hits"]
        ),
        "downstream_value": (
            winner["final_target_hits"]
            >= int(config["promotion"]["direct_promotion_target_hits"])
            or (
                winner["final_target_hits"] == int(production_summary["final_target_hits"])
                and rho_reduction
                >= float(config["promotion"]["minimum_same_hit_median_rho_reduction"])
            )
        ),
        "refinement_rate_guard": refinement_rate_guard,
        "frozen_top32_no_regression": (
            winner["frozen_top32_target_hits"]
            >= arm_results["s0_legacy_mse"]["frozen_top32_target_hits"]
        ),
        "v2_1_cost_guard": (
            winner["total_v2_1_calls"]
            <= int(production_summary["c_v2_1_calls"])
            + int(production_summary["d_v2_1_calls"])
            + int(config["promotion"]["maximum_extra_v2_1_calls"])
        ),
    }
    new_loss_wins = winning_arm != "s0_legacy_mse"
    preliminary = new_loss_wins and all(checks.values())
    if preliminary and config["promotion"]["require_multiseed_for_production"]:
        phase_decision = "RUN_CONDITIONAL_MULTI_SEED_CLOSURE"
    elif preliminary:
        phase_decision = "ELIGIBLE_FOR_PRODUCTION_PROMOTION"
    elif not new_loss_wins:
        phase_decision = "STOP_LOSS_TUNING_AND_MOVE_TO_GATE_L"
    else:
        phase_decision = "KEEP_BASELINE_NO_MULTI_SEED"
    summary = {
        "protocol_version": GATE_K2_VERSION,
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_use_allowed": False,
        "production_pipeline_modified": False,
        "device": run_device,
        "representatives": representatives,
        "production_reference": {
            "final_target_hits": int(production_summary["final_target_hits"]),
            "median_best_true_rho": production_rho,
            "total_v2_1_calls": int(production_summary["c_v2_1_calls"])
            + int(production_summary["d_v2_1_calls"]),
            "hit_targets": sorted(production_hits),
            "refinement": production_refinement,
        },
        "arms": arm_results,
        "selection": {
            "method": "lexicographic_hits_c_hits_rho_retention_refinement_cost_frozen32",
            "winning_arm": winning_arm,
            "new_loss_wins": new_loss_wins,
            "median_rho_reduction_vs_production": rho_reduction,
            "promotion_checks": checks,
            "preliminary_promotion_eligible": preliminary,
            "phase_a_decision": phase_decision,
        },
        "new_v2_1_calls": sum(item["total_v2_1_calls"] for item in arm_results.values()),
        "frozen_pool_new_v2_1_calls": 0,
        "hashes": {
            "config_sha256": sha256_file(config_source),
            "training_summary_sha256": sha256_file(training_path),
            "gate_i_baseline_sha256": sha256_file(root / config["gate_i_baseline_summary"]),
        },
    }
    _write_json(runtime / config["artifacts"]["summary"], summary)
    report_path = runtime / config["artifacts"]["report"]
    report_path.write_text(_render_report(summary), encoding="utf-8")
    _write_json(
        runtime / config["artifacts"]["manifest"],
        {
            "protocol_version": GATE_K2_VERSION,
            "summary_sha256": sha256_file(runtime / config["artifacts"]["summary"]),
            "report_sha256": sha256_file(report_path),
        },
    )
    return summary
