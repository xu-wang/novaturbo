"""Gate I: equal-budget ablation of hit-aligned Geometry19 search."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping

import yaml

from src.ai.geometry19_candidates import run_candidate_protocol
from src.ai.geometry19_physics import run_true_physics_stage
from src.ai.geometry19_pipeline import build_target_mapping, resolve_generator_checkpoint
from src.ai.geometry19_refinement import run_refinement_stage
from src.ai.geometry19_surrogate import sha256_file


GATE_I_VERSION = "wp19_23_gate_i_hit_alignment_ablation_v1"


class Geometry19GateIError(ValueError):
    """Raised when the Gate I campaign contract is invalid."""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def _best_rho(summary: Mapping[str, Any]) -> float | None:
    best = summary.get("best_approximation")
    if not isinstance(best, Mapping):
        return None
    alignment = best.get("hit_alignment")
    return float(alignment["rho"]) if isinstance(alignment, Mapping) else None


def _power_of_two_floor(value: int) -> int:
    return 0 if value <= 0 else 1 << (value.bit_length() - 1)


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Gate I：命中对齐目标等预算消融",
        "",
        "四个实验臂使用同一个 bounded-sigmoid 生成器、相同目标和相同 C 阶段 V2.1 预算。",
        "",
        "| 实验臂 | 排序 | D-Flow/梯度损失 | 真实接受 | C 命中 | D 救援 | 最终命中 | C 调用 | D 调用 | 中位最终 rho |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for arm_id, arm in summary["arms"].items():
        config = arm["configuration"]
        rho = arm["median_final_true_rho"]
        rho_text = f"{rho:.3f}" if rho is not None else "n/a"
        lines.append(
            f"| {arm_id} | {config['candidate_ranking_mode']} | "
            f"{config['dflow_target_loss_mode']}/{config['refinement_target_loss_mode']} | "
            f"{config['true_acceptance_mode']} | {arm['c_target_hits']}/{summary['target_count']} | "
            f"{arm['d_rescue_hits']}/{summary['target_count']} | "
            f"{arm['final_target_hits']}/{summary['target_count']} | "
            f"{arm['c_v2_1_calls']} | {arm['d_v2_1_calls']} | "
            f"{rho_text} |"
        )
    lines += [
        "",
        f"C 阶段共同预算：每目标 {summary['common_c_budget_per_target']} 次真实 V2.1。",
        "",
        "严格命中等价于 true rho <= 1；D 阶段仅在 C 未命中时运行，并按命中提前停止。",
        "",
        "## 完整命中对齐臂逐目标结果",
        "",
        "| 目标 | C 命中 | D 救援 | 最终命中 | 最佳 true rho |",
        "| --- | --- | --- | --- | ---: |",
    ]
    full = summary["arms"].get("i3_full_hit_alignment")
    if full:
        for sample_id, row in full["targets"].items():
            rho = row["final_best_true_rho"]
            lines.append(
                f"| {sample_id} | {'是' if row['c_hit'] else '否'} | "
                f"{'是' if row['d_hit'] else '否'} | {'是' if row['final_hit'] else '否'} | "
                f"{rho:.3f} |"
            )
    lines += [
        "",
        "单次固定种子实验用于验证改动方向，不替代多随机种子置信区间。",
        "",
    ]
    return "\n".join(lines)


def run_gate_i_campaign(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    device: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    if config.get("protocol_version") != GATE_I_VERSION:
        raise Geometry19GateIError("Unsupported Gate I protocol version")
    arms = {str(item["arm_id"]): dict(item) for item in config["arms"]}
    runtime = Path(output_root).resolve() if output_root else root / config["artifacts"]["runtime_root"]
    if runtime.exists():
        raise Geometry19GateIError(f"Refusing to overwrite Gate I run: {runtime}")
    runtime.mkdir(parents=True)
    run_device = str(device or config.get("device", "cpu"))
    target_entries: list[tuple[str, Path]] = []
    for item in config["targets"]:
        target = build_target_mapping(
            thrust_N=item["thrust_N"], tsfc_kg_N_s=item["tsfc_kg_N_s"],
            mass_kg=item["mass_kg"], target_type=item.get("target_type", "ID"),
            expected_status=item.get("expected_status", "KNOWN_FEASIBLE"),
        )
        target_path = runtime / "targets" / f"sample_{item['sample_id']}.yaml"
        _write_yaml(target_path, target)
        target_entries.append((str(item["sample_id"]), target_path))

    records: dict[str, dict[str, Any]] = {arm_id: {} for arm_id in arms}
    # Generate all arms for one target first so the C budget is the common available N_keep.
    common_budgets: list[int] = []
    for sample_id, target_path in target_entries:
        b_summaries: dict[str, Mapping[str, Any]] = {}
        for arm_id, arm in arms.items():
            arm_root = runtime / "runs" / f"sample_{sample_id}" / arm_id
            b_root = arm_root / "candidates" / "dflow"
            b_summaries[arm_id] = run_candidate_protocol(
                root, target_path, mode="dflow", device=run_device, output_root=b_root,
                generator_checkpoint=resolve_generator_checkpoint(root, "bounded-sigmoid"),
                dflow_target_loss_mode=arm["dflow_target_loss_mode"],
                candidate_ranking_mode=arm["candidate_ranking_mode"],
            )
        available = min(int(value["candidate_protocol"]["n_keep"]) for value in b_summaries.values())
        cap = int(config["budgets"]["maximum_true_physics_per_target_per_arm"])
        common_budget = _power_of_two_floor(min(available, cap))
        if common_budget <= 0:
            raise Geometry19GateIError(f"No common candidate budget for sample {sample_id}")
        common_budgets.append(common_budget)
        for arm_id, arm in arms.items():
            arm_root = runtime / "runs" / f"sample_{sample_id}" / arm_id
            c_summary = run_true_physics_stage(
                root, target_path, candidate_root=arm_root / "candidates",
                output_root=arm_root / "wp19_23c", sampling_mode="dflow",
                evaluation_budget=common_budget,
            )
            d_summary = None
            if not c_summary["downstream_stop"]:
                d_summary = run_refinement_stage(
                    root, target_path, physics_root=arm_root / "wp19_23c",
                    output_root=arm_root / "wp19_23d", device=run_device,
                    target_loss_mode=arm["refinement_target_loss_mode"],
                    true_acceptance_mode=arm["true_acceptance_mode"],
                )
            final_summary = d_summary or c_summary
            c_best_rho = _best_rho(c_summary)
            d_best_rho = _best_rho(d_summary) if d_summary else None
            available_rhos = [value for value in (c_best_rho, d_best_rho) if value is not None]
            records[arm_id][sample_id] = {
                "n_keep": int(b_summaries[arm_id]["candidate_protocol"]["n_keep"]),
                "c_budget": common_budget,
                "c_hit": bool(c_summary["downstream_stop"]),
                "c_best_true_rho": c_best_rho,
                "c_v2_1_calls": int(c_summary["counts"]["unique_v2_1_evaluations"]),
                "d_ran": d_summary is not None,
                "d_hit": bool(d_summary and d_summary["downstream_stop"]),
                "d_best_true_rho": d_best_rho,
                "d_v2_1_calls": int(d_summary["counts"]["true_v2_1_evaluations"]) if d_summary else 0,
                "accepted_updates": int(d_summary["counts"]["strictly_accepted_updates"]) if d_summary else 0,
                "final_hit": bool(final_summary["downstream_stop"]),
                "final_best_true_rho": min(available_rhos) if available_rhos else None,
            }

    arm_results: dict[str, Any] = {}
    for arm_id, arm in arms.items():
        rows = records[arm_id]
        rho_values = [row["final_best_true_rho"] for row in rows.values() if row["final_best_true_rho"] is not None]
        arm_results[arm_id] = {
            "configuration": arm,
            "targets": rows,
            "c_target_hits": sum(row["c_hit"] for row in rows.values()),
            "d_rescue_hits": sum(row["d_hit"] for row in rows.values()),
            "final_target_hits": sum(row["final_hit"] for row in rows.values()),
            "c_v2_1_calls": sum(row["c_v2_1_calls"] for row in rows.values()),
            "d_v2_1_calls": sum(row["d_v2_1_calls"] for row in rows.values()),
            "accepted_updates": sum(row["accepted_updates"] for row in rows.values()),
            "median_final_true_rho": median(rho_values) if rho_values else None,
        }
    result = {
        "protocol_version": GATE_I_VERSION,
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_use_allowed": False,
        "device": run_device,
        "target_count": len(target_entries),
        "common_c_budget_per_target": min(common_budgets) if len(set(common_budgets)) == 1 else common_budgets,
        "arms": arm_results,
        "hashes": {"config_sha256": sha256_file(config_source)},
    }
    _write_json(runtime / config["artifacts"]["summary"], result)
    (runtime / config["artifacts"]["report"]).write_text(_report(result), encoding="utf-8")
    _write_json(runtime / config["artifacts"]["manifest"], {
        "protocol_version": GATE_I_VERSION,
        "summary_sha256": sha256_file(runtime / config["artifacts"]["summary"]),
        "report_sha256": sha256_file(runtime / config["artifacts"]["report"]),
    })
    return result
