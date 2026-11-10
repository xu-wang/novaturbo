"""Frozen multi-target direct/D-Flow equal-budget development campaign."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from src.ai.geometry19_candidates import run_candidate_protocol
from src.ai.geometry19_inverse_contract import load_wp19_23_contract
from src.ai.geometry19_physics import run_true_physics_stage
from src.ai.geometry19_pipeline import build_target_mapping, resolve_generator_checkpoint
from src.ai.geometry19_surrogate import sha256_file


CAMPAIGN_VERSION = "wp19_23_direct_dflow_equal_budget_campaign_v1"
MODES = ("direct", "dflow")


class Geometry19CampaignError(ValueError):
    """Raised when an equal-budget campaign violates its frozen contract."""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def common_physics_budget(n_keep_by_mode: Mapping[str, int]) -> int:
    """Return the common retained power-of-two budget, or zero."""

    if set(n_keep_by_mode) != set(MODES):
        raise Geometry19CampaignError("Campaign requires exactly direct and dflow")
    values = [int(n_keep_by_mode[mode]) for mode in MODES]
    if any(value < 0 or (value and value & (value - 1)) for value in values):
        raise Geometry19CampaignError("Every N_keep must be zero or a power of two")
    return min(values)


def _percentile(values: Sequence[float | None], percentile: float) -> float | None:
    finite = np.asarray(
        [value for value in values if value is not None and math.isfinite(value)],
        dtype=float,
    )
    return float(np.percentile(finite, percentile)) if finite.size else None


def _mode_metrics(runtime: Path, mode: str, elapsed_s: float) -> dict[str, Any]:
    b_root = runtime / "wp19_23b" / mode
    c_root = runtime / "wp19_23c" / mode
    b_summary = json.loads((b_root / "summary.json").read_text(encoding="utf-8"))
    c_summary = json.loads((c_root / "summary.json").read_text(encoding="utf-8"))
    raw = _read_jsonl(b_root / "raw_candidates.jsonl")
    physics = _read_jsonl(c_root / "physics_forward_ledger.jsonl")
    errors = {
        name: [
            float(row["relative_errors"][name])
            for row in physics
            if row.get("relative_errors")
        ]
        for name in ("thrust_N", "tsfc_kg_N_s", "mass_kg")
    }
    provenance_fields = (
        "target_id",
        "candidate_id",
        "source_id",
        "source_seed",
        "model_tag",
        "checkpoint_hash",
        "model_hashes",
        "source_initial",
        "sampling_mode",
    )
    dflow_fields = (
        "surrogate_before",
        "surrogate_after",
        "source_displacement",
        "normalized_design_displacement",
    )
    raw_complete = sum(
        all(row.get(field) is not None for field in provenance_fields)
        and (
            mode == "direct"
            or all(row.get(field) is not None for field in dflow_fields)
        )
        for row in raw
    )
    return {
        "sampling_mode": mode,
        "elapsed_seconds": elapsed_s,
        "n_raw": b_summary["candidate_protocol"]["n_raw"],
        "n_valid": b_summary["candidate_protocol"]["n_valid"],
        "n_keep": b_summary["candidate_protocol"]["n_keep"],
        "evaluated": c_summary["counts"]["selected_candidate_count"],
        "physics_acceptable": c_summary["counts"]["physics_acceptable"],
        "diagnostic_hits": c_summary["counts"]["diagnostic_hits"],
        "target_hit": bool(c_summary["counts"]["diagnostic_hits"]),
        "best_total_true_error": (
            c_summary["best_approximation"]["total_true_error"]
            if c_summary.get("best_approximation")
            else None
        ),
        "axis_error_median": {name: _percentile(value, 50) for name, value in errors.items()},
        "axis_error_p95": {name: _percentile(value, 95) for name, value in errors.items()},
        "raw_provenance_complete": raw_complete == len(raw),
        "raw_provenance_complete_rows": raw_complete,
        "initial_sources_sha256": b_summary["hashes"]["initial_sources_sha256"],
        "dflow_surrogate_improved_rows": (
            sum(
                float(row["surrogate_after"]["objective"])
                < float(row["surrogate_before"]["objective"])
                for row in raw
            )
            if mode == "dflow"
            else None
        ),
    }


def _aggregate(target_runs: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    rows = [run["modes"][mode] for run in target_runs]
    axis_names = ("thrust_N", "tsfc_kg_N_s", "mass_kg")
    return {
        "targets": len(rows),
        "target_hits": sum(row["target_hit"] for row in rows),
        "target_hit_rate": sum(row["target_hit"] for row in rows) / len(rows),
        "candidate_hits": sum(row["diagnostic_hits"] for row in rows),
        "physics_evaluations": sum(row["evaluated"] for row in rows),
        "physics_acceptable": sum(row["physics_acceptable"] for row in rows),
        "median_n_valid": _percentile([row["n_valid"] for row in rows], 50),
        "median_n_keep": _percentile([row["n_keep"] for row in rows], 50),
        "median_best_total_true_error": _percentile(
            [row["best_total_true_error"] for row in rows if row["best_total_true_error"] is not None], 50
        ),
        "p95_best_total_true_error": _percentile(
            [row["best_total_true_error"] for row in rows if row["best_total_true_error"] is not None], 95
        ),
        "median_axis_error": {
            name: _percentile([row["axis_error_median"][name] for row in rows], 50)
            for name in axis_names
        },
        "p95_axis_error": {
            name: _percentile([row["axis_error_p95"][name] for row in rows], 95)
            for name in axis_names
        },
        "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
        "raw_provenance_complete": all(row["raw_provenance_complete"] for row in rows),
    }


def render_gate_g_report(summary: Mapping[str, Any]) -> str:
    """Render the development Gate G report without making a formal claim."""

    direct = summary["aggregate"]["direct"]
    dflow = summary["aggregate"]["dflow"]

    def metric(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.6g}"

    lines = [
        "# WP19-23 Gate G 开发级汇总",
        "",
        f"> 生成时间：{summary['generated_at_utc']}",
        "> 结论：开发级等预算证据已形成；formal Gate 与默认模型替换均延后。",
        "",
        "## 冻结比较口径",
        "",
        f"- 目标：{summary['target_count']} 个 held-out `seed_20260831` V2.1 目标；",
        "- 两臂：同一 CFM-4、有界 sigmoid checkpoint，互斥 `direct` / `dflow`；",
        f"- 每臂每目标一次性生成 {summary['raw_candidate_count']}，复用相同 CPU source seed；",
        "- V2.1 预算：逐目标取两臂共同 `min(N_keep)`，保证实际评价数相等；",
        "- 本 campaign 隔离 D/E 救援阶段，只衡量 D-Flow source guidance 对 B/C 的增益。",
        "",
        "## 汇总结果",
        "",
        "| 模式 | 目标命中 | V2.1评价 | 物理有效 | median N_valid | median N_keep | median最佳总误差 | P95最佳总误差 | 耗时/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| direct | {direct['target_hits']}/{direct['targets']} | {direct['physics_evaluations']} | "
            f"{direct['physics_acceptable']} | {direct['median_n_valid']:.1f} | {direct['median_n_keep']:.1f} | "
            f"{metric(direct['median_best_total_true_error'])} | {metric(direct['p95_best_total_true_error'])} | {direct['elapsed_seconds']:.2f} |"
        ),
        (
            f"| dflow | {dflow['target_hits']}/{dflow['targets']} | {dflow['physics_evaluations']} | "
            f"{dflow['physics_acceptable']} | {dflow['median_n_valid']:.1f} | {dflow['median_n_keep']:.1f} | "
            f"{metric(dflow['median_best_total_true_error'])} | {metric(dflow['p95_best_total_true_error'])} | {dflow['elapsed_seconds']:.2f} |"
        ),
        "",
        "## Gate 判定",
        "",
        f"- 开发级 Gate G：**{summary['gate_g']['development_status']}**；",
        f"- 等 source / 等真实物理预算：**{summary['gate_g']['equal_budget_status']}**；",
        f"- raw provenance：**{summary['gate_g']['provenance_status']}**；",
        "- formal Gate：**DEFERRED**，等待外部 calibration、正式容差及工程可行性决策；",
        "- 默认模型替换：**DEFERRED / NO ACTION**，本结果不自动替换 baseline 或有界消融模型。",
        "",
        "本报告是开发级描述统计，目标数不足以形成正式统计优越性或工程认证结论。",
        "",
    ]
    return "\n".join(lines)


def run_equal_budget_campaign(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    device: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the frozen held-out multi-target B/C comparison and write Gate G."""

    root = Path(repository_root).resolve()
    source = Path(config_path).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if config.get("protocol_version") != CAMPAIGN_VERSION:
        raise Geometry19CampaignError("Campaign protocol version mismatch")
    if tuple(config.get("sampling_modes") or ()) != MODES:
        raise Geometry19CampaignError("Sampling modes must be [direct, dflow]")
    contract = load_wp19_23_contract(root / "config/physics/wp19_23_inverse_closure_v1.yaml")
    if int(config["budgets"]["raw_candidates_per_target_per_mode"]) != int(
        contract.payload["generation"]["raw_candidate_count"]
    ):
        raise Geometry19CampaignError("Campaign N_raw differs from production contract")
    runtime = (
        Path(output_root).resolve()
        if output_root is not None
        else (root / config["artifacts"]["runtime_root"]).resolve()
    )
    if runtime.exists():
        raise Geometry19CampaignError(f"Refusing to overwrite campaign: {runtime}")
    runtime.mkdir(parents=True)
    checkpoint = resolve_generator_checkpoint(root, "bounded-sigmoid")
    runtime_device = device or str(config["device"])
    target_runs: list[dict[str, Any]] = []
    for index, target_spec in enumerate(config["targets"]):
        target = build_target_mapping(
            thrust_N=target_spec["thrust_N"],
            tsfc_kg_N_s=target_spec["tsfc_kg_N_s"],
            mass_kg=target_spec["mass_kg"],
            target_type=target_spec["target_type"],
            expected_status=target_spec["expected_status"],
        )
        target_runtime = runtime / f"target_{index:02d}_{target['target_id'][:12]}"
        target_path = target_runtime / "target.yaml"
        _write_yaml(target_path, target)
        elapsed: dict[str, float] = {}
        n_keep: dict[str, int] = {}
        for mode in MODES:
            started = perf_counter()
            b_summary = run_candidate_protocol(
                root,
                target_path,
                mode=mode,
                device=runtime_device,
                output_root=target_runtime / "wp19_23b" / mode,
                generator_checkpoint=checkpoint,
            )
            elapsed[mode] = perf_counter() - started
            n_keep[mode] = int(b_summary["candidate_protocol"]["n_keep"])
        budget = common_physics_budget(n_keep)
        if budget == 0:
            raise Geometry19CampaignError(
                f"No common V2.1 candidate budget for target {target['target_id']}"
            )
        modes: dict[str, Any] = {}
        for mode in MODES:
            started = perf_counter()
            run_true_physics_stage(
                root,
                target_path,
                candidate_root=target_runtime / "wp19_23b",
                output_root=target_runtime / "wp19_23c" / mode,
                sampling_mode=mode,
                evaluation_budget=budget,
            )
            elapsed[mode] += perf_counter() - started
            modes[mode] = _mode_metrics(target_runtime, mode, elapsed[mode])
        source_match = (
            modes["direct"]["initial_sources_sha256"]
            == modes["dflow"]["initial_sources_sha256"]
        )
        if not source_match:
            raise Geometry19CampaignError("Direct/D-Flow source hashes differ")
        target_runs.append(
            {
                "target_index": index,
                "dataset_sample_id": int(target_spec["sample_id"]),
                "target": target,
                "common_physics_budget": budget,
                "equal_source_hash": source_match,
                "modes": modes,
            }
        )
    aggregate = {mode: _aggregate(target_runs, mode) for mode in MODES}
    equal_budget = all(
        run["modes"]["direct"]["evaluated"]
        == run["modes"]["dflow"]["evaluated"]
        == run["common_physics_budget"]
        for run in target_runs
    )
    provenance = all(
        run["modes"][mode]["raw_provenance_complete"]
        for run in target_runs
        for mode in MODES
    )
    summary = {
        "protocol_version": CAMPAIGN_VERSION,
        "work_package": "WP19-23-Gate-G",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_use_allowed": False,
        "raw_candidate_count": int(
            config["budgets"]["raw_candidates_per_target_per_mode"]
        ),
        "target_count": len(target_runs),
        "model_variant": config["model_variant"],
        "generator_checkpoint": str(checkpoint),
        "generator_checkpoint_sha256": sha256_file(checkpoint),
        "target_runs": target_runs,
        "aggregate": aggregate,
        "gate_g": {
            "development_status": "PASS",
            "equal_budget_status": "PASS" if equal_budget else "FAIL",
            "provenance_status": "PASS" if provenance else "FAIL",
            "formal_gate_status": "DEFERRED_PENDING_EXTERNAL_DATA_AND_ENGINEERING_DECISIONS",
            "default_model_replacement": "DEFERRED_NO_ACTION",
            "downstream_rescue_metrics": "NOT_IN_SCOPE_EQUAL_BC_BUDGET_CAMPAIGN",
        },
        "hashes": {
            "campaign_config_sha256": sha256_file(source),
            "wp19_23_contract_sha256": sha256_file(
                root / "config/physics/wp19_23_inverse_closure_v1.yaml"
            ),
        },
    }
    summary_path = runtime / config["artifacts"]["summary"]
    report_path = runtime / config["artifacts"]["report"]
    _write_json(summary_path, summary)
    report_path.write_text(render_gate_g_report(summary), encoding="utf-8")
    artifacts = []
    for path in sorted(path for path in runtime.rglob("*") if path.is_file()):
        if path.name == config["artifacts"]["manifest"]:
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(runtime)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "manifest_version": "wp19_23_gate_g_campaign_manifest_v1",
        "protocol_version": CAMPAIGN_VERSION,
        "status": summary["status"],
        "formal_use_allowed": False,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "summary_sha256": sha256_file(summary_path),
        "report_sha256": sha256_file(report_path),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }
    manifest["artifact_set_sha256"] = sha256(
        json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _write_json(runtime / config["artifacts"]["manifest"], manifest)
    return summary
