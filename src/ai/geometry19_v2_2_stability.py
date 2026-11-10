"""Multi-target software-stability audit for the simplified V2.2 pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import yaml

from src.ai.geometry19_pipeline import run_wp19_23_pipeline
from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.ai.geometry19_pipeline_smoke import (
    Geometry19PipelineSmokeError,
    validate_v2_2_development_smoke,
)
from src.ai.geometry19_surrogate import sha256_file


CAMPAIGN_VERSION = "geometry19_v2_2_multitarget_software_stability_v1"
DOMAIN_CANDIDATE_VERSION = "geometry19_v2_2_development_target_domain_candidate_v1"


class Geometry19V22StabilityError(ValueError):
    """Raised when the software-stability campaign contract is invalid."""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_v2_2_stability_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the bounded, development-only campaign."""

    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Geometry19V22StabilityError("Campaign config must be a mapping")
    if payload.get("campaign_version") != CAMPAIGN_VERSION:
        raise Geometry19V22StabilityError("Campaign version mismatch")
    if payload.get("formal_use_allowed") is not False:
        raise Geometry19V22StabilityError("Campaign must remain non-formal")
    if payload.get("training_enabled") is not False:
        raise Geometry19V22StabilityError("Campaign must not enable training")
    if payload.get("doe_generation_enabled") is not False:
        raise Geometry19V22StabilityError("Campaign must not enable DOE generation")
    if payload.get("pipeline_profile") != "v2_2_development":
        raise Geometry19V22StabilityError("Campaign profile must be v2_2_development")
    domain_candidate = payload.get("domain_candidate_version")
    if domain_candidate is not None and domain_candidate != DOMAIN_CANDIDATE_VERSION:
        raise Geometry19V22StabilityError("Target-domain candidate version mismatch")
    execution = payload.get("execution")
    if not isinstance(execution, dict) or (
        execution.get("model_variant") != "baseline"
        or execution.get("sampling_mode") != "direct"
        or execution.get("device") != "cpu"
        or execution.get("refinement_device") != "cpu"
        or int(execution.get("raw_candidate_count_per_target", 0)) != 256
    ):
        raise Geometry19V22StabilityError("Campaign execution contract mismatch")
    maximum_seconds = float(execution.get("maximum_runtime_seconds_per_target", 0.0))
    if not math.isfinite(maximum_seconds) or maximum_seconds <= 0.0:
        raise Geometry19V22StabilityError("Campaign runtime budget must be positive")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not 5 <= len(targets) <= 10:
        raise Geometry19V22StabilityError("Campaign requires 5 to 10 targets")
    labels: set[str] = set()
    for target in targets:
        if not isinstance(target, dict):
            raise Geometry19V22StabilityError("Every target must be a mapping")
        label = str(target.get("label", ""))
        if not label or label in labels or "/" in label or ".." in label:
            raise Geometry19V22StabilityError("Target labels must be unique safe names")
        labels.add(label)
        if target.get("expected_status") != "UNKNOWN":
            raise Geometry19V22StabilityError(
                "V2.2 stability targets must use UNKNOWN expected status"
            )
        for name in ("thrust_N", "tsfc_kg_N_s", "mass_kg"):
            value = float(target.get(name, 0.0))
            if not math.isfinite(value) or value <= 0.0:
                raise Geometry19V22StabilityError(
                    f"Target {label} has invalid {name}"
                )
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        raise Geometry19V22StabilityError("Campaign acceptance contract is missing")
    minimum_rate = float(acceptance.get("minimum_pipeline_pass_rate", -1.0))
    minimum_valid = int(acceptance.get("minimum_targets_with_physics_acceptable", -1))
    if not 0.0 <= minimum_rate <= 1.0 or not 0 <= minimum_valid <= len(targets):
        raise Geometry19V22StabilityError("Campaign acceptance thresholds are invalid")
    if acceptance.get("require_all_artifact_contracts") is not True:
        raise Geometry19V22StabilityError("Artifact contracts must remain mandatory")
    return payload


def _target_result(runtime: Path, elapsed_seconds: float) -> dict[str, Any]:
    pipeline = _read_json(runtime / "pipeline_summary.json")
    generation = _read_json(runtime / "wp19_23b/direct/summary.json")
    physics = _read_json(runtime / "wp19_23c/summary.json")
    source_name = str(pipeline["terminal_source_stage"])
    source_directory = {
        "WP19-23C": "wp19_23c",
        "WP19-23D": "wp19_23d",
        "WP19-23E": "wp19_23e",
    }[source_name]
    terminal = _read_json(runtime / source_directory / "summary.json")
    best = terminal.get("best_approximation") or physics.get("best_approximation")
    physics_evaluations = int(physics["counts"]["unique_physics_evaluations"])
    for directory, count_name in (
        ("wp19_23d", "true_physics_evaluations"),
        ("wp19_23e", "unique_physics_evaluations"),
    ):
        summary_path = runtime / directory / "summary.json"
        if summary_path.is_file():
            physics_evaluations += int(
                _read_json(summary_path).get("counts", {}).get(count_name, 0)
            )
    c_physics_acceptable = int(physics["counts"]["physics_acceptable"])
    terminal_domain_valid = bool(
        best
        and best.get("constraint_assessment", {}).get("physics_domain_valid")
    )
    return {
        "status": "PASS",
        "elapsed_seconds": elapsed_seconds,
        "target_id": pipeline["target_id"],
        "completed_stages": pipeline["completed_stages"],
        "terminal_source_stage": source_name,
        "terminal_source_status": pipeline["terminal_source_status"],
        "delivery_status": pipeline["delivery_status"],
        "delivered_count": len(pipeline["delivered_design_ids"]),
        "n_raw": int(generation["candidate_protocol"]["n_raw"]),
        "n_valid": int(generation["candidate_protocol"]["n_valid"]),
        "n_keep": int(generation["candidate_protocol"]["n_keep"]),
        "physics_evaluations": physics_evaluations,
        "c_physics_acceptable": c_physics_acceptable,
        "terminal_physics_domain_valid": terminal_domain_valid,
        "has_physics_domain_valid": c_physics_acceptable > 0 or terminal_domain_valid,
        "best_rho": (
            float(best["hit_alignment"]["rho"])
            if best and best.get("hit_alignment")
            else None
        ),
        "best_engineering_feasible": bool(
            best
            and best.get("constraint_assessment", {}).get("engineering_feasible")
        ),
        "pipeline_summary_hash": sha256_file(runtime / "pipeline_summary.json"),
        "pipeline_manifest_hash": sha256_file(runtime / "pipeline_manifest.json"),
        "simplified_model": pipeline["simplified_model"],
        "development_only": pipeline["development_only"],
        "formal_use_allowed": pipeline["formal_use_allowed"],
        "target_hit": pipeline["target_hit"],
    }


def _partial_target_metrics(runtime: Path) -> dict[str, Any]:
    """Recover completed B/C metrics without treating a failed run as PASS."""

    metrics: dict[str, Any] = {
        "n_raw": 0,
        "n_valid": 0,
        "n_keep": 0,
        "physics_evaluations": 0,
        "c_physics_acceptable": 0,
        "terminal_physics_domain_valid": False,
        "has_physics_domain_valid": False,
        "best_rho": None,
    }
    generation_path = runtime / "wp19_23b/direct/summary.json"
    if generation_path.is_file():
        candidate = _read_json(generation_path)["candidate_protocol"]
        metrics.update(
            {
                "n_raw": int(candidate["n_raw"]),
                "n_valid": int(candidate["n_valid"]),
                "n_keep": int(candidate["n_keep"]),
            }
        )
    physics_path = runtime / "wp19_23c/summary.json"
    if physics_path.is_file():
        physics = _read_json(physics_path)
        metrics["physics_evaluations"] = int(
            physics["counts"]["unique_physics_evaluations"]
        )
        metrics["c_physics_acceptable"] = int(
            physics["counts"]["physics_acceptable"]
        )
        best = physics.get("best_approximation")
        metrics["terminal_physics_domain_valid"] = bool(
            best
            and best.get("constraint_assessment", {}).get("physics_domain_valid")
        )
        metrics["has_physics_domain_valid"] = bool(
            metrics["c_physics_acceptable"]
            or metrics["terminal_physics_domain_valid"]
        )
        if best and best.get("hit_alignment"):
            metrics["best_rho"] = float(best["hit_alignment"]["rho"])
    return metrics


def aggregate_v2_2_stability(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Aggregate target outcomes and apply the pre-registered software gate."""

    target_count = len(rows)
    passed = sum(row.get("status") == "PASS" for row in rows)
    valid_targets = sum(bool(row.get("has_physics_domain_valid")) for row in rows)
    artifact_passed = sum(row.get("artifact_contract_status") == "PASS" for row in rows)
    elapsed = [float(row["elapsed_seconds"]) for row in rows if row.get("elapsed_seconds")]
    rho = sorted(
        float(row["best_rho"])
        for row in rows
        if row.get("best_rho") is not None and math.isfinite(float(row["best_rho"]))
    )
    minimum_rate = float(config["acceptance"]["minimum_pipeline_pass_rate"])
    minimum_valid = int(
        config["acceptance"]["minimum_targets_with_physics_acceptable"]
    )
    maximum_seconds = float(
        config["execution"]["maximum_runtime_seconds_per_target"]
    )
    checks = {
        "pipeline_pass_rate": passed / target_count >= minimum_rate,
        "targets_with_physics_acceptable": valid_targets >= minimum_valid,
        "all_artifact_contracts": artifact_passed == target_count,
        "per_target_runtime_budget": all(value <= maximum_seconds for value in elapsed)
        and len(elapsed) == target_count,
        "no_formal_claim": all(row.get("formal_use_allowed") is False for row in rows),
        "no_target_hit_claim": all(row.get("target_hit") is False for row in rows),
    }
    return {
        "target_count": target_count,
        "pipeline_pass_count": passed,
        "pipeline_pass_rate": passed / target_count,
        "targets_with_physics_acceptable": valid_targets,
        "artifact_contract_pass_count": artifact_passed,
        "total_c_physics_acceptable": sum(
            int(row.get("c_physics_acceptable", 0)) for row in rows
        ),
        "total_physics_evaluations": sum(
            int(row.get("physics_evaluations", 0)) for row in rows
        ),
        "total_elapsed_seconds": sum(elapsed),
        "maximum_elapsed_seconds": max(elapsed) if elapsed else None,
        "median_best_rho": median(rho) if rho else None,
        "checks": checks,
        "software_stability_ready": all(checks.values()),
    }


def render_v2_2_stability_report(summary: Mapping[str, Any]) -> str:
    aggregate = summary["aggregate"]
    lines = [
        "# Geometry19 简化 V2.2 多目标软件稳定性审计",
        "",
        f"> 生成时间：{summary['generated_at_utc']}",
        "> 用途：验证简化模型的软件流程覆盖；不评价实验精度，不形成正式可行性声明。",
        "",
        "## 结论",
        "",
        f"- 软件稳定性门：**{'PASS' if aggregate['software_stability_ready'] else 'FAIL'}**；",
        f"- Pipeline PASS：{aggregate['pipeline_pass_count']}/{aggregate['target_count']}；",
        f"- 至少一个 physics-domain valid 候选：{aggregate['targets_with_physics_acceptable']}/{aggregate['target_count']}；",
        f"- 物理复算：{aggregate['total_physics_evaluations']} 次；C 阶段接受 {aggregate['total_c_physics_acceptable']} 次；",
        f"- 总 CPU 时间：{aggregate['total_elapsed_seconds']:.2f} s；单目标最大：{aggregate['maximum_elapsed_seconds']:.2f} s。",
        "",
        "## 逐目标结果",
        "",
        "| 目标 | Pipeline | B–F | N_valid | physics valid | best rho | 时间/s | 失败阶段 |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["targets"]:
        rho = "N/A" if row.get("best_rho") is None else f"{row['best_rho']:.4g}"
        lines.append(
            f"| {row['label']} | {row['status']} | "
            f"{len(row.get('completed_stages', []))}/5 | {row.get('n_valid', 0)} | "
            f"{row.get('c_physics_acceptable', 0)} | {rho} | "
            f"{row.get('elapsed_seconds', 0.0):.2f} | {row.get('failure_stage') or '-'} |"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 全部目标在 V2.2 中均按 `UNKNOWN` 处理；V2.1 来源标签不继承为 V2.2 可行性证据；",
            "- `target_hit=false` 表示没有正式命中声明，不否认开发诊断误差仍可用于排序；",
            "- 本审计不生成 DOE、不训练代理/CFM、不改变默认 `v2_1` profile。",
            "",
        ]
    )
    return "\n".join(lines)


def run_v2_2_stability_campaign(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    runtime_root: str | Path | None = None,
    evidence_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run all targets independently, preserving failures and producing one audit."""

    root = Path(repository_root).resolve()
    source = Path(config_path).resolve()
    config = load_v2_2_stability_config(source)
    runtime = (
        Path(runtime_root).resolve()
        if runtime_root is not None
        else (root / config["artifacts"]["runtime_root"]).resolve()
    )
    evidence = (
        Path(evidence_root).resolve()
        if evidence_root is not None
        else (root / config["artifacts"]["evidence_root"]).resolve()
    )
    if runtime.exists() or evidence.exists():
        raise Geometry19V22StabilityError(
            "Refusing to overwrite stability campaign runtime or evidence"
        )
    runtime.mkdir(parents=True)
    evidence.mkdir(parents=True)
    context = build_pipeline_context(root, pipeline_profile="v2_2_development")
    rows: list[dict[str, Any]] = []
    maximum_seconds = float(
        config["execution"]["maximum_runtime_seconds_per_target"]
    )
    for target in config["targets"]:
        label = str(target["label"])
        target_runtime = runtime / label
        started = perf_counter()
        try:
            run_wp19_23_pipeline(
                root,
                target_values={
                    "thrust_N": float(target["thrust_N"]),
                    "tsfc_kg_N_s": float(target["tsfc_kg_N_s"]),
                    "mass_kg": float(target["mass_kg"]),
                },
                target_type=str(target["target_type"]),
                expected_status="UNKNOWN",
                model_variant="baseline",
                sampling_mode="direct",
                device="cpu",
                refinement_device="cpu",
                output_root=target_runtime,
                storage_root=runtime / "mesh_storage",
                pipeline_profile="v2_2_development",
                pipeline_context=context,
            )
            elapsed = perf_counter() - started
            artifact = validate_v2_2_development_smoke(
                target_runtime,
                expected_raw_candidates=int(
                    config["execution"]["raw_candidate_count_per_target"]
                ),
                elapsed_seconds=elapsed,
                maximum_runtime_seconds=maximum_seconds,
            )
            row = _target_result(target_runtime, elapsed)
            row["artifact_contract_status"] = artifact["status"].removeprefix(
                "SMOKE_"
            )
            row["failure_stage"] = None
            row["error_type"] = None
            row["message"] = None
        except Exception as exc:
            elapsed = perf_counter() - started
            failure_path = target_runtime / "pipeline_failure.json"
            failure = _read_json(failure_path) if failure_path.is_file() else {}
            completed = list(failure.get("completed_stages", ()))
            row = {
                "status": "FAILED",
                "elapsed_seconds": elapsed,
                "target_id": failure.get("target_id"),
                "completed_stages": completed,
                "failure_stage": (
                    "B"
                    if not completed
                    else {1: "C", 2: "D", 3: "E"}.get(len(completed), "F")
                ),
                "artifact_contract_status": "FAILED",
                **_partial_target_metrics(target_runtime),
                "formal_use_allowed": False,
                "target_hit": False,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            if isinstance(exc, Geometry19PipelineSmokeError):
                row["failure_stage"] = "ARTIFACT_AUDIT"
        row.update(
            {
                "label": label,
                "source": target["source"],
                "target_type": target["target_type"],
                "expected_status": "UNKNOWN",
                "target": {
                    "thrust_N": float(target["thrust_N"]),
                    "tsfc_kg_N_s": float(target["tsfc_kg_N_s"]),
                    "mass_kg": float(target["mass_kg"]),
                },
                "runtime_root": str(target_runtime),
            }
        )
        rows.append(row)
        _write_json(evidence / "targets" / f"{label}.json", row)
    aggregate = aggregate_v2_2_stability(rows, config)
    summary = {
        "campaign_version": CAMPAIGN_VERSION,
        "domain_candidate_version": config.get("domain_candidate_version"),
        "work_package": config["work_package"],
        "status": "PASS" if aggregate["software_stability_ready"] else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "simplified_model": True,
        "development_only": True,
        "formal_use_allowed": False,
        "training_enabled": False,
        "doe_generation_enabled": False,
        "pipeline_profile": "v2_2_development",
        "pipeline_context_hash": context.context_hash,
        "physics_bundle_hash": context.physics_runtime.physics_bundle_hash,
        "config_hash": sha256_file(source),
        "code_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "runtime_root": str(runtime),
        "evidence_root": str(evidence),
        "aggregate": aggregate,
        "targets": rows,
    }
    summary_path = evidence / "summary.json"
    report_path = evidence / "REPORT.md"
    _write_json(summary_path, summary)
    report_path.write_text(render_v2_2_stability_report(summary), encoding="utf-8")
    _write_json(
        evidence / "manifest.json",
        {
            "campaign_version": CAMPAIGN_VERSION,
            "config_hash": summary["config_hash"],
            "pipeline_context_hash": summary["pipeline_context_hash"],
            "physics_bundle_hash": summary["physics_bundle_hash"],
            "summary_hash": sha256_file(summary_path),
            "report_hash": sha256_file(report_path),
            "target_artifact_hashes": {
                row["label"]: sha256_file(evidence / "targets" / f"{row['label']}.json")
                for row in rows
            },
        },
    )
    return summary
