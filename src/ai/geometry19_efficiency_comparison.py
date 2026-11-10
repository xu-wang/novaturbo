"""Compare the WP19-23 pipeline with a true cold-start NSGA-II baseline."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from src.ai.geometry19_delivery import run_delivery_stage
from src.ai.geometry19_inverse_contract import load_wp19_23_contract
from src.ai.geometry19_nsga2 import run_nsga2_stage
from src.ai.geometry19_pipeline import build_target_mapping, run_wp19_23_pipeline
from src.ai.geometry19_refinement import Geometry19RefinementError
from src.ai.geometry19_surrogate import sha256_file


COMPARISON_VERSION = "wp19_23_pipeline_vs_cold_nsga2_256_v1"


class Geometry19EfficiencyComparisonError(ValueError):
    """Raised when the pipeline/cold-start comparison is inconsistent."""


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


def _read_summary(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _project_counts(runtime: Path) -> dict[str, Any]:
    c = _read_summary(runtime / "wp19_23c/summary.json")
    d = _read_summary(runtime / "wp19_23d/summary.json")
    e = _read_summary(runtime / "wp19_23e/summary.json")
    f = _read_summary(runtime / "wp19_23f/summary.json")
    c_calls = int(c["counts"]["unique_v2_1_evaluations"]) if c else 0
    d_calls = int(d["counts"]["true_v2_1_evaluations"]) if d else 0
    e_calls = int(e["counts"]["unique_v2_1_evaluations"]) if e else 0
    delivery_calls = int(f["counts"]["v2_1_recalculation_passed"]) if f else 0
    hit = bool(f and f["counts"]["development_diagnostic_hits"])
    if c and c["downstream_stop"]:
        terminal_stage = "WP19-23C"
    elif d and d["downstream_stop"]:
        terminal_stage = "WP19-23D"
    elif e:
        terminal_stage = "WP19-23E"
    else:
        terminal_stage = "WP19-23C_NO_TRUE_PHYSICS_VALID_REFINEMENT_SEED"
    return {
        "wp19_23c_v2_1_calls": c_calls,
        "wp19_23d_v2_1_calls": d_calls,
        "wp19_23e_v2_1_calls": e_calls,
        "delivery_recalculation_calls": delivery_calls,
        "search_v2_1_calls": c_calls + d_calls + e_calls,
        "total_v2_1_calls": c_calls + d_calls + e_calls + delivery_calls,
        "diagnostic_hit_delivered": hit,
        "terminal_search_stage": terminal_stage,
        "method_completed": bool(f),
    }


def _cold_counts(e: Mapping[str, Any], f: Mapping[str, Any]) -> dict[str, Any]:
    search = int(e["counts"]["unique_v2_1_evaluations"])
    delivery = int(f["counts"]["v2_1_recalculation_passed"])
    return {
        "wp19_23e_v2_1_calls": search,
        "delivery_recalculation_calls": delivery,
        "search_v2_1_calls": search,
        "total_v2_1_calls": search + delivery,
        "diagnostic_hit_delivered": bool(
            f["counts"]["development_diagnostic_hits"]
        ),
        "stopped_after_generation": e["stopped_after_generation"],
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], target_class: str) -> dict[str, Any]:
    selected = [row for row in rows if row["target_class"] == target_class]
    savings = [float(row["saved_v2_1_calls"]) for row in selected]
    project_calls = [float(row["project"]["total_v2_1_calls"]) for row in selected]
    cold_calls = [float(row["cold_start"]["total_v2_1_calls"]) for row in selected]
    return {
        "runs": len(selected),
        "project_hits": sum(row["project"]["diagnostic_hit_delivered"] for row in selected),
        "cold_start_hits": sum(
            row["cold_start"]["diagnostic_hit_delivered"] for row in selected
        ),
        "project_total_v2_1_calls": int(sum(project_calls)),
        "cold_start_total_v2_1_calls": int(sum(cold_calls)),
        "saved_v2_1_calls": int(sum(savings)),
        "saved_fraction": (
            float(sum(savings) / sum(cold_calls)) if sum(cold_calls) else None
        ),
        "median_saved_v2_1_calls": float(np.median(savings)),
    }


def render_report(summary: Mapping[str, Any]) -> str:
    labels = {"ID": "数据集已有", "INTERPOLATION": "内插", "EXTRAPOLATION": "外插"}
    lines = [
        "# WP19-23 256候选 pipeline 与 cold-start NSGA-II 对照",
        "",
        f"> 生成时间：{summary['generated_at_utc']}",
        "> 正节省表示本项目pipeline少调用V2.1；负值表示pipeline反而多调用。",
        "",
        "## 冻结口径",
        "",
        f"- 本项目方法：`{summary['method']['model_variant']} CFM-4 + "
        f"{summary['method']['sampling_mode']}`，一次性 `N_raw={summary['raw_candidate_count']}`，随后按命中触发C/D/E/F；",
        "- 对照：同一目标、同一Geometry19边界、seed 42、64人口、最多640次V2.1的纯LHS cold-start NSGA-II；",
        "- 总调用数包含搜索阶段真实调用和WP19-23F强制交付复算；缓存命中不重复计数。",
        "",
        "## 分类汇总",
        "",
        "| 目标类型 | 运行数 | pipeline命中 | cold命中 | pipeline V2.1 | cold V2.1 | 节省调用 | 节省比例 | median节省 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("ID", "INTERPOLATION", "EXTRAPOLATION"):
        row = summary["aggregate"][key]
        fraction = "N/A" if row["saved_fraction"] is None else f"{row['saved_fraction']:.2%}"
        lines.append(
            f"| {labels[key]} | {row['runs']} | {row['project_hits']} | {row['cold_start_hits']} | "
            f"{row['project_total_v2_1_calls']} | {row['cold_start_total_v2_1_calls']} | "
            f"{row['saved_v2_1_calls']:+d} | {fraction} | {row['median_saved_v2_1_calls']:+.0f} |"
        )
    overall = summary["overall"]
    lines.extend(
        [
            "",
            "## 总计",
            "",
            f"pipeline共调用 `{overall['project_total_v2_1_calls']}` 次，cold-start共调用 "
            f"`{overall['cold_start_total_v2_1_calls']}` 次，净节省 "
            f"`{overall['saved_v2_1_calls']:+d}` 次（`{overall['saved_fraction']:.2%}`）。",
            "",
            f"结果一致性：双方命中 {overall['both_hit_runs']}，仅pipeline命中 "
            f"{overall['project_only_hit_runs']}，仅cold-start命中 "
            f"{overall['cold_only_hit_runs']}，双方均未命中 {overall['neither_hit_runs']}。",
            "",
            "只有双方达到同一成功判据时，调用差才能解释为成功求解效率。对于双方均未命中的运行，正节省仅表示pipeline在较小的动态人口/代数预算后返回最佳近似，并不表示更快求得命中解。",
            "",
        ]
    )
    return "\n".join(lines)


def run_efficiency_comparison(
    repository_root: str | Path,
    config_path: str | Path,
    *,
    device: str | None = None,
    output_root: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_source = Path(config_path).resolve()
    config = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    if config.get("protocol_version") != COMPARISON_VERSION:
        raise Geometry19EfficiencyComparisonError("Comparison protocol version mismatch")
    contract_path = root / "config/physics/wp19_23_inverse_closure_v1.yaml"
    contract = load_wp19_23_contract(contract_path)
    expected_raw = int(config["method"]["raw_candidate_count"])
    if expected_raw != int(contract.payload["generation"]["raw_candidate_count"]):
        raise Geometry19EfficiencyComparisonError(
            "Comparison N_raw differs from the active production contract"
        )
    if "targets_from_config" in config:
        target_config_path = root / str(config["targets_from_config"])
        target_config = yaml.safe_load(target_config_path.read_text(encoding="utf-8"))
        targets = target_config["targets"]
    else:
        target_config_path = config_source
        targets = config["targets"]
    runtime = (
        Path(output_root).resolve()
        if output_root is not None
        else (root / config["artifacts"]["runtime_root"]).resolve()
    )
    if runtime.exists() and not resume:
        raise Geometry19EfficiencyComparisonError(
            f"Refusing to overwrite comparison: {runtime}"
        )
    runtime.mkdir(parents=True, exist_ok=True)
    runtime_device = device or str(config["method"]["device"])
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(targets):
        target = build_target_mapping(
            thrust_N=spec["thrust_N"],
            tsfc_kg_N_s=spec["tsfc_kg_N_s"],
            mass_kg=spec["mass_kg"],
            target_type=spec["target_type"],
            expected_status=spec["expected_status"],
        )
        target_runtime = runtime / f"{index:02d}_{spec['label']}"
        target_path = target_runtime / "target.yaml"
        if target_path.is_file():
            if yaml.safe_load(target_path.read_text(encoding="utf-8")) != target:
                raise Geometry19EfficiencyComparisonError("Resume target mismatch")
        else:
            _write_yaml(target_path, target)
        project_root = target_runtime / "project_pipeline"
        project_summary_path = project_root / "pipeline_summary.json"
        method_stop_path = project_root / "comparison_method_stop.json"
        refinement_failure_path = project_root / "wp19_23d/failure.json"
        if (
            not method_stop_path.is_file()
            and refinement_failure_path.is_file()
        ):
            failure = json.loads(refinement_failure_path.read_text(encoding="utf-8"))
            if (
                failure.get("error_type") == "Geometry19RefinementError"
                and failure.get("message")
                == "No true-physics-valid refinement seed exists"
            ):
                _write_json(
                    method_stop_path,
                    {
                        "status": "NO_TRUE_PHYSICS_VALID_REFINEMENT_SEED",
                        "target_id": target["target_id"],
                        "formal_use_allowed": False,
                    },
                )
        if not project_summary_path.is_file() and not method_stop_path.is_file():
            try:
                run_wp19_23_pipeline(
                    root,
                    target_path=target_path,
                    model_variant=config["method"]["model_variant"],
                    sampling_mode=config["method"]["sampling_mode"],
                    device=runtime_device,
                    refinement_device=runtime_device,
                    output_root=project_root,
                    storage_root=runtime / "mesh_storage/project",
                    resume=resume and project_root.exists(),
                )
            except Geometry19RefinementError as exc:
                if str(exc) != "No true-physics-valid refinement seed exists":
                    raise
                _write_json(
                    method_stop_path,
                    {
                        "status": "NO_TRUE_PHYSICS_VALID_REFINEMENT_SEED",
                        "target_id": target["target_id"],
                        "formal_use_allowed": False,
                    },
                )
        project = _project_counts(project_root)
        cold_root = target_runtime / "cold_start_nsga2"
        cold_e_root = cold_root / "wp19_23e"
        cold_e = _read_summary(cold_e_root / "summary.json")
        if cold_e is None:
            cold_e = run_nsga2_stage(
                root,
                target_path,
                cold_start_comparison=True,
                output_root=cold_e_root,
            )
        cold_f_root = cold_root / "wp19_23f"
        cold_f = _read_summary(cold_f_root / "summary.json")
        if cold_f is None:
            cold_f = run_delivery_stage(
                root,
                target_path,
                source_root=cold_e_root,
                output_root=cold_f_root,
                storage_root=runtime / "mesh_storage/cold_start",
                network_architecture_modified=False,
            )
        cold = _cold_counts(cold_e, cold_f)
        saved = cold["total_v2_1_calls"] - project["total_v2_1_calls"]
        target_class = (
            "EXTRAPOLATION"
            if spec["target_type"] == "OUT_OF_EMPIRICAL_SUPPORT"
            else spec["target_type"]
        )
        rows.append(
            {
                "label": spec["label"],
                "target_class": target_class,
                "source_kind": spec["source_kind"],
                "source_sample_ids": list(spec["source_sample_ids"]),
                "target": target,
                "project": project,
                "cold_start": cold,
                "saved_v2_1_calls": saved,
                "saved_fraction": saved / cold["total_v2_1_calls"],
                "comparison_interpretation": (
                    "BOTH_HIT"
                    if project["diagnostic_hit_delivered"]
                    and cold["diagnostic_hit_delivered"]
                    else "PROJECT_ONLY_HIT"
                    if project["diagnostic_hit_delivered"]
                    else "COLD_ONLY_HIT"
                    if cold["diagnostic_hit_delivered"]
                    else "NEITHER_HIT_BUDGET_COMPARISON_ONLY"
                ),
            }
        )
    aggregate = {
        target_class: _aggregate(rows, target_class)
        for target_class in ("ID", "INTERPOLATION", "EXTRAPOLATION")
    }
    project_total = sum(row["project"]["total_v2_1_calls"] for row in rows)
    cold_total = sum(row["cold_start"]["total_v2_1_calls"] for row in rows)
    summary = {
        "protocol_version": COMPARISON_VERSION,
        "work_package": config["work_package"],
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_use_allowed": False,
        "raw_candidate_count": expected_raw,
        "method": dict(config["method"]),
        "target_runs": rows,
        "aggregate": aggregate,
        "overall": {
            "runs": len(rows),
            "project_total_v2_1_calls": project_total,
            "cold_start_total_v2_1_calls": cold_total,
            "saved_v2_1_calls": cold_total - project_total,
            "saved_fraction": (cold_total - project_total) / cold_total,
            "both_hit_runs": sum(
                row["comparison_interpretation"] == "BOTH_HIT" for row in rows
            ),
            "project_only_hit_runs": sum(
                row["comparison_interpretation"] == "PROJECT_ONLY_HIT" for row in rows
            ),
            "cold_only_hit_runs": sum(
                row["comparison_interpretation"] == "COLD_ONLY_HIT" for row in rows
            ),
            "neither_hit_runs": sum(
                row["comparison_interpretation"]
                == "NEITHER_HIT_BUDGET_COMPARISON_ONLY"
                for row in rows
            ),
        },
        "hashes": {
            "config_sha256": sha256_file(config_source),
            "target_config_sha256": sha256_file(target_config_path),
            "protocol_sha256": sha256_file(contract_path),
        },
    }
    summary_path = runtime / config["artifacts"]["summary"]
    report_path = runtime / config["artifacts"]["report"]
    _write_json(summary_path, summary)
    report_path.write_text(render_report(summary), encoding="utf-8")
    artifacts = [
        {
            "path": str(path.relative_to(runtime)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(path for path in runtime.rglob("*") if path.is_file())
        if path.name != config["artifacts"]["manifest"]
    ]
    _write_json(
        runtime / config["artifacts"]["manifest"],
        {
            "manifest_version": "wp19_23_efficiency_comparison_manifest_v1",
            "protocol_version": COMPARISON_VERSION,
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
        },
    )
    return summary
