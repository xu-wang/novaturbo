"""Compare aligned WP19-23 baseline and enhanced method-arm summaries."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from src.ai.geometry19_surrogate import sha256_file


UPLIFT_VERSION = "wp19_23_method_uplift_256_v1"


class Geometry19MethodUpliftError(ValueError):
    """Raised when two method arms cannot be compared target by target."""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _aggregate(rows: list[Mapping[str, Any]], target_class: str) -> dict[str, Any]:
    selected = [row for row in rows if row["target_class"] == target_class]
    count = len(selected)
    baseline_hits = sum(row["baseline_hit"] for row in selected)
    enhanced_hits = sum(row["enhanced_hit"] for row in selected)
    baseline_calls = sum(row["baseline_v2_1_calls"] for row in selected)
    enhanced_calls = sum(row["enhanced_v2_1_calls"] for row in selected)
    return {
        "runs": count,
        "baseline_hits": baseline_hits,
        "enhanced_hits": enhanced_hits,
        "hit_count_uplift": enhanced_hits - baseline_hits,
        "baseline_hit_rate": baseline_hits / count,
        "enhanced_hit_rate": enhanced_hits / count,
        "hit_rate_uplift_percentage_points": 100.0
        * (enhanced_hits - baseline_hits)
        / count,
        "baseline_v2_1_calls": baseline_calls,
        "enhanced_v2_1_calls": enhanced_calls,
        "additional_v2_1_calls": enhanced_calls - baseline_calls,
    }


def render_report(summary: Mapping[str, Any]) -> str:
    labels = {"ID": "数据集已有", "INTERPOLATION": "内插", "EXTRAPOLATION": "外插"}
    lines = [
        "# WP19-23 bounded-sigmoid + D-Flow + gradient命中率提升",
        "",
        f"> 生成时间：{summary['generated_at_utc']}",
        "",
        "## 对齐口径",
        "",
        "- 基线臂：baseline CFM-4 + direct + 条件gradient refinement；",
        "- 增强臂：bounded-sigmoid CFM-4 + D-Flow + 条件gradient refinement；",
        "- 两臂共享同一9目标、N_raw=256、V2.1容差、NSGA-II与交付复算协议；",
        "- 命中率提升按同目标逐一配对计算。",
        "",
        "## 分类汇总",
        "",
        "| 目标类型 | 目标数 | 基线命中 | 增强命中 | 提升/百分点 | 基线调用 | 增强调用 | 调用变化 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("ID", "INTERPOLATION", "EXTRAPOLATION"):
        row = summary["aggregate"][key]
        lines.append(
            f"| {labels[key]} | {row['runs']} | {row['baseline_hits']} | {row['enhanced_hits']} | "
            f"{row['hit_rate_uplift_percentage_points']:+.2f} | {row['baseline_v2_1_calls']} | "
            f"{row['enhanced_v2_1_calls']} | {row['additional_v2_1_calls']:+d} |"
        )
    overall = summary["overall"]
    lines.extend(
        [
            "",
            "## 总体",
            "",
            f"命中由 `{overall['baseline_hits']}/{overall['runs']}` "
            f"（`{overall['baseline_hit_rate']:.2%}`）提高到 "
            f"`{overall['enhanced_hits']}/{overall['runs']}` "
            f"（`{overall['enhanced_hit_rate']:.2%}`），提升 "
            f"`{overall['hit_rate_uplift_percentage_points']:+.2f}` 个百分点。",
            "",
            f"增强臂相对基线实际V2.1调用变化为 `{overall['additional_v2_1_calls']:+d}`。",
            "",
            "外插预期不可行目标应优先解释为OOD压力测试；总体命中率与域内命中率必须同时报告。",
            "",
        ]
    )
    return "\n".join(lines)


def compare_method_arms(
    baseline_summary_path: str | Path,
    enhanced_summary_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    baseline_path = Path(baseline_summary_path).resolve()
    enhanced_path = Path(enhanced_summary_path).resolve()
    runtime = Path(output_root).resolve()
    if runtime.exists():
        raise Geometry19MethodUpliftError(f"Refusing to overwrite uplift run: {runtime}")
    runtime.mkdir(parents=True)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    enhanced = json.loads(enhanced_path.read_text(encoding="utf-8"))
    baseline_rows = {row["label"]: row for row in baseline["target_runs"]}
    enhanced_rows = {row["label"]: row for row in enhanced["target_runs"]}
    if set(baseline_rows) != set(enhanced_rows):
        raise Geometry19MethodUpliftError("Method arms have different target labels")
    rows = []
    for label in baseline_rows:
        left = baseline_rows[label]
        right = enhanced_rows[label]
        if left["target"] != right["target"] or left["target_class"] != right["target_class"]:
            raise Geometry19MethodUpliftError(f"Target mismatch for {label}")
        baseline_hit = bool(left["project"]["diagnostic_hit_delivered"])
        enhanced_hit = bool(right["project"]["diagnostic_hit_delivered"])
        rows.append(
            {
                "label": label,
                "target_class": left["target_class"],
                "target_id": left["target"]["target_id"],
                "baseline_hit": baseline_hit,
                "enhanced_hit": enhanced_hit,
                "hit_change": (
                    "GAINED_HIT" if enhanced_hit and not baseline_hit else
                    "LOST_HIT" if baseline_hit and not enhanced_hit else
                    "UNCHANGED_HIT" if baseline_hit else "UNCHANGED_MISS"
                ),
                "baseline_terminal_stage": left["project"]["terminal_search_stage"],
                "enhanced_terminal_stage": right["project"]["terminal_search_stage"],
                "baseline_v2_1_calls": int(left["project"]["total_v2_1_calls"]),
                "enhanced_v2_1_calls": int(right["project"]["total_v2_1_calls"]),
                "additional_v2_1_calls": int(
                    right["project"]["total_v2_1_calls"]
                    - left["project"]["total_v2_1_calls"]
                ),
            }
        )
    aggregate = {
        key: _aggregate(rows, key) for key in ("ID", "INTERPOLATION", "EXTRAPOLATION")
    }
    all_runs = len(rows)
    baseline_hits = sum(row["baseline_hit"] for row in rows)
    enhanced_hits = sum(row["enhanced_hit"] for row in rows)
    overall = {
        "runs": all_runs,
        "baseline_hits": baseline_hits,
        "enhanced_hits": enhanced_hits,
        "baseline_hit_rate": baseline_hits / all_runs,
        "enhanced_hit_rate": enhanced_hits / all_runs,
        "hit_count_uplift": enhanced_hits - baseline_hits,
        "hit_rate_uplift_percentage_points": 100.0
        * (enhanced_hits - baseline_hits)
        / all_runs,
        "baseline_v2_1_calls": sum(row["baseline_v2_1_calls"] for row in rows),
        "enhanced_v2_1_calls": sum(row["enhanced_v2_1_calls"] for row in rows),
        "additional_v2_1_calls": sum(row["additional_v2_1_calls"] for row in rows),
        "gained_hits": sum(row["hit_change"] == "GAINED_HIT" for row in rows),
        "lost_hits": sum(row["hit_change"] == "LOST_HIT" for row in rows),
    }
    summary = {
        "protocol_version": UPLIFT_VERSION,
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_use_allowed": False,
        "target_runs": rows,
        "aggregate": aggregate,
        "overall": overall,
        "hashes": {
            "baseline_summary_sha256": sha256_file(baseline_path),
            "enhanced_summary_sha256": sha256_file(enhanced_path),
        },
    }
    summary_path = runtime / "method_uplift_summary.json"
    report_path = runtime / "REPORT.md"
    _write_json(summary_path, summary)
    report_path.write_text(render_report(summary), encoding="utf-8")
    _write_json(
        runtime / "manifest.json",
        {
            "manifest_version": "wp19_23_method_uplift_manifest_v1",
            "summary_sha256": sha256_file(summary_path),
            "report_sha256": sha256_file(report_path),
            "inputs": summary["hashes"],
        },
    )
    return summary
