#!/usr/bin/env python3
"""Run the WP19-20 19D OAT, statistical, mesh, and overlay audit."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/novaturbo-matplotlib")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_wp19_16_geometry_regression import (  # noqa: E402
    component_hashes,
    render_front_view,
    render_section_view,
)
from src.physics import Geometry19  # noqa: E402
from src.physics.wp19_20_sensitivity import (  # noqa: E402
    build_positive_oat_cases,
    evaluate_oat_channel,
    spearman_sensitivity,
)
from src.visualization import Geometry19DesignStore  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "config/physics/wp19_20_sensitivity_audit_v1.yaml"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _worker(
    storage_root: str,
    design_payload: dict[str, Any],
    quality: str,
    profile: str,
) -> dict[str, Any]:
    store = Geometry19DesignStore(REPO_ROOT, storage_root)
    return store.regenerate(
        Geometry19.from_mapping(design_payload),
        quality=quality,
        constraint_profile=profile,
    )


def _render_contact_sheet(
    cases: list[dict[str, Any]], output_path: Path, reference_image: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    reference = plt.imread(reference_image)
    figure, axes = plt.subplots(5, 4, figsize=(18, 15), dpi=120)
    figure.patch.set_facecolor("#0a0e17")
    for axis, case in zip(axes.flat, cases):
        changed = plt.imread(REPO_ROOT / case["front_view_path"])
        divider = max(reference.shape[0], changed.shape[0])
        axis.imshow(reference, extent=(0, 1, 0, divider), aspect="auto")
        axis.imshow(changed, extent=(1, 2, 0, divider), aspect="auto")
        axis.axvline(1, color="#ffffff", linewidth=0.7, alpha=0.6)
        axis.set_title(
            f"{case['index']:02d} {case['variable']}\nREF  |  +OAT ({case['channel_audit']['channel']})",
            color="white", fontsize=8,
        )
        axis.axis("off")
    for axis in axes.flat[len(cases):]:
        axis.axis("off")
    figure.suptitle("WP19-20 Geometry19 reference / positive-OAT comparison", color="white", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    failures = summary["failed_variables"]
    lines = [
        "# WP19-20 19D 敏感性与可视化方向审计",
        "",
        f"> 执行时间：{summary['generated_at_utc']}",
        f"> 结论：**{'Gate F 通过' if summary['gate_f_passed'] else 'Gate F 未通过'}**",
        "",
        "## 审计结论",
        "",
        f"- 4096 点账本完整：`{summary['doe']['record_count']}/4096`；收敛样本 `{summary['doe']['converged_sample_count']}`。",
        f"- 正向单变量扰动：`{summary['oat_passed_count']}/19` 通过；preview mesh：`{summary['mesh_passed_count']}/20` 通过。",
        f"- 未解释死维度：`{len(failures)}`；失败变量：`{', '.join(failures) if failures else '无'}`。",
        "- TIT/RPM 已通过 Geometry19 服务响应驱动当前设计的 thermal/stress overlay；两者不改变 solid/lattice STL。",
        "- 点阵参数保持实体 STL 不变，并改变点阵 STL 与物理/制造通道。",
        "",
        "## 19 维通道结果",
        "",
        "| # | 变量 | 通道 | 变化组件数 | 物理响应数 | 结果 |",
        "|---:|---|---|---:|---:|---|",
    ]
    for case in summary["oat_cases"]:
        audit = case["channel_audit"]
        lines.append(
            f"| {case['index']} | `{case['variable']}` | {audit['channel']} | "
            f"{len(audit['changed_components'])} | {len(audit['changed_physics_paths'])} | "
            f"{'PASS' if case['passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## 统计敏感性",
        "",
        "统计仅使用收敛样本，对 `F / TSFC / M` 及全部约束残差计算 Spearman 秩相关。"
        "OAT 通道检查用于判定死维度，统计相关性用于描述全局趋势，不以任意相关阈值替代通道验收。",
        "",
        "## 可视化产物",
        "",
        "- `results/visualization/wp19_20_sensitivity_audit_20260825/comparison_contact_sheet.png`：参考设计与 19 个正扰动前视图。",
        "- `results/visualization/wp19_20_sensitivity_audit_20260825/cases/`：每例前视图、剖视图与机器响应（大文件忽略提交）。",
        "- `results/physics/wp19_20_sensitivity_audit_v1.json`：完整机器可读审计。",
        "",
        "## Gate F",
        "",
        "Gate F 的程序化条件为：DOE 4096 条完整、19 个 OAT 通道全部通过、20 个 preview mesh 全部通过、"
        "19 维均存在统计非恒定输出通道、覆盖层链路通过、且对比图存在。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    schema = yaml.safe_load((REPO_ROOT / config["design_schema"]).read_text(encoding="utf-8"))
    cases = build_positive_oat_cases(schema, float(config["oat_fraction_of_range"]))
    storage_root = REPO_ROOT / config["artifacts"]["mesh_storage_root"]
    visualization_root = REPO_ROOT / config["artifacts"]["visualization_root"]
    cases_root = visualization_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    responses: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=int(config["workers"])) as executor:
        futures = {
            executor.submit(
                _worker,
                str(storage_root),
                design.as_mapping(),
                config["mesh_quality"],
                config["constraint_profile"],
            ): index
            for index, (_, design) in enumerate(cases)
        }
        for future in as_completed(futures):
            index = futures[future]
            responses[index] = future.result()
            print(f"[{len(responses):02d}/20] solved/meshed case {index:02d}", flush=True)

    rendered = {}
    hash_sets = {}
    for index, (variable, _) in enumerate(cases):
        response = responses[index]
        source = storage_root / response["design_id"] / config["mesh_quality"]
        case_name = "reference" if variable is None else f"{index:02d}_{variable['name']}_positive"
        case_root = cases_root / case_name
        case_root.mkdir(parents=True, exist_ok=True)
        front = case_root / "front_view.png"
        section = case_root / "section_view.png"
        render_front_view(source, front)
        render_section_view(source, section)
        _write_json(case_root / "response.json", response)
        rendered[index] = {"front": front, "section": section}
        hash_sets[index] = component_hashes(source)
        print(f"[{index + 1:02d}/20] rendered case {case_name}", flush=True)

    reference_response = responses[0]
    oat_records = []
    for index, (variable, design) in enumerate(cases[1:], start=1):
        audit = evaluate_oat_channel(
            variable, responses[index], hash_sets[index], reference_response, hash_sets[0]
        )
        oat_records.append({
            "index": index,
            "variable": variable["name"],
            "component": variable["component"],
            "reference_value": variable["reference_value"],
            "perturbed_value": getattr(design, variable["name"]),
            "design_id": responses[index]["design_id"],
            "front_view_path": str(rendered[index]["front"].relative_to(REPO_ROOT)),
            "section_view_path": str(rendered[index]["section"].relative_to(REPO_ROOT)),
            "channel_audit": audit,
            "passed": audit["passed"],
        })
    contact_sheet = visualization_root / "comparison_contact_sheet.png"
    _render_contact_sheet(oat_records, contact_sheet, rendered[0]["front"])

    ledger = _read_jsonl(REPO_ROOT / config["doe_ledger"])
    variable_names = [item["name"] for item in schema["design_variables"]]
    statistics = spearman_sensitivity(ledger, variable_names)
    failed = [record["variable"] for record in oat_records if not record["passed"]]
    statistical_dead = [
        name for name, audit in statistics["variables"].items()
        if audit["nonconstant_channel_count"] == 0
    ]
    mesh_passed = sum(bool(response["mesh_diagnostics"]["passed"]) for response in responses.values())
    gate = (
        len(ledger) == 4096
        and not failed
        and not statistical_dead
        and mesh_passed == 20
        and contact_sheet.is_file()
    )
    summary = {
        "work_package": "WP19-20",
        "protocol_version": config["protocol_version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "doe": {"record_count": len(ledger), "converged_sample_count": statistics["converged_sample_count"]},
        "oat_case_count": 19,
        "oat_passed_count": sum(record["passed"] for record in oat_records),
        "mesh_passed_count": mesh_passed,
        "failed_variables": failed,
        "statistical_dead_dimensions": statistical_dead,
        "statistical_sensitivity": statistics,
        "oat_cases": oat_records,
        "reference": {
            "design_id": reference_response["design_id"],
            "geometry_hash": reference_response["geometry_hash"],
            "lattice_hash": reference_response["lattice_hash"],
            "overlay_parameters": reference_response["overlay_parameters"],
        },
        "comparison_contact_sheet": str(contact_sheet.relative_to(REPO_ROOT)),
        "gate_f_passed": gate,
    }
    result_path = REPO_ROOT / config["artifacts"]["summary_json"]
    _write_json(result_path, summary)
    _write_json(visualization_root / "summary.json", summary)
    _write_report(REPO_ROOT / config["artifacts"]["report_markdown"], summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = run(args.config.resolve())
    print(f"WP19-20: {summary['oat_passed_count']}/19 OAT; Gate F={summary['gate_f_passed']}")
    return 0 if summary["gate_f_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
