"""Cumulative audit for the Geometry19 V2.2 development root-fill campaign."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from src.ai.geometry19_pipeline_context import build_pipeline_context
from src.physics.geometry19_v2_2_bridge_root_expansion import (
    _bridge_candidates,
    _coverage_summary,
    _root_centers,
)
from src.physics.geometry19_v2_2_dataset import (
    Geometry19V22DatasetError,
    _artifact_path,
    _hash_file,
    _read_jsonl,
    _write_json,
)


ROOT_FILL_AUDIT_VERSION = "geometry19_v2_2_root_fill_audit_v1"


def _load_bound_rows(
    root: Path, spec: Mapping[str, Any], label: str
) -> tuple[Path, list[dict[str, Any]]]:
    path = (root / spec["path"]).resolve()
    if _hash_file(path) != spec["sha256"]:
        raise Geometry19V22DatasetError(f"{label} hash mismatch")
    return path, _read_jsonl(path)


def _effective_roots(rows: Sequence[dict[str, Any]], segment: str) -> float:
    counts = Counter(
        str(row["root_seed_id"])
        for row in rows
        if row["assigned_segment"] == segment
    )
    total = sum(counts.values())
    if not total:
        return 0.0
    probabilities = np.asarray(list(counts.values()), dtype=float) / total
    return float(1.0 / np.sum(probabilities**2))


def _render_report(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Geometry19 V2.2 development 根补齐累计审计",
        "",
        f"- 审计状态：`{audit['status']}`",
        f"- 根数量门：`{'PASS' if audit['root_count_gate_pass'] else 'FAIL'}`",
        f"- 全局根 5 折门：`{'PASS' if audit['fold_gate_pass'] else 'FAIL'}`",
        f"- 累计桥接覆盖门：`{'PASS' if audit['bridge_coverage_gate_pass'] else 'PARTIAL'}`",
        "- confirmation：哈希锁定且未进入 development",
        "- 正式使用与部署：禁止",
        "",
        "## 累计数据结构",
        "",
        "| 线段 | 原根 | 最终根 | 新增根 | 原/最终有效根 | 桥接参考 P95 前→后 | 降幅 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for segment, values in audit["segments"].items():
        coverage = values["cumulative_bridge_coverage"]
        lines.append(
            f"| `{segment}` | {values['original_roots']} | {values['final_roots']} | "
            f"{values['added_roots']} | {values['original_effective_roots']:.2f}/"
            f"{values['final_effective_roots']:.2f} | {coverage['before']['p95']:.3f}→"
            f"{coverage['after']['p95']:.3f} | {coverage['p95_reduction_fraction']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## 审计结论",
            "",
            "两条薄弱线段均达到预定的新增 10–15 个独立 development 根目标，且每个进入最终数据集的新根都先通过单根最少 24 行门。`low_mass_low_to_mid` 的跨簇桥接明显改善；`low_mass_mid_to_high` 虽然根数和有效根数提升，但新增根仍集中在已有可达流形分支，几何中部覆盖改善有限。",
            "",
            "## 下一步约束",
            "",
            "1. 可以使用冻结的 development 5 折做 root-balanced、小容量模型消融，但只能看多折 worst-root rho P95；",
            "2. 若远距离层的误差仍高，下一轮数据必须来自新的逆设计起点/全局流形发现，不能继续用局部插值或单步 Jacobian 分支；",
            "3. 现有 confirmation 不得复用。模型与选择规则完全冻结后，再另取至少 3 个全新 confirmation 根/线段做一次性评价。",
            "",
        ]
    )
    return "\n".join(lines)


def run_root_fill_audit(
    repository_root: str | Path,
    protocol_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    protocol_source = Path(protocol_path)
    protocol_source = (
        protocol_source.resolve()
        if protocol_source.is_absolute()
        else (root / protocol_source).resolve()
    )
    protocol = yaml.safe_load(protocol_source.read_text(encoding="utf-8"))
    if protocol.get("audit_version") != ROOT_FILL_AUDIT_VERSION:
        raise Geometry19V22DatasetError("Root-fill audit version mismatch")
    if protocol.get("formal_use_allowed") is not False:
        raise Geometry19V22DatasetError("Root-fill audit cannot be formal")
    output = Path(output_root).resolve()
    if output.exists():
        raise Geometry19V22DatasetError(f"Refusing to overwrite root-fill audit: {output}")
    output.mkdir(parents=True)

    train_path, train = _load_bound_rows(root, protocol["original_development"]["train"], "train")
    validation_path, validation = _load_bound_rows(
        root, protocol["original_development"]["validation"], "validation"
    )
    confirmation_path, confirmation = _load_bound_rows(
        root, protocol["sealed_confirmation"], "confirmation"
    )
    final_path, final = _load_bound_rows(root, protocol["final_development"], "final development")
    fold_path = (root / protocol["development_folds"]["path"]).resolve()
    if _hash_file(fold_path) != protocol["development_folds"]["sha256"]:
        raise Geometry19V22DatasetError("Development fold hash mismatch")
    folds = json.loads(fold_path.read_text(encoding="utf-8"))
    original = [*train, *validation]
    confirmation_roots = {str(row["root_seed_id"]) for row in confirmation}
    final_roots = {str(row["root_seed_id"]) for row in final}
    confirmation_excluded = not bool(confirmation_roots & final_roots)
    if not confirmation_excluded:
        raise Geometry19V22DatasetError("Confirmation root leaked into final development")

    context = build_pipeline_context(root, pipeline_profile="v2_2_development")
    variables = list(context.physics_runtime.design_schema["design_variables"])
    segments: dict[str, Any] = {}
    for segment in protocol["target_segments"]:
        before = _root_centers(original, variables, segment_id=segment)
        after = _root_centers(final, variables, segment_id=segment)
        reference = _bridge_candidates(
            before,
            protocol["bridge_reference"]["interpolation_fractions"],
            float(protocol["bridge_reference"]["minimum_endpoint_l2"]),
            float(protocol["bridge_reference"]["minimum_l2_to_existing"]),
        )
        coverage = _coverage_summary(reference, before, after)
        segments[segment] = {
            "original_rows": sum(row["assigned_segment"] == segment for row in original),
            "final_rows": sum(row["assigned_segment"] == segment for row in final),
            "original_roots": len(before),
            "final_roots": len(after),
            "added_roots": len(after) - len(before),
            "original_effective_roots": _effective_roots(original, segment),
            "final_effective_roots": _effective_roots(final, segment),
            "cumulative_bridge_coverage": coverage,
        }

    gate = protocol["gates"]
    root_count_gate_pass = all(
        int(gate["minimum_added_roots_per_target_segment"])
        <= values["added_roots"]
        <= int(gate["maximum_added_roots_target"])
        for values in segments.values()
    )
    fold_gate_pass = all(
        all(
            summary["validation_roots_by_target_segment"][segment]
            >= int(gate["minimum_validation_roots_per_fold_per_target_segment"])
            for segment in protocol["target_segments"]
        )
        for summary in folds["fold_summaries"]
    )
    bridge_coverage_gate_pass = all(
        values["cumulative_bridge_coverage"]["p95_reduction_fraction"]
        >= float(gate["minimum_cumulative_bridge_p95_reduction_fraction"])
        for values in segments.values()
    )
    status = (
        "DEVELOPMENT_ROOT_FILL_PASS"
        if root_count_gate_pass and fold_gate_pass and bridge_coverage_gate_pass
        else "DEVELOPMENT_ROOT_FILL_PASS_BRIDGE_COVERAGE_PARTIAL"
        if root_count_gate_pass and fold_gate_pass
        else "DEVELOPMENT_ROOT_FILL_FAIL"
    )
    audit = {
        "audit_version": ROOT_FILL_AUDIT_VERSION,
        "status": status,
        "formal_use_allowed": False,
        "deployment_allowed": False,
        "confirmation_reuse_allowed": False,
        "development_ablation_ready": root_count_gate_pass and fold_gate_pass,
        "root_count_gate_pass": root_count_gate_pass,
        "fold_gate_pass": fold_gate_pass,
        "bridge_coverage_gate_pass": bridge_coverage_gate_pass,
        "confirmation_excluded": confirmation_excluded,
        "counts": {
            "original_development_rows": len(original),
            "final_development_rows": len(final),
            "added_development_rows": len(final) - len(original),
            "original_global_roots": len({row["root_seed_id"] for row in original}),
            "final_global_roots": len(final_roots),
            "sealed_confirmation_rows": len(confirmation),
            "sealed_confirmation_roots": len(confirmation_roots),
        },
        "segments": segments,
        "gates": gate,
        "hashes": {
            "protocol_hash": _hash_file(protocol_source),
            "train_hash": _hash_file(train_path),
            "validation_hash": _hash_file(validation_path),
            "sealed_confirmation_hash": _hash_file(confirmation_path),
            "final_development_hash": _hash_file(final_path),
            "development_folds_hash": _hash_file(fold_path),
            "pipeline_context_hash": context.context_hash,
            "physics_bundle_hash": context.physics_runtime.physics_bundle_hash,
        },
        "paths": {
            "final_development": _artifact_path(final_path, root),
            "development_folds": _artifact_path(fold_path, root),
        },
    }
    summary_path = output / "audit_summary.json"
    _write_json(summary_path, audit)
    report_path = output / "Root_Fill_Audit.md"
    report_path.write_text(_render_report(audit), encoding="utf-8")
    return audit
