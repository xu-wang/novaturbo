#!/usr/bin/env python3
"""Collect the V2.1 completion evidence into one Markdown report.

Reads the artefacts produced by the retraining / verification / pipeline runs and
writes a self-contained report (hashes, metrics, statuses) so the result can be
audited without re-running anything.

Usage::

    python tools/v21/collect_evidence.py \
        --pipeline-root data/inverse_design/wp19_23_pipeline_v1/<target>/<variant>/<tag> \
        --bundle-root v21_capability \
        --output docs/V2_1_WEIGHTS_AND_STANDALONE_CAPABILITY.md
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.geometry19_surrogate import sha256_file  # noqa: E402

GENERATOR_DIR = Path(
    "data/generator_models/geometry19/wp19_23_cfm4_bounded_sigmoid_seed43_v1/baseline"
)
GENERATOR_RUN = Path(
    "data/inverse_design/wp19_23_geometry19_cfm4_bounded_sigmoid_seed43_v1/baseline/summary.json"
)
SURROGATE_DIR = Path(
    "data/inverse_design/wp19_22_performance_surrogate_v2_ablation/formal/models/performance3_log/seed_43"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _size(path: Path) -> str:
    return f"{path.stat().st_size / 1024:.1f} KiB" if path.is_file() else "missing"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, default=None)
    parser.add_argument("--bundle-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs/V2_1_WEIGHTS_AND_STANDALONE_CAPABILITY.md")
    args = parser.parse_args()

    lines: list[str] = []
    add = lines.append
    add("# NovaTurbo V2.1 权重补全与独立能力抽取 · 验收证据")
    add("")
    add(f"> 生成时间：{datetime.now(timezone.utc).isoformat()}（UTC）")
    add("> 本机：Windows + NVIDIA GeForce RTX 3090（物理索引 0，`CUDA_VISIBLE_DEVICES=0`）")
    add("> 运行环境：conda `novaturbo`，Python 3.13.14，torch 2.7.1+cu118")
    add("")

    add("## 1. 生成器权重（CFM-4 bounded-sigmoid, seed 43）")
    add("")
    generator_dir = REPO_ROOT / GENERATOR_DIR
    for name in ("best.pt", "normalizer.json", "last.pt", "training_history.json"):
        path = generator_dir / name
        state = "存在" if path.is_file() else "缺失"
        digest = sha256_file(path) if path.is_file() else "-"
        add(f"- `{GENERATOR_DIR / name}`：{state}，{_size(path)}，sha256 `{digest}`")
    summary_path = REPO_ROOT / GENERATOR_RUN
    if summary_path.is_file():
        summary = _read_json(summary_path)
        training = summary.get("training", {})
        validation = summary.get("validation", {})
        metrics = validation.get("sample_metrics", {})
        add("")
        add("| 项 | 值 |")
        add("|---|---|")
        add(f"| 状态 | `{summary.get('status')}` |")
        add(f"| 训练设备 | `{summary.get('device')}` |")
        add(f"| 完成 epoch | {training.get('epochs_completed')}（best {training.get('best_epoch')}）|")
        add(f"| 早停 | {training.get('early_stopped')} / {training.get('termination_reason')} |")
        add(f"| 参数量 | {training.get('total_parameters')} |")
        add(f"| 显存峰值 | {training.get('peak_gpu_memory_bytes')} B |")
        add(f"| 训练耗时 | {training.get('elapsed_seconds'):.1f} s |")
        add(f"| 源码积分一致 | {validation.get('source_integration_exact')} |")
        add(f"| checkpoint 重载一致 | {validation.get('checkpoint_reload_exact')} |")
        add(f"| 采样在界 | {metrics.get('in_bounds_count')}/{metrics.get('count')} |")
        add(f"| 样本去重率(1e-8) | {metrics.get('duplicate_rate')} |")
    else:
        add("- 训练 summary 缺失")

    add("")
    add("## 2. 性能代理权重（performance3_log, 三输出自然对数）")
    add("")
    surrogate_dir = REPO_ROOT / SURROGATE_DIR
    for name in ("weights_best_val.pt", "normalization.npz", "model_manifest.json", "training_metrics.json"):
        path = surrogate_dir / name
        state = "存在" if path.is_file() else "缺失"
        digest = sha256_file(path) if path.is_file() else "-"
        add(f"- `{SURROGATE_DIR / name}`：{state}，{_size(path)}，sha256 `{digest}`")
    metrics_path = surrogate_dir / "training_metrics.json"
    if metrics_path.is_file():
        metrics = _read_json(metrics_path)
        add("")
        add(f"- model_tag：`{metrics.get('model_tag')}`；output_transform：`{metrics.get('output_transform')}`；device：`{metrics.get('device')}`")
        add(f"- 完成 epoch：{metrics.get('completed_epochs')}（best {metrics.get('best_epoch')}）")
        add(f"- 数据划分：{metrics.get('split_counts')}")
        add("")
        add("| 轴 | 测试集 p50 相对误差 | 测试集 p90 | 验证集 p50 | R²（test）|")
        add("|---|---:|---:|---:|---:|")
        test_rel = metrics["test"]["relative"]
        val_rel = metrics["validation"]["relative"]
        test_r2 = metrics["test"]["r2"]
        for name in ("thrust_N", "tsfc_kg_N_s", "mass_kg"):
            add(
                f"| {name} | {test_rel[name]['relative_error_p50']:.4%} | "
                f"{test_rel[name]['relative_error_p90']:.4%} | "
                f"{val_rel[name]['relative_error_p50']:.4%} | "
                f"{test_r2[name]:.4f} |"
            )
    else:
        add("- 训练指标缺失")

    add("")
    add("## 3. V2.1 流水线端到端")
    add("")
    if args.pipeline_root:
        root = args.pipeline_root.resolve()
        summary_file = root / "pipeline_summary.json"
        manifest_file = root / "pipeline_manifest.json"
        if summary_file.is_file():
            summary = _read_json(summary_file)
            add(f"- 运行目录：`{root}`")
            add(f"- 状态：`{summary.get('status')}`；终止阶段：`{summary.get('terminal_source_stage')}`；交付状态：`{summary.get('delivery_status')}`")
            add(f"- 交付 design_id：`{summary.get('delivered_design_ids')}`")
            delivers = summary.get("delivered_candidates")
            if not isinstance(delivers, list):
                delivers = []
                for candidate in (root / "wp19_23c" / "delivered_candidates.json",
                                  root / "wp19_23f" / "delivered_candidates.json",
                                  root / "delivered_candidates.json"):
                    if candidate.is_file():
                        loaded = _read_json(candidate)
                        if isinstance(loaded, list):
                            delivers = loaded
                        break
            if isinstance(delivers, list) and delivers:
                for row in delivers[:5]:
                    hit = row.get("development_diagnostic_hit")
                    alignment = row.get("hit_alignment") or {}
                    rho = alignment.get("rho")
                    perf = row.get("performance") or {}
                    add(
                        f"  - candidate {row.get('candidate_id')}：hit={hit}，rho={rho}，"
                        f"failed_axes={alignment.get('failed_axis_count')}，"
                        f"thrust={perf.get('thrust_N')}，tsfc={perf.get('tsfc_kg_N_s')}，mass={perf.get('mass_kg')}"
                    )
            add(f"- 聚合 manifest：{'存在' if manifest_file.is_file() else '缺失'}（sha256 `{sha256_file(manifest_file) if manifest_file.is_file() else '-'}`）")
        else:
            add(f"- `{root}` 下没有 `pipeline_summary.json`")
    else:
        add("- 未提供 `--pipeline-root`")

    add("")
    add("## 4. 独立能力包")
    add("")
    if args.bundle_root:
        bundle = args.bundle_root.resolve()
        manifest_file = bundle / "bundle_manifest.json"
        if manifest_file.is_file():
            manifest = _read_json(manifest_file)
            add(f"- 目录：`{bundle}`")
            add(f"- 模块数：{manifest.get('modules_vendored')}；mesh 支持：{manifest.get('mesh_enabled')}")
            add(f"- 生成器权重：`{manifest.get('generator_weights')}`")
            add(f"- 性能代理权重：`{manifest.get('surrogate_weights')}`")
            total = sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file())
            add(f"- 体积：{total / 1024 / 1024:.1f} MiB")
        else:
            add(f"- `{bundle}` 下没有 `bundle_manifest.json`")
    else:
        add("- 未提供 `--bundle-root`")

    add("")
    add("## 5. 边界声明")
    add("")
    add("- `formal_use_allowed=false`、`deployment_allowed=false`：物理为项目内低阶闭式模型，不等于 CFD、台架试验或制造验证。")
    add("- 两份权重均为**本地按仓库冻结配置重训**的产物，不是发布版 checkpoint；不得引用原仓库的命中率或对比结论。")
    add("- 命中口径为 `rho = max(相对误差 / 轴容差)`，容差 2.7451% / 0.8573% / 1.1707%，仅作开发诊断。")

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
