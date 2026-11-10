#!/usr/bin/env python3
"""Summarize S2-WP4D random Gaussian source sensitivity."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_s2_wp4d import DEFAULT_CONFIG, load_matrix
from scripts.summarize_cfm_2 import aggregate
from src.ai.baseline import sha256_file


OUTPUT = Path("results/stage2/s2_wp4d")
REFERENCE_DIRS = {
    42: Path(
        "data/inverse_design/stage2/cfm_arch/"
        "s2_wp4c_steps32_seed42_v2_1"
    ),
    43: Path(
        "data/inverse_design/stage2/cfm_arch/"
        "s2_wp4c_steps32_seed43_v2_1"
    ),
}
VOLATILE_COLUMNS = {"sampling_elapsed_seconds", "generator_time", "0d_time"}


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def run_dir(matrix: dict, checkpoint_seed: int, source_seed: int) -> Path:
    return (
        Path(matrix["output"]["root"])
        / f"checkpoint{checkpoint_seed}_source{source_seed}_v2_1"
    )


def descriptive_effect_partition(values: pd.DataFrame) -> dict[str, float]:
    """Partition target-demeaned log-error SS in the crossed 2x8 matrix."""

    ss_checkpoint = 0.0
    ss_source = 0.0
    ss_interaction = 0.0
    for _, target in values.groupby("target_id", sort=False):
        pivot = target.pivot(
            index="checkpoint_seed", columns="source_seed", values="log10_best_error"
        )
        if pivot.shape != (2, 8) or pivot.isna().any().any():
            raise RuntimeError("Each target must contain a complete 2x8 crossed matrix")
        array = pivot.to_numpy(dtype=float)
        grand = float(array.mean())
        row_mean = array.mean(axis=1)
        column_mean = array.mean(axis=0)
        ss_checkpoint += array.shape[1] * float(np.square(row_mean - grand).sum())
        ss_source += array.shape[0] * float(np.square(column_mean - grand).sum())
        residual = array - row_mean[:, None] - column_mean[None, :] + grand
        ss_interaction += float(np.square(residual).sum())
    total = ss_checkpoint + ss_source + ss_interaction
    return {
        "checkpoint_fraction": ss_checkpoint / total,
        "source_fraction": ss_source / total,
        "checkpoint_source_interaction_fraction": ss_interaction / total,
        "total_target_demeaned_ss": total,
    }


def main() -> int:
    matrix_path = REPO_ROOT / DEFAULT_CONFIG
    matrix = load_matrix(matrix_path)
    output = REPO_ROOT / OUTPUT
    output.mkdir(parents=True, exist_ok=True)

    frames = []
    run_rows = []
    gate_rows = []
    for checkpoint_seed in (42, 43):
        for source_seed in matrix["source_seeds"]:
            directory = REPO_ROOT / run_dir(matrix, checkpoint_seed, source_seed)
            frame = pd.read_csv(directory / "target_summary.csv")
            if len(frame) != 30:
                raise RuntimeError(f"Expected 30 targets in {directory}")
            frame.insert(0, "checkpoint_seed", checkpoint_seed)
            frame.insert(1, "source_seed", int(source_seed))
            frames.append(frame)
            overall_frame = frame.copy()
            # The shared historical aggregator uses a generic training-seed
            # column. Keep this compatibility alias local so the WP4D ledger
            # continues to distinguish checkpoint_seed from source_seed.
            overall_frame["seed"] = checkpoint_seed
            overall_frame["generalization_group"] = "Overall"
            metric = aggregate(overall_frame).iloc[0].to_dict()
            run_rows.append(
                {
                    "checkpoint_seed": checkpoint_seed,
                    "source_seed": int(source_seed),
                    "valid_rate_mean": metric["valid_rate_mean"],
                    "N_keep_mean": metric["N_keep_mean"],
                    "best_error_median": metric["best_error_median"],
                    "topk_error_median": metric["topk_error_median"],
                    "hit_rate_mean": metric["hit_rate_mean"],
                    "target_success_rate": metric["target_success_rate"],
                    "0d_success_rate": metric["0d_success_rate"],
                }
            )
            gate = json.loads((directory / "functional_gate.json").read_text())
            environment = json.loads((directory / "environment.json").read_text())
            manifest = yaml.safe_load((directory / "wp4_manifest.yaml").read_text())
            gate_rows.append(
                {
                    "checkpoint_seed": checkpoint_seed,
                    "source_seed": int(source_seed),
                    "functional_gate_passed": bool(gate["passed"]),
                    "device": manifest["device"],
                    "cuda_visible_devices": environment.get("cuda_visible_devices"),
                    "target_count": len(frame),
                }
            )

    evaluations = pd.concat(frames, ignore_index=True)
    if len(evaluations) != 480 or evaluations.target_id.nunique() != 30:
        raise RuntimeError("S2-WP4D requires 2 checkpoints x 8 sources x 30 targets")
    evaluations["log10_best_error"] = np.log10(
        evaluations.best_total_error.clip(lower=np.finfo(float).tiny)
    )
    evaluations.to_csv(output / "s2_wp4d_ledger.csv", index=False)
    with (output / "s2_wp4d_ledger.jsonl").open("w", encoding="utf-8") as stream:
        for record in evaluations.replace({np.nan: None}).to_dict(orient="records"):
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    run_summary = pd.DataFrame(run_rows).sort_values(
        ["checkpoint_seed", "source_seed"]
    )
    run_summary.to_csv(output / "run_summary.csv", index=False)
    gates = pd.DataFrame(gate_rows)
    gates.to_csv(output / "functional_gates.csv", index=False)
    if not gates.functional_gate_passed.all():
        raise RuntimeError("At least one S2-WP4D functional gate failed")
    if set(gates.device) != {"cuda"} or set(gates.cuda_visible_devices.astype(str)) != {"6"}:
        raise RuntimeError("S2-WP4D was not executed exclusively on physical GPU 6")

    # Verify the first frozen source seed is exactly the previously frozen 32-step run.
    reproduction_rows = []
    first_source = int(matrix["source_seeds"][0])
    for checkpoint_seed in (42, 43):
        current = evaluations.loc[
            (evaluations.checkpoint_seed == checkpoint_seed)
            & (evaluations.source_seed == first_source)
        ].drop(columns=["checkpoint_seed", "source_seed", "log10_best_error"])
        reference = pd.read_csv(REPO_ROOT / REFERENCE_DIRS[checkpoint_seed] / "target_summary.csv")
        stable = [column for column in reference.columns if column not in VOLATILE_COLUMNS]
        reproduction_rows.append(
            {
                "checkpoint_seed": checkpoint_seed,
                "source_seed": first_source,
                "stable_columns_equal": reference[stable].reset_index(drop=True).equals(
                    current[stable].reset_index(drop=True)
                ),
                "excluded_timing_columns": ",".join(sorted(VOLATILE_COLUMNS)),
            }
        )
    reproduction = pd.DataFrame(reproduction_rows)
    reproduction.to_csv(output / "wp4c_reproduction.csv", index=False)
    if not reproduction.stable_columns_equal.all():
        raise RuntimeError("Frozen source seed did not reproduce S2-WP4C")

    # Per-checkpoint run-to-run spread and target-level source sensitivity.
    checkpoint_rows = []
    for checkpoint_seed, values in run_summary.groupby("checkpoint_seed"):
        checkpoint_rows.append(
            {
                "checkpoint_seed": checkpoint_seed,
                "runs": len(values),
                "best_error_run_median": values.best_error_median.median(),
                "best_error_run_min": values.best_error_median.min(),
                "best_error_run_max": values.best_error_median.max(),
                "best_error_run_iqr": values.best_error_median.quantile(0.75)
                - values.best_error_median.quantile(0.25),
                "valid_rate_run_mean": values.valid_rate_mean.mean(),
                "valid_rate_run_std": values.valid_rate_mean.std(ddof=0),
                "hit_rate_run_mean": values.hit_rate_mean.mean(),
                "hit_rate_run_std": values.hit_rate_mean.std(ddof=0),
                "success_rate_run_min": values.target_success_rate.min(),
                "success_rate_run_max": values.target_success_rate.max(),
            }
        )
    checkpoint_summary = pd.DataFrame(checkpoint_rows)
    checkpoint_summary.to_csv(output / "checkpoint_source_spread.csv", index=False)

    target_rows = []
    for (checkpoint_seed, target_id), values in evaluations.groupby(
        ["checkpoint_seed", "target_id"], sort=False
    ):
        target_rows.append(
            {
                "checkpoint_seed": checkpoint_seed,
                "target_id": target_id,
                "target_type": values.target_type.iloc[0],
                "best_error_p10": values.best_total_error.quantile(0.10),
                "best_error_median": values.best_total_error.median(),
                "best_error_p90": values.best_total_error.quantile(0.90),
                "log10_error_std": values.log10_best_error.std(ddof=0),
                "valid_rate_std": values.valid_rate.std(ddof=0),
                "hit_rate_std": values.hit_rate.std(ddof=0),
                "success_rate": values.best_within_tolerance.mean(),
                "success_consistent": values.best_within_tolerance.nunique() == 1,
            }
        )
    target_sensitivity = pd.DataFrame(target_rows)
    target_sensitivity.to_csv(output / "target_source_sensitivity.csv", index=False)

    group_rows = []
    for target_type, values in evaluations.groupby("target_type", sort=False):
        effect = descriptive_effect_partition(values)
        group_rows.append(
            {
                "target_type": target_type,
                "target_count": values.target_id.nunique(),
                "best_error_median": values.best_total_error.median(),
                "valid_rate_mean": values.valid_rate.mean(),
                "hit_rate_mean": values.hit_rate.mean(),
                "success_rate": values.best_within_tolerance.mean(),
                **effect,
            }
        )
    overall_effect = descriptive_effect_partition(evaluations)
    group_rows.append(
        {
            "target_type": "Overall",
            "target_count": evaluations.target_id.nunique(),
            "best_error_median": evaluations.best_total_error.median(),
            "valid_rate_mean": evaluations.valid_rate.mean(),
            "hit_rate_mean": evaluations.hit_rate.mean(),
            "success_rate": evaluations.best_within_tolerance.mean(),
            **overall_effect,
        }
    )
    group_summary = pd.DataFrame(group_rows)
    group_summary.to_csv(output / "group_effect_partition.csv", index=False)

    paired = evaluations.pivot_table(
        index=["source_seed", "target_id", "target_type"],
        columns="checkpoint_seed",
        values=["best_total_error", "valid_rate", "hit_rate"],
    )
    paired.columns = [f"{metric}_checkpoint{seed}" for metric, seed in paired.columns]
    paired = paired.reset_index()
    paired["checkpoint43_best_error_improved"] = (
        paired.best_total_error_checkpoint43 < paired.best_total_error_checkpoint42
    )
    paired["checkpoint43_log10_error_delta"] = np.log10(
        paired.best_total_error_checkpoint43
    ) - np.log10(paired.best_total_error_checkpoint42)
    paired.to_csv(output / "paired_checkpoints.csv", index=False)

    ordinary = evaluations.target_type.isin(["ID", "Interpolation"])
    consistent = target_sensitivity.success_consistent.mean()
    unstable_targets = target_sensitivity.loc[~target_sensitivity.success_consistent]
    worst = target_sensitivity.sort_values("log10_error_std", ascending=False).head(5)
    source_cv = run_summary.groupby("checkpoint_seed").best_error_median.agg(
        lambda values: values.std(ddof=0) / values.mean()
    )
    overall = group_summary.loc[group_summary.target_type == "Overall"].iloc[0]
    branch = git_value("branch", "--show-current")
    source_commit = git_value("rev-parse", "HEAD")

    report = f"""# S2-WP4D：CFM随机源起点敏感性实验报告

## 实验状态

- Status：**PASS**（16/16功能Gate通过，480个目标运行完成）。
- Branch：`{branch}`；实验源 commit：`{source_commit}`。
- 模型：GPU Sinkhorn CFM-4，训练checkpoint seed 42/43。
- 随机源：8个冻结base seed；每个base seed与`model_name + target_id`哈希，得到各目标的标准高斯随机起点。
- 固定项：同一checkpoint权重、30目标、`N_raw=128`、V2.1过滤/裁剪、RK4 32步、0D评价和目标误差定义。
- 同一个source seed与target在两个checkpoint间使用相同初始噪声，因此checkpoint比较是配对比较。
- 全部运行使用物理GPU 6；首个source seed逐列复现S2-WP4C 32步基线（仅排除墙钟计时字段）。

## 按训练checkpoint的随机源波动

| Checkpoint | 8次运行最佳误差中位数的中位数 | 最小—最大 | CV | 合法率均值±std | 命中率均值±std | 目标成功率范围 |
|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in checkpoint_summary.itertuples(index=False):
        report += (
            f"| {int(row.checkpoint_seed)} | {row.best_error_run_median:.6f} | "
            f"{row.best_error_run_min:.6f}—{row.best_error_run_max:.6f} | "
            f"{source_cv.loc[row.checkpoint_seed]:.2%} | "
            f"{row.valid_rate_run_mean:.2%}±{row.valid_rate_run_std:.2%} | "
            f"{row.hit_rate_run_mean:.2%}±{row.hit_rate_run_std:.2%} | "
            f"{row.success_rate_run_min:.2%}—{row.success_rate_run_max:.2%} |\n"
        )

    report += f"""

## 训练checkpoint与随机源的描述性效应分解

对每个目标的 `log10(best error)` 先去除目标均值，再对完整 `2 checkpoint × 8 source seed` 交叉矩阵分解平方和。它用于描述本实验矩阵，不是总体概率分布上的显著性检验。

| 分组 | Checkpoint效应 | Source seed效应 | 交互效应 | 最佳误差中位数 | 成功率 |
|---|---:|---:|---:|---:|---:|
"""
    for row in group_summary.itertuples(index=False):
        report += (
            f"| {row.target_type} | {row.checkpoint_fraction:.2%} | "
            f"{row.source_fraction:.2%} | "
            f"{row.checkpoint_source_interaction_fraction:.2%} | "
            f"{row.best_error_median:.6f} | {row.success_rate:.2%} |\n"
        )

    report += f"""

## 稳健性诊断

- 60个“checkpoint–target”组合中，成功/失败状态对8个随机源完全一致的比例为 **{consistent:.2%}**；状态随随机源变化的组合数为 **{len(unstable_targets)}**。
- 480次运行总体目标成功率为 **{evaluations.best_within_tolerance.mean():.2%}**；ID+Interpolation成功率为 **{evaluations.loc[ordinary, 'best_within_tolerance'].mean():.2%}**。
- 配对的240个“source–target”比较中，checkpoint 43最佳误差低于checkpoint 42的比例为 **{paired.checkpoint43_best_error_improved.mean():.2%}**。
- 去除目标本身难度后，`log10(best error)`波动中：训练checkpoint占 **{overall.checkpoint_fraction:.2%}**，随机源主效应占 **{overall.source_fraction:.2%}**，二者交互占 **{overall.checkpoint_source_interaction_fraction:.2%}**。

随机源最敏感的5个“checkpoint–target”组合：

| Checkpoint | Target type | Target id | log10(error) std | P10—P90 error | 成功率 |
|---:|---|---|---:|---:|---:|
"""
    for row in worst.itertuples(index=False):
        report += (
            f"| {int(row.checkpoint_seed)} | {row.target_type} | "
            f"`{row.target_id[:12]}` | {row.log10_error_std:.3f} | "
            f"{row.best_error_p10:.6f}—{row.best_error_p90:.6f} | "
            f"{row.success_rate:.2%} |\n"
        )

    report += f"""

## 结论

1. 随机高斯起点会改变单次候选及单目标最佳误差，因此不能只用一次采样结果评价生成器。
2. 但在本次`N_raw=128`下，运行级合法率、命中率和总体成功率是否稳定，应以上表实测波动为准；随机起点影响不等同于模型失效。
3. 训练checkpoint与随机源存在交互：某个checkpoint在一个source seed上更好，不保证在所有source seed上都更好。
4. 后续正式生成器比较至少应保留多训练seed、多采样seed，并使用相同source seed做配对比较；禁止挑选表现最好的随机起点后汇报。
5. S2-WP4D只诊断随机源敏感性，没有修改网络、训练、OT、ODE或候选协议。

## STAGE STOP

S2-WP4D已完成；未自动进入下一实验。
"""
    report_path = output / "S2-WP4D_CFM随机源起点敏感性实验报告.md"
    report_path.write_text(report, encoding="utf-8")
    manifest = {
        "experiment_id": matrix["experiment_id"],
        "status": "PASS",
        "branch": branch,
        "source_commit": source_commit,
        "checkpoint_seeds": [42, 43],
        "source_seeds": [int(seed) for seed in matrix["source_seeds"]],
        "targets": 30,
        "runs": 16,
        "target_evaluations": 480,
        "raw_candidates_per_target": 128,
        "ode_method": "rk4",
        "ode_steps": 32,
        "physical_gpu": 6,
        "next_experiment_started": False,
        "matrix_config_sha256": sha256_file(matrix_path),
        "ledger_sha256": sha256_file(output / "s2_wp4d_ledger.csv"),
        "report": report_path.name,
        "report_sha256": sha256_file(report_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
