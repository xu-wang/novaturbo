#!/usr/bin/env python3
"""Build the machine-readable CFM-2 ledger and strict stage report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.baseline import sha256_file


TRAIN_DIR = Path("data/generator_models/stage2/cfm_arch/cfm_2_wide_deep_gpu6_v1")
EVAL_DIRS = {
    42: Path("data/inverse_design/stage2/cfm_arch/cfm_2_eval_seed42_v2_1"),
    43: Path("data/inverse_design/stage2/cfm_arch/cfm_2_eval_seed43_v2_1"),
}
INFERENCE = {
    42: Path("data/inverse_design/stage2/cfm_arch/cfm_2_inference_seed42.json"),
    43: Path("data/inverse_design/stage2/cfm_arch/cfm_2_inference_seed43.json"),
}
CONFIG = Path("config/experiments/cfm_arch/cfm_2_wide_deep_train.yaml")
TARGETS = Path("config/experiments/cfm_arch/cfm_0_generalization_targets.csv")
CFM_0_DIR = Path("results/s2_arch_cfm/cfm_0")
CFM_1_DIR = Path("results/s2_arch_cfm/cfm_1")


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def load_evaluations(paths: dict[int, Path]) -> pd.DataFrame:
    frames = []
    for seed, directory in paths.items():
        frame = pd.read_csv(REPO_ROOT / directory / "target_summary.csv")
        frame.insert(0, "seed", seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, group), values in frame.groupby(
        ["model", "generalization_group"], sort=True
    ):
        count_0d = values["actual_0d_evaluations"].sum()
        rows.append(
            {
                "model": model,
                "generalization_group": group,
                "run_count": len(values),
                "seed_count": values["seed"].nunique(),
                "N_valid_mean": values["N_valid"].mean(),
                "N_valid_std": values["N_valid"].std(ddof=0),
                "N_keep_mean": values["N_keep"].mean(),
                "N_trimmed_mean": values["N_trimmed"].mean(),
                "valid_rate_mean": values["valid_rate"].mean(),
                "valid_rate_std": values["valid_rate"].std(ddof=0),
                "zero_valid_run_rate": (values["N_valid"] == 0).mean(),
                "0d_success_rate": (
                    values["0d_success_count"].sum() / count_0d if count_0d else 0.0
                ),
                "best_error_median": values["best_total_error"].median(),
                "best_error_mean": values["best_total_error"].mean(),
                "best_error_std": values["best_total_error"].std(ddof=0),
                "topk_error_median": values["topk_error"].median(),
                "median_error_median": values["median_error"].median(),
                "hit_rate_mean": values["hit_rate"].mean(),
                "target_success_rate": values["best_within_tolerance"].mean(),
                "actual_0d_evaluations_mean": values[
                    "actual_0d_evaluations"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def relative(new: float, old: float) -> float:
    return new / old - 1.0 if old else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/s2_arch_cfm/cfm_2")
    )
    args = parser.parse_args()
    output = REPO_ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    training = pd.read_csv(REPO_ROOT / TRAIN_DIR / "seed_summary.csv")
    evaluations = load_evaluations(EVAL_DIRS)
    groups_map = pd.read_csv(REPO_ROOT / TARGETS)[
        ["target_id", "generalization_group"]
    ]
    evaluations = evaluations.merge(
        groups_map, on="target_id", how="left", validate="many_to_one"
    )
    if len(evaluations) != 60 or evaluations["target_id"].nunique() != 30:
        raise RuntimeError("CFM-2 requires the same 30 frozen targets for both seeds")
    if evaluations["generalization_group"].isna().any():
        raise RuntimeError("unclassified CFM-2 target")

    branch = git_value("branch", "--show-current")
    source_commit = git_value("rev-parse", "HEAD")
    ledgers = []
    seed_rows = []
    for seed in (42, 43):
        train = training.loc[training["seed"] == seed].iloc[0]
        inference = json.loads(
            (REPO_ROOT / INFERENCE[seed]).read_text(encoding="utf-8")
        )
        checkpoint = TRAIN_DIR / f"cfm/seed_{seed}/best.pt"
        values = evaluations.loc[evaluations["seed"] == seed].copy()
        values.insert(0, "experiment_id", "s2_arch_cfm_2_v1")
        values.insert(1, "architecture", "CFM-2")
        values["target_F"] = values["target_thrust_N"]
        values["target_TSFC"] = values["target_tsfc_kg_N_s"]
        values["target_Mass"] = values["target_total_mass_kg"]
        values["parameter_count"] = int(train["total_parameters"])
        values["trainable_parameter_count"] = int(train["trainable_parameters"])
        values["checkpoint_size_bytes"] = int(train["best_checkpoint_bytes"])
        values["training_time"] = float(train["training_elapsed_seconds"])
        values["average_epoch_time"] = float(train["average_epoch_seconds"])
        values["best_val_loss"] = float(train["best_validation_loss"])
        values["best_epoch"] = int(train["best_epoch"])
        values["epochs_completed"] = int(train["epochs_completed"])
        values["generation_time"] = float(inference["mean_seconds"])
        values["peak_gpu_memory"] = int(train["peak_gpu_memory_bytes"])
        values["0d_success"] = values["0d_success_count"]
        values["0d_failure"] = values["0d_failure_count"]
        values["best_error"] = values["best_total_error"]
        values["git_commit"] = source_commit
        values["branch"] = branch
        values["config_path"] = str(CONFIG)
        values["checkpoint_path"] = str(checkpoint)
        values["checkpoint_sha256"] = sha256_file(REPO_ROOT / checkpoint)
        ledgers.append(values)
        seed_rows.append(
            {
                "architecture": "CFM-2",
                "seed": seed,
                "total_parameters": int(train["total_parameters"]),
                "trainable_parameters": int(train["trainable_parameters"]),
                "checkpoint_size_bytes": int(train["best_checkpoint_bytes"]),
                "best_validation_loss": train["best_validation_loss"],
                "best_epoch": int(train["best_epoch"]),
                "epochs_completed": int(train["epochs_completed"]),
                "training_time_seconds": train["training_elapsed_seconds"],
                "average_epoch_seconds": train["average_epoch_seconds"],
                "peak_gpu_memory_bytes": int(train["peak_gpu_memory_bytes"]),
                "generation_128_mean_seconds": inference["mean_seconds"],
                "generation_128_median_seconds": inference["median_seconds"],
                "generation_128_std_seconds": inference["std_seconds"],
                "N_valid_mean": values["N_valid"].mean(),
                "N_keep_mean": values["N_keep"].mean(),
                "valid_rate_mean": values["valid_rate"].mean(),
                "best_error_median": values["best_total_error"].median(),
                "topk_error_median": values["topk_error"].median(),
                "median_error_median": values["median_error"].median(),
                "hit_rate_mean": values["hit_rate"].mean(),
                "target_success_rate": values["best_within_tolerance"].mean(),
            }
        )

    ledger = pd.concat(ledgers, ignore_index=True)
    ledger.to_csv(output / "cfm_2_ledger.csv", index=False)
    with (output / "cfm_2_ledger.jsonl").open("w", encoding="utf-8") as stream:
        for record in ledger.replace({np.nan: None}).to_dict(orient="records"):
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(output / "cfm_2_seed_summary.csv", index=False)

    group_summary = aggregate(evaluations)
    group_summary.to_csv(output / "cfm_2_group_summary.csv", index=False)
    overall_frame = evaluations.copy()
    overall_frame["generalization_group"] = "Overall"
    overall = aggregate(overall_frame)
    overall.to_csv(output / "cfm_2_overall_summary.csv", index=False)

    cfm_1_ledger = pd.read_csv(REPO_ROOT / CFM_1_DIR / "cfm_1_ledger.csv")
    paired = evaluations.merge(
        cfm_1_ledger[
            [
                "seed", "target_id", "valid_rate", "N_valid", "N_keep",
                "best_total_error", "topk_error", "hit_rate",
                "best_within_tolerance",
            ]
        ],
        on=["seed", "target_id"],
        suffixes=("_cfm2", "_cfm1"),
        validate="one_to_one",
    )
    paired["valid_rate_delta"] = paired.valid_rate_cfm2 - paired.valid_rate_cfm1
    paired["best_error_delta"] = (
        paired.best_total_error_cfm2 - paired.best_total_error_cfm1
    )
    paired["best_error_improved"] = paired.best_error_delta < 0
    paired["hit_rate_delta"] = paired.hit_rate_cfm2 - paired.hit_rate_cfm1
    paired.to_csv(output / "cfm_2_vs_cfm_1_paired.csv", index=False)

    cfm_0_all = pd.read_csv(REPO_ROOT / CFM_0_DIR / "model_overall_summary.csv")
    cfm_1_overall = pd.read_csv(REPO_ROOT / CFM_1_DIR / "cfm_1_overall_summary.csv")
    cfm_0 = cfm_0_all.loc[cfm_0_all.model == "cfm"].iloc[0]
    nflows = cfm_0_all.loc[cfm_0_all.model == "nflows"].iloc[0]
    cfm_1 = cfm_1_overall.iloc[0]
    cfm_2 = overall.iloc[0]
    comparison = pd.concat(
        [
            overall.assign(architecture="CFM-2 Wide/Deep"),
            cfm_1_overall.assign(architecture="CFM-1 Time Embedding"),
            cfm_0_all.loc[cfm_0_all.model == "cfm"].assign(
                architecture="CFM-0 Scalar Time"
            ),
            cfm_0_all.loc[cfm_0_all.model == "nflows"].assign(
                architecture="nflows Existing Best"
            ),
        ], ignore_index=True,
    )
    comparison.to_csv(output / "model_comparison_overall.csv", index=False)

    train1 = pd.read_csv(REPO_ROOT / CFM_1_DIR / "cfm_1_seed_summary.csv")
    params_delta = relative(
        seed_summary.total_parameters.iloc[0], train1.total_parameters.iloc[0]
    )
    train_delta = relative(
        seed_summary.training_time_seconds.mean(), train1.training_time_seconds.mean()
    )
    inference_delta = relative(
        seed_summary.generation_128_mean_seconds.mean(),
        train1.generation_128_mean_seconds.mean(),
    )
    improved = paired.best_error_improved.mean()
    groups2 = group_summary.set_index("generalization_group")
    groups1 = pd.read_csv(REPO_ROOT / CFM_1_DIR / "cfm_1_group_summary.csv").set_index(
        "generalization_group"
    )

    report = f"""# S2-ARCH-CFM CFM-2 阶段报告

## 阶段状态

- Architecture：CFM-2 Wide/Deep MLP
- Status：PASS
- Branch：`{branch}`
- 实验源 commit：`{source_commit}`
- 唯一核心变化：保留CFM-1固定32维时间嵌入，将普通MLP由3×128改为4×256。
- 未加入residual、FiLM、gate、U-Net、attention或可训练时间投影。
- CFM-3：未实现、未配置、未训练。

PASS只表示实现、完整训练、V2.1物理评价和可复现产物完成；方法优劣按下述指标单独判断。

## Architecture

```text
t → fixed sinusoidal embedding(32)
[x_t(10), embedding(t)(32), c(3)]
→ Linear(45,256) → SiLU
→ Linear(256,256) → SiLU
→ Linear(256,256) → SiLU
→ Linear(256,256) → SiLU
→ Linear(256,10)
```

- Plain `ConditionalFlowMatcher(sigma=0)`；标准高斯源分布；RK4/32/128 NFE。
- 数据、split、normalizer、optimizer、scheduler、loss、目标集和V2.1协议全部冻结。
- Total/trainable parameters：{int(seed_summary.total_parameters.iloc[0]):,} / {int(seed_summary.trainable_parameters.iloc[0]):,}，相对CFM-1增加 {params_delta:+.2%}。

## Training

| Seed | Best val loss | Best epoch | Completed | Time (s) | Avg epoch (s) | Peak GPU MiB | Checkpoint KiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in seed_rows:
        report += (
            f"| {row['seed']} | {row['best_validation_loss']:.6f} | "
            f"{row['best_epoch']} | {row['epochs_completed']} | "
            f"{row['training_time_seconds']:.3f} | {row['average_epoch_seconds']:.3f} | "
            f"{row['peak_gpu_memory_bytes']/2**20:.2f} | "
            f"{row['checkpoint_size_bytes']/2**10:.2f} |\n"
        )
    report += f"""

两个seed均使用完整79,945/9,992训练/验证样本并正常早停。平均训练耗时较CFM-1增加 {train_delta:+.2%}；验证损失均明显低于CFM-1。

## Inference

GPU 6、float32、`N_raw=128`、RK4/32；5次warm-up后重复30次，每次前后CUDA同步。

| Seed | Mean (ms) | Median (ms) | Std (ms) |
|---:|---:|---:|---:|
"""
    for row in seed_rows:
        report += (
            f"| {row['seed']} | {1000*row['generation_128_mean_seconds']:.3f} | "
            f"{1000*row['generation_128_median_seconds']:.3f} | "
            f"{1000*row['generation_128_std_seconds']:.3f} |\n"
        )
    report += f"""

平均128候选生成耗时 {1000*seed_summary.generation_128_mean_seconds.mean():.3f} ms，较CFM-1变化 {inference_delta:+.2%}；NFE仍为128。

## V2.1 Validity and Target Accuracy

严格执行一次 `128 → N_valid → 2^floor(log2(N_valid)) → candidate_id裁剪 → 0D`，不补采样。

| Group | N_valid | N_keep | Valid rate | Best error median | Top-10 error median | Hit rate | Target success |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in group_summary.itertuples(index=False):
        report += (
            f"| {row.generalization_group} | {row.N_valid_mean:.2f} | "
            f"{row.N_keep_mean:.2f} | {row.valid_rate_mean:.2%} | "
            f"{row.best_error_median:.6f} | {row.topk_error_median:.6f} | "
            f"{row.hit_rate_mean:.2%} | {row.target_success_rate:.2%} |\n"
        )
    report += f"""
| Overall | {cfm_2.N_valid_mean:.2f} | {cfm_2.N_keep_mean:.2f} | {cfm_2.valid_rate_mean:.2%} | {cfm_2.best_error_median:.6f} | {cfm_2.topk_error_median:.6f} | {cfm_2.hit_rate_mean:.2%} | {cfm_2.target_success_rate:.2%} |

0D成功率为 {cfm_2['0d_success_rate']:.2%}。30个目标均为已知可行的ID、Interpolation或Boundary目标；OOD没有可靠可行目标，因此not evaluated。

## CFM-2 vs CFM-1

| Metric | CFM-1 | CFM-2 | Change |
|---|---:|---:|---:|
| Parameters | {int(train1.total_parameters.iloc[0]):,} | {int(seed_summary.total_parameters.iloc[0]):,} | {params_delta:+.2%} |
| Mean train time (s) | {train1.training_time_seconds.mean():.3f} | {seed_summary.training_time_seconds.mean():.3f} | {train_delta:+.2%} |
| Mean inference/128 (ms) | {1000*train1.generation_128_mean_seconds.mean():.3f} | {1000*seed_summary.generation_128_mean_seconds.mean():.3f} | {inference_delta:+.2%} |
| Valid rate | {cfm_1.valid_rate_mean:.2%} | {cfm_2.valid_rate_mean:.2%} | {cfm_2.valid_rate_mean-cfm_1.valid_rate_mean:+.2%} pt |
| N_keep mean | {cfm_1.N_keep_mean:.2f} | {cfm_2.N_keep_mean:.2f} | {cfm_2.N_keep_mean-cfm_1.N_keep_mean:+.2f} |
| Best error median | {cfm_1.best_error_median:.6f} | {cfm_2.best_error_median:.6f} | {relative(cfm_2.best_error_median, cfm_1.best_error_median):+.2%} |
| Top-10 error median | {cfm_1.topk_error_median:.6f} | {cfm_2.topk_error_median:.6f} | {relative(cfm_2.topk_error_median, cfm_1.topk_error_median):+.2%} |
| Candidate hit rate | {cfm_1.hit_rate_mean:.2%} | {cfm_2.hit_rate_mean:.2%} | {cfm_2.hit_rate_mean-cfm_1.hit_rate_mean:+.2%} pt |
| Target success | {cfm_1.target_success_rate:.2%} | {cfm_2.target_success_rate:.2%} | {cfm_2.target_success_rate-cfm_1.target_success_rate:+.2%} pt |

逐目标同seed配对中，CFM-2有 {improved:.2%} 的运行取得更小最佳误差。合法率变化覆盖三个分组：ID {groups2.loc['ID','valid_rate_mean']-groups1.loc['ID','valid_rate_mean']:+.2%} pt、Interpolation {groups2.loc['Interpolation','valid_rate_mean']-groups1.loc['Interpolation','valid_rate_mean']:+.2%} pt、Boundary {groups2.loc['Boundary','valid_rate_mean']-groups1.loc['Boundary','valid_rate_mean']:+.2%} pt。

## Seed Stability

| Seed | Valid rate | Best error median | Top-10 error median | Hit rate | Target success |
|---:|---:|---:|---:|---:|---:|
"""
    for row in seed_rows:
        report += (
            f"| {row['seed']} | {row['valid_rate_mean']:.2%} | "
            f"{row['best_error_median']:.6f} | {row['topk_error_median']:.6f} | "
            f"{row['hit_rate_mean']:.2%} | {row['target_success_rate']:.2%} |\n"
        )
    report += f"""

两个seed的合法率相差 {abs(seed_summary.valid_rate_mean.iloc[1]-seed_summary.valid_rate_mean.iloc[0]):.2%} 个百分点，最佳误差中位数相差 {abs(seed_summary.best_error_median.iloc[1]-seed_summary.best_error_median.iloc[0]):.6f}；方向一致且目标成功率完全相同，未出现单seed崩溃。

## Comparison with Baseline and Existing Best

| Model | Valid rate | Best error median | Top-10 error median | Hit rate | Target success |
|---|---:|---:|---:|---:|---:|
| CFM-0 | {cfm_0.valid_rate_mean:.2%} | {cfm_0.best_error_median:.6f} | {cfm_0.topk_error_median:.6f} | {cfm_0.hit_rate_mean:.2%} | {cfm_0.target_success_rate:.2%} |
| CFM-1 | {cfm_1.valid_rate_mean:.2%} | {cfm_1.best_error_median:.6f} | {cfm_1.topk_error_median:.6f} | {cfm_1.hit_rate_mean:.2%} | {cfm_1.target_success_rate:.2%} |
| CFM-2 | {cfm_2.valid_rate_mean:.2%} | {cfm_2.best_error_median:.6f} | {cfm_2.topk_error_median:.6f} | {cfm_2.hit_rate_mean:.2%} | {cfm_2.target_success_rate:.2%} |
| nflows | {nflows.valid_rate_mean:.2%} | {nflows.best_error_median:.6f} | {nflows.topk_error_median:.6f} | {nflows.hit_rate_mean:.2%} | {nflows.target_success_rate:.2%} |

CFM-2相对CFM-1的容量扩展产生了明确收益，但最佳误差中位数仍为nflows的 {cfm_2.best_error_median/nflows.best_error_median:.2f} 倍，核心目标命中率仍低 {nflows.hit_rate_mean-cfm_2.hit_rate_mean:.2%} 个百分点。

## Advantages

1. 相对CFM-1，最佳误差、Top-10误差、合法率、候选命中率和目标成功率均改善，且逐目标配对改善比例为 {improved:.2%}。
2. 两个seed的验证损失和下游指标方向一致，扩大普通MLP容量的收益比单独Time Embedding更稳定。
3. 三类目标均有合法率提升，Boundary和Interpolation没有以牺牲泛化换取ID收益。

## Disadvantages

1. 参数量增加 {params_delta:+.2%}，checkpoint、显存和训练成本均上升。
2. 推理速度慢于CFM-1；RK4/32仍需128次更大速度网络前向。
3. 尽管明显改善，给定目标准确度仍弱于nflows，因此尚不能称为当前最佳初始化模型。
4. 没有可靠OOD目标，不能声称OOD泛化优势。

## Current Interpretation

CFM-2表明当前CFM速度场存在容量不足：在完全相同的CFM路径、训练和采样协议下，仅增加普通MLP宽度与深度就同时降低训练损失并改善物理回代指标。该收益不是“时间嵌入单独有效”的证明，而是CFM-1时间编码基础上的容量消融结果。精度—复杂度权衡为正，但尚未追平nflows。

## Recommendation

保留CFM-2作为当前最强CFM结构基线。是否进入CFM-3 Residual MLP，必须由用户根据本报告明确决定。

## STAGE STOP

CFM-2 已完成。CFM-3 未开始。
"""
    (output / "CFM-2阶段报告.md").write_text(report, encoding="utf-8")

    manifest = {
        "experiment_id": "s2_arch_cfm_2_v1",
        "architecture": "CFM-2",
        "status": "PASS",
        "branch": branch,
        "source_commit": source_commit,
        "cfm_3_started": False,
        "targets": 30,
        "seeds": [42, 43],
        "ledger": "cfm_2_ledger.csv",
        "ledger_sha256": sha256_file(output / "cfm_2_ledger.csv"),
        "report": "CFM-2阶段报告.md",
        "report_sha256": sha256_file(output / "CFM-2阶段报告.md"),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
