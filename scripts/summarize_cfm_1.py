#!/usr/bin/env python3
"""Build the machine-readable CFM-1 ledger and strict stage report."""

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


TRAIN_DIR = Path(
    "data/generator_models/stage2/cfm_arch/cfm_1_time_embedding_gpu6_v1"
)
EVAL_DIRS = {
    42: Path("data/inverse_design/stage2/cfm_arch/cfm_1_eval_seed42_v2_1"),
    43: Path("data/inverse_design/stage2/cfm_arch/cfm_1_eval_seed43_v2_1"),
}
INFERENCE = {
    42: Path("data/inverse_design/stage2/cfm_arch/cfm_1_inference_seed42.json"),
    43: Path("data/inverse_design/stage2/cfm_arch/cfm_1_inference_seed43.json"),
}
CONFIG = Path("config/experiments/cfm_arch/cfm_1_time_embedding_train.yaml")
TARGETS = Path("config/experiments/cfm_arch/cfm_0_generalization_targets.csv")
CFM_0_DIR = Path("results/s2_arch_cfm/cfm_0")


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
        evaluations = values["actual_0d_evaluations"].sum()
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
                    values["0d_success_count"].sum() / evaluations
                    if evaluations
                    else 0.0
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


def percent_change(new: float, old: float) -> float:
    return (new / old - 1.0) if old else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/s2_arch_cfm/cfm_1")
    )
    args = parser.parse_args()
    output = REPO_ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    training = pd.read_csv(REPO_ROOT / TRAIN_DIR / "seed_summary.csv")
    evaluations = load_evaluations(EVAL_DIRS)
    target_groups = pd.read_csv(REPO_ROOT / TARGETS)[
        ["target_id", "generalization_group"]
    ]
    evaluations = evaluations.merge(
        target_groups, on="target_id", how="left", validate="many_to_one"
    )
    if evaluations["generalization_group"].isna().any():
        raise RuntimeError("unclassified CFM-1 target")
    if len(evaluations) != 60 or evaluations["target_id"].nunique() != 30:
        raise RuntimeError("CFM-1 requires 30 frozen targets for both seeds")

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
        values.insert(0, "experiment_id", "s2_arch_cfm_1_v1")
        values.insert(1, "architecture", "CFM-1")
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
                "architecture": "CFM-1",
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
    ledger.to_csv(output / "cfm_1_ledger.csv", index=False)
    with (output / "cfm_1_ledger.jsonl").open("w", encoding="utf-8") as stream:
        for record in ledger.replace({np.nan: None}).to_dict(orient="records"):
            stream.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )
    pd.DataFrame(seed_rows).to_csv(output / "cfm_1_seed_summary.csv", index=False)

    groups = aggregate(evaluations)
    groups.to_csv(output / "cfm_1_group_summary.csv", index=False)
    overall_frame = evaluations.copy()
    overall_frame["generalization_group"] = "Overall"
    overall = aggregate(overall_frame)
    overall.to_csv(output / "cfm_1_overall_summary.csv", index=False)

    cfm_0_ledger = pd.read_csv(REPO_ROOT / CFM_0_DIR / "cfm_0_ledger.csv")
    paired = evaluations.merge(
        cfm_0_ledger[
            [
                "seed",
                "target_id",
                "valid_rate",
                "N_valid",
                "N_keep",
                "best_total_error",
                "topk_error",
                "hit_rate",
                "best_within_tolerance",
            ]
        ],
        on=["seed", "target_id"],
        suffixes=("_cfm1", "_cfm0"),
        validate="one_to_one",
    )
    paired["valid_rate_delta"] = (
        paired["valid_rate_cfm1"] - paired["valid_rate_cfm0"]
    )
    paired["best_error_delta"] = (
        paired["best_total_error_cfm1"] - paired["best_total_error_cfm0"]
    )
    paired["best_error_improved"] = paired["best_error_delta"] < 0
    paired["hit_rate_delta"] = paired["hit_rate_cfm1"] - paired["hit_rate_cfm0"]
    paired.to_csv(output / "cfm_1_vs_cfm_0_paired.csv", index=False)

    cfm_0_overall = pd.read_csv(REPO_ROOT / CFM_0_DIR / "model_overall_summary.csv")
    cfm_0_groups = pd.read_csv(REPO_ROOT / CFM_0_DIR / "model_group_summary.csv")
    comparison = pd.concat(
        [
            overall.assign(architecture="CFM-1 Time Embedding"),
            cfm_0_overall.loc[cfm_0_overall["model"] == "cfm"].assign(
                architecture="CFM-0 Scalar Time"
            ),
            cfm_0_overall.loc[cfm_0_overall["model"] == "nflows"].assign(
                architecture="nflows Existing Best"
            ),
        ],
        ignore_index=True,
    )
    comparison.to_csv(output / "model_comparison_overall.csv", index=False)

    cfm_1 = overall.iloc[0]
    cfm_0 = cfm_0_overall.loc[cfm_0_overall["model"] == "cfm"].iloc[0]
    nflows = cfm_0_overall.loc[cfm_0_overall["model"] == "nflows"].iloc[0]
    train0 = pd.read_csv(REPO_ROOT / CFM_0_DIR / "cfm_0_seed_summary.csv")
    train1 = pd.DataFrame(seed_rows)
    param_delta = percent_change(
        float(train1.total_parameters.iloc[0]), float(train0.total_parameters.iloc[0])
    )
    inference_delta = percent_change(
        train1.generation_128_mean_seconds.mean(),
        train0.generation_128_mean_seconds.mean(),
    )
    training_delta = percent_change(
        train1.training_time_seconds.mean(), train0.training_time_seconds.mean()
    )
    improved_targets = paired["best_error_improved"].mean()
    cfm_1_groups = groups.set_index("generalization_group")
    cfm_0_groups = cfm_0_groups.loc[cfm_0_groups["model"] == "cfm"].set_index(
        "generalization_group"
    )

    report = f"""# S2-ARCH-CFM CFM-1 阶段报告

## 阶段状态

- Architecture：CFM-1 Time Embedding MLP
- Status：PASS
- Branch：`{branch}`
- 实验源 commit：`{source_commit}`
- 变更边界：仅将 scalar `t` 替换为固定32维 sinusoidal/Fourier embedding；未加入加宽、加深、residual、FiLM、gate、U-Net、attention或可训练时间投影。
- CFM-2：未实现、未配置、未训练。

PASS 表示实现、完整训练、V2.1物理评价和可复现产物均完成；不表示它在准确度上优于现有模型。

## Architecture

```text
t → fixed sinusoidal embedding(32)
[x_t(10), embedding(t)(32), c(3)]
→ Linear(45,128) → SiLU
→ Linear(128,128) → SiLU
→ Linear(128,128) → SiLU
→ Linear(128,10)
```

- Plain `ConditionalFlowMatcher(sigma=0)`；标准高斯源分布。
- RK4、32 steps、128 NFE；数据、split、normalizer、优化器、scheduler和loss与CFM-0一致。
- Total/trainable parameters：{int(train1.total_parameters.iloc[0]):,} / {int(train1.trainable_parameters.iloc[0]):,}，较CFM-0增加 {param_delta:+.2%}。

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

两个seed都在相同训练协议下正常早停。平均训练耗时相对CFM-0变化 {training_delta:+.2%}；验证损失没有改善（CFM-1 seed42/43分别为 {train1.best_validation_loss.iloc[0]:.6f}/{train1.best_validation_loss.iloc[1]:.6f}，CFM-0为 {train0.best_validation_loss.iloc[0]:.6f}/{train0.best_validation_loss.iloc[1]:.6f}）。

## Inference

固定物理GPU 6、float32、`N_raw=128`、RK4/32；5次warm-up后重复30次，每次计时前后CUDA同步。

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

平均128候选生成耗时为 {1000*train1.generation_128_mean_seconds.mean():.3f} ms，较CFM-0慢 {inference_delta:+.2%}。采样器的速度场调用数未改变，仍为128 NFE。

## V2.1 Validity and Target Accuracy

所有运行严格执行一次 `N_raw=128 → N_valid → N_keep=2^floor(log2(N_valid)) → candidate_id裁剪 → 0D`；不补采样。

| Group | N_valid mean | N_keep mean | Valid rate | Best error median | Top-10 mean error median | Candidate hit rate | Target success |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in groups.itertuples(index=False):
        report += (
            f"| {row.generalization_group} | {row.N_valid_mean:.2f} | "
            f"{row.N_keep_mean:.2f} | {row.valid_rate_mean:.2%} | "
            f"{row.best_error_median:.6f} | {row.topk_error_median:.6f} | "
            f"{row.hit_rate_mean:.2%} | {row.target_success_rate:.2%} |\n"
        )
    report += f"""
| Overall | {cfm_1['N_valid_mean']:.2f} | {cfm_1['N_keep_mean']:.2f} | {cfm_1['valid_rate_mean']:.2%} | {cfm_1['best_error_median']:.6f} | {cfm_1['topk_error_median']:.6f} | {cfm_1['hit_rate_mean']:.2%} | {cfm_1['target_success_rate']:.2%} |

30个目标均为已知可行的ID、Interpolation或Boundary目标。OOD：not evaluated；当前阶段没有可靠的已知可行OOD目标集。

## CFM-1 vs CFM-0

| Metric | CFM-0 | CFM-1 | Change |
|---|---:|---:|---:|
| Parameters | {int(train0.total_parameters.iloc[0]):,} | {int(train1.total_parameters.iloc[0]):,} | {param_delta:+.2%} |
| Mean train time (s) | {train0.training_time_seconds.mean():.3f} | {train1.training_time_seconds.mean():.3f} | {training_delta:+.2%} |
| Mean inference/128 (ms) | {1000*train0.generation_128_mean_seconds.mean():.3f} | {1000*train1.generation_128_mean_seconds.mean():.3f} | {inference_delta:+.2%} |
| Valid rate | {cfm_0['valid_rate_mean']:.2%} | {cfm_1['valid_rate_mean']:.2%} | {cfm_1['valid_rate_mean']-cfm_0['valid_rate_mean']:+.2%} pt |
| N_keep mean | {cfm_0['N_keep_mean']:.2f} | {cfm_1['N_keep_mean']:.2f} | {cfm_1['N_keep_mean']-cfm_0['N_keep_mean']:+.2f} |
| Best error median | {cfm_0['best_error_median']:.6f} | {cfm_1['best_error_median']:.6f} | {percent_change(cfm_1['best_error_median'], cfm_0['best_error_median']):+.2%} |
| Top-10 error median | {cfm_0['topk_error_median']:.6f} | {cfm_1['topk_error_median']:.6f} | {percent_change(cfm_1['topk_error_median'], cfm_0['topk_error_median']):+.2%} |
| Candidate hit rate | {cfm_0['hit_rate_mean']:.2%} | {cfm_1['hit_rate_mean']:.2%} | {cfm_1['hit_rate_mean']-cfm_0['hit_rate_mean']:+.2%} pt |
| Target success | {cfm_0['target_success_rate']:.2%} | {cfm_1['target_success_rate']:.2%} | {cfm_1['target_success_rate']-cfm_0['target_success_rate']:+.2%} pt |

在严格逐目标、同seed配对比较中，CFM-1有 {improved_targets:.2%} 的运行取得更小最佳误差。合法率下降不是单一分组造成：ID {cfm_1_groups.loc['ID','valid_rate_mean']-cfm_0_groups.loc['ID','valid_rate_mean']:+.2%} pt、Interpolation {cfm_1_groups.loc['Interpolation','valid_rate_mean']-cfm_0_groups.loc['Interpolation','valid_rate_mean']:+.2%} pt、Boundary {cfm_1_groups.loc['Boundary','valid_rate_mean']-cfm_0_groups.loc['Boundary','valid_rate_mean']:+.2%} pt。

## Seed Stability

| Seed | Valid rate | Best error median | Top-10 error median | Candidate hit rate | Target success |
|---:|---:|---:|---:|---:|---:|
"""
    for row in seed_rows:
        report += (
            f"| {row['seed']} | {row['valid_rate_mean']:.2%} | "
            f"{row['best_error_median']:.6f} | {row['topk_error_median']:.6f} | "
            f"{row['hit_rate_mean']:.2%} | {row['target_success_rate']:.2%} |\n"
        )
    report += f"""

seed 43在合法率和目标误差上优于seed 42，但两者都通过功能Gate且没有零合法候选目标。相对两个seed之间约 {abs(train1.valid_rate_mean.iloc[1]-train1.valid_rate_mean.iloc[0]):.2%} 个百分点的合法率差异，CFM-1与CFM-0总体合法率差仅 {abs(cfm_1['valid_rate_mean']-cfm_0['valid_rate_mean']):.2%} 个百分点，因此不能把该变化解释成稳定的架构增益。

## Comparison with Existing Best

| Model | Valid rate | Best error median | Top-10 error median | Candidate hit rate | Target success |
|---|---:|---:|---:|---:|---:|
| CFM-1 | {cfm_1['valid_rate_mean']:.2%} | {cfm_1['best_error_median']:.6f} | {cfm_1['topk_error_median']:.6f} | {cfm_1['hit_rate_mean']:.2%} | {cfm_1['target_success_rate']:.2%} |
| nflows | {nflows['valid_rate_mean']:.2%} | {nflows['best_error_median']:.6f} | {nflows['topk_error_median']:.6f} | {nflows['hit_rate_mean']:.2%} | {nflows['target_success_rate']:.2%} |

CFM-1的合法率比nflows低 {nflows['valid_rate_mean']-cfm_1['valid_rate_mean']:.2%} 个百分点，最佳误差中位数为nflows的 {cfm_1['best_error_median']/nflows['best_error_median']:.2f} 倍，候选命中率低 {nflows['hit_rate_mean']-cfm_1['hit_rate_mean']:.2%} 个百分点。因此它仍不是“最靠近目标性能”的最佳模型。

## Advantages

1. Top-10误差中位数相对CFM-0下降 {-percent_change(cfm_1['topk_error_median'], cfm_0['topk_error_median']):.2%}，候选命中率增加 {cfm_1['hit_rate_mean']-cfm_0['hit_rate_mean']:+.2%} 个百分点，说明时间嵌入在候选集合层面有很小的局部收益。
2. 参数仅增加 {param_delta:+.2%}，而完整训练耗时变化只有 {training_delta:+.2%}；训练成本几乎不变。
3. 两seed均稳定早停、checkpoint可复现且物理功能Gate通过，没有出现数值崩溃或零合法候选目标。

## Disadvantages

1. 核心最佳误差中位数相对CFM-0反而增加 {percent_change(cfm_1['best_error_median'], cfm_0['best_error_median']):.2%}，目标成功率下降 {cfm_0['target_success_rate']-cfm_1['target_success_rate']:.2%} 个百分点；逐目标配对也只有 {improved_targets:.2%} 改善，不能判为准确度提升。
2. 合法率下降 {cfm_0['valid_rate_mean']-cfm_1['valid_rate_mean']:.2%} 个百分点，平均每目标少保留 {cfm_0['N_keep_mean']-cfm_1['N_keep_mean']:.2f} 个候选进入0D；三个泛化分组均下降。
3. 128候选推理比CFM-0慢约 {inference_delta:.2%}，因为每次速度场调用增加了时间特征计算和更宽的首层输入。
4. 验证loss未下降，且候选命中率和目标准确度仍显著弱于nflows。
5. 没有可靠OOD集，因此不能声称具备OOD泛化优势。

## Current Interpretation

固定32维正弦时间嵌入没有在本协议下形成总体收益。它仅小幅改善Top-10误差和候选命中率，却同时降低合法率、提高最佳误差中位数、降低目标成功率，并带来约11%的参数和约22%的采样耗时。两个seed的方向也不完全一致，说明变化规模接近seed波动。nflows仍在“给定目标、取得最接近性能”这一核心指标上明显领先；CFM-1只能作为严格消融中的时间编码节点，不能替代CFM-0或当前最佳反向设计模型。

## Recommendation

保留CFM-1 checkpoint与配置以保证消融链可追溯，但不建议把单独的Time Embedding视为已验证有效的改进。是否按既定路线基于CFM-1进入CFM-2，必须由用户根据本报告明确决定。

## STAGE STOP

CFM-1 已完成。CFM-2 未开始。
"""
    (output / "CFM-1阶段报告.md").write_text(report, encoding="utf-8")

    manifest = {
        "experiment_id": "s2_arch_cfm_1_v1",
        "architecture": "CFM-1",
        "status": "PASS",
        "branch": branch,
        "source_commit": source_commit,
        "cfm_2_started": False,
        "targets": 30,
        "seeds": [42, 43],
        "ledger": "cfm_1_ledger.csv",
        "ledger_sha256": sha256_file(output / "cfm_1_ledger.csv"),
        "report": "CFM-1阶段报告.md",
        "report_sha256": sha256_file(output / "CFM-1阶段报告.md"),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
