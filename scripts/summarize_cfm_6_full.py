#!/usr/bin/env python3
"""Summarize the original-width CFM-6 repeat without overwriting scaled results."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_cfm_2 import aggregate, relative
from src.ai.baseline import sha256_file


TRAIN_DIR = Path("data/generator_models/stage2/cfm_arch/cfm_6_mlp_unet_full_gpu6_v1")
EVAL_DIRS = {
    42: Path("data/inverse_design/stage2/cfm_arch/cfm_6_full_eval_seed42_v2_1"),
    43: Path("data/inverse_design/stage2/cfm_arch/cfm_6_full_eval_seed43_v2_1"),
}
INFERENCE = {
    42: Path("data/inverse_design/stage2/cfm_arch/cfm_6_full_inference_seed42.json"),
    43: Path("data/inverse_design/stage2/cfm_arch/cfm_6_full_inference_seed43.json"),
}
CONFIG = Path("config/experiments/cfm_arch/cfm_6_mlp_unet_full_train.yaml")
TARGETS = Path("config/experiments/cfm_arch/cfm_0_generalization_targets.csv")
RESULTS = Path("results/s2_arch_cfm")
OUTPUT = RESULTS / "cfm_6_full"


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def load_evaluations() -> pd.DataFrame:
    frames = []
    for seed, directory in EVAL_DIRS.items():
        frame = pd.read_csv(REPO_ROOT / directory / "target_summary.csv")
        frame.insert(0, "seed", seed)
        frames.append(frame)
    values = pd.concat(frames, ignore_index=True)
    groups = pd.read_csv(REPO_ROOT / TARGETS)[
        ["target_id", "generalization_group"]
    ]
    values = values.merge(groups, on="target_id", how="left", validate="many_to_one")
    if len(values) != 60 or values.target_id.nunique() != 30:
        raise RuntimeError("CFM-6-FULL requires both seeds on all 30 frozen targets")
    if values.generalization_group.isna().any():
        raise RuntimeError("CFM-6-FULL contains an unclassified target")
    return values


def paired_comparison(
    current: pd.DataFrame, reference_path: Path, reference_name: str
) -> pd.DataFrame:
    reference = pd.read_csv(REPO_ROOT / reference_path)
    columns = [
        "seed", "target_id", "N_valid", "N_keep", "valid_rate",
        "best_total_error", "topk_error", "hit_rate", "best_within_tolerance",
    ]
    paired = current.merge(
        reference[columns],
        on=["seed", "target_id"],
        suffixes=("_full", "_reference"),
        validate="one_to_one",
    )
    paired.insert(0, "reference", reference_name)
    paired["valid_rate_delta"] = paired.valid_rate_full - paired.valid_rate_reference
    paired["best_error_delta"] = (
        paired.best_total_error_full - paired.best_total_error_reference
    )
    paired["best_error_improved"] = paired.best_error_delta < 0
    paired["hit_rate_delta"] = paired.hit_rate_full - paired.hit_rate_reference
    return paired


def main() -> int:
    output = REPO_ROOT / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    training = pd.read_csv(REPO_ROOT / TRAIN_DIR / "seed_summary.csv")
    evaluations = load_evaluations()
    branch = git_value("branch", "--show-current")
    source_commit = git_value("rev-parse", "HEAD")

    ledger_parts = []
    seed_rows = []
    for seed in (42, 43):
        train = training.loc[training.seed == seed].iloc[0]
        timing = json.loads((REPO_ROOT / INFERENCE[seed]).read_text(encoding="utf-8"))
        checkpoint = TRAIN_DIR / f"cfm/seed_{seed}/best.pt"
        values = evaluations.loc[evaluations.seed == seed].copy()
        values.insert(0, "experiment_id", "s2_arch_cfm_6_full_v1")
        values.insert(1, "architecture", "CFM-6-FULL")
        values["parameter_count"] = int(train.total_parameters)
        values["checkpoint_size_bytes"] = int(train.best_checkpoint_bytes)
        values["training_time"] = float(train.training_elapsed_seconds)
        values["best_val_loss"] = float(train.best_validation_loss)
        values["generation_time"] = float(timing["mean_seconds"])
        values["peak_gpu_memory"] = int(train.peak_gpu_memory_bytes)
        values["git_commit"] = source_commit
        values["branch"] = branch
        values["config_path"] = str(CONFIG)
        values["checkpoint_path"] = str(checkpoint)
        values["checkpoint_sha256"] = sha256_file(REPO_ROOT / checkpoint)
        ledger_parts.append(values)
        seed_rows.append(
            {
                "architecture": "CFM-6-FULL",
                "seed": seed,
                "total_parameters": int(train.total_parameters),
                "checkpoint_size_bytes": int(train.best_checkpoint_bytes),
                "best_validation_loss": train.best_validation_loss,
                "best_epoch": int(train.best_epoch),
                "epochs_completed": int(train.epochs_completed),
                "training_time_seconds": train.training_elapsed_seconds,
                "average_epoch_seconds": train.average_epoch_seconds,
                "peak_gpu_memory_bytes": int(train.peak_gpu_memory_bytes),
                "generation_128_mean_seconds": timing["mean_seconds"],
                "generation_128_median_seconds": timing["median_seconds"],
                "generation_128_std_seconds": timing["std_seconds"],
                "N_valid_mean": values.N_valid.mean(),
                "N_keep_mean": values.N_keep.mean(),
                "valid_rate_mean": values.valid_rate.mean(),
                "best_error_median": values.best_total_error.median(),
                "topk_error_median": values.topk_error.median(),
                "median_error_median": values.median_error.median(),
                "hit_rate_mean": values.hit_rate.mean(),
                "target_success_rate": values.best_within_tolerance.mean(),
            }
        )

    ledger = pd.concat(ledger_parts, ignore_index=True)
    ledger.to_csv(output / "cfm_6_full_ledger.csv", index=False)
    with (output / "cfm_6_full_ledger.jsonl").open("w", encoding="utf-8") as stream:
        for record in ledger.replace({np.nan: None}).to_dict(orient="records"):
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(output / "cfm_6_full_seed_summary.csv", index=False)
    groups = aggregate(evaluations)
    groups.to_csv(output / "cfm_6_full_group_summary.csv", index=False)
    overall_input = evaluations.copy()
    overall_input["generalization_group"] = "Overall"
    overall = aggregate(overall_input)
    overall.to_csv(output / "cfm_6_full_overall_summary.csv", index=False)

    references = {
        "CFM-6-SCALED": RESULTS / "cfm_6/cfm_6_ledger.csv",
        "CFM-3": RESULTS / "cfm_3/cfm_3_ledger.csv",
        "CFM-4": RESULTS / "cfm_4/cfm_4_ledger.csv",
        "CFM-5": RESULTS / "cfm_5/cfm_5_ledger.csv",
    }
    paired_frames = []
    for name, path in references.items():
        paired_frames.append(paired_comparison(evaluations, path, name))
    paired = pd.concat(paired_frames, ignore_index=True)
    paired.to_csv(output / "cfm_6_full_paired_comparisons.csv", index=False)

    comparison_parts = [overall.assign(architecture="CFM-6-FULL")]
    for name, directory in (
        ("CFM-6-SCALED", "cfm_6"),
        ("CFM-3", "cfm_3"),
        ("CFM-4", "cfm_4"),
        ("CFM-5", "cfm_5"),
    ):
        path = REPO_ROOT / RESULTS / directory / f"{directory}_overall_summary.csv"
        comparison_parts.append(pd.read_csv(path).assign(architecture=name))
    baseline = pd.read_csv(REPO_ROOT / RESULTS / "cfm_0/model_overall_summary.csv")
    comparison_parts.append(
        baseline.loc[baseline.model == "nflows"].assign(architecture="nflows")
    )
    comparison = pd.concat(comparison_parts, ignore_index=True)
    comparison.to_csv(output / "model_comparison_overall.csv", index=False)

    current = overall.iloc[0]
    scaled = comparison.loc[comparison.architecture == "CFM-6-SCALED"].iloc[0]
    cfm3 = comparison.loc[comparison.architecture == "CFM-3"].iloc[0]
    cfm4 = comparison.loc[comparison.architecture == "CFM-4"].iloc[0]
    cfm5 = comparison.loc[comparison.architecture == "CFM-5"].iloc[0]
    nflows = comparison.loc[comparison.architecture == "nflows"].iloc[0]
    scaled_seeds = pd.read_csv(REPO_ROOT / RESULTS / "cfm_6/cfm_6_seed_summary.csv")
    cfm3_seeds = pd.read_csv(REPO_ROOT / RESULTS / "cfm_3/cfm_3_seed_summary.csv")
    cfm4_seeds = pd.read_csv(REPO_ROOT / RESULTS / "cfm_4/cfm_4_seed_summary.csv")
    cfm5_seeds = pd.read_csv(REPO_ROOT / RESULTS / "cfm_5/cfm_5_seed_summary.csv")
    wins = {
        name: paired.loc[paired.reference == name, "best_error_improved"].mean()
        for name in references
    }
    group_index = groups.set_index("generalization_group")

    report = f"""# CFM-6 原规划宽度复现实验报告

## 结论

原规划 `64/128/256/512` 已在 GPU 6 完整复现。工程 Gate 为 **PASS**，但它仍未达到 CFM-4 或 nflows 的目标精度，也没有证明 MLP-U-Net 层级结构优于普通 Residual/FiLM MLP。

与 `32/64/128/256` 缩小版相比，原宽度显著改善了拟合损失、合法率、目标误差和命中率，说明缩小版确实存在容量/早期压缩瓶颈；不过这些恢复不足以抵消结构带来的复杂度和推理成本。

## 冻结设置

- Branch：`{branch}`；实验源 commit：`{source_commit}`。
- 输入：`x_t(10) + sinusoidal time embedding(32) + condition(3) = 45D`。
- Encoder：`64 → 128 → 256 → 512`。
- Bottleneck：一个512维双层 Residual block。
- Decoder：`512 → 256 → 128 → 64`，固定 concat skip 和线性融合。
- 输出：10D velocity；参数量：{int(seeds.total_parameters.iloc[0]):,}。
- 保持 ConditionalFlowMatcher、sigma=0、RK4/32步、NFE=128、数据、训练策略、目标集和V2.1协议不变。
- V2.1：一次采样128，剔除越界，按 candidate_id 裁剪到不超过合法数的最大2次幂，不补采样。

## 训练和复杂度

| Seed | Best val loss | Best epoch | Epochs | Train (s) | Peak GPU MiB | Inference/128 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in seed_rows:
        report += (
            f"| {row['seed']} | {row['best_validation_loss']:.6f} | "
            f"{row['best_epoch']} | {row['epochs_completed']} | "
            f"{row['training_time_seconds']:.3f} | "
            f"{row['peak_gpu_memory_bytes']/2**20:.2f} | "
            f"{1000*row['generation_128_mean_seconds']:.3f} |\n"
        )
    report += f"""

相对缩小版：参数量 {relative(seeds.total_parameters.iloc[0], scaled_seeds.total_parameters.iloc[0]):+.2%}，平均训练时间 {relative(seeds.training_time_seconds.mean(), scaled_seeds.training_time_seconds.mean()):+.2%}，平均同步推理时间 {relative(seeds.generation_128_mean_seconds.mean(), scaled_seeds.generation_128_mean_seconds.mean()):+.2%}。

## V2.1 分组结果

| Group | N_valid | N_keep | Valid rate | Best error median | Top-10 error median | Hit rate | Target success |
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
| Overall | {current.N_valid_mean:.2f} | {current.N_keep_mean:.2f} | {current.valid_rate_mean:.2%} | {current.best_error_median:.6f} | {current.topk_error_median:.6f} | {current.hit_rate_mean:.2%} | {current.target_success_rate:.2%} |

0D成功率为 {current['0d_success_rate']:.2%}。没有可靠已知可行OOD目标，因此OOD不评价。

## 与缩小版严格对照

| Metric | CFM-6缩小版 | CFM-6原宽度 | Change |
|---|---:|---:|---:|
| Parameters | {int(scaled_seeds.total_parameters.iloc[0]):,} | {int(seeds.total_parameters.iloc[0]):,} | {relative(seeds.total_parameters.iloc[0], scaled_seeds.total_parameters.iloc[0]):+.2%} |
| Best validation loss（双seed均值） | {scaled_seeds.best_validation_loss.mean():.6f} | {seeds.best_validation_loss.mean():.6f} | {relative(seeds.best_validation_loss.mean(), scaled_seeds.best_validation_loss.mean()):+.2%} |
| Valid rate | {scaled.valid_rate_mean:.2%} | {current.valid_rate_mean:.2%} | {current.valid_rate_mean-scaled.valid_rate_mean:+.2%} pt |
| Best error median | {scaled.best_error_median:.6f} | {current.best_error_median:.6f} | {relative(current.best_error_median, scaled.best_error_median):+.2%} |
| Top-10 error median | {scaled.topk_error_median:.6f} | {current.topk_error_median:.6f} | {relative(current.topk_error_median, scaled.topk_error_median):+.2%} |
| Hit rate | {scaled.hit_rate_mean:.2%} | {current.hit_rate_mean:.2%} | {current.hit_rate_mean-scaled.hit_rate_mean:+.2%} pt |
| Target success | {scaled.target_success_rate:.2%} | {current.target_success_rate:.2%} | {current.target_success_rate-scaled.target_success_rate:+.2%} pt |

原宽度在同 seed、同目标的60次配对中，有 {wins['CFM-6-SCALED']:.2%} 取得更小最佳误差。因此容量缩小是上一轮退化的重要原因，但不是全部原因。

## 与已有模型比较

| Model | Parameters | Inference/128 | Valid rate | Best error median | Top-10 error median | Hit rate | Target success |
|---|---:|---:|---:|---:|---:|---:|---:|
| CFM-3 | {int(cfm3_seeds.total_parameters.iloc[0]):,} | {1000*cfm3_seeds.generation_128_mean_seconds.mean():.2f} ms | {cfm3.valid_rate_mean:.2%} | {cfm3.best_error_median:.6f} | {cfm3.topk_error_median:.6f} | {cfm3.hit_rate_mean:.2%} | {cfm3.target_success_rate:.2%} |
| CFM-4 | {int(cfm4_seeds.total_parameters.iloc[0]):,} | {1000*cfm4_seeds.generation_128_mean_seconds.mean():.2f} ms | {cfm4.valid_rate_mean:.2%} | {cfm4.best_error_median:.6f} | {cfm4.topk_error_median:.6f} | {cfm4.hit_rate_mean:.2%} | {cfm4.target_success_rate:.2%} |
| CFM-5 | {int(cfm5_seeds.total_parameters.iloc[0]):,} | {1000*cfm5_seeds.generation_128_mean_seconds.mean():.2f} ms | {cfm5.valid_rate_mean:.2%} | {cfm5.best_error_median:.6f} | {cfm5.topk_error_median:.6f} | {cfm5.hit_rate_mean:.2%} | {cfm5.target_success_rate:.2%} |
| CFM-6缩小版 | {int(scaled_seeds.total_parameters.iloc[0]):,} | {1000*scaled_seeds.generation_128_mean_seconds.mean():.2f} ms | {scaled.valid_rate_mean:.2%} | {scaled.best_error_median:.6f} | {scaled.topk_error_median:.6f} | {scaled.hit_rate_mean:.2%} | {scaled.target_success_rate:.2%} |
| **CFM-6原宽度** | **{int(seeds.total_parameters.iloc[0]):,}** | **{1000*seeds.generation_128_mean_seconds.mean():.2f} ms** | **{current.valid_rate_mean:.2%}** | **{current.best_error_median:.6f}** | **{current.topk_error_median:.6f}** | **{current.hit_rate_mean:.2%}** | **{current.target_success_rate:.2%}** |
| nflows | — | — | {nflows.valid_rate_mean:.2%} | {nflows.best_error_median:.6f} | {nflows.topk_error_median:.6f} | {nflows.hit_rate_mean:.2%} | {nflows.target_success_rate:.2%} |

原宽度 CFM-6 对 CFM-3、CFM-4、CFM-5 的逐目标最佳误差配对胜率分别为 {wins['CFM-3']:.2%}、{wins['CFM-4']:.2%}、{wins['CFM-5']:.2%}。其 Boundary 合法率为 {group_index.loc['Boundary','valid_rate_mean']:.2%}、最佳误差中位数为 {group_index.loc['Boundary','best_error_median']:.6f}，没有形成边界泛化优势。

## 判断

1. **容量影响成立**：增宽显著修复缩小版的拟合和生成质量，因此不能再把 CFM-6 的全部退化归结为 MLP-U-Net 思路本身。
2. **结构优势不成立**：在约{seeds.total_parameters.iloc[0]/cfm4_seeds.total_parameters.iloc[0]:.2f}倍于 CFM-4 的参数量下，原宽度仍然在合法率、最佳误差、Top-10误差和命中率上落后 CFM-4。
3. **层级结构性价比低**：推理没有因增宽明显恶化于缩小版，但约104.7万参数没有换来超过普通 FiLM Residual MLP 的精度。
4. **当前选择不变**：CFM 内部最高综合精度仍是 CFM-4；跨模型的目标精度仍以 nflows 最强。

## STAGE STOP

原宽度 CFM-6 复现实验已完成。CFM-7 未开始。
"""
    report_path = output / "CFM-6原规划宽度复现实验报告.md"
    report_path.write_text(report, encoding="utf-8")
    manifest = {
        "experiment_id": "s2_arch_cfm_6_full_v1",
        "architecture": "CFM-6-FULL",
        "status": "PASS",
        "branch": branch,
        "source_commit": source_commit,
        "encoder_dims": [64, 128, 256, 512],
        "parameters": int(seeds.total_parameters.iloc[0]),
        "seeds": [42, 43],
        "targets": 30,
        "cfm_7_started": False,
        "ledger_sha256": sha256_file(output / "cfm_6_full_ledger.csv"),
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
