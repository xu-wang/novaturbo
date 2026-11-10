#!/usr/bin/env python3
"""Build the CFM-7 ledger and final CFM-0--7 architecture comparison."""

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


TRAIN_DIR = Path("data/generator_models/stage2/cfm_arch/cfm_7_attention_gpu6_v1")
EVAL_DIRS = {
    42: Path("data/inverse_design/stage2/cfm_arch/cfm_7_eval_seed42_v2_1"),
    43: Path("data/inverse_design/stage2/cfm_arch/cfm_7_eval_seed43_v2_1"),
}
INFERENCE = {
    42: Path("data/inverse_design/stage2/cfm_arch/cfm_7_inference_seed42.json"),
    43: Path("data/inverse_design/stage2/cfm_arch/cfm_7_inference_seed43.json"),
}
CONFIG = Path("config/experiments/cfm_arch/cfm_7_attention_train.yaml")
TARGETS = Path("config/experiments/cfm_arch/cfm_0_generalization_targets.csv")
RESULTS = Path("results/s2_arch_cfm")
OUTPUT = RESULTS / "cfm_7"


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
        raise RuntimeError("CFM-7 requires both seeds on all 30 frozen targets")
    if values.generalization_group.isna().any():
        raise RuntimeError("CFM-7 contains an unclassified target")
    return values


def paired(current: pd.DataFrame, path: Path, name: str) -> pd.DataFrame:
    reference = pd.read_csv(REPO_ROOT / path)
    columns = [
        "seed", "target_id", "N_valid", "N_keep", "valid_rate",
        "best_total_error", "topk_error", "hit_rate", "best_within_tolerance",
    ]
    values = current.merge(
        reference[columns],
        on=["seed", "target_id"],
        suffixes=("_cfm7", "_reference"),
        validate="one_to_one",
    )
    values.insert(0, "reference", name)
    values["valid_rate_delta"] = values.valid_rate_cfm7 - values.valid_rate_reference
    values["best_error_delta"] = (
        values.best_total_error_cfm7 - values.best_total_error_reference
    )
    values["best_error_improved"] = values.best_error_delta < 0
    values["hit_rate_delta"] = values.hit_rate_cfm7 - values.hit_rate_reference
    return values


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
        values.insert(0, "experiment_id", "s2_arch_cfm_7_v1")
        values.insert(1, "architecture", "CFM-7")
        values["parameter_count"] = int(train.total_parameters)
        values["trainable_parameter_count"] = int(train.trainable_parameters)
        values["checkpoint_size_bytes"] = int(train.best_checkpoint_bytes)
        values["training_time"] = float(train.training_elapsed_seconds)
        values["average_epoch_time"] = float(train.average_epoch_seconds)
        values["best_val_loss"] = float(train.best_validation_loss)
        values["best_epoch"] = int(train.best_epoch)
        values["epochs_completed"] = int(train.epochs_completed)
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
                "architecture": "CFM-7",
                "seed": seed,
                "total_parameters": int(train.total_parameters),
                "trainable_parameters": int(train.trainable_parameters),
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
    ledger.to_csv(output / "cfm_7_ledger.csv", index=False)
    with (output / "cfm_7_ledger.jsonl").open("w", encoding="utf-8") as stream:
        for record in ledger.replace({np.nan: None}).to_dict(orient="records"):
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(output / "cfm_7_seed_summary.csv", index=False)
    groups = aggregate(evaluations)
    groups.to_csv(output / "cfm_7_group_summary.csv", index=False)
    overall_input = evaluations.copy()
    overall_input["generalization_group"] = "Overall"
    overall = aggregate(overall_input)
    overall.to_csv(output / "cfm_7_overall_summary.csv", index=False)

    references = {
        "CFM-3": RESULTS / "cfm_3/cfm_3_ledger.csv",
        "CFM-4": RESULTS / "cfm_4/cfm_4_ledger.csv",
        "CFM-5": RESULTS / "cfm_5/cfm_5_ledger.csv",
        "CFM-6-FULL": RESULTS / "cfm_6_full/cfm_6_full_ledger.csv",
    }
    paired_values = pd.concat(
        [paired(evaluations, path, name) for name, path in references.items()],
        ignore_index=True,
    )
    paired_values.to_csv(output / "cfm_7_paired_comparisons.csv", index=False)
    wins = {
        name: paired_values.loc[
            paired_values.reference == name, "best_error_improved"
        ].mean()
        for name in references
    }

    architecture_rows = []
    specifications = [
        ("CFM-0", "cfm_0", "model_overall_summary.csv", "cfm_0_seed_summary.csv"),
        ("CFM-1", "cfm_1", "cfm_1_overall_summary.csv", "cfm_1_seed_summary.csv"),
        ("CFM-2", "cfm_2", "cfm_2_overall_summary.csv", "cfm_2_seed_summary.csv"),
        ("CFM-3", "cfm_3", "cfm_3_overall_summary.csv", "cfm_3_seed_summary.csv"),
        ("CFM-4", "cfm_4", "cfm_4_overall_summary.csv", "cfm_4_seed_summary.csv"),
        ("CFM-5", "cfm_5", "cfm_5_overall_summary.csv", "cfm_5_seed_summary.csv"),
        ("CFM-6-SCALED", "cfm_6", "cfm_6_overall_summary.csv", "cfm_6_seed_summary.csv"),
        ("CFM-6-FULL", "cfm_6_full", "cfm_6_full_overall_summary.csv", "cfm_6_full_seed_summary.csv"),
    ]
    for name, directory, overall_name, seed_name in specifications:
        summary = pd.read_csv(REPO_ROOT / RESULTS / directory / overall_name)
        if name == "CFM-0":
            summary = summary.loc[summary.model == "cfm"]
        metric = summary.iloc[0]
        complexity = pd.read_csv(REPO_ROOT / RESULTS / directory / seed_name)
        architecture_rows.append(
            {
                "architecture": name,
                "parameters": int(complexity.total_parameters.iloc[0]),
                "validation_loss_mean": complexity.best_validation_loss.mean(),
                "training_time_mean_seconds": complexity.training_time_seconds.mean(),
                "inference_128_mean_seconds": complexity.generation_128_mean_seconds.mean(),
                "peak_gpu_memory_mean_bytes": complexity.peak_gpu_memory_bytes.mean(),
                "valid_rate": metric.valid_rate_mean,
                "N_keep_mean": metric.N_keep_mean,
                "best_error_median": metric.best_error_median,
                "topk_error_median": metric.topk_error_median,
                "hit_rate": metric.hit_rate_mean,
                "target_success": metric.target_success_rate,
            }
        )
    current = overall.iloc[0]
    architecture_rows.append(
        {
            "architecture": "CFM-7",
            "parameters": int(seeds.total_parameters.iloc[0]),
            "validation_loss_mean": seeds.best_validation_loss.mean(),
            "training_time_mean_seconds": seeds.training_time_seconds.mean(),
            "inference_128_mean_seconds": seeds.generation_128_mean_seconds.mean(),
            "peak_gpu_memory_mean_bytes": seeds.peak_gpu_memory_bytes.mean(),
            "valid_rate": current.valid_rate_mean,
            "N_keep_mean": current.N_keep_mean,
            "best_error_median": current.best_error_median,
            "topk_error_median": current.topk_error_median,
            "hit_rate": current.hit_rate_mean,
            "target_success": current.target_success_rate,
        }
    )
    architectures = pd.DataFrame(architecture_rows)
    architectures.to_csv(output / "cfm_0_to_7_comparison.csv", index=False)
    architecture_index = architectures.set_index("architecture")
    cfm0 = architecture_index.loc["CFM-0"]
    cfm1 = architecture_index.loc["CFM-1"]
    cfm2 = architecture_index.loc["CFM-2"]
    cfm3 = architecture_index.loc["CFM-3"]
    cfm4 = architecture_index.loc["CFM-4"]
    cfm5 = architecture_index.loc["CFM-5"]
    cfm6_scaled = architecture_index.loc["CFM-6-SCALED"]
    cfm6 = architecture_index.loc["CFM-6-FULL"]
    cfm7 = architecture_index.loc["CFM-7"]
    nflows = pd.read_csv(REPO_ROOT / RESULTS / "cfm_0/model_overall_summary.csv")
    nflows = nflows.loc[nflows.model == "nflows"].iloc[0]
    group_index = groups.set_index("generalization_group")

    report = f"""# S2-ARCH-CFM CFM-7 阶段与最终结构对照报告

## 阶段结论

- Architecture：CFM-7 Lightweight Attention。
- Status：**PASS**（工程和实验闭环完成，不代表目标精度最优）。
- Branch：`{branch}`；实验源 commit：`{source_commit}`。
- CFM-0～7 结构序列已全部完成；未启动任何后续方法实验。

CFM-7 在未使用 tanh、clip 或其他有界输出机制的情况下，将合法率提高到 **{current.valid_rate_mean:.2%}**，显著高于所有已有 CFM；但最佳目标误差中位数为 **{current.best_error_median:.6f}**，没有超过 CFM-4，且两个 seed 的误差差异较明显。Attention 的主要优势是合法候选生成，而不是最高目标精度。

## Architecture

```text
x_t的10个标量 → 10个design tokens
condition(3)   → 1个condition token
time embedding(32) → 1个time token
12 tokens × 128
→ 2层 Pre-Norm Transformer Encoder
→ 4 heads / FFN 256 / dropout 0
→ 读取10个design tokens
→ shared Linear(128,1)
→ 10D velocity
```

- 参数量：{int(seeds.total_parameters.iloc[0]):,}，与 CFM-3 的 {int(cfm3.parameters):,} 基本等量。
- Plain ConditionalFlowMatcher、sigma=0、RK4/32步、NFE=128保持冻结。
- V2.1固定：一次采样128，剔除越界，按candidate_id裁剪到最大2次幂，不补采样。

## Training and Complexity

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

相对同参数量级的 CFM-3，CFM-7 平均训练时间变化 {relative(cfm7.training_time_mean_seconds, cfm3.training_time_mean_seconds):+.2%}，推理时间变化 {relative(cfm7.inference_128_mean_seconds, cfm3.inference_128_mean_seconds):+.2%}，峰值显存变化 {relative(cfm7.peak_gpu_memory_mean_bytes, cfm3.peak_gpu_memory_mean_bytes):+.2%}。FLOPs/MACs未采用可靠工具测量，记录为N/A。

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

0D成功率为 {current['0d_success_rate']:.2%}。Boundary合法率为 {group_index.loc['Boundary','valid_rate_mean']:.2%}，说明高合法率覆盖边界目标，而非只来自ID目标。没有可靠已知可行OOD目标，因此OOD不评价。

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

合法率在两个 seed 上稳定，但最佳误差存在明显差异，因此 CFM-7 的合法性结论稳定，目标精度结论不如 CFM-4/5 稳定。

## CFM-0～7 完整结构比较

| Model | Params | Inference/128 | Valid rate | N_keep | Best error | Top-10 error | Hit rate | Success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CFM-0 | {int(cfm0.parameters):,} | {1000*cfm0.inference_128_mean_seconds:.2f} ms | {cfm0.valid_rate:.2%} | {cfm0.N_keep_mean:.2f} | {cfm0.best_error_median:.6f} | {cfm0.topk_error_median:.6f} | {cfm0.hit_rate:.2%} | {cfm0.target_success:.2%} |
| CFM-1 | {int(cfm1.parameters):,} | {1000*cfm1.inference_128_mean_seconds:.2f} ms | {cfm1.valid_rate:.2%} | {cfm1.N_keep_mean:.2f} | {cfm1.best_error_median:.6f} | {cfm1.topk_error_median:.6f} | {cfm1.hit_rate:.2%} | {cfm1.target_success:.2%} |
| CFM-2 | {int(cfm2.parameters):,} | {1000*cfm2.inference_128_mean_seconds:.2f} ms | {cfm2.valid_rate:.2%} | {cfm2.N_keep_mean:.2f} | {cfm2.best_error_median:.6f} | {cfm2.topk_error_median:.6f} | {cfm2.hit_rate:.2%} | {cfm2.target_success:.2%} |
| CFM-3 | {int(cfm3.parameters):,} | {1000*cfm3.inference_128_mean_seconds:.2f} ms | {cfm3.valid_rate:.2%} | {cfm3.N_keep_mean:.2f} | {cfm3.best_error_median:.6f} | {cfm3.topk_error_median:.6f} | {cfm3.hit_rate:.2%} | {cfm3.target_success:.2%} |
| CFM-4 | {int(cfm4.parameters):,} | {1000*cfm4.inference_128_mean_seconds:.2f} ms | {cfm4.valid_rate:.2%} | {cfm4.N_keep_mean:.2f} | {cfm4.best_error_median:.6f} | {cfm4.topk_error_median:.6f} | {cfm4.hit_rate:.2%} | {cfm4.target_success:.2%} |
| CFM-5 | {int(cfm5.parameters):,} | {1000*cfm5.inference_128_mean_seconds:.2f} ms | {cfm5.valid_rate:.2%} | {cfm5.N_keep_mean:.2f} | {cfm5.best_error_median:.6f} | {cfm5.topk_error_median:.6f} | {cfm5.hit_rate:.2%} | {cfm5.target_success:.2%} |
| CFM-6缩小版 | {int(cfm6_scaled.parameters):,} | {1000*cfm6_scaled.inference_128_mean_seconds:.2f} ms | {cfm6_scaled.valid_rate:.2%} | {cfm6_scaled.N_keep_mean:.2f} | {cfm6_scaled.best_error_median:.6f} | {cfm6_scaled.topk_error_median:.6f} | {cfm6_scaled.hit_rate:.2%} | {cfm6_scaled.target_success:.2%} |
| CFM-6原宽度 | {int(cfm6.parameters):,} | {1000*cfm6.inference_128_mean_seconds:.2f} ms | {cfm6.valid_rate:.2%} | {cfm6.N_keep_mean:.2f} | {cfm6.best_error_median:.6f} | {cfm6.topk_error_median:.6f} | {cfm6.hit_rate:.2%} | {cfm6.target_success:.2%} |
| **CFM-7** | **{int(cfm7.parameters):,}** | **{1000*cfm7.inference_128_mean_seconds:.2f} ms** | **{cfm7.valid_rate:.2%}** | **{cfm7.N_keep_mean:.2f}** | **{cfm7.best_error_median:.6f}** | **{cfm7.topk_error_median:.6f}** | **{cfm7.hit_rate:.2%}** | **{cfm7.target_success:.2%}** |
| nflows | — | — | {nflows.valid_rate_mean:.2%} | {nflows.N_keep_mean:.2f} | {nflows.best_error_median:.6f} | {nflows.topk_error_median:.6f} | {nflows.hit_rate_mean:.2%} | {nflows.target_success_rate:.2%} |

CFM-7 对 CFM-3、CFM-4、CFM-5、CFM-6原宽度的逐目标最佳误差配对胜率分别为 {wins['CFM-3']:.2%}、{wins['CFM-4']:.2%}、{wins['CFM-5']:.2%}、{wins['CFM-6-FULL']:.2%}。

## 最终判断

1. **显式变量交互有效，但作用主要体现在合法率**：Attention 使平均约 {current.N_valid_mean:.1f}/128 个候选合法，裁剪后约 {current.N_keep_mean:.1f} 个进入0D，明显降低了网络采样浪费。
2. **目标精度不是最优**：CFM-7 的最佳误差和Top-10误差均落后CFM-4，也远落后nflows。
3. **复杂度代价明显**：虽然参数量和CFM-3接近，但序列注意力使训练、推理和显存成本显著增加。
4. **推荐按用途选择**：目标精度优先选CFM-4；合法候选覆盖优先可选CFM-7；跨生成方法的给定目标精度仍选nflows。
5. **不建议仅凭综合单分数选择**：CFM-4与CFM-7代表“精度”和“合法率/候选利用率”两种不同优势，应在后续物理筛选和最终应用成本中分别验证。

## STAGE STOP

CFM-7 已完成。CFM-0～7 结构消融阶段完成；未自动进入后续实验。
"""
    report_path = output / "CFM-7阶段与最终结构对照报告.md"
    report_path.write_text(report, encoding="utf-8")
    manifest = {
        "experiment_id": "s2_arch_cfm_7_v1",
        "architecture": "CFM-7",
        "status": "PASS",
        "branch": branch,
        "source_commit": source_commit,
        "parameters": int(seeds.total_parameters.iloc[0]),
        "seeds": [42, 43],
        "targets": 30,
        "next_stage_started": False,
        "ledger_sha256": sha256_file(output / "cfm_7_ledger.csv"),
        "comparison_sha256": sha256_file(output / "cfm_0_to_7_comparison.csv"),
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
