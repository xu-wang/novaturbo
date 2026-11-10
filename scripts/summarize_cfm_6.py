#!/usr/bin/env python3
"""Build the CFM-6 ledger, comparisons, and strict stage report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_cfm_2 import aggregate, git_value, relative
from src.ai.baseline import sha256_file


TRAIN_DIR = Path("data/generator_models/stage2/cfm_arch/cfm_6_mlp_unet_gpu6_v1")
EVAL_DIRS = {
    42: Path("data/inverse_design/stage2/cfm_arch/cfm_6_eval_seed42_v2_1"),
    43: Path("data/inverse_design/stage2/cfm_arch/cfm_6_eval_seed43_v2_1"),
}
INFERENCE = {
    42: Path("data/inverse_design/stage2/cfm_arch/cfm_6_inference_seed42.json"),
    43: Path("data/inverse_design/stage2/cfm_arch/cfm_6_inference_seed43.json"),
}
CONFIG = Path("config/experiments/cfm_arch/cfm_6_mlp_unet_train.yaml")
TARGETS = Path("config/experiments/cfm_arch/cfm_0_generalization_targets.csv")
RESULTS = Path("results/s2_arch_cfm")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS / "cfm_6")
    args = parser.parse_args()
    output = REPO_ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    training = pd.read_csv(REPO_ROOT / TRAIN_DIR / "seed_summary.csv")
    frames = []
    for seed, directory in EVAL_DIRS.items():
        frame = pd.read_csv(REPO_ROOT / directory / "target_summary.csv")
        frame.insert(0, "seed", seed)
        frames.append(frame)
    evaluations = pd.concat(frames, ignore_index=True)
    target_groups = pd.read_csv(REPO_ROOT / TARGETS)[
        ["target_id", "generalization_group"]
    ]
    evaluations = evaluations.merge(
        target_groups, on="target_id", how="left", validate="many_to_one"
    )
    if len(evaluations) != 60 or evaluations.target_id.nunique() != 30:
        raise RuntimeError("CFM-6 requires the same 30 frozen targets for both seeds")
    if evaluations.generalization_group.isna().any():
        raise RuntimeError("unclassified CFM-6 target")

    branch = git_value("branch", "--show-current")
    source_commit = git_value("rev-parse", "HEAD")
    ledgers = []
    seed_rows = []
    for seed in (42, 43):
        train = training.loc[training.seed == seed].iloc[0]
        inference = json.loads(
            (REPO_ROOT / INFERENCE[seed]).read_text(encoding="utf-8")
        )
        checkpoint = TRAIN_DIR / f"cfm/seed_{seed}/best.pt"
        values = evaluations.loc[evaluations.seed == seed].copy()
        values.insert(0, "experiment_id", "s2_arch_cfm_6_v1")
        values.insert(1, "architecture", "CFM-6")
        values["target_F"] = values.target_thrust_N
        values["target_TSFC"] = values.target_tsfc_kg_N_s
        values["target_Mass"] = values.target_total_mass_kg
        values["parameter_count"] = int(train.total_parameters)
        values["trainable_parameter_count"] = int(train.trainable_parameters)
        values["checkpoint_size_bytes"] = int(train.best_checkpoint_bytes)
        values["training_time"] = float(train.training_elapsed_seconds)
        values["average_epoch_time"] = float(train.average_epoch_seconds)
        values["best_val_loss"] = float(train.best_validation_loss)
        values["best_epoch"] = int(train.best_epoch)
        values["epochs_completed"] = int(train.epochs_completed)
        values["generation_time"] = float(inference["mean_seconds"])
        values["peak_gpu_memory"] = int(train.peak_gpu_memory_bytes)
        values["0d_success"] = values["0d_success_count"]
        values["0d_failure"] = values["0d_failure_count"]
        values["best_error"] = values.best_total_error
        values["git_commit"] = source_commit
        values["branch"] = branch
        values["config_path"] = str(CONFIG)
        values["checkpoint_path"] = str(checkpoint)
        values["checkpoint_sha256"] = sha256_file(REPO_ROOT / checkpoint)
        ledgers.append(values)
        seed_rows.append(
            {
                "architecture": "CFM-6",
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
                "generation_128_mean_seconds": inference["mean_seconds"],
                "generation_128_median_seconds": inference["median_seconds"],
                "generation_128_std_seconds": inference["std_seconds"],
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

    ledger = pd.concat(ledgers, ignore_index=True)
    ledger.to_csv(output / "cfm_6_ledger.csv", index=False)
    with (output / "cfm_6_ledger.jsonl").open("w", encoding="utf-8") as stream:
        for record in ledger.replace({np.nan: None}).to_dict(orient="records"):
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(output / "cfm_6_seed_summary.csv", index=False)
    groups = aggregate(evaluations)
    groups.to_csv(output / "cfm_6_group_summary.csv", index=False)
    overall_frame = evaluations.copy()
    overall_frame["generalization_group"] = "Overall"
    overall = aggregate(overall_frame)
    overall.to_csv(output / "cfm_6_overall_summary.csv", index=False)

    old_ledger = pd.read_csv(REPO_ROOT / RESULTS / "cfm_5/cfm_5_ledger.csv")
    paired = evaluations.merge(
        old_ledger[
            [
                "seed", "target_id", "valid_rate", "N_valid", "N_keep",
                "best_total_error", "topk_error", "hit_rate", "best_within_tolerance",
            ]
        ],
        on=["seed", "target_id"],
        suffixes=("_cfm6", "_cfm5"),
        validate="one_to_one",
    )
    paired["valid_rate_delta"] = paired.valid_rate_cfm6 - paired.valid_rate_cfm5
    paired["best_error_delta"] = paired.best_total_error_cfm6 - paired.best_total_error_cfm5
    paired["best_error_improved"] = paired.best_error_delta < 0
    paired["hit_rate_delta"] = paired.hit_rate_cfm6 - paired.hit_rate_cfm5
    paired.to_csv(output / "cfm_6_vs_cfm_5_paired.csv", index=False)

    old_overall = pd.read_csv(REPO_ROOT / RESULTS / "cfm_5/cfm_5_overall_summary.csv")
    film_overall = pd.read_csv(REPO_ROOT / RESULTS / "cfm_4/cfm_4_overall_summary.csv")
    cfm2_overall = pd.read_csv(REPO_ROOT / RESULTS / "cfm_2/cfm_2_overall_summary.csv")
    baseline_all = pd.read_csv(REPO_ROOT / RESULTS / "cfm_0/model_overall_summary.csv")
    nflows = baseline_all.loc[baseline_all.model == "nflows"].iloc[0]
    old = old_overall.iloc[0]
    film = film_overall.iloc[0]
    cfm2 = cfm2_overall.iloc[0]
    new = overall.iloc[0]
    pd.concat(
        [
            overall.assign(architecture="CFM-6 MLP-U-Net"),
            old_overall.assign(architecture="CFM-5 Gated"),
            film_overall.assign(architecture="CFM-4 FiLM Residual"),
            cfm2_overall.assign(architecture="CFM-2 Wide/Deep"),
            baseline_all.loc[baseline_all.model == "nflows"].assign(
                architecture="nflows Existing Best"
            ),
        ], ignore_index=True,
    ).to_csv(output / "model_comparison_overall.csv", index=False)

    old_seeds = pd.read_csv(REPO_ROOT / RESULTS / "cfm_5/cfm_5_seed_summary.csv")
    param_delta = relative(seeds.total_parameters.iloc[0], old_seeds.total_parameters.iloc[0])
    train_delta = relative(seeds.training_time_seconds.mean(), old_seeds.training_time_seconds.mean())
    inference_delta = relative(
        seeds.generation_128_mean_seconds.mean(), old_seeds.generation_128_mean_seconds.mean()
    )
    improved = paired.best_error_improved.mean()
    groups6 = groups.set_index("generalization_group")
    groups5 = pd.read_csv(REPO_ROOT / RESULTS / "cfm_5/cfm_5_group_summary.csv").set_index(
        "generalization_group"
    )

    report = f"""# S2-ARCH-CFM CFM-6 阶段报告

## 阶段状态

- Architecture：CFM-6 MLP-U-Net
- Status：PASS
- Branch：`{branch}`
- 实验源 commit：`{source_commit}`
- 原建议64/128/256/512预计参数量过高；按实验规则统一缩小一级为32/64/128/256。
- 固定concat skip和一个256维双层Residual bottleneck；未加入FiLM、gate、attention或卷积。
- CFM-7：未实现、未配置、未训练。

PASS表示完整工程与实验流程完成，不表示性能优于现有CFM。

## Architecture

```text
input45 → 32 → 64 → 128 → 256
                  Residual bottleneck(256)
256 → 128 + concat(skip128) → fusion128
128 → 64  + concat(skip64)  → fusion64
64  → 32  + concat(skip32)  → fusion32
→ output10
```

- Plain `ConditionalFlowMatcher(sigma=0)`；RK4/32/128 NFE。
- 数据、训练、采样、V2.1、目标集和物理评价全部冻结。
- 参数量：{int(seeds.total_parameters.iloc[0]):,}，较CFM-5 {param_delta:+.2%}，与CFM-3同量级。

## Training and Complexity

| Seed | Best val loss | Best epoch | Epochs | Train (s) | Peak GPU MiB | Inference/128 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in seed_rows:
        report += (
            f"| {row['seed']} | {row['best_validation_loss']:.6f} | {row['best_epoch']} | "
            f"{row['epochs_completed']} | {row['training_time_seconds']:.3f} | "
            f"{row['peak_gpu_memory_bytes']/2**20:.2f} | "
            f"{1000*row['generation_128_mean_seconds']:.3f} |\n"
        )
    report += f"""

两个seed均使用完整79,945/9,992样本并正常早停。相对CFM-5，平均训练时间变化 {train_delta:+.2%}，同步推理时间变化 {inference_delta:+.2%}。

## V2.1 Validity and Target Accuracy

严格执行一次 `128 → N_valid → 2^floor(log2(N_valid)) → candidate_id裁剪 → 0D`，不补采样。

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
| Overall | {new.N_valid_mean:.2f} | {new.N_keep_mean:.2f} | {new.valid_rate_mean:.2%} | {new.best_error_median:.6f} | {new.topk_error_median:.6f} | {new.hit_rate_mean:.2%} | {new.target_success_rate:.2%} |

0D成功率为 {new['0d_success_rate']:.2%}；没有可靠已知可行OOD目标，因此OOD not evaluated。

## CFM-6 vs CFM-5

| Metric | CFM-5 Gated | CFM-6 MLP-U-Net | Change |
|---|---:|---:|---:|
| Parameters | {int(old_seeds.total_parameters.iloc[0]):,} | {int(seeds.total_parameters.iloc[0]):,} | {param_delta:+.2%} |
| Mean train time (s) | {old_seeds.training_time_seconds.mean():.3f} | {seeds.training_time_seconds.mean():.3f} | {train_delta:+.2%} |
| Mean inference/128 (ms) | {1000*old_seeds.generation_128_mean_seconds.mean():.3f} | {1000*seeds.generation_128_mean_seconds.mean():.3f} | {inference_delta:+.2%} |
| Valid rate | {old.valid_rate_mean:.2%} | {new.valid_rate_mean:.2%} | {new.valid_rate_mean-old.valid_rate_mean:+.2%} pt |
| N_keep mean | {old.N_keep_mean:.2f} | {new.N_keep_mean:.2f} | {new.N_keep_mean-old.N_keep_mean:+.2f} |
| Best error median | {old.best_error_median:.6f} | {new.best_error_median:.6f} | {relative(new.best_error_median, old.best_error_median):+.2%} |
| Top-10 error median | {old.topk_error_median:.6f} | {new.topk_error_median:.6f} | {relative(new.topk_error_median, old.topk_error_median):+.2%} |
| Candidate hit rate | {old.hit_rate_mean:.2%} | {new.hit_rate_mean:.2%} | {new.hit_rate_mean-old.hit_rate_mean:+.2%} pt |
| Target success | {old.target_success_rate:.2%} | {new.target_success_rate:.2%} | {new.target_success_rate-old.target_success_rate:+.2%} pt |

逐目标同seed配对中，CFM-6只有 {improved:.2%} 的运行取得更小最佳误差。合法率分组变化：ID {groups6.loc['ID','valid_rate_mean']-groups5.loc['ID','valid_rate_mean']:+.2%} pt、Interpolation {groups6.loc['Interpolation','valid_rate_mean']-groups5.loc['Interpolation','valid_rate_mean']:+.2%} pt、Boundary {groups6.loc['Boundary','valid_rate_mean']-groups5.loc['Boundary','valid_rate_mean']:+.2%} pt。

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

两个seed的合法率和误差量级接近，退化具有一致性，不是单seed失败。

## Comparison with Strong CFM and Existing Best

| Model | Valid rate | Best error median | Top-10 error median | Hit rate | Target success |
|---|---:|---:|---:|---:|---:|
| CFM-2 | {cfm2.valid_rate_mean:.2%} | {cfm2.best_error_median:.6f} | {cfm2.topk_error_median:.6f} | {cfm2.hit_rate_mean:.2%} | {cfm2.target_success_rate:.2%} |
| CFM-4 | {film.valid_rate_mean:.2%} | {film.best_error_median:.6f} | {film.topk_error_median:.6f} | {film.hit_rate_mean:.2%} | {film.target_success_rate:.2%} |
| CFM-5 | {old.valid_rate_mean:.2%} | {old.best_error_median:.6f} | {old.topk_error_median:.6f} | {old.hit_rate_mean:.2%} | {old.target_success_rate:.2%} |
| CFM-6 | {new.valid_rate_mean:.2%} | {new.best_error_median:.6f} | {new.topk_error_median:.6f} | {new.hit_rate_mean:.2%} | {new.target_success_rate:.2%} |
| nflows | {nflows.valid_rate_mean:.2%} | {nflows.best_error_median:.6f} | {nflows.topk_error_median:.6f} | {nflows.hit_rate_mean:.2%} | {nflows.target_success_rate:.2%} |

CFM-6最佳误差中位数为nflows的 {new.best_error_median/nflows.best_error_median:.2f} 倍，候选命中率低 {nflows.hit_rate_mean-new.hit_rate_mean:.2%} 个百分点。

## Advantages

1. 两个seed都稳定完成训练和物理回代，层级结构本身工程可行。
2. Boundary目标成功率保持100%，但没有形成误差或合法率优势。
3. 参数量控制在CFM-3同量级，没有依靠超大容量测试层级结构。

## Disadvantages

1. 相对CFM-5，准确度、合法率、N_keep、命中率和目标成功率全面退化，配对胜率仅 {improved:.2%}。
2. 虽然参数更少，但多层小矩阵和concat融合导致训练及推理更慢。
3. Interpolation误差最高，层级压缩没有改善边界或插值泛化。
4. 核心目标精度显著落后nflows，且没有OOD结论。

## Current Interpretation

当前10D连续向量没有图像式空间层级，32维早期表示和逐级压缩形成了信息瓶颈；concat skip未能补偿该损失。结果不支持在本问题上使用当前规模的MLP-U-Net。由于按公平性规则缩小了原建议网络，本结论针对固定的32/64/128/256版本，不外推到更大但计算代价更高的U-Net。

## Recommendation

不保留CFM-6作为候选模型。最高精度仍选CFM-4，效率优先选CFM-2，折中选CFM-5。是否进入CFM-7 Attention Mixer，必须由用户根据本报告明确决定。

## STAGE STOP

CFM-6 已完成。CFM-7 未开始。
"""
    (output / "CFM-6阶段报告.md").write_text(report, encoding="utf-8")

    manifest = {
        "experiment_id": "s2_arch_cfm_6_v1",
        "architecture": "CFM-6",
        "status": "PASS",
        "branch": branch,
        "source_commit": source_commit,
        "cfm_7_started": False,
        "targets": 30,
        "seeds": [42, 43],
        "ledger": "cfm_6_ledger.csv",
        "ledger_sha256": sha256_file(output / "cfm_6_ledger.csv"),
        "report": "CFM-6阶段报告.md",
        "report_sha256": sha256_file(output / "CFM-6阶段报告.md"),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
