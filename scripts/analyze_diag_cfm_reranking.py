#!/usr/bin/env python3
"""Evaluate Diag-CFM reverse-prediction scores on frozen CFM-4 candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.generator_dataset import GeneratorNormalizer
from src.ai.generators import load_generator


def analyze(
    evaluation_dir: Path,
    checkpoint: Path,
    normalizer_path: Path,
    output_csv: Path,
    device: str,
) -> pd.DataFrame:
    if output_csv.exists():
        raise FileExistsError(f"reranking analysis output exists: {output_csv}")
    generator = load_generator(checkpoint, device=device)
    generator.model_config["ode_method"] = "rk4"
    generator.model_config["ode_steps"] = 32
    normalizer = GeneratorNormalizer.load(normalizer_path)
    ledger = pd.read_csv(evaluation_dir / "evaluation_ledger.csv")
    ledger = ledger[
        (ledger["model"] == "cfm") & ledger["forward_valid"].astype(bool)
    ].copy()
    targets = pd.read_csv(evaluation_dir / "evaluated_targets.csv").set_index(
        "target_id"
    )
    design_columns = [f"normalized_{name}" for name in normalizer.design_names]

    scored_frames = []
    for target_id, frame in ledger.groupby("target_id", sort=False):
        target = targets.loc[target_id, normalizer.condition_names]
        normalized_target = normalizer.transform_conditions(target)
        designs = torch.as_tensor(
            frame[design_columns].to_numpy(dtype=np.float32, copy=True), device=device
        )
        predicted = generator.predict_conditions(designs).detach().cpu().numpy()
        score = np.mean((predicted - normalized_target[None, :]) ** 2, axis=1)
        scored = frame.copy()
        scored["diag_reverse_condition_mse"] = score
        scored_frames.append(scored)
    result = pd.concat(scored_frames, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)

    selected = []
    for _, frame in result.groupby("target_id", sort=False):
        selected.append(
            frame.nsmallest(max(1, len(frame) // 2), "diag_reverse_condition_mse")
        )
    retained = pd.concat(selected, ignore_index=True)
    all_target_best = result.groupby("target_id")["total_error"].min()
    retained_target_best = retained.groupby("target_id")["total_error"].min()
    all_target_success = result.groupby("target_id")["within_tolerance"].any()
    retained_target_success = retained.groupby("target_id")[
        "within_tolerance"
    ].any()
    print(
        {
            "candidate_count": len(result),
            "spearman_score_vs_0d_error": float(
                result["diag_reverse_condition_mse"].corr(
                    result["total_error"], method="spearman"
                )
            ),
            "all_hit_rate": float(result["within_tolerance"].mean()),
            "lowest_half_score_hit_rate": float(
                retained["within_tolerance"].mean()
            ),
            "all_median_error": float(result["total_error"].median()),
            "lowest_half_score_median_error": float(
                retained["total_error"].median()
            ),
            "all_median_target_best_error": float(all_target_best.median()),
            "lowest_half_score_median_target_best_error": float(
                retained_target_best.median()
            ),
            "all_target_success_rate": float(all_target_success.mean()),
            "lowest_half_score_target_success_rate": float(
                retained_target_success.mean()
            ),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    analyze(
        args.evaluation_dir,
        args.checkpoint,
        args.normalizer,
        args.output_csv,
        args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
